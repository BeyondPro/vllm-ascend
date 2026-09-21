"""kt-kernel LLAMAFILE backend for CPU routed experts.

This adapter reuses ktransformers' pinned buffers, CPUInfer worker pool, and
GGUF/LLAMAFILE MoE implementation.  Two execution modes are supported:

* **eager** -- ``submit`` blocks the host thread into kt-kernel's WorkerPool
  and ``wait`` drains it.  Correct, and the only mode that allows CPU/NPU
  overlap today.
* **graph** -- during ACL graph capture the layer is recorded as a *pair* of
  stream host callbacks around kt-kernel's pinned buffers: a submit half that
  enqueues the CPU MoE as soon as the inputs have been copied down, and a join
  half that waits for it at the point the result is first needed.  The input
  D2H and the output H2D are ordinary device ops, so like the callbacks they
  re-run on every replay.  Splitting submit from join is what makes the CPU
  MoE overlap: the NPU work captured between the two halves runs concurrently
  with the WorkerPool instead of stalling behind it.

The mode is chosen per call from whether the current stream is being captured,
so one backend serves a warmup, a capture, and every replay without being told
which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass(frozen=True, slots=True)
class KtKernelBackendConfig:
    """Immutable construction parameters for one CPU MoE layer."""

    layer_idx: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    weight_path: str
    cpuinfer_threads: int = 32
    threadpool_count: int = 1
    max_num_tokens: int = 1
    numa_nodes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        positive_fields = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "cpuinfer_threads": self.cpuinfer_threads,
            "threadpool_count": self.threadpool_count,
            "max_num_tokens": self.max_num_tokens,
        }
        if self.layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.numa_nodes is not None and len(
                self.numa_nodes) != self.threadpool_count:
            raise ValueError(
                "numa_nodes must contain one node per thread pool; "
                f"got nodes={len(self.numa_nodes)}, "
                f"threadpool_count={self.threadpool_count}")

    def resolved_weight_path(self) -> str:
        """Resolve the reference project's per-layer GGUF template."""
        try:
            path = self.weight_path.format(layer_idx=self.layer_idx)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "weight_path must be literal or use the {layer_idx} field") from exc
        if not Path(path).is_file():
            raise FileNotFoundError(f"CPU MoE GGUF not found: {path}")
        return path


# Batch sizes whose pinned buffers a captured graph has already baked in.
#
# ``KExpertsCPUBuffer`` keeps a strong reference only to buffers registered
# through ``set_capture_batch_sizes``; every other batch size lives in a
# single-slot temp buffer that a later call at a different batch size replaces.
# A captured graph has the pinned *addresses* baked in, so dropping that
# reference would leave replay writing into freed host memory -- a corruption
# that shows up as garbage output, not as an error.  Module-level because the
# cache it protects (``KExpertsCPUBuffer.capture_buffers``) is class-level and
# therefore process-wide.
_GRAPH_PINNED_BATCH_SIZES: set[int] = set()

# The kt-kernel cache is keyed only by token count. Keep the shape contract
# beside the process-wide batch registry so incompatible wrappers cannot
# silently reuse a captured buffer with the wrong shape or dtype.
_GRAPH_BATCH_SIGNATURES: dict[int, tuple[int, int, torch.dtype, str, int | None]] = {}

# Stream ids this module has registered for callback reporting.  Keyed by the
# integer stream id rather than the Stream object so the check cannot depend on
# torch's stream equality semantics.
_SUBSCRIBED_STREAM_IDS: set[int] = set()

# Methods the wrapper must expose for the graph path.  Checked when a capture
# first asks for them rather than at construction, so an older kt-kernel build
# keeps working for eager decode.
_GRAPH_REQUIRED_METHODS = (
    "copy_inputs_to_cpu_buffers",
    "forward_on_pinned_buffers",
    "drain_pinned_forward",
    "copy_forward_output_to_device",
    "set_capture_batch_sizes",
)


def _graph_capture_active() -> bool:
    """Whether this forward is being recorded into an ACL graph.

    vLLM-Ascend sets ``_EXTRA_CTX.capturing`` from this exact stream query
    (``worker/v2/aclgraph_utils.ModelWithContext.forward``), so the two agree
    by construction.  The flag is read first because it is cheaper; the stream
    query is the fallback for a forward that runs while a capture is in
    progress but outside that wrapper -- taking the eager path there would
    submit the CPU MoE at capture time only, so every replay would run the
    layer with no CPU half at all and silently produce a wrong result.
    """
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        if bool(getattr(_EXTRA_CTX, "capturing", False)):
            return True
    except Exception:
        pass
    try:
        import torch_npu  # noqa: F401  -- registers the torch.npu namespace

        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


def _current_stream(device: torch.device):
    """The stream device work is currently being issued on."""
    if device.type == "npu":
        import torch_npu  # noqa: F401  -- registers the torch.npu namespace

        return torch.npu.current_stream(device)
    if device.type == "cuda":
        return torch.cuda.current_stream(device)
    raise TypeError(
        "kt-kernel hybrid execution expects NPU or CUDA input; "
        f"got device={device}")


def _ensure_subscribed(stream) -> None:
    """Register ``stream`` for callback reporting, tolerating a prior one.

    ``_launch_host_func`` rejects a stream that no thread has subscribed (ERR
    107015), while ``aclrtSubscribeReport`` rejects one that is *already*
    subscribed (ERR 107011) -- and ACL offers no way to ask which state a
    stream is in.  Other components subscribe this same compute stream
    (vLLM-Ascend's expert_offload_manager does it on the prefetch path), so
    "already subscribed" is treated as the state being asked for rather than
    tracking a private idea of the truth that can drift from the runtime's.
    """
    import torch_npu

    stream_id = int(stream.npu_stream)
    if stream_id in _SUBSCRIBED_STREAM_IDS:
        return
    try:
        torch_npu.npu._subscribe_report(stream)
    except RuntimeError as exc:
        text = str(exc)
        if "107011" not in text and "already" not in text.lower():
            raise
    _SUBSCRIBED_STREAM_IDS.add(stream_id)


class _KtKernelExpertTask:
    """One in-flight call using kt-kernel's per-layer pinned buffers."""

    def __init__(self, backend: "KtKernelCPUExpertBackend",
                 hidden_states: torch.Tensor, stream_handle: int) -> None:
        self._backend = backend
        self._hidden_states = hidden_states
        self._stream_handle = stream_handle
        self._result: torch.Tensor | None = None
        self._error: BaseException | None = None
        self._finished = False

    def wait(self) -> torch.Tensor:
        if not self._finished:
            try:
                self._result = self._backend._wrapper.sync_forward(
                    self._hidden_states, self._stream_handle)
            except BaseException as exc:
                self._error = exc
                raise
            finally:
                self._finished = True
                self._backend._task_finished(self)
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError(
                "kt-kernel task completed without an output tensor")
        return self._result


class _KtKernelGraphExpertTask:
    """One in-flight call whose CPU work runs in ACL stream host callbacks.

    The layer is recorded as a *pair* of callbacks.  The submit half fires once
    the inputs have been copied down and returns immediately; the join half
    fires where the result is first needed and waits there.  Everything the
    NPU executes in between overlaps the CPU MoE.

    ``wait`` registers the join half and then issues the device-side H2D of
    kt-kernel's pinned output buffer.  The stream orders that copy behind the
    join callback, so it cannot read a half-written ``output_cpu`` -- the join
    is what does the synchronising, not a side stream or an explicit event.
    """

    def __init__(self, backend: "KtKernelCPUExpertBackend",
                 hidden_states: torch.Tensor, stream) -> None:
        self._backend = backend
        self._hidden_states = hidden_states
        self._stream = stream
        self._result: torch.Tensor | None = None
        self._error: BaseException | None = None
        self._finished = False

    def wait(self) -> torch.Tensor:
        if not self._finished:
            try:
                self._backend._join_pinned_forward(self._stream)
                self._result = (
                    self._backend._wrapper.copy_forward_output_to_device(
                        self._hidden_states))
            except BaseException as exc:
                self._error = exc
                raise
            finally:
                self._finished = True
                self._backend._task_finished(self)
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError(
                "kt-kernel graph task completed without an output tensor")
        return self._result


def _current_stream_handle(device: torch.device) -> int:
    stream = _current_stream(device)
    if device.type == "npu":
        return int(stream.npu_stream)
    return int(stream.cuda_stream)


def _default_wrapper_factory(**kwargs: Any) -> Any:
    try:
        from kt_kernel import KTMoEWrapper
    except ImportError as exc:
        raise RuntimeError(
            "kt-kernel is not importable. Build the reference project's "
            "Ascend-enabled kt-kernel and add it to PYTHONPATH before enabling "
            "the CPU MoE backend.") from exc
    return KTMoEWrapper(method="LLAMAFILE", mode="inference", **kwargs)


class KtKernelCPUExpertBackend:
    """Adapt the reference kt-kernel LLAMAFILE wrapper to vLLM-Ascend.

    The CPU/NPU split is deliberately *not* configured here.
    ``HybridExpertExecutor.prepare()`` derives it from ``log2phy`` on every
    step and marks every NPU-owned route with a ``-1`` in ``cpu_topk_ids``;
    kt-kernel's ``should_skip_expert`` skips negative ids unconditionally.
    Passing a construction-time resident mask as well would add a second,
    frozen source of truth that can drift from ``log2phy`` -- and when it
    does, both sides skip the route and the expert vanishes silently.
    """

    def __init__(
        self,
        config: KtKernelBackendConfig,
        *,
        wrapper_factory: Callable[..., Any] = _default_wrapper_factory,
        stream_handle_provider: Callable[[torch.device], int] =
        _current_stream_handle,
    ) -> None:
        self.config = config
        self._stream_handle_provider = stream_handle_provider
        self._in_flight: (_KtKernelExpertTask
                          | _KtKernelGraphExpertTask | None) = None
        self._graph_methods_checked = False

        self._wrapper = wrapper_factory(
            layer_idx=config.layer_idx,
            num_experts=config.num_experts,
            num_experts_per_tok=config.top_k,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.intermediate_size,
            gpu_experts_mask=None,
            cpuinfer_threads=config.cpuinfer_threads,
            threadpool_count=config.threadpool_count,
            weight_path=config.resolved_weight_path(),
            chunked_prefill_size=config.max_num_tokens,
            cpu_save=False,
            max_deferred_experts_per_token=0,
            numa_nodes=(list(config.numa_nodes)
                        if config.numa_nodes is not None else None),
        )
        required_methods = ("load_weights", "submit_forward", "sync_forward")
        missing_methods = [
            name for name in required_methods
            if not callable(getattr(self._wrapper, name, None))
        ]
        if missing_methods:
            raise TypeError(
                "kt-kernel wrapper is missing required methods: "
                + ", ".join(missing_methods))
        # Per-layer GGUFs use logical expert order; kt-kernel's identity mapping
        # is therefore the correct default.  Every expert stays resident on the
        # CPU side -- the mask never gated weight loading.
        self._wrapper.load_weights(None)

    @property
    def max_num_tokens(self) -> int:
        """Capacity of kt-kernel's per-layer pinned token buffers."""
        return self.config.max_num_tokens

    def submit(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        if layer_idx != self.config.layer_idx:
            raise ValueError(
                f"backend belongs to layer {self.config.layer_idx}, "
                f"got submit for layer {layer_idx}")
        if self._in_flight is not None:
            raise RuntimeError(
                "kt-kernel backend received overlapping work for one layer; "
                "wait for the previous task before reusing its pinned buffers")
        if topk_ids.shape != topk_weights.shape:
            raise ValueError("topk ids and weights must have identical shapes")
        if topk_ids.ndim != 2 or topk_ids.shape[1] != self.config.top_k:
            raise ValueError(
                f"routing must have shape [T, {self.config.top_k}]; "
                f"got {tuple(topk_ids.shape)}")
        num_tokens = hidden_states.reshape(-1,
                                           hidden_states.shape[-1]).shape[0]
        if topk_ids.shape[0] != num_tokens:
            raise ValueError(
                "routing token count must match hidden states; "
                f"routes={topk_ids.shape[0]}, hidden={num_tokens}")
        if num_tokens > self.config.max_num_tokens:
            raise ValueError(
                f"token count {num_tokens} exceeds configured kt-kernel buffer "
                f"capacity {self.config.max_num_tokens}")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"hidden size {hidden_states.shape[-1]} does not match backend "
                f"configuration {self.config.hidden_size}")
        if topk_ids.device != hidden_states.device:
            raise ValueError(
                "topk ids and hidden states must be on the same device; "
                f"ids={topk_ids.device}, hidden={hidden_states.device}")
        if topk_weights.device != hidden_states.device:
            raise ValueError(
                "topk weights and hidden states must be on the same device; "
                f"weights={topk_weights.device}, hidden={hidden_states.device}")
        if topk_ids.dtype not in (torch.int8, torch.int16, torch.int32,
                                  torch.int64):
            raise TypeError(
                "topk ids must use a signed integer dtype; "
                f"got {topk_ids.dtype}")
        if not topk_weights.dtype.is_floating_point:
            raise TypeError(
                "topk weights must use a floating-point dtype; "
                f"got {topk_weights.dtype}")

        hidden_states = hidden_states.contiguous()
        if _graph_capture_active():
            return self._submit_graph(hidden_states, topk_ids, topk_weights)

        stream_handle = self._stream_handle_provider(hidden_states.device)
        self._wrapper.submit_forward(hidden_states, topk_ids, topk_weights,
                                     stream_handle)
        task = _KtKernelExpertTask(self, hidden_states, stream_handle)
        self._in_flight = task
        return task

    def _submit_graph(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> _KtKernelGraphExpertTask:
        """Record this layer's CPU MoE into the graph being captured.

        Nothing here may block the host.  ``submit_forward`` and
        ``sync_forward`` both drain the WorkerPool on the calling thread, so
        inside a capture they would run exactly once -- at capture time -- and
        every replay would then run the NPU half of the layer with no CPU half
        at all, which surfaces as a quietly wrong result rather than as a
        failure.
        """
        self._require_graph_methods()
        self._pin_graph_batch_size(hidden_states)

        # D2H into kt-kernel's pinned buffers.  An ordinary device op, so it is
        # recorded into the graph and re-runs on every replay.
        self._wrapper.copy_inputs_to_cpu_buffers(hidden_states, topk_ids,
                                                 topk_weights)

        import torch_npu

        stream = _current_stream(hidden_states.device)
        _ensure_subscribed(stream)
        # Submit half.  It only enqueues, so the NPU work captured after this
        # point runs while the WorkerPool chews on the layer.
        torch_npu.npu._launch_host_func(
            stream,
            self._launch_pinned_forward,
            (hidden_states, int(stream.npu_stream)),
        )
        task = _KtKernelGraphExpertTask(self, hidden_states, stream)
        self._in_flight = task
        return task

    def _launch_pinned_forward(self, args: tuple[torch.Tensor, int]) -> None:
        """Submit half: enqueue this layer's CPU MoE and return at once.

        Invoked once per graph replay on the ACL callback dispatch thread.
        The stream has already reached this point, so the D2H copies recorded
        by ``copy_inputs_to_cpu_buffers`` have landed and no device wait is
        needed here -- and none is legal either, since a ``synchronize`` on
        this stream would target a captured one (ERR 107027 / 107030).

        Deliberately does not wait.  Blocking here would put the CPU MoE back
        on the critical path of the graph, which is the serial behaviour this
        split exists to remove.
        """
        hidden_states, stream_handle = args
        self._wrapper.forward_on_pinned_buffers(hidden_states, stream_handle)

    def _drain_pinned_forward(self, _user_data: None) -> None:
        """Join half: block until this layer's CPU MoE has finished.

        This is where the graph finally waits on the CPU, so the stream
        carrying the captured H2D of ``output_cpu`` is ordered behind a
        finished buffer rather than a half-written one.
        """
        self._wrapper.drain_pinned_forward()

    def _join_pinned_forward(self, stream) -> None:
        """Register the join half, or run it inline when no capture is active.

        The inline branch is a safety net for a task waited on outside the
        capture window; without it the following H2D would read an output the
        WorkerPool may not have written yet.
        """
        if _graph_capture_active():
            import torch_npu

            _ensure_subscribed(stream)
            torch_npu.npu._launch_host_func(stream, self._drain_pinned_forward,
                                            None)
        else:
            self._wrapper.drain_pinned_forward()

    def _pin_graph_batch_size(self, hidden_states: torch.Tensor) -> None:
        """Register this batch size so its pinned buffers are never freed.

        Must run *before* the D2H, because registration only makes
        ``KExpertsCPUBuffer.get_buffer`` store the tuple it hands out -- the
        buffer the D2H writes into has to be the one that gets cached.
        """
        batch_size = int(
            hidden_states.view(-1, hidden_states.shape[-1]).shape[0])
        device_index = (hidden_states.device.index
                         if hidden_states.device.type != "cpu" else None)
        signature = (int(hidden_states.shape[-1]), self.config.top_k,
                     hidden_states.dtype, hidden_states.device.type,
                     device_index)
        previous = _GRAPH_BATCH_SIGNATURES.get(batch_size)
        if previous is not None and previous != signature:
            raise RuntimeError(
                "incompatible graph-captured CPU MoE buffer for token batch "
                f"{batch_size}: existing={previous}, requested={signature}")
        _GRAPH_PINNED_BATCH_SIZES.add(batch_size)
        _GRAPH_BATCH_SIGNATURES[batch_size] = signature
        # Reapply on every capture in case the process-wide cache was cleared.
        self._wrapper.set_capture_batch_sizes(sorted(_GRAPH_PINNED_BATCH_SIZES))

    def _require_graph_methods(self) -> None:
        """Fail loudly if this kt-kernel build cannot support graph capture."""
        if self._graph_methods_checked:
            return
        missing = [
            name for name in _GRAPH_REQUIRED_METHODS
            if not callable(getattr(self._wrapper, name, None))
        ]
        if missing:
            raise TypeError(
                "kt-kernel wrapper cannot run the CPU MoE inside a graph; "
                "missing methods: " + ", ".join(missing)
                + ". Rebuild kt-kernel with NPU graph host-callback support, "
                "or run this configuration with --enforce-eager.")
        self._graph_methods_checked = True

    def _task_finished(
            self, task: "_KtKernelExpertTask | _KtKernelGraphExpertTask") -> None:
        if self._in_flight is task:
            self._in_flight = None


__all__ = ["KtKernelBackendConfig", "KtKernelCPUExpertBackend"]
