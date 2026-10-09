"""kt-kernel LLAMAFILE backend for CPU routed experts.

This adapter reuses ktransformers' pinned buffers, CPUInfer worker pool, and
GGUF/LLAMAFILE MoE implementation.  Two execution modes are supported:

* **eager** -- submit stages inputs and queues CPU work; wait drains the
  worker pool and copies the output back to the device.
* **graph** -- a shared per-device CPU communication stream waits for input readiness,
  records D2H, submit and drain host callbacks, then H2D and output readiness.
  Expert-transfer mode defers drain/output H2D until NPU work is recorded,
  allowing the weight-loading callback to progress after CPU submission.
  The compute stream runs NPU experts concurrently and joins the output event
  only where the CPU result is consumed. All dependencies replay with the graph.

The mode is chosen per call from whether the current stream is being captured,
so one backend serves a warmup, a capture, and every replay without being told
which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch


def _eager_trace(event: str, **fields: Any) -> None:
    path = os.environ.get("VLLM_ASCEND_CPU_MOE_TRACE")
    if not path:
        return
    # Shared NFS writes can stall the runtime callback dispatcher. Keep
    # diagnostics on the container's local temporary filesystem instead.
    if os.path.abspath(path).startswith("/mnt/share/"):
        path = f"/tmp/cpu_moe_trace_{os.getpid()}.jsonl"
    # ts_ns is CLOCK_MONOTONIC: durations stay correct if the wall clock is
    # stepped. wall_ns is CLOCK_REALTIME: it shares its epoch with the Ascend
    # profiler, so the spans can be placed on that timeline.
    record = {
        "event": event,
        "ts_ns": time.perf_counter_ns(),
        "wall_ns": time.time_ns(),
        "pid": os.getpid(),
        "tid": threading.get_native_id(),
    }
    record.update(fields)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass


def _cpu_affinity() -> list[int] | None:
    """CPUs the calling thread may run on, or None where the OS will not say."""
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _distinct_ids(ids: torch.Tensor) -> list[int]:
    """Sorted distinct non-negative expert ids in a routed-id tensor."""
    return sorted({int(x) for x in ids.reshape(-1) if int(x) >= 0})


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

# One CPU communication stream per process/device, shared by all layer backends.
# Each layer still owns its capture events and task/buffer references.
_CPU_GRAPH_STREAMS: dict[int, Any] = {}
_CPU_GRAPH_STREAMS_LOCK = threading.Lock()


def _shared_cpu_graph_stream(device: torch.device):
    if device.type != "npu":
        raise ValueError("CPU graph communication requires an NPU device")
    device_index = device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    with _CPU_GRAPH_STREAMS_LOCK:
        stream = _CPU_GRAPH_STREAMS.get(device_index)
        if stream is None:
            stream = torch.npu.Stream(device=device_index)
            _CPU_GRAPH_STREAMS[device_index] = stream
        return stream


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
        self._created_ns = time.perf_counter_ns()

    def wait(self) -> torch.Tensor:
        if not self._finished:
            sync_begin = time.perf_counter_ns()
            _eager_trace("cpu_moe.sync_begin", layer=self._backend.config.layer_idx, task_created_ns=self._created_ns, wait_queue_ns=sync_begin - self._created_ns)
            try:
                self._result = self._backend._wrapper.sync_forward(
                    self._hidden_states, self._stream_handle)
            except BaseException as exc:
                self._error = exc
                raise
            finally:
                sync_end = time.perf_counter_ns()
                _eager_trace("cpu_moe.sync_end", layer=self._backend.config.layer_idx, sync_duration_ns=sync_end - sync_begin, total_task_ns=sync_end - self._created_ns)
                self._finished = True
                self._backend._task_finished(self)
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError(
                "kt-kernel task completed without an output tensor")
        return self._result


class _KtKernelGraphExpertTask:
    """A graph task with CPU callbacks and transfers on a separate stream.

    Input readiness forks the CPU branch from the compute stream. Both host
    callbacks and output H2D run on the CPU branch; wait only joins its output
    readiness back into the compute stream before the routed results merge.
    """

    def __init__(self, backend: "KtKernelCPUExpertBackend",
                 hidden_states: torch.Tensor, stream, output_ready,
                 result: torch.Tensor | None, inputs_ready,
                 cpu_stream=None, submitted_event=None) -> None:
        self._backend = backend
        self._hidden_states = hidden_states
        self._stream = stream
        # Retain events and buffers for the capture/replay lifetime.
        self._output_ready = output_ready
        self._inputs_ready = inputs_ready
        self._cpu_stream = cpu_stream
        self.submitted_event = submitted_event
        self._npu_done = None
        self._result: torch.Tensor | None = result
        self._error: BaseException | None = None
        self._finished = False

    def wait(self) -> torch.Tensor:
        if not self._finished:
            try:
                if self._cpu_stream is not None:
                    # In transfer mode a blocking join must not occupy the
                    # callback dispatcher before the weight callback/NPU phase.
                    # Record before waiting: safe on the first graph replay.
                    self._npu_done = torch.npu.Event()
                    self._stream.record_event(self._npu_done)
                    with torch.npu.stream(self._cpu_stream):
                        self._cpu_stream.wait_event(self._npu_done)
                        self._backend._join_pinned_forward(self._cpu_stream)
                        self._result = self._backend._wrapper.copy_forward_output_to_device(
                            self._hidden_states)
                        self._cpu_stream.record_event(self._output_ready)
                # A graph device dependency, not a host-side synchronize.
                # 给主计算流添加设备依赖：消费 CPU 结果前，等通信流 H2D 完成。
                # 不是让当前 Python 线程同步等待计算。
                self._stream.wait_event(self._output_ready)
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
        self._graph_tasks: list[_KtKernelGraphExpertTask] = []
        # Set by ``_submit_graph`` at capture and read by the join half, which
        # has no task object of its own to carry the timestamp.
        self._graph_task_created_ns: int | None = None

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
        defer_graph_join: bool = False,
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
            return self._submit_graph(hidden_states, topk_ids, topk_weights,
                                      defer_graph_join=defer_graph_join)

        stream_handle = self._stream_handle_provider(hidden_states.device)
        submit_begin = time.perf_counter_ns()
        self._wrapper.submit_forward(hidden_states, topk_ids, topk_weights,
                                     stream_handle)
        submit_end = time.perf_counter_ns()
        try:
            expert_ids = _distinct_ids(topk_ids.detach().cpu())
        except Exception:
            expert_ids = []
        _eager_trace("cpu_moe.submit_end", layer=layer_idx, tokens=num_tokens, experts=expert_ids, submit_duration_ns=submit_end-submit_begin, cpu_affinity=_cpu_affinity(), stream_handle=stream_handle)
        task = _KtKernelExpertTask(self, hidden_states, stream_handle)
        self._in_flight = task
        return task

    def _submit_graph(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        defer_graph_join: bool = False,
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

        import torch_npu

        # 旧版：D2H -> 提交回调 -> NPU 专家 -> 等待回调 -> H2D，都在主计算流。
        # 现在：主计算流算 NPU 专家；共享通信流负责 D2H、两个回调和 H2D。
        # 此处捕获图，记录执行顺序；两个回调在每次 replay 时执行。
        stream = _current_stream(hidden_states.device)
        cpu_stream = _shared_cpu_graph_stream(hidden_states.device)
        inputs_ready = torch.npu.Event()
        output_ready = torch.npu.Event()
        submitted_event = torch.npu.Event() if defer_graph_join else None
        # 输入及路由就绪后记录事件，通信流等此事件才开始拷贝。
        # with 切换后续设备操作的入队流，本身不会新建 Python 线程。
        stream.record_event(inputs_ready)
        with torch.npu.stream(cpu_stream):
            # Fork after routing/input production; compute stream can proceed
            # directly to NPU experts without waiting for either host callback.
            cpu_stream.wait_event(inputs_ready)
            # D2H：把 hidden states、专家 ID 和权重拷到 CPU 缓冲区。
            self._wrapper.copy_inputs_to_cpu_buffers(hidden_states, topk_ids,
                                                     topk_weights)
            _ensure_subscribed(cpu_stream)
            # 前面的 D2H 完成后，运行时的 CPU 回调线程才执行提交函数。
            # 提交只把 CPU 任务入队，不等待整层计算完成。
            torch_npu.npu._launch_host_func(
                cpu_stream, self._launch_pinned_forward,
                (hidden_states, int(cpu_stream.npu_stream)),
            )
            if defer_graph_join:
                # Nonblocking CPU submission completes before main-stream H2D.
                # wait() will record drain/output H2D after NPU computation.
                cpu_stream.record_event(submitted_event)
                result = None
            else:
                # Preserve the original CPU-only path: return results ASAP.
                self._join_pinned_forward(cpu_stream)
                result = self._wrapper.copy_forward_output_to_device(hidden_states)
                cpu_stream.record_event(output_ready)
        task = _KtKernelGraphExpertTask(
            self, hidden_states, stream, output_ready, result, inputs_ready,
            cpu_stream=cpu_stream if defer_graph_join else None,
            submitted_event=submitted_event)
        self._graph_task_created_ns = time.perf_counter_ns()
        self._in_flight = task
        self._graph_tasks.append(task)
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
        submit_begin = time.perf_counter_ns()
        callback_begin_wall_ns = time.time_ns()
        # Python -> C++ CPUInfer.submit -> MoE 入口 -> TaskQueue.enqueue。
        # 队列后台线程执行 forward，再调度 CPU 计算线程池。
        self._wrapper.forward_on_pinned_buffers(hidden_states, stream_handle)
        submit_end = time.perf_counter_ns()
        try:
            expert_ids = self._pinned_expert_ids(hidden_states)
        except Exception:
            expert_ids = []
        metadata_end = time.perf_counter_ns()
        _eager_trace("cpu_moe.submit_end", layer=self.config.layer_idx, tokens=int(hidden_states.view(-1, hidden_states.shape[-1]).shape[0]), experts=expert_ids, submit_duration_ns=submit_end-submit_begin, callback_begin_ns=submit_begin, callback_begin_wall_ns=callback_begin_wall_ns, submit_end_ns=submit_end, metadata_end_ns=metadata_end, metadata_duration_ns=metadata_end-submit_end, cpu_affinity=_cpu_affinity(), stream_handle=stream_handle)

    def _pinned_expert_ids(self, hidden_states: torch.Tensor) -> list[int]:
        """Distinct CPU expert ids this layer routed to, read on the host.

        Graph mode has no Python frame holding ``topk_ids``: the ids reach the
        CPU only through the D2H that ``copy_inputs_to_cpu_buffers`` records
        into kt-kernel's pinned buffers, and that op re-runs on every replay
        rather than at capture.  By the time the submit callback fires the copy
        has landed, so the ids can be read straight out of the buffer -- and
        reading beats copying them down again, which a capture would reject
        anyway.
        """
        from kt_kernel.experts_base import KExpertsCPUBuffer

        flat = hidden_states.view(-1, hidden_states.shape[-1])
        _, immediate_ids_cpu, _, _, _, _, _ = KExpertsCPUBuffer.get_buffer(
            flat, self.config.top_k)
        slot = self.config.layer_idx % KExpertsCPUBuffer.buffer_depth
        return _distinct_ids(immediate_ids_cpu[slot])

    def _drain_pinned_forward(self, _user_data: None) -> None:
        """Join half: block until this layer's CPU MoE has finished.

        This is where the graph finally waits on the CPU, so the stream
        carrying the captured H2D of ``output_cpu`` is ordered behind a
        finished buffer rather than a half-written one.
        """
        sync_begin = time.perf_counter_ns()
        self._wrapper.drain_pinned_forward()
        sync_end = time.perf_counter_ns()
        created = self._graph_task_created_ns
        _eager_trace("cpu_moe.sync_end", layer=self.config.layer_idx, sync_duration_ns=sync_end-sync_begin, total_task_ns=sync_end-created if created is not None else None)

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
