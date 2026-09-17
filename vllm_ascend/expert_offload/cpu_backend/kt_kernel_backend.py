"""kt-kernel LLAMAFILE backend for CPU routed experts.

This adapter reuses ktransformers' pinned buffers, CPUInfer worker pool, and
GGUF/LLAMAFILE MoE implementation. The first integration targets eager decode;
graph host callbacks and side-stream scheduling are intentionally separate.
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


def _current_stream_handle(device: torch.device) -> int:
    if device.type == "npu":
        stream = torch.npu.current_stream(device)
        return int(stream.npu_stream)
    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
        return int(stream.cuda_stream)
    raise TypeError(
        "kt-kernel hybrid execution expects NPU or CUDA input; "
        f"got device={device}")


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
        self._in_flight: _KtKernelExpertTask | None = None

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
    ) -> _KtKernelExpertTask:
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
        stream_handle = self._stream_handle_provider(hidden_states.device)
        self._wrapper.submit_forward(hidden_states, topk_ids, topk_weights,
                                     stream_handle)
        task = _KtKernelExpertTask(self, hidden_states, stream_handle)
        self._in_flight = task
        return task

    def _task_finished(self, task: _KtKernelExpertTask) -> None:
        if self._in_flight is task:
            self._in_flight = None


__all__ = ["KtKernelBackendConfig", "KtKernelCPUExpertBackend"]
