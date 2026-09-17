"""CPU/NPU hybrid routed-expert execution contracts.

This module owns only the device split and task lifecycle.  CPU weight loading
and CPU kernels live behind :class:`CPUExpertBackend` and can be implemented
independently of the Ascend MoE pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class CPUExpertTask(Protocol):
    """One submitted CPU routed-expert computation.

    ``wait`` returns a token-major, route-weighted and expert-reduced tensor
    with shape ``[num_tokens, hidden_size]``.  The tensor may be on CPU or NPU.
    """

    def wait(self) -> torch.Tensor:
        """Wait for CPU computation and return its routed-expert output."""
        ...


@runtime_checkable
class CPUExpertBackend(Protocol):
    """Backend implemented by a concrete CPU MoE kernel integration."""

    @property
    def max_num_tokens(self) -> int:
        """Largest token batch accepted by one submitted task."""
        ...

    def submit(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> CPUExpertTask:
        """Submit the CPU-owned routes without blocking NPU execution.

        ``topk_ids`` contains global logical expert ids for CPU routes and -1
        for routes owned by NPU.  The backend must skip negative ids.  It owns
        any D2H transfer and must apply each non-zero routing weight exactly
        once before reducing expert outputs into token-major order.
        """
        ...


@dataclass(frozen=True, slots=True)
class HybridExpertPlan:
    """NPU-safe routing plus the concurrently submitted CPU task."""

    npu_topk_ids: torch.Tensor
    npu_topk_weights: torch.Tensor
    cpu_topk_ids: torch.Tensor
    cpu_topk_weights: torch.Tensor
    cpu_task: CPUExpertTask


class HybridExpertExecutor:
    """Split selected routes according to the current NPU resident map."""

    def __init__(self, backend: CPUExpertBackend, layer_idx: int) -> None:
        if not isinstance(backend, CPUExpertBackend):
            raise TypeError("backend does not implement CPUExpertBackend")
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative; got {layer_idx}")
        self.backend = backend
        self.layer_idx = layer_idx

    def prepare(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        log2phy: torch.Tensor,
    ) -> HybridExpertPlan:
        """Submit CPU misses and produce a safe zero-weight NPU route.

        Logical ids remain unchanged.  The existing Ascend communication path
        maps them through ``log2phy`` and clamps unmapped ids to physical slot
        zero.  Zeroing their weights makes that placeholder slot contribute
        nothing while keeping every routed id valid for NPU operators.
        """
        if topk_ids.shape != topk_weights.shape:
            raise ValueError(
                "topk ids and weights must have the same shape; "
                f"got ids={tuple(topk_ids.shape)}, "
                f"weights={tuple(topk_weights.shape)}")
        if topk_ids.ndim != 2:
            raise ValueError(
                f"topk routing must be rank 2; got rank={topk_ids.ndim}")
        if log2phy.ndim != 1:
            raise ValueError(
                f"log2phy must be rank 1; got rank={log2phy.ndim}")

        physical_ids = log2phy[topk_ids]
        npu_route_mask = physical_ids >= 0
        zero = torch.zeros((), dtype=topk_weights.dtype,
                           device=topk_weights.device)
        npu_topk_weights = torch.where(
            npu_route_mask, topk_weights, zero)
        cpu_topk_weights = torch.where(
            npu_route_mask, zero, topk_weights)
        if topk_ids.dtype not in (torch.int8, torch.int16, torch.int32,
                                  torch.int64):
            raise TypeError(
                "topk ids must use a signed integer dtype so -1 can mark "
                f"NPU-owned routes; got {topk_ids.dtype}")
        cpu_topk_ids = torch.where(
            npu_route_mask,
            torch.full((), -1, dtype=topk_ids.dtype,
                       device=topk_ids.device),
            topk_ids,
        )

        cpu_task = self.backend.submit(
            layer_idx=self.layer_idx,
            hidden_states=hidden_states,
            topk_ids=cpu_topk_ids,
            topk_weights=cpu_topk_weights,
        )
        return HybridExpertPlan(
            npu_topk_ids=topk_ids,
            npu_topk_weights=npu_topk_weights,
            cpu_topk_ids=cpu_topk_ids,
            cpu_topk_weights=cpu_topk_weights,
            cpu_task=cpu_task,
        )

    @staticmethod
    def finish(
        npu_routed_out: torch.Tensor,
        cpu_task: CPUExpertTask,
    ) -> torch.Tensor:
        """Wait for CPU output, validate its contract and merge on NPU."""
        cpu_routed_out = cpu_task.wait()
        if not isinstance(cpu_routed_out, torch.Tensor):
            raise TypeError("CPU expert task must return a torch.Tensor")
        if cpu_routed_out.shape != npu_routed_out.shape:
            raise ValueError(
                "CPU and NPU routed outputs must have the same shape; "
                f"cpu={tuple(cpu_routed_out.shape)}, "
                f"npu={tuple(npu_routed_out.shape)}")
        cpu_routed_out = cpu_routed_out.to(
            device=npu_routed_out.device,
            dtype=npu_routed_out.dtype,
            non_blocking=cpu_routed_out.device.type == "cpu",
        )
        return npu_routed_out + cpu_routed_out


def maybe_prepare_hybrid_routes(
    layer: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    log2phy: torch.Tensor,
    max_tokens: int | None = None,
) -> HybridExpertPlan | None:
    """Split this layer's routes onto CPU and NPU, if a backend owns it.

    Returns ``None`` when the layer has no attached executor, or when the step
    is larger than ``max_tokens`` -- the caller must then run the existing
    weight-paging path instead.  A returned plan and paging are mutually
    exclusive: ``update_weights`` rewrites the very ``log2phy`` this split is
    derived from, so doing both lets the CPU and NPU halves disagree about who
    owns a route.  A route both halves skip contributes nothing to the output
    and raises nothing.
    """
    executor = get_hybrid_expert_executor(layer)
    if executor is None:
        return None
    backend_max_tokens = executor.backend.max_num_tokens
    if backend_max_tokens <= 0:
        raise ValueError(
            "CPU expert backend max_num_tokens must be positive; "
            f"got {backend_max_tokens}")
    effective_max_tokens = backend_max_tokens
    if max_tokens is not None:
        effective_max_tokens = min(max_tokens, backend_max_tokens)
    if topk_ids.shape[0] > effective_max_tokens:
        return None
    return executor.prepare(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        log2phy=log2phy,
    )


def attach_hybrid_expert_executor(
    layer: torch.nn.Module,
    backend: CPUExpertBackend,
    layer_idx: int,
) -> HybridExpertExecutor:
    """Attach a CPU backend to one routed-expert layer.

    A future CPU kernel integration calls this after its layer weights are
    ready.  Until then the layer has no executor and existing offload behavior
    remains unchanged.
    """
    executor = HybridExpertExecutor(backend, layer_idx)
    layer._hybrid_expert_executor = executor
    return executor


def get_hybrid_expert_executor(
    layer: torch.nn.Module,
) -> HybridExpertExecutor | None:
    """Return a previously attached executor, if CPU compute is available."""
    executor = getattr(layer, "_hybrid_expert_executor", None)
    if executor is not None and not isinstance(executor, HybridExpertExecutor):
        raise TypeError("layer._hybrid_expert_executor has an invalid type")
    return executor


__all__ = [
    "CPUExpertBackend",
    "CPUExpertTask",
    "HybridExpertExecutor",
    "HybridExpertPlan",
    "attach_hybrid_expert_executor",
    "get_hybrid_expert_executor",
    "maybe_prepare_hybrid_routes",
]
