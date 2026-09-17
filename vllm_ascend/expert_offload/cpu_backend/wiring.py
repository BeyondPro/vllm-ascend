"""Model-load-time wiring for the CPU routed-expert backend.

This is the only place that turns a ``CpuMoeConfig`` into an attached
executor.  It lives beside the backend rather than in ``hybrid_executor``
(which owns the contract and must stay importable without kt-kernel) or inside
``kt_kernel_backend`` (which owns a single layer's adapter).

Attachment happens once per MoE layer during ``AscendMoERunner.__init__``,
after the static resident map (``log2phy``) has been built.  That ordering is
load-bearing: the executor derives its CPU/NPU split from ``log2phy`` on every
step, so the map must be final before anything reads it.
"""

from __future__ import annotations

from typing import Any

from vllm_ascend.expert_offload.hybrid_executor import (
    CPUExpertBackend,
    attach_hybrid_expert_executor,
    get_hybrid_expert_executor,
)

from .kt_kernel_backend import KtKernelBackendConfig, KtKernelCPUExpertBackend


def model_layer_index(layer_name: str) -> int:
    """Resolve the MODEL layer index from a vLLM layer name.

    ``cpu_moe.layers`` and the GGUF ``blk.{layer_idx}`` prefix share the model
    layer namespace, which is *not* the MoE registration ordinal
    (``moe_instance_id``): models interleave dense layers with MoE ones, so the
    two indices diverge after the first dense block.  The layer name is the
    only place the model index survives to construction time.
    """
    from vllm.model_executor.models.utils import extract_layer_index

    try:
        return int(extract_layer_index(layer_name))
    except (AssertionError, ValueError, IndexError) as exc:
        raise ValueError(
            "cpu_moe.layers is expressed in MODEL layer indices, which are "
            "parsed from the layer name, but none could be parsed from "
            f"{layer_name!r}") from exc


def maybe_attach_cpu_expert_backend(
    *,
    routed_experts: Any,
    layer_name: str,
    moe_config: Any,
    cpu_moe_config: Any,
    enable_multi_card: bool,
    tp_size: int,
) -> CPUExpertBackend | None:
    """Attach a kt-kernel backend when CPU/NPU hybrid owns this layer.

    Returns ``None`` whenever hybrid is off or this layer is not selected; the
    caller then keeps the untouched weight-paging path.  Misconfiguration that
    would otherwise fail silently raises here instead -- a layer that pages
    experts in while a CPU backend also owns its routes would let the paging
    rewrite ``log2phy`` out from under the executor, and the two halves would
    then disagree about which side owns a route.
    """
    if not cpu_moe_config.enabled:
        return None

    existing = get_hybrid_expert_executor(routed_experts)
    if existing is not None:
        return existing.backend

    model_layer_idx = model_layer_index(layer_name)
    if not cpu_moe_config.includes(model_layer_idx):
        return None

    # Hybrid v1 is single-card decode.  Refuse the rest loudly rather than
    # attaching a backend whose routes would never reach it.
    if enable_multi_card or tp_size != 1 or moe_config.ep_size > 1:
        raise ValueError(
            "expert_offload_config.cpu_moe supports single-card decode only, "
            f"but layer {model_layer_idx} runs with enable_multi_card="
            f"{enable_multi_card}, tp_size={tp_size}, "
            f"ep_size={moe_config.ep_size}")

    config = KtKernelBackendConfig(
        # The MODEL layer index, matching both cpu_moe.layers and the GGUF
        # blk.{layer_idx} prefix.  kt-kernel's per-layer GGUFs are written in
        # logical expert order, so its identity expert mapping is correct here.
        layer_idx=model_layer_idx,
        # Every expert lives in the CPU-side GGUF; which of them actually
        # compute on CPU is decided per step from log2phy, not here.
        num_experts=moe_config.num_logical_experts,
        top_k=moe_config.experts_per_token,
        hidden_size=moe_config.hidden_dim,
        intermediate_size=moe_config.intermediate_size_per_partition,
        weight_path=cpu_moe_config.weight_path,
        cpuinfer_threads=cpu_moe_config.cpuinfer_threads,
        threadpool_count=cpu_moe_config.threadpool_count,
        max_num_tokens=cpu_moe_config.max_num_tokens,
        numa_nodes=cpu_moe_config.numa_nodes,
    )
    backend = KtKernelCPUExpertBackend(config)
    # Both sides take the model layer index: submit() cross-checks the index it
    # is handed against config.layer_idx and raises on a mismatch.
    attach_hybrid_expert_executor(routed_experts, backend, model_layer_idx)
    return backend


__all__ = ["maybe_attach_cpu_expert_backend", "model_layer_index"]
