"""Transfer one CPU miss while the other hybrid experts compute on CPU."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time

import torch

from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import _eager_trace


CPU_EXPERT_TRANSFER_MIN_MISSES = 1


def select_transfer_expert(topk_ids, log2phy, slot_count):
    """Pick the first distinct miss and a slot unused by ANY current route.

    Any CPU miss can trigger one transfer. Duplicate routes count once. If every
    slot is needed now, preserve the CPU path rather than evict an active
    resident expert. The incoming expert remains resident for future steps.
    """
    routed = list(dict.fromkeys(int(e) for e in topk_ids.reshape(-1).tolist()
                                if int(e) >= 0))
    misses = [e for e in routed if int(log2phy[e]) < 0]
    if len(misses) < CPU_EXPERT_TRANSFER_MIN_MISSES:
        return None
    protected_slots = {int(log2phy[e]) for e in routed
                       if int(log2phy[e]) >= 0}
    for slot in range(slot_count):
        if slot not in protected_slots:
            return misses[0], slot
    return None


@dataclass
class _TransferState:
    ids_h: torch.Tensor
    map_h: torch.Tensor
    input_ready: object
    decision_ready: object
    stream: object
    graph: bool
    load_and_sync: object
    selection: tuple | None = None

    def wait_for_weights(self, cpu_task):
        # Submit CPU work first. Reuse offload's H2D + synchronize on a host
        # callback in the main stream, so subsequent NPU kernels are gated.
        if self.graph:
            from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import _ensure_subscribed
            stream = torch.npu.current_stream()
            stream.wait_event(cpu_task.submitted_event)
            _ensure_subscribed(stream)
            torch.npu._launch_host_func(stream, self.load_and_sync, self)
        else:
            self.load_and_sync(self)


class HybridExpertTransfer:
    """Stage routing on a shared stream; reuse offload's H2D synchronization.

    Routing D2H -> choose/update mapping -> decision_ready -> CPU submit.
    The main stream then runs a load-and-sync callback before NPU kernels.
    CPU work runs asynchronously during H2D and NPU expert execution. Its
    blocking join is deferred until the NPU expert phase has completed.
    Every replay re-evaluates current routing and the >=1 miss threshold.
    """

    def __init__(self, manager, layer):
        if manager.exclusive_dynamic_enabled:
            raise ValueError(
                "cpu_moe.transfer_one_expert requires replicated CPU weights")
        self.manager = manager
        self.layer = layer
        self.layer_idx = manager.moe_layers.index(layer)
        self._states = []  # Retain pinned addresses/events baked into graphs.

    def _scheduling_stream(self, device):
        # One stream per manager/device; capture buffers/events remain private.
        streams = getattr(self.manager, "_hybrid_transfer_streams", None)
        if streams is None:
            streams = {}
            self.manager._hybrid_transfer_streams = streams
        index = device.index
        if index is None:
            index = torch.npu.current_device()
        if index not in streams:
            streams[index] = torch.npu.Stream(device=index)
        return streams[index]

    def _enqueue(self, state):
        state.selection = select_transfer_expert(
            state.ids_h, state.map_h,
            self.manager.num_device_experts_for_layer(self.layer_idx))
        if state.selection is None:
            return
        eid, slot = state.selection
        state.map_h[state.map_h == slot] = -1
        state.map_h[eid] = slot

    def _load_and_sync(self, state):
        tracing = bool(os.environ.get("VLLM_ASCEND_CPU_MOE_TRACE"))
        if tracing:
            callback_begin_ns = time.perf_counter_ns()
            callback_begin_wall_ns = time.time_ns()
        if state.selection is not None:
            eid, slot = state.selection
            if tracing:
                load_begin_ns = time.perf_counter_ns()
            with torch.npu.stream(self.manager.load_stream):
                self.manager._load_expert_weights_into_slot(
                    self.layer, self.layer_idx, eid, slot)
            if tracing:
                load_end_ns = time.perf_counter_ns()
            self.manager._synchronize_h2d()
            if tracing:
                sync_end_ns = time.perf_counter_ns()
                # One record after all measured phases; no file writes are
                # inserted between weight enqueue and synchronization.
                _eager_trace(
                    "cpu_moe.transfer_weights", layer=self.layer_idx,
                    expert=eid, slot=slot,
                    callback_begin_ns=callback_begin_ns,
                    callback_begin_wall_ns=callback_begin_wall_ns,
                    load_begin_ns=load_begin_ns, load_end_ns=load_end_ns,
                    sync_end_ns=sync_end_ns,
                    load_duration_ns=load_end_ns-load_begin_ns,
                    sync_duration_ns=sync_end_ns-load_end_ns,
                    callback_work_duration_ns=sync_end_ns-callback_begin_ns,
                )
        elif tracing:
            _eager_trace("cpu_moe.transfer_skipped", layer=self.layer_idx,
                         callback_begin_ns=callback_begin_ns,
                         callback_begin_wall_ns=callback_begin_wall_ns)

    def prepare(self, topk_ids, log2phy):
        from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import (
            _ensure_subscribed,
            _graph_capture_active,
        )

        if topk_ids.device.type != "npu":
            raise ValueError("hybrid expert transfer requires NPU routing")
        graph = _graph_capture_active()
        compute_stream = torch.npu.current_stream(topk_ids.device)
        transfer_stream = self._scheduling_stream(topk_ids.device)
        # Normal mutable tensors are required for replay callbacks even if
        # model warmup/capture runs under inference_mode.
        with torch.inference_mode(False):
            ids_h = torch.empty(topk_ids.shape, dtype=topk_ids.dtype,
                                device="cpu", pin_memory=True)
            map_h = torch.empty(log2phy.shape, dtype=log2phy.dtype,
                                device="cpu", pin_memory=True)
        state = _TransferState(
            ids_h, map_h, torch.npu.Event(), torch.npu.Event(),
            transfer_stream, graph, self._load_and_sync)
        if not graph:
            ids_h.copy_(topk_ids)
            map_h.copy_(log2phy)
            self._enqueue(state)
            log2phy.copy_(map_h, non_blocking=True)
            return state

        self._states.append(state)
        compute_stream.record_event(state.input_ready)
        with torch.npu.stream(transfer_stream):
            transfer_stream.wait_event(state.input_ready)
            ids_h.copy_(topk_ids, non_blocking=True)
            map_h.copy_(log2phy, non_blocking=True)
            _ensure_subscribed(transfer_stream)
            torch.npu._launch_host_func(transfer_stream, self._enqueue, state)
            log2phy.copy_(map_h, non_blocking=True)
            transfer_stream.record_event(state.decision_ready)
            # The main-stream load/sync callback is registered only after
            # the reduced CPU task has been submitted.
        compute_stream.wait_event(state.decision_ready)
        return state
