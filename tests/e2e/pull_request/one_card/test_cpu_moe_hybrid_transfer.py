"""Actual NPU H2D/graph regression with a small synthetic CPU worker.

This checks scheduling and route ownership, not the real kt-kernel's numerics
or model-level performance. Run on an idle NPU, outside serving benchmarks.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import KtKernelCPUExpertBackend
from vllm_ascend.expert_offload.hybrid_executor import HybridExpertExecutor
from vllm_ascend.expert_offload.hybrid_transfer import (
    HybridExpertTransfer,
    select_transfer_expert,
)


@pytest.mark.parametrize("layer_count", (1, 2))
def test_changed_routes_and_cpu_h2d_overlap(layer_count):
    device = torch.device("npu", torch.npu.current_device())
    pool = ThreadPoolExecutor(max_workers=1)
    load_stream = torch.npu.Stream(device=device)
    # Empty SimpleNamespace instances compare equal; the real layers use
    # identity. Distinct objects keep manager.moe_layers.index(layer) correct.
    layers = [object() for _ in range(layer_count)]
    slots = [torch.tensor([1., 2.], device=device) for _ in layers]
    source = torch.arange(1., 17., pin_memory=True)
    loads = []
    overlaps = []
    wrappers = []

    class Wrapper:
        def __init__(self):
            self.x = torch.empty((1, 32), pin_memory=True)
            self.ids = torch.empty((1, 6), dtype=torch.int64, pin_memory=True)
            self.weights = torch.empty((1, 6), pin_memory=True)
            self.out = torch.empty((1, 32), pin_memory=True)
            self.device_out = torch.empty((1, 32), device=device)
            self.started = threading.Event()
            self.loaded = threading.Event()
            self.future = None

        def copy_inputs_to_cpu_buffers(self, x, ids, weights):
            self.x.copy_(x, non_blocking=True)
            self.ids.copy_(ids, non_blocking=True)
            self.weights.copy_(weights, non_blocking=True)

        def forward_on_pinned_buffers(self, x, stream):
            self.started.clear()
            self.loaded.clear()
            inputs, ids, weights = self.x.clone(), self.ids.clone(), self.weights.clone()

            def compute():
                self.started.set()
                # A selected expert now owns an NPU route. Keep this synthetic
                # worker active through that expert's H2D to verify the fork.
                # With no selection, there is no weight callback to release it.
                if bool((ids < 0).any()):
                    # NPU routes also occur without a transfer, so completion
                    # is released by the drain in that case below.
                    self.loaded.wait(5)
                gains = (ids.clamp_min(0) + 1) * (ids >= 0)
                self.out.copy_(inputs * (weights * gains).sum())

            self.future = pool.submit(compute)

        def drain_pinned_forward(self):
            self.loaded.set()
            self.future.result(timeout=10)

        def copy_forward_output_to_device(self, x):
            self.device_out.copy_(self.out, non_blocking=True)
            return self.device_out

    def load(layer, idx, eid, slot):
        wrapper = wrappers[idx]
        # Record callback evidence and assert on the test thread. Exceptions
        # escaping a runtime host callback can obscure the original failure.
        started = wrapper.started.wait(5)
        overlaps.append((idx, started, wrapper.future is not None and not wrapper.future.done()))
        slots[idx][slot:slot + 1].copy_(source[eid:eid + 1], non_blocking=True)
        loads.append((idx, eid, slot))

    # No blocking CPU join is queued until after the main-stream weight
    # callback and NPU work, so these callbacks can complete on first replay.
    manager = SimpleNamespace(exclusive_dynamic_enabled=False, moe_layers=layers,
                              load_stream=load_stream, num_device_experts_for_layer=lambda _: 2,
                              _load_expert_weights_into_slot=load,
                              _synchronize_h2d=load_stream.synchronize)
    executors, transfers, mappings = [], [], []
    for idx, layer in enumerate(layers):
        wrapper = Wrapper()
        wrappers.append(wrapper)
        backend = KtKernelCPUExpertBackend.__new__(KtKernelCPUExpertBackend)
        backend.config = SimpleNamespace(layer_idx=idx, top_k=6, max_num_tokens=1, hidden_size=32)
        backend._wrapper = wrapper
        backend._in_flight = None
        backend._graph_tasks = []
        backend._graph_task_created_ns = None
        backend._require_graph_methods = lambda: None
        backend._pin_graph_batch_size = lambda x: None
        backend._pinned_expert_ids = lambda x, w=wrapper: sorted(set(w.ids.reshape(-1).tolist()))
        executors.append(HybridExpertExecutor(backend, idx))
        transfers.append(HybridExpertTransfer(manager, layer))
        mapping = torch.full((16,), -1, dtype=torch.int32, device=device)
        mapping[:2].copy_(torch.tensor([0, 1], dtype=torch.int32, device=device))
        mappings.append(mapping)

    ids = torch.tensor([[2, 3, 4, 5, 6, 7]], device=device)
    weights = torch.full((1, 6), 1. / 6, device=device)
    x = torch.ones((1, 32), device=device)

    def forward():
        outputs = []
        for idx, executor in enumerate(executors):
            plan = executor.prepare(hidden_states=x, topk_ids=ids, topk_weights=weights,
                                    log2phy=mappings[idx], weight_transfer=transfers[idx])
            physical = mappings[idx][plan.npu_topk_ids].clamp_min(0).long()
            npu = x * (slots[idx][physical] * plan.npu_topk_weights).sum()
            outputs.append(executor.finish(npu, plan.cpu_task))
        return outputs

    try:
        # Compile the small kernels before capture. No model is loaded.
        (x * (slots[0][ids.clamp_max(1)] * weights).sum()).sum()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = forward()
        torch.npu.synchronize()
        cases = [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7],
                 [8, 9, 10, 11, 12, 13], [8, 9, 10, 11, 12, 13], [0, 0, 1, 1, 2, 2],
                 [14, 14, 14, 14, 14, 14]]
        for step in range(24):
            route = cases[step % len(cases)]
            ids.copy_(torch.tensor([route], device=device))
            x.fill_(step + 1)
            expected_loads = []
            for idx, mapping in enumerate(mappings):
                selected = select_transfer_expert(torch.tensor([route]), mapping.cpu(), 2)
                if selected is not None:
                    expected_loads.append((idx, *selected))
            before = len(loads)
            before_overlap = len(overlaps)
            graph.replay()
            torch.npu.synchronize()
            assert loads[before:] == expected_loads
            assert all(started and active for _, started, active in overlaps[before_overlap:])
            expected = (step + 1) * sum(eid + 1 for eid in route) / 6
            for output in outputs:
                torch.testing.assert_close(output.cpu(), torch.full((1, 32), expected))
    finally:
        for wrapper in wrappers:
            wrapper.loaded.set()
        pool.shutdown(wait=True)
