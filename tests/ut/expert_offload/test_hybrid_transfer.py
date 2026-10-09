import sys
import io
import os
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.cpu_backend import kt_kernel_backend as cpu_module
from vllm_ascend.expert_offload.hybrid_executor import (
    HybridExpertExecutor,
    attach_hybrid_expert_executor,
    maybe_prepare_hybrid_routes,
)
from vllm_ascend.expert_offload.hybrid_transfer import (
    HybridExpertTransfer,
    _TransferState,
    select_transfer_expert,
)


@pytest.mark.parametrize('misses', range(7))
def test_any_cpu_miss_triggers_one_transfer(misses):
    ids = torch.arange(misses).reshape(1, -1)
    selected = select_transfer_expert(ids, torch.full((8,), -1), 2)
    assert selected == ((0, 0) if misses >= 1 else None)


def test_duplicates_and_negative_sentinels_do_not_inflate_misses():
    ids = torch.tensor([[-1, 0, 1, 2, 3, 0, 1, 2, 3]])
    assert select_transfer_expert(ids, torch.full((8,), -1), 2) == (0, 0)


def test_one_miss_requires_an_unused_slot():
    ids = torch.tensor([[0, 1, 2]])
    mapping = torch.tensor([0, 1, -1])
    assert select_transfer_expert(ids, mapping, 2) is None


def test_only_resident_routes_do_not_transfer():
    assert select_transfer_expert(torch.tensor([[0, 0]]), torch.tensor([0]), 2) is None


def test_slot_selection_protects_all_current_routes():
    ids = torch.tensor([[0, 2, 3, 4, 5, 6]])
    mapping = torch.tensor([0, 1, -1, -1, -1, -1, -1])
    assert select_transfer_expert(ids, mapping, 2) == (2, 1)
    assert select_transfer_expert(ids, mapping, 1) is None


def test_first_miss_follows_routing_order():
    ids = torch.tensor([[-1, 5, 4, 3, 2, 1, 0]])
    assert select_transfer_expert(ids, torch.full((6,), -1), 2) == (5, 0)


def make_manager(layer, load=None, synchronize=None):
    return SimpleNamespace(
        exclusive_dynamic_enabled=False,
        moe_layers=[layer], load_stream=object(),
        offload_config=SimpleNamespace(cpu_moe=SimpleNamespace(transfer_one_expert=True)),
        num_device_experts_for_layer=lambda _: 2,
        _load_expert_weights_into_slot=load or (lambda *args: None),
        _synchronize_h2d=synchronize or (lambda: None),
    )


def test_decision_does_not_load_weights_and_only_evicts_unused_slot(monkeypatch):
    layer = object()
    calls = []
    manager = make_manager(layer, lambda *args: calls.append(('load', args)),
                           lambda: calls.append(('sync',)))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(stream=lambda _: nullcontext()), raising=False)
    transfer = HybridExpertTransfer(manager, layer)
    state = SimpleNamespace(ids_h=torch.tensor([[0, 2, 3, 4, 5, 6]]),
                            map_h=torch.tensor([0, 1, -1, -1, -1, -1, -1]))
    transfer._enqueue(state)
    assert calls == []
    assert state.map_h.tolist() == [0, -1, 1, -1, -1, -1, -1]
    transfer._load_and_sync(state)
    assert calls == [('load', (layer, 0, 2, 1)), ('sync',)]
    # All slots are now protected by current routes, despite remaining misses.
    assert select_transfer_expert(state.ids_h, state.map_h, 2) is None


def test_no_selection_does_not_copy_or_sync():
    layer = object()
    calls = []
    transfer = HybridExpertTransfer(make_manager(layer, lambda *a: calls.append(a),
                                                lambda: calls.append('sync')), layer)
    transfer._load_and_sync(SimpleNamespace(selection=None))
    assert calls == []


def test_weight_phase_trace_records_selection_and_durations(monkeypatch):
    import vllm_ascend.expert_offload.hybrid_transfer as module
    records = []
    monkeypatch.setenv('VLLM_ASCEND_CPU_MOE_TRACE', '/unused/test.jsonl')
    monkeypatch.setattr(module, '_eager_trace', lambda event, **fields: records.append((event, fields)))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(stream=lambda _: nullcontext()), raising=False)
    layer = object()
    transfer = HybridExpertTransfer(make_manager(layer), layer)
    transfer._load_and_sync(SimpleNamespace(selection=(3, 1)))
    event, fields = records.pop()
    assert event == 'cpu_moe.transfer_weights'
    assert (fields['layer'], fields['expert'], fields['slot']) == (0, 3, 1)
    assert fields['callback_begin_ns'] <= fields['load_begin_ns'] <= fields['load_end_ns'] <= fields['sync_end_ns']
    assert fields['load_duration_ns'] + fields['sync_duration_ns'] <= fields['callback_work_duration_ns']
    transfer._load_and_sync(SimpleNamespace(selection=None))
    assert records[0][0] == 'cpu_moe.transfer_skipped'


@pytest.mark.parametrize('configured, expected', [
    ('/mnt/share/y50063808/diagnostic.jsonl', None),
    ('/tmp/diagnostic.jsonl', '/tmp/diagnostic.jsonl'),
])
def test_host_trace_uses_local_disk(monkeypatch, configured, expected):
    writes = []

    @contextmanager
    def writer(path, *args, **kwargs):
        output = io.StringIO()
        yield output
        writes.append((path, output.getvalue()))

    monkeypatch.setenv('VLLM_ASCEND_CPU_MOE_TRACE', configured)
    monkeypatch.setattr(cpu_module, 'open', writer, raising=False)
    cpu_module._eager_trace('test_event', layer=0)
    assert writes[0][0] == (expected or f'/tmp/cpu_moe_trace_{os.getpid()}.jsonl')
    assert 'test_event' in writes[0][1]


def test_single_miss_promoted_leaves_zero_cpu_routes(monkeypatch):
    monkeypatch.delenv('VLLM_ASCEND_CPU_MOE_TRACE', raising=False)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(stream=lambda _: nullcontext()), raising=False)
    layer = object()
    transfer = HybridExpertTransfer(make_manager(layer), layer)
    mapping = torch.tensor([0, 1, -1])

    def prepare(ids, log2phy):
        state = _TransferState(ids, log2phy, None, None, None, False, transfer._load_and_sync)
        transfer._enqueue(state)
        return state

    monkeypatch.setattr(transfer, 'prepare', prepare)

    def submit(**kwargs):
        assert kwargs['topk_ids'].tolist() == [[-1, -1]]
        assert torch.count_nonzero(kwargs['topk_weights']) == 0
        return SimpleNamespace(wait=lambda: torch.zeros_like(kwargs['hidden_states']))

    executor = HybridExpertExecutor(SimpleNamespace(max_num_tokens=1, submit=submit), 0)
    plan = executor.prepare(hidden_states=torch.ones(1, 4), topk_ids=torch.tensor([[2, 2]]),
                            topk_weights=torch.tensor([[0.4, 0.6]]), log2phy=mapping,
                            weight_transfer=transfer)
    assert mapping.tolist() == [-1, 1, 0]
    torch.testing.assert_close(plan.npu_topk_weights, torch.tensor([[0.4, 0.6]]))
    torch.testing.assert_close(executor.finish(torch.ones(1, 4), plan.cpu_task), torch.ones(1, 4))


def test_exclusive_storage_rejected():
    layer = object()
    manager = make_manager(layer)
    manager.exclusive_dynamic_enabled = True
    with pytest.raises(ValueError, match='replicated CPU weights'):
        HybridExpertTransfer(manager, layer)


def test_cpu_submission_precedes_h2d_and_promoted_routes_count_once():
    order = []
    ids = torch.tensor([[0, 1, 2, 3, 4, 0]])
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.2, 0.1]])
    mapping = torch.full((6,), -1, dtype=torch.int32)
    hidden = torch.zeros(1, 2)

    class Transfer:
        graph = False

        def prepare(self, topk_ids, log2phy):
            order.append('decision')
            log2phy[0] = 0
            return self

        def wait_for_weights(self, task):
            order.append('load_and_sync')

    class Backend:
        max_num_tokens = 1

        def submit(self, **kwargs):
            order.append('cpu_submit')
            assert 'defer_graph_join' not in kwargs
            self.ids, self.weights = kwargs['topk_ids'], kwargs['topk_weights']
            output = (self.weights * (self.ids.clamp_min(0) + 1)).sum().expand_as(hidden)
            return SimpleNamespace(wait=lambda: output)

    backend = Backend()
    executor = HybridExpertExecutor(backend, 0)
    plan = executor.prepare(hidden_states=hidden, topk_ids=ids, topk_weights=weights,
                            log2phy=mapping, weight_transfer=Transfer())
    assert order == ['decision', 'cpu_submit', 'load_and_sync']
    assert backend.ids.tolist() == [[-1, 1, 2, 3, 4, -1]]
    torch.testing.assert_close(plan.npu_topk_weights + backend.weights, weights)
    npu = (plan.npu_topk_weights * (ids + 1)).sum().expand_as(hidden)
    torch.testing.assert_close(executor.finish(npu, plan.cpu_task),
                               (weights * (ids + 1)).sum().expand_as(hidden))


def test_cpu_worker_stays_active_during_weight_loading(monkeypatch):
    # A real background worker cannot finish until H2D synchronization releases
    # it. An early CPU wait therefore fails instead of hiding serialization.
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    workers = []
    layer = object()
    mapping = torch.full((6,), -1, dtype=torch.int32)

    def load(*args):
        assert started.wait(2)
        assert not finished.is_set()

    def sync():
        assert started.is_set() and not finished.is_set()
        release.set()

    transfer = HybridExpertTransfer(make_manager(layer, load, sync), layer)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(stream=lambda _: nullcontext()), raising=False)

    def prepare(ids, log2phy):
        state = _TransferState(ids, log2phy, None, None, None, False, transfer._load_and_sync)
        transfer._enqueue(state)
        return state

    monkeypatch.setattr(transfer, 'prepare', prepare)

    class Backend:
        max_num_tokens = 1

        def submit(self, **kwargs):
            def compute():
                started.set()
                release.wait(3)
                finished.set()
            worker = threading.Thread(target=compute)
            workers.append(worker)
            worker.start()

            def wait():
                assert release.is_set(), 'CPU joined before weight loading'
                worker.join(2)
                assert finished.is_set()
                return torch.zeros_like(kwargs['hidden_states'])
            return SimpleNamespace(wait=wait)

    try:
        executor = HybridExpertExecutor(Backend(), 0)
        plan = executor.prepare(hidden_states=torch.ones(1, 4), topk_ids=torch.arange(6).reshape(1, -1),
                                topk_weights=torch.ones(1, 6), log2phy=mapping, weight_transfer=transfer)
        executor.finish(torch.ones(1, 4), plan.cpu_task)
    finally:
        release.set()
        for worker in workers:
            worker.join(3)


def test_manager_switch_disabled_and_capacity_fallback_do_not_construct_transfer(monkeypatch):
    layer = torch.nn.Module()
    calls = []
    backend = SimpleNamespace(max_num_tokens=1, submit=lambda **kw: calls.append(kw) or SimpleNamespace(wait=lambda: None))
    attach_hybrid_expert_executor(layer, backend, 0)
    manager = make_manager(layer)
    manager.offload_config.cpu_moe.transfer_one_expert = False
    args = dict(hidden_states=torch.ones(1, 4), topk_ids=torch.tensor([[0]]),
                topk_weights=torch.ones(1, 1), log2phy=torch.tensor([0]), transfer_manager=manager)
    plan = maybe_prepare_hybrid_routes(layer, **args)
    assert plan is not None and 'defer_graph_join' not in calls[0]
    manager.offload_config.cpu_moe.transfer_one_expert = True
    args.update(hidden_states=torch.ones(2, 4), topk_ids=torch.zeros(2, 1, dtype=torch.long),
                topk_weights=torch.ones(2, 1))
    assert maybe_prepare_hybrid_routes(layer, **args) is None
    assert len(calls) == 1
    assert layer._hybrid_expert_executor._weight_transfer is None


def test_manager_enables_and_reuses_transfer(monkeypatch):
    import vllm_ascend.expert_offload.hybrid_transfer as module
    layer = torch.nn.Module()
    count = []
    submissions = []

    class Transfer:
        graph = True

        def __init__(self, manager, owned_layer):
            assert owned_layer is layer
            count.append(manager)

        def prepare(self, *args):
            return self

        def wait_for_weights(self, task):
            assert submissions[-1]['defer_graph_join'] is True

    monkeypatch.setattr(module, 'HybridExpertTransfer', Transfer)
    backend = SimpleNamespace(max_num_tokens=1,
                              submit=lambda **kw: submissions.append(kw) or SimpleNamespace(wait=lambda: None))
    attach_hybrid_expert_executor(layer, backend, 0)
    manager = make_manager(layer)
    for _ in range(2):
        maybe_prepare_hybrid_routes(layer, hidden_states=torch.ones(1, 4), topk_ids=torch.tensor([[0]]),
                                    topk_weights=torch.ones(1, 1), log2phy=torch.tensor([0]), transfer_manager=manager)
    assert count == [manager]


@pytest.fixture
def fake_streams(monkeypatch):
    calls, events = [], []

    class Stream:
        npu_stream = 123

        def __init__(self, name):
            self.name = name

        def record_event(self, event):
            calls.append((self.name, 'record', event))

        def wait_event(self, event):
            calls.append((self.name, 'wait', event))

    main, cpu, scheduling = Stream('main'), Stream('cpu'), Stream('scheduling')
    active = [main]

    @contextmanager
    def context(stream):
        prev, active[0] = active[0], stream
        try:
            yield
        finally:
            active[0] = prev

    def event():
        value = object()
        events.append(value)
        return value

    def callback(stream, fn, args):
        calls.append((stream.name, fn.__name__))

    monkeypatch.setattr(cpu_module, '_current_stream', lambda _: main)
    monkeypatch.setattr(cpu_module, '_shared_cpu_graph_stream', lambda _: cpu)
    monkeypatch.setattr(cpu_module, '_ensure_subscribed', lambda _: None)
    monkeypatch.setattr(cpu_module, '_graph_capture_active', lambda: True)
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(npu=SimpleNamespace(_launch_host_func=callback)))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(Event=event, Stream=lambda **kw: scheduling,
        stream=context, current_stream=lambda *args: main, _launch_host_func=callback), raising=False)
    return SimpleNamespace(calls=calls, events=events, main=main, cpu=cpu, scheduling=scheduling, active=active)


def test_graph_submission_weight_wait_and_deferred_join_order(tmp_path, monkeypatch, fake_streams):
    from vllm_ascend.expert_offload.cpu_backend import KtKernelBackendConfig, KtKernelCPUExpertBackend
    f = fake_streams
    gguf = tmp_path / 'weights.gguf'
    gguf.touch()
    result = torch.ones(1, 4)
    wrapper = SimpleNamespace(load_weights=lambda _: None, submit_forward=lambda *a: None, sync_forward=lambda *a: None)
    wrapper.copy_inputs_to_cpu_buffers = lambda *args: f.calls.append((f.active[0].name, 'd2h'))

    def output(*args):
        f.calls.append((f.active[0].name, 'result_h2d'))
        return result

    wrapper.copy_forward_output_to_device = output
    backend = KtKernelCPUExpertBackend(KtKernelBackendConfig(0, 8, 2, 4, 256, str(gguf)),
                                       wrapper_factory=lambda **kw: wrapper, stream_handle_provider=lambda _: 123)
    monkeypatch.setattr(backend, '_require_graph_methods', lambda: None)
    monkeypatch.setattr(backend, '_pin_graph_batch_size', lambda _: None)
    task = backend._submit_graph(torch.ones(1, 4), torch.ones(1, 2), torch.ones(1, 2), defer_graph_join=True)
    inputs, output_ready, submitted = f.events
    assert f.calls == [('main', 'record', inputs), ('cpu', 'wait', inputs), ('cpu', 'd2h'),
                       ('cpu', '_launch_pinned_forward'), ('cpu', 'record', submitted)]
    state = _TransferState(None, None, None, None, f.scheduling, True, lambda _: None)
    state.wait_for_weights(task)
    assert f.calls[-2:] == [('main', 'wait', submitted), ('main', '<lambda>')]
    f.calls.append(('main', 'npu_experts'))
    assert task.wait() is result
    done = f.events[-1]
    assert f.calls[-7:] == [('main', 'npu_experts'), ('main', 'record', done), ('cpu', 'wait', done),
        ('cpu', '_drain_pinned_forward'), ('cpu', 'result_h2d'), ('cpu', 'record', output_ready),
        ('main', 'wait', output_ready)]
    # Every event is first recorded before any wait, including first replay.
    recorded = set()
    for call in f.calls:
        if len(call) == 3 and call[1] == 'record':
            recorded.add(call[2])
        if len(call) == 3 and call[1] == 'wait':
            assert call[2] in recorded
    count = len(f.calls)
    assert task.wait() is result and len(f.calls) == count
    assert backend._in_flight is None


def test_scheduling_stream_shared_by_manager_not_capture(fake_streams):
    manager = make_manager(object())
    layer = manager.moe_layers[0]
    first, second = HybridExpertTransfer(manager, layer), HybridExpertTransfer(manager, layer)
    device = SimpleNamespace(index=0)
    assert first._scheduling_stream(device) is second._scheduling_stream(device)
    assert first._states is not second._states


def test_graph_capture_uses_mutable_pinned_buffers(fake_streams, monkeypatch):
    f = fake_streams
    original_empty = torch.empty
    allocations = []

    def empty(*args, **kwargs):
        allocations.append((kwargs.pop('pin_memory', False), torch.is_inference_mode_enabled()))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, 'empty', empty)

    class DeviceTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type='npu', index=0)

    ids = torch.arange(6).reshape(1, -1).as_subclass(DeviceTensor)
    mapping = torch.full((6,), -1, dtype=torch.int32)
    layer = object()
    transfer = HybridExpertTransfer(make_manager(layer), layer)
    with torch.inference_mode():
        state = transfer.prepare(ids, mapping)
    assert allocations == [(True, False), (True, False)]
    assert not state.ids_h.is_inference() and not state.map_h.is_inference()
    assert f.calls == [('main', 'record', state.input_ready), ('scheduling', 'wait', state.input_ready),
                       ('scheduling', '_enqueue'), ('scheduling', 'record', state.decision_ready),
                       ('main', 'wait', state.decision_ready)]
    assert transfer._states == [state]
