import torch

from vllm_ascend.expert_offload.hybrid_executor import HybridExpertExecutor


class _CompletedTask:

    def __init__(self, output: torch.Tensor) -> None:
        self.output = output

    def wait(self) -> torch.Tensor:
        return self.output


class _RecordingBackend:

    def __init__(self, max_num_tokens=16) -> None:
        self.call = None
        self.max_num_tokens = max_num_tokens

    def submit(self, **kwargs):
        self.call = kwargs
        hidden_states = kwargs["hidden_states"]
        return _CompletedTask(torch.zeros_like(hidden_states))


def test_prepare_splits_cpu_and_npu_routes_by_log2phy():
    backend = _RecordingBackend()
    executor = HybridExpertExecutor(backend, layer_idx=4)
    hidden_states = torch.randn(2, 8)
    topk_ids = torch.tensor([[0, 2, 3], [3, 1, 2]])
    topk_weights = torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]])
    log2phy = torch.tensor([1, -1, -1, 0], dtype=torch.int32)

    plan = executor.prepare(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        log2phy=log2phy,
    )

    torch.testing.assert_close(plan.npu_topk_ids, topk_ids)
    torch.testing.assert_close(
        plan.npu_topk_weights,
        torch.tensor([[0.5, 0.0, 0.2], [0.6, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        plan.cpu_topk_ids,
        torch.tensor([[-1, 2, -1], [-1, 1, 2]]),
    )
    torch.testing.assert_close(
        plan.cpu_topk_weights,
        torch.tensor([[0.0, 0.3, 0.0], [0.0, 0.3, 0.1]]),
    )
    torch.testing.assert_close(
        backend.call["topk_weights"],
        torch.tensor([[0.0, 0.3, 0.0], [0.0, 0.3, 0.1]]),
    )
    assert backend.call["layer_idx"] == 4
    assert backend.call["hidden_states"] is hidden_states
    torch.testing.assert_close(
        backend.call["topk_ids"],
        torch.tensor([[-1, 2, -1], [-1, 1, 2]]))


def test_finish_adds_token_major_cpu_output():
    npu_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    cpu_output = torch.tensor([[0.5, 1.0], [1.5, 2.0]])

    output = HybridExpertExecutor.finish(
        npu_output, _CompletedTask(cpu_output))

    torch.testing.assert_close(
        output, torch.tensor([[1.5, 3.0], [4.5, 6.0]]))


def test_finish_rejects_wrong_output_shape():
    npu_output = torch.zeros(2, 8)
    task = _CompletedTask(torch.zeros(1, 8))

    try:
        HybridExpertExecutor.finish(npu_output, task)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("expected shape validation to fail")


def test_prepare_falls_back_when_backend_capacity_is_exceeded():
    from vllm_ascend.expert_offload.hybrid_executor import (
        attach_hybrid_expert_executor,
        maybe_prepare_hybrid_routes,
    )

    layer = torch.nn.Module()
    backend = _RecordingBackend(max_num_tokens=1)
    attach_hybrid_expert_executor(layer, backend, layer_idx=4)

    plan = maybe_prepare_hybrid_routes(
        layer,
        hidden_states=torch.randn(2, 8),
        topk_ids=torch.tensor([[0, 1], [1, 0]]),
        topk_weights=torch.tensor([[0.6, 0.4], [0.7, 0.3]]),
        log2phy=torch.tensor([0, -1], dtype=torch.int32),
        max_tokens=8,
    )

    assert plan is None
    assert backend.call is None
