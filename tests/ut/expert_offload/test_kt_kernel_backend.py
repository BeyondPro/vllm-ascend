from pathlib import Path

import pytest
import torch

from vllm_ascend.expert_offload.cpu_backend import (
    KtKernelBackendConfig,
    KtKernelCPUExpertBackend,
)


class _FakeWrapper:

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.loaded_mapping = object()
        self.submitted = None
        self.synced = None

    def load_weights(self, mapping):
        self.loaded_mapping = mapping

    def submit_forward(self, hidden_states, topk_ids, topk_weights,
                       stream_handle):
        self.submitted = (hidden_states, topk_ids, topk_weights, stream_handle)

    def sync_forward(self, hidden_states, stream_handle):
        self.synced = (hidden_states, stream_handle)
        return torch.full_like(hidden_states, 2.0)


def _make_backend(tmp_path: Path):
    gguf = tmp_path / "layer3.gguf"
    gguf.touch()
    wrappers = []

    def factory(**kwargs):
        wrapper = _FakeWrapper(kwargs)
        wrappers.append(wrapper)
        return wrapper

    config = KtKernelBackendConfig(
        layer_idx=3,
        num_experts=8,
        top_k=2,
        hidden_size=4,
        intermediate_size=256,
        weight_path=str(tmp_path / "layer{layer_idx}.gguf"),
        max_num_tokens=2,
    )
    backend = KtKernelCPUExpertBackend(
        config,
        wrapper_factory=factory,
        stream_handle_provider=lambda _device: 123,
    )
    return backend, wrappers[0]


def test_backend_builds_llamafile_wrapper_and_runs_task(tmp_path):
    backend, wrapper = _make_backend(tmp_path)
    hidden_states = torch.randn(2, 4)
    topk_ids = torch.tensor([[-1, 1], [3, -1]])
    topk_weights = torch.tensor([[0.0, 0.4], [0.6, 0.0]])

    task = backend.submit(
        layer_idx=3,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    output = task.wait()

    assert wrapper.loaded_mapping is None
    # No construction-time resident mask: the -1 sentinel that
    # HybridExpertExecutor writes into cpu_topk_ids is the only CPU-side
    # routing authority.
    assert wrapper.kwargs["gpu_experts_mask"] is None
    assert wrapper.submitted == (hidden_states, topk_ids, topk_weights, 123)
    assert wrapper.synced == (hidden_states, 123)
    torch.testing.assert_close(output, torch.full_like(hidden_states, 2.0))


def test_backend_rejects_overlapping_tasks(tmp_path):
    backend, _wrapper = _make_backend(tmp_path)
    hidden_states = torch.randn(1, 4)
    ids = torch.tensor([[-1, 1]])
    weights = torch.tensor([[0.0, 1.0]])
    backend.submit(
        layer_idx=3,
        hidden_states=hidden_states,
        topk_ids=ids,
        topk_weights=weights,
    )

    with pytest.raises(RuntimeError, match="overlapping work"):
        backend.submit(
            layer_idx=3,
            hidden_states=hidden_states,
            topk_ids=ids,
            topk_weights=weights,
        )


def test_backend_takes_no_resident_mask(tmp_path):
    """A resident mask must not be accepted as a second routing authority.

    The split is derived from ``log2phy`` by HybridExpertExecutor on every
    step.  A frozen mask handed to the backend can disagree with ``log2phy``;
    kt-kernel then skips the route on the CPU side while the NPU side has
    already zeroed its weight, so the expert is dropped from both.
    """
    (tmp_path / "layer0.gguf").touch()
    config = KtKernelBackendConfig(
        layer_idx=0,
        num_experts=8,
        top_k=2,
        hidden_size=4,
        intermediate_size=256,
        weight_path=str(tmp_path / "layer{layer_idx}.gguf"),
    )
    with pytest.raises(TypeError):
        KtKernelCPUExpertBackend(config, torch.zeros(8, dtype=torch.bool))


def test_config_rejects_missing_gguf(tmp_path):
    config = KtKernelBackendConfig(
        layer_idx=0,
        num_experts=8,
        top_k=2,
        hidden_size=4,
        intermediate_size=256,
        weight_path=str(tmp_path / "missing-{layer_idx}.gguf"),
    )
    with pytest.raises(FileNotFoundError, match="GGUF not found"):
        config.resolved_weight_path()
