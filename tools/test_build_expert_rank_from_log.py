import pytest

from build_expert_rank_from_log import build_expert_rank, parse_lines


def test_parses_accumulates_and_normalizes_complete_counts():
    lines = [
        "prefix [EXPERT-STATS] phase=decode layer=0 "
        "global_counts=[2, 0, 1, 1]",
        "prefix [EXPERT-STATS] phase=prefill layer=0 "
        "global_counts=[100, 100, 100, 100]",
        "prefix [EXPERT-STATS] phase=decode layer=0 "
        "global_counts=[0, 2, 1, 1]",
        "prefix [EXPERT-STATS] phase=decode layer=1 "
        "global_counts=[0, 3, 1, 0]",
    ]

    totals = parse_lines(lines)
    result = build_expert_rank(totals)

    assert totals == {0: [2, 2, 2, 2], 1: [0, 3, 1, 0]}
    assert result == {
        "0": [[0, 0.25], [1, 0.25], [2, 0.25], [3, 0.25]],
        "1": [[1, 0.75], [2, 0.25]],
    }


def test_parses_sparse_selected_counts_and_ignores_prefill():
    lines = [
        "[EXPERT-STATS] phase=decode layer=0 num_experts=4 "
        "selected_counts=[[0, 2], [2, 1], [3, 1]]",
        "[EXPERT-STATS] phase=prefill layer=0 num_experts=4 "
        "selected_counts=[[1, 100]]",
        "[EXPERT-STATS] phase=decode layer=0 num_experts=4 "
        "selected_counts=[[1, 2], [2, 1], [3, 1]]",
    ]

    totals = parse_lines(lines)
    result = build_expert_rank(totals)

    assert totals == {0: [2, 2, 2, 2]}
    assert result == {
        "0": [[0, 0.25], [1, 0.25], [2, 0.25], [3, 0.25]],
    }


def test_rejects_expert_count_changes_within_a_layer():
    lines = [
        "[EXPERT-STATS] phase=decode layer=0 global_counts=[1, 2]",
        "[EXPERT-STATS] phase=decode layer=0 global_counts=[1, 2, 3]",
    ]

    with pytest.raises(ValueError, match="expert count changed"):
        parse_lines(lines)
