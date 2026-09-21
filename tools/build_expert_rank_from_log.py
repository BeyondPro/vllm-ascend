#!/usr/bin/env python3
"""Build an offline expert hotness ranking from vLLM-Ascend logs."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Iterable


EXPERT_STATS_RE = re.compile(
    r"\[EXPERT-STATS\]\s+phase=(?P<phase>\w+)\s+"
    r"layer=(?P<layer>\d+)\s+"
    r"global_counts=(?P<counts>\[[^\]]*\])"
)
SPARSE_EXPERT_STATS_RE = re.compile(
    r"\[EXPERT-STATS\]\s+phase=(?P<phase>\w+)\s+"
    r"layer=(?P<layer>\d+)\s+"
    r"num_experts=(?P<num_experts>\d+)\s+"
    r"selected_counts=(?P<counts>\[[^\r\n]*\])"
)


def parse_lines(
    lines: Iterable[str],
    phase: str = "decode",
) -> dict[int, list[int]]:
    """Accumulate complete per-expert route counts from matching log lines."""
    totals: dict[int, list[int]] = {}
    for line in lines:
        sparse = SPARSE_EXPERT_STATS_RE.search(line)
        match = sparse or EXPERT_STATS_RE.search(line)
        if not match or match.group("phase") != phase:
            continue

        layer = int(match.group("layer"))
        parsed_counts = ast.literal_eval(match.group("counts"))
        if sparse:
            num_experts = int(match.group("num_experts"))
            counts = [0] * num_experts
            if not isinstance(parsed_counts, list):
                raise ValueError(
                    f"layer {layer}: selected_counts must be a list")
            for item in parsed_counts:
                if (not isinstance(item, list) or len(item) != 2
                        or not all(isinstance(value, int) for value in item)):
                    raise ValueError(
                        f"layer {layer}: invalid selected_counts item")
                expert_id, count = item
                if not 0 <= expert_id < num_experts or count < 0:
                    raise ValueError(
                        f"layer {layer}: invalid expert id or count")
                counts[expert_id] += count
        else:
            counts = parsed_counts
        if not isinstance(counts, list) or not all(
                isinstance(value, int) and value >= 0 for value in counts):
            raise ValueError(
                f"layer {layer}: global_counts must be non-negative integers")

        if layer not in totals:
            totals[layer] = [0] * len(counts)
        elif len(totals[layer]) != len(counts):
            raise ValueError(
                f"layer {layer}: expert count changed from "
                f"{len(totals[layer])} to {len(counts)}")

        for expert_id, count in enumerate(counts):
            totals[layer][expert_id] += count
    return totals


def merge_totals(
    destination: dict[int, list[int]],
    source: dict[int, list[int]],
) -> None:
    for layer, counts in source.items():
        if layer not in destination:
            destination[layer] = counts.copy()
            continue
        if len(destination[layer]) != len(counts):
            raise ValueError(
                f"layer {layer}: expert count differs across log files")
        for expert_id, count in enumerate(counts):
            destination[layer][expert_id] += count


def build_expert_rank(
    totals: dict[int, list[int]],
    precision: int = 6,
) -> dict[str, list[list[int | float]]]:
    """Normalize and rank experts using the hot_experts_file JSON schema."""
    result: dict[str, list[list[int | float]]] = {}
    for layer, counts in sorted(totals.items()):
        layer_total = sum(counts)
        if layer_total == 0:
            result[str(layer)] = []
            continue
        pairs = [
            [expert_id, round(count / layer_total, precision)]
            for expert_id, count in enumerate(counts)
            if count > 0
        ]
        pairs.sort(key=lambda item: (-item[1], item[0]))
        result[str(layer)] = pairs
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="one or more vLLM log files")
    parser.add_argument("-o", "--output", required=True,
                        help="output expert-rank JSON path")
    parser.add_argument("--phase", default="decode",
                        help="statistics phase to include (default: decode)")
    parser.add_argument("--precision", type=int, default=6,
                        help="decimal places for normalized weights")
    args = parser.parse_args()

    totals: dict[int, list[int]] = {}
    for log_path in args.logs:
        if log_path == "-":
            parsed = parse_lines(sys.stdin, phase=args.phase)
        else:
            with Path(log_path).open(
                    "r", encoding="utf-8", errors="replace") as log_file:
                parsed = parse_lines(log_file, phase=args.phase)
        merge_totals(totals, parsed)

    if not totals:
        raise ValueError(
            f"no [EXPERT-STATS] phase={args.phase} records found")

    result = build_expert_rank(totals, precision=args.precision)
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print(f"wrote {len(result)} layers to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
