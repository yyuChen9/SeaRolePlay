#!/usr/bin/env python3
"""Aggregate per-chunk judge scores into a model comparison report.

Reports per-dimension means, the weighted overall, bootstrap confidence
intervals, and -- most importantly -- PAIRED differences between models.

Pairing matters here. Role-play scenarios differ wildly in difficulty, so the
variance between seeds dwarfs the variance between models. Comparing two
models' overall means with independent CIs will call almost everything a tie.
Since every model is evaluated on the same seeds, the per-seed difference is the
statistic with any power, and that is what `--baseline` reports.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DIMENSIONS = ("basics", "logic", "knowledge", "diversity", "content_logic", "interaction")
WEIGHTS = {
    "basics": 0.50 / 3,
    "logic": 0.50 / 3,
    "knowledge": 0.50 / 3,
    "diversity": 0.25 / 2,
    "content_logic": 0.25 / 2,
    "interaction": 0.25,
}
OBJECTIVES = {
    "Worlds (50%)": ("basics", "logic", "knowledge"),
    "Stories (25%)": ("diversity", "content_logic"),
    "Preferences (25%)": ("interaction",),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def session_scores(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Collapse chunks -> sessions. Returns model -> session_key -> dimension mean.

    The card scores at session level with chunked judging, so chunks of one
    session are averaged before anything else. Averaging chunks directly across
    sessions would silently weight long sessions more heavily.
    """
    chunks: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = f"{record['seed_id']}::{record['run_id']}"
        chunks[record["model_name"]][key].append(record)

    sessions: dict[str, dict[str, dict[str, float]]] = {}
    for model, by_session in chunks.items():
        sessions[model] = {}
        for key, group in by_session.items():
            means = {
                dimension: statistics.mean(chunk[dimension] for chunk in group)
                for dimension in DIMENSIONS
            }
            means["overall"] = sum(WEIGHTS[d] * means[d] for d in DIMENSIONS)
            means["_chunks"] = float(len(group))
            sessions[model][key] = means
    return sessions


def bootstrap_ci(
    values: list[float], iterations: int, rng: random.Random, alpha: float = 0.05
) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    means = []
    size = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()
    low = means[int((alpha / 2) * iterations)]
    high = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (low, high)


def paired_bootstrap(
    deltas: list[float], iterations: int, rng: random.Random
) -> tuple[float, float, float, float]:
    """Return (mean delta, CI low, CI high, two-sided p) for paired differences."""
    mean_delta = statistics.mean(deltas)
    low, high = bootstrap_ci(deltas, iterations, rng)

    # Permutation test: under the null the sign of each paired difference is
    # arbitrary, so flip signs at random and count how often |mean| is at least
    # as extreme as observed.
    size = len(deltas)
    at_least_as_extreme = 0
    for _ in range(iterations):
        flipped = sum(d if rng.random() < 0.5 else -d for d in deltas) / size
        if abs(flipped) >= abs(mean_delta):
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (iterations + 1)
    return mean_delta, low, high, p_value


def format_table(sessions: dict[str, dict[str, dict[str, float]]], iterations: int, rng: random.Random) -> str:
    lines = []
    header = f"{'model':<18}{'n':>4}{'overall':>10}{'95% CI':>18}"
    for dimension in DIMENSIONS:
        header += f"{dimension[:9]:>11}"
    lines.append(header)
    lines.append("-" * len(header))

    ranked = sorted(
        sessions.items(),
        key=lambda item: statistics.mean(s["overall"] for s in item[1].values()),
        reverse=True,
    )
    for model, by_session in ranked:
        overalls = [s["overall"] for s in by_session.values()]
        mean_overall = statistics.mean(overalls)
        low, high = bootstrap_ci(overalls, iterations, rng)
        row = f"{model:<18}{len(overalls):>4}{mean_overall:>10.2f}"
        row += f"{f'[{low:.2f}, {high:.2f}]':>18}"
        for dimension in DIMENSIONS:
            row += f"{statistics.mean(s[dimension] for s in by_session.values()):>11.2f}"
        lines.append(row)
    return "\n".join(lines)


def format_objectives(sessions: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = []
    header = f"{'model':<18}"
    for name in OBJECTIVES:
        header += f"{name:>20}"
    lines.append(header)
    lines.append("-" * len(header))
    for model, by_session in sessions.items():
        row = f"{model:<18}"
        for dimensions in OBJECTIVES.values():
            value = statistics.mean(
                statistics.mean(s[d] for d in dimensions) for s in by_session.values()
            )
            row += f"{value:>20.2f}"
        lines.append(row)
    return "\n".join(lines)


def format_paired(
    sessions: dict[str, dict[str, dict[str, float]]],
    baseline: str,
    iterations: int,
    rng: random.Random,
) -> str:
    if baseline not in sessions:
        return f"[WARN] baseline '{baseline}' 不在结果中，跳过配对比较"

    lines = []
    lines.append(f"配对比较（相对 {baseline}，按 seed+run 配对）")
    header = f"{'model':<18}{'n_pairs':>9}{'delta':>9}{'95% CI':>18}{'p':>9}  {'':<12}"
    lines.append(header)
    lines.append("-" * len(header))

    base_sessions = sessions[baseline]
    for model, by_session in sessions.items():
        if model == baseline:
            continue
        shared = sorted(set(base_sessions) & set(by_session))
        if len(shared) < 2:
            lines.append(f"{model:<18}{len(shared):>9}   配对样本不足，跳过")
            continue
        deltas = [by_session[k]["overall"] - base_sessions[k]["overall"] for k in shared]
        mean_delta, low, high, p_value = paired_bootstrap(deltas, iterations, rng)
        verdict = "显著" if p_value < 0.05 else "不显著"
        wins = sum(1 for d in deltas if d > 0)
        row = f"{model:<18}{len(shared):>9}{mean_delta:>+9.2f}"
        row += f"{f'[{low:+.2f}, {high:+.2f}]':>18}{p_value:>9.4f}  {verdict:<8} ({wins}/{len(deltas)} 胜)"
        lines.append(row)

    lines.append("")
    lines.append("逐维度 delta：")
    dim_header = f"{'model':<18}"
    for dimension in DIMENSIONS:
        dim_header += f"{dimension[:9]:>11}"
    lines.append(dim_header)
    lines.append("-" * len(dim_header))
    for model, by_session in sessions.items():
        if model == baseline:
            continue
        shared = sorted(set(base_sessions) & set(by_session))
        if len(shared) < 2:
            continue
        row = f"{model:<18}"
        for dimension in DIMENSIONS:
            delta = statistics.mean(
                by_session[k][dimension] - base_sessions[k][dimension] for k in shared
            )
            row += f"{delta:>+11.2f}"
        lines.append(row)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", help="judge.py 产出的评分 JSONL（可多个）")
    parser.add_argument("--baseline", default="base", help="配对比较的基准模型名")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--json-out", default="", help="额外写出机器可读的汇总 JSON")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in args.scores:
        records.extend(read_jsonl(Path(path)))
    if not records:
        raise SystemExit("没有评分记录")

    rng = random.Random(args.seed)
    sessions = session_scores(records)

    total_chunks = len(records)
    total_sessions = sum(len(v) for v in sessions.values())
    print(f"评分块 {total_chunks} 个，会话 {total_sessions} 个，模型 {len(sessions)} 个")
    print()
    print(format_table(sessions, args.bootstrap, rng))
    print()
    print(format_objectives(sessions))
    print()
    print(format_paired(sessions, args.baseline, args.bootstrap, rng))
    print()
    print(
        "注意：本表由自建 judge 产出，仅用于横向比较，不可与 MiniMax 官方榜单数值直接对比。"
    )

    if args.json_out:
        summary = {
            model: {
                "n_sessions": len(by_session),
                "overall": statistics.mean(s["overall"] for s in by_session.values()),
                **{
                    dimension: statistics.mean(s[dimension] for s in by_session.values())
                    for dimension in DIMENSIONS
                },
            }
            for model, by_session in sessions.items()
        }
        Path(args.json_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n汇总已写出: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
