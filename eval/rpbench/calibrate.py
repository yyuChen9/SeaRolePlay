#!/usr/bin/env python3
"""Check our reimplemented judge against role-play-bench's published scores.

Because MiniMax did not publish the judge prompt, judge.py is our own
reconstruction from the dataset card. This script quantifies how much to trust
it: score a sample of the PUBLISHED transcripts (whose official scores we know)
with our judge, then measure agreement.

What matters is rank correlation across models, not absolute agreement. Our
judge sitting uniformly 5 points high is harmless for a base-vs-SFT-vs-DPO
comparison; our judge ranking a known-weak model above a known-strong one is
not, and means the report's conclusions should not be trusted.

Usage is two steps -- build a transcript subset, judge it, then compare:

  python eval/rpbench/calibrate.py sample  --output eval/rpbench/data/calib.jsonl
  python eval/rpbench/judge.py --dialogues eval/rpbench/data/calib.jsonl \\
      --output eval/rpbench/results/calib_scores.jsonl
  python eval/rpbench/calibrate.py compare --scores eval/rpbench/results/calib_scores.jsonl
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


def read_records(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array (prepare_seeds.py output) or JSONL (judge.py output)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            end = index
            while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
                end += 1
            average = (index + end) / 2 + 1
            for position in range(index, end + 1):
                ranks[order[position]] = average
            index = end + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) ** 0.5 * sum((b - my) ** 2 for b in ry) ** 0.5
    )
    return num / den if den else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (
        sum((a - mx) ** 2 for a in xs) ** 0.5 * sum((b - my) ** 2 for b in ys) ** 0.5
    )
    return num / den if den else float("nan")


def cmd_sample(args: argparse.Namespace) -> int:
    dialogues = read_records(Path(args.dialogues))
    rng = random.Random(args.seed)

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dialogues:
        by_model[record["model_name"]].append(record)

    selected = []
    for model in sorted(by_model):
        pool = sorted(by_model[model], key=lambda r: (r["seed_id"], r["run_id"]))
        rng.shuffle(pool)
        selected.extend(pool[: args.per_model])

    # Trim turns so calibration cost stays proportional to the real eval, which
    # uses far fewer than the published 102 turns.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            turns = record["dialogue"]
            if isinstance(turns, str):
                turns = json.loads(turns)
            handle.write(
                json.dumps(
                    {
                        "seed_id": record["seed_id"],
                        "model_name": record["model_name"],
                        "run_id": record["run_id"],
                        "dialogue": turns[: args.num_turns],
                        "num_turns": min(len(turns), args.num_turns),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"[DONE] 抽取 {len(selected)} 段已发布对话"
        f"（{len(by_model)} 个模型 x {args.per_model} 段，各取前 {args.num_turns} 轮）"
        f"  ->  {output_path}"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    ours = read_records(Path(args.scores))
    official = read_records(Path(args.evaluations))

    # Ours: chunk -> session -> model.
    sessions: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in ours:
        sessions[(record["model_name"], record["seed_id"], record["run_id"])].append(record)

    our_by_model: dict[str, list[float]] = defaultdict(list)
    our_dims: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    paired_ours: list[float] = []
    paired_official: list[float] = []

    official_index = {
        (r["model_name"], r["seed_id"], r["run_id"]): r for r in official
    }

    for key, chunks in sessions.items():
        model, seed_id, run_id = key
        means = {d: statistics.mean(c[d] for c in chunks) for d in DIMENSIONS}
        overall = sum(WEIGHTS[d] * means[d] for d in DIMENSIONS)
        our_by_model[model].append(overall)
        for d in DIMENSIONS:
            our_dims[model][d].append(means[d])

        reference = official_index.get(key)
        if reference is not None:
            official_overall = sum(
                WEIGHTS[d] * reference[f"{d}_score"] for d in DIMENSIONS
            )
            paired_ours.append(overall)
            paired_official.append(official_overall)

    official_by_model: dict[str, list[float]] = defaultdict(list)
    for record in official:
        official_by_model[record["model_name"]].append(
            sum(WEIGHTS[d] * record[f"{d}_score"] for d in DIMENSIONS)
        )

    models = sorted(our_by_model, key=lambda m: statistics.mean(our_by_model[m]), reverse=True)
    print(f"{'model':<32}{'n':>4}{'ours':>9}{'official':>10}{'diff':>9}")
    print("-" * 64)
    our_means, official_means = [], []
    for model in models:
        our_mean = statistics.mean(our_by_model[model])
        official_mean = statistics.mean(official_by_model[model])
        our_means.append(our_mean)
        official_means.append(official_mean)
        print(
            f"{model:<32}{len(our_by_model[model]):>4}{our_mean:>9.2f}"
            f"{official_mean:>10.2f}{our_mean - official_mean:>+9.2f}"
        )

    print()
    if len(models) >= 3:
        print(f"模型级 Spearman 秩相关: {spearman(our_means, official_means):.3f}")
        print(f"模型级 Pearson 相关:    {pearson(our_means, official_means):.3f}")
        print(f"平均绝对偏移:            {statistics.mean(abs(a - b) for a, b in zip(our_means, official_means)):.2f}")
    if len(paired_ours) >= 3:
        print(f"会话级 Spearman（n={len(paired_ours)}）: {spearman(paired_ours, paired_official):.3f}")

    print()
    print("判读标准：模型级 Spearman >= 0.7 时，本 judge 可用于横向比较；")
    print("低于该值说明 judge 与官方口径分歧过大，报告结论需谨慎。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="从已发布对话中抽取校准子集")
    sample.add_argument("--dialogues", default="eval/rpbench/data/dialogues_en.json")
    sample.add_argument("--output", default="eval/rpbench/data/calib.jsonl")
    sample.add_argument("--per-model", type=int, default=3)
    sample.add_argument("--num-turns", type=int, default=40)
    sample.add_argument("--seed", type=int, default=20260823)
    sample.set_defaults(func=cmd_sample)

    compare = sub.add_parser("compare", help="对比自建 judge 与官方分数")
    compare.add_argument("--scores", required=True)
    compare.add_argument("--evaluations", default="eval/rpbench/data/evaluations_en.json")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
