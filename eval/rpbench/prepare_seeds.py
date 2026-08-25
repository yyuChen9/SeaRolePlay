#!/usr/bin/env python3
"""Download the MiniMaxAI/role-play-bench seeds used as evaluation inputs.

The benchmark publishes three tables: `seeds` (character cards + opening lines),
`dialogues` (100-turn transcripts from 11 reference models) and `evaluations`
(6-dimension scores for those transcripts). It does NOT publish the judge prompt
or rubric, so only `seeds` is usable as input for evaluating a new model --
the scoring side is reimplemented in judge.py from the dataset card's
dimension definitions.

`dialogues` and `evaluations` are downloaded too, but purely as reference: they
let us sanity-check our judge against published scores for known models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - exercised on a fresh machine
    raise SystemExit(
        "缺少 huggingface_hub，请先执行: python -m pip install -U huggingface_hub"
    ) from exc


REPO_ID = "MiniMaxAI/role-play-bench"
# Pinned so the eval inputs stay identical across machines and reruns, matching
# the convention in scripts/prepare_hf_character_data.py.
REVISION = "3c1be2a56afbcaab19ae6b40b8a24429eae792f5"

SEED_FIELDS = (
    "id",
    "ai_name",
    "ai_setting",
    "user_name",
    "user_setting",
    "ai_prologue",
    "initial_user_input",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 解析失败: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: 样本必须是对象")
            records.append(record)
    return records


def validate_seeds(seeds: list[dict[str, Any]], lang: str) -> None:
    if not seeds:
        raise ValueError(f"{lang}: seeds 为空")
    ids = set()
    for index, seed in enumerate(seeds):
        missing = [field for field in SEED_FIELDS if not str(seed.get(field, "")).strip()]
        if missing:
            raise ValueError(f"{lang}: seed #{index} 缺少字段 {missing}")
        if seed["id"] in ids:
            raise ValueError(f"{lang}: seed id 重复: {seed['id']}")
        ids.add(seed["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="eval/rpbench/data")
    parser.add_argument("--lang", default="en", choices=["en", "zh", "both"])
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="同时下载 dialogues/evaluations（仅用于校准 judge，评测本身不需要）",
    )
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()

    langs = ["en", "zh"] if args.lang == "both" else [args.lang]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        seed_path = hf_hub_download(
            REPO_ID,
            f"data/{lang}/seeds.jsonl",
            repo_type="dataset",
            revision=args.revision,
        )
        seeds = read_jsonl(Path(seed_path))
        validate_seeds(seeds, lang)

        destination = output_dir / f"seeds_{lang}.json"
        destination.write_text(
            json.dumps(seeds, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{lang:>3} seeds  {len(seeds):>5} 条  ->  {destination}")

        if args.with_reference:
            for name in ("dialogues", "evaluations"):
                source = hf_hub_download(
                    REPO_ID,
                    f"data/{lang}/{name}.jsonl",
                    repo_type="dataset",
                    revision=args.revision,
                )
                records = read_jsonl(Path(source))
                reference = output_dir / f"{name}_{lang}.json"
                reference.write_text(
                    json.dumps(records, ensure_ascii=False), encoding="utf-8"
                )
                print(f"{lang:>3} {name:<12} {len(records):>5} 条  ->  {reference}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
