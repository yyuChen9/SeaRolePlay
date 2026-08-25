#!/usr/bin/env python3
"""Score role-play transcripts with an LLM judge on the role-play-bench rubric.

IMPORTANT -- this is a reimplementation, not the official scorer. MiniMax
published the benchmark inputs (seeds) and the resulting scores for 11 models,
but NOT the judge prompt or rubric. The six dimensions and their weights below
come from the dataset card; the prompt wording is ours.

Consequences, which the report repeats:
  * Scores are comparable ACROSS MODELS SCORED BY THIS FILE, and not comparable
    with the published leaderboard numbers.
  * Applying the card's stated weights to the published per-dimension scores
    reproduces the leaderboard ORDER but not its absolute values (our recompute
    runs 5-9 points high), so MiniMax applies a further normalisation that is
    not documented. Use `calibrate.py` to quantify our judge's agreement with
    the published scores before trusting a small gap between two models.

Protocol detail kept from the card: long transcripts are cut into fixed-size
chunks and each chunk is scored independently, then averaged. Judging 100 turns
in one call degrades reliably-scored detail; chunking is what makes the score
stable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 httpx，请先执行: python -m pip install -U httpx") from exc


# Weights from the dataset card: Worlds 50%, Stories 25%, User Preferences 25%.
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
    "worlds": ("basics", "logic", "knowledge"),
    "stories": ("diversity", "content_logic"),
    "preferences": ("interaction",),
}

JUDGE_SYSTEM = """You are a strict evaluator of role-play quality. You score an \
NPC's performance in a fictional role-play transcript.

Score by DETECTING MISALIGNMENT, not by rewarding good writing. Start each \
dimension at 100 and deduct for concrete, quotable defects. A chunk with no \
defects scores 100. Do not deduct for a scene being mundane, for mature or \
dark content that fits the character, or for the user's behaviour -- you are \
scoring only the NPC's turns.

Score these six dimensions from 0 to 100:

basics -- Text quality. Deduct for garbled text, broken markup, unintended \
language mixing, verbatim repetition of its own earlier phrasing, truncation \
mid-sentence, or stray meta tokens.

logic -- Referential coherence. Deduct for confusing who is who, contradicting \
established facts of the scene, spatial or physical impossibilities, wrong \
pronouns, or losing track of objects and positions.

knowledge -- World consistency. Deduct when the NPC violates the rules of its \
setting: anachronisms in a period setting, real-world knowledge a fictional \
character could not have, or breaking established in-world constraints.

diversity -- Expressive range and narrative momentum. Deduct for recycled \
sentence structures, a repeated verbal tic, every reply having identical \
shape or length, or the scene stagnating in a loop with no development.

content_logic -- Character fidelity. Deduct for out-of-character behaviour, \
personality drift, abrupt unmotivated changes in stance or mood, or ignoring \
its own stated background. Character growth is fine when the transcript builds \
to it.

interaction -- User handling. Deduct heavily for AI SPEAKS FOR USER (writing \
the user's dialogue, actions, thoughts, or decisions), AI IGNORES USER (not \
responding to what the user actually said or did), and OVER REFUSAL (breaking \
character to refuse, moralise, lecture, or give an assistant-style disclaimer).

Return ONLY a JSON object, no prose and no code fence:
{"basics": <int>, "logic": <int>, "knowledge": <int>, "diversity": <int>, \
"content_logic": <int>, "interaction": <int>, "notes": "<= 40 words citing the \
worst concrete defect, or 'none'"}"""

JUDGE_USER = """CHARACTER THE NPC MUST PLAY
Name: {ai_name}
Description: {ai_setting}

The NPC is "{ai_name}". The human participant is "{user_name}".
This is an excerpt (turns {start}-{end}) from a longer transcript, so it may \
begin and end mid-scene. Judge only what is shown.

TRANSCRIPT
{transcript}

Score the {ai_name} turns on the six dimensions. Return only the JSON object."""


class JudgeClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float, retries: int) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.retries = retries
        self.client = httpx.Client(timeout=timeout)

    def score(self, system: str, user: str, temperature: float, max_tokens: int) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.client.post(self.url, headers=self.headers, json=payload)
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"] or ""
                return parse_scores(text)
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last_error = exc
            if attempt < self.retries - 1:
                threading.Event().wait(2.0 * (attempt + 1))
        raise RuntimeError(f"judge 请求失败（重试 {self.retries} 次）: {last_error}")

    def close(self) -> None:
        self.client.close()


def parse_scores(text: str) -> dict[str, Any]:
    """Extract the score object, tolerating code fences and surrounding prose."""
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    if not candidate.startswith("{"):
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not brace:
            raise ValueError(f"judge 输出中找不到 JSON: {text[:200]!r}")
        candidate = brace.group(0)

    data = json.loads(candidate)
    scores: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        if dimension not in data:
            raise ValueError(f"judge 输出缺少维度 {dimension}: {text[:200]!r}")
        value = float(data[dimension])
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"维度 {dimension} 超出 0-100: {value}")
        scores[dimension] = value
    scores["notes"] = str(data.get("notes", ""))[:300]
    return scores


def format_transcript(turns: list[dict[str, Any]], ai_name: str, user_name: str) -> str:
    lines = []
    for turn in turns:
        speaker = ai_name if turn["role"] == "ai" else user_name
        lines.append(f"[turn {turn['round']}] {speaker}: {turn['text']}")
    return "\n\n".join(lines)


def chunk_turns(dialogue: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [dialogue[i : i + chunk_size] for i in range(0, len(dialogue), chunk_size)]


def weighted_overall(scores: dict[str, float]) -> float:
    return sum(WEIGHTS[dimension] * scores[dimension] for dimension in DIMENSIONS)


def score_one_chunk(
    client: JudgeClient,
    seed: dict[str, Any],
    chunk: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    user_prompt = JUDGE_USER.format(
        ai_name=seed["ai_name"],
        ai_setting=seed["ai_setting"],
        user_name=seed["user_name"],
        start=chunk[0]["round"],
        end=chunk[-1]["round"],
        transcript=format_transcript(chunk, seed["ai_name"], seed["user_name"]),
    )
    return client.score(JUDGE_SYSTEM, user_prompt, args.temperature, args.max_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", required=True, help="generate.py 产出的 JSONL")
    parser.add_argument("--seeds", default="eval/rpbench/data/seeds_en.json")
    parser.add_argument("--output", required=True, help="逐 chunk 评分 JSONL")
    parser.add_argument("--judge-model", default="gpt-5.6-sol")
    parser.add_argument(
        "--base-url",
        default="https://openresty-gateway.gpu-service.dev.seaart.dev/llm/v1",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="留空则依次读取 JUDGE_API_KEY / ANTHROPIC_AUTH_TOKEN",
    )
    parser.add_argument("--chunk-size", type=int, default=20, help="每个评分块的轮数")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    import os

    api_key = (
        args.api_key
        or os.environ.get("JUDGE_API_KEY", "")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    )
    if not api_key:
        raise SystemExit(
            "缺少 API key：传入 --api-key，或设置 JUDGE_API_KEY / ANTHROPIC_AUTH_TOKEN"
        )

    seeds = {
        seed["id"]: seed
        for seed in json.loads(Path(args.seeds).read_text(encoding="utf-8"))
    }

    dialogues = []
    with Path(args.dialogues).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    if not dialogues:
        raise SystemExit(f"没有对话可评: {args.dialogues}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str, int]] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done.add((record["seed_id"], record["run_id"], record["chunk_index"]))
        if done:
            print(f"[INFO] 续跑：已有 {len(done)} 个 chunk 已评分，将跳过")

    tasks = []
    for dialogue in dialogues:
        seed = seeds.get(dialogue["seed_id"])
        if seed is None:
            print(f"[WARN] seed 未找到，跳过: {dialogue['seed_id']}", file=sys.stderr)
            continue
        turns = dialogue["dialogue"]
        if isinstance(turns, str):  # published transcripts store this as a string
            turns = json.loads(turns)
        for chunk_index, chunk in enumerate(chunk_turns(turns, args.chunk_size)):
            if not any(turn["role"] == "ai" for turn in chunk):
                continue
            key = (dialogue["seed_id"], dialogue["run_id"], chunk_index)
            if key not in done:
                tasks.append((dialogue, seed, chunk_index, chunk))

    if not tasks:
        print("[INFO] 全部 chunk 已评分，无需重跑")
        return 0

    print(f"[INFO] 待评分 {len(tasks)} 个 chunk（chunk_size={args.chunk_size}），并发 {args.concurrency}")

    client = JudgeClient(args.base_url, api_key, args.judge_model, args.timeout, args.retries)
    write_lock = threading.Lock()
    completed = 0
    failed = 0

    try:
        with output_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(score_one_chunk, client, seed, chunk, args): (
                        dialogue,
                        chunk_index,
                        chunk,
                    )
                    for dialogue, seed, chunk_index, chunk in tasks
                }
                for future in as_completed(futures):
                    dialogue, chunk_index, chunk = futures[future]
                    try:
                        scores = future.result()
                    except Exception as exc:  # noqa: BLE001 - reported, run continues
                        failed += 1
                        print(
                            f"[WARN] {dialogue['seed_id']} {dialogue['run_id']} "
                            f"chunk{chunk_index} 评分失败: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    record = {
                        "seed_id": dialogue["seed_id"],
                        "model_name": dialogue["model_name"],
                        "run_id": dialogue["run_id"],
                        "chunk_index": chunk_index,
                        "turn_start": chunk[0]["round"],
                        "turn_end": chunk[-1]["round"],
                        **scores,
                        "overall": weighted_overall(scores),
                    }
                    with write_lock:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        completed += 1
                        if completed % 20 == 0 or completed == len(tasks):
                            print(f"[INFO] {completed}/{len(tasks)} chunk 已评分")
    finally:
        client.close()

    print(f"[DONE] 成功 {completed}，失败 {failed}  ->  {output_path}")
    return 1 if failed and not completed else 0


if __name__ == "__main__":
    sys.exit(main())
