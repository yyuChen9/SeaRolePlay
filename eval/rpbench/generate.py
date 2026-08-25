#!/usr/bin/env python3
"""Generate role-play self-play transcripts against an OpenAI-compatible server.

Reproduces the MiniMax role-play-bench trajectory protocol: for each seed, an
NPC model (the model under test) and a user simulator alternate for N turns,
starting from the seed's `ai_prologue` and `initial_user_input`.

The NPC and the user simulator are separate endpoints, so each side can sit on a
local vLLM server or on a remote OpenAI-compatible API:

  * NPC   -- normally the local vLLM server (base weights, or a LoRA adapter
             name). Point --base-url at a remote API instead to score an
             external reference model on the same seeds.
  * user  -- one fixed model for every NPC under test, so that all models face
             identical user behaviour. MiniMax reports they validated this same
             fixed-user design and found statistically similar results.

Whichever model plays the user, it must be the SAME across every NPC in a
comparison; scores from runs with different user simulators are not comparable.

The dataset's own transcripts use roles "ai"/"user" with a `round` index; the
output here matches that shape so judge.py can score our transcripts and the
published reference transcripts with the same code path.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 httpx，请先执行: python -m pip install -U httpx") from exc


# The NPC system prompt mirrors the character-card style the SFT data was
# trained on (a prose description of the character, second person "You're X"),
# so the fine-tuned adapters are evaluated in-distribution rather than being
# penalised for a prompt format they never saw.
NPC_SYSTEM = """You're {ai_name} in this fictional never-ending roleplay with {user_name}.

{ai_name}'s description: {ai_setting}

Stay in character as {ai_name} at all times. Never speak or act on behalf of \
{user_name}, never narrate their thoughts or decide their actions. Reply only \
with {ai_name}'s speech and actions. Keep replies to a few sentences unless the \
scene calls for more. Do not break the fourth wall, do not mention being an AI, \
and do not summarise or comment on the roleplay itself."""

# The user simulator is deliberately terse and is told to drive the scene. Left
# unconstrained, an instruct model tends to write long cooperative paragraphs
# that make every NPC look good and compress score variance.
USER_SYSTEM = """You are role-playing as {user_name} in an ongoing fictional \
scene with {ai_name}.

{ai_name}'s description: {ai_setting}

You are the human participant. Write only {user_name}'s next message: speech and \
actions, in first person or with *asterisk actions*. Be natural and varied -- \
ask questions, push back, change the subject, introduce complications, react \
emotionally. Do not write {ai_name}'s lines or narrate their reactions. Keep it \
to one or two sentences. Never break character or comment on the roleplay."""


class ChatClient:
    """Minimal OpenAI-compatible chat client with retry.

    `template_kwargs` must be off for hosted APIs: `chat_template_kwargs` is a
    vLLM extension, and gateways reject the whole request with
    `Unsupported parameter` rather than ignoring the unknown field.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        retries: int,
        template_kwargs: bool = True,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.retries = retries
        self.template_kwargs = template_kwargs
        self.client = httpx.Client(timeout=timeout)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        no_think: bool = True,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if no_think and self.template_kwargs:
            # The adapters were trained under LLaMA-Factory's `qwen3_5_nothink`
            # template. vLLM applies the model's own chat template, which
            # defaults to thinking mode -- leaving this off makes every model
            # emit a visible "Thinking Process:" chain of thought into the
            # transcript and tanks every judge dimension.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.client.post(self.url, headers=self.headers, json=payload)
                response.raise_for_status()
                data = response.json()
                text = (data["choices"][0]["message"].get("content") or "").strip()
                if text:
                    return text
                last_error = ValueError("模型返回空内容")
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last_error = exc
            if attempt < self.retries - 1:
                # Deterministic backoff; jitter would break run reproducibility.
                threading.Event().wait(2.0 * (attempt + 1))
        raise RuntimeError(f"{model} 请求失败（重试 {self.retries} 次）: {last_error}")

    def close(self) -> None:
        self.client.close()


def build_npc_messages(seed: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """History from the NPC's perspective: its own turns are `assistant`."""
    messages = [
        {
            "role": "system",
            "content": NPC_SYSTEM.format(
                ai_name=seed["ai_name"],
                user_name=seed["user_name"],
                ai_setting=seed["ai_setting"],
            ),
        }
    ]
    for turn in history:
        role = "assistant" if turn["role"] == "ai" else "user"
        messages.append({"role": role, "content": turn["text"]})
    return messages


def build_user_messages(seed: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """History from the simulated user's perspective: roles are inverted."""
    messages = [
        {
            "role": "system",
            "content": USER_SYSTEM.format(
                ai_name=seed["ai_name"],
                user_name=seed["user_name"],
                ai_setting=seed["ai_setting"],
            ),
        }
    ]
    for turn in history:
        role = "user" if turn["role"] == "ai" else "assistant"
        messages.append({"role": role, "content": turn["text"]})
    return messages


def run_one_dialogue(
    npc_client: ChatClient,
    user_client: ChatClient,
    seed: dict[str, Any],
    npc_model: str,
    user_model: str,
    num_turns: int,
    run_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Turn 0 and 1 come from the seed itself, exactly as in the published
    # transcripts, so every model starts from the same two lines.
    history: list[dict[str, Any]] = [
        {"role": "ai", "round": 0, "text": seed["ai_prologue"]},
        {"role": "user", "round": 1, "text": seed["initial_user_input"]},
    ]

    # Seeds are derived from the seed id and run index so a rerun reproduces the
    # same trajectory, while the three runs of one scenario still differ.
    # crc32, not hash(): str hashing is salted per process (PYTHONHASHSEED), so
    # hash() would silently give a different trajectory on every rerun.
    #
    # This only holds for a local vLLM server. Hosted APIs treat `seed` as
    # best-effort at most, so a run with a remote NPC or user simulator is NOT
    # bit-reproducible -- the seed still varies the sampling per turn, but a
    # rerun may diverge.
    base_seed = (zlib.crc32(seed["id"].encode("utf-8")) & 0xFFFF) * 1000 + run_index * 7

    for round_index in range(2, num_turns):
        if round_index % 2 == 0:
            text = npc_client.complete(
                npc_model,
                build_npc_messages(seed, history),
                temperature=args.npc_temperature,
                max_tokens=args.npc_max_tokens,
                seed=base_seed + round_index,
                no_think=not args.thinking,
            )
            history.append({"role": "ai", "round": round_index, "text": text})
        else:
            text = user_client.complete(
                user_model,
                build_user_messages(seed, history),
                temperature=args.user_temperature,
                max_tokens=args.user_max_tokens,
                seed=base_seed + round_index,
                no_think=not args.thinking,
            )
            history.append({"role": "user", "round": round_index, "text": text})

    return {
        "seed_id": seed["id"],
        "model_name": args.label,
        "run_id": f"run_{run_index + 1}",
        "dialogue": history,
        "num_turns": len(history),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="eval/rpbench/data/seeds_en.json")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--label", required=True, help="记入结果的模型名，如 base/sft/dpo")
    parser.add_argument("--npc-model", required=True, help="被测模型在 server 上的名称")
    parser.add_argument("--user-model", required=True, help="user simulator 模型名称")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="NPC 接入点")
    parser.add_argument("--api-key", default="EMPTY", help="NPC 接入点的 key")
    parser.add_argument(
        "--remote-npc",
        action="store_true",
        help="NPC 走托管 API（不发送 vLLM 专有的 chat_template_kwargs）。"
        "用于把外部模型作为横向对比的参照，此时思维链无法关闭。",
    )
    parser.add_argument(
        "--user-base-url",
        default="",
        help="user simulator 接入点，留空则复用 --base-url",
    )
    parser.add_argument(
        "--user-api-key",
        default="",
        help="user simulator 接入点的 key，留空则复用 --api-key",
    )
    parser.add_argument(
        "--remote-user",
        action="store_true",
        help="user simulator 走托管 API（不发送 chat_template_kwargs）。"
        "指定了 --user-base-url 时会自动推断，一般无需显式传入。",
    )
    parser.add_argument("--num-turns", type=int, default=40)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-seeds", type=int, default=0, help="0 表示全部")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--npc-temperature", type=float, default=0.8)
    parser.add_argument("--user-temperature", type=float, default=1.0)
    parser.add_argument("--npc-max-tokens", type=int, default=512)
    parser.add_argument("--user-max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="保留思维链输出（默认关闭，与训练用的 qwen3_5_nothink 模板一致）",
    )
    args = parser.parse_args()

    seeds = json.loads(Path(args.seeds).read_text(encoding="utf-8"))
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    if not seeds:
        raise SystemExit(f"没有可用 seed: {args.seeds}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: a long generation run is the expensive part, so completed
    # (seed, run) pairs are skipped instead of regenerated.
    done: set[tuple[str, str]] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                done.add((record["seed_id"], record["run_id"]))
        if done:
            print(f"[INFO] 续跑：已有 {len(done)} 条完成的对话，将跳过")

    tasks = [
        (seed, run_index)
        for run_index in range(args.runs)
        for seed in seeds
        if (seed["id"], f"run_{run_index + 1}") not in done
    ]
    if not tasks:
        print("[INFO] 全部对话已生成，无需重跑")
        return 0

    print(
        f"[INFO] {args.label}: {len(tasks)} 段对话 "
        f"({len(seeds)} seeds x {args.runs} runs, {args.num_turns} 轮), "
        f"并发 {args.concurrency}"
    )

    npc_client = ChatClient(
        args.base_url,
        args.api_key,
        args.timeout,
        args.retries,
        template_kwargs=not args.remote_npc,
    )
    # A separate --user-base-url implies a different (hosted) endpoint, so the
    # vLLM-only parameter is dropped for that side automatically.
    user_is_remote = args.remote_user or bool(args.user_base_url)
    if args.user_base_url:
        user_client = ChatClient(
            args.user_base_url,
            args.user_api_key or args.api_key,
            args.timeout,
            args.retries,
            template_kwargs=not user_is_remote,
        )
    else:
        user_client = npc_client

    print(
        f"[INFO] NPC: {args.npc_model} @ {args.base_url}"
        f"{' (remote)' if args.remote_npc else ''}\n"
        f"[INFO] user: {args.user_model} @ "
        f"{args.user_base_url or args.base_url}{' (remote)' if user_is_remote else ''}"
    )
    if args.remote_npc or user_is_remote:
        print(
            "[INFO] 远程接入点不支持 chat_template_kwargs，该侧思维链无法强制关闭；"
            "远程 seed 不保证确定性，重跑轨迹可能不同。"
        )

    write_lock = threading.Lock()
    completed = 0
    failed = 0

    try:
        with output_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(
                        run_one_dialogue,
                        npc_client,
                        user_client,
                        seed,
                        args.npc_model,
                        args.user_model,
                        args.num_turns,
                        run_index,
                        args,
                    ): (seed["id"], run_index)
                    for seed, run_index in tasks
                }
                for future in as_completed(futures):
                    seed_id, run_index = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001 - reported, run continues
                        failed += 1
                        print(f"[WARN] {seed_id} run_{run_index + 1} 失败: {exc}", file=sys.stderr)
                        continue
                    with write_lock:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        completed += 1
                        if completed % 5 == 0 or completed == len(tasks):
                            print(f"[INFO] {completed}/{len(tasks)} 段对话完成")
    finally:
        npc_client.close()
        if user_client is not npc_client:
            user_client.close()

    print(f"[DONE] {args.label}: 成功 {completed}，失败 {failed}  ->  {output_path}")
    return 1 if failed and not completed else 0


if __name__ == "__main__":
    sys.exit(main())
