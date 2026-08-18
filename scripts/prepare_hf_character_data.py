#!/usr/bin/env python3
"""Download and normalize the three Character datasets used by this project.

The two StoryPlay/Sonnet datasets are converted to ShareGPT SFT records. The
StoryPlay DPO dataset is kept as prompt/chosen/rejected ranking records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - exercised on a fresh machine
    raise SystemExit(
        "缺少 huggingface_hub，请先执行: python -m pip install -U huggingface_hub"
    ) from exc


DATASETS = {
    "storyplay_sft": {
        "repo_id": "rx1lora/StoryPlay_Sonnet3.5-Charcard-Roleplay",
        "filename": "sonnet35-charcard-roleplay-sharegpt.jsonl",
        "output": "storyplay_sonnet35_charcard.json",
        "revision": "8ed694f86627d82f669337308ecb5f0bf751e097",
    },
    "storyplay_dpo": {
        "repo_id": "rx1lora/StoryPlay_DPO_Pairs-Roleplay-NSFW",
        "filename": "DPO_Pairs-Roleplay-NSFW.jsonl",
        "output": "storyplay_dpo_roleplay_nsfw.json",
        "revision": "6b029adf3801fedd12ed7ff657a388914aad5dc2",
    },
    "gryphe_sft": {
        "repo_id": "Gryphe/Sonnet3.5-Charcard-Roleplay",
        "filename": "sonnet35-charcard-roleplay-sharegpt.jsonl",
        "output": "gryphe_sonnet35_charcard.json",
        "revision": "0b47a7695233107ad40da25db044610ddb378830",
    },
}

ROLE_ALIASES = {"human": "user", "gpt": "assistant", "bot": "assistant"}
VALID_ROLES = {"system", "user", "assistant"}


def jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            position = 0
            while position < len(line):
                while position < len(line) and line[position].isspace():
                    position += 1
                if position == len(line):
                    break
                try:
                    record, end = decoder.raw_decode(line, position)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: JSON 解析失败（第 {position + 1} 列）: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: 样本必须是对象")
                yield record
                position = end


def normalize_role(value: Any) -> str:
    role = ROLE_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if role not in VALID_ROLES:
        raise ValueError(f"未知角色 {value!r}")
    return role


def normalize_conversations(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    conversations = record.get("conversations") or record.get("messages")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("缺少非空 conversations/messages 数组")
    messages: list[dict[str, str]] = []
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError("消息必须是对象")
        role_value = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("消息 content/value 不能为空")
        messages.append({"role": normalize_role(role_value), "content": content.strip()})
    if not any(item["role"] == "assistant" for item in messages):
        raise ValueError("样本没有 assistant 回复")
    return {"messages": messages}


def normalize_dpo(record: dict[str, Any]) -> dict[str, str]:
    fields = {key: record.get(key) for key in ("prompt", "chosen", "rejected")}
    for key, value in fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"DPO 字段 {key} 不能为空")
        fields[key] = value.strip()
    return fields  # type: ignore[return-value]


def write_json(path: Path, records: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def record_key(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def download_dataset(name: str, cache_dir: Path, revision: str | None) -> Path:
    spec = DATASETS[name]
    local_path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        repo_type="dataset",
        revision=revision or spec["revision"],
        local_dir=cache_dir / name,
    )
    return Path(local_path)


def build_dataset_info(output_dir: Path) -> None:
    info: dict[str, Any] = {
        "storyplay_sft": {
            "file_name": "storyplay_sonnet35_charcard.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        },
        "gryphe_sft": {
            "file_name": "gryphe_sonnet35_charcard.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        },
        "character_sft": {
            "file_name": "character_sft_combined.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        },
        "character": {
            "file_name": "character_sft_combined.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        },
        "storyplay_dpo": {
            "file_name": "storyplay_dpo_roleplay_nsfw.json",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    write_json(output_dir / "dataset_info.json", info)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--revision", default=None, help="可选 Hugging Face revision/commit")
    parser.add_argument("--skip-download", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force-download", action="store_true", help="忽略已有原始文件并重新下载")
    parser.add_argument("--no-combined", action="store_true", help="不生成合并 SFT 文件")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    cache_dir = (args.cache_dir or output_dir / "raw").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    converted_sft: list[dict[str, Any]] = []
    for name in ("storyplay_sft", "gryphe_sft"):
        raw_path = cache_dir / name / DATASETS[name]["filename"]
        if args.force_download or not raw_path.exists():
            print(f"下载 {DATASETS[name]['repo_id']} ...", file=sys.stderr)
            raw_path = download_dataset(name, cache_dir, args.revision)
        records, errors = [], []
        for index, record in enumerate(jsonl_records(raw_path)):
            try:
                records.append(normalize_conversations(record))
            except ValueError as exc:
                errors.append(f"{raw_path}:{index + 1}: {exc}")
        if errors:
            preview = "\n".join(errors[:10])
            raise SystemExit(f"SFT 数据校验失败，共 {len(errors)} 条错误:\n{preview}")
        write_json(output_dir / DATASETS[name]["output"], records)
        converted_sft.extend(records)
        print(f"{name}: {len(records)} 条")

    if not args.no_combined:
        unique: dict[str, dict[str, Any]] = {}
        for record in converted_sft:
            unique.setdefault(record_key(record), record)
        write_json(output_dir / "character_sft_combined.json", list(unique.values()))
        print(f"character_sft_combined: {len(unique)} 条（原始 {len(converted_sft)} 条，已去重）")

    dpo_path = cache_dir / "storyplay_dpo" / DATASETS["storyplay_dpo"]["filename"]
    if args.force_download or not dpo_path.exists():
        print(f"下载 {DATASETS['storyplay_dpo']['repo_id']} ...", file=sys.stderr)
        dpo_path = download_dataset("storyplay_dpo", cache_dir, args.revision)
    dpo_records, dpo_errors, dpo_skipped = [], [], []
    for index, record in enumerate(jsonl_records(dpo_path)):
        try:
            dpo_records.append(normalize_dpo(record))
        except ValueError as exc:
            message = f"{dpo_path}:{index + 1}: {exc}"
            if "rejected 不能为空" in str(exc):
                dpo_skipped.append(message)
            else:
                dpo_errors.append(message)
    if dpo_errors:
        preview = "\n".join(dpo_errors[:10])
        raise SystemExit(f"DPO 数据校验失败，共 {len(dpo_errors)} 条错误:\n{preview}")
    write_json(output_dir / DATASETS["storyplay_dpo"]["output"], dpo_records)
    print(f"storyplay_dpo: {len(dpo_records)} 条")
    if dpo_skipped:
        print(f"storyplay_dpo: 跳过 {len(dpo_skipped)} 条 rejected 为空的样本", file=sys.stderr)
    build_dataset_info(output_dir)
    print(f"已写入数据集注册: {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
