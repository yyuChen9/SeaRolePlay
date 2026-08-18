#!/usr/bin/env python3
"""Convert common instruction/chat datasets to LLaMA-Factory ShareGPT JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROLES = {"system", "user", "assistant"}
ROLE_ALIASES = {"human": "user", "gpt": "assistant", "bot": "assistant"}


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
    else:
        raise ValueError("输入必须是 JSON 数组，或包含 data 数组的 JSON 对象")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError("每条数据必须是 JSON 对象")
    return records


def normalize_role(value: Any) -> str:
    role = str(value).strip().lower()
    role = ROLE_ALIASES.get(role, role)
    if role not in ROLES:
        raise ValueError(f"未知消息角色: {value!r}")
    return role


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = record.get("messages") or record.get("conversations")
    if raw_messages is not None:
        if not isinstance(raw_messages, list):
            raise ValueError("messages/conversations 必须是数组")
        messages = []
        for item in raw_messages:
            if not isinstance(item, dict):
                raise ValueError("每条消息必须是对象")
            role = normalize_role(item.get("role", item.get("from")))
            content = item.get("content", item.get("value"))
            if not isinstance(content, str) or not content.strip():
                raise ValueError("消息 content/value 不能为空")
            messages.append({"role": role, "content": content.strip()})
    else:
        instruction = record.get("instruction", record.get("prompt"))
        output = record.get("output", record.get("response", record.get("answer")))
        extra_input = record.get("input", "")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("缺少非空 instruction/prompt 字段")
        if not isinstance(output, str) or not output.strip():
            raise ValueError("缺少非空 output/response/answer 字段")
        user_content = instruction.strip()
        if isinstance(extra_input, str) and extra_input.strip():
            user_content = f"{user_content}\n{extra_input.strip()}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output.strip()},
        ]

    if not messages or not any(message["role"] == "assistant" for message in messages):
        raise ValueError("样本必须至少包含一条 assistant 消息")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = read_records(args.input)
    converted = []
    errors = []
    for index, record in enumerate(records):
        try:
            converted.append({"messages": normalize_messages(record)})
        except ValueError as exc:
            errors.append(f"第 {index} 条: {exc}")
    if errors:
        preview = "\n".join(errors[:10])
        suffix = "" if len(errors) <= 10 else f"\n... 另有 {len(errors) - 10} 条错误"
        raise SystemExit(f"数据校验失败，共 {len(errors)} 条错误:\n{preview}{suffix}")
    if not converted:
        raise SystemExit("输入数据为空")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"已转换 {len(converted)} 条数据 -> {args.output}")


if __name__ == "__main__":
    main()
