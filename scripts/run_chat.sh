#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-base}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
ADAPTER_PATH="${ADAPTER_PATH:-$ROOT_DIR/saves/qwen3.5-0.8b/character-lora}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$MODE" in
    base) EXTRA_ARGS=() ;;
    lora)
        [[ -d "$ADAPTER_PATH" ]] || { printf '[ERROR] Adapter not found: %s\n' "$ADAPTER_PATH" >&2; exit 1; }
        EXTRA_ARGS=(--adapter_name_or_path "$ADAPTER_PATH")
        ;;
    *) printf 'Usage: %s [base|lora]\n' "$0" >&2; exit 2 ;;
esac

if command -v llamafactory-cli >/dev/null 2>&1; then
    LLAMA_FACTORY=(llamafactory-cli)
else
    LLAMA_FACTORY=("$PYTHON_BIN" -m llamafactory.cli)
fi

exec "${LLAMA_FACTORY[@]}" chat --model_name_or_path "$MODEL_NAME" --template qwen3 --trust_remote_code true "${EXTRA_ARGS[@]}"
