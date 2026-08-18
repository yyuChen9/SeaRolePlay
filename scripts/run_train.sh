#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/qwen3_5_0_8b_lora_sft.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_FILE="${DATA_FILE:-$ROOT_DIR/data/character_sft_combined.json}"

[[ -f "$DATA_FILE" ]] || {
    printf '[ERROR] Missing %s. Run scripts/prepare_hf_character_data.py first.\n' "$DATA_FILE" >&2
    exit 1
}
[[ -f "$CONFIG_PATH" ]] || { printf '[ERROR] Config not found: %s\n' "$CONFIG_PATH" >&2; exit 1; }
cd "$ROOT_DIR"

if command -v llamafactory-cli >/dev/null 2>&1; then
    LLAMA_FACTORY=(llamafactory-cli)
else
    LLAMA_FACTORY=("$PYTHON_BIN" -m llamafactory.cli)
fi

printf '[INFO] Starting LLaMA-Factory training with config: %s\n' "$CONFIG_PATH"
exec "${LLAMA_FACTORY[@]}" train "$CONFIG_PATH"
