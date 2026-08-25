#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/qwen3_5_9b_lora_sft.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
[[ -f "$CONFIG_PATH" ]] || { printf '[ERROR] Config not found: %s\n' "$CONFIG_PATH" >&2; exit 1; }
cd "$ROOT_DIR"

# Resolve the dataset files this config actually needs by looking up its `dataset`
# entries in dataset_info.json, instead of assuming the SFT file. DPO configs use a
# different dataset, and hardcoding one path made them fail with a misleading error.
if [[ -n "${DATA_FILE:-}" ]]; then
    MISSING=$([[ -f "$DATA_FILE" ]] || printf '%s' "$DATA_FILE")
else
    MISSING="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import json, sys, pathlib, yaml

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
data_dir = pathlib.Path(config.get("dataset_dir") or "data")
names = config.get("dataset") or ""
if isinstance(names, str):
    names = [n.strip() for n in names.split(",") if n.strip()]

info_path = data_dir / "dataset_info.json"
if not info_path.is_file():
    print(info_path)
    sys.exit(0)

info = json.loads(info_path.read_text(encoding="utf-8"))
for name in names:
    entry = info.get(name)
    if entry is None:
        print(f"{info_path} (dataset '{name}' is not registered)")
        continue
    file_name = entry.get("file_name")
    if file_name and not (data_dir / file_name).is_file():
        print(data_dir / file_name)
PY
)"
fi

[[ -z "$MISSING" ]] || {
    printf '[ERROR] Missing dataset file(s):\n%s\n' "$MISSING" >&2
    printf 'Run scripts/prepare_hf_character_data.py first.\n' >&2
    exit 1
}

if command -v llamafactory-cli >/dev/null 2>&1; then
    LLAMA_FACTORY=(llamafactory-cli)
else
    LLAMA_FACTORY=("$PYTHON_BIN" -m llamafactory.cli)
fi

printf '[INFO] Starting LLaMA-Factory training with config: %s\n' "$CONFIG_PATH"
exec "${LLAMA_FACTORY[@]}" train "$CONFIG_PATH"
