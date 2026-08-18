#!/usr/bin/env bash

set -Eeuo pipefail

LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-/workspace/LLaMA-Factory}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '[ERROR] Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

if [[ -d "$LLAMA_FACTORY_DIR/.git" ]]; then
    printf '[INFO] Installing existing LLaMA-Factory checkout: %s\n' "$LLAMA_FACTORY_DIR"
    "$PYTHON_BIN" -m pip install -e "$LLAMA_FACTORY_DIR"
else
    printf '[INFO] Installing LLaMA-Factory from PyPI.\n'
    "$PYTHON_BIN" -m pip install -U llamafactory
fi

if command -v llamafactory-cli >/dev/null 2>&1; then
    llamafactory-cli version
else
    "$PYTHON_BIN" -m llamafactory.cli version
fi
