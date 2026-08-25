#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/qwen3_5_9b_lora_sft.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
[[ -f "$CONFIG_PATH" ]] || { printf '[ERROR] Config not found: %s\n' "$CONFIG_PATH" >&2; exit 1; }
cd "$ROOT_DIR"

# Load .env before applying any default, so nothing defaulted here can silently
# shadow the file. Already-exported variables still win, so a one-off
# `SWANLAB_MODE=local bash scripts/run_train.sh` overrides it.
# .env is gitignored -- never commit it. See .env.example for the template.
# (Same loader as eval/rpbench/run_eval.sh; training needs SWANLAB_API_KEY from it.)
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%$'\r'}"
        [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        key="${key#"${key%%[![:space:]]*}"}"
        key="${key%"${key##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        [[ "$value" == \"*\" && "$value" == *\" ]] && value="${value:1:-1}"
        [[ "$value" == \'*\' && "$value" == *\' ]] && value="${value:1:-1}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ -z "${!key:-}" ]] && export "$key=$value"
    done < "$ENV_FILE"
fi

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

# The ctx-mask patch lives in site-packages, not in this repo, so any
# `pip install -U llamafactory` silently reverts it. Without it a dataset that
# carries the __ROLEPLAY_CTX_TURNS__ payload dies deep inside the tokenizer with
# "Invalid JSON format in tool description" -- loud, but only after the model has
# loaded. Check up front instead, and only for configs that actually need it.
NEEDS_PATCH="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import json, pathlib, sys, yaml

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
data_dir = pathlib.Path(config.get("dataset_dir") or "data")
names = []
for key in ("dataset", "eval_dataset"):
    value = config.get(key) or ""
    if isinstance(value, str):
        value = [n.strip() for n in value.split(",") if n.strip()]
    names.extend(value)

info_path = data_dir / "dataset_info.json"
if not info_path.is_file():
    sys.exit(0)

info = json.loads(info_path.read_text(encoding="utf-8"))
# `tools` is how the per-sample ctx-turn count reaches the patched processor;
# a dataset that declares it cannot train correctly on a stock LLaMA-Factory.
for name in names:
    if (info.get(name) or {}).get("columns", {}).get("tools"):
        print(name)
PY
)"

if [[ -n "$NEEDS_PATCH" ]]; then
    if ! "$PYTHON_BIN" "$ROOT_DIR/data_process/patch_lf.py" --check >/dev/null 2>&1; then
        printf '[ERROR] 数据集 %s 依赖 ctx-mask 补丁, 但补丁未生效。\n' "$(echo "$NEEDS_PATCH" | tr '\n' ' ')" >&2
        printf '  补丁在 site-packages 中, pip install -U llamafactory 会将其冲掉。\n' >&2
        printf '  修复:\n' >&2
        printf '    %s data_process/patch_lf.py            # 打补丁\n' "$PYTHON_BIN" >&2
        printf '    %s data_process/verify_loss_mask.py    # token 级验证(不可省)\n' "$PYTHON_BIN" >&2
        exit 1
    fi
    printf '[INFO] ctx-mask 补丁已生效\n'
fi

if command -v llamafactory-cli >/dev/null 2>&1; then
    LLAMA_FACTORY=(llamafactory-cli)
else
    LLAMA_FACTORY=("$PYTHON_BIN" -m llamafactory.cli)
fi

# ---- SwanLab -----------------------------------------------------------------
# LLaMA-Factory has first-class SwanLab support (`use_swanlab`,
# finetuning_args.py:405) and attaches its own callback (tuner.py:78) rather than
# going through transformers' report_to. So this passes use_swanlab=true instead
# of report_to=swanlab -- parser.py:465 strips the latter back out anyway.
#
# The SFT run is long and will be interrupted and resumed more than once.
# LLaMA-Factory auto-resumes from the last checkpoint (hparams/parser.py:491),
# but SwanLab would start a *fresh* experiment each time and the loss curve
# would come out in disconnected fragments. Pinning SWANLAB_RUN_ID to a value
# derived from output_dir makes every resume land back in the same run.
#
# Set SWANLAB=0 to disable entirely; SWANLAB_MODE=local to log to disk only.
SWANLAB="${SWANLAB:-1}"
LOG_ARGS=()
if [[ "$SWANLAB" == "0" ]]; then
    printf '[INFO] swanlab 已禁用 (SWANLAB=0)\n'
else
    OUTPUT_DIR="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PYCFG'
import pathlib, sys, yaml
config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(config.get("output_dir") or "")
PYCFG
)"
    [[ -n "$OUTPUT_DIR" ]] || { printf '[ERROR] 配置里没有 output_dir\n' >&2; exit 1; }

    # Deliberately NOT named SWANLAB_PROJECT. swanlab 0.9.7 builds its Settings via
    # pydantic-settings with env_prefix="SWANLAB_", and `project` there is a *nested*
    # model, so a plain string in SWANLAB_PROJECT makes swanlab.init() die with
    #   SettingsError: error parsing value for field "project"
    # The legacy scalar spelling is SWANLAB_PROJ_NAME; we don't need either, because
    # the project reaches LF through the swanlab_project CLI arg below.
    SL_PROJECT="${SL_PROJECT:-seaart-roleplay}"
    # saves/qwen3.5-9b/seaart-sft-lora -> qwen3.5-9b-seaart-sft-lora
    RUN_SLUG="$(printf '%s' "${OUTPUT_DIR#saves/}" | tr -c 'A-Za-z0-9._-' '-')"
    export SWANLAB_RUN_ID="${SWANLAB_RUN_ID:-$RUN_SLUG}"
    export SWANLAB_RESUME="${SWANLAB_RESUME:-allow}"
    SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
    LOG_ARGS=(
        use_swanlab=true
        "swanlab_project=$SL_PROJECT"
        "swanlab_run_name=$RUN_SLUG"
        "swanlab_mode=$SWANLAB_MODE"
    )
    # swanlab_api_key is deliberately NOT passed as a CLI arg -- it would show up
    # in the process list. SwanLab reads SWANLAB_API_KEY from the environment,
    # which the .env loader above already exported.

    # Fail before the model loads. swanlab.init() runs inside the trainer's
    # on_train_begin, i.e. ~10 minutes in on a 9B model, so anything wrong with the
    # swanlab environment must be caught here instead of after that whole wait.
    #
    # Two separate failures, hence two exit codes from the probe below:
    #   2 = Settings() itself blew up. Every SWANLAB_* name in the environment *and*
    #       in ./.env feeds pydantic-settings (it reads the dotenv from cwd on its
    #       own, so our loader is not the only path in), and the nested fields reject
    #       plain strings. This is how SWANLAB_PROJECT=... kills a run.
    #   1 = Settings() is fine but there is no api_key. Only fatal in cloud mode;
    #       without credentials and without a TTY swanlab.init() raises
    #       RuntimeError: Failed to initialize SwanLab in online mode: no TTY ...
    # `|| SL_PROBE_RC=$?` is load-bearing: under `set -e` a failing command
    # substitution aborts the script at the assignment, so the diagnostics below
    # would never print.
    SL_PROBE_RC=0
    SL_PROBE_ERR="$(
        "$PYTHON_BIN" - <<'PYAUTH' 2>&1
import sys

try:
    from swanlab.sdk.internal.settings import Settings

    settings = Settings()
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(2)
sys.exit(0 if getattr(settings, "api_key", None) else 1)
PYAUTH
    )" || SL_PROBE_RC=$?
    if [[ $SL_PROBE_RC -eq 2 ]]; then
        printf '[ERROR] swanlab 配置无法解析, 训练会在 swanlab.init() 崩溃:\n' >&2
        printf '%s\n' "$SL_PROBE_ERR" >&2
        printf '  检查环境变量与 %s 里的 SWANLAB_* 项。常见原因: SWANLAB_PROJECT\n' "$ENV_FILE" >&2
        printf '  在 swanlab 0.9.7 是嵌套字段, 不接受字符串 —— 项目名请用 SL_PROJECT=。\n' >&2
        exit 1
    fi
    if [[ "$SWANLAB_MODE" == "cloud" && $SL_PROBE_RC -ne 0 ]]; then
        printf '[ERROR] swanlab 未登录, 无法上报。三选一:\n' >&2
        printf '    %s -m swanlab login        # 交互式登录\n' "$PYTHON_BIN" >&2
        printf '    在 .env 中设置 SWANLAB_API_KEY=...\n' >&2
        printf '    SWANLAB_MODE=local bash scripts/run_train.sh   # 只写本地\n' >&2
        printf '  或 SWANLAB=0 bash scripts/run_train.sh 完全关闭上报。\n' >&2
        exit 1
    fi
    printf '[INFO] swanlab: project=%s run=%s mode=%s\n' \
        "$SL_PROJECT" "$SWANLAB_RUN_ID" "$SWANLAB_MODE"
fi

printf '[INFO] Starting LLaMA-Factory training with config: %s\n' "$CONFIG_PATH"
# Extra args go last so they override both the yaml and LOG_ARGS, e.g.
#   bash scripts/run_train.sh max_steps=2 output_dir=/tmp/smoke
exec "${LLAMA_FACTORY[@]}" train "$CONFIG_PATH" "${LOG_ARGS[@]}" "$@"
