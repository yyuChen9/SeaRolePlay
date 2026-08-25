#!/usr/bin/env bash
# End-to-end role-play-bench evaluation of any mix of local and hosted models.
#
# NPC_MODELS lists the models under test. A bare name is served by the local
# vLLM process (base weights loaded once, LoRA adapters hot-swapped per
# request); a `label=model` pair is called over a hosted OpenAI-compatible API.
# vLLM only starts when at least one bare name is present, so an API-only
# comparison never touches the GPU.
#
# vLLM lives in its own conda env (`rpbench`) because it pulls torch 2.13,
# which would break the training env's pinned torch 2.8.0. The generation and
# judging clients only need httpx and run from either env.
#
# Usage:
#   bash eval/rpbench/run_eval.sh                 # 45 seeds x 1 run x 40 turns
#   NUM_TURNS=60 RUNS=2 bash eval/rpbench/run_eval.sh
#   MAX_SEEDS=3 NUM_TURNS=8 bash eval/rpbench/run_eval.sh   # quick smoke test
#
#   # hosted baselines only -- no GPU, no adapters needed
#   NPC_MODELS="gpt52=gpt-5.2-chat" bash eval/rpbench/run_eval.sh
#
#   # local adapters plus a hosted baseline, scored side by side
#   NPC_MODELS="base sft dpo gpt52=gpt-5.2-chat" bash eval/rpbench/run_eval.sh
#
# Re-running resumes: completed dialogues and scored chunks are skipped, so an
# interrupted run continues where it stopped. To start clean, delete the
# results directory (see RESULTS_DIR below).

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Load .env FIRST, before any default is applied. The loader below only fills
# variables that are still empty, so anything defaulted above this point would
# silently shadow the .env value and the file would appear to be ignored.
# Values already exported in the environment still win, so a one-off
# `JUDGE_API_KEY=... bash run_eval.sh` overrides the file.
# .env is gitignored -- never commit it. See .env.example for the template.
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
    printf '[INFO] 加载环境配置: %s\n' "$ENV_FILE"
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%$'\r'}"
        [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        # Trim surrounding whitespace and optional quotes around the value.
        key="${key#"${key%%[![:space:]]*}"}"
        key="${key%"${key##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        [[ "$value" == \"*\" && "$value" == *\" ]] && value="${value:1:-1}"
        [[ "$value" == \'*\' && "$value" == *\' ]] && value="${value:1:-1}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        # Only set if not already present, so explicit env vars take precedence.
        [[ -z "${!key:-}" ]] && export "$key=$value"
    done < "$ENV_FILE"
fi

# Conda entry point. Derived from whatever conda is on PATH so no machine-local
# path is baked in; override CONDA_SH in .env when conda is not on PATH.
if [[ -z "${CONDA_SH:-}" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
VLLM_ENV="${VLLM_ENV:-rpbench}"
CLIENT_ENV="${CLIENT_ENV:-roleplay}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
SFT_ADAPTER="${SFT_ADAPTER:-saves/qwen3.5-9b/character-lora}"
DPO_ADAPTER="${DPO_ADAPTER:-saves/qwen3.5-9b/character-dpo}"

PORT="${PORT:-8000}"
SERVER_URL="http://127.0.0.1:${PORT}/v1"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

SEEDS="${SEEDS:-eval/rpbench/data/seeds_en.json}"
NUM_TURNS="${NUM_TURNS:-40}"
RUNS="${RUNS:-1}"
MAX_SEEDS="${MAX_SEEDS:-0}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-16}"
BASELINE="${BASELINE:-base}"

# Judge endpoint. No default -- the gateway is deployment-specific, so it must
# come from .env (see .env.example).
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_URL="${JUDGE_URL:-}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"

# User simulator. One fixed model plays the user for every NPC, so all models
# under test face identical user behaviour. Leave USER_BASE_URL empty to use the
# local vLLM server (USER_MODEL must then be a name it serves, e.g. `base`).
# No default model name: silently defaulting to a local name would point a
# remote-only run at a vLLM server that was never started.
USER_MODEL="${USER_MODEL:-}"
USER_BASE_URL="${USER_BASE_URL:-}"
USER_API_KEY="${USER_API_KEY:-}"

# Models under test. A bare name is served by the local vLLM (a served-model
# name or a LoRA adapter); `label=model` is called over a hosted API, where
# `label` is what the results are recorded under. vLLM starts only when at
# least one bare name is present.
#   NPC_MODELS="base sft dpo"                     # local only
#   NPC_MODELS="gpt52=gpt-5.2-chat"               # hosted only, no GPU
#   NPC_MODELS="base sft gpt52=gpt-5.2-chat"      # mixed
NPC_MODELS="${NPC_MODELS:-base sft dpo}"
NPC_BASE_URL="${NPC_BASE_URL:-}"
NPC_API_KEY="${NPC_API_KEY:-}"

# Scores produced against different user simulators are not comparable, so the
# results directory is named after the user model to keep them physically
# apart. tr maps path-hostile characters (a `vendor/model` name would otherwise
# create a nested directory).
USER_SLUG="$(printf '%s' "$USER_MODEL" | tr -c 'A-Za-z0-9._-' '-')"
RESULTS_DIR="${RESULTS_DIR:-eval/rpbench/results_${USER_SLUG}}"

# The judge is a paid API; fail early rather than after hours of generation.
MISSING_ENV=()
[[ -n "${ANTHROPIC_AUTH_TOKEN:-}" || -n "${JUDGE_API_KEY:-}" ]] || MISSING_ENV+=(JUDGE_API_KEY)
[[ -n "$JUDGE_URL" ]] || MISSING_ENV+=(JUDGE_URL)
[[ -n "$JUDGE_MODEL" ]] || MISSING_ENV+=(JUDGE_MODEL)
if [[ ${#MISSING_ENV[@]} -gt 0 ]]; then
    printf '[ERROR] 缺少必填配置: %s\n' "${MISSING_ENV[*]}" >&2
    printf '  1. cp .env.example .env 并填入这些值（推荐）\n' >&2
    printf '  2. 或 export 同名环境变量\n' >&2
    printf '  3. 或 JUDGE_API_KEY=... bash eval/rpbench/run_eval.sh\n' >&2
    exit 1
fi
export ANTHROPIC_AUTH_TOKEN="${JUDGE_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}"

# A remote user simulator is billed per turn and is called far more often than
# the judge (one call per user turn, per model, with the full history), so a
# missing key must fail here rather than after hours of generation.
if [[ -n "$USER_BASE_URL" && -z "$USER_API_KEY" ]]; then
    printf '[ERROR] USER_BASE_URL 已设置但缺少 USER_API_KEY（在 .env 中填写）\n' >&2
    exit 1
fi
if [[ -n "$REF_MODELS" && -z "$REF_BASE_URL" ]]; then
    printf '[ERROR] REF_MODELS 已设置但缺少 REF_BASE_URL（或 USER_BASE_URL）\n' >&2
    exit 1
fi

if [[ -z "$LOCAL_MODELS" && -z "$REF_MODELS" ]]; then
    printf '[ERROR] LOCAL_MODELS 和 REF_MODELS 不能同时为空\n' >&2
    exit 1
fi

# Every `source "$CONDA_SH"` below assumes this resolved; fail here rather than
# midway through with a confusing "No such file or directory".
if [[ -z "${CONDA_SH:-}" || ! -f "$CONDA_SH" ]]; then
    printf '[ERROR] 找不到 conda.sh: %s\n' "${CONDA_SH:-（未设置，且 conda 不在 PATH 上）}" >&2
    printf '在 .env 中设置 CONDA_SH，例如 CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh\n' >&2
    exit 1
fi

# Every model in one comparison must face the same user simulator; mixing runs
# generated against different user models makes the scores incomparable.
printf '[INFO] user simulator: %s @ %s\n' "$USER_MODEL" "${USER_BASE_URL:-本地 vLLM}"
printf '[INFO] judge: %s @ %s\n' "$JUDGE_MODEL" "$JUDGE_URL"
printf '[INFO] 本地被测: %s\n' "${LOCAL_MODELS:-（无）}"
printf '[INFO] 外部参照: %s\n' "${REF_MODELS:-（无）}"

if [[ -n "$LOCAL_MODELS" ]]; then
    for path in "$SFT_ADAPTER" "$DPO_ADAPTER"; do
        [[ -d "$path" ]] || { printf '[ERROR] adapter 不存在: %s\n' "$path" >&2; exit 1; }
    done
fi

if [[ ! -f "$SEEDS" ]]; then
    printf '[INFO] seeds 不存在，正在下载: %s\n' "$SEEDS"
    # shellcheck disable=SC1090
    source "$CONDA_SH" && conda activate "$CLIENT_ENV"
    python eval/rpbench/prepare_seeds.py --lang en --with-reference
fi

mkdir -p "$RESULTS_DIR"

# --- start the server -------------------------------------------------------
# Only needed for the local models; a reference-only run skips the GPU entirely.
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        printf '[INFO] 关闭 vLLM server (pid %s)\n' "$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

if [[ -n "$LOCAL_MODELS" ]]; then
    SERVER_LOG="$RESULTS_DIR/vllm_server.log"
    printf '[INFO] 启动 vLLM（base + 2 个 adapter），日志: %s\n' "$SERVER_LOG"

    # shellcheck disable=SC1090
    source "$CONDA_SH" && conda activate "$VLLM_ENV"
    python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --served-model-name base \
        --enable-lora \
        --lora-modules "sft=$SFT_ADAPTER" "dpo=$DPO_ADAPTER" \
        --max-lora-rank 8 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --host 127.0.0.1 --port "$PORT" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    trap cleanup EXIT INT TERM

    printf '[INFO] 等待 server 就绪（首次加载权重+编译 kernel 约 4-6 分钟）...\n'
    for attempt in $(seq 1 120); do
        if curl -sf --max-time 3 "$SERVER_URL/models" >/dev/null 2>&1; then
            printf '[INFO] server 已就绪\n'
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            printf '[ERROR] server 启动失败，日志末尾：\n' >&2
            tail -30 "$SERVER_LOG" >&2
            exit 1
        fi
        sleep 5
        [[ $attempt -eq 120 ]] && { printf '[ERROR] server 启动超时（10 分钟）\n' >&2; exit 1; }
    done
else
    printf '[INFO] LOCAL_MODELS 为空，跳过 vLLM 启动\n'
fi

# --- generate + judge -------------------------------------------------------
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$CLIENT_ENV"

MAX_SEEDS_ARG=()
[[ "$MAX_SEEDS" -gt 0 ]] && MAX_SEEDS_ARG=(--max-seeds "$MAX_SEEDS")

# Shared user-simulator flags. Empty when the user simulator is the local vLLM
# server, in which case generate.py reuses the NPC endpoint.
USER_ARGS=(--user-model "$USER_MODEL")
if [[ -n "$USER_BASE_URL" ]]; then
    USER_ARGS+=(--user-base-url "$USER_BASE_URL" --user-api-key "$USER_API_KEY")
fi

ALL_LABELS=()

for MODEL in $LOCAL_MODELS; do
    printf '\n========== 生成对话: %s (本地) ==========\n' "$MODEL"
    ALL_LABELS+=("$MODEL")
    python eval/rpbench/generate.py \
        --label "$MODEL" \
        --npc-model "$MODEL" \
        --base-url "$SERVER_URL" \
        "${USER_ARGS[@]}" \
        --seeds "$SEEDS" \
        --output "$RESULTS_DIR/dialogues_${MODEL}.jsonl" \
        --num-turns "$NUM_TURNS" \
        --runs "$RUNS" \
        --concurrency "$GEN_CONCURRENCY" \
        "${MAX_SEEDS_ARG[@]}"
done

# The GPU is idle from here on; free it before the reference and judging phases,
# which only need the remote APIs and can run for a while.
cleanup
trap - EXIT INT TERM

for PAIR in $REF_MODELS; do
    LABEL="${PAIR%%=*}"
    REF_MODEL="${PAIR#*=}"
    if [[ "$LABEL" == "$PAIR" || -z "$LABEL" || -z "$REF_MODEL" ]]; then
        printf '[ERROR] REF_MODELS 格式应为 label=model-name，收到: %s\n' "$PAIR" >&2
        exit 1
    fi
    printf '\n========== 生成对话: %s (外部 %s) ==========\n' "$LABEL" "$REF_MODEL"
    ALL_LABELS+=("$LABEL")
    python eval/rpbench/generate.py \
        --label "$LABEL" \
        --npc-model "$REF_MODEL" \
        --base-url "$REF_BASE_URL" \
        --api-key "$REF_API_KEY" \
        --remote-npc \
        "${USER_ARGS[@]}" \
        --seeds "$SEEDS" \
        --output "$RESULTS_DIR/dialogues_${LABEL}.jsonl" \
        --num-turns "$NUM_TURNS" \
        --runs "$RUNS" \
        --concurrency "$GEN_CONCURRENCY" \
        "${MAX_SEEDS_ARG[@]}"
done

for MODEL in "${ALL_LABELS[@]}"; do
    printf '\n========== 评分: %s ==========\n' "$MODEL"
    python eval/rpbench/judge.py \
        --dialogues "$RESULTS_DIR/dialogues_${MODEL}.jsonl" \
        --seeds "$SEEDS" \
        --output "$RESULTS_DIR/scores_${MODEL}.jsonl" \
        --judge-model "$JUDGE_MODEL" \
        --base-url "$JUDGE_URL" \
        --chunk-size "$CHUNK_SIZE" \
        --concurrency "$JUDGE_CONCURRENCY"
done

# --- report -----------------------------------------------------------------
printf '\n========== 结果 ==========\n'
SCORE_FILES=()
for MODEL in "${ALL_LABELS[@]}"; do
    SCORE_FILES+=("$RESULTS_DIR/scores_${MODEL}.jsonl")
done

python eval/rpbench/report.py \
    "${SCORE_FILES[@]}" \
    --baseline "$BASELINE" \
    --json-out "$RESULTS_DIR/summary.json" \
    | tee "$RESULTS_DIR/report.txt"

printf '\n[DONE] 结果目录: %s\n' "$RESULTS_DIR"
