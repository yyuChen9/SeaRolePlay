#!/usr/bin/env bash
#
# 用 .env 里的 SWANLAB_API_KEY 完成 swanlab 登录, 并校验它对服务端确实有效。
#
# 为什么需要这个脚本, 而不是直接 `swanlab login`:
#   原生 `swanlab login` 是交互式的, 会在 TTY 里问你要 key。训练常常跑在
#   nohup/后台里, 没有 TTY —— 那时 swanlab.init() 会抛
#       RuntimeError: Failed to initialize SwanLab in online mode: no TTY ...
#   而且这一步发生在 trainer 的 on_train_begin, 即 9B 模型加载完之后,
#   要等十几分钟才看得到。所以正式开跑前先用本脚本把凭证坐实。
#
# 与 run_train.sh 的分工:
#   run_train.sh 只做本地预检(Settings().api_key 是否存在), 不联网 ——
#   它不能替你发现"key 写错了但格式合法"。本脚本会真的打一次服务端。
#
# 用法
#     bash scripts/swanlab_login.sh              # 校验 + 写入本地凭证
#     bash scripts/swanlab_login.sh --check      # 只校验, 不落盘
#     SWANLAB_API_KEY=xxx bash scripts/swanlab_login.sh   # 不经 .env, 一次性

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT_DIR"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

# .env 加载器与 run_train.sh 保持一致: 已导出的变量优先, 便于一次性覆盖。
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

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
    printf '[ERROR] 没有 SWANLAB_API_KEY。在 %s 里加一行:\n' "$ENV_FILE" >&2
    printf '    SWANLAB_API_KEY=<你的 key>\n' >&2
    printf '  key 在 https://swanlab.cn 的设置页获取。\n' >&2
    printf '  不想上报时: SWANLAB=0 bash scripts/run_train.sh\n' >&2
    exit 1
fi

# key 不作为命令行参数传给 python —— 那会让它出现在进程列表里。走环境变量。
SAVE_MODE="$CHECK_ONLY" "$PYTHON_BIN" - <<'PYLOGIN'
import os
import sys

try:
    import swanlab
except ImportError:
    print("[ERROR] 没装 swanlab: pip install swanlab", file=sys.stderr)
    sys.exit(1)

api_key = os.environ["SWANLAB_API_KEY"]
check_only = os.environ.get("SAVE_MODE") == "1"

# relogin=True 强制真的走一次服务端校验, 否则 swanlab 见到已有凭证会直接短路,
# 那样就测不出新 key 是否有效了。
# save=False 时只校验不落盘, 供 --check 使用。
try:
    swanlab.login(api_key=api_key, relogin=True, save=not check_only, timeout=20)
except Exception as exc:  # 网络不通与 key 无效都在这里, 消息里已含区分信息
    print(f"[ERROR] swanlab 登录失败: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"  key 长度 {len(api_key)}, 前 4 位 {api_key[:4]}... (不打印全文)", file=sys.stderr)
    print("  若是 key 无效, 去 https://swanlab.cn 重新获取并更新 .env", file=sys.stderr)
    sys.exit(1)

# 回读一次, 确认凭证确实对后续进程可见 —— run_train.sh 的预检查的就是这个。
from swanlab.sdk.internal.settings import Settings

if not getattr(Settings(), "api_key", None):
    print("[ERROR] 登录报告成功, 但 Settings() 读不到 api_key。", file=sys.stderr)
    sys.exit(1)

print("[INFO] swanlab 登录成功" + ("(--check: 未落盘)" if check_only else ""))
PYLOGIN

printf '[INFO] 现在可以直接开训: PYTHON_BIN=%s bash scripts/run_train.sh\n' "$PYTHON_BIN"
