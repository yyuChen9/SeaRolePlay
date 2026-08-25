#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sft_sharegpt_v2.jsonl  ->  LLaMA-Factory sharegpt 格式  (Qwen3.5-9B)

配合打过补丁的 supervised.py 使用 —— 见 patch_lf.py。原生 LF 无法表达
"前 K 个 turn 的 assistant 不算 loss", 因为 align_dataset 会 remove_columns
掉除固定七字段外的一切, 消息上的 loss_mask 在归一化时被静默丢弃。
补丁借 tools 字段偷渡 K, 本脚本负责生成该载荷。

用法
    python3 data_process/build_data.py --dry-run     # 只统计, 不写文件
    python3 data_process/build_data.py               # 正式生成到 data/lf_data/

【必须先打补丁】没打补丁时 tools 载荷会被当作工具描述编码进 system,
且 ctx 轮次会照常参与 loss —— 训练不会报错, 只会静默训错。
build_data.py 不做这项检查(它不 import llamafactory), 由 scripts/run_train.sh 把关。
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# 原始数据留在仓库外: 4.5 GB, 且是上游产物而非本仓库资产。
#
# 这是 SeaArt 内部 dump, 没有公开下载地址 —— 换机器时必须自己拿到一份, 用
# ROLEPLAY_SRC 或 --src 指向它。期望的字段契约与校验方式见 data/README.md;
# 已验证过的那一份的指纹记在 EXPECTED_FINGERPRINT, 对不上不会拦你, 只会提醒。
SRC_DEFAULT = os.environ.get(
    "ROLEPLAY_SRC",
    "/workspace/roleplay_data/seaart-s1-roleplay-dataset-2026-08-25/sft_sharegpt_v2.jsonl",
)

# 2026-08-25 那份 dump 的指纹。HANDOFF 与 data/README 里的一切统计口径
# (74543 条 / 2527424 可训轮 / 12789 session) 都是在它之上测出来的。
EXPECTED_FINGERPRINT = {
    "bytes": 4486203029,
    "lines": 74543,
    "sha256": "bb7ea742f6dc8c3d9f773cd1b723c78cab14458c583e3018c635fdcf3f9c12c7",
}


ROLE_MAP = {"human": "user", "gpt": "assistant"}

# tools 字段载荷前缀, supervised.py 补丁按此解析
CTX_TAG = "__ROLEPLAY_CTX_TURNS__:"

# greeting 前置的占位 user。渲染为 <|im_start|>user\n<|im_end|>\n, 实测 5 token
# (不是 0 —— 空 content 也要付 im_start/role/im_end 的结构开销)。
# 相对 P50 约 10.9k token 的样本可忽略, 但别当成免费的。
# 【重要】推理时必须拼上同样的占位轮, 否则训练/推理分布不一致。
PLACEHOLDER_USER = ""


def detect_script(text: str, n: int = 4000) -> str:
    """按 Unicode 字符名判主要文字体系。

    原始 lang_tag 不可信: v1 全量复核发现 205 条标错或缺失
    (138 条标 en 实为阿拉伯文, 14 条标 en 实为西里尔文, 42 条为 None)。
    v2 仍带 lang_tag, 同样不作为分层依据 —— 一律以字符体系为准。
    """
    c = Counter()
    for ch in text[:n]:
        if ch.isalpha():
            try:
                c[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                pass
    return c.most_common(1)[0][0] if c else "NONE"


def validate(row: dict) -> str:
    """返回错误信息, 空串表示健康。

    重点复核"前缀不训 / 其后 gpt 全训"规律 —— 不符合的样本无法用 turn 粒度
    表达。这类样本必须丢弃并计数, 绝不能静默改变监督信号。
    """
    msgs = row.get("messages") or []
    mask = row.get("loss_mask") or []
    n_ctx = row.get("n_ctx_msgs")

    if not msgs:
        return "messages 为空"
    if len(msgs) != len(mask):
        return f"messages({len(msgs)}) 与 loss_mask({len(mask)}) 不等长"
    if not isinstance(n_ctx, int) or not (0 <= n_ctx <= len(msgs)):
        return f"n_ctx_msgs 非法: {n_ctx}"
    if not (row.get("system") or "").strip():
        return "system 为空"
    if not row.get("session_id"):
        return "session_id 缺失"      # 按 session 切分依赖此字段

    for j, (m, k) in enumerate(zip(msgs, mask)):
        if m.get("from") not in ROLE_MAP:
            return f"未知角色 {m.get('from')!r} @{j}"
        expect = 0 if (j < n_ctx or m["from"] == "human") else 1
        if k != expect:
            return f"loss_mask 违反前缀规律 @{j}: 实际{k} 期望{expect}"
    return ""


def convert(row: dict) -> tuple[dict | None, str]:
    """返回 (结果, 丢弃原因)。"""
    msgs, mask, n_ctx = row["messages"], row["loss_mask"], row["n_ctx_msgs"]

    # ---- greeting 处理: 前置占位 user ----
    # v2 中 72607/74543 条以 gpt(greeting) 开头。greeting 属 ctx(mask=0), 但 LF
    # 要求首条为 user 且消息数为偶数。插入空 user 后 greeting 成为第 0 个 turn
    # 的 assistant。
    #
    # v2 全量复核的奇偶性质(零反例):
    #     有 greeting -> n_ctx_msgs 必为奇数 (72607/72607)
    #     无 greeting -> n_ctx_msgs 必为偶数 ( 1936/ 1936)
    # 故插入占位后 ctx 必为偶数, 恰好落在 turn 边界, 无需取整补救。
    # 下面仍保留显式检查: 这是"监督信号未被悄悄改变"的前提, 不能只靠注释。
    body = list(msgs)
    ctx = n_ctx
    used_placeholder = False
    if body[0]["from"] == "gpt":
        body = [{"from": "human", "value": PLACEHOLDER_USER}] + body
        ctx = n_ctx + 1
        used_placeholder = True

    if body[0]["from"] != "human":
        return None, f"首条非 human: {body[0]['from']}"
    if ctx % 2 != 0:
        return None, f"ctx 未落在 turn 边界: n_ctx_msgs={n_ctx} 占位={used_placeholder}"

    ctx = min(ctx, len(body))
    n_ctx_turns = ctx // 2

    # 尾部必须是 assistant, 且严格交替
    if body[-1]["from"] != "gpt":
        body = body[:-1]
    if not body or body[-1]["from"] != "gpt":
        return None, "无法对齐成 assistant 结尾"
    if len(body) % 2 != 0:
        return None, f"对齐后消息数为奇数: {len(body)}"
    for a, b in zip(body, body[1:]):
        if a["from"] == b["from"]:
            return None, "出现同角色连续"

    conv = [{"from": ROLE_MAP[m["from"]], "value": m["value"]} for m in body]

    n_total_turns = len(conv) // 2
    n_train_turns = n_total_turns - n_ctx_turns
    if n_train_turns <= 0:
        return None, "无可训练轮次"

    # 仅用参训 assistant 文本判语言, 避免 ctx/user 内容干扰分层
    tgt_text = "".join(
        m["value"] for i, m in enumerate(conv)
        if m["from"] == "assistant" and i // 2 >= n_ctx_turns
    )

    return {
        "conversations": conv,
        "system": row["system"],
        # 载荷: 前 n_ctx_turns 个 turn 的 assistant 不算 loss
        "tools": f"{CTX_TAG}{n_ctx_turns}",
        # 下划线开头为元数据: 不在 LF 白名单内, 会被 remove_columns 丢掉,
        # 仅供本地分层划分与事后排查
        "_id": row.get("id"),
        "_session": row["session_id"],
        "_mode": row.get("mode"),
        "_script": detect_script(tgt_text),
        "_n_ctx_turns": n_ctx_turns,
        "_n_train_turns": n_train_turns,
        "_n_total_turns": n_total_turns,
        "_orig_train_turns": sum(mask),
        "_placeholder": used_placeholder,
    }, ""


def split_by_session(kept: list, val_ratio: float, seed: int) -> tuple[list, list]:
    """按 session 整组切分, 保证 train/val 的 session 集合不相交。

    为什么不能按记录切:
        v2 中 12789 个 session 产出 74543 条记录(p50=4, max=103 条/session)。
        同一 session 的记录共享 system(同一角色卡)与大量重叠对话前缀 —— 实测
        gpt 消息重复率 11%, 正是滑窗产生的。按记录随机切会让 val session 几乎
        全部同时出现在 train 中(v1 实测 100% 泄漏), eval_loss 因此偏低失真,
        失去早停与过拟合判断的参考价值。

    分层: 以 session 为单位, 按该 session 的主导 (mode, script) 分层,
    使 val 在语言与 sfw/nsfw 构成上仍能代表整体。
    """
    by_sess = defaultdict(list)
    for r in kept:
        by_sess[r["_session"]].append(r)

    # 每个 session 的分层键 = 其记录中最常见的 (mode, script)
    sess_stratum = {}
    for sid, rs in by_sess.items():
        key = Counter((r["_mode"], r["_script"]) for r in rs).most_common(1)[0][0]
        sess_stratum[sid] = key

    strata = defaultdict(list)
    for sid, key in sess_stratum.items():
        strata[key].append(sid)

    rng = random.Random(seed)
    val_sessions = set()
    for key in sorted(strata):
        sids = sorted(strata[key])          # 先排序, 保证可复现
        rng.shuffle(sids)
        # 层内 session 数 >= 10 才抽, 否则该层全部留给 train
        n_val = max(1, round(len(sids) * val_ratio)) if len(sids) >= 10 else 0
        val_sessions.update(sids[:n_val])

    train = [r for r in kept if r["_session"] not in val_sessions]
    val = [r for r in kept if r["_session"] in val_sessions]
    return train, val


def fingerprint(path: Path, do_sha: bool = True) -> dict:
    """输入文件指纹。

    转换本身有「监督量对账」保证*内部*一致, 但那只能证明转换没改变监督信号,
    证明不了「这次读的和上次读的是同一份 dump」。上游 dump 不进仓库, 也没有
    公开地址, 所以换机器后唯一能坐实输入版本的东西就是这个指纹。

    sha256 要额外整读一遍 4.5 GB(约 30 秒), 相对 2 分钟的转换可以接受;
    确实不想付这个钱时用 --no-checksum, 但那样 BUILD_INFO 就只剩弱指纹。
    """
    st = path.stat()
    fp = {"path": str(path), "bytes": st.st_size, "sha256": None, "lines": None}
    if not do_sha:
        return fp
    h = hashlib.sha256()
    lines = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
            lines += chunk.count(b"\n")
    fp["sha256"] = h.hexdigest()
    fp["lines"] = lines
    return fp


def git_commit() -> str | None:
    """记录转换脚本自身的版本 —— 产物是否需要重建, 取决于代码有没有变。"""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def compare_fingerprint(fp: dict) -> None:
    """与已验证的那份 dump 比对。不匹配只警告, 不拦 —— 上游出新版本是正常的,
    但此时仓库里记的所有统计数字都需要重新确认, 必须让人看见。"""
    diffs = [
        f"{k}: 期望 {EXPECTED_FINGERPRINT[k]} 实际 {fp[k]}"
        for k in ("bytes", "lines", "sha256")
        if fp.get(k) is not None and fp[k] != EXPECTED_FINGERPRINT[k]
    ]
    if not diffs:
        print("  ✅ 与已验证的 2026-08-25 dump 一致")
        return
    print("  ⚠️  与已验证的 dump 不一致:")
    for d in diffs:
        print(f"      {d}")
    print("      -> 这不是错误, 但 docs/HANDOFF.md 与 data/README.md 里的统计数字")
    print("         (74543 条 / 2527424 可训轮 / 12789 session 等) 均以那份为准,")
    print("         换了输入就需要按本次输出重新核对, 别直接沿用旧数字。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)

    # 产物落在仓库 data/ 下, 与 dataset_info.json 的注册项对应。
    # data/*.json(l) 已被 gitignore, 不会误提交。
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "lf_data"),
    )
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-checksum", action="store_true",
        help="跳过输入 sha256(省约 30 秒), BUILD_INFO 只留大小作为弱指纹",
    )
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(
            f"找不到输入文件: {src}\n"
            "\n"
            "这是 SeaArt 内部 dump(sft_sharegpt_v2.jsonl, 约 4.5 GB), 不在本仓库内,\n"
            "也没有公开下载地址 —— 需要通过内部渠道取得一份, 然后指定路径:\n"
            "\n"
            "    ROLEPLAY_SRC=/path/to/sft_sharegpt_v2.jsonl python3 data_process/build_data.py\n"
            "    # 或  python3 data_process/build_data.py --src /path/to/sft_sharegpt_v2.jsonl\n"
            "\n"
            "拿到 dump 后先跑 --dry-run: 它会校验字段契约并打印丢弃原因, 两分钟的\n"
            "转换不会白跑。期望的字段契约见 data/README.md「上游 dump 字段契约」一节。"
        )

    fp = fingerprint(src, do_sha=not args.no_checksum)
    print("=== 输入指纹 ===")
    print(f"  {src}")
    print(f"  bytes  : {fp['bytes']}")
    print(f"  lines  : {fp['lines']}")
    print(f"  sha256 : {fp['sha256']}")
    compare_fingerprint(fp)
    print()

    kept, dropped = [], Counter()
    total = 0
    with src.open(encoding="utf-8") as f:
        for line in f:
            if args.limit and total >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped["JSON 解析失败"] += 1
                continue

            err = validate(row)
            if err:
                dropped[f"校验: {err}"] += 1
                continue

            out, why = convert(row)
            if out is None:
                dropped[why] += 1
                continue
            kept.append(out)

    print(f"读入 {total} -> 保留 {len(kept)}, 丢弃 {sum(dropped.values())}")
    if dropped:
        print("\n=== 丢弃原因 ===")
        for r, c in dropped.most_common(15):
            print(f"  {c:5d}  {r}")
    if not kept:
        raise SystemExit("无样本通过, 请检查数据")

    # ---- 监督量对账: 转换后可训轮数必须等于原始 sum(loss_mask) ----
    conv_turns = sum(r["_n_train_turns"] for r in kept)
    orig_turns = sum(r["_orig_train_turns"] for r in kept)
    print("\n=== 监督量对账 ===")
    print(f"  转换后可训 assistant 轮 : {conv_turns}")
    print(f"  原始 sum(loss_mask)     : {orig_turns}")
    print(f"  差异                    : {conv_turns - orig_turns}")
    if conv_turns != orig_turns:
        raise SystemExit(
            "监督量不一致 —— 转换逻辑有误, 拒绝写出。\n"
            "ctx 应天然落在 turn 边界(见 README 奇偶规律), 差异非 0 说明前提被打破。"
        )
    print("  ✅ 完全一致")

    print("\n=== 样本画像 ===")
    print("  mode    :", dict(Counter(r["_mode"] for r in kept)))
    print("  文字体系:", dict(Counter(r["_script"] for r in kept)))
    n_ph = sum(1 for r in kept if r["_placeholder"])
    print(f"  占位user : {n_ph} 条 / 原生 user 开头 {len(kept)-n_ph} 条")
    p = lambda a, q: sorted(a)[min(len(a) - 1, int(len(a) * q))]
    ct = [r["_n_ctx_turns"] for r in kept]
    tt = [r["_n_train_turns"] for r in kept]
    print(f"  ctx turns  : P50={p(ct,.5)} P90={p(ct,.9)} max={max(ct)}")
    print(f"  train turns: P50={p(tt,.5)} P90={p(tt,.9)} max={max(tt)}")

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return

    train, val = split_by_session(kept, args.val_ratio, args.seed)

    # ---- 泄漏自检: 这是本脚本存在的主要理由, 必须硬断言 ----
    s_tr = {r["_session"] for r in train}
    s_va = {r["_session"] for r in val}
    overlap = s_tr & s_va
    print("\n=== 切分 ===")
    print(f"  train: {len(train):5d} 条 / {len(s_tr)} session")
    print(f"  val  : {len(val):5d} 条 / {len(s_va)} session")
    print(f"  session 交集: {len(overlap)}")
    if overlap:
        raise SystemExit(f"session 泄漏 {len(overlap)} 个 —— 切分逻辑有误, 拒绝写出")
    print("  ✅ 无泄漏")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, rows in [("roleplay_train.json", train), ("roleplay_val.json", val)]:
        p_ = out_dir / name
        with p_.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        sizes[name] = p_.stat().st_size
        print(f"  写出 {p_}  ({len(rows)} 条, {sizes[name]/1e6:.1f} MB)")

    # ---- 溯源记录 ----
    # 产物 4.3 GB, 被 gitignore, 只存在于跑过转换的那台机器上。没有这份记录,
    # 事后无从判断 data/lf_data/ 里的东西是哪份输入、哪版脚本、哪组参数产出的
    # —— 而 train/val 切分依赖 seed, 换了参数重建出的 val 就不是同一批 session。
    # 与产物同目录, 因此同样不进仓库; 它服务的是"这台机器上的产物可追溯"。
    build_info = {
        "source": fp,
        "source_matches_verified_dump": all(
            fp.get(k) is None or fp[k] == EXPECTED_FINGERPRINT[k]
            for k in ("bytes", "lines", "sha256")
        ),
        "script_git_commit": git_commit(),
        "args": {
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "limit": args.limit,
        },
        "records": {
            "read": total,
            "kept": len(kept),
            "dropped": sum(dropped.values()),
            "train": len(train),
            "val": len(val),
        },
        "sessions": {"train": len(s_tr), "val": len(s_va), "overlap": len(overlap)},
        "supervision": {
            "converted_train_turns": conv_turns,
            "orig_sum_loss_mask": orig_turns,
        },
        "output_bytes": sizes,
    }
    info_path = out_dir / "BUILD_INFO.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(build_info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"  写出 {info_path}  (溯源记录)")

    print("\n=== val 分层构成 ===")
    for k, c in sorted(Counter((r["_mode"], r["_script"]) for r in val).items()):
        print(f"  {k}: {c}")


if __name__ == "__main__":
    main()
