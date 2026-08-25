#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给 LLaMA-Factory 打补丁: 支持"前 K 个 turn 的 assistant 不算 loss"。

【注意】本补丁作用于 LF 源码, 与 27B 项目**共用同一份** supervised.py。
若 27B 侧已打过, 此处 --check 会直接通过, 无需重打。
补丁内容与模型无关, 只依赖 LF 版本(实测 0.9.6.dev0)。

为什么需要这个补丁, 见 README.md 第 2 节(三条原生路径全部堵死的验证)。

原理
----
LF 已内建"把某轮 target 置为 IGNORE_INDEX"的机制, 但判断条件写死为
    if self.data_args.mask_history and turn_idx != 0:   # train on the last turn only
即只支持"只训最后一轮"。我们要的是"前 K 轮不训、其后全训", 且 K 逐样本不同
(实测 K ∈ [0, 39])。

本补丁在两处插入代码:
    1) 从 tools 字段解析 __ROLEPLAY_CTX_TURNS__:K, 随后把 tools 置空
    2) 在原 mask_history 分支后加 elif turn_idx < K -> IGNORE_INDEX

改动挂在 elif 上, 原生 mask_history 行为完全不变。

用法
    python3 patch_lf.py --check     # 只检查现状, 不修改  <- 建议先跑这个
    python3 patch_lf.py             # 打补丁
    python3 patch_lf.py --revert    # 从备份还原

--check 的退出码是脚本契约(scripts/run_train.sh 依赖它):
    0 = 补丁已生效且与预期逐字节一致
    1 = 未打补丁, 或与预期不一致
    2 = 已打补丁但无备份, 无法核对
"""

import argparse
import shutil
import sys
from pathlib import Path

CTX_TAG = "__ROLEPLAY_CTX_TURNS__:"
MARK = "# >>> roleplay ctx-mask patch"

# 锚点必须与源码逐字符一致。不一致说明 LF 版本与预期不符 -> 拒绝盲改。
ANCHOR1 = """        discarding_history_cot = self.data_args.mask_history and not self.template.preserve_thinking
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools, discarding_history_cot)"""

REPLACE1 = f'''        discarding_history_cot = self.data_args.mask_history and not self.template.preserve_thinking
{MARK} : 从 tools 字段解析出"前 K 个 turn 不算 loss", 并将 tools 置空
        _roleplay_ctx_turns = 0
        if tools and tools.startswith("{CTX_TAG}"):
            try:
                _roleplay_ctx_turns = int(tools[len("{CTX_TAG}"):])
            except ValueError:
                _roleplay_ctx_turns = 0
            tools = ""   # 置空, 否则载荷会被当成工具描述编码进 prompt
{MARK} end
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools, discarding_history_cot)'''

ANCHOR2 = """            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids"""

REPLACE2 = f'''            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
{MARK} : 前 K 个 turn 属于上下文, 其 assistant 不计 loss
            elif turn_idx < _roleplay_ctx_turns:
                target_label = [IGNORE_INDEX] * target_len
{MARK} end
            else:
                target_label = target_ids'''


def locate() -> Path:
    """定位 supervised.py。用 import 而非硬编码路径, 避免改错副本。"""
    import llamafactory
    p = Path(llamafactory.__file__).parent / "data" / "processor" / "supervised.py"
    if not p.exists():
        raise SystemExit(f"找不到 {p}")
    return p


def expected_source(orig: str) -> str:
    """由原生源码推出打完补丁应有的内容。"""
    for i, a in enumerate([ANCHOR1, ANCHOR2], 1):
        if a not in orig:
            raise SystemExit(f"锚点 {i} 未在原生源码中匹配 —— LF 版本不符, 拒绝处理")
    return orig.replace(ANCHOR1, REPLACE1, 1).replace(ANCHOR2, REPLACE2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查现状, 不修改")
    ap.add_argument("--revert", action="store_true", help="从备份还原")
    args = ap.parse_args()

    target = locate()
    backup = target.with_suffix(".py.roleplay.bak")
    src = target.read_text(encoding="utf-8")
    patched = MARK in src

    print(f"目标文件: {target}")
    print(f"备份文件: {backup}  {'(存在)' if backup.exists() else '(不存在)'}")
    print(f"补丁标记: {'已存在' if patched else '不存在'}")

    # ---- --check: 逐字节核对现状是否等于预期 ----
    # 退出码是给脚本用的契约, 只有"补丁确实生效"才返回 0:
    #   0 = 已打且与预期一致   1 = 未打 / 与预期不一致   2 = 已打但无备份可核对
    # 未打补丁曾返回 0, 导致 run_train.sh 的守卫在补丁被 pip 冲掉后照常放行。
    if args.check:
        if not patched:
            print("\n状态: 原生未打补丁。运行 `python3 patch_lf.py` 打补丁。")
            return 1
        if not backup.exists():
            print("\n⚠️ 已打补丁但无备份, 无法核对原生内容。")
            return 2
        want = expected_source(backup.read_text(encoding="utf-8"))
        if src == want:
            print("\n✅ 现有补丁与预期逐字节一致 —— 可直接复用, 无需重打。")
            return 0
        print("\n❌ 现有补丁与预期不一致! 建议 --revert 后重打。")
        import difflib
        for line in list(difflib.unified_diff(
                want.splitlines(), src.splitlines(),
                "expected", "actual", lineterm=""))[:40]:
            print("   ", line)
        return 1

    # ---- --revert ----
    if args.revert:
        if not backup.exists():
            raise SystemExit(f"找不到备份 {backup}")
        shutil.copy2(backup, target)
        print(f"\n已从备份还原 -> {target}")
        return 0

    # ---- 打补丁 ----
    if patched:
        print("\n补丁已存在, 不重复打。如需重打请先 --revert。")
        print("建议运行 `--check` 核对现有补丁是否与预期一致。")
        return 0

    expected_source(src)      # 先验锚点, 不匹配则在此终止
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"\n已备份 -> {backup}")
    target.write_text(expected_source(src), encoding="utf-8")
    print(f"补丁已写入 {target}")
    print("\n下一步: python3 verify_loss_mask.py   (token 级独立验证, 不可省)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
