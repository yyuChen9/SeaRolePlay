#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
token 级验证: 打过补丁的 LF 是否真的把"前 K 个 turn 的 assistant"排除出 loss。

【为什么不可省】
patch_lf.py 改的是 loss 计算。改错了不会抛异常, 不会打 warning ——
训练照常跑完, 只是监督信号是错的。唯一能发现这类错误的手段, 就是把
样本真正喂进 LF 的数据管道, 再逐 token 检查 labels。

本脚本不复用 build_data.py 的中间量, 而是直接走 LF 的公开入口
(get_dataset -> align_dataset -> SupervisedDatasetProcessor), 独立地把
labels 解码回文本, 与原始 messages 比对。两条路径都对, 才说明补丁生效。

检查项
    1. 前 K 个 turn 的 assistant 文本, 一个 token 都不在 loss 内
    2. 第 K 个 turn 起的 assistant 文本, 全部在 loss 内
    3. 所有 user 文本都不在 loss 内(原生行为, 顺带回归)
    4. tools 载荷没有泄漏进 input_ids(置空那一步生效)
    5. 未打补丁时本脚本必须失败 —— 见 --expect-unpatched 自测

用法
    python3 data_process/verify_loss_mask.py                    # 默认取 20 条
    python3 data_process/verify_loss_mask.py --num-samples 200
    python3 data_process/verify_loss_mask.py --expect-unpatched # 自测: 确认能抓到未打补丁
"""

import argparse
import json
import sys
from pathlib import Path

CTX_TAG = "__ROLEPLAY_CTX_TURNS__:"
IGNORE_INDEX = -100

REPO = Path(__file__).resolve().parent.parent


def build_one(processor, row, tokenizer):
    """把一条 build_data 产物喂进 LF 的 SupervisedDatasetProcessor。

    绕开 datasets.map 直接调 _encode_data_example, 是因为我们要的就是
    (input_ids, labels) 这一对原始输出; 走 map 还得再拆 batch, 没有额外信息。
    对齐格式必须与 align_dataset 的输出一致: role/content, 且 prompt 为
    奇数条、response 恰好 1 条。
    """
    conv = row["conversations"]
    msgs = [{"role": m["from"], "content": m["value"]} for m in conv]
    return processor._encode_data_example(
        prompt=msgs[:-1],
        response=msgs[-1:],
        system=row.get("system") or "",
        tools=row.get("tools") or "",
        images=[],
        videos=[],
        audios=[],
    )


def check_sample(row, input_ids, labels, tokenizer):
    """返回错误列表, 空列表表示通过。"""
    errs = []
    conv = row["conversations"]
    tools = row.get("tools") or ""
    k = int(tools[len(CTX_TAG):]) if tools.startswith(CTX_TAG) else 0

    if len(input_ids) != len(labels):
        return [f"input_ids({len(input_ids)}) 与 labels({len(labels)}) 不等长"]

    # 落在 loss 内的 token, 解码成一整段文本。
    # 逐 turn 精确定位起止点需要复刻 template 的渲染逻辑, 那等于把被测代码
    # 抄一遍 —— 用"文本包含"判定更独立: 只要 ctx 的 assistant 原文出现在
    # 参训文本里, 就是漏屏蔽。
    kept_ids = [t for t in labels if t != IGNORE_INDEX]
    kept_text = tokenizer.decode(kept_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    if not kept_ids:
        errs.append("没有任何 token 参与 loss")

    # 1 + 2: 按 turn 分组的 assistant 文本
    for turn_idx in range(len(conv) // 2):
        asst = conv[turn_idx * 2 + 1]
        if asst["from"] != "assistant":
            errs.append(f"turn {turn_idx} 的第二条不是 assistant: {asst['from']}")
            continue
        text = (asst["value"] or "").strip()
        if len(text) < 40:
            continue  # 太短容易与别处文本偶然重合, 判定不可靠
        probe = text[:60]
        present = probe in kept_text
        if turn_idx < k and present:
            errs.append(f"ctx turn {turn_idx} 的 assistant 出现在 loss 内(应被屏蔽)")
        if turn_idx >= k and not present:
            errs.append(f"训练 turn {turn_idx} 的 assistant 不在 loss 内(应参训)")

    # 3: user 文本不应参训
    for turn_idx in range(len(conv) // 2):
        user = conv[turn_idx * 2]
        text = (user["value"] or "").strip()
        if len(text) < 40:
            continue
        if text[:60] in kept_text:
            errs.append(f"turn {turn_idx} 的 user 出现在 loss 内")

    # 4: tools 载荷不得泄漏进 prompt
    if CTX_TAG in full_text:
        errs.append(f"tools 载荷泄漏进 input_ids: {CTX_TAG}")

    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(REPO / "data" / "lf_data" / "roleplay_val.json"),
        help="build_data.py 的产物。默认用 val (80 MB); train 是 4.2 GB, "
             "光 json.load 就要几分钟, 而两者走的是同一条编码路径。",
    )
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--template", default="qwen3_5_nothink")
    ap.add_argument("--cutoff-len", type=int, default=32768)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument(
        "--expect-unpatched",
        action="store_true",
        help="自测: 断言补丁未生效。用于确认本脚本确实能抓到问题。",
    )
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"找不到 {data_path}\n先跑: python3 data_process/build_data.py")

    import llamafactory
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    lf_src = Path(llamafactory.__file__).parent / "data" / "processor" / "supervised.py"
    patched = "roleplay ctx-mask patch" in lf_src.read_text(encoding="utf-8")
    print(f"LF {llamafactory.__version__} @ {lf_src}")
    print(f"补丁标记: {'已存在' if patched else '不存在'}")

    if args.expect_unpatched and patched:
        raise SystemExit("--expect-unpatched 但补丁已打, 无法自测。先 patch_lf.py --revert。")
    if not args.expect_unpatched and not patched:
        raise SystemExit("补丁未打。先运行: python3 data_process/patch_lf.py")

    # 走 LF 自己的 load_tokenizer 而不是 AutoTokenizer: Qwen3.5 是多模态模型,
    # 其 template 带 image_token, mm_plugin 会强制要求 processor 存在, 否则
    # 抛 "Processor was not found"。训练时 LF 正是这样加载的。
    print(f"加载 tokenizer/processor: {args.model}")
    model_args = ModelArguments(model_name_or_path=args.model, trust_remote_code=True)
    tok_module = load_tokenizer(model_args)
    tokenizer = tok_module["tokenizer"]
    data_args = DataArguments(template=args.template, cutoff_len=args.cutoff_len)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=tok_module["processor"],
        data_args=data_args,
    )

    rows = json.loads(data_path.read_text(encoding="utf-8"))
    # 优先挑 K>0 的样本: K=0 时补丁分支根本不会触发, 验证不到东西。
    # P50 的 K 是 1, 但 K=0 的样本确实存在, 全随机抽会浪费掉大半配额。
    # 按索引分流而非按值比较 —— `r not in with_ctx` 是对每条巨型 dict 做
    # 深度相等比较, 在 7 万条上是 O(n^2), 实测跑不完。
    def ctx_of(row):
        tools = row.get("tools") or ""
        return int(tools[len(CTX_TAG):]) if tools.startswith(CTX_TAG) else 0

    with_ctx, without = [], []
    for row in rows:
        bucket = with_ctx if ctx_of(row) > 0 else without
        if len(bucket) < args.num_samples:
            bucket.append(row)
        if len(with_ctx) >= args.num_samples and len(without) >= 2:
            break
    # 尽量覆盖大 K: 只有 K 大时"前 K 轮被屏蔽"才是强断言。
    with_ctx.sort(key=ctx_of, reverse=True)
    sample = with_ctx[: max(0, args.num_samples - 2)] + without[:2]
    print(f"样本: {len(sample)} 条 (含 ctx 的 {len(sample)-len(without[:2])} 条, "
          f"K=0 的 {len(without[:2])} 条)\n")

    failed = 0
    for i, row in enumerate(sample):
        k = int((row.get("tools") or CTX_TAG + "0")[len(CTX_TAG):])
        # 未打补丁时 tools 载荷会被 format_tools 当成工具描述去 json.loads,
        # 直接抛 RuntimeError。这是好事(响亮地失败, 而非静默训错), 但意味着
        # 自测里必须把异常算作"检出", 否则脚本自己会崩在这里。
        try:
            input_ids, labels = build_one(processor, row, tokenizer)
        except Exception as exc:
            print(f"[FAIL] #{i} K={k:2d} 数据管道抛异常: {type(exc).__name__}: {str(exc)[:80]}")
            failed += 1
            continue
        errs = check_sample(row, input_ids, labels, tokenizer)
        n_loss = sum(1 for t in labels if t != IGNORE_INDEX)
        status = "OK  " if not errs else "FAIL"
        print(f"[{status}] #{i} K={k:2d} turns={len(row['conversations'])//2:2d} "
              f"tokens={len(input_ids):6d} loss_tokens={n_loss:6d}")
        for e in errs[:4]:
            print(f"         - {e}")
        if errs:
            failed += 1

    print()
    if args.expect_unpatched:
        # 自测语义相反: 未打补丁时 ctx 必然参训, 所以"全都失败"才是预期。
        if failed == 0:
            print("❌ 自测失败: 未打补丁却全部通过 —— 本脚本检不出问题, 验证无效。")
            return 1
        print(f"✅ 自测通过: {failed}/{len(sample)} 条被判失败, 说明检查确实有效。")
        return 0

    if failed:
        print(f"❌ {failed}/{len(sample)} 条未通过 —— 补丁未按预期生效, 不要开始训练。")
        return 1
    print(f"✅ {len(sample)}/{len(sample)} 条全部通过: ctx 轮已排除出 loss, 训练轮完整参训。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
