# 数据处理: SeaArt 角色扮演 SFT

把上游的 `sft_sharegpt_v2.jsonl` 转成 LLaMA-Factory 能吃的格式，并解决一个
原生 LF 表达不了的需求：**前 K 个 turn 的 assistant 不算 loss**。

## 为什么需要打补丁

上游数据是**滑动窗口**切出来的：12,789 个 session 切成 74,543 条记录，
每条带着上一窗口的尾巴当上文。这段上文必须让模型看见（否则对话失去连续性），
但**不能在上面算 loss**（否则同一段文本会被反复训练，且 greeting 不是模型该学的目标）。

数据里用 `loss_mask` 逐消息标注了这件事。但 LF 收不到它：

1. **`align_dataset` 会 `remove_columns` 掉一切**（`data/converter.py:410,423`），
   只保留固定七字段 `_prompt/_response/_system/_tools/_images/_videos/_audios`。
   消息上的 `loss_mask` 在归一化时被静默丢弃。
2. **`dataset_info.json` 的列名是白名单硬编码的**（`data/parser.py:81-82`），
   没法声明额外的列。
3. **loss 判定条件写死**：`supervised.py` 里是
   `if mask_history and turn_idx != 0`，只能表达"只训最后一轮"。
   而我们要的 K 是**逐样本不同的**（实测 K ∈ [0, 39]）。

打标能力是现成的（`IGNORE_INDEX` 机制），缺的是**把逐样本参数传进去的通道**。

`tools` 是唯一幸存的自由文本字段：它在白名单内、能装任意字符串、
且一路活到 loss 计算现场。补丁借它偷渡 K，读完立刻置空
（不置空的话载荷会被拼进 system prompt，模型会真的学到它）。

## 用法

```bash
# 0. 一次性: 打补丁 (改的是 site-packages, 不是本仓库)
python3 data_process/patch_lf.py --check    # 先看现状
python3 data_process/patch_lf.py            # 打

# 1. 生成数据 -> data/lf_data/{roleplay_train,roleplay_val}.json
python3 data_process/build_data.py --dry-run    # 只统计
python3 data_process/build_data.py             # 正式生成 (约 2 分钟, 4.3 GB)

# 2. token 级验证 —— 不可省
python3 data_process/verify_loss_mask.py

# 3. 训练
bash scripts/run_train.sh
```

## ⚠️ 补丁会被 pip 冲掉

补丁在 `site-packages/llamafactory/data/processor/supervised.py`，
**不在本仓库**。任何 `pip install -U llamafactory` 都会把它还原，且没有提示。

好消息是它不会静默训错：补丁没了以后 `tools` 载荷会被
`format_tools` 拿去 `json.loads`，直接抛
`RuntimeError: Invalid JSON format in tool description` —— 响亮地失败。

但那是在模型加载完之后才炸。所以 `scripts/run_train.sh` 会在启动前先查：
只要配置里的数据集声明了 `tools` 列，就跑 `patch_lf.py --check`，
不通过就拒绝训练。

## 三个脚本

| 脚本 | 作用 |
|---|---|
| `patch_lf.py` | 给 LF 打/查/撤补丁。锚点逐字符匹配，不符就拒绝改 |
| `build_data.py` | 转换 + 按 session 切分 train/val |
| `verify_loss_mask.py` | 把样本真喂进 LF 管道，逐 token 检查 labels |

### build_data.py 的两道硬断言

**监督量对账**：转换后的可训轮数必须精确等于原始 `sum(loss_mask)`，
不等就拒绝写文件。这把"转换有没有悄悄改变监督信号"变成可验证的等式。

**session 泄漏检查**：同一 session 的记录共享 system（同一角色卡）且
对话前缀大量重叠（实测 gpt 消息重复率 11%，是滑窗的正常产物）。
按记录随机切会让 val session 几乎全部同时出现在 train 里，`eval_loss` 失真。
故按 session 整组切，并按 `(mode, script)` 分层，切完断言交集为空。

### 为什么不信 lang_tag

原始 `lang_tag` 有错（v1 复核发现 205 条标错，138 条标 `en` 实为阿拉伯文）。
分层一律用 `detect_script()` 按 Unicode 字符名判断，不用 `lang_tag`。

## v2 数据概况

| 项 | 值 |
|---|---|
| 记录数 | 74,543（12,789 session） |
| 转换后 | train 73,153 / val 1,390，session 零交集 |
| 可训 assistant 轮 | 2,527,424（与原始 `loss_mask` 精确一致） |
| ctx turns | P50=1, P90=17, max=39 |
| train turns | P50=39, max=40 |
| token 长度 | P50≈10.9k, P90≈13.8k, max≈19.1k |
| 其中 system | P50≈5.9k（超过总长一半） |
| 文字体系 | LATIN 59,923 / CYRILLIC 10,961 / ARABIC 2,810 / 其他 |
| mode | nsfw 61,773 / sfw 12,770 |

**`cutoff_len` 必须 ≥ 32768。** 原配置的 2048 连 system 都装不下，
一条 assistant 回复都学不到 —— 不是调优问题，是训练完全无效。

## 已知遗留

- **DPO 阶段不受此补丁影响。** `pairwise.py` 有自己独立的 `IGNORE_INDEX`
  逻辑（`pairwise.py:66-68`），完全没被触及。DPO 的 ctx 屏蔽是另一件事。
- **1,501 条记录含未替换的 `{{user}}`/`{{char}}` 占位符**
  （含 `{{usser}}`、`{{имя}}` 等拼写变体）。目前未做处理。
- **占位 user 轮实测 5 token**（`<|im_start|>user\n<|im_end|>\n`），不是 0。
  相对 P50 10.9k 可忽略，但推理时必须拼上同样的占位轮，否则训练/推理分布不一致。
