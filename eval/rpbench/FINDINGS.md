# 首轮评测结论（2026-08-23）

45 个 en seed × 1 run × 40 轮，自建 judge（见 [README](README.md) 的口径限制）。
原始数据：`results_base-user/report.txt`、`results_base-user/summary.json`。
对话转录与 vLLM 日志体积过大未入库，需要时按 [README](README.md) 的步骤重跑生成。

## 结果

| model | n | overall | 95% CI | 相对 base |
| --- | --- | --- | --- | --- |
| dpo | 44 | 73.47 | [70.82, 75.95] | **+0.47** [-1.56, +2.51] p=0.645 不显著 |
| base | 45 | 73.16 | [70.98, 75.35] | — |
| sft | 34 | 53.53 | [49.59, 57.58] | **-20.23** [-24.36, -15.94] p=0.0001 显著 |

**SFT 大幅劣于 base；DPO 与 base 无统计差异。**

## 结论一：SFT 退化是真的，不是 harness 故障

全维度均匀塌陷（basics -21.4、logic -15.1、knowledge -13.0、diversity -22.6、
content_logic -25.5、interaction -23.9）本身很像生成故障，所以逐条排除过：

- `vllm_server.log` 里的 `EngineDeadError` 是**关机噪音**（先 `shutdown triggered` 才报错），引擎全程没崩。
- 34 段完成的 SFT 转录**结构干净**：40/40 轮、0 空回复、0 占位符泄漏、0 思维链泄漏。
- SFT 存在复读病理（逐字重复轮次占比 4.3% vs base 0.1%，2/34 段严重塌陷），
  但**剔除全部含复读的会话后，剩下 24 段零复读会话仍只有 57.2 分**（base 73.2）。

复读只解释了一小部分。Judge 给出的是可引用的具体缺陷，不是模板化低分：

| seed | judge notes 摘要 |
| --- | --- |
| en_dialogue_008 | NPC 自称第三人称 "Miss Ashford"，同时替 Nurse Halloway 说话 |
| en_dialogue_003 | 第 18 轮截断在词中间 `nond`；角色与用户身份对调 |
| en_dialogue_010 | 语法崩坏：`The merchant fearfully, is correct` |
| en_dialogue_032 | 用户的法杖与预言被 NPC 认领（角色反转） |

## 结论二：根因是 SFT 数据里未替换的 `{{user}}` 占位符

`data/character_sft_combined.json`（9736 样本）实测：

| 位置 | 含 `{{user}}` |
| --- | --- |
| system | 9736 / 9736 = 100% |
| user 消息 | 95242 / 95243 ≈ 100% |
| **assistant 消息** | **53037 / 95243 = 55.7%** |

超过一半的 assistant 目标输出里，用户名是字面量 `{{user}}` 而非真实姓名。模型学到的
"用户是谁"没有可指代的落点，推理时就表现为**身份混淆、替用户发言、角色反转** ——
正是 judge 记录的主症状，也解释了 basics / content_logic / interaction 为何同时塌。

`data/storyplay_dpo_roleplay_nsfw.json`（3425 样本）**不含**任何占位符。这个不对称
与"DPO 恢复到 base 水平"一致：DPO 阶段用干净数据把 SFT 学坏的部分覆盖了回去。

> 注意 DPO 配置 `create_new_adapter: true`，隐式参考模型是 SFT。DPO 相对 base 打平，
> 意味着它主要在**修复 SFT 的损伤**，而非在 base 之上取得增益。

## 结论三：11 段缺失会话使 45 vs 34 的对比严格来说不合法

`dialogues_sft.jsonl` 缺 11 个 seed：`en_dialogue_002 / 005 / 007 / 011 / 012 / 019 /
020 / 026 / 035 / 039 / 042`（base 全部成功，dpo 只缺 036）。

`generate.py:105` 在模型连续 4 次返回空内容后抛错并丢弃整段。因为用了确定性 seed，
**重试拿到的是同一个空输出，重试机制对这个失败模式完全无效**。

这些 seed 在 base/dpo 下都成功，故非种子本身的问题，而是 SFT 特有的退化。
缺失的多半是退化**最严重**的那批（严重到吐空），因此 **-20.23 更可能是低估而非高估** ——
方向不变，但配对比较只用了 34 对，报告时必须声明。

## 后续改进

1. **修 SFT 数据**（最高优先级）：把 `{{user}}` / `{{char}}` 替换成角色卡里的真实姓名后重跑 SFT。
   这是唯一一个有明确根因、可直接验证的改动。
2. **修重试逻辑**：`generate.py` 对"空内容"重试时应变动 seed，否则重试无意义；
   同时把失败 seed 落盘（当前 `[WARN]` 只进 stderr，本轮已丢失）。
3. **补 `--only-seeds` 参数**，用于只补跑失败会话，不必整轮重来。
4. **跑 `calibrate.py`**（尚未执行过）：验证自建 judge 与官方评分的 Spearman ≥ 0.7。
   -20 这个量级的差距不太可能是 judge 噪音，但 DPO 的 +0.47 本就落在不显著区间，
   校准前后都不影响"DPO 与 base 无差异"的结论。
5. 重跑后补齐 45 vs 45 的配对比较。
