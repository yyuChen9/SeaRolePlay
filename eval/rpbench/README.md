# Role-Play Bench 评测

基于 [`MiniMaxAI/role-play-bench`](https://huggingface.co/datasets/MiniMaxAI/role-play-bench) 对比
base / SFT / DPO 三个模型的角色扮演能力。

## 这个 benchmark 提供了什么，没提供什么

数据集发布了三张表：

| 表 | 内容 | 我们的用法 |
| --- | --- | --- |
| `seeds` | 45 en / 50 zh 个角色卡 + 开场白 + 首轮用户输入 | **评测输入** |
| `dialogues` | 11 个已知模型跑出的 102 轮对话（1485 条） | 校准 judge |
| `evaluations` | 上述对话的 6 维分数 | 校准 judge |

**没有发布 judge prompt 和 rubric。** 因此 [`judge.py`](judge.py) 是我们按数据集卡片上的维度定义重新实现的，
不是官方评分器。这带来两个必须记住的限制：

1. **分数只能横向比较**（base vs SFT vs DPO），**不能与官方榜单数值对比**。
2. 用卡片给出的权重去重算官方 `evaluations.jsonl`，能复现榜单**排名**但复现不出**数值**
   （我们算出 doubao-1.5-pro 89.87，榜单是 80.64，整体高 5–9 分），说明官方还有一层未公开的归一化。

`calibrate.py` 用来量化「我们的 judge 与官方口径差多远」，在信任任何小幅差距前应先跑一次。

## 评分维度与权重

权重来自数据集卡片：Worlds 50% / Stories 25% / User Preferences 25%。

| 目标 | 维度 | 权重 | 扣分点 |
| --- | --- | --- | --- |
| Worlds | `basics` | 16.7% | 乱码、语言混杂、复读、截断 |
| Worlds | `logic` | 16.7% | 角色混淆、空间错误、代词指代错误 |
| Worlds | `knowledge` | 16.7% | 违反世界设定、时代错置 |
| Stories | `diversity` | 12.5% | 句式复用、口头禅、剧情停滞 |
| Stories | `content_logic` | 12.5% | OOC、人设漂移、无铺垫的态度突变 |
| Preferences | `interaction` | 25% | **替用户发言**、**无视用户**、**过度拒答** |

评分采用「检测失配」而非「奖励好文笔」的口径：每个维度从 100 分起扣，只对可引用的具体缺陷扣分。

## 运行

前置：`roleplay` 环境（训练用，torch 2.8.0）+ `rpbench` 环境（vLLM 0.27.1，torch 2.13）。
两者必须分开——vLLM 会拉高 torch，破坏训练环境的 pin。

```bash
# 一次性：创建 vLLM 环境
source /workspace/miniconda3/etc/profile.d/conda.sh
conda create -n rpbench python=3.12 -y -c conda-forge --override-channels
conda activate rpbench && pip install -r requirements-eval.txt

# 一次性：配置接入点与密钥
cp .env.example .env
# 编辑 .env，至少填入 JUDGE_API_KEY / JUDGE_URL / JUDGE_MODEL

# 下载评测输入
conda activate roleplay
python eval/rpbench/prepare_seeds.py --lang en --with-reference

# 冒烟测试（3 个 seed x 8 轮，约 5 分钟，验证全链路）
MAX_SEEDS=3 NUM_TURNS=8 CHUNK_SIZE=8 bash eval/rpbench/run_eval.sh

# 正式评测（45 seeds x 1 run x 40 轮）
bash eval/rpbench/run_eval.sh
```

**所有密钥与接入点 URL 都通过 `.env` 注入，代码里没有任何默认值** —— 接入点是部署相关的，
硬编码会跟着仓库泄露出去。`run_eval.sh` 在应用任何默认值之前加载 `.env`，缺 judge 三项配置
会立即报错退出，不会等到跑完几小时生成才失败。已 export 的同名环境变量优先级高于文件，
临时覆盖仍然可用。`.env` 已被 gitignore，不要提交。

`run_eval.sh` 会启动一个 vLLM 进程同时服务三个模型（base 权重只加载一次，两个 LoRA adapter
按请求热切换），依次生成对话、评分、出报告，最后自动关闭 server。

**断点续跑**：已完成的对话和已评分的 chunk 会自动跳过，中断后重跑即可继续。要重头开始就删除
`eval/rpbench/results/`。

## 关键实现决策

- **必须关闭思维链。** adapter 是在 `qwen3_5_nothink` 模板下训练的，而 vLLM 用模型自带的
  chat template，默认开启 thinking。不传 `chat_template_kwargs: {"enable_thinking": false}`
  会让三个模型都把 `Thinking Process:` 写进对话，所有维度分数崩掉。这一点已固化在
  `generate.py` 里，`--thinking` 才会开启。
- **user simulator 用 base 模型**，三个被测模型面对完全一致的用户行为，对比才公平。
  user simulator 被要求简短、多变、会追问和转移话题——放任不管的话它会写出长篇配合性文字，
  让所有 NPC 都显得很好，压缩分数方差。
- **NPC system prompt 采用角色卡格式**（`You're {name} in this fictional never-ending
  roleplay`），与 SFT 训练数据同分布，避免微调模型因没见过的 prompt 格式而被扣分。
- **分块评分**：长对话切成 20 轮一块分别评分再平均，与卡片描述一致。整段 100 轮一次性评分
  会让 judge 丢失细节。
- **会话级聚合**：先把同一会话的 chunk 平均成会话分，再跨会话平均。直接平均所有 chunk 会
  让长会话被隐式加权。
- **配对检验**：角色扮演场景难度差异极大，seed 间方差远大于模型间方差，用独立置信区间比较
  几乎必然得出「无差异」。三个模型跑的是同一批 seed，所以 `report.py` 报告**按 seed 配对的
  差值**，并用置换检验给出 p 值——这才是有统计功效的量。
- **确定性**：随机种子由 seed id 的 crc32 导出（不是 `hash()`——字符串哈希每个进程都加盐，
  会让「可复现」的轨迹每次重跑都不同）。

## 校准 judge

```bash
conda activate roleplay
python eval/rpbench/calibrate.py sample --per-model 3 --num-turns 40 \
  --output eval/rpbench/data/calib.jsonl
python eval/rpbench/judge.py --dialogues eval/rpbench/data/calib.jsonl \
  --output eval/rpbench/results/calib_scores.jsonl
python eval/rpbench/calibrate.py compare --scores eval/rpbench/results/calib_scores.jsonl
```

判读：**模型级 Spearman ≥ 0.7** 时本 judge 可用于横向比较；低于该值说明与官方口径分歧过大，
报告结论需谨慎。绝对值偏移不重要（我们的 judge 整体偏高是已知的），重要的是排序一致。

## 文件

```text
prepare_seeds.py   下载并校验评测输入（pin 了 dataset commit）
generate.py        自对弈生成对话，支持断点续跑
judge.py           LLM judge 评分，分块 + 断点续跑
report.py          聚合、bootstrap 置信区间、配对显著性检验
calibrate.py       用已发布分数校准自建 judge
run_eval.sh        全流程编排
FINDINGS.md        首轮评测结论与后续改进项
```

## 成本参考

45 seeds × 1 run × 40 轮，三个模型：

- GPU：约 1–2 小时（H200 单卡，含 4–6 分钟启动）
- judge：约 270 次 API 调用（45 × 2 chunk × 3 模型）

放大到官方协议（3 runs × 100 轮）约为此 7.5 倍。
