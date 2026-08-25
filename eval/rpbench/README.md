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

官方协议是 **45 seeds × 3 runs × 102 轮**，由已发布数据反推确认：`dialogues_en.json` 恰好
1485 = 11 模型 × 45 seed × 3 run，run_1/run_2/run_3 各 495 条，每个 (模型, seed) 组合都是 3 个
run，且 1485 段对话轮数全部为 102（min = max）。

**本仓库默认是 `RUNS=1` / `NUM_TURNS=40`**，是有意的成本削减，不是配置错误 —— user simulator
按轮计费且每次发全量历史，官方协议约为默认的 7.5 倍。代价是每个 seed 只有一条轨迹，采样噪音和
seed 难度混在一起拆不开；`RUNS=3` 能把同一 seed 内的采样方差估出来，配对数从 45 涨到 135，
是提升统计功效最直接的手段。`generate.py` 的随机种子含 `run_index * 7` 偏移，三个 run 轨迹不同
且各自可复现（仅本地 vLLM 成立）。

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
# 一次性：创建 vLLM 环境（只跑远程 API 模型时可跳过）
source /workspace/miniconda3/etc/profile.d/conda.sh
conda create -n rpbench python=3.12 -y -c conda-forge --override-channels
conda activate rpbench && pip install -r requirements-eval.txt

# 一次性：配置接入点与密钥
cp .env.example .env
# 编辑 .env，至少填入 JUDGE_API_KEY / JUDGE_URL / JUDGE_MODEL / USER_MODEL

# 下载评测输入
conda activate roleplay
python eval/rpbench/prepare_seeds.py --lang en --with-reference

# 冒烟测试（3 个 seed x 8 轮，约 5 分钟，验证全链路）
MAX_SEEDS=3 NUM_TURNS=8 CHUNK_SIZE=8 bash eval/rpbench/run_eval.sh

# 正式评测（45 seeds x 1 run x 40 轮）
bash eval/rpbench/run_eval.sh
```

被测模型由 `NPC_MODELS` 指定，空格分隔，**裸名走本地 vLLM，`label=model` 走远程 API**：

```bash
# 纯远程基线：不启动 vLLM，不占 GPU，也不校验 adapter
NPC_MODELS="gpt52=gpt-5.2-chat fable=claude-fable-5" bash eval/rpbench/run_eval.sh

# 纯本地（默认）
NPC_MODELS="base sft dpo" bash eval/rpbench/run_eval.sh

# 混合：本地 adapter 与远程模型跑同一批 seed，直接可比
NPC_MODELS="base sft dpo gpt52=gpt-5.2-chat" bash eval/rpbench/run_eval.sh
```

本地条目只支持 `base` / `sft` / `dpo`，且**只校验实际点名的 adapter** —— 只跑 `base` 时不会因为
DPO 还没训练而报错。存在本地条目时 vLLM 先跑完并立即释放显存，之后才轮到远程生成与评分。

**所有密钥与接入点 URL 都通过 `.env` 注入，代码里没有任何默认值** —— 接入点是部署相关的，
硬编码会跟着仓库泄露出去。judge、user simulator 和远程 NPC 是三组独立配置，可以指向不同网关。
`run_eval.sh` 在应用任何默认值之前加载 `.env`，缺 judge 三项配置或 `USER_MODEL`
会立即报错退出，不会等到跑完几小时生成才失败。已 export 的同名环境变量优先级高于文件，
临时覆盖仍然可用。`.env` 已被 gitignore，不要提交。

**断点续跑**：已完成的对话和已评分的 chunk 会自动跳过，中断后重跑即可继续。要重头开始就删除
对应的结果目录。

## 常用命令

所有参数都可以用命令行前缀临时覆盖，**不必改 `.env`** —— `run_eval.sh` 的加载器只填当前为空的
变量（`[[ -z "${!key:-}" ]] && export`），已在环境里的值优先。

```bash
# 冒烟测试：单独的结果目录，别混进正式结果（见下面「换参数前先想清楚」）
MAX_SEEDS=2 NUM_TURNS=6 CHUNK_SIZE=6 RUNS=1 \
  RESULTS_DIR=eval/rpbench/results_smoke BASELINE=<某个 label> \
  bash eval/rpbench/run_eval.sh
wc -l eval/rpbench/results_smoke/*.jsonl   # 确认真的有内容，别只看退出码

# 正式评测：45 seeds x 3 runs x 40 轮
RUNS=3 BASELINE=<某个 label> bash eval/rpbench/run_eval.sh

# 临时换被测模型（label 是结果文件名，model 是发给网关的名字）
NPC_MODELS="doubao=doubao-1.5-pro" bash eval/rpbench/run_eval.sh

# 多模型横向对比：同 seed、同 user simulator、同 judge，直接可比
RUNS=3 BASELINE=<基准 label> \
  NPC_MODELS="a=model-a b=model-b" bash eval/rpbench/run_eval.sh
```

中断后重跑同一条命令即可续跑。

### 换参数前先想清楚

续跑的去重键是 `(seed_id, run_id)`，**不含被测模型、不含轮数**。由此：

- **调大 `RUNS` 是增量的**：跑完 `RUNS=1` 再跑 `RUNS=3`，只补 run_2/run_3，run_1 复用。
- **改 `NUM_TURNS` 不触发重跑**：40 轮的 run_1 已存在，改成 102 也会被当成「已完成」跳过。
- **换模型但复用同一个 label 会静默混数据**：`qwen-c=A` 跑过一半改成 `qwen-c=B`，已有记录照样
  跳过，一个文件里混两个模型的对话。**换模型就换 label**，或先删掉对应的
  `dialogues_<label>.jsonl` 和 `scores_<label>.jsonl`。

换协议（runs / 轮数）或换模型时，改 `RESULTS_DIR` 写到新目录是最省心的做法。

### `BASELINE` 必须指向结果里真实存在的 label

默认是 `base`。跑纯远程对比时结果里没有 `base`，`report.py` 只会打印
`[WARN] baseline 'base' 不在结果中，跳过配对比较` 然后继续 —— 不报错，但**唯一有统计功效的
配对检验就这么没了**，只剩几乎必然判「无差异」的独立置信区间。每次都显式指定。

### 排查

- **401 Unauthorized**：judge / user / NPC 是三组独立配置，各自取自己的 key。只配了
  `NPC_BASE_URL` 而漏掉 `NPC_API_KEY` 时，`generate.py` 的 `if api_key:` 为假，**整个
  `Authorization` 头都不会发出**，网关返回 401。`run_eval.sh` 目前只对 user 侧做了
  「有 URL 就必须有 key」的前置校验，NPC 侧没有，所以这个错误要等到生成阶段才暴露。
  症状是 `dialogues_<label>.jsonl` 为 0 字节。
- **重试对确定性失败无效**：payload 在重试循环外构建一次，`seed` 固定，`judge.py` 还是
  `temperature=0`。所以「模型返回空内容」「judge 输出解析失败」这类失败重试 4 次拿到的是同一个
  结果，整段对话/整个 chunk 被丢弃。失败的 seed 只进 stderr 的 `[WARN]`，不落盘。

## 关键实现决策

- **必须关闭思维链。** adapter 是在 `qwen3_5_nothink` 模板下训练的，而 vLLM 用模型自带的
  chat template，默认开启 thinking。不传 `chat_template_kwargs: {"enable_thinking": false}`
  会让模型把 `Thinking Process:` 写进对话，所有维度分数崩掉。这一点已固化在
  `generate.py` 里，`--thinking` 才会开启。**但这是 vLLM 的私有扩展**，托管网关会拒绝整个
  请求而非忽略未知字段，因此走远程 API 的模型（包括远程 user simulator）关不掉思维链；
  若模型默认输出思考过程，会在 `basics` 维度被扣分，解读远程模型分数时需注意。
- **user simulator 全局固定。** 一个模型扮演所有对手方，三个被测模型面对完全一致的用户行为，
  对比才公平。它可以是本地 vLLM 上的模型，也可以是远程 API，但**同一次比较里必须始终是同一个**
  —— 换了 user simulator 的两轮结果不可比，所以结果目录默认按 user model 命名
  （`results_<user-model>/`）来物理隔离。user simulator 被要求简短、多变、会追问和转移话题
  ——放任不管的话它会写出长篇配合性文字，让所有 NPC 都显得很好，压缩分数方差。
- **NPC system prompt 采用角色卡格式**（`You're {name} in this fictional never-ending
  roleplay`），与 SFT 训练数据同分布，避免微调模型因没见过的 prompt 格式而被扣分。
- **分块评分**：长对话切成 20 轮一块分别评分再平均，与卡片描述一致。整段 100 轮一次性评分
  会让 judge 丢失细节。
- **会话级聚合**：先把同一会话的 chunk 平均成会话分，再跨会话平均。直接平均所有 chunk 会
  让长会话被隐式加权。
- **配对检验**：角色扮演场景难度差异极大，seed 间方差远大于模型间方差，用独立置信区间比较
  几乎必然得出「无差异」。所有模型跑的是同一批 seed，所以 `report.py` 报告**按 seed 配对的
  差值**，并用置换检验给出 p 值——这才是有统计功效的量。
- **确定性**：随机种子由 seed id 的 crc32 导出（不是 `hash()`——字符串哈希每个进程都加盐，
  会让「可复现」的轨迹每次重跑都不同）。这只对本地 vLLM 成立：托管 API 至多把 `seed` 当作
  尽力而为，远程模型重跑可能得到不同轨迹。

## 校准 judge

judge 换模型后，之前的校准结论即失效，应重跑一次再看正式评测。校准与主评测互不干扰，可以并行跑；
约 33 次 judge 调用，很便宜。

```bash
conda activate roleplay
mkdir -p eval/rpbench/results_calibration
python eval/rpbench/calibrate.py sample --per-model 3 --num-turns 40 \
  --output eval/rpbench/data/calib.jsonl
python eval/rpbench/judge.py --dialogues eval/rpbench/data/calib.jsonl \
  --output eval/rpbench/results_calibration/calib_scores.jsonl
python eval/rpbench/calibrate.py compare \
  --scores eval/rpbench/results_calibration/calib_scores.jsonl \
  | tee eval/rpbench/results_calibration/calibration.txt
```

判读：**模型级 Spearman ≥ 0.7** 时本 judge 可用于横向比较；低于该值说明与官方口径分歧过大，
报告结论需谨慎。绝对值偏移不重要（我们的 judge 整体偏高是已知的），重要的是排序一致。

校准只涉及 judge 对**已发布转录**的打分，与 user simulator 和被测模型都无关，因此单独放在
`results_calibration/`，不随 user simulator 变化。

## 文件

```text
prepare_seeds.py   下载并校验评测输入（pin 了 dataset commit）
generate.py        自对弈生成对话，支持断点续跑
judge.py           LLM judge 评分，分块 + 断点续跑
report.py          聚合、bootstrap 置信区间、配对显著性检验
calibrate.py       用已发布分数校准自建 judge
run_eval.sh        全流程编排
```

## 成本参考

45 seeds × 1 run × 40 轮，每个被测模型：

- judge：约 90 次调用（45 seeds × 2 chunk）
- user simulator：约 855 次调用（45 seeds × 19 个 user 轮次），且每次都要发全量历史。
  user 走远程时这是**最大的成本项**，约为 judge 的 10 倍
- NPC 本地时另需 GPU 约 20–40 分钟／模型（首次含 4–6 分钟启动）

被测模型数量线性放大以上数字；放大到官方协议（3 runs × 100 轮）约为此 7.5 倍。
