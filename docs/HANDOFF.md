# SeaRolePlay 交接文档

更新时间：2026-08-25

上一版（2026-08-14）描述的是 `/workspace/Character` 下的 0.8B 实验，与当前项目已完全不同，本文档整体重写。

常用命令（启动 / 停止 / 看进度 / 启动后必查项 / 陷阱）见 **[RUNBOOK.md](RUNBOOK.md)**。本文档讲背景和设计理由。

---

## 0. 给接手 4 卡 SFT 的人：先读这一节

**当前机器只有 1 张 GPU。** 实测：

```text
$ nvidia-smi -L
GPU 0: NVIDIA H200 (UUID: GPU-b24a4f98-...)
$ python -c "import torch; print(torch.cuda.device_count())"
1
```

4 卡并发在**这台机器上跑不起来**。要么换到 4 卡机器，要么把计划改成单卡。下面三条是切到 4 卡时必须先处理的，尤其第一条。

### 0.1 有效 batch 会静默变成 4 倍（必须改配置）

Transformers 的 `get_total_train_batch_size()` 的公式是
`micro_batch × grad_accum × dp_world_size`（`trainer.py:2360`）。当前配置是
`per_device_train_batch_size: 1 × gradient_accumulation_steps: 16`：

| | 有效 batch | 优化器步数 | warmup 占比 | checkpoint 数 |
|---|---|---|---|---|
| 1 卡（当前配置调参依据） | 16 | 9144 | 1.1% | 45 |
| 4 卡（**同一份配置**） | **64** | **2286** | 4.4% | 11 |

也就是说，什么都不改直接上 4 卡，**等于换了一组超参在训练**：batch 翻 4 倍、总更新步数只剩四分之一。`learning_rate: 1.0e-4` 是在 batch=16 下定的，batch=64 还用同一个 LR 通常需要上调（常见做法是按 √4=2 倍或线性 4 倍缩放），不调则收敛偏慢。

**要保持与单卡完全等价**，把梯度累积除以卡数：

```bash
FORCE_TORCHRUN=1 bash scripts/run_train.sh gradient_accumulation_steps=4
```

`1 × 4 × 4 = 16`，与单卡一致，超参、步数、warmup 占比全部不变。**这是推荐做法** —— 现有 LR/warmup 都是按 batch=16 定的，先拿到能和单卡对比的结果，再单独做扩 batch 的实验。

若确实想用 batch=64 换吞吐，那就是一次独立调参，至少要同步调 `learning_rate` 和 `warmup_steps`（100 步在 2286 步的总长里占比偏高），并且**换个 `output_dir`**，别和单卡结果混在一个目录里。

### 0.2 启动方式：多卡必须先修 PATH

LF 会自己判断，`launcher.py:61`：

```python
is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and ...)
```

只要能看到多张卡就自动起 torchrun，`nproc_per_node` 默认取卡数。

**但 `launcher.py:115` 是按裸名调 `torchrun` 的**，走 `PATH` 解析。本机 `/usr/local/bin/torchrun` 属于系统 python（`/usr/local/lib/python3.12/dist-packages/`），排在 conda env 前面，于是 4 个 rank 全部起在系统解释器下，齐刷刷报：

```
ModuleNotFoundError: No module named 'llamafactory'
```

父 shell 里的检查全部通过（它用的是 conda python），只有子进程炸 —— 所以错误看起来毫无来由。多卡启动必须显式前置 env 的 bin：

```bash
PATH=/workspace/miniconda3/envs/roleplay/bin:$PATH \
FORCE_TORCHRUN=1 bash scripts/run_train.sh gradient_accumulation_steps=4
```

指定用哪几张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PATH=/workspace/miniconda3/envs/roleplay/bin:$PATH \
FORCE_TORCHRUN=1 bash scripts/run_train.sh gradient_accumulation_steps=4
```

（本节原写作"不需要改脚本"，那是在单卡机器上写的，torchrun 这条路径当时从未被走到过。2026-08-26 首次多卡启动即命中。）

脚本里的 ctx-mask 补丁检查、swanlab 预检都在父 shell 里跑**一次**，然后才 exec 给 LF，不会每个 rank 重复执行 —— 这点已确认。

### 0.3 首次 4 卡启动前，先单卡把 tokenize 缓存打出来

`overwrite_cache: false`，但目前**缓存是空的**（`~/.cache/huggingface/datasets` 不存在）。74543 条 × 32k 的 tokenize 很重。DDP 下各 rank 会各自处理数据集，首启动可能出现多份重复 tokenize，也可能触碰 `ddp_timeout`（已设成很大的 180000000，不太会超时，但会白等）。

稳妥做法：先用单卡跑几步让缓存落盘，再切 4 卡。

```bash
bash scripts/run_train.sh max_steps=2 output_dir=/tmp/warm_cache
```

### 0.4 其他 4 卡注意事项

- **显存**：单卡 H200 143 GB，32k 序列 + gradient_checkpointing + LoRA 在单卡跑通过。4 卡每卡负载与单卡相同（micro_batch 仍是 1），显存不是瓶颈。
- **swanlab**：callback 有 `is_world_process_zero` 保护（`trainer_utils.py:867`），只有 rank 0 上报，不会 4 个 rank 开 4 条曲线。
- **`SWANLAB_RUN_ID` 固定**是按 `output_dir` 推导的，所以如果你按 0.1 的建议换了 `output_dir`，实验名也会自动跟着换，不会覆盖旧曲线。

---

## 1. 项目目标

在 `Qwen/Qwen3.5-9B` 上用 SeaArt 角色扮演数据做 LoRA SFT，之后接 DPO，用 RPBench 评测 base / SFT / DPO 三者的角色扮演能力。

项目根目录：`/workspace/SeaRolePlay`

当前阶段：**数据、训练链路、日志、评测框架均已就绪且验证通过，正式 SFT 尚未开跑。**

## 2. 环境

conda 环境 `roleplay`，解释器 `/workspace/miniconda3/envs/roleplay/bin/python`：

```text
Python            3.12.13
LLaMA-Factory     0.9.5
Transformers      5.6.0
PyTorch           2.8.0+cu128
peft              0.18.1
trl               0.24.0
swanlab           0.9.7
```

精确复现：`pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128`。人工维护的宽松版本在 `requirements.txt`。

注意 vLLM 环境（评测用）与训练环境必须分开，torch 版本不兼容。

调用脚本时统一传 `PYTHON_BIN`：

```bash
PYTHON_BIN=/workspace/miniconda3/envs/roleplay/bin/python bash scripts/run_train.sh
```

## 3. 数据

原始 dump 在仓库外：`/workspace/roleplay_data/`（4.5 GB，不进仓库）。

`data_process/build_data.py` 负责转换与切分，产出：

```text
data/lf_data/roleplay_train.json    73153 条 (4.2 GB)
data/lf_data/roleplay_val.json       1390 条 (79 MB)
```

**训练数据含成人/NSFW 内容**（74543 条中 61773 条为 `mode: nsfw`）。使用、分发、发布模型前必须重新核对许可证与平台政策。

### 3.1 train/val 切分

按 **session 整组**切分，按 `(mode, script)` 分层，`--val-ratio 0.02 --seed 42`。构建时**硬断言 session 交集为 0**，不通过拒绝写出（`build_data.py:288`）。

不能按记录切：同一 session 的记录共享角色卡与大量重叠对话前缀，按记录随机切会让 val 几乎全部泄漏进 train（v1 实测 100% 泄漏），eval_loss 会偏低失真。

**没有独立的 test 集。** val 只用于看 `eval_loss`，且当前**不参与任何决策** —— 三个配置里都没有 `load_best_model_at_end` / `metric_for_best_model` / 早停。事实上的测试集是 RPBench（见第 7 节），45 个 seed，与训练数据完全独立。

### 3.2 已知数据问题

1501 条记录含未替换的 `{{user}}` / `{{char}}` 占位符，尚未处理。

## 4. ctx-mask 补丁（重要，改的是 LF 源码）

数据要求「**前 K 个 turn 的 assistant 不算 loss**」，K 逐样本不同（实测 0–39）。LF 原生只支持 `mask_history`（只训最后一轮），三条原生路径全部堵死，因此改了库源码。

补丁文件：`data_process/patch_lf.py`，作用于
`site-packages/llamafactory/data/processor/supervised.py`，共 13 行两处：

1. 从 `tools` 字段解析 `__ROLEPLAY_CTX_TURNS__:K`，**随后把 `tools` 置空**（否则载荷会被当工具描述编码进 prompt）
2. 在原 `mask_history` 分支后加 `elif turn_idx < K: target_label = [IGNORE_INDEX] * target_len`

挂在 `elif` 上，原生行为完全不变。

```bash
python data_process/patch_lf.py --check    # 0=已生效且逐字节一致 1=未打/不一致 2=已打但无备份
python data_process/patch_lf.py            # 打补丁（自动备份 .roleplay.bak）
python data_process/patch_lf.py --revert   # 还原
python data_process/verify_loss_mask.py    # token 级验证，不可省
```

**风险：补丁在 site-packages，不在仓库里，任何 `pip install -U llamafactory` 都会冲掉它。**
`run_train.sh` 启动前会 `--check`，被冲掉会拒绝启动而非静默训错。

`verify_loss_mask.py` 不可省的理由：补丁改的是 loss 计算，改错了**不抛异常、不打 warning，训练照常跑完，只是监督信号是错的**。最近一次运行 20/20 通过。

**DPO 完全没覆盖** —— `pairwise.py:66-68` 有自己独立的 IGNORE_INDEX 逻辑，补丁碰不到。DPO 目前是原生行为，待办。

## 5. SFT 配置

`configs/qwen3_5_9b_lora_sft.yaml`，输出 `saves/qwen3.5-9b/seaart-sft-lora`。

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
template: qwen3_5_nothink
dataset: seaart_sft
eval_dataset: seaart_sft_val
cutoff_len: 32768
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 2.0
warmup_steps: 100
gradient_checkpointing: true
bf16: true
```

两个参数值得说明：

- **`cutoff_len: 32768`**（不是 2048）。实测 400 条抽样：单条 P50=10.9k、P90=13.8k、max=19.1k token，其中 system 一项就占 P50 5.9k。2048 下连 system 都装不下，一条 assistant 回复都学不到 —— 不是调优问题，是训练完全无效。
- **`1 × 16`**（原为 `8 × 2`）。序列长度涨 16 倍后单卡放不下 batch 8，把并行度让给梯度累积，等效 batch 仍为 16。

覆盖参数用 OmegaConf 的 `key=value` 形式（`hparams/parser.py:90-93` 把它们 merge 进 YAML，命令行优先）：

```bash
bash scripts/run_train.sh max_steps=3 output_dir=/tmp/smoke
```

**不要用 `--flag value` 形式**，HfArgumentParser 会报 "Some keys are not used"。

## 6. 训练日志（swanlab）

```bash
bash scripts/swanlab_login.sh          # 校验 key + 落盘（~/.swanlab/.netrc）
bash scripts/swanlab_login.sh --check  # 只校验不落盘
bash scripts/run_train.sh              # 默认上报云端
SWANLAB_MODE=local bash scripts/run_train.sh   # 只写本地
SWANLAB=0 bash scripts/run_train.sh            # 完全不上报
```

已登录账号 **yyuchen**，project 默认 `seaart-roleplay`。

四个设计点：

- **走 LF 自己的 `use_swanlab`，不走 `report_to`** —— 后者会被 `hparams/parser.py:465` 摘掉。
- **run id 由 `output_dir` 推导并固定**。LF 会自动从最后 checkpoint 续训（`parser.py:491`），但 swanlab 默认每次重启开新实验，曲线会碎成几段。固定 `SWANLAB_RUN_ID` + `SWANLAB_RESUME=allow` 让续训落回同一条曲线。
- **项目名用 `SL_PROJECT`，绝不能写 `SWANLAB_PROJECT`**。swanlab 0.9.7 以 `SWANLAB_` 为前缀读环境变量，其中 `project` 是**嵌套子模型**，塞字符串会让 `swanlab.init()` 抛 `SettingsError` 打断训练。而且 swanlab 会自行读取当前目录的 `.env`，所以这个名字写进 `.env` 一样会踩雷。`run_train.sh` 启动前会实例化一次 `Settings()` 预检。
- **`swanlab_login.sh` 联网校验，`run_train.sh` 只做本地预检**。后者只看 key 存不存在，抓不出「格式合法但 key 是错的」；前者用 `relogin=True` 强制真打服务端（不加这个参数，swanlab 见到已有凭证会直接短路返回 True，等于没测）。

API key 只经环境变量传递，不作为命令行参数 —— LF 支持 `swanlab_api_key=`，但那会让密钥出现在进程列表里。

## 7. 评测（RPBench）

`eval/rpbench/run_eval.sh`。被测模型统一用 `NPC_MODELS`：裸名 = 本地 vLLM，`label=model` = 远程 API。无本地条目时完全跳过 GPU。

结果目录按 user simulator 自动命名（`results_<user-model>/`），因为不同 user simulator 下的分数不可比。

**judge 换模型后需要重新校准。** 旧 judge 的模型级 Spearman 是 0.736，刚过 0.7 门槛；换 judge 后这个数字不再适用，跑正式评测前应重跑 `calibrate.py`。

## 8. 已验证 / 未验证

**已验证：**

- 数据转换、session 级切分、零泄漏断言
- ctx-mask 补丁 token 级验证 20/20 通过
- 训练链路端到端（3 步 SFT，loss 1.649 → 1.688，eval_loss 1.664）
- swanlab 本地模式与云端模式均上报成功（服务端回查确认 `FINISHED`，指标齐全）
- swanlab 登录脚本四条路径：好 key / 坏 key(401) / 未配置 / `SWANLAB=0`
- `run_train.sh` 四条守卫路径
- RPBench 纯远程链路冒烟

**未验证：**

- **多卡 DDP 从未跑过**（本机只有 1 卡，没有条件验证）
- 完整 2 epoch 的实际耗时、显存峰值、磁盘占用
- DPO 训练从未实跑
- 合并 adapter 后的推理

## 9. 待办（按优先级）

1. **4 卡前先解决 0.1 的 batch 问题**，否则跑出来的不是当前配置的结果。
2. **`data_process/` 仍未提交**（`?? data_process/`），`run_train.sh` 的补丁守卫依赖它存在。`scripts/swanlab_login.sh` 同样未提交。
3. **DPO 的 adapter 路径不匹配**：SFT 实际输出到 `saves/qwen3.5-9b/seaart-sft-lora`，但下面三处仍指向旧的 `character-lora`，DPO 会找不到 adapter：
   - `configs/qwen3_5_9b_lora_dpo.yaml:8`（及 `:6` 注释）
   - `configs/qwen3_5_9b_lora_dpo_smoke.yaml:19`（及 `:15` 注释）
   - `README.md:187`

   我没有直接改 —— 这三处到底该跟随 SFT 新路径，还是 SFT 该改回旧名，属于命名约定的决定，留给你定。
4. **把 ctx-mask 扩展到 DPO**（`pairwise.py` 独立逻辑）。
5. 处理 1501 条 `{{user}}`/`{{char}}` 占位符。
6. 决定 val 是否要真正发挥作用（加 `load_best_model_at_end` + 早停）。
7. GPU 争用：当前有 `/workspace/miniconda3/envs/urag/bin/python3.10` 占着 24 GB、GPU 利用率 97%。这会显著拖慢训练（实测受争用时 432 s/step，无争用时约 117 s/step）。正式开跑前先确认它是否还在。

## 10. 关键文件

```text
README.md                              # 面向使用者的操作手册
docs/HANDOFF.md                        # 本文档
configs/qwen3_5_9b_lora_sft.yaml       # SFT 配置
configs/qwen3_5_9b_lora_dpo.yaml       # DPO 配置（adapter 路径待修，见待办 3）
scripts/run_train.sh                   # 训练入口（含补丁守卫 + swanlab 预检）
scripts/swanlab_login.sh               # swanlab 登录与联网校验
data_process/build_data.py             # 数据转换与 session 级切分
data_process/patch_lf.py               # ctx-mask 补丁
data_process/verify_loss_mask.py       # token 级 loss mask 验证
eval/rpbench/run_eval.sh               # RPBench 评测编排
data/dataset_info.json                 # LF 数据集注册
.env                                   # 密钥（gitignored，绝不提交）
.env.example                           # 模板，值一律留空
```

**所有密钥与接入点 URL 都通过 `.env` 注入，代码里没有任何默认值** —— 接入点是部署相关的，硬编码会跟着仓库泄露。`.gitignore` 已排除 `.env`、`.swanlab/`（`swanlab login --save local` 会把 key 明文写进去）、`data/*.json`、`data/lf_data/`、`data/raw/`、`saves/*`、`swanlog/`。
