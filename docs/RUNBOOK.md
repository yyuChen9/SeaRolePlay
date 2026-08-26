# RUNBOOK — 常用命令

速查用。背景、设计理由、未验证项见 `HANDOFF.md`。

环境：4 x H200 (143 GB)，conda env `roleplay`，工作目录 `/workspace/SeaRolePlay`。

---

## 1. 启动训练

```bash
cd /workspace/SeaRolePlay && mkdir -p logs
PATH=/workspace/miniconda3/envs/roleplay/bin:$PATH \
PYTHON_BIN=/workspace/miniconda3/envs/roleplay/bin/python \
FORCE_TORCHRUN=1 \
nohup bash scripts/run_train.sh > logs/sft_rank32.log 2>&1 &
```

**`PATH=` 那行不能省。** LF 在 `launcher.py:115` 按裸名调 `torchrun`，不加会解析到
`/usr/local/bin/torchrun`（系统 torch），4 个 rank 全部 `ModuleNotFoundError:
No module named 'llamafactory'`。父 shell 的检查全过（它用的是 conda python），
只有子进程炸，错误看起来毫无来由。

**`> logs/... 2>&1` 也不能省。** 不重定向的话输出只进当前终端（`/dev/pts/N`），
磁盘上不留文件，事后无法排查。

临时覆盖参数用 OmegaConf 的 `key=value` 追加在后面（**不是** `--flag value`）：

```bash
... nohup bash scripts/run_train.sh per_device_train_batch_size=2 learning_rate=1e-4 > logs/x.log 2>&1 &
```

---

## 2. 停止训练

```bash
pkill -f llamafactory/launcher.py
```

确认干净退出（4 张卡都要回到 0 MiB）：

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

有残留（rank 卡在 NCCL 等待）时补一刀：

```bash
pkill -9 -f llamafactory
```

停之前先看有没有 checkpoint 会丢：`ls saves/qwen3.5-9b/<run>/`。

---

## 3. 看进度

```bash
tail -f logs/sft_rank32.log
```

只看步进（进度条是 `\r` 刷新的，`grep` 要用 `-a` 或 `-oE`）：

```bash
grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s/it\]" logs/sft_rank32.log | tail -3
```

显存和利用率：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

远端看：swanlab cloud，project `seaart-roleplay`。
https://swanlab.cn/@yyuchen/seaart-roleplay

---

## 4. 启动后必查的三件事

按顺序，出问题就停，别浪费几十小时。

**其一，linear attention 有没有走融合 kernel。**

```bash
grep -c "fast path is not available" logs/sft_rank32.log
```

必须是 `0`。非 0 说明 fla / causal-conv1d 没被认到，那 24 层在跑 PyTorch 参考实现，
慢 5 倍以上（实测 MFU 从 11% 到 60% 量级）。

**其二，tilelang 后端有没有被选中。**

```bash
grep -c "TileLang completes to compile" logs/sft_rank32.log   # 应为 4（每 rank 一次）
grep -E "rejected" logs/sft_rank32.log                        # 应为空
```

看到 `TileLang begins/completes to compile` 是正常的 JIT 预热，约 19 秒一次性成本。
出现 `rejected: <原因>` 说明准入条件没过，会退回去撞 triton 的正确性 guard 而报错。

**其三，单步耗时。**

进度条显示的是累计平均，前几步含 JIT 编译，会虚高。看第 10 步之后的稳定值，
或自己算增量（相邻两步的时间戳差）。

---

## 5. 环境依赖（加速相关）

装这四个之前，24/32 层 linear attention 跑的是 PyTorch fallback，MFU 只有 11%。

| 包 | 版本 | 作用 |
|---|---|---|
| `flash-linear-attention` | 0.5.2 | 24 层 linear attention 的融合 kernel（大头） |
| `fla-core` | 0.5.2 | 同上 |
| `causal-conv1d` | 1.7.0 | 那 24 层里的 conv1d 算子 |
| `tilelang` | 0.1.13 | 绕开 Hopper + triton 3.4 的 gated bwd 正确性 bug |

```bash
PATH=/workspace/miniconda3/envs/roleplay/bin:$PATH pip install flash-linear-attention tilelang
PATH=/workspace/miniconda3/envs/roleplay/bin:$PATH CUDA_HOME=/usr/local/cuda MAX_JOBS=32 \
  pip install causal-conv1d --no-build-isolation
```

**`--no-build-isolation` 不能省。** 不加的话 pip 会建个干净环境、从 PyPI 拉最新
torch（cu130）去编译，然后报 `The detected CUDA version (12.8) mismatches ... (13.0)`。
本机只有一个 nvcc（12.8），torch 也是 cu128，**本来就是匹配的** —— 那个报错是
build isolation 造出来的假象，不是环境坏了。

验证（import 在网络挂载上要十几分钟，急的话直接看训练日志的第 4 节）：

```bash
/workspace/miniconda3/envs/roleplay/bin/python -c "
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
print('chunk_gated_delta_rule :', 'OK' if m.chunk_gated_delta_rule else 'None')
print('causal_conv1d_fn       :', 'OK' if m.causal_conv1d_fn else 'None')"
```

`is_fast_path_available` 为 False **可以无视** —— 它（`modeling_qwen3_5.py:205`）只
决定打不打 warning，不参与实际分派。真正的分派是逐函数独立的（第 407、465 行），
所以装一半也能生效一半。

### 为什么是 tilelang 而不是升 triton

fla 在 `ops/common/chunk_o.py:704` 主动拦截：

```
RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect
results for gated chunk_bwd_dqkwg (see #640).
```

这是**正确性**拦截，不是性能问题 —— 不拦的话那 24 层梯度是错的，不报错不告警，
loss 曲线照样好看，训完模型是废的。torch 2.8.0 把 `triton==3.4.0` 硬钉死了，
硬升风险大；tilelang 是 fla 官方给的解法（错误信息里就写着），装完不动 torch/triton。

tilelang 后端在 Hopper + triton>=3.4.0 时**默认启用**，无需设环境变量
（`ops/common/backends/tilelang/__init__.py`）。强制开关：`FLA_TILELANG=1` / `=0`。

---

## 6. 陷阱

### tokenized_path 会屏蔽后续数据参数改动

`configs/*.yaml` 里的 `tokenized_path` 一旦指向已存在的目录，LF 会 `load_from_disk`
后**直接 return**（`data/loader.py:287-295`），日志只留一句
`Loading dataset from disk will ignore other data arguments`。

此后 `cutoff_len` / `packing` / `template` / 数据文件的改动**全部失效，不报错不告警**，
只是静默拿旧数据训。所以路径名里嵌了决定性参数
（`seaart_c32768_pack_qwen3_5_nothink`）。改这些参数时必须换路径名，或：

```bash
rm -rf data/tokenized/<旧目录>
```

首次 tokenize 很重（74k 条 x 32k），产物约 20 GB；之后重启直接读 Arrow。
`data/tokenized/` 已在 `.gitignore`。

### 会从 checkpoint 自动续训

`overwrite_output_dir: false` 时 LF 自动从最后的 checkpoint 恢复（`parser.py:491`）。
**改了超参再重启，会拿新超参接着旧优化器状态跑**，而不是重头开始。

调参时要么换 `output_dir`，要么清空目录。

### swanlab resume mode

`SWANLAB_RUN_ID` 从 `output_dir` 派生。同名 run 重启会进 resume mode，日志显示
`Hardware information collection, monitor, and terminal proxy have been disabled`。
指标照常记，但硬件监控图没有。想要干净的一条曲线就换 `output_dir`。

### 在网页上删掉 swanlab 实验 → 下次启动直接崩

删除不可逆，那个 run ID 就废了。而 `run_train.sh` 每次都从 `output_dir` 派生出
同一个 ID，于是带着已删除的 ID 去请求，服务端拒绝：

```
RuntimeError: Failed to start run: API Request Failed:
[Disabled_Resource] 实验已被删除
```

**这个错误藏在 launcher 的 `exitcode 1` / `ChildFailedError` 后面**，尾部只看到
`CalledProcessError: Command '['torchrun', ...]' returned non-zero exit status 1`，
第一眼完全看不出跟 swanlab 有关。往上翻日志找 `[rank0]` 开头的行才是真错误。

挂在 `on_train_begin` 回调里，一步都没跑，不会写坏 checkpoint。修法是换
`output_dir`（得到新 ID），或 `SWANLAB=0 bash scripts/run_train.sh` 先不上报。

### rank 级错误都在日志更靠上的位置

torchrun 尾部的 `Root Cause ... exitcode: 1` 只是"子进程死了"的结果，不是原因。
真正的 traceback 要往上找：

```bash
grep -nE "^\[rank0\]" logs/sft_rank32.log | grep -viE "use_return_dict" | tail -25
```

### flash_attn: fa2 未安装时静默回退

flash-attn 没装时不报错，`model/model_utils/attention.py:80` 打条 warning 就回退到
sdpa。所以配置写着 fa2 不代表真在用。

**但这层不值得折腾**：Qwen3.5-9B 是混合架构，`full_attention_interval: 4`，
32 层里只有 8 层是 full attention，在 cutoff_len=32768 下仅占约 19% 的 FLOPs，
且 sdpa 本身已是 flash 算法。真正决定速度的是另外 24 层（见第 5 节）。

---

## 7. 参数速查（截至 2026-08-26 的 rank32 配置）

改参数直接编辑 `configs/qwen3_5_9b_lora_sft.yaml`，或用第 1 节的 `key=value` 覆盖。

| 项 | 值 | 备注 |
|---|---|---|
| `lora_rank` / `lora_alpha` | 32 / 64 | 比值须保持 2.0，见 yaml 注释 |
| 可训参数 | 86,556,672 | 约占 9B 的 0.96% |
| `cutoff_len` | 32768 | 实测 P50 10.9k、max 19.1k，无截断 |
| `per_device_train_batch_size` | 4 | bsz>7 会撞 conv1d 的 int32 索引上限 |
| `gradient_accumulation_steps` | 4 | |
| 等效 batch | **64** | = 4 x 4 x 4卡；改任一项都是一次独立调参 |
| `learning_rate` | 2.0e-4 | |
| `num_train_epochs` | 2.0 | |
| 总步数 | 920 | 29,397 packed 样本 / 64 x 2 epoch |
| `bf16` | true | H200 上换 fp16 无收益，还有动态范围风险 |
| `enable_liger_kernel` | true | fused CE，避免materialize 全量 logits |

**显存**：约 118 GB / 143 GB。大头是激活侧，不是参数侧 —— vocab 248,320 让
logits 张量比 9B 权重本身还大。LoRA 省的是参数侧（86.6M 可训参数的优化器状态
才约 1 GB），省不到激活侧。

---

## 8. 数据

```bash
# 重建 train/val（约 4.2 GB，会跑两个硬断言：监督对账 + session 泄漏检查）
/workspace/miniconda3/envs/roleplay/bin/python data_process/build_data.py --help

# ctx-mask 补丁状态（0 = 已打且字节一致）
/workspace/miniconda3/envs/roleplay/bin/python data_process/patch_lf.py --check

# loss mask token 级验证
/workspace/miniconda3/envs/roleplay/bin/python data_process/verify_loss_mask.py
```

补丁未打时 LF 会在 `format_tools` 里抛 `RuntimeError: Invalid JSON format in tool
description` —— 响亮失败，不会静默训错。`scripts/run_train.sh` 启动前会先行拦截。
