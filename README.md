# Character LLM 微调实验

这个目录是一个围绕 LLaMA-Factory 的最小、可复现学习项目。LLaMA-Factory 负责训练和推理，本项目负责依赖安装、数据格式转换、数据集注册和实验配置。

当前模型为 `Qwen/Qwen3.5-9B`，训练方式为 LoRA SFT，后续计划使用偏好对做 DPO。

当前状态、环境版本、已知问题和下一步工作见 [`docs/HANDOFF.md`](docs/HANDOFF.md)。

## 环境

已验证环境：Python 3.12 + torch 2.8.0+cu128，GPU 为 NVIDIA H200（sm90）。

```bash
conda create -n roleplay python=3.12 -y
conda activate roleplay
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

bash /workspace/setup_huggingface.sh
source ~/.bashrc
```

版本约束的具体原因写在 [`requirements.txt`](requirements.txt) 的注释里，简要说：

- `torch` / `torchvision` / `torchaudio` 三者版本必须严格匹配，且都要 pin。torchaudio 是 LLaMA-Factory 的硬依赖，其编译扩展绑定特定 torch 版本，不匹配会在 import 阶段报 `undefined symbol: torch_library_impl`。
- `torch` 必须避开 `2.9.x`。LLaMA-Factory 0.9.5 会对含 `Conv3d` 的模型硬拦截该区间（见 `llamafactory/model/loader.py`）。Qwen3.5 是多模态模型，视觉塔含 `Conv3d`，即使只训练文本也会触发。
- 选择降级到 2.8.0 而非升级到 2.10+，是为了保留 flash-attn 预编译 wheel（cp312 只覆盖到 `torch2.8`）。flash-attn 是可选项，不装也能训练。

需要字节级复现已验证环境时使用 [`requirements-lock.txt`](requirements-lock.txt)。

注意：`scripts/install_llamafactory.sh` 使用 `pip install -U llamafactory`，会连带升级 torch 并破坏上述 pin。已按 `requirements.txt` 装好环境后不要再运行它；只在需要安装 `/workspace/LLaMA-Factory` 源码 checkout 时使用，之后重新执行一次 `pip install -r requirements.txt` 校正版本。

验证安装：

```bash
python -c "import torch, torchaudio, torchvision; print(torch.__version__, torch.cuda.is_available())"
llamafactory-cli version
```

预期输出 `2.8.0+cu128 True` 和 `LLaMA Factory 0.9.5`。

## 准备 Character 数据

内部统一使用 ShareGPT 格式：

```json
[
  {
    "messages": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好，很高兴认识你。"}
    ]
  }
]
```

转换常见的 Character 数据：

```bash
python scripts/prepare_character_data.py \
  --input /path/to/character.json \
  --output data/character_sft_combined.json
```

脚本支持 ShareGPT、Alpaca（`instruction`/`input`/`output`）以及 `conversations` 字段，并会拒绝空消息、未知角色和没有 assistant 回复的样本。

准备本项目使用的三个 Hugging Face 数据集（两个 SFT，一个 DPO）：

```bash
python scripts/prepare_hf_character_data.py --output-dir data
```

脚本会下载固定文件、校验并转换。两个 SFT 数据集中的 `human/gpt` 会统一成 `user/assistant`；DPO 数据中 `rejected` 为空的记录会跳过并报告，其余格式错误会终止流程。

- `storyplay_sonnet35_charcard.json`：`rx1lora/StoryPlay_Sonnet3.5-Charcard-Roleplay`
- `gryphe_sonnet35_charcard.json`：`Gryphe/Sonnet3.5-Charcard-Roleplay`
- `character_sft_combined.json`：上述两个 SFT 数据去重合并
- `storyplay_dpo_roleplay_nsfw.json`：`prompt/chosen/rejected` DPO 数据

原始下载文件默认放在 `data/raw/`。脚本默认使用三个仓库各自记录的 commit，保证以后迁移环境时输入版本一致；已有原始文件会自动复用，只有传入 `--force-download` 才会重新下载。需要测试其他 revision 时可传入 `--revision`（该值会覆盖三个仓库的默认 revision）。

三个数据集的许可证和内容限制请以 Hugging Face 仓库页面为准，训练前应确认符合你的使用场景。

## LoRA 训练与推理

```bash
bash scripts/run_train.sh
bash scripts/run_chat.sh base
bash scripts/run_chat.sh lora
```

训练配置位于 [`configs/qwen3_5_9b_lora_sft.yaml`](configs/qwen3_5_9b_lora_sft.yaml)，adapter 输出到 `saves/qwen3.5-9b/seaart-sft-lora`。

模板使用 `qwen3_5_nothink`。LLaMA-Factory 0.9.5 中 `qwen3` 和 `qwen3_5` 是两个不同模板，Qwen3.5 必须用后者。`Qwen/Qwen3.5-9B` 在 LLaMA-Factory 中注册为 Thinking 版本，`_nothink` 变体用于抑制思维链输出；如需保留思维链，改为 `qwen3_5`。

切换实验时复制一份 YAML，并通过环境变量指定，避免 YAML 和命令行参数互相覆盖：

```bash
CONFIG_PATH=/path/to/config.yaml bash scripts/run_train.sh
```

不要用 `--flag value` 形式覆盖 YAML。在 LLaMA-Factory 中 `llamafactory-cli train config.yaml --max_samples 20` 会让 HfArgumentParser 把附加参数视为未使用键并报错（`Some keys are not used by the HfArgumentParser`）。若确实要临时覆盖，用 OmegaConf 的 `key=value` 形式（`hparams/parser.py:90-93` 把它们 merge 进 YAML，命令行优先）：

```bash
bash scripts/run_train.sh max_steps=3 output_dir=/tmp/smoke
```

正式训练前建议先用少量样本做 smoke test，覆盖 `max_samples`、`max_steps` 和 `output_dir`，不要覆盖正式配置的输出目录。

### 多卡训练

**本机目前只有 1 张 H200**（`nvidia-smi -L` / `torch.cuda.device_count()` 均确认），下面是换到多卡机器时的做法，未在本机验证过。

启动方式不需要改脚本。LLaMA-Factory 自己判断卡数并拉起 torchrun（`launcher.py:61`），`run_train.sh` 里没有任何单卡假设：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 \
  bash scripts/run_train.sh gradient_accumulation_steps=4
```

**`gradient_accumulation_steps=4` 不是可选项，是必须的。** 有效 batch 的公式是
`micro_batch × grad_accum × world_size`（`transformers/trainer.py:2360`），当前配置
`1 × 16` 在 4 卡下会变成 **64**，优化器步数从 9144 掉到 2286 —— 等于换了一组超参在训练，
而 `learning_rate: 1.0e-4` 和 `warmup_steps: 100` 都是按 batch=16 定的。改成
`1 × 4 × 4 = 16` 才与单卡等价，结果才可比。

若确实想用 batch=64 换吞吐，那是一次独立调参：同步调 LR 与 warmup，并换一个
`output_dir`，别和单卡结果混在同一目录（`SWANLAB_RUN_ID` 由 `output_dir` 推导，
换目录会自动开新曲线，不会覆盖旧的）。

另外首次多卡启动前，先用单卡把 tokenize 缓存打出来（`overwrite_cache: false`，但缓存
目前是空的，74543 条 × 32k 的 tokenize 很重）：

```bash
bash scripts/run_train.sh max_steps=2 output_dir=/tmp/warm_cache
```

swanlab 的 callback 有 `is_world_process_zero` 保护，只有 rank 0 上报，不会开出多条曲线。

### 训练日志（swanlab）

`run_train.sh` 默认把训练指标上报到 swanlab，project 为 `seaart-roleplay`（改用
`SL_PROJECT=`，不是 `SWANLAB_PROJECT=`，原因见下）。

```bash
# 前置：在 .env 里填 SWANLAB_API_KEY，然后登录一次（凭证存 ~/.swanlab/.netrc）
bash scripts/swanlab_login.sh
bash scripts/swanlab_login.sh --check                # 只校验不落盘

bash scripts/run_train.sh                            # 上报到 swanlab 云端
SWANLAB_MODE=local bash scripts/run_train.sh         # 只写本地，swanlab watch 查看
SWANLAB=0 bash scripts/run_train.sh                  # 完全不上报
```

四个设计点：

- **走 LLaMA-Factory 自己的 `use_swanlab`，不走 `report_to`。** LF 内建 SwanLab 支持
  （`hparams/finetuning_args.py:405`）并自行挂载 callback（`train/tuner.py:78`）。
  即使写 `report_to=swanlab`，`hparams/parser.py:465` 也会把它从 `report_to` 里摘掉。
- **run id 由 `output_dir` 推导并固定**（`saves/qwen3.5-9b/seaart-sft-lora` →
  实验名 `qwen3.5-9b-seaart-sft-lora`）。SFT 全量耗时很长，必然会中断重跑；LF 会自动
  从最后一个 checkpoint 续训（`hparams/parser.py:491`），但 swanlab 默认每次重启开一条
  新实验，loss 曲线会碎成互不相连的几段。固定 `SWANLAB_RUN_ID` + `SWANLAB_RESUME=allow`
  让每次续训落回同一条曲线。
- **`swanlab_login.sh` 联网校验，`run_train.sh` 只做本地预检。** 两者分工不同：
  后者只看 `Settings().api_key` 存不存在，抓不出"格式合法但 key 是错的"；前者会
  真的打一次服务端（`relogin=True` 强制走网络，否则见到已有凭证会直接短路）。
  正式开跑前先跑一次 login 脚本，别把 401 留到训练里。
- **未登录时在加载模型之前就拒绝启动**。`swanlab.init()` 发生在 trainer 的
  `on_train_begin`，即 9B 模型加载完之后。不预检的话要等十几分钟才会看到
  `RuntimeError: Failed to initialize SwanLab in online mode: no TTY is available`。
- **项目名用 `SL_PROJECT`，不占用 `SWANLAB_` 前缀。** swanlab 0.9.7 用
  pydantic-settings 以 `env_prefix="SWANLAB_"` 读环境变量，其中 `project` 是嵌套
  子模型，`SWANLAB_PROJECT=seaart-roleplay` 这种字符串会让它在 `swanlab.init()` 里抛
  `SettingsError: error parsing value for field "project"`，训练直接中断（旧的标量
  写法是 `SWANLAB_PROJ_NAME`）。而且 swanlab 会自行读取当前目录的 `.env`，所以这个名字
  写进 `.env` 一样会踩雷 —— 项目名经 `swanlab_project=` 命令行参数传给 LF 即可。
  `run_train.sh` 启动前会实例化一次 `Settings()` 预检，配错时当场报错。

API key 只经环境变量传递，不作为命令行参数 —— LF 支持 `swanlab_api_key=...`，但那会
让密钥出现在进程列表里。

YAML 里的 `report_to: none` 是刻意保留的：直接 `llamafactory-cli train <config>` 时
不会因为缺凭证而失败，上报由 `run_train.sh` 在命令行追加 `use_swanlab=true` 开启。

## DPO 训练

DPO 接在 SFT 之上，需先完成 SFT 训练并生成 `saves/qwen3.5-9b/character-lora`。

```bash
# 先跑 smoke test 验证启动路径（约 15 秒）
CONFIG_PATH=configs/qwen3_5_9b_lora_dpo_smoke.yaml bash scripts/run_train.sh
rm -rf saves/_smoke_dpo

# 正式训练
CONFIG_PATH=configs/qwen3_5_9b_lora_dpo.yaml bash scripts/run_train.sh
```

配置为 [`configs/qwen3_5_9b_lora_dpo.yaml`](configs/qwen3_5_9b_lora_dpo.yaml)，adapter 输出到 `saves/qwen3.5-9b/character-dpo`。

几个关键设计：

- `create_new_adapter: true` 会先把 SFT adapter 合并进 base 权重，再随机初始化一个新 adapter 用于 DPO。LoRA 场景下 LLaMA-Factory 不单独加载参考模型，而是用「禁用 adapter 后的模型」作为隐式参考 —— 合并之后它恰好等于 SFT 后的模型，符合 DPO 语义，因此无需指定 `ref_model`。
- 学习率为 `5.0e-6`，比 SFT 的 `1.0e-4` 低约 20 倍。沿用 SFT 量级的学习率会迅速破坏已对齐的模型，是 DPO 最常见的失败原因。
- `cutoff_len: 1024`。实测 DPO 数据 `prompt + max(chosen, rejected)` 的 p99 约 683 tokens、最长 728，1024 不会产生截断。
- `per_device_train_batch_size: 4` 配 `gradient_accumulation_steps: 4`。DPO 每步需前向 chosen 和 rejected 两条序列，显存约为同 batch SFT 的两倍，故 per-device 减半、有效 batch 仍与 SFT 保持 16。
- 可调超参：`pref_beta`（默认 0.1，越小越允许偏离参考模型）、`pref_loss`（可选 `sigmoid`/`hinge`/`ipo`/`kto_pair`/`orpo`/`simpo`）、`pref_ftx`（叠加 SFT 正则项，角色一致性退化时可设为 0.1 左右）。

训练日志中应关注 `rewards/accuracies`（应逐步上升）和 `rewards/margins`（chosen 与 rejected 的奖励差，应逐步扩大）。第一步这些值为 0 是正常的 —— 此时策略模型与参考模型完全相同。

已知数据局限：SFT 数据是带角色卡 system prompt 的多轮 ShareGPT 对话，而 DPO 数据是不含 system、单轮的 `prompt`/`chosen`/`rejected` 纯字符串。两个阶段的 prompt 分布不一致，DPO 可能冲淡 SFT 学到的角色卡遵循能力。评测时应重点对比 DPO 前后的角色一致性；若出现退化，可考虑调高 `pref_ftx`，或在转换脚本中把 DPO prompt 包装成与 SFT 一致的对话格式。

