# Character LLM 项目交接文档

更新时间：2026-08-14

## 1. 项目目标

基于 LLaMA-Factory，在 `Qwen/Qwen3.5-0.8B` 上进行角色扮演（Character/Roleplay）数据的 LoRA SFT，后续再使用偏好对进行 DPO。项目根目录为：

```text
/workspace/Character
```

当前阶段是“数据准备完成，SFT 启动参数已修复，等待验证首个训练 step”。

## 2. 当前环境

本次交接时检测到：

```text
Python            3.14.6
LLaMA-Factory     0.9.5
Transformers      5.6.0
PyTorch           2.13.0
huggingface_hub   1.27.0
```

Python 环境来自 `/workspace/miniconda3`。Hugging Face 配置脚本位于 `/workspace/setup_huggingface.sh`，默认缓存目录为 `/workspace/.cache/huggingface`。

注意：Python 3.14 较新。如果后续遇到 CUDA 扩展、bitsandbytes、flash-attn 或第三方包兼容问题，优先考虑创建 Python 3.11/3.12 的独立训练环境，不要直接破坏当前 base 环境。

## 3. 数据来源与版本

脚本固定了以下 Hugging Face commit，保证迁移后的数据一致：

| 用途 | 数据集 | Commit |
| --- | --- | --- |
| SFT | `rx1lora/StoryPlay_Sonnet3.5-Charcard-Roleplay` | `8ed694f86627d82f669337308ecb5f0bf751e097` |
| SFT | `Gryphe/Sonnet3.5-Charcard-Roleplay` | `0b47a7695233107ad40da25db044610ddb378830` |
| DPO | `rx1lora/StoryPlay_DPO_Pairs-Roleplay-NSFW` | `6b029adf3801fedd12ed7ff657a388914aad5dc2` |

数据包含成人/NSFW 内容。使用、分发和发布模型前，必须重新核对各仓库许可证、平台政策和目标部署环境限制。

## 4. 数据准备状态

运行命令：

```bash
cd /workspace/Character
python scripts/prepare_hf_character_data.py --output-dir data
```

当前结果：

```text
storyplay_sft             9736 条
gryphe_sft                9736 条
character_sft_combined    9736 条（19472 条精确去重后）
storyplay_dpo             3425 条有效样本
```

两个 SFT 来源在当前 commit 下转换后完全重复，因此合并数据仍为 9736 条。不要通过简单拼接把重复样本训练两遍，除非明确希望改变其采样权重。

DPO 原始文件有两个已知质量问题：

1. 第 239 个物理行包含两个直接拼接的 JSON 对象。读取器通过 `JSONDecoder.raw_decode()` 正确拆分。
2. 3428 条解析结果中有 3 条 `rejected` 为空。脚本会报告并跳过，最终保留 3425 条。

生成文件：

```text
data/character_sft_combined.json
data/storyplay_sonnet35_charcard.json
data/gryphe_sonnet35_charcard.json
data/storyplay_dpo_roleplay_nsfw.json
data/dataset_info.json
```

原始文件缓存在 `data/raw/`。默认会复用已有文件；强制重新下载使用：

```bash
python scripts/prepare_hf_character_data.py --output-dir data --force-download
```

## 5. LLaMA-Factory 数据注册

`data/dataset_info.json` 注册了：

| 名称 | 文件 | 用途 |
| --- | --- | --- |
| `character` | `character_sft_combined.json` | 默认 SFT 别名 |
| `character_sft` | `character_sft_combined.json` | SFT |
| `storyplay_sft` | `storyplay_sonnet35_charcard.json` | 单独使用 StoryPlay SFT |
| `gryphe_sft` | `gryphe_sonnet35_charcard.json` | 单独使用 Gryphe SFT |
| `storyplay_dpo` | `storyplay_dpo_roleplay_nsfw.json` | DPO ranking 数据 |

当前 SFT 配置使用 `dataset: character`。

## 6. SFT 配置

配置文件：`configs/qwen3_5_0_8b_lora_sft.yaml`

关键参数：

```yaml
model_name_or_path: Qwen/Qwen3.5-0.8B
template: qwen3
stage: sft
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all
dataset: character
cutoff_len: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 2.0
warmup_steps: 100
bf16: true
```

输出目录：

```text
saves/qwen3.5-0.8b/character-lora
```

## 7. 启动训练

```bash
cd /workspace/Character
bash scripts/run_train.sh
```

训练脚本只把 YAML 路径传给 LLaMA-Factory。不要恢复以下旧写法：

```bash
llamafactory-cli train config.yaml --model_name_or_path Qwen/Qwen3.5-0.8B
```

在当前 `LLaMA-Factory 0.9.5` 中，这会让 OmegaConf/HfArgumentParser 把 `--model_name_or_path` 视为未使用键并报错。模型名应放在 YAML 中。

切换配置文件使用：

```bash
CONFIG_PATH=/path/to/another.yaml bash scripts/run_train.sh
```

`warmup_ratio` 已因 Transformers 5.6 弃用警告改为 `warmup_steps: 100`。

## 8. 已验证与未验证

已验证：

- 三个 Hugging Face 源文件可以下载。
- 两个 SFT 数据可以转换为 ShareGPT `messages` 格式。
- DPO 拼接 JSON 和空 rejected 可以正确处理。
- 数据集注册 JSON 有效。
- Shell/Python 语法检查通过。
- 修正后的训练 YAML 可由本地 YAML 解析器读取。
- 训练启动脚本不再追加不兼容的 CLI 参数。

尚未验证：

- `Qwen/Qwen3.5-0.8B` 是否是实际可访问的最终模型仓库名。
- 当前 LLaMA-Factory 是否为该模型提供正确的 `qwen3` 模板。
- 模型下载、tokenization、首个 forward/backward step。
- 当前 GPU 是否支持 `bf16`。
- 完整两轮训练的显存、速度和磁盘需求。
- `scripts/run_chat.sh` 在当前 LLaMA-Factory 版本上的实际推理行为。
- DPO 训练配置尚未创建。

## 9. 下一步建议

1. 运行 `bash scripts/run_train.sh`，只观察到模型加载、数据预处理和第一个训练 step。
2. 如果显存不足，优先降低 `cutoff_len`，再考虑 QLoRA、gradient checkpointing 或更小 batch。
3. 检查训练日志中的有效 token 比例和截断比例。角色对话很长，`cutoff_len: 2048` 可能截掉大量后续轮次。
4. 先用少量样本做 smoke test，再进行完整训练；建议新增独立 smoke-test YAML，不要覆盖正式配置。
5. SFT 成功后再创建 DPO 配置，并以 SFT adapter/合并模型作为 DPO 起点。
6. 建立固定评测集，比较 base、SFT 和 DPO 的角色一致性、拒答行为、重复输出和越界内容。

## 10. 关键文件

```text
README.md
docs/HANDOFF.md
configs/qwen3_5_0_8b_lora_sft.yaml
scripts/install_llamafactory.sh
scripts/prepare_hf_character_data.py
scripts/prepare_character_data.py
scripts/run_train.sh
scripts/run_chat.sh
data/dataset_info.json
```

训练数据、原始下载和 `saves/` 已由 `.gitignore` 排除，不应提交到代码仓库。
