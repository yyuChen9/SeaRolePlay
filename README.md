# Character LLM 微调实验

这个目录是一个围绕 LLaMA-Factory 的最小、可复现学习项目。LLaMA-Factory 负责训练和推理，本项目负责依赖安装、数据格式转换、数据集注册和实验配置。

当前状态、环境版本、已知问题和下一步工作见 [`docs/HANDOFF.md`](docs/HANDOFF.md)。

## 安装

```bash
bash scripts/install_llamafactory.sh
bash /workspace/setup_huggingface.sh
source ~/.bashrc
```

`install_llamafactory.sh` 默认优先安装 `/workspace/LLaMA-Factory` 的源码 checkout；不存在时从 PyPI 安装。

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

脚本会下载固定文件、校验并转换为。两个 SFT 数据集中的 `human/gpt` 会统一成 `user/assistant`；DPO 数据中 `rejected` 为空的记录会跳过并报告，其余格式错误会终止流程。

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

默认模型配置为 `Qwen/Qwen3.5-0.8B`，训练配置位于 `configs/qwen3_5_0_8b_lora_sft.yaml`。需要切换实验时复制一份 YAML，并通过 `CONFIG_PATH=/path/to/config.yaml bash scripts/run_train.sh` 指定，避免 YAML 和命令行参数互相覆盖。

`Qwen3.5-0.8B` 的实际 Hugging Face 仓库名和模板需要以模型发布版本及本地 LLaMA-Factory 版本为准。若 `qwen3` 模板不存在，请按 `llamafactory-cli chat --help` 显示的模板名修改配置。
