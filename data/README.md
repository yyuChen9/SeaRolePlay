# Character 数据

将原始开源 Character 数据放在任意位置，然后转换为项目标准文件：

```bash
python scripts/prepare_character_data.py \
  --input /path/to/raw_character.json \
  --output data/character_sft_combined.json
```

`character_sft_combined.json` 默认被 `.gitignore` 忽略，避免训练数据和个人修改被误提交。
