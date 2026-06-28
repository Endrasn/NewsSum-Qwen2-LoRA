# 中文新闻文本自动摘要实验台

本项目用于对比中文新闻文本的两种自动摘要方法：

- TextRank：抽取式摘要，直接从原文中选取关键句。
- Qwen2.5-7B-Instruct + LoRA：生成式摘要，加载本地基础模型和本地 LoRA 权重。

前端采用三页式结构：

1. 摘要对比：输入新闻正文和总结示例，展示 TextRank 与模型摘要结果。
2. ROUGE 图谱：在同一张三角图中展示 ROUGE-1、ROUGE-2、ROUGE-L，使用不同半透明颜色对比两种方法。
3. 参数设置：调节 TextRank 句数、选句比例，以及 Qwen 推理参数。

## 一键启动

Windows 下双击：

```text
start.bat
```

脚本会自动：

1. 创建 `.venv` 虚拟环境。
2. 安装 `requirements.txt` 依赖。
3. 打开浏览器访问 `http://127.0.0.1:7860`。
4. 启动 Flask 后端服务。

如果依赖安装失败，通常是网络或 PyTorch 版本问题。可以先按 PyTorch 官网命令安装适配 CUDA 的 `torch`，再运行：

```bash
pip install -r requirements.txt
python app.py
```

## 大模型放置位置

当前目录已有：

```text
adapter_model.safetensors
adapter_config.json
```

它们只是 LoRA 权重和 LoRA 配置，不包含 Qwen2.5-7B-Instruct 基础模型本体。请将基础模型下载到：

```text
models/Qwen2.5-7B-Instruct/
```

目录中应包含类似文件：

```text
config.json
tokenizer.json
tokenizer_config.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

如果你希望整理 LoRA，也可以放到：

```text
models/qwen2.5-7b-news-lora/
├── adapter_model.safetensors
└── adapter_config.json
```

然后在第三页把 LoRA 目录改成：

```text
models/qwen2.5-7b-news-lora
```

## 缺少 adapter_config.json 时

如果只有 `adapter_model.safetensors`，可以生成一个基础 PEFT 配置：

```bash
python scripts/prepare_lora_adapter.py --adapter-dir . --base-model models/Qwen2.5-7B-Instruct
```

注意：最准确的做法是使用 LoRA 发布方提供的 `adapter_config.json`。当前脚本根据权重形状生成基础配置，默认 `r=16`、`lora_alpha=32`，目标模块为 Qwen 常见投影层。

## 单独运行脚本

TextRank：

```bash
python scripts/textrank_summarizer.py --input news.txt --max-sentences 4
```

Qwen LoRA 推理：

```bash
python scripts/qwen_lora_infer.py --text "这里填写新闻正文" --model-path models/Qwen2.5-7B-Instruct --adapter-path .
```

ROUGE 评分：

```bash
python scripts/rouge_score.py --reference "人工参考摘要" --hypothesis "模型生成摘要"
```

## 项目结构

```text
.
├── app.py
├── start.bat
├── adapter_model.safetensors
├── adapter_config.json
├── requirements.txt
├── scripts/
│   ├── textrank_summarizer.py
│   ├── rouge_score.py
│   ├── qwen_lora_infer.py
│   └── prepare_lora_adapter.py
└── static/
    ├── index.html
    ├── styles.css
    └── app.js
```

## 鸣谢

感谢 Endrasn 对本项目的支持与帮助。
