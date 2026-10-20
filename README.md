# NewsSum-Qwen2-LoRA

LCSTS news summarization via LoRA-finetuned Qwen2.5-7B-Instruct (5k subset) and TextRank, evaluated with ROUGE metrics.

## 项目简介 (Overview)

本项目基于 **LCSTS 新闻摘要数据集**（抽取 5,000 条高质量样本），分别实现了基于深度学习的 **Qwen2.5-7B-Instruct LoRA 微调方案** 以及传统的 **TextRank 抽取式摘要算法**，并使用 **ROUGE (1/2/L)** 指标对两种方案的生成效果进行量化评估与对比。

## 特性 (Features)

* **模型微调**: 基于 `Qwen2.5-7B-Instruct` 基座模型，依托 **LLaMA-Factory** 微调框架实现 LoRA 轻量化高效微调。
* **硬件支持**: 原始训练在单张 **NVIDIA RTX 4090 (24GB)** 显卡上完成，推理可使用 4-bit 量化降低显存占用。
* **基线对比**: 集成 TextRank 算法作为无监督 Baseline。
* **自动评估**: 内置 ROUGE-1、ROUGE-2、ROUGE-L 评分系统，一键输出评价指标与效果对比。
* **数据集**: 使用 LCSTS (Large-scale Chinese Short Text Summarization) 过滤后的 5k 核心训练集。

## LoRA 权重

最终可推理的 LoRA adapter 已整理到：

```text
adapters/qwen2.5-7b-lcsts-lora/
```

该目录包含 `adapter_model.safetensors`、`adapter_config.json`、tokenizer、chat template、训练参数和训练结果。`adapter_model.safetensors` 使用 Git LFS 存储；克隆仓库后如未自动下载大文件，请运行：

```bash
git lfs pull
```

注意：仓库只包含 LoRA adapter，不包含 `Qwen/Qwen2.5-7B-Instruct` 基座模型。推理时需要先下载基座模型，再加载本仓库 adapter。

## Web 界面

页面支持直接输入或导入新闻稿，并一次完成：

1. TextRank 抽取式摘要
2. Qwen2.5 LoRA 生成式摘要
3. 两种摘要相对人工参考摘要的 ROUGE-1/2/L 评分

先启动提供 OpenAI 兼容 `/v1/chat/completions` 接口的 Qwen2.5 LoRA 推理服务，默认地址为：

```text
http://127.0.0.1:8000/v1/chat/completions
```

然后启动本项目网页服务：

```bash
pip install -r requirements.txt
python textrank_summarizer.py --serve
```

浏览器访问 `http://127.0.0.1:8080`。模型地址、模型名称和 API Key 可在页面的“模型设置”中修改，也可通过 `QWEN_API_URL`、`QWEN_MODEL`、`QWEN_API_KEY` 环境变量配置默认值。

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 硬件与环境配置 (Hardware & Environment)

* **GPU**: NVIDIA GeForce RTX 4090 (24GB) 用于训练；8GB 显存设备建议使用 4-bit 量化推理。
* **主要依赖**:
  * CUDA 12.x / PyTorch 2.x
  * LLaMA-Factory
  * Transformers, PEFT, Accelerate
  * jieba（TextRank 中文分词，可选）

## 声明与致谢 (Disclaimers & Acknowledgments)

### 1. 模型与工具链声明

* **基座模型**: 本项目微调所使用的基座模型为阿里巴巴团队开源的 **Qwen2.5-7B-Instruct**。模型权重的非商业与商业用途需严格遵守 [Qwen 开源模型许可协议](https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE)。特此对 Qwen 团队的开源工作表示感谢。
* **微调框架**: 本项目的训练流程基于 **LLaMA-Factory** 开源框架实现，特此对相关开发团队的卓越工作表示感谢。

### 2. 数据集 (LCSTS) 版权与使用限制

本项目训练与测试所使用的文本摘要数据源自哈尔滨工业大学（HIT）发布的 **LCSTS (Large-scale Chinese Short Text Summarization)** 大规模中文短文本摘要数据集。

* **版权归属**: 数据集版权完全归原作者及发布机构所有。
* **使用限制**: 根据 LCSTS 数据集官方许可，该数据**仅供学术研究使用 (Academic Research Only)，严格禁止任何形式的商业用途**。本项目仓库内包含的 5,000 条数据子集仅用于微调算法的复现与学术评估，任何第三方因违反此限制而导致的合规风险由使用者自行承担。

### 3. 免责声明

本项目所提供的代码、微调权重及对比实验结果仅用于个人学习、学术交流与技术分享。作者不对因使用本项目代码、模型或数据所导致的任何直接或间接损失承担责任。
