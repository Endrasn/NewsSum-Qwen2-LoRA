# NewsSum-Qwen2-LoRA

LCSTS news summarization via LoRA-finetuned Qwen2-7B (5k subset) and TextRank, evaluated with ROUGE metrics on an NVIDIA RTX 4090.

## 项目简介 (Overview)

本项目基于 **LCSTS 新闻摘要数据集**（抽取 5,000 条高质量样本），分别实现了基于深度学习的 **Qwen2-7B LoRA 微调方案** 以及传统的 **TextRank 抽取式摘要算法**，并使用 **ROUGE (1/2/L)** 指标对两种方案的生成效果进行量化评估与对比。

## 特性 (Features)

* **模型微调**: 基于 `Qwen2-7B-Instruct` 基座模型，依托 **LLaMA-Factory** 微调框架实现 LoRA 轻量化高效微调。
* **硬件支持**: 针对单张 **NVIDIA RTX 4090 (24GB)** 显卡进行训练优化（支持 bfloat16 与 QLoRA/LoRA 混合精度训练）。
* **基线对比**: 集成 TextRank 算法作为无监督 Baseline。
* **自动评估**: 内置 ROUGE-1、ROUGE-2、ROUGE-L 评分系统，一键输出评价指标与效果对比。
* **数据集**: 使用 LCSTS (Large-scale Chinese Short Text Summarization) 过滤后的 5k 核心训练集。

## 硬件与环境配置 (Hardware & Environment)

* **GPU**: NVIDIA GeForce RTX 4090 (24GB)
* **主要依赖**:
  * CUDA 12.x / PyTorch 2.x
  * LLaMA-Factory
  * Transformers, PEFT, Accelrate
  * rouge-chinese (用于 ROUGE 指标评估)

## 声明与致谢 (Disclaimers & Acknowledgments)

### 1. 模型与工具链声明

* **基座模型**: 本项目微调所使用的基座模型为阿里巴巴团队开源的 **Qwen2-7B-Instruct**。模型权重的非商业与商业用途需严格遵守 [Qwen2 开源模型许可协议](https://github.com/QwenLM/Qwen2/blob/main/LICENSE)。特此对 Qwen 团队的开源工作表示感谢。
* **微调框架**: 本项目的训练流程基于 **LLaMA-Factory** 开源框架实现，特此对相关开发团队的卓越工作表示感谢。

### 2. 数据集 (LCSTS) 版权与使用限制

本项目训练与测试所使用的文本摘要数据源自哈尔滨工业大学（HIT）发布的 **LCSTS (Large-scale Chinese Short Text Summarization)** 大规模中文短文本摘要数据集。

* **版权归属**: 数据集版权完全归原作者及发布机构所有。
* **使用限制**: 根据 LCSTS 数据集官方许可，该数据**仅供学术研究使用 (Academic Research Only)，严格禁止任何形式的商业用途**。本项目仓库内包含的 5,000 条数据子集仅用于微调算法的复现与学术评估，任何第三方因违反此限制而导致的合规风险由使用者自行承担。

### 3. 免责声明

本项目所提供的代码、微调权重及对比实验结果仅用于个人学习、学术交流与技术分享。作者不对因使用本项目代码、模型或数据所导致的任何直接或间接损失承担责任。
