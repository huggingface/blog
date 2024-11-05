---
title: "SynthID Text：在 AI 生成文本中应用不可见水印的新技术"
thumbnail: /blog/assets/synthid-text/thumbnail.png
authors:
  - user: sumedhghaisas
    org: Google DeepMind
    guest: true
  - user: sdathath
    org: Google DeepMind
    guest: true
  - user: RyanMullins
    org: Google DeepMind
    guest: true
  - user: joaogante
  - user: marcsun13
  - user: RaushanTurganbay
translators:
- user: chenglu
---

# SynthID Text：在 AI 生成文本中应用不可见水印的新技术

你是否难以分辨一段文本是由人类撰写的，还是 AI 生成的？识别 AI 生成内容对于提升信息可信度、解决归因错误以及抑制错误信息至关重要。

今天，[Google DeepMind](https://deepmind.google/) 和 Hugging Face 共同宣布，在 [Transformers v4.46.0](https://huggingface.co/docs/transformers/v4.46.0) 版本中，我们正式推出了 [SynthID Text](https://deepmind.google/technologies/synthid/) 技术。这项技术能够通过使用 [logits 处理器](https://huggingface.co/docs/transformers/v4.46.0/en/internal/generation_utils#transformers.SynthIDTextWatermarkLogitsProcessor) 为生成任务添加水印，并利用 [分类器](https://huggingface.co/docs/transformers/v4.46.0/en/internal/generation_utils#transformers.SynthIDTextWatermarkDetector) 检测这些水印。

详细的技术实现请参考发表在《自然》（_Nature_）上的 [SynthID Text 论文](https://www.nature.com/articles/s41586-024-08025-4)，以及 Google 的 [负责任生成式 AI 工具包](https://ai.google.dev/responsible/docs/safeguards/synthid)，了解如何将 SynthID Text 应用到你的产品中。

### 工作原理

SynthID Text 的核心目标是为 AI 生成的文本嵌入水印，从而让你能判断文本是否由你的大语言模型 (LLM) 生成，同时不影响模型的功能或生成质量。Google DeepMind 开发了一种水印技术，使用一个伪随机函数（g 函数）增强任何 LLM 的生成过程。这个水印对人类来说不可见，但能被训练好的模型检测。这项功能被实现为一个 [生成工具](https://huggingface.co/docs/transformers/v4.46.0/en/internal/generation_utils#transformers.SynthIDTextWatermarkLogitsProcessor)，可使用 `model.generate()` API 与任何 LLM 兼容，无需对模型做修改，并提供一个完整的 [端到端示例](https://github.com/huggingface/transformers/tree/v4.46.0/examples/research_projects/synthid_text/detector_training.py)，展示如何训练检测器来识别水印文本。具体细节可参考 [研究论文](https://www.nature.com/articles/s41586-024-08025-4)。

### 配置水印

水印通过一个 [数据类](https://huggingface.co/docs/transformers/v4.46.0/en/internal/generation_utils#transformers.SynthIDTextWatermarkingConfig) 进行配置，这个类参数化 g 函数，并定义它在抽样过程中的应用方式。每个模型都应有其专属的水印配置，并且必须**安全私密地存储**，否则他人可能会复制你的水印。

在水印配置中，必须定义两个关键参数：

- `keys` 参数：这是一个整数列表，用于计算 g 函数在模型词汇表上的分数。建议使用 20 到 30 个唯一的随机数，以在可检测性和生成质量之间取得平衡。
- `ngram_len` 参数：用于平衡稳健性和可检测性。值越大，水印越易被检测，但也更易受到干扰影响。推荐值为 5，最小值应为 2。

你还可以根据实际性能需求调整配置。更多信息可查阅 [`SynthIDTextWatermarkingConfig` 类](https://huggingface.co/docs/transformers/v4.46.0/en/internal/generation_utils#transformers.SynthIDTextWatermarkingConfig)。研究论文还分析了不同配置值如何影响水印性能的具体影响。

### 应用水印

将水印应用到文本生成中非常简单。你只需定义配置，并将 `SynthIDTextWatermarkingConfig` 对象作为 `watermarking_config=` 参数传递给 `model.generate()`，生成的文本就会自动携带水印。你可以在 [SynthID Text Space](https://huggingface.co/spaces/google/synthid-text) 中体验交互式示例，看看你是否能察觉到水印的存在。

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
)

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained('repo/id')
model = AutoModelForCausalLM.from_pretrained('repo/id')

# 配置 SynthID Text
watermarking_config = SynthIDTextWatermarkingConfig(
    keys=[654, 400, 836, 123, 340, 443, 597, 160, 57, ...],
    ngram_len=5,
)

# 使用水印生成文本
tokenized_prompts = tokenizer(["your prompts here"])
output_sequences = model.generate(
    **tokenized_prompts,
    watermarking_config=watermarking_config,
    do_sample=True,
)
watermarked_text = tokenizer.batch_decode(output_sequences)
```

### 检测水印

水印设计为对人类几乎不可察觉，但能被训练好的分类器检测。每个水印配置都需要一个对应的检测器。

训练检测器的基本步骤如下：

1. 确定一个水印配置。
2. 收集一个包含带水印和未带水印文本的训练集，分为训练集和测试集，推荐至少 10,000 个示例。
3. 使用模型生成不带水印的文本。
4. 使用模型生成带水印的文本。
5. 训练水印检测分类器。
6. 将水印配置及相应检测器投入生产环境。

Transformers 提供了一个 [贝叶斯检测器类](https://huggingface.co/docs/transformers/v4.46.0/en/internal/generation_utils#transformers.BayesianDetectorModel)，并附带一个 [端到端示例](https://github.com/huggingface/transformers/tree/v4.46.0/examples/research_projects/synthid_text/detector_training.py)，展示如何使用特定水印配置训练检测器。如果多个模型使用相同的分词器，可以共享水印配置和检测器，前提是训练集中包含所有相关模型的样本。这个训练好的检测器可以上传到私有的 Hugging Face Hub，使其在组织内部可用。Google 的 [负责任生成式 AI 工具包](https://ai.google.dev/responsible/docs/safeguards/synthid) 提供了更多关于将 SynthID Text 投入生产的指南。

### 限制

SynthID Text 的水印在某些文本变形下依然有效，如截断、少量词汇修改或轻微的改写，但也有其局限性：

- 在事实性回复中，水印应用效果较弱，因为增强生成的空间有限，否则可能降低准确性。
- 如果 AI 生成的文本被彻底改写或翻译为其他语言，检测器的置信度可能显著降低。

虽然 SynthID Text 不能直接阻止有目的的攻击者，但它可以增加滥用 AI 生成内容的难度，并与其他方法结合，覆盖更多内容类型和平台。
