---
title: "Google 发布最新开放大语言模型 Gemma 2，现已登陆 Hugging Face Hub"
thumbnail: /blog/assets/gemma2/thumbnail.jpg
authors:
- user: philschmid
- user: osanseviero
- user: pcuenq
- user: lewtun
- user: tomaarsen
- user: reach-vb
translators:
- user: chenglu
---

# 欢迎使用 Gemma 2 - Google 最新的开放大语言模型

Google 发布了最新的开放大语言模型 Gemma 2，我们非常高兴与 Google 合作，确保其在 Hugging Face 生态系统中的最佳集成。你可以在 Hub 上找到 4 个开源模型（2 个基础模型和 2 个微调模型）。发布的功能和集成包括：

- [Hub 上的模型](https://huggingface.co/collections/google/g-667d6600fd5220e7b967f315)
- Hugging Face [Transformers 集成](https://github.com/huggingface/transformers/releases/tag/v4.42.0)
- 与 Google Cloud 和推理端点的集成

## 目录

- [什么是 Gemma 2？](#what-is-gemma-2)
- [Gemma 2 的技术进展](#technical-advances-in-gemma-2)
  - [滑动窗口注意力](#sliding-window-attention)
  - [软上限和注意力实现](#soft-capping-and-attention-implementations)
  - [知识蒸馏](#knowledge-distillation)
  - [模型合并](#model-merging)
- [Gemma 2 的评估](#gemma-2-evaluation)
  - [技术报告结果](#technical-report-results)
  - [开源 LLM 排行榜结果](#open-llm-leaderboard-results)
- [如何提示 Gemma 2](#how-to-prompt-gemma-2)
- [演示](#demo)
- [使用 Hugging Face Transformers](#using-hugging-facetransformers)
- [与 Google Cloud 的集成](#integration-with-google-cloud)
- [与推理端点的集成](#integration-with-inference-endpoints)
- [使用 🤗 TRL 进行微调](#fine-tuning-with-trl)
- [其他资源](#additional-resources)
- [致谢](#acknowledgments)

## Gemma 2 是什么？

Gemma 2 是 Google 最新的开放大语言模型。它有两种规模：90 亿参数和 270 亿参数，分别具有基础（预训练）和指令调优版本。Gemma 基于 Google DeepMind 的 Gemini，拥有 8K Tokens 的上下文长度：

- [gemma-2-9b](https://huggingface.co/google/gemma-2-9b): 90 亿基础模型。
- [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it): 90 亿基础模型的指令调优版本。
- [gemma-2-27b](https://huggingface.co/google/gemma-2-27b): 270 亿基础模型。
- [gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it): 270 亿基础模型的指令调优版本。

Gemma 2 模型的训练数据量约为其第一代的两倍，总计 13 万亿 Tokens（270 亿模型）和 8 万亿 Tokens（90 亿模型）的网页数据（主要是英语）、代码和数学数据。我们不知道训练数据混合的具体细节，只能猜测更大和更仔细的数据整理是性能提高的重要因素之一。

Gemma 2 与第一代使用相同的许可证，这是一个允许再分发、微调、商业用途和衍生作品的宽松许可证。

## Gemma 2 的技术进展

Gemma 2 与第一代有许多相似之处。它有 8192 Tokens 的上下文长度，并使用旋转位置嵌入 (RoPE)。与原始 Gemma 相比，Gemma 2 的主要进展有四点：

- [滑动窗口注意力](#sliding-window-attention): 交替使用滑动窗口和全二次注意力以提高生成质量。
- [Logit 软上限](#soft-capping-and-attention-implementations): 通过将 logits 缩放到固定范围来防止其过度增长，从而改进训练。
- [知识蒸馏](#knowledge-distillation): 利用较大的教师模型来训练较小的模型（适用于 90 亿模型）。
- [模型合并](#model-merging): 将两个或多个大语言模型合并成一个新的模型。

Gemma 2 使用 [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) 和 [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/) 在 [Google Cloud TPU (27B on v5p](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer?hl=en) 和 [9B on TPU v4)](https://cloud.google.com/tpu/docs/v4) 上进行训练。Gemma 2 Instruct 已针对对话应用进行了优化，并使用监督微调 (SFT)、大模型蒸馏、人类反馈强化学习 (RLHF) 和模型合并 (WARP) 来提高整体性能。

与预训练数据集混合类似，关于微调数据集或与 SFT 和 [RLHF](https://huggingface.co/blog/rlhf) 相关的超参数的细节尚未共享。

### 滑动窗口注意力

[滑动窗口注意力](https://huggingface.co/papers/2004.05150) 是一种用于减少 Transformer 模型中注意力计算的内存和时间需求的方法，已在 [Mistral](https://huggingface.co/papers/2310.06825) 等模型中使用。Gemma 2 的新颖之处在于每隔一层应用滑动窗口（局部 - 4096 Tokens），而中间层仍使用全局二次注意力（8192 Tokens）。我们推测这是为了在长上下文情况下提高质量（半数层仍然关注所有 Tokens），同时部分受益于滑动注意力的优势。

### 软上限和注意力实现

软上限是一种防止 logits 过度增长而不截断它们的技术。它通过将 logits 除以最大值阈值 (`soft_cap`)，然后通过 `tanh` 层（确保它们在 `(-1, 1)` 范围内），最后再乘以阈值。这确保了最终值在 `(-soft_cap, +soft_cap)` 区间内，不会丢失太多信息但稳定了训练。

综合起来，logits 的计算公式为：`logits ← soft_cap ∗ tanh(logits/soft_cap)`

Gemma 2 对最终层和每个注意力层都采用了软上限。注意力 logits 上限为 50.0，最终 logits 上限为 30.0。

在发布时，软上限与 Flash Attention / SDPA 不兼容，但它们仍可用于推理以实现最高效率。Gemma 2 团队观察到，在推理过程中不使用软上限机制时，差异非常小。

**注意：对于稳定的微调运行，仍需启用软上限，因此我们建议使用 `eager` 注意力进行微调，而不是 SDPA。**

### 知识蒸馏

知识蒸馏是一种常用技术，用于训练较小的 **学生** 模型以模仿较大但表现更好的 **教师** 模型的行为。这是通过将大语言模型的下一个 Token 预测任务与教师提供的 Token 概率分布（例如 GPT-4、Claude 或 Gemini）结合起来，从而为学生提供更丰富的学习信号。

根据 Gemma 2 技术报告，知识蒸馏用于预训练 90 亿模型，而 270 亿模型则是从头开始预训练的。

在后期训练中，Gemma 2 团队生成了来自教师（报告中未指定，但可能是 Gemini Ultra）的多样化补全集，然后使用这些合成数据通过 SFT 训练学生模型。这也是许多开源模型的基础，如 [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) 和 [OpenHermes](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)，它们完全基于较大大语言模型的合成数据进行训练。

尽管有效，但这种方法存在缺点，因为学生和教师之间的模型容量不匹配可能导致 **训练-推理不匹配**，即学生在推理期间生成的文本与训练期间看到的文本不同。

为解决这个问题，Gemma 2 团队采用了[“在线蒸馏”](https://arxiv.org/pdf/2306.13649)，其中学生从 SFT 提示生成补全。这些补全用于计算教师和学生 logits 之间的 KL 散度。通过在整个训练过程中最小化 KL 散度，学生能够准确地模拟教师的行为，同时最小化训练-推理不匹配。

这种方法非常有趣，正如我们在社区中看到的那样，在线 DPO 等在线方法会产生更强的模型，而在线蒸馏的一个优势在于只需要教师的 logits，因此无需依赖奖励模型或大语言模型作为评审员来改进模型。我们期待看到这种方法在未来几个月中是否会在微调人员中变得更受欢迎！

### 模型合并

[模型合并](https://huggingface.co/blog/mlabonne/merge-models) 是一种将两个或多个大语言模型合并成一个新模型的技术。这是相对较新和实验性的，可以不使用加速器进行。[Mergekit](https://github.com/arcee-ai/mergekit) 是一个流行的开源工具包，用于合并大语言模型。它实现了线性、SLERP、TIES、DARE 和其他合并技术。

根据技术报告，Gemma 2 使用了 [Warp](https://arxiv.org/abs/2406.16768)，这是一种新型合并技术，分三个独特阶段进行合并：

1. 指数移动平均 (EMA)：在强化学习 (RL) 微调过程中应用。
2. 球形线性插值 (SLERP)：在多个策略的 RL 微调后应用。
3. 向初始化线性插值 (LITI)：在 SLERP 阶段之后应用。

## Gemma 2 的评估

Gemma 模型的表现如何？以下是根据技术报告和新版 [开源 LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 对其他开源开放模型的性能比较。

### 技术报告结果

Gemma 2 的技术报告比较了不同开源 LLM 在之前开源 LLM 排行榜基准上的性能。

|            | Llama 3 (70B) | Qwen 1.5 (32B) | Gemma 2 (27B) |
| ---------- | ------------- | -------------- | ------------- |
| MMLU       | **79.2**      | 74.3           | 75.2          |
| GSM8K      | **76.9**      | 61.1           | 75.1          |
| ARC-c      | 68.8          | 63.6           | **71.4**      |
| HellaSwag  | **88.0**      | 85.0           | 86.4          |
| Winogrande | **85.3**      | 81.5           | 83.7          |

该报告还比较了小型语言模型的性能。

| Benchmark  | Mistral (7B) | Llama 3 (8B) | Gemma (8B) | Gemma 2 (9B) |
| ---------- | ------------ | ------------ | ---------- | ------------ |
| MMLU       | 62.5         | 66.6         | 64.4       | **71.3**     |
| GSM8K      | 34.5         | 45.7         | 50.9       | **62.3**     |
| ARC-C      | 60.5         | 59.2         | 61.1       | **68.4**     |
| HellaSwag  | **83.0**     | 82.0         | 82.3       | 81.9         |
| Winogrande | 78.5         | 78.5         | 79.0       | **80.6**     |

### 开源 LLM 排行榜结果

**注意：我们目前正在新的开源 LLM 排行榜基准上单独评估 Google Gemma 2，并将在今天晚些时候更新此部分。**

## 如何提示 Gemma 2

基础模型没有提示格式。像其他基础模型一样，它们可以用于继续输入序列的合理延续或零样本/少样本推理。指令版本有一个非常简单的对话结构：

```bash
<start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
LaMDA<end_of_turn>
<start_of_turn>model
LaMDA who?<end_of_turn><eos>
```

必须精确地复制此格式才能有效使用。稍后我们将展示如何使用 `transformers` 中的聊天模板轻松地复制指令提示。

## 演示

你可以在 Hugging Chat 上与 Gemma 27B 指令模型聊天！查看此链接：
https://huggingface.co/chat/models/google/gemma-2-27b-it

## 使用 Hugging Face Transformers

随着 Transformers [版本 4.42](https://github.com/huggingface/transformers/releases/tag/v4.42.0) 的发布，你可以使用 Gemma 并利用 Hugging Face 生态系统中的所有工具。要使用 Transformers 使用 Gemma 模型，请确保使用最新的 `transformers` 版本：

```bash
pip install "transformers==4.42.1" --upgrade
```

以下代码片段展示了如何使用 `transformers` 使用 `gemma-2-9b-it`。它需要大约 18 GB 的 RAM，适用于许多消费者 GPU。相同的代码片段适用于 `gemma-2-27b-it`，需要 56GB 的 RAM，使其非常适合生产用例。通过加载 8-bit 或 4-bit 模式，可以进一步减少内存消耗。

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=False,
)
assistant_response = outputs[0]["generated_text"][-1]["content"]
print(assistant_response)
```

> 啊哈，船长！我是数字海洋上的一艘谦卑的词语之船。他们叫我 Gemma，是 Google DeepMind 的杰作。我被训练在一堆文本宝藏上，学习如何像一个真正的海盗一样说话和写作。
>
> 问我你的问题吧，我会尽力回答，啊哈！🦜📚

**我们使用 bfloat16 因为这是指令调优模型的参考精度。在你的硬件上运行 float16 可能会更快，90 亿模型的结果应该是相似的。然而，使用 float16 时，270 亿指令调优模型会产生不稳定的输出：对于该模型权重，你必须使用 bfloat16。**

你还可以自动量化模型，以 8-bit 甚至 4-bit 模式加载。加载 4-bit 模式的 270 亿版本需要大约 18 GB 的内存，使其兼容许多消费者显卡和 Google Colab 中的 GPU。这是你在 4-bit 模式下加载生成管道的方式：

```python
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": {"load_in_4bit": True}
    },
)
```

有关使用 Transformers 模型的更多详细信息，请查看[模型卡](https://huggingface.co/gg-hf/gemma-2-9b)。

## 与 Google Cloud 和推理端点的集成

**注意：我们目前正在为 GKE 和 Vertex AI 添加新的容器，以高效运行 Google Gemma 2。我们将在容器可用时更新此部分。**

## 其他资源

- [Hub 上的模型](https://huggingface.co/collections/google/g-667d6600fd5220e7b967f315)
- [开放 LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Hugging Chat 上的聊天演示](https://huggingface.co/chat/models/google/gemma-2-27b-it)
- [Google 博客](https://blog.google/technology/developers/google-gemma-2/)
- Google Notebook 即将推出
- Vertex AI 模型花园 即将推出

## 致谢

在生态系统中发布此类模型及其支持和评估离不开许多社区成员的贡献，包括 [Clémentine](https://huggingface.co/clefourrier) 和 [Nathan](https://huggingface.co/SaylorTwift) 对 LLM 的评估；[Nicolas](https://huggingface.co/Narsil) 对文本生成推理的支持；[Arthur](https://huggingface.co/ArthurZ)、[Sanchit](https://huggingface.co/sanchit-gandhi)、[Joao](https://huggingface.co/joaogante) 和 [Lysandre](https://huggingface.co/lysandre) 对 Gemma 2 集成到 Transformers 中的支持；[Nathan](https://huggingface.co/nsarrazin) 和 [Victor](https://huggingface.co/victor) 使 Gemma 2 在 Hugging Chat 中可用。

感谢 Google 团队发布 Gemma 2 并使其对开源 AI 社区开放！
