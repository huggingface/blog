---
title: "Llama 3.1：405B/70B/8B 模型的多语言与长上下文能力解析" 
thumbnail: /blog/assets/llama31/thumbnail.jpg
authors:
- user: philschmid
- user: osanseviero
- user: alvarobartt
- user: lvwerra
- user: dvilasuero
- user: reach-vb
- user: marcsun13
- user: pcuenq
translators:
- user: AdinaY
---

# Llama 3.1 - 405B、70B 和 8B 的多语言与长上下文能力解析

Llama 3.1 发布了！今天我们迎来了 Llama 家族的新成员 Llama 3.1 进入 Hugging Face 平台。我们很高兴与 Meta 合作，确保在 Hugging Face 生态系统中实现最佳集成。Hub 上现有八个开源权重模型 (3 个基础模型和 5 个微调模型)。

Llama 3.1 有三种规格: 8B 适合在消费者级 GPU 上进行高效部署和开发，70B 适合大规模 AI 原生应用，而 405B 则适用于合成数据、大语言模型 (LLM) 作为评判者或蒸馏。这三个规格都提供基础版和指令调优版。

除了六个生成模型，Meta 还发布了两个新模型: Llama Guard 3 和 Prompt Guard。Prompt Guard 是一个小型分类器，可以检测提示注入和越狱。Llama Guard 3 是一个保护模型，能够分类 LLM 输入和生成的内容。

此次发布的一些功能和集成包括:

- [Hub 上的模型](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
- Hugging Face Transformers 和 TGI 集成
- [Meta Llama 3.1 405B Instruct 的 Hugging Chat 集成](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405b-instruct/)
- 使用推理端点、Google Cloud、Amazon SageMaker 和 DELL Enterprise Hub 进行推理和部署集成
- FP8、AWQ 和 GPTQ 的量化，便于推理
- 使用 🤗 TRL 在单个 GPU 上微调 Llama 3.1 8B
- 使用 Distilabel 生成 Llama 3.1 70B 和 405B 的合成数据

## 目录

  - [Llama 3.1 的新功能](#whats-new-with-llama-31)
  - [Llama 3.1 需要多少内存？](#how-much-memory-does-llama-31-need)
    - [推理内存需求](#inference-memory-requirements)
    - [训练内存需求](#training-memory-requirements)
  - [Llama 3.1 评估](#llama-31-evaluation)
  - [使用 Hugging Face Transformers](#using-hugging-face-transformers)
  - [如何使用 Llama 3.1](#how-to-prompt-llama-31)
    - [内置工具调用](#built-in-tool-calling)
  - [自定义工具调用](#custom-tool-calling)
  - [演示](#demo)
  - [Llama 3.1 405B 的 FP8、AWQ 和 GPTQ 量化](#llama-31-405b-quantization-with-fp8-awq-and-gptq)
  - [推理集成](#inference-integrations)
    - [Hugging Face 推理 API](#hugging-face-inference-api)
    - [Hugging Face 推理端点](#hugging-face-inference-endpoints)
  - [Hugging Face 合作伙伴集成](#hugging-face-partner-integrations)
  - [使用 Hugging Face TRL 进行微调](#fine-tuning-with-hugging-face-trl)
  - [使用 distilabel 生成合成数据](#synthetic-data-generation-with-distilabel)
  - [附加资源](#additional-resources)
  - [致谢](#acknowledgments)

## Llama 3.1 的新功能

Llama 3.1 为什么令人兴奋？在前代产品的基础上，Llama 3.1 增加了一些关键新功能:

- 128K token 的长上下文能力 (相较于原来的 8K)
- 多语言支持
- 工具使用功能
- 拥有 4050 亿参数的超大稠密模型
- 更宽松的许可证

让我们深入了解这些新功能！

Llama 3.1 版本引入了基于 Llama 3 架构的六个新开源 LLM 模型。它们有三种规格: 8B、70B 和 405B 参数，每种都有基础版 (预训练) 和指令调优版。所有版本都支持 128K token 的上下文长度和 8 种语言，包括英语、德语、法语、意大利语、葡萄牙语、印地语、西班牙语和泰语。Llama 3.1 继续使用分组查询注意力 (GQA)，这是一种高效的表示方式，有助于处理更长的上下文。

- [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B): 基础 8B 模型
- [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct): 基础 8B 模型的指令调优版
- [Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B): 基础 70B 模型
- [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct): 基础 70B 模型的指令调优版
- [Meta-Llama-3.1-405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B): 基础 405B 模型
- [Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct): 基础 405B 模型的指令调优版

除了这六个语言模型，还发布了 Llama Guard 3 和 Prompt Guard。

- [Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B) 是 Llama Guard 家族的最新版本，基于 Llama 3.1 8B 进行微调。它为生产用例而设计，具有 128k 的上下文长度和多语言能力。Llama Guard 3 可以分类 LLM 的输入 (提示) 和输出，以检测在风险分类中被认为不安全的内容。
- [Prompt Guard](https://huggingface.co/meta-llama/Prompt-Guard-86M)，另一方面，是一个小型 279M 参数的基于 BERT 的分类器，可以检测提示注入和越狱。它在大规模攻击语料库上训练，并建议使用特定应用的数据进行进一步微调。

与 Llama 3 相比，Llama 3.1 的新特点是指令模型在工具调用方面进行了微调，适用于智能体用例。内置了两个工具 (搜索，使用 Wolfram Alpha 进行数学推理)，可以扩展为自定义 JSON 功能。

Llama 3.1 模型在定制 GPU 集群上训练了超过 15 万亿 token，总计 39.3M GPU 小时 (8B 1.46M，70B 7.0M，405B 30.84M)。我们不知道训练数据集混合的具体细节，但我们猜测它在多语言方面有更广泛的策划。Llama 3.1 Instruct 已优化用于指令跟随，并在公开可用的指令数据集以及超过 2500 万合成生成的示例上进行监督微调 (SFT) 和人类反馈的强化学习 (RLHF)。Meta 开发了基于 LLM 的分类器，以在数据混合创建过程中过滤和策划高质量的提示和响应。

关于许可条款，Llama 3.1 具有非常相似的许可证，但有一个关键区别: **它允许使用模型输出来改进其他 LLM**。这意味着合成数据生成和蒸馏是允许的，即使是不同的模型！这对 405B 模型尤其重要，如后面所讨论的。许可证允许再分发、微调和创建衍生作品，仍然要求派生模型在其名称的开头包括 “Llama”，并且任何衍生作品或服务必须提及 “Built with Llama”。有关完整详情，请确保阅读 [官方许可证](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/LICENSE)。

## Llama 3.1 需要多少内存？

Llama 3.1 带来了令人兴奋的进步。然而，运行它需要仔细考虑硬件资源。我们分解了三种模型规格在训练和推理中的内存需求。

### 推理内存需求

对于推理，内存需求取决于模型规格和权重的精度。以下是不同配置所需的近似内存:

<table>
  <tr>
   <td><strong> 模型规格 </strong>
   </td>
   <td><strong>FP16</strong>
   </td>
   <td><strong>FP8</strong>
   </td>
   <td><strong>INT4</strong>
   </td>
  </tr>
  <tr>
   <td>8B
   </td>
   <td>16 GB
   </td>
   <td>8 GB
   </td>
   <td>4 GB
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td>140 GB
   </td>
   <td>70 GB
   </td>
   <td>35 GB
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>810 GB
   </td>
   <td>405 GB
   </td>
   <td>203 GB
   </td>
  </tr>
</table>

_注意: 上面引用的数字表示仅加载模型检查点所需的 GPU VRAM。它们不包括内核或 CUDA 图形的 torch 保留空间。_

例如，一个 H100 节点 (8x H100) 有约 640GB 的 VRAM，因此 405B 模型需要在多节点设置中运行或以较低精度 (例如 FP8) 运行，这是推荐的方法。

请记住，较低精度 (例如 INT4) 可能会导致一些精度损失，但可以显著减少内存需求并提高推理速度。除了模型权重外，您还需要将 KV 缓存保持在内存中。它包含模型上下文中所有 token 的键和值，以便在生成新 token 时不需要重新计算。特别是当利用可用的长上下文长度时，它变得至关重要。在 FP16 中，KV 缓存内存需求如下:

<table>
  <tr>
   <td><strong> 模型规格 </strong>
   </td>
   <td><strong>1k token</strong>
   </td>
   <td><strong>16k token</strong>
   </td>
   <td><strong>128k token</strong>
   </td>
  </tr>
  <tr>
   <td>8B
   </td>
   <td>0.125 GB
   </td>
   <td>1.95 GB
   </td>
   <td>15.62 GB
   </td>
</tr>
  <tr>
   <td>70B
   </td>
   <td>0.313 GB
   </td>
   <td>4.88 GB
   </td>
   <td>39.06 GB
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>0.984 GB
   </td>
   <td>15.38
   </td>
   <td>123.05 GB
   </td>
  </tr>
</table>

特别是对于小规格模型，当接近上下文长度上限时，缓存使用的内存与权重一样多。

### 训练内存需求

以下表格概述了使用不同技术训练 Llama 3.1 模型的大致内存需求:

<table>
  <tr>
   <td><strong>模型规格</strong>
   </td>
   <td><strong>1k token</strong>
   </td>
   <td><strong>16k token</strong>
   </td>
   <td><strong>128k token</strong>
   </td>
  </tr>
  <tr>
   <td>8B
   </td>
   <td>0.125 GB
   </td>
   <td>1.95 GB
   </td>
   <td>15.62 GB
   </td>


  </tr>
  <tr>
   <td>70B
   </td>
   <td>0.313 GB
   </td>
   <td>4.88 GB
   </td>
   <td>39.06 GB
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>0.984 GB
   </td>
   <td>15.38
   </td>
   <td>123.05 GB
   </td>
  </tr>
</table>

_注意: 这些是估计值，可能会根据具体实现细节和优化情况有所不同。_

## Llama 3.1 评估

_注意: 我们目前正在新的 [Open LLM Leaderboard 2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) 上单独评估 Llama 3.1，并将在今天晚些时候更新此部分。以下是 Meta 官方评估的摘录。_


<table>
  <tr>
   <td><strong><em>类别</em></strong>
   </td>
   <td><strong><em>基准</em></strong>
   </td>
   <td><strong><em>样本数</em></strong>
   </td>
   <td><strong><em>指标</em></strong>
   </td>
   <td><strong><em>Llama 3 8B</em></strong>
   </td>
   <td><strong><em>Llama 3.1 8B</em></strong>
   </td>
   <td><strong><em>Llama 3 70B</em></strong>
   </td>
   <td><strong><em>Llama 3.1 70B</em></strong>
   </td>
   <td><strong><em>Llama 3.1 405B</em></strong>
   </td>
  </tr>
  <tr>
   <td><em>综合</em>
   </td>
   <td><em>MMLU</em>
   </td>
   <td><em>5</em>
   </td>
   <td><em>宏观平均/字符准确率</em></td>
   <td><em>66.7</em>
   </td>
   <td><em>66.7</em>
   </td>
   <td><em>79.5</em>
   </td>
   <td><em>79.3</em>
   </td>
   <td><em>85.2</em></td>
  </tr>
  <tr>
   <td></td>
   <td><em>MMLU PRO（CoT）</em></td>
   <td><em>5</em></td>
   <td><em>宏观平均/字符准确率</em></td>
   <td><em>36.2</em></td>
   <td><em>37.1</em></td>
   <td><em>55.0</em></td>
   <td><em>53.8</em></td>
   <td><em>61.6</em></td>
  </tr>
  <tr>
   <td></td>
   <td><em>AGIEval 英语</em></td>
   <td><em>3-5</em></td>
   <td><em>平均/字符准确率</em></td>
   <td><em>47.1</em></td>
   <td><em>47.8</em></td>
   <td><em>63.0</em></td>
   <td><em>64.6</em></td>
   <td><em>71.6</em></td>
  </tr>
  <tr>
   <td></td>
   <td><em>CommonSenseQA</em></td>
   <td><em>7</em></td>
   <td><em>字符准确率</em></td>
   <td><em>72.6</em></td>
   <td><em>75.0</em></td>
   <td><em>83.8</em></td>
   <td><em>84.1</em></td>
   <td><em>85.8</em></td>
  </tr>
  <tr>
   <td></td>
   <td><em>Winogrande</em></td>
   <td><em>5</em></td>
   <td><em>字符准确率</em></td>
   <td><em>-</em></td>
   <td><em>60.5</em></td>
   <td><em>-</em></td>
   <td><em>83.3</em></td>
   <td><em>86.7</em></td>
  </tr>
  <tr>
   <td></td>
   <td><em>BIG-Bench Hard（CoT）</em></td>
   <td><em>3</em></td>
   <td><em>平均/完全匹配</em></td>
   <td><em>61.1</em></td>
   <td><em>64.2</em></td>
   <td><em>81.3</em></td>
   <td><em>81.6</em></td>
   <td><em>85.9</em></td>
  </tr>
  <tr>
   <td></td>
   <td><em>ARC-Challenge</em></td>
   <td><em>25</em></td>
   <td><em>字符准确率</em></td>
   <td><em>79.4</em></td>
   <td><em>79.7</em></td>
   <td><em>93.1</em></td>
   <td><em>92.9</em></td>
   <td><em>96.1</em></td>
  </tr>
  <tr>
   <td><em>知识推理</em></td>
   <td><em>TriviaQA-Wiki</em></td>
   <td><em>5</em></td>
   <td><em>完全匹配</em></td>
   <td><em>78.5</em></td>
   <td><em>77.6</em></td>
   <td><em>89.7</em></td>
   <td><em>89.8</em></td>
   <td><em>91.8</em></td>
  </tr>
  <tr>
   <td></td>
   <td><em>SQuAD</em></td>
   <td><em>1</em></td>
   <td><em>完全匹配</em></td>
   <td><em>76.4</em></td>
   <td><em>77.0</em></td>
   <td><em>85.6</em></td>
   <td><em>81.8</em></td>
   <td><em>89.3</em></td>
  </tr>
  <tr>
   <td><em>阅读理解</em></td>
   <td><em>QuAC（F1）</em></td>
   <td><em>1</em></td>
   <td><em>F1</em></td>
   <td><em>44.4</em></td>
   <td><em>44.9</em></td>
   <td><em>51.1</em></td>
   <td><em>51.1</em></td>
   <td><em>53.6</em></td>
  </tr>
  <tr>
   <td></td>
   <td><em>BoolQ</em></td>
   <td><em>0

</em></td>
   <td><em>字符准确率</em></td>
   <td><em>75.7</em></td>
   <td><em>75.0</em></td>
   <td><em>79.0</em></td>
   <td><em>79.4</em></td>
   <td><em>80.0</em></td>
  </tr>
  <tr>
   <td></td>
   <td><em>DROP（F1）</em></td>
   <td><em>3</em></td>
   <td><em>F1</em></td>
   <td><em>58.4</em></td>
   <td><em>59.5</em></td>
   <td><em>79.7</em></td>
   <td><em>79.6</em></td>
   <td><em>84.8</em></td>
  </tr>
</table>

## 使用 Hugging Face Transformers

Llama 3.1 需要进行少量建模更新，以有效处理 RoPE 缩放。使用 Transformers [4.43 版](https://github.com/huggingface/transformers/tags)，您可以使用新的 Llama 3.1 模型，并利用 Hugging Face 生态系统中的所有工具。确保使用最新的 `transformers` 版本:

```bash
pip install "transformers>=4.43" --upgrade
```

几个细节:

- Transformers 默认以 bfloat16 加载模型。这是 Meta 发布的原始检查点使用的类型，因此这是确保最佳精度或进行评估的推荐方法。
- 助手响应可能以特殊 token `<|eot_id|>` 结尾，但我们还必须在找到常规 EOS token 时停止生成。我们可以通过在 `eos_token_id` 参数中提供终止符列表来提前停止生成。
- 我们使用了 Meta 代码库中的默认采样参数 (`temperature` 和 `top_p` )。我们还没有时间进行广泛测试，请随意探索！

以下代码段显示了如何使用 `meta-llama/Meta-Llama-3.1-8B-Instruct` 。它大约需要 16 GB 的 VRAM，适合许多消费者级 GPU。相同的代码段适用于 `meta-llama/Meta-Llama-3.1-70B-Instruct` ，在 140GB VRAM 和 `meta-llama/Meta-Llama-3.1-405B-Instruct` (需要 810GB VRAM)，使其成为生产用例的非常有趣的模型。可以通过以 8 位或 4 位模式加载进一步减少内存消耗。

```python
from transformers import pipeline
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
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
# Arrrr, me hearty! Yer lookin' fer a bit o' information about meself, eh? Alright then, matey! I be a language-generatin' swashbuckler, a digital buccaneer with a penchant fer spinnin' words into gold doubloons o' knowledge! Me name be... (dramatic pause)...Assistant! Aye, that be me name, and I be here to help ye navigate the seven seas o' questions and find the hidden treasure o' answers! So hoist the sails and set course fer adventure, me hearty! What be yer first question?
```

您还可以自动量化模型，以 8 位甚至 4 位模式加载，使用 bitsandbytes。4 位加载大 70B 版本大约需要 34 GB 的内存运行。这是如何以 4 位模式加载生成管道:

```python
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": {"load_in_4bit": True}
    },
)
```

有关使用 `transformers` 模型的更多详细信息，请查看 [模型卡片](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)。

_注意: Transformers 处理所有棘手的提示模板问题，如果您想了解更多关于提示的信息，请查看下一部分。_

## 如何使用 Llama 3.1

基础模型没有提示格式。像其他基础模型一样，它们可以用于继续输入序列并进行合理的延续或零样本/少样本推理。它们也是微调您自己用例的绝佳基础。

指令版本支持具有 4 个角色的对话格式:

1. **system:** 设置对话的上下文。它允许包括规则、指南或必要的信息，帮助有效响应。它也用于在适当情况下启用工具使用。
2. **user:** 用户输入、命令和对模型的问题。
3. **assistant:** 助手的响应，基于 `system` 和 `user` 提示中提供的上下文。
4. **ipython:** Llama 3.1 中引入的新角色。当工具调用返回给 LLM 时作为输出使用。

指令版本使用以下对话结构进行简单对话:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ model_answer_1 }}<|eot_id|>
```

Llama 3.1 指令模型现在支持工具调用，包括三个内置工具 (brave_search、wolfram_alpha 和 code_interpreter) 和通过 JSON 函数调用的自定义工具调用。内置工具使用 Python 语法。生成 Python 代码以进行函数调用是代码解释器工具的一部分，必须在系统提示中使用 `Environment` 关键字启用，如下所示。

### 内置工具调用

包括 "Environment: ipython" 会打开代码解释器模式，模型可以生成它期望被执行的 Python 代码。助手响应的消息体以特殊标记 `<|python_tag|>` 开头，以 `<|eom_id|>` 结尾，而不是标准 `<|eot_id|>`。后者表示回合结束，而前者表示继续多步推理。

<details close>
<summary>内置工具调用示例</summary>

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>


Environment: ipython
Tools: brave_search, wolfram_alpha

Cutting Knowledge Date: 01 March 2023
Today's Date: 13 July 2024


You are a helpful Assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Weather in Menlo Park, California<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

此时模型的响应将包括调用其中一个支持的工具 (在本例中为 `brave_search` ) 的 Python 代码:

```
<|python_tag|>brave_search.call(query="current weather in Menlo Park, California")<|eom_id|>
```

执行调用的响应然后发送回模型以检索最终响应。为了简洁，以下内容将附加到前面代码段中显示的消息:

```
<|python_tag|>brave_search.call(query="Menlo Park California weather")<|eom_id|><|start_header_id|>ipython<|end_header_id|>

{"query": "Menlo Park California weather", "top_k": [{"title": "10-Day Weather Forecast for West Menlo Park, CA - The Weather Channel | weather.com", "url": "https://weather.com/weather/tenday/l/West+Menlo+Park+CA?canonicalCityId=b2375713aa1943aad7d1a13a85e1c0adad13c1b10563b2bbaad70734dc61cf11", "description": "Be prepared with the most accurate 10-day forecast for West <strong>Menlo</strong> <strong>Park</strong>, CA with highs, lows, chance of precipitation from The <strong>Weather</strong> Channel and <strong>Weather</strong>.com", "type": "search_result"},....}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

LLM 的最终响应将是:

```
The current weather in Menlo Park, California is mostly sunny with a high of 77°F and a low of 56°F.<|eot_id|>
```

</details>

## 自定义工具调用

Llama 3.1 指令支持从单个用户消息中调用自定义函数。以下提示提供了如何从模型输出调用自定义函数的示例。在自定义函数调用中，模型输出 `<|eot_id|>` 而不是 `<|eom_id|>` 。需要调整系统提示以告知模型如何处理函数调用输出。

<details close>
<summary>自定义工具调用 JSON 函数</summary>

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the orginal user question.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{
    "type": "function",
    "function": {
    "name": "get_current_conditions",
    "description": "Get the current weather conditions for a specific location",
    "parameters": {
        "type": "object",
        "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g., San Francisco, CA"
        },
        "unit": {
            "type": "string",
            "enum": ["Celsius", "Fahrenheit"],
            "description": "The temperature unit to use. Infer this from the user's location."
        }
        },
        "required": ["location", "unit"]
    }
    }
}

Question: what is the weather like in Menlo Park?<|eot_id|><|start_header_id|>assitant<|end_header_id|>

{"name": "get_current_conditions", "parameters": {"location": "Menlo Park, CA", "unit": "Fahrenheit"}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>
```

当我们从选定的工具检索输出时，我们将其传回模型，使用相同的 `<|python_tag|>` 分隔符。`<|python_tag|>` 不意味着使用 Python。它仅用于表示任何工具的输出开始。

```
<|python_tag|>{
    "tool_call_id": "get_current_conditions"
    "output": "Clouds giving way to sun Hi: 76° Tonight: Mainly clear early, then areas of low clouds forming Lo: 56°"
}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The weather in Menlo Park is currently cloudy with a high of 76° and a low of 56°, with clear skies expected tonight.<|eot_id|>
```

这种格式必须精确复制才能有效使用。transformers 中可用的聊天模板使其易于正确格式化提示。

</details>

## 演示

您可以在以下演示中试验三种指令模型:

- Llama 3.1 405B 的 Hugging Chat [https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405b-instruct/](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405b-instruct/)
- Llama 3.1 70B 的 Hugging Chat [https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-70b-instruct/](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-70b-instruct/)
- Llama 3.1 8B 演示的 Gradio 驱动的 Space [https://huggingface.co/spaces/ysharma/Chat_with_Meta_llama3_1_8b](https://huggingface.co/spaces/ysharma/Chat_with_Meta_llama3_1_8b)

整个堆栈都是开源的。Hugging Chat 由 [chat-ui](https://github.com/huggingface/chat-ui) 和 [text-generation-inference](https://github.com/huggingface/text-generation-inference) 提供支持。

## Llama 3.1 405B 的 FP8、AWQ 和 GPTQ 量化

Meta 创建了 [Llama 3.1 405B 的官方 FP8 量化版本](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8)，精度损失最小。为实现这一目标，FP8 量化仅应用于模型的主要线性运算符，例如 FFNs 的门和上升及下降投影 (涵盖 75% 的推理 FLOPs)。我们共同努力，确保此 FP8 量化检查点在社区中兼容 (transformers, TGI, VLLM)。

此外，我们使用 AutoAWQ 和 AutoGPTQ 创建了 INT4 的 AWQ 和 GPTQ 量化变体。对于 AWQ，所有线性层都使用 GEMM 内核进行量化，将零点量化到 4 位，组大小为 128; 对于 GPTQ，相同的设置仅使用 GPTQ 内核。我们确保 INT4 检查点与 transformers 和 TGI 兼容，包括 Marlin 内核支持，以加快 TGI 中 GPTQ 量化的推理速度。

可用的 Llama 3.1 405B 的量化权重:

- [meta-llama/Meta-Llama-3.1-405B-Base-FP8](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-FP8): 官方 FP8 量化权重，可在 8xH100 上运行
- [meta-llama/Meta-Llama-3.1-405B-Instruct-FP8](https://huggingface.co/sllhf/Meta-Llama-3.1-405B-Instruct-FP8): 官方 FP8 量化权重，可在 8xH100 上运行
- [hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4): Hugging Face 量化权重，可在 8xA100 80GB, 8xH100 80GB 和 8xA100 40GB (减少 KV 缓存且无 CUDA 图形) 上运行
- [hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4:](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4): Hugging Face 量化权重，可在 8xA100 80GB, 8xH100 80GB 和 8xA100 40GB (减少 KV 缓存且无 CUDA 图形) 上运行
- [hugging-quants/Meta-Llama-3.1-405B-BNB-NF4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-BNB-NF4): Hugging Face 量化权重，适用于 QLoRA 微调
- [hugging-quants/Meta-Llama-3.1-405B-Instruct-BNB-NF4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-BNB-NF4): Hugging Face 量化权重，适用于在 8xA100 和 4xH100 上推理

[Hugging Quants 组织](https://huggingface.co/hugging-quants) 还包含 70B 和 8B 版本的量化检查点。

## 推理集成

### Hugging Face 推理 API

[Hugging Face PRO 用户现在可以访问独家 API 端点](https://huggingface.co/blog/inference-pro)，托管 Llama 3.1 8B Instruct、Llama 3.1 70B Instruct 和 Llama 3.1 405B Instruct AWQ，由 [text-generation-inference](https://github.com/huggingface/text-generation-inference) 提供支持。所有版本都支持 Messages API，因此与 OpenAI 客户端库兼容，包括 LangChain 和 LlamaIndex。

_注意: 使用 `pip install "huggingface_hub>=0.24.1"` 更新到最新的 `huggingface_hub` 版本。_

```python
from huggingface_hub import InferenceClient

# 初始化客户端，指向一个可用的模型
client = InferenceClient()

chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful and honest programming assistant."},
        {"role": "user", "content": "Is Rust better than Python?"},
    ],
    stream=True,
    max_tokens=500
)

# 迭代并打印流
for message in chat_completion:
    print(message.choices[0].delta.content, end="")
```

有关使用 Messages API 的更多详细信息，请查看 [此帖子](https://huggingface.co/blog/tgi-messages-api)。

### Hugging Face 推理端点

您可以在 Hugging Face 的 [推理端点](https://ui.endpoints.huggingface.co/) 上部署 Llama 3.1，它使用 Text Generation Inference 作为后端。Text Generation Inference 是 Hugging Face 开发的生产就绪推理容器，支持 FP8、连续批处理、token 流、张量并行，以便在多个 GPU 上快速推理。要部署 Llama 3.1，请转到 [模型页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct) 并点击部署 -> 推理端点小部件:

- [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) 推荐在 1x NVIDIA A10G 或 L4 GPU 上运行
- [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) 推荐在 4x NVIDIA A100 或量化为 AWQ/GPTQ 在 2x A100 上运行
- [Meta-Llama-3.1-405B-Instruct-FP8](https://huggingface.co/sllhf/Meta-Llama-3.1-405B-Instruct-FP8) 推荐在 8x NVIDIA H100 上以 FP 运行或量化为 [AWQ](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4)/[GPTQ](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4) 在 8x A100 上运行

```python
from huggingface_hub import InferenceClient

# 初始化客户端，指向一个可用的模型
client = InferenceClient(
    base_url="<ENDPOINT_URL>",
)

# 创建一个聊天完成
chat_completion = client.chat.completions.create(
    model="ENDPOINT",
    messages=[
        {"role": "system", "content": "You are a helpful and honest programming assistant."},
        {"role": "user", "content": "Is Rust better than Python?"},
    ],
    stream=True,
    max_tokens=500
)

# 迭代并打印流
for message in chat_completion:
    print(message.choices[0].delta.content, end="")
```

## Hugging Face 合作伙伴集成

_注意: 我们目前正在与我们的合作伙伴 AWS、Google Cloud、Microsoft Azure 和 DELL 合作，将 Llama 3.1 8B、70B 和 405B 添加到 Amazon SageMaker、Google Kubernetes Engine、Vertex AI Model Catalog、Azure AI Studio、DELL Enterprise Hub。我们将在容器可用时更新此部分 - 您可以 [订阅 Hugging Squad 以获取电子邮件更新](https://mailchi.mp/huggingface/squad)。_

## 使用 Hugging Face TRL 进行微调

在本节中，我们将查看 Hugging Face 生态系统中可用的工具，以便在消费者级 GPU 上高效训练 Llama 3.1。下面是一个示例命令，用于在 OpenAssistant 的 [chat 数据集](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) 上微调 Llama 3.1 8B。我们使用 4 位量化和 [QLoRA](https://arxiv.org/abs/2305.14314) 来节省内存，以针对所有注意力块的线性层。

<details close>
<summary>使用 Hugging Face TRL 的微调示例</summary>

首先，安装最新版本的 🤗 TRL 并克隆 repo 以访问 [训练脚本](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py):

```
pip install "transformers>=4.43" --upgrade
pip install --upgrade bitsandbytes
pip install --ugprade peft
pip install git+https://github.com/huggingface/trl
git clone https://github.com/huggingface/trl
cd trl
```

然后你可以运行脚本:

```
python \
    examples/scripts/sft.py \
    --model_name meta-llama/Meta-Llama-3.1-8B \
    --dataset_name OpenAssistant/oasst_top1_2023-08-25 \
    --dataset_text_field="text" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --report_to "none" \
    --bf16 \
    --max_seq_length 1024 \
    --lora_r 16 --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --load_in_4bit \
    --use_peft \
    --attn_implementation "flash_attention_2" \
    --logging_steps=10 \
    --gradient_checkpointing \
    --output_dir llama31
```

如果您有更多的 GPU，可以使用 DeepSpeed 和 ZeRO Stage 3 运行训练:

```
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft.py \
    --model_name meta-llama/Meta-Llama-3.1-8B \
    --dataset_name OpenAssistant/oasst_top1_2023-08-25 \
    --dataset_text_field="text" \
    --per_device train batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --report_to wandb \
    --bf16 \
    --max_seq_length 1024 \
    --attn_implementation eager \
    --logging_steps=10 \
    --gradient_checkpointing \
    --output_dir models/llama
```

</details>

## 使用 distilabel 生成合成数据

Llama 3.1 许可证的一个重大变化是，它允许使用模型输出来改进其他 LLM，这意味着您可以使用 Llama 3.1 模型生成合成数据集，并使用它们来微调更小、更专业的模型。

让我们看一个示例，如何使用 [distilabel](https://github.com/argilla-io/distilabel)，一个用于生成合成数据的开源框架，生成一个偏好数据集。该数据集可用于使用 TRL 提供的偏好优化方法 (如 DPO 或 KTO) 微调模型。

首先安装最新的 `distilabel` 版本，包括 `hf-inference-endpoints` 额外组件，使用 `pip` 如下:

```bash
pip install “distilabel[hf-inference-endpoints]” --upgrade
```

然后定义一个管道:

- 从 Hugging Face Hub 加载带有指令的数据集。
- 使用 Hugging Face 推理端点，通过 Llama 3.1 70B Instruct 和 Llama 3.1 405B Instruct 生成响应。
- 最后，使用 Llama 3.1 405B Instruct 作为裁判，使用 UltraFeedback 提示对响应进行评分。从这些评分中，可以选择和拒绝响应，并使用偏好优化方法微调模型。

请参阅下面的代码以定义管道，或使用此 [Colab 笔记本](https://colab.research.google.com/drive/1o0ALge7DHBmcKgdyrk59yOL70tcGS3v4?usp=sharing) 自行运行并探索生成的数据集。

```python
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, CombineColumns
from distilabel.steps.tasks import TextGeneration, UltraFeedback

llama70B = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct"
)
llama405B = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
)

with Pipeline(name="synthetic-data-with-llama3") as pipeline:
    # 加载带有提示的数据集
    load_dataset = LoadDataFromHub(
        repo_id="argilla/10Kprompts-mini"
    )
    # 为每个提示生成两个响应
    generate = [
        TextGeneration(llm=llama70B),
        TextGeneration(llm=llama405B)
    ]
    # 将响应组合到一个列中
    combine = CombineColumns(
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"]
    )
    # 使用 405B LLM-as-a-judge 对响应进行评分
    rate = UltraFeedback(aspect="overall-rating", llm=llama405B)
    # 定义管道
    load_dataset >> generate >> combine >> rate

if __name__ == "__main__":
    distiset = pipeline.run()
```

接下来是什么？除了上述示例， `distilabel` 还提供了使用 LLM 在广泛的场景和主题中生成合成数据的令人兴奋的方法。它包括当前 SOTA 文献中的实现，用于任务如使用 LLM-as-a-judge 方法评估输出、进化指令、数据过滤以及定义自定义组件。

## 附加资源

- [Hub 上的模型](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
- [Hugging Face Llama Recipes](https://github.com/huggingface/huggingface-llama-recipes)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Llama 3.1 405B Instruct 的 Hugging Chat 演示](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405b-instruct/)
- [Meta 博客](https://ai.meta.com/blog/meta-llama-3-1/)

## 致谢

没有成千上万社区成员对 transformers、tgi、vllm、pytorch、LM Eval Harness 和许多其他项目的贡献，这些模型的发布和生态系统中的支持与评估是不可能实现的。这次发布离不开 [Clémentine](https://huggingface.co/clefourrier) 和 [Nathan](https://huggingface.co/SaylorTwift) 对 LLM 评估的支持; [Nicolas](https://huggingface.co/Narsil)、[Olivier Dehaene](https://huggingface.co/olivierdehaene) 和 [Daniël de Kok](https://huggingface.co/danieldk) 对 Text Generation Inference 支持的贡献; [Arthur](https://huggingface.co/ArthurZ)、[Matthew Carrigan](https://huggingface.co/Rocketknight1)、[Zachary Mueller](https://huggingface.co/muellerzr)、[Joao](https://huggingface.co/joaogante)、[Joshua Lochner](https://huggingface.co/Xenova) 和 [Lysandre](https://huggingface.co/lysandre) 对 Llama 3.1 集成到 `transformers` 的贡献; [Matthew Douglas](https://huggingface.co/mdouglas) 对量化支持的贡献; [Gabriel Martín Blázquez](https://huggingface.co/gabrielmbmb) 对 `distilabel` 支持的贡献; [Merve Noyan](https://huggingface.co/merve) 和 [Aymeric Roucher](https://huggingface.co/m-ric) 对审核的贡献; [hysts](huggingface.co/hysts) 和 [Yuvi](huggingface.co/ysharma) 对演示的贡献; [Ellie](https://huggingface.co/eliebak) 对微调测试的贡献; [Brigitte Tousignant](https://huggingface.co/BrigitteTousi) 和 [Florent Daudens](https://huggingface.co/fdaudens) 对沟通的贡献; [Nathan](https://huggingface.co/nsarrazin) 和 [Victor](https://huggingface.co/victor) 对 Hugging Chat 中 Llama 3.1 的可用性的贡献。

感谢 Meta 团队发布 Llama 3.1 并使其在开源 AI 社区中可用！