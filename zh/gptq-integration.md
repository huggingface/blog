---
title: "使用 AutoGPTQ 和 transformers 让大语言模型更轻量化" 
thumbnail: /blog/assets/159_autogptq_transformers/thumbnail.jpg
authors:
- user: marcsun13
- user: fxmarty
- user: PanEa
  guest: true
- user: qwopqwop
  guest: true
- user: ybelkada
- user: TheBloke
  guest: true
translators:
- user: PanEa
  guest: true
- user: zhongdongy
  proofreader: true
---

# 使用 AutoGPTQ 和 transformers 让大语言模型更轻量化


大语言模型在理解和生成人类水平的文字方面所展现出的非凡能力，正在许多领域带来应用上的革新。然而，在消费级硬件上训练和部署大语言模型的需求也变得越来越难以满足。

🤗 Hugging Face 的核心使命是 _让优秀的机器学习普惠化_ ，而这正包括了尽可能地让所有人都能够使用上大模型。本着 [与 bitsandbytes 合作](https://huggingface.co/blog/4bit-transformers-bitsandbytes) 一样的精神，我们将 [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 代码库集成到了 Transformers 中，让用户使用 GPTQ 算法 ([Frantar et al. 2023](https://arxiv.org/pdf/2210.17323.pdf)) 在 8 位、4 位、3 位，甚至是 2 位精度下量化和运行模型成为可能。当使用 int4 量化时，精度的下降可以忽略不计，同时在小批量推理上保持着与 `fp16` 基线相当的速度。 需要注意的是，GPTQ 方法与 bitsandbytes 提出的训练后量化方法有所不同：它需要在量化阶段提供一个校准数据集。

本次集成支持英伟达 GPU 和基于 RoCm 的 AMD GPU。

## 目录

- [相关资源](#相关资源)
- [**GPTQ 论文总结**](#--gptq-论文总结--)
- [AutoGPTQ 代码库——一站式地将 GPTQ 方法应用于大语言模型](#autogptq-代码库——一站式地将-gptq-方法应用于大语言模型)
- [🤗 Transformers 对 GPTQ 模型的本地化支持](#---transformers-对-gptq-模型的本地化支持)
- [使用 **Optimum 代码库** 量化模型](#使用---optimum-代码库---量化模型)
- [通过 ***Text-Generation-Inference*** 使用 GPTQ 模型](#通过----text-generation-inference----使用-gptq-模型)
- [**使用 PEFT 微调量化后的模型**](#--使用-peft-微调量化后的模型--)
- [改进空间](#改进空间)
  * [已支持的模型](#已支持的模型)
- [结论和结语](#结论和结语)
- [致谢](#致谢)


## 相关资源

本文及相关版本发布提供了一些资源来帮助用户开启 GPTQ 量化的旅程：

- [原始论文](https://arxiv.org/pdf/2210.17323.pdf)
- [运行于 Google Colab 笔记本上的基础用例](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) —— 该笔记本上的用例展示了如何使用 GPTQ 方法量化你的 transformers 模型、如何进行量化模型的推理，以及如何使用量化后的模型进行微调。
- Transformers 中集成 GPTQ 的 [说明文档](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)
- Optimum 中集成 GPTQ 的 [说明文档](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
- TheBloke [模型仓库](https://huggingface.co/TheBloke?sort_models=likes#models) 中的 GPTQ 模型。


## **GPTQ 论文总结**

通常，量化方法可以分为以下两类：

1. 训练后量化 (Post Training Quantization, PTQ)：适度地使用一些资源来量化预训练好的模型，如一个校准数据集和几小时的算力。
2. 量化感知训练 (Quantization Aware Training, QAT)：在训练或进一步微调之前执行量化。

GPTQ 属于训练后量化，这对于大模型而言格外有趣且有意义，因为对其进行全参数训练以及甚至仅仅是微调都十分昂贵。

具体而言，GPTQ 采用 int4/fp16 (W4A16) 的混合量化方案，其中模型权重被量化为 int4 数值类型，而激活值则保留在 float16。在推理阶段，模型权重被动态地反量化回 float16 并在该数值类型下进行实际的运算。

该方案有以下两方面的优点：

- int4 量化能够节省接近4倍的内存，这是因为反量化操作发生在算子的计算单元附近，而不是在 GPU 的全局内存中。
- 由于用于权重的位宽较低，因此可以节省数据通信的时间，从而潜在地提升了推理速度。

GPTQ 论文解决了分层压缩的问题：

给定一个拥有权重矩阵 \\(W_{l}\\) 和输入 \\(X_{l}\\) 的网络层 \\(l\\)，我们期望获得一个量化版本的权重矩阵 \\(\hat{W}_{l}\\) 以最小化均方误差 (MSE)：

\\({\hat{W}_{l}}^{*} = argmin_{\hat{W_{l}}} \|W_{l}X-\hat{W}_{l}X\|^{2}_{2}\\)

一旦每层都实现了上述目标，就可以通过组合各网络层量化结果的方式来获得一个完整的量化模型。

为解决这一分层压缩问题，论文作者采用了最优脑量化 (Optimal Brain Quantization, OBQ) 框架 ([Frantar et al 2022](https://arxiv.org/abs/2208.11580)) 。OBQ 方法的出发点在于其观察到：以上等式可以改写成权重矩阵 \\(W_{l}\\) 每一行的平方误差之和

\\( \sum_{i=0}^{d_{row}} \|W_{l[i,:]}X-\hat{W}_{l[i,:]}X\|^{2}_{2} \\)

这意味着我们可以独立地对每一行执行量化。即所谓的 per-channel quantization。对每一行 \\(W_{l[i,:]}\\)，OBQ 在每一时刻只量化一个权重，同时更新所有未被量化的权重，以补偿量化单个权重所带来的误差。所选权重的更新采用一个闭环公式，并利用了海森矩阵 (Hessian Matrices)。

GPTQ 论文通过引入一系列优化措施来改进上述量化框架，在降低量化算法复杂度的同时保留了模型的精度。

相较于 OBQ，GPTQ 的量化步骤本身也更快：OBQ 需要花费 2 个 GPU 时来完成 BERT 模型 (336M) 的量化，而使用 GPTQ，量化一个 Bloom 模型 (176B) 则只需不到 4 个 GPU 时。

为了解算法的更多细节以及在困惑度 (perplexity, PPL) 指标和推理速度上的不同测评数据，可查阅原始 [论文](https://arxiv.org/pdf/2210.17323.pdf) 。

## AutoGPTQ 代码库——一站式地将 GPTQ 方法应用于大语言模型

AutoGPTQ 代码库让用户能够使用 GPTQ 方法量化 🤗 Transformers 中支持的大量模型，而社区中的其他平行工作如 [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) 、[Exllama](https://github.com/turboderp/exllama) 和 [llama.cpp](https://github.com/ggerganov/llama.cpp/) 则主要针对 Llama 模型架构实现量化策略。相较之下，AutoGPTQ 因其对丰富的 transformers 架构的平滑覆盖而广受欢迎。

正因为 AutoGPTQ 代码库覆盖了大量的 transformers 模型，我们决定提供一个 🤗 Transformers 的 API 集成，让每个人都能够更容易地使用大语言模型量化技术。截止目前，我们已经集成了包括 CUDA 算子在内的最常用的优化选项。对于更多高级选项如使用 Triton 算子和（或）兼容注意力的算子融合，请查看 [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 代码库。

## 🤗 Transformers 对 GPTQ 模型的本地化支持

在 [安装 AutoGPTQ 代码库](https://github.com/PanQiWei/AutoGPTQ#quick-installation) 和 `optimum` (`pip install optimum`) 之后，在 Transformers 中运行 GPTQ 模型将非常简单：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", torch_dtype=torch.float16, device_map="auto")
```

请查阅 Transformers 的 [说明文档](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) 以了解有关所有特性的更多信息。

我们的 AutoGPTQ 集成有以下诸多优点：

- 量化模型可被序列化并在 Hugging Face Hub 上分享。
- GPTQ 方法大大降低运行大语言模型所需的内存，同时保持着与 FP16 相当的推理速度。
- AutoGPTQ 在更广泛的 transformers 架构上支持 Exllama 算子。
- 该集成带有基于 RoCm 的 AMD GPU 的本地化支持。
- 能够 [**使用 PEFT 微调量化后的模型**](#--使用-peft-微调量化后的模型--) 。

你可以在 Hugging Face Hub 上查看你所喜爱的模型是否已经拥有 GPTQ 量化版本。TheBloke，Hugging Face 的顶级贡献者之一，已经使用 AutoGPTQ 量化了大量的模型并分享在 Hugging Face Hub 上。在我们的共同努力下，这些模型仓库都将可以与我们的集成一起开箱即用。


以下是一个使用 batch size = 1 的测评结果示例。该测评结果通过在英伟达 A100-SXM4-80GB GPU 上运行得到。我们使用长度为 512 个词元的提示文本，并精确地生成 512 个新词元。表格的第一行展示的是未量化的 `fp16` 基线，另外两行则展示使用 AutoGPTQ 不同算子的内存开销和推理性能。

| gptq  | act_order | bits | group_size | kernel            | Load time (s) | Per-token latency (ms) | Throughput (tokens/s) | Peak memory (MB) |
|-------|-----------|------|------------|-------------------|---------------|------------------------|-----------------------|------------------|
| False | None      | None | None       | None              | 26.0          | 36.958                 | 27.058                | 29152.98         |
| True  | False     | 4    | 128        | exllama           | 36.2          | 33.711                 | 29.663                | 10484.34         |
| True  | False     | 4    | 128        | autogptq-cuda-old | 36.2          | 46.44                  | 21.53                 | 10344.62         |

一个更全面的、可复现的测评结果可以在[这里](https://github.com/huggingface/optimum/tree/main/tests/benchmark#gptq-benchmark) 取得。


## 使用 **Optimum 代码库** 量化模型

为了将 AutoGPTQ 无缝集成到 Transformers 中，我们使用了 AutoGPTQ API 的一个极简版本，其可在 [Optimum](https://github.com/huggingface/optimum) 中获得 —— 这是 Hugging Face 针对训练和推理优化而开发的一个工具包。通过这种方式，我们轻松地实现了与 Transformers 的集成，同时，如果人们想要量化他们自己的模型，他们也完全可以单独使用 Optimum 的 API！如果想要量化你自己的大语言模型，请查阅 Optimum 的 [说明文档](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization) 。

只需数行代码，即可使用 GPTQ 方法量化 🤗 Transformers 的模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
```

量化一个模型可能花费较长的时间。对于一个 175B 参数量的模型，如果使用一个大型校准数据集（如 “c4”），至少需要 4 个 GPU 时。正如上面提到的那样，许多 GPTQ 模型已经可以在 Hugging Face Hub 上被取得，这让你在大多数情况下无需自行量化模型。当然，你仍可以使用你所专注的特定领域的数据集来量化模型。

## 通过 ***Text-Generation-Inference*** 使用 GPTQ 模型

在将 GPTQ 集成到 Transformers 中的同时，[Text-Generation-Inference 代码库](https://github.com/huggingface/text-generation-inference) (TGI) 已经添加了 GPTQ 的支持，旨在为生产中的大语言模型提供服务。现在，GPTQ 已经可以与动态批处理、paged attention、flash attention 等特性一起被应用于 [广泛的 transformers 模型架构](https://huggingface.co/docs/text-generation-inference/main/en/supported_models) 。

例如，这一集成允许在单个 A100-80GB GPU上服务 70B 模型！而这在使用 fp16 的模型权重时是不可能的，因为它超出了最大可用的 GPU 内存。

你可以在 TGI 的 [说明文档](https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/preparing_model#quantization) 中找到更多有关 GPTQ 的用法。

需要注意的时，TGI 中集成的算子不能很好地扩展到较大的批处理大小。因此，这一方式虽然节省了内存，但在较大的批处理大小上发生速度的下降是符合预期的。

## **使用 PEFT 微调量化后的模型**

在常规的方法下，你无法进一步微调量化后的模型。然而，通过使用 PEFT 代码库，你可以在量化后的模型之上训练适应性网络！为实现这一目标，我们冻结了量化过的基座模型的所有网络层，并额外添加可训练的适应性网络。这里是一些关于如何使用 PEFT 训练 GPTQ 模型的例子：[Colab 笔记本](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) 和 [微调脚本](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) 。

## 改进空间

虽然我们的 AutoGPTQ 集成在极小的预测质量损失代价下，带来了引人瞩目的优势。但在量化技术应用和算子实现方面仍有提升的空间。

首先，尽管 AutoGPTQ （在我们的认知范围内）已经集成了 [exllama](https://github.com/turboderp/exllama) 中所实现的最佳性能的 W4A16 算子（权重为 int4 数值类型，激活值为 fp16 数值类型），其仍有很大的改进空间。来自 [Kim 等人](https://arxiv.org/pdf/2211.10017.pdf) 的实现和 [MIT Han Lab](https://github.com/mit-han-lab/llm-awq) 的方法似乎十分可靠。此外，根据我们的内部测评，似乎暂未有开源的高性能的 Triton 版本的 W4A16 算子实现，这也是一个值得探索的方向。

在量化层面，我们需要再次强调 GPTQ 方法只对模型权重进行量化。而针对大语言模型的量化，存在其他的方法，提供了以较小的预测质量损失为代价，同时量化权重和激活值的方案。如 [LLM-QAT](https://arxiv.org/pdf/2305.17888.pdf) 采用 int4/int8 的混合精度方案，同时还对 KV Cache 施行量化。这一技术的强大优点是能实际使用整数运算算法来进行计算，一个例子是 [英伟达的张量核心支持 int8 计算](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) 。然而，据我们所知，目前暂无开源的 W4A8 量化算子，但这可能是一个 [值得探索的方向](https://www.qualcomm.com/news/onq/2023/04/floating-point-arithmetic-for-ai-inference-hit-or-miss) 。

在算子层面，为更大的批处理大小设计高性能的 W4A16 算子仍然是一大挑战。

### 已支持的模型

在初始实现中，暂时只支持纯编码器或纯解码器架构的大语言模型。这听起来似乎有较大的局限性，但其实已经涵盖了当前绝大多数最先进的大语言模型，如 Llama、OPT、GPT-Neo、GPT-NeoX 等。

大型的视觉、语音和多模态模型在现阶段暂不被支持。

## 结论和结语

本文中，我们介绍了 Transformers 对 [AutoGPTQ 代码库](https://github.com/PanQiWei/AutoGPTQ) 的集成，使得社区中的任何人都可以更方便地利用 GPTQ 方法量化大语言模型，助力令人激动的大语言模型工具和应用的构建。

这一集成支持英伟达 GPU 和基于 RoCm 的 AMD GPU，这是向支持更广泛 GPU 架构的量化模型的普惠化迈出的一大步。

与 AutoGPTQ 团队的合作非常富有成效，我们非常感谢他们的支持和他们在该代码库上的工作。

我们希望本次集成将使每个人都更容易地在他们的应用程序中使用大语言模型，我们迫不及待地想要看到大家即将使用它所创造出的一切！

再次提醒不要错过文章开头分享的有用资源，以便更好地理解本次集成的特性以及如何快速开始使用 GPTQ 量化。

- [原始论文](https://arxiv.org/pdf/2210.17323.pdf)
- [运行于 Google Colab 笔记本上的基础用例](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) —— 该笔记本上的用例展示了如何使用 GPTQ 方法量化你的 transformers 模型、如何进行量化模型的推理，以及如何使用量化后的模型进行微调。
- Transformers 中集成 GPTQ 的 [说明文档](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)
- Optimum 中集成 GPTQ 的 [说明文档](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
- TheBloke [模型仓库](https://huggingface.co/TheBloke?sort_models=likes#models) 中的 GPTQ 模型。


## 致谢

感谢 [潘其威](https://github.com/PanQiWei) 对杰出的 AutoGPTQ 代码库的支持和所作的工作，以及他对本次集成的帮助。
感谢 [TheBloke](https://huggingface.co/TheBloke) 使用 AutoGPTQ 量化大量的模型并分享在 Hugging Face Hub 上，以及他在本次集成中所提供的帮助。
感谢 [qwopqwop200](https://github.com/qwopqwop200) 对 AutoGPTQ 代码库的持续贡献，目前，他正致力于将该代码库的使用场景拓展至 CPU ，这一特性将在 AutoGPTQ 的下一版本中发布。

最后，我们还要感谢 [Pedro Cuenca](https://github.com/pcuenca) 对本文的撰写所提供的帮助。