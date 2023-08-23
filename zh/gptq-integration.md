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
---

# 使用 AutoGPTQ 和 transformers 让大语言模型更轻量化

大语言模型在理解和生成人类水平的文字方面所展现出的非凡能力，正在许多领域带来应用上的革新。然而，在消费级硬件上训练和部署大语言模型的需求也变得越来越难以满足。

🤗 Hugging Face 的核心使命是 _让优秀的机器学习大众化_ (TODO: 是否有官方正式的中译版本？)，而这正包括了尽可能地让所有人都能够使用上大模型。本着[与 bitsandbytes 合作](https://huggingface.co/blog/4bit-transformers-bitsandbytes) 一样的精神，我们将 [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 代码库集成到了 Transformers 中，让用户使用 GPTQ 算法 ([Frantar et al. 2023](https://arxiv.org/pdf/2210.17323.pdf)) 在8位、4位、3位，甚至是2位精度下量化和运行模型成为可能。当使用4位量化时，精度的下降可以忽略不计，同时在小批量推理上保持着与 `fp16` 基线相当的速度。 需要注意的是，GPTQ 方法与 bitsandbytes 提出的训练后量化方法有所不同：它需要在量化阶段提供一个校准数据集。

这一集成支持英伟达 GPU 和基于 RoCm 的 AMD GPU。

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

本文及有关版本发布提供了一些资源来帮助用户开启 GPTQ 量化的旅程：

- [原始论文](https://arxiv.org/pdf/2210.17323.pdf)
- [运行于 Google Colab 笔记本上的基础用例](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) —— 该笔记本上的用例展示了如何使用 GPTQ 方法量化你的 transformers 模型、如何进行量化模型的推理，以及如何使用量化后的模型进行微调。
- Transformers 中集成 GPTQ 的[说明文档](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)
- Optimum 中集成 GPTQ 的[说明文档](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
- TheBloke [模型仓库](https://huggingface.co/TheBloke?sort_models=likes#models) 中的 GPTQ 模型。


## **GPTQ 论文总结**

通常，量化方法可以分为以下两类：

1. 训练后量化 (Post Training Quantization, PTQ)：适度地使用一些资源来量化预训练好的模型，如一个校准数据集和几小时的算力。
2. 量化感知训练 (Quantization Aware Training, QAT)：在训练或进一步微调之前执行量化。

GPTQ 属于训练后量化，这对于大模型而言格外有意义(TODO: 还是遵循原文的 interesting 翻译成有趣？)，因为对其进行全参数的训练甚至仅仅是微调都十分昂贵。

具体而言，GPTQ 采用 int4/fp16 (W4A16) 的混合量化方案，其中模型权重被量化为 int4 数值类型，而激活值则保留在 float16。在推理阶段，模型权重被动态地反量化回 float16 并在该数值类型下进行实际的运算。

该方案有以下两方面的优点：

- int4 量化能够节省接近4倍的内存，这是因为反量化操作发生在算子的计算单元附近，而不是在 GPU 的全局内存中。
- 由于用于权重的位宽较低，因此可以节省数据通信的时间，从而潜在地提升了推理速度。

GPTQ 论文解决了分层压缩的问题：

给定一个拥有权重矩阵 $W_{l}$ 和输入 $X_{l}$ 的网络层 $l$，我们期望获得一个量化版本的权重矩阵 $\hat{W}_{l}$ 以最小化均方误差（MSE）：

\\({\hat{W}_{l}}^{*} = argmin_{\hat{W_{l}}} \|W_{l}X-\hat{W}_{l}X\|^{2}_{2})

一旦每层实现了上述目标，就可以通过组合各网络层量化方案的方式来获得全模型的量化方案。

为解决这一分层压缩问题，论文作者采用了最优脑量化 (Optimal Brain Quantization, OBQ) 框架 ([Frantar et al 2022](https://arxiv.org/abs/2208.11580)) 。OBQ 方法的出发点在于其观察到：以上等式可以改写成权重矩阵 $W_{l}$ 每一行的平方误差之和

\\( \sum_{i=0}^{d_{row}} \|W_{l[i,:]}X-\hat{W}_{l[i,:]}X\|^{2}_{2} )

这意味着我们可以独立地对每一行执行量化。即所谓的 per-channel quantization。对每一行 $W_{l[i,:]}$，OBQ 在每一时刻只量化一个权重，同时更新所有未被量化的权重，以补偿量化单个权重所带来的误差。所选权重的更新采用一个闭环公式，并利用了海森矩阵 (Hessian Matrices)。

GPTQ 论文通过引入一系列优化措施来改进上述量化框架，在降低量化算法复杂度的同时保留了模型的精度。

相较于 OBQ，GPTQ 的量化步骤本身也更快：OBQ 需要花费2个 GPU 时来完成 BERT 模型 (336M) 的量化，而使用 GPTQ，量化一个 Bloom 模型 (176B) 则只需不到4个 GPU 时。

为了解算法的更多细节以及在困惑度 (perplexity, PPL) 指标和推理速度上的不同测评数据，可查阅原始[论文](https://arxiv.org/pdf/2210.17323.pdf) 。

## AutoGPTQ 代码库——一站式地将 GPTQ 方法应用于大语言模型

AutoGPTQ 代码库使得用户能够使用 GPTQ 方法量化 🤗 Transformers 中支持的大量模型，而社区中的其他平行工作如 [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) 、[Exllama](https://github.com/turboderp/exllama) 和 [llama.cpp](https://github.com/ggerganov/llama.cpp/) 则主要针对 Llama 模型架构实现量化策略。相较之下，AutoGPTQ 因其对丰富的 transformers 架构的平滑覆盖而广受欢迎。

正因为 AutoGPTQ 代码库覆盖了大量的 transformers 模型，我们决定提供一个 🤗 Transformers 的 API 集成，让每个人都能够更容易地接触到大语言模型量化。截止目前，我们已经集成了包括 CUDA 算子在内的最常用的优化选项。对于更多高级选项如 Triton 算子和（或）兼容注意力的算子融合，请查看 [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 代码库。

## 🤗 Transformers 对 GPTQ 模型的本地化支持

在[安装 AutoGPTQ 代码库](https://github.com/PanQiWei/AutoGPTQ#quick-installation) 和 `optimum` (`pip install optimum`) 之后，在 Transformers 中运行 GPTQ 模型将非常简单：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", torch_dtype=torch.float16, device_map="auto")
```

请查阅 Transformers 的[说明文档](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) 以了解有关所有特性的更多信息。

我们的 AutoGPTQ 集成有以下诸多优点：

- 量化模型可被序列化并在 Hugging Face Hub 上分享。
- GPTQ 方法大大降低运行大语言模型所需的内存，同时保持着与 FP16 相当的推理延迟。
- AutoGPTQ 在更广泛的 transformers 架构上支持 Exllama 算子。
- 该集成带有基于 RoCm 的 AMD GPU 的本地化支持。
- 能够[**使用 PEFT 微调量化后的模型**](#--使用-peft-微调量化后的模型--) 。

你可以在 Hugging Face Hub 上查找你所喜爱的模型是否已经拥有 GPTQ 量化版本。TheBloke，Hugging Face 的顶级贡献者之一，已经使用 AutoGPTQ 量化了大量的模型并分享在 Hugging Face Hub 上。在我们的共同努力下，这些模型仓库都将可以与我们的集成一起开箱即用。


以下是一个使用 batch size = 1 的测评结果示例。该测评结果通过在英伟达 A100-SXM4-80GB GPU 上运行得到。我们使用长度为512个词元的提示文本，并精确地生成512个新词元。表格的第一行展示的是未量化的 `fp16` 基线，另外两行则展示使用 AutoGPTQ 不同算子的内存开销和性能。

| gptq  | act_order | bits | group_size | kernel            | Load time (s) | Per-token latency (ms) | Throughput (tokens/s) | Peak memory (MB) |
|-------|-----------|------|------------|-------------------|---------------|------------------------|-----------------------|------------------|
| False | None      | None | None       | None              | 26.0          | 36.958                 | 27.058                | 29152.98         |
| True  | False     | 4    | 128        | exllama           | 36.2          | 33.711                 | 29.663                | 10484.34         |
| True  | False     | 4    | 128        | autogptq-cuda-old | 36.2          | 46.44                  | 21.53                 | 10344.62         |

一个更全面的、可复现的测评结果可以在[这里](https://github.com/huggingface/optimum/tree/main/tests/benchmark#gptq-benchmark) 取得。


## 使用 **Optimum 代码库** 量化模型

To seamlessly integrate AutoGPTQ into Transformers, we used a minimalist version of the AutoGPTQ API that is available 
in [Optimum](https://github.com/huggingface/optimum), Hugging Face's toolkit for training and inference optimization. 
By following this approach, we achieved easy integration with Transformers, while allowing people to use the Optimum API 
if they want to quantize their own models! Check out the Optimum [documentation](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization) 
if you want to quantize your own LLMs. 

Quantizing 🤗 Transformers models with the GPTQ method can be done in a few lines:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
```

Quantizing a model may take a long time. Note that for a 175B model, at least 4 GPU-hours are required if one uses a large dataset (e.g. `"c4"``). As mentioned above, many GPTQ models are already available on the Hugging Face Hub, which bypasses the need to quantize a model yourself in most use cases. Nevertheless, you can also quantize a model using your own dataset appropriate for the particular domain you are working on.

## 通过 ***Text-Generation-Inference*** 使用 GPTQ 模型

In parallel to the integration of GPTQ in Transformers, GPTQ support was added to the [Text-Generation-Inference library](https://github.com/huggingface/text-generation-inference) (TGI), aimed at serving large language models in production. GPTQ can now be used alongside features such as dynamic batching, paged attention and flash attention for a [wide range of architectures](https://huggingface.co/docs/text-generation-inference/main/en/supported_models).

As an example, this integration allows to serve a 70B model on a single A100-80GB GPU! This is not possible using a fp16 checkpoint as it exceeds the available GPU memory.

You can find out more about the usage of GPTQ in TGI in [the documentation](https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/preparing_model#quantization).

Note that the kernel integrated in TGI does not scale very well with larger batch sizes. Although this approach saves memory, slowdowns are expected at larger batch sizes.

## **使用 PEFT 微调量化后的模型**

You can not further train a quantized model using the regular methods. However, by leveraging the PEFT library, you can train adapters on top! To do that, we freeze all the layers of the quantized model and add the trainable adapters. Here are some examples on how to use PEFT with a GPTQ model: [colab notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) and [finetuning](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) script. 

## 改进空间

Our AutoGPTQ integration already brings impressive benefits at a small cost in the quality of prediction. There is still room for improvement, both in the quantization techniques and the kernel implementations.

First, while AutoGPTQ integrates (to the best of our knowledge) with the most performant W4A16 kernel (weights as int4, activations as fp16) from the [exllama implementation](https://github.com/turboderp/exllama), there is a good chance that the kernel can still be improved. There have been other promising implementations [from Kim et al.](https://arxiv.org/pdf/2211.10017.pdf) and from [MIT Han Lab](https://github.com/mit-han-lab/llm-awq) that appear to be promising. Moreover, from internal benchmarks, there appears to still be no open-source performant W4A16 kernel written in Triton, which could be a direction to explore.

On the quantization side, let’s emphasize again that this method only quantizes the weights. There have been other approaches proposed for LLM quantization that can quantize both weights and activations at a small cost in prediction quality, such as [LLM-QAT](https://arxiv.org/pdf/2305.17888.pdf) where a mixed int4/int8 scheme can be used, as well as quantization of the key-value cache. One of the strong advantages of this technique is the ability to use actual integer arithmetic for the compute, with e.g. [Nvidia Tensor Cores supporting int8 compute](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf). However, to the best of our knowledge, there are no open-source W4A8 quantization kernels available, but this may well be [an interesting direction to explore](https://www.qualcomm.com/news/onq/2023/04/floating-point-arithmetic-for-ai-inference-hit-or-miss).

On the kernel side as well, designing performant W4A16 kernels for larger batch sizes remains an open challenge.

### 已支持的模型

In this initial implementation, only large language models with a decoder or encoder only architecture are supported. This may sound a bit restrictive, but it encompasses most state of the art LLMs such as Llama, OPT, GPT-Neo, GPT-NeoX.

Very large vision, audio, and multi-modal models are currently not supported.

## 结论和结语

In this blogpost we have presented the integration of the [AutoGPTQ library](https://github.com/PanQiWei/AutoGPTQ) in Transformers, making it possible to quantize LLMs with the GPTQ method to make them more accessible for anyone in the community and empower them to build exciting tools and applications with LLMs. 

This integration is available both for Nvidia GPUs, and RoCm-powered AMD GPUs, which is a huge step towards democratizing quantized models for broader GPU architectures.

The collaboration with the AutoGPTQ team has been very fruitful, and we are very grateful for their support and their work on this library.

We hope that this integration will make it easier for everyone to use LLMs in their applications, and we are looking forward to seeing what you will build with it!

Do not miss the useful resources shared above for better understanding the integration and how to quickly get started with GPTQ quantization.

- [Original Paper](https://arxiv.org/pdf/2210.17323.pdf)
- [Basic usage Google Colab notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) -  This notebook shows how to quantize your transformers model with GPTQ method, how to do inference, and how to do fine-tuning with the quantized model.
- Transformers integration [documentation](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)
- Optimum integration [documentation](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
- The Bloke [repositories](https://huggingface.co/TheBloke?sort_models=likes#models) with compatible GPTQ models.


## 致谢

We would like to thank [William](https://github.com/PanQiWei) for his support and his work on the amazing AutoGPTQ library and for his help in the integration. 
We would also like to thank [TheBloke](https://huggingface.co/TheBloke) for his work on quantizing many models with AutoGPTQ and sharing them on the Hub and for his help with the integration. 
We would also like to aknowledge [qwopqwop200](https://github.com/qwopqwop200) for his continuous contributions on AutoGPTQ library and his work on extending the library for CPU that is going to be released in the next versions of AutoGPTQ. 

Finally, we would like to thank [Pedro Cuenca](https://github.com/pcuenca) for his help with the writing of this blogpost.