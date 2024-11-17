---
title: "通用辅助生成：使用任意辅助模型加速解码"
thumbnail: /blog/assets/optimum_intel/intel_thumbnail.png
authors:
- user: danielkorat
  guest: true
  org: Intel
- user: orenpereg
  guest: true
  org: Intel
- user: mber
  guest: true
  org: Intel
- user: jmamou
  guest: true
  org: Intel
- user: joaogante
- user: lewtun
- user: Nadav-Timor
  guest: true
  org: weizmannscience
- user: moshew
  guest: true
  org: Intel
translators:
- user: MatrixYao
---

# 通用辅助生成：使用任意辅助模型加速解码

<em>太长不看版</em>：许多 LLM（如 `gemma-2-9b`、`Mixtral-8x22B-Instruct-v0.1`等）苦于缺乏对应小模型，而无法适用[辅助生成](https://huggingface.co/blog/zh/assisted-generation)方案。本文，我们将介绍由英特尔研究院和 Hugging Face 合作开发的*通用辅助生成*技术。有了这项技术，LLM 可与**任意** SLM 搭配组成辅助生成方案。从而，我们可以用辅助生成技术加速*任意*解码器模型或[混合专家](https://huggingface.co/blog/zh/moe)模型以获得 **1.5x-2.0x** 的加速比。重要的是，开销几乎为零 🔥🔥 🔥！一起了解一下吧！

## 引言

如今，风头最劲的开放权重 LLM 参数量一般都有数十亿到数千亿（说你呢 Llama-3.1-405B 👋），这给在生产环境中部署这些饿兽带来了一系列工程挑战。挑战之一就是：大模型文本生成速度很慢。为此，社区开发了很多不同的技术来加速解码过程。辅助生成，也称为[投机解码](https://arxiv.org/abs/2211.17192)，是其中一种非常常用且实用的方法，可在不损失准确性的情况下加速 LLM 推理。本文，我们将了解辅助生成的工作原理，并分享我们的最新研究成果，该成果使得对 Hugging Face Hub [14 万个语言模型](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)中的*任意一个*模型进行加速成为可能，🚀！ 

## 辅助生成

辅助生成的核心是一对模型，分别称为*目标模型*和*辅助模型*，其中辅助模型是目标模型的小版，举个例子，你可以使用 [`Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) 作为较大的 [`Llama-3.1-70b`](https://huggingface.co/meta-llama/Llama-3.1-70b) 目标模型的辅助模型。整个生成过程是一个迭代过程：每一轮，辅助模型会先一个一个自回归地生成多个词元；接着，目标模型通过一次前向传播验证辅助模型本轮生成的所有词元。加速的奥秘就在于目标模型每次前向传播中可以验证多个词元，而不像原本每次只能生成一个词元。更详细的解释，请参阅[原博文](https://huggingface.co/blog/zh/assisted-generation)。结合新近推出的[动态投机](https://huggingface.co/blog/dynamic_speculation_lookahead) 策略，辅助生成可将文本生成速度提高 1.5 至 3 倍，具体倍数取决于任务类型及所使用的模型。

但，辅助生成并非无往而不利，一个最明显的问题就是：其要求目标模型和辅助模型必须使用相同的分词器，这意味着两者必须来自同一个模型系列。然而，许多广泛使用的模型缺乏合适的“矮小紧”模型，因此与如此大幅的延迟降低无缘。根据我们的经验，一般来说，辅助模型需要至少比目标模型小 50-100 倍，才会看到有意义的加速。举几个例子，[`CodeLlama-13b`](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf) 没有小模型；[`gemma-2-9b`](https://huggingface.co/google/gemma-2-9b) 只有一个 `2b` 的小模型，显然不够小、不够快，因此加速注定不会太明显。

## 通用辅助生成

为了缓解这个痛点，英特尔研究院与 Hugging Face 合作开发了通用辅助生成（Universal Assisted Generation，UAG）技术。UAG 可以无视分词器的差异，配对任意目标模型和辅助模型。例如，可以使用 `gemma-2-9b` 作为目标模型，并选取 [`vicuna-68m`](https://huggingface.co/double7/vicuna-68m) 作为辅助模型。

该技术背后的主要思想是双路分词器映射：每一轮，辅助模型生成完词元后，就将其输出词元序列解码为文本，再使用目标模型的分词器将文本编码成词元序列；同样地，在目标模型验证完后，将目标模型的词元序列用相同的方法转换回辅助模型的词元序列，再将其添加至辅助模型的上下文用于下一轮迭代。

由于辅助模型和目标模型的分词器的词汇表不同，因此还需要处理由此带来的差异。为了准确地对辅助模型新生成的词元序列进行重编码，必须再多给它一些上文词元。然后，将整个序列重新编码为目标模型的词元格式，并与之前生成的最新的目标词元对齐，以锚定新生成词元的确切位置。下面的视频对此过程进行了图解。

<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 80%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/universal-assisted-generation/method-animation.mov"
    ></video>
</figure>

从目标模型到辅助模型的词元重编码也遵循与上述视频类似的过程。此时，如遇不匹配的词元，需从辅助模型的键值 (KV) 缓存中将它们丢弃掉，以保证数据的完整性。

## 基准测试

下表展示了不同目标模型与异分词器辅助模型形成辅助解码方案时测得的延迟改进。

| 目标模型 | 辅助模型 | 数据集 | 任务 | 加速比 |
|----------------------|---------------------|---------------------------|---------------------------|---------------------------|
| `codellama/CodeLlama-13b-Instruct-hf` | `bigcode/tiny_starcoder_py` | [`openai/humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) | 代码生成 | **1.90x** |
| [`mistralai/Mixtral-8x22B-Instruct-v0.1`](mistralai/Mixtral-8x22B-Instruct-v0.1) | `double7/vicuna-68m`  | [`cnn_dailymail`](https://huggingface.co/datasets/cnn_dailymail)   | 摘要 | **1.52x** |
| `google/gemma-2-9b` | `double7/vicuna-68m`  | [`cnn_dailymail`](https://huggingface.co/datasets/cnn_dailymail)   | 摘要 | **1.76x** |
| `mistralai/Mixtral-8x22B-Instruct-v0.1` | `Qwen/Qwen2-0.5B-Instruct`  | [`tau/scrolls`](https://huggingface.co/datasets/tau/scrolls)   | 长文摘要 | **1.78x** |
| `meta-llama/Llama-3.1-70B` | `Qwen/Qwen2-0.5B-Instruct`  | [`tau/scrolls`](https://huggingface.co/datasets/tau/scrolls)   | 长文摘要 | **1.78x** |
| `microsoft/Phi-3-medium-128k-instruct` | `Qwen/Qwen2-0.5B-Instruct`  | [`tau/scrolls`](https://huggingface.co/datasets/tau/scrolls)   | 长文摘要 | **1.91x** |

请注意，在标准辅助解码方案下，上表中所有目标模型都会苦于没有合适的小模型（低于 10 亿参数）。

上述实验均在 100 个随机样本上完成。`Llama` 和 `Mixtral` 目标模型的实验分别用了 2 张和 4 张 A100 GPU；其他所有实验均使用单张 A6000 GPU。

## 代码

通用辅助生成技术已集成至 🤗 Transformers [4.46.0](https://github.com/huggingface/transformers/releases/tag/v4.46.0) 版。

要使能该技术，需将 `tokenizer` 和 `assistant_tokenizer` 传递给 `generate()`，示例代码如下：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = "Alice and Bob"
checkpoint = "google/gemma-2-9b"
assistant_checkpoint = "double7/vicuna-68m"

assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(checkpoint)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
outputs = model.generate(**inputs, assistant_model=assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```
输出如下：
```
['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']
```

## 下一步

标准辅助生成方案在 `do_sample=True` 时，使用的投机采样算法为[该论文的算法 1](https://arxiv.org/pdf/2211.17192.pdf)，但 UAG 
目前仅实现了多项分布采样。在多项分布采样中，如果目标模型与辅助模型采样得的词元不相同时，会自动拒绝该词元，这与投机采样对此情况的处理不同。在实践中，这意味着与共享分词器的标准方案相比，UAG 方案在 `do_sample=True` 时吞吐量会较低。将来，我们计划增加对 UAG 投机采样的支持。

此外，我们还打算将 UAG 集成到 🤗 Transformers 流水线中，以使用户能够更简单、轻松地利用它。

## 参考资源

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)
- [辅助生成: 低延迟文本生成的新方向](https://huggingface.co/blog/zh/assisted-generation)

> 英文原文: <url> https://huggingface.co/blog/universal_assisted_generation </url>
> 原文作者：Daniel Korat，Oren Pereg，Moshe Berchansky，Jonathan Mamou，Joao Gante，Lewis Tunstall，Nadav Timor，Moshe Wasserblat
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。
