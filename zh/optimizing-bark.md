---
title: "使用 🤗 Transformers 优化 Bark" 
thumbnail: /blog/assets/bark_optimization/thumbnail.png
authors:
- user: ylacombe
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 使用 🤗 Transformers 优化文本转语音模型 Bark


<a target="_blank" href="https://colab.research.google.com/github/ylacombe/notebooks/blob/main/Benchmark_Bark_HuggingFace.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"/>
</a>

🤗 Transformers 提供了许多最新最先进 (state-of-the-art，SoTA) 的模型，这些模型横跨多个领域及任务。为了使这些模型能以最佳性能运行，我们需要优化其推理速度及内存使用。

🤗 Hugging Face 生态系统为满足上述需求提供了现成且易于使用的优化工具，这些工具可应用于库中的所有模型。用户只需添加几行代码就可以轻松 **减少内存占用** 并 **提高推理速度**。

在本实战教程中，我将演示如何用三个简单的优化技巧来优化 [Bark](https://huggingface.co/docs/transformers/main/en/model_doc/bark#overview) 模型。Bark 是🤗 Transformers 支持的一个文本转语音 (Text-To-Speech，TTS) 模型。所有优化仅依赖于 [Transformers](https://github.com/huggingface/transformers)、[Optimum](https://github.com/huggingface/optimum) 以及 [Accelerate](https://github.com/huggingface/accelerate) 这三个 🤗 生态系统库。

本教程还演示了如何对模型及其不同的优化方案进行性能基准测试。

本文对应的 Google Colab 在 [此](https://colab.research.google.com/github/ylacombe/notebooks/blob/main/Benchmark_Bark_HuggingFace.ipynb)。

本文结构如下:

## 目录

1. Bark 模型 [简介](#bark-模型架构)
2. 不同优化技巧及其优点 [概述](#优化技术)
3. 基准测试结果 [展示](#基准测试结果)

# Bark 模型架构

**Bark** 是 Suno AI 提出的基于 transformer 的 TTS 模型，其原始代码库为 [suno-ai/bark](https://github.com/suno-ai/bark)。该模型能够生成各种音频输出，包括语音、音乐、背景噪音以及简单的音效。此外，它还可以产生非语言语音，如笑声、叹息声和抽泣声等。

自 v4.31.0 起，Bark 已集成入 🤗 Transformers！

你可以通过 [这个 notebook](https://colab.research.google.com/github/ylacombe/notebooks/blob/main/Bark_HuggingFace_Demo.ipynb) 试试 Bark 并探索其功能。

Bark 主要由 4 个模型组成:

- `BarkSemanticModel` (也称为 **文本** 模型): 一个因果自回归 transformer 模型，其输入为分词后的词元序列，并输出能捕获文义的语义词元。
- `BarkCoarseModel` (也称为 **粗声学** 模型): 一个因果自回归 transformer 模型，其接收 `BarkSemanticModel` 模型的输出，并据此预测 EnCodec 所需的前两个音频码本。
- `BarkFineModel` (也称为 **细声学** 模型)，这次是个非因果自编码器 transformer 模型，它对 _先前码本的嵌入和_ 进行迭代，从而生成最后一个码本。
- 在 [`EncodecModel`](https://huggingface.co/docs/transformers/v4.31.0/model_doc/encodec) 的编码器部分预测出所有码本通道后，Bark 继续用其解码器来解码并输出音频序列。

截至本文撰写时，共有两个 Bark checkpoint 可用，其中一个是 [小版](https://huggingface.co/suno/bark-small)，一个是 [大版](https://huggingface.co/suno/bark)。

## 加载模型及其处理器

预训练的 Bark [小 checkpoint](https://huggingface.co/suno/bark-small) 和 [大 checkpoint]((https://huggingface.co/suno/bark)) 均可从 Hugging Face Hub 上加载。你可根据实际需要加载相应的 repo-id。

为了使实验运行起来快点，我们默认使用小 checkpoint，即 `“suno/bark-small”` 。但你可以随意改成 `“suno/bark”` 来尝试大 checkpoint。

```python
from transformers import BarkModel

model = BarkModel.from_pretrained("suno/bark-small")
```

将模型放到加速器上以优化其速度:

```python
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

加载处理器，它主要处理分词以及说话人嵌入 (若有)。

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("suno/bark-small")
```

# 优化技巧

本节，我们将探索如何使用 🤗 Optimum 和 🤗 Accelerate 库中的现成功能来以最少的代码改动达到优化 Bark 模型的目的。

## 设置实验环境

首先，我们准备一个输入文本并定义一个函数来测量 Bark 生成过程的延迟及其 GPU 显存占用情况。

```python
text_prompt = "Let's try generating speech, with Bark, a text-to-speech model"
inputs = processor(text_prompt).to(device)
```

测量延迟和 GPU 内存占用需要使用特定的 CUDA 函数。我们实现了一个工具函数，用于测量模型的推理延迟及 GPU 内存占用。为了确保结果的准确性，每次测量我们会运行 `nb_loops` 次求均值:

```python
import torch
from transformers import set_seed

def measure_latency_and_memory_use(model, inputs, nb_loops = 5):

  # define Events that measure start and end of the generate pass
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  # reset cuda memory stats and empty cache
  torch.cuda.reset_peak_memory_stats(device)
  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  # get the start time
  start_event.record()

  # actually generate
  for _ in range(nb_loops):
        # set seed for reproducibility
        set_seed(0)
        output = model.generate(**inputs, do_sample = True, fine_temperature = 0.4, coarse_temperature = 0.8)

  # get the end time
  end_event.record()
  torch.cuda.synchronize()

  # measure memory footprint and elapsed time
  max_memory = torch.cuda.max_memory_allocated(device)
  elapsed_time = start_event.elapsed_time(end_event)* 1.0e-3

  print('Execution time:', elapsed_time/nb_loops, 'seconds')
  print('Max memory footprint', max_memory*1e-9, ' GB')

  return output
```

## 基线

在优化之前，我们先测量下模型的基线性能并听一下生成的音频，我们测量五次并求均值:

```python

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**输出:**

```
Execution time: 9.3841625 seconds
Max memory footprint 1.914612224 GB
```

现在，我们可以播放一下输出音频:

```python
from IPython.display import Audio

# now, listen to the output
sampling_rate = model.generation_config.sample_rate
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)
```

按下面的播放键听一下吧 ([下载该音频文件](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_base.wav)):

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_base.wav" type="audio/wav">
当前浏览器不支持音频元素。
</audio>

### 重要说明

上例中运行次数较少。为了测量和后续对比的准确性，运行次数需要增加到至少 100。

增加 `nb_loops` 一个主要原因是，同一输入的多次运行所生成的语音长度差异也很大。因此当运行次数较少时，有可能通过 `measure_latency_and_memory_use` 测出的延迟并不能反映出优化方法的实际性能！文末的基准测试取的是 100 次运行的均值，用以逼近模型的真实性能。

## 1. 🤗 Better Transformer

Better Transformer 是  🤗 Optimum 的一个功能，它可以帮助在后台执行算子融合。这意味着模型的某些操作在 GPU 上的性能将会得到进一步优化，从而加速模型的最终运行速度。

再具体一点，🤗 Transformers 支持的大多数模型都依赖于注意力，这使得模型在生成输出时可以选择性地关注输入的某些部分，因而能够有效地处理远程依赖关系并捕获数据中复杂的上下文关系。

Dao 等人于 2022 年提出了一项名为 [Flash Attention](https://arxiv.org/abs/2205.14135) 的技术，极大地优化了朴素注意力的性能。

Flash Attention 是一种更快、更高效的注意力算法，它巧妙地结合了一些传统方法 (如平铺和重计算)，以最大限度地减少内存使用并提高速度。与之前的算法不同，Flash Attention 将内存使用量从与序列长度呈平方关系降低到线性关系，这对关注内存效率的应用尤其重要。

🤗 Better Transformer 可以开箱即用地支持 Flash Attention！只需一行代码即可将模型导出到 🤗 Better Transformer 并启用 Flash Attention:

```python
model =  model.to_bettertransformer()

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**输出:**

```
Execution time: 5.43284375 seconds
Max memory footprint 1.9151841280000002 GB
```

按下面的播放键听一下输出吧 ([下载该音频文件](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_bettertransformer.wav)):

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_bettertransformer.wav" type="audio/wav">
当前浏览器不支持音频元素。
</audio>

**利弊**

效果不会下降，这意味着你可以获得与基线版本完全相同的结果，同时提速 20% 到 30%！想要了解更多有关 Better Transformer 的详细信息，请参阅此 [博文](https://pytorch.org/blog/out-of-the-box-acceleration/)。

## 2. 半精度

大多数人工智能模型通常使用称为单精度浮点的存储格式，即 `fp32` ，这在实践中意味着每个数都用 32 比特来存储。

你也可以选择使用 16 比特对每个数进行编码，即所谓的半精度浮点，即 `fp16` (译者注: 或 `bf16` )，这时每个数占用的存储空间就变成了原来的一半！除此以外，你还可以获得计算上的加速！

但天下没有免费的午餐，半精度会带来较小的效果下降，因为模型内部的操作不如 `fp32` 精确了。

你可以通过简单地在 `BarkModel.from_pretrained(...)` 的入参中添加 `torch_dtype=torch.float16` 来将 Transformers 模型加载为半精度！

代码如下:

```python
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**输出:**

```
Execution time: 7.00045390625 seconds
Max memory footprint 2.7436124160000004 GB
```

照例，按下面的播放键听一下输出吧 ([下载该音频文件](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_fp16.wav)):

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_fp16.wav" type="audio/wav">
当前浏览器不支持音频元素。
</audio>

**利弊**

虽然效果略有下降，但内存占用量减少了 50%，速度提高了 5%。

## 3. CPU 卸载

正如本文第一部分所述，Bark 包含 4 个子模型，这些子模型在音频生成过程中按序调用。 **换句话说，当一个子模型正在使用时，其他子模型处于空闲状态。**

为什么要讨论这个问题呢？ 因为 GPU 显存在 AI 工作负载中非常宝贵，显存中的运算速度是最快的，而很多情况下显存不足是推理速度的瓶颈。

一个简单的解决方案是将空闲子模型从 GPU 显存中卸载至 CPU 内存，该操作称为 CPU 卸载。

**好消息: ** Bark 的 CPU 卸载已集成至 🤗 Transformers 中，只需一行代码即可使能。唯一条件是，仅需确保安装了 🤗 Accelerate 即可！

```python
model = BarkModel.from_pretrained("suno/bark-small")

# Enable CPU offload
model.enable_cpu_offload()

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**输出:**

```
Execution time: 8.97633828125 seconds
Max memory footprint 1.3231160320000002 GB
```

按下面的播放键听一下输出吧 ([下载该音频文件](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_cpu_offload.wav)):
<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_cpu_offload.wav" type="audio/wav">
当前浏览器不支持音频元素。
</audio>

**利弊**

速度略有下降 (10%)，换得内存占用的巨大降低 (60% 🤯)。

启用此功能后， `bark-large` 占用空间从原先的 5GB 降至 2GB，与 `bark-small` 的内存占用相同！

如果你还想要降更多的话，可以试试启用 `fp16` ，内存占用甚至可以降至 1GB。具体可以参见下一节的数据。

## 4. 组合优化

我们把上述所有优化组合到一起，这意味着你可以合并 CPU 卸载、半精度以及 🤗 Better Transformer 带来的收益！

```python
# load in fp16
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)

# convert to bettertransformer
model = BetterTransformer.transform(model, keep_original_model=False)

# enable CPU offload
model.enable_cpu_offload()

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**输出:**

```
Execution time: 7.4496484375000005 seconds
Max memory footprint 0.46871091200000004 GB
```

按下面的播放键听一下输出吧 ([下载该音频文件](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_optimized.wav)):

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_optimized.wav" type="audio/wav">
当前浏览器不支持音频元素。
</audio>

**利弊**

最终，你将获得 23% 的加速并节约 80% 的内存！

## 批处理

得陇望蜀？

加个批处理吧，上述 3 种优化技巧加上批处理可以进一步提升速度。批处理即将多个样本组合起来一起推理，这样会使这些样本的总生成时间低于逐样本生成时的总生成时间。

下面给出了一个批处理的简单代码:

```python
text_prompt = [
    "Let's try generating speech, with Bark, a text-to-speech model",
    "Wow, batching is so great!",
    "I love Hugging Face, it's so cool."]

inputs = processor(text_prompt).to(device)

with torch.inference_mode():
  # samples are generated all at once
  speech_output = model.generate(**inputs, do_sample = True, fine_temperature = 0.4, coarse_temperature = 0.8)
```

输出音频如下 (下载 [第一个](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_0.wav)、[第二个](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_1.wav) 以及 [第三个](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_2.wav) 音频文件):

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_0.wav" type="audio/wav">
当前浏览器不支持音频元素。
</audio>

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_1.wav" type="audio/wav">
当前浏览器不支持音频元素。
</audio>

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_2.wav" type="audio/wav">
当前浏览器不支持音频元素。
</audio>

# 基准测试结果

上文我们进行的这些小实验更多是想法验证，我们需要将其扩展以更准确地衡量性能。另外，在每次正式测量性能之前，还需要先跑几轮以预热 GPU。

以下是扩展至 100 个样本的基准测量的结果，使用的模型为 **大 Bark**。

该基准测试在 NVIDIA TITAN RTX 24GB 上运行，最大词元数为 256。

## 如何解读结果？

### 延迟

该指标主要测量每次调用生成函数的平均时间，无论 batch size 如何。

换句话说，它等于 $\frac{elapsedTime}{nbLoops}$。

**延迟越小越好。**

### 最大内存占用

它主要测量生成函数在每次调用期间使用的最大内存。

**内存占用越小越好。**

### 吞吐量

它测量每秒生成的样本数。这次，batch size 的因素已被考虑在内。

换句话说，它等于 $\frac{nbLoops*batchSize}{elapsedTime}$。

**吞吐量越高越好。**

## 单样本推理

下表为 `batch_size=1` 的结果。

| 绝对性能           | 延迟 | 内存占用  |
|-----------------------------|---------|---------|
| 无优化           |   10.48 | 5025.0M |
| 仅 bettertransformer      |    7.70 | 4974.3M |
| CPU 卸载 + bettertransformer |    8.90 | 2040.7M |
| CPU 卸载 + bettertransformer + fp16   |    8.10 | 1010.4M |

| 相对性能              | 延迟 | 内存占用 |
|-----------------------------|---------|--------|
| 无优化            |      0% |     0% |
| 仅 bettertransformer      |    -27% |    -1% |
| CPU 卸载 + bettertransformer |    -15% |   -59% |
| CPU 卸载 + bettertransformer + fp16   |    -23% |   -80% |

### 点评

不出所料，CPU 卸载极大地减少了内存占用，同时略微增加了延迟。

然而，结合 bettertransformer 和 `fp16` ，我们得到了两全其美的效果，巨大的延迟和内存降低！

## batch size 为 8

以下是 `batch_size=8` 时的吞吐量基准测试结果。

请注意，由于 `bettertransformer` 是一种免费优化，它执行与非优化模型完全相同的操作并具有相同的内存占用，同时速度更快，因此所有的基准测试均 **默认开启此优化**。

| 绝对性能              | 延迟 | 内存占用  | 吞吐量 |
|-------------------------------|---------|---------|-----------|
| 基线 (bettertransformer) |   19.26 | 8329.2M |      0.42 |
| + fp16                          |   10.32 | 4198.8M |      0.78 |
| + CPU 卸载                       |   20.46 | 5172.1M |      0.39 |
| + CPU 卸载 + fp16                |   10.91 | 2619.5M |      0.73 |

| 相对性能                | 延迟 | 内存占用 | 吞吐量 |
|-------------------------------|---------|--------|------------|
| + 基线  (bettertransformer) |      0% |     0% |         0% |
| + fp16                          |    -46% |   -50% |        87% |
| + CPU 卸载                        |      6% |   -38% |        -6% |
| + CPU 卸载  + fp16                |    -43% |   -69% |       77% |

### 点评

这里，我们看到了组合所有三个优化技巧后的性能潜力！

`fp16` 对延迟的影响在 `batch_size = 1` 时不太明显，但在 `batch_size = 1` 时的表现非常有趣，它可以将延迟减少近一半，吞吐量几乎翻倍！

# 结束语

本文展示了 🤗 生态系统中的一些现成的、简单的优化技巧。使用这些技巧中的任何一种或全部三种都可以极大地改善 Bark 的推理速度和内存占用。

- **使用🤗 Better Transformer 和 CPU 卸载**，你可以对大 Bark 模型进行推理，而不会出现任何性能下降，占用空间仅为 2GB (而不是 5GB)，同时速度提高 15%。
- 如果你钟情于高吞吐，可以 **把 batch size 打到 8，并利用 🤗 Better Transformer 和 fp16**。
- 如果你“既要，又要，还要”，试试 **fp16、🤗 Better Transformer 加 CPU 卸载** 组合优化吧！