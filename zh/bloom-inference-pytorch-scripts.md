---
title: "使用 DeepSpeed 和 Accelerate 进行超快 BLOOM 模型推理"
thumbnail: /blog/assets/bloom-inference-pytorch-scripts/thumbnail.png
authors:
- user: stas
- user: sgugger
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 使用 DeepSpeed 和 Accelerate 进行超快 BLOOM 模型推理


本文展示了如何使用 1760 亿 (176B) 参数的 [BLOOM 模型](https://huggingface.co/bigscience/bloom) 生成文本时如何获得超快的词吞吐 (per token throughput)。

因为在使用 bf16 (bfloat16) 权重时该模型内存占用为 352 GB (`176*2`)，所以最高效的硬件配置是使用 8x80GB 的 A100 GPU。也可使用 2x8x40GB 的 A100 或者 2x8x48GB 的 A6000。使用这些 GPU 的主要原因是截至本文成稿时为止它们是能提供最大显存的 GPU，但你也可以使用其他 GPU。比如，可以使用 24x32GB V100。

一般来讲，使用单节点会带来最快的吞吐，因为大多数时候节点内的 GPU 互联硬件比节点间的快，但未必总是如此。

如果你没有这么高端的硬件或没有这么多硬件，你仍可能通过 CPU 卸载 (CPU offload) 或是 NVMe 卸载 (NVMe offload) 的方式在更小的 GPU 上对 BLOOM 进行推理。当然，生成时间会慢很多。

我们计划涉及 [8 比特量化方案](https://huggingface.co/blog/hf-bitsandbytes-integration)，该方案以稍慢的吞吐为代价将显存需求减少到一半。我们还会讨论 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 和 [Deepspeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) 库。

## 测试基准

事不宜迟，我们先展示一些数据吧。

为了保持一致性，除非另有说明，本文的测试基准都是在相同的配有 512GB CPU 内存的 8x80GB A100 节点上完成的，该节点来自 法国 [Jean Zay 超算中心](http://www.idris.fr/eng/jean-zay/index.html)。这一配置对于节省检查点加载时间非常重要，如果磁盘加载缓慢，就需要更长的检查点加载时间。我们在多个进程中并行执行 IO 任务的情况下更是如此。

所有的测试基准都是使用 [贪心搜索](https://huggingface.co/blog/how-to-generate#greedy-search) 完成最多 100 个词的生成任务:

```
Generate args {'max_length': 100, 'do_sample': False}
```

输入提示词仅包含几个词。我们会缓存先前见到的词，因为每次重新计算它们相当慢。

首先，让我们快速看一下从开始到准备好花了多长时间， 即模型加载和准备花了多长时间:

| 方法                    | 秒   |
| :---------------------- | :--- |
| accelerate              |  121 |
| ds-inference shard-int8 |   61 |
| ds-inference shard-fp16 |   60 |
| ds-inference unsharded  |  662 |
| ds-zero                 |  462 |

Deepspeed-Inference 使用了预分片的权重仓库，整个加载时间大约在 1 分钟。Accelerrate 的加载时间也很优秀，只有大约 2 分钟。其他方案就慢得多。

加载时间有可能重要也可能并不重要，因为一旦加载成功你可以一遍遍持续不断地生成词而不再需要额外地加载开销。

接着是最重要的测试基准指标：词生成吞吐 (token generation throughput)。这个吞吐的度量比较简单，即：生成 100 个新词的时间除以 100 和 batch size (也就是除以生成的总词数)。

下面列出了 8x80GB GPU 的吞吐，单位为毫秒:

| 方法         \ bs |      1 |     8 |    16 |    32 |   64 |  128 |  256 | 512  |
| :---------------- | :----- | :---- | :---- | :---- | :--- | :--- | :--- | :--- |
| accelerate   bf16 | 230.38 | 31.78 | 17.84 | 10.89 |  oom |      |      |      |
| accelerate   int8 | 286.56 | 40.92 | 22.65 | 13.27 |  oom |      |      |      |
| ds-inference fp16 |  44.02 |  5.70 |  3.01 |  1.68 | 1.00 | 0.69 |  oom |      |
| ds-inference int8 |  89.09 | 11.44 |  5.88 |  3.09 | 1.71 | 1.02 | 0.71 | oom  |
| ds-zero      bf16 |    283 | 34.88 |   oom |       |      |      |      |      |

这里， 当内存耗尽 (Out Of Memory，OOM) 时即表明 batch size 太大 GPU 显存放不下了。

使用 Deepspeed-Inference 的张量并行 (Tensor Parallelism，TP) 和定制化融合 CUDA 核函数可以得到小于 1 毫秒的吞吐！太棒了！尽管使用这个方案去推理那些尚未被验证过的模型时，你可能会需要花一些时间去开发从而让它工作起来。

Accelerate 也超级快。它使用了非常简单的管线并行 (Pipeline Parallelism，PP)。因为它非常简单，所以它应该对任何模型都是开箱即用的。

因为 Deepspeed-ZeRO 可以并行处理多路生成流，其吞吐可以再除以 8 或者 16，具体数值取决于在调用 `generate` 时用了 8 个 GPU 还是 16 个 GPU。当然，这也意味着在 8x80GB A100 的情况下 (见上表) ，可以处理的 batch size 为 64 且吞吐可至大约 4 毫秒。因此，这 3 种方案的性能是接近的。

让我们再重新看一下这些数字是怎么计算出来的。举个例子，使用 Deepspeed-Inference fp16 模式实时生成 batch size 为 128、长度为 100 个新词的文本花了 8832 毫秒，因此我们可以这样计算吞吐：钟面时间 / ( batch size * 新词数 ) 或 `8821/(128*100) = 0.69`。

现在我们一起看看 Deepspeed-Inference 和 BitsAndBytes 提供的 int8 量化模型的威力，它仅需占用 bfloat16 或 float16 推理所需显存的一半。

以下为 4x80GB GPU 的吞吐，单位为毫秒:

| 方法          bs  |      1 |     8 |    16 |    32 |   64 | 128  |
| :---------------- | :----- | :---- | :---- | :---- | :--- | :--- |
| accelerate   int8 | 284.15 | 40.14 | 21.97 |  oom  |      |      |
| ds-inference int8 | 156.51 | 20.11 | 10.38 |  5.50 | 2.96 | oom  |

你只需在下述 3 个脚本里添加 `--benchmark` 即可重现这些测试基准的结果。


## 方案

首先获取最新的演示代码仓库:

```
git clone https://github.com/huggingface/transformers-bloom-inference
cd transformers-bloom-inference
```

本文我们准备使用 `bloom-inference-scripts/` 文件夹下的 3 个脚本。

下面我们按框架的字母序逐一展示相关方案。

## HuggingFace Accelerate

[Accelerate](https://github.com/huggingface/accelerate)

Accelerate 按如下步骤进行大模型推理:
1. 用空的权重实例化模型。
2. 分析每层的大小以及每个设备 (CPU, CPU) 的可用空间，并决定每层应该在哪个设备上推理。
3. 逐比特加载模型 checkpoint 并把权重加载到相应的设备。

然后，它会使用钩子代码 (hook) 来确保模型正确运行，钩子代码被用于在正确的设备间传输输入和输出，并在前向轮运行前加载那些卸载到 CPU (甚至硬盘) 上的权重到 GPU，然后在前向轮结束后再次把权重卸载。

在有多个 GPU 且有足够空间放下整个模型的情形下，该方案在 GPU 间逐个切换直至所有层运行完毕。每个给定的时间只有一个 GPU 工作，这听起来很没效率。但尽管该方案 GPU 存在空闲，它的吞吐却相当不错。

因为相同的代码可以运行在任意给定的设置中，所以本方案非常灵活。Accelerate 首先使用所有可用的 GPU，当显存已满时会卸载到 CPU 内存直至卸载到硬盘。卸载到 CPU 或硬盘会让推理变慢。举个例子，与 8x80 A100 上的 10 毫秒相比，已有用户报告，不作任何代码改动，在 2 个 A100 上运行 BLOOM 吞吐是每词 15 秒。

你可以你从 [Accelerate 文档](https://huggingface.co/docs/accelerate/big_modeling) 中获取本方案的更多信息。

### 设置

```
pip install transformers>=4.21.3 accelerate>=0.12.0
```


### 运行

简单执行如下命令:

```
python bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 1 --benchmark
```

如需使用 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 的 8 比特量化方案，首先要安装 `bitsandbytes`:

```
pip install bitsandbytes
```

然后在前述命令行中增加 `--dtype int8`:

```
python bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --dtype int8 --batch_size 1 --benchmark
```

如果你有 4 个以上 GPU，你可以通过如下命令限制脚本只使用其中 4 个 GPU:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --dtype int8 --batch_size 1 --benchmark
```

在这个例子中，不 OOM 的最大 batch size 是 40。如果你深入研究脚本，你会看到我们需要调整显存分配映射从而把第一个 GPU 解放出来去仅处理激活和先前词的缓存。

## DeepSpeed-Inference

[DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) 使用张量并行 (Tensor Parallelism) 以及高效的融合 CUDA 核函数在 128 这个大 batch size 下达到了每词 1 毫秒的超快推理性能。

### 设置

```
pip install deepspeed>=0.7.3
```

### 运行

1. 最快的方法是使用 TP 预分片 (TP = Tensor Parallel) 的 checkpoint，与非预分片的 bloom checkpoint 相比，它仅需大约 1 分钟即可加载:


```
deepspeed --num_gpus 8 bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-fp16
```

1a. 如果你想要运行原始 bloom checkpoint，这个 checkpoint 一旦加载就会跟之前的方案跑到相同的吞吐，但加载需要花 10 到 20 分钟:

```
deepspeed --num_gpus 8 bloom-inference-scripts/bloom-ds-inference.py --name bigscience/bloom
```

2a. 8 比特量化版本与一般的半精度版本相比仅需一半 GPU 显存:


```
deepspeed --num_gpus 8 bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8
```

这里我们使用 `microsoft/bloom-deepspeed-inference-int8` checkpoint 并告诉脚本跑在 `int8` 模式。

当然，现在仅需 4x80GB A100 GPU 就够了:

```
deepspeed --num_gpus 4 bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8
```

这种情况下，不 OOM 的最大 batch size 是 128。

可以看到，本方案中有两个因素在获得更好的性能上起到了主导作用。

1. 本方案的吞吐提高主要来自于张量并行 (Tensor Parallelism，TP) 而不是 Acclerate 的管线并行 (Pipeline Parallelism，PP)。因为 Accelerate 旨在成为非常通用的方案，因此也非常不幸地很难最大化 GPU 使用率。它首先在 GPU 0 上完成所有计算，然后是 GPU 1，等等，一直到 GPU 8，这意味着任何时刻都有 7 个 GPU 是空闲的。而另一方面，DeepSpeed-Inference 使用了 TP，意味着它会向所有 GPU 发送张量，在每个 GPU 上计算部分生成结果，然后在所有的 GPU 间通信计算结果，并继续做下一层。这就是说 TP 所有的 GPU 都同时是活跃的，但它需要比 PP 多得多的通信。
2. DeepSpeed-Inference 还使用了定制的 CUDA 核函数以避免分配太多内存以及太多进出 GPU 的张量拷贝。这么做会减少显存需求及核函数启动次数从而提高吞吐，另外还可以支持更大的 batch size 从而进一步增加总吞吐。

如果你对更多的例子感兴趣，可以看看t [Accelerate GPT-J inference with DeepSpeed-Inference on GPUs](https://www.philschmid.de/gptj-deepspeed-inference) 或 [Accelerate BERT inference with DeepSpeed-Inference on GPUs](https://www.philschmid.de/bert-deepspeed-inference)。

## Deepspeed ZeRO-Inference

[Deepspeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) 使用一个魔术般的分片方法，使得它可以输入几乎任何模型并将它扩展到少至几个多至上百个 GPU，进行训练或推理。

### 设置

```
pip install deepspeed
```


### 运行
注意到现在为止的脚本都是所有 GPU 都处理相同的输入，但你其实可以在每个 GPU 上运行不同的流，从而得到 `n_gpu` 倍的吞吐。你不能用 Deepspeed-Inference 达到这个目的。

```
deepspeed --num_gpus 8 bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 1 --benchmark
```

请记住用户可以用 ZeRO 同时创建多个不同的流，因此总性能应该是每秒每词的吞吐除以参与计算的 GPU 的数目，因此根据你是使用 16 个 GPU 还是 8 个 GPU，可以获得 8 倍或者 16 倍的更快性能。

你还可以在一个小型 GPU 上试试卸载方案，运行的时间会很长，但是如果你没有 8 个巨型 GPU 的话这也是一个聊甚于无的方案。

CPU 卸载 (1x GPUs):

```
deepspeed --num_gpus 1 bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --cpu_offload --benchmark
```

NVMe 卸载 (1x GPUs):

```
deepspeed --num_gpus 1 bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --nvme_offload_path=/path/to/nvme_offload --benchmark
```

请确保在你的高速 NVMe 盘上预留约 400GB 的空间，并把 `/path/to/nvme_offload` 设成它。


## 更多客户端及服务器方案

你可以从 [transformers-bloom-inference](https://github.com/huggingface/transformers-bloom-inference) 找到更多非常高效的方案，包括服务器方案。

这里我们提供一些预览。

服务器方案：

* [Mayank Mishra](https://github.com/mayank31398) 拿着本博文中讨论的所有演示代码并把它们变成了一个网络服务包，你可以从 [这里](https://github.com/huggingface/transformers-bloom-inference) 下载。

* [Nicolas Patry](https://github.com/Narsil) 开发了一个超高效的 [基于 Rust 的网络服务方案]((https://github.com/Narsil/bloomserver)。

更多的客户端方案:

* [Thomas Wang](https://github.com/thomasw21) 正在开发一个很快的 [定制 CUDA 核函数的 BLOOM 模型](https://github.com/huggingface/transformers_bloom_parallel)。

* HuggingFace 的 JAX 组已开发了一个 [基于 JAX 的方案](https://github.com/huggingface/bloom-jax-inference)。


因为如果你在本博文发布几个月后读到它，很有可能它已经不能反映最新的状态了，你可以去 [transformers-bloom-inference 的 GitHub 仓库](https://github.com/huggingface/transformers-bloom-inference) 找到最新的方案。

## 致谢

万分感谢如下这些人，他们提出了好的问题并帮助提高了本文的可读性：Olatunji Ruwase 和 Philipp Schmid。
