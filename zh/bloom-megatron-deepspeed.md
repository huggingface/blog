---
title: "千亿参数开源大模型 BLOOM 背后的技术"
thumbnail: /blog/assets/86_bloom_megatron_deepspeed/thumbnail.png
authors:
- user: stas
translators:
- user: MatrixYao
- user: inferjay
  proofreader: true
---

# 千亿参数开源大模型 BLOOM 背后的技术


> 假设你现在有了数据，也搞到了预算，一切就绪，准备开始训练一个大模型，一显身手了，“一朝看尽长安花”似乎近在眼前 …… 且慢！训练可不仅仅像这两个字的发音那么简单，看看 BLOOM 的训练或许对你有帮助。

近年来，语言模型越训越大已成为常态。大家通常会诟病这些大模型本身的信息未被公开以供研究，但很少关注大模型训练技术这种背后的知识。本文旨在以 1760 亿参数的语言模型 [BLOOM](https://hf.co/bigscience/bloom) 为例，阐明训练此类模型背后的软硬件工程和技术要点，以促进大家对大模型训练技术的讨论。

首先，我们要感谢促成或赞助我们这个小组最终完成了训练 1760 亿参数模型这一惊人壮举的公司、个人和团体。

然后，我们开始讨论硬件配置和主要技术组件。

![BLOOM](../assets/86_bloom_megatron_deepspeed/bloom-banner.png)

以下是对本项目的简要总结:

|               |                             |
| :-----        | :-------------              |
| 硬件      | 384 张 80GB A100 GPU         |
| 软件      | Megatron-DeepSpeed          |
| 模型架构  | 基于 GPT3            |
| 数据集       | 含 59 种语言，共 3500 亿词元 |
| 训练时长 | 3.5 个月                  |

## 人员组成

该项目由 Thomas Wolf (Hugging Face 联合创始人兼 CSO) 发想，他敢于与大公司竞争，提出不仅要训练出立于世界上最大的多语言模型之林的模型，还要让所有人都可以公开访问训练结果，圆了大多数人的梦想

本文主要关注模型训练的工程方面。BLOOM 背后的技术中最重要的部分是分享专业知识并帮助我们进行编码和训练的人员和公司。

我们主要需要感谢 6 个群体:

1. HuggingFace 的 BigScience 团队投入了六名以上的全职员工全程参与了训练的研究和运行，他们还提供或报销了 Jean Zay 计算机之外的所有基础设施。
2. Microsoft DeepSpeed 团队，开发了 DeepSpeed，后来将其与 Megatron-LM 集成，其开发人员花费数周时间研究项目需求，并在训练前和训练期间提供了许多很棒的实用经验建议。
3. NVIDIA Megatron-LM 团队开发了 Megatron-LM，他们非常乐于回答我们的大量问题并提供一流的使用建议。
4. IDRIS / GENCI 团队管理着 Jean Zay 超级计算机，他们为该项目捐赠了大量的算力和强大的系统管理支持。
5. PyTorch 团队创建了一个超强的框架，其余软件都基于该框架，并且在准备训练期间非常支持我们，修复了多个 bug 并提高了我们所依赖的 PyTorch 组件的训练可用性。
6. BigScience 工程工作组志愿者 很难说出所有为该项目的工程方面做出贡献的杰出人物的名字，所以我只列举 Hugging Face 之外的几个关键人物，他们在过去 14 个月中为该项目奠定了工程基础:

Olatunji Ruwase、Deepak Narayanan、Jeff Rasley、Jared Casper、Samyam Rajbhandari 和 Rémi Lacroix

我们也感谢所有允许其员工为该项目做出贡献的公司。

## 概述

BLOOM 的模型架构与 [GPT3](https://en.wikipedia.org/wiki/GPT-3) 非常相似，只是增加了一些改进，本文稍后将对此进行讨论。

该模型是在 [Jean Zay](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html) 上训练的，Jean Zay 是由 GENCI 管理的法国政府资助的超级计算机，安装在法国国家科学研究中心 (CNRS) 的国家计算中心 [IDRIS](http://www.idris.fr/)。训练所需的算力由 GENCI 慷慨捐赠给本项目 (捐赠号 2021-A0101012475)。

训练硬件:

- GPU: 384 张 NVIDIA A100 80GB GPU (48 个节点) + 32 张备用 GPU
- 每个节点 8 张 GPU，4 条 NVLink 卡间互联，4 条 OmniPath 链路
- CPU: AMD EPYC 7543 32 核处理器
- CPU 内存: 每个节点 512GB
- GPU 显存: 每个节点 640GB
- 节点间连接: 使用 Omni-Path Architecture (OPA) 网卡，网络拓扑为无阻塞胖树
- NCCL - 通信网络: 一个完全专用的子网
- 磁盘 IO 网络: GPFS 与其他节点和用户共享

Checkpoints:

- [主 checkpoints](https://huggingface.co/bigscience/bloom)
- 每个 checkpoint 含精度为 fp32 的优化器状态和精度为 bf16+fp32 的权重，占用存储空间为 2.3TB。如只保存 bf16 的权重，则仅占用 329GB 的存储空间。

数据集:

- 41.5TB 经过大量去重和清洗的文本，包含 46 种语言，最终转换为 350B 个词元
- 模型的词汇表含 250,680 个词元
- 更详细信息，请参阅 [The BigScience Corpus A 1.6TB Composite Multilingual Dataset](https://openreview.net/forum?id=UoEw6KigkUn)

176B BLOOM 模型的训练于 2022 年 3 月至 7 月期间，耗时约 3.5 个月完成 (约 100 万计算时)。

## Megatron-DeepSpeed

176B BLOOM 模型使用 [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) 进行训练，它结合了两种主要技术:

* [DeepSpeed](https://github.com/microsoft/DeepSpeed) 是一个深度学习优化库，让分布式训练变得简单、高效且有效。
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 是由 NVIDIA 的应用深度学习研究团队开发的大型、强大的 transformer 模型框架。

DeepSpeed 团队通过将 DeepSpeed 库中的 ZeRO 分片和流水线并行 (Pipeline Parallelism) 与 Megatron-LM 中的张量并行 (Tensor Parallelism) 相结合，开发了一种基于 3D 并行的方案。有关每个组件的更多详细信息，请参见下表。

请注意，BigScience 的 [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) 是基于原始 [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) 代码库，我们还在其上添加了不少代码。

下表列出了我们在训练 BLOOM 时各采用了两个框架的哪些组件

| 组件                                       | DeepSpeed | Megatron-LM |
| :----                                           | :----     | :----       |
| [ZeRO 数据并行](#zero-data-parallelism) | 是         |             |
| [张量并行](#tensor-parallelism)       |           | 是           |
| [流水线并行](#pipeline-parallelism)   | 是         |             |
| [BF16 优化器](#bf16optimizer)                 | 是         |             |
| [CUDA 融合核函数](#fused-cuda-kernels)       |           | 是           |
| [DataLoader](#datasets)                         |           | 是           |

请注意，Megatron-LM 和 DeepSpeed 都有流水线并行和 BF16 优化器实现，但我们使用 DeepSpeed 的实现，因为它们集成进了 ZeRO。

Megatron-DeepSpeed 实现了 3D 并行以允许大模型以非常有效的方式进行训练。我们简要讨论一下有哪些 3D 组件。

1. 数据并行 (Data Parallelism，DP) - 相同的设置和模型被复制多份，每份每次都被馈送不同的一份数据。处理是并行完成的，所有份在每个训练步结束时同步。
2. 张量并行 (Tensor Parallelism，TP) - 每个张量都被分成多个块，因此张量的每个分片都位于其指定的 GPU 上，而不是让整个张量驻留在单个 GPU 上。在处理过程中，每个分片在不同的 GPU 上分别并行处理，结果在步骤结束时同步。这就是所谓的水平并行，因为是做的水平拆分。
3. 流水线并行 (Pipeline Parallelism，PP) - 模型在多个 GPU 上垂直 (即按层) 拆分，因此只有一个或多个模型层放置在单个 GPU 上。每个 GPU 并行处理流水线的不同阶段，并处理 batch 的一部分数据。
4. 零冗余优化器 (Zero Redundancy Optimizer，ZeRO) - 也执行与 TP 相类似的张量分片，但整个张量会及时重建以进行前向或反向计算，因此不需要修改模型。它还支持各种卸载技术以补偿有限的 GPU 内存。

## 数据并行

大多数只有几张 GPU 的用户可能比较熟悉 `DistributedDataParallel`(DDP)，这是相应的  [PyTorch 文档](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)。在该方法中，模型被完全复制到每个 GPU，然后在每次迭代后所有模型相互同步各自的状态。这种方法可以通过投入更多 GPU 资源的方式加快训练速度，解决问题。但它有个限制，即只有当模型能够放进单个 GPU 时才有效。

### ZeRO 数据并行

下图很好地描述了 ZeRO 数据并行 (来自这篇 [博文](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/))。

![DeepSpeed-Image-1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png)

看上去比较高大上，可能让你很难专心去理解，但实际上，这个概念非常简单。这只是通常的 DDP，只是没有每个 GPU 都复制完整的模型参数、梯度和优化器状态，而是每个 GPU 只存储其中的一部分。在随后的运行过程中，当需要给定层的完整层参数时，所有 GPU 同步以相互提供它们缺失的部分 —— 仅此而已。

该组件由 DeepSpeed 实现。

## 张量并行

在张量并行 (TP) 中，每个 GPU 仅处理张量的一部分，并且仅当某些算子需要完整的张量时才触发聚合操作。

在本节中，我们使用 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 论文: [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)。

Transformer 类模型的主要模块为: 一个全连接层 `nn.Linear`，后面跟一个非线性激活层 `GeLU`。

沿用 Megatron 论文的符号，我们可以将其点积部分写为 `Y = GeLU (XA)`，其中 `X` 和 `Y` 是输入和输出向量， `A` 是权重矩阵。

如果以矩阵形式表示的话，很容易看出矩阵乘法可以如何在多个 GPU 之间拆分:

![并行 GEMM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_gemm.png)

如果我们将权重矩阵 `A` 按列拆分到 `N` 个 GPU 上，然后并行执行矩阵乘法 `XA_1` 到  `XA_n`，那么我们最终将得到 `N` 个输出向量 `Y_1、Y_2、…… 、 Y_n` ，它们可以独立输入 `GeLU`:

![independent GeLU](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-independent-gelu.png)

注意因为 `Y` 矩阵是按列拆分的，因此随后的 GEMM 我们可以选择按行拆分方案，这样它就可以直接获取前面层的 GeLU 的输出，而无需任何额外的通信。

使用该原理，我们可以更新任意深度的 MLP，只需在每个 `拆列 - 拆行` 序列之后同步 GPU。Megatron-LM 论文作者为此提供了一个不错的图示:

![并行分片处理](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_shard_processing.png)

这里 `f` 是前向传播中的恒等运算符，后向传播中的 all reduce，而 `g` 是前向传播中的 all reduce 和后向传播中的恒等式。

并行化多头注意力层甚至更简单，因为它们本来就是并行的，因为有多个独立的头！

![并行自注意力](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_self_attention.png)

需要特别考虑的是: 由于前向和后向传播中每层都有两个 all reduce，因此 TP 需要设备间有非常快速的互联。因此，除非你有一个非常快的网络，否则不建议跨多个节点进行 TP。我们训练 BLOOM 的硬件配置中，节点间的速度比 PCIe 慢很多。实际上，如果节点有 4 个 GPU，则最高 TP 度设为 4 比较好。如果需要 TP 度为 8，则需要使用至少有 8 个 GPU 的节点。

该组件由 Megatron-LM 实现。Megatron-LM 最近扩展了张量并行能力，新增了序列并行的能力，用于难以使用前述切分算法的算子，如 LayerNorm。[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) 论文提供了此技术的详细信息。序列并行是在训练 BLOOM 之后开发的，所以 BLOOM 训练时并未采用此技术。

## 流水线并行

朴素流水线并行 (naive PP) 是将模型各层分组分布在多个 GPU 上，并简单地将数据从 GPU 移动到 GPU，就好像它是一个大型复合 GPU 一样。该机制相对简单 - 将所需层用 `.to()` 方法绑到相应设备，现在只要数据进出这些层，这些层就会将数据切换到与该层相同的设备，其余部分保持不变。

这其实就是垂直模型并行，因为如果你还记得我们是怎么画大多数模型的拓扑图的，我们其实是垂直切分模型各层的。例如，如果下图显示一个 8 层模型:

```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        GPU0                 GPU1
```
我们将它垂直切成 2 部分，将层 0-3 放置在 GPU0 上，将层 4-7 放置在 GPU1 上。

现在，当数据从第 0 层传到第 1 层、第 1 层传到第 2 层以及第 2 层传到第 3 层时，这就跟单 GPU 上的普通前向传播一样。但是当数据需要从第 3 层传到第 4 层时，它需要从 GPU0 传输到 GPU1，这会引入通信开销。如果参与的 GPU 位于同一计算节点 (例如同一台物理机器) 上，则传输非常快，但如果 GPU 位于不同的计算节点 (例如多台机器) 上，通信开销可能会大得多。

然后第 4 到 5 到 6 到 7 层又像普通模型一样，当第 7 层完成时，我们通常需要将数据发送回标签所在的第 0 层 (或者将标签发送到最后一层)。现在可以计算损失，然后使用优化器来进行更新参数了。

问题:

- 该方法为什么被称为 朴素 流水线并行呢，它又有什么缺陷呢？主要是因为该方案在任意给定时刻除了一个 GPU 之外的其他所有 GPU 都是空闲的。因此，如果使用 4 个 GPU，则几乎等同于将单个 GPU 的内存量翻两番，而其他资源 (如计算) 相当于没用上。另外还需要加上在设备之间复制数据的开销。所以 4 张 使用朴素流水线并行的 6GB 卡将能够容纳与 1 张 24GB 卡相同大小的模型，而后者训练得更快，因为它没有数据传输开销。但是，比如说，如果你有 40GB 卡，但需要跑 45GB 模型，你可以使用 4x 40GB 卡 (也就刚刚够用，因为还有梯度和优化器状态需要显存)。

- 共享嵌入可能需要在 GPU 之间来回复制。我们使用的流水线并行 (PP) 与上述朴素 PP 几乎相同，但它解决了 GPU 闲置问题，方法是将传入的 batch 分块为 micros batch 并人工创建流水线，从而允许不同的 GPU 同时参与计算过程。

下图来自于 [GPipe 论文](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)，其上半部分表示朴素 PP 方案，下半部分是 PP 方法:

![mp-pp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png)

从图的下半部分很容易看出 PP 的死区 (指 GPU 处于空闲状态) 更少，即 “气泡” 更少。

图上两种方案的并行度均为 4 ，即由 4 张 GPU 组成流水线。于是就有了 F0、F1、F2、F3 这 4 个管级的前向路径，然后是 B3、B2、B1、B0 的逆序后向路径。

PP 引入了一个新的超参数来调整，称为 `块 (chunks)`。它定义了通过同一管级按顺序发送多少数据块。例如，在图的下半部分，你可以看到 `chunks = 4`。GPU0 在 chunk 0、1、2 和 3 (F0,0、F0,1、F0,2、F0,3) 上执行相同的前向路径，然后等待，等其他 GPU 完成工作后，GPU0 会再次开始工作，为块 3、2、1 和 0 (B0,3、B0,2、B0,1、B0,0) 执行后向路径。

请注意，从概念上讲，这与梯度累积 (gradient accumulation steps，GAS) 的意思相同。PyTorch 叫它 `块`，而 DeepSpeed 叫它 `GAS`。

因为 `块`，PP 引入了 micro-batches (MBS) 的概念。DP 将全局 batch size 拆分为小 batch size，因此如果 DP 度为 4，则全局 batch size 1024 将拆分为 4 个小 batch size，每个小 batch size 为 256 (1024/4)。而如果 `块` (或 GAS) 的数量为 32，我们最终得到的 micro batch size 为 8 (256/32)。每个管级一次处理一个 micro batch。

计算 DP + PP 设置的全局批量大小的公式为: `mbs*chunks*dp_degree` (`8*32*4=1024`).

我们回过头再看一下图。

使用 `chunks=1` 你最终得到的是朴素 PP，这是非常低效的。而使用非常大的 `块` 数，你最终会得到很小的微批量大小，这很可能也不是很有效。因此，必须通过实验来找到能最有效地利用 GPU 的 `块` 数。

该图显示存在无法并行化的 “死” 时间气泡，因为最后一个 `forward` 阶段必须等待 `backward` 完成流水。那么，找到最佳的 `块` 数，从而使所有参与的 GPU 达到高的并发利用率，这一问题其实就转化为最小化气泡数了。

这种调度机制被称为 `全前全后`。其他一些可选方案有 [一前一后](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/) 和 [交错一前一后](https://arxiv.org/abs/2104.04473)。

虽然 Megatron-LM 和 DeepSpeed 都有自己的 PP 协议实现，但 Megatron-DeepSpeed 使用的是 DeepSpeed 实现，因为它与 DeepSpeed 的其他功能集成在一起。

这里的另一个重要问题是词嵌入矩阵的大小。虽然通常词嵌入矩阵比 transfomer 块所需的内存更少，但在 BLOOM 有 250k 词汇表的情况下，嵌入层需要 7.2GB 的 bf16 权重，而变换器块仅为 4.9GB。因此，我们不得不让 Megatron-Deepspeed 将嵌入层视为一个转换器块。所以我们有一个 72 级的流水线，其中 2 个是专门用于嵌入的 (第一个和最后一个)。这使得我们可以平衡 GPU 的内存消耗。如果我们不这样做，我们就会让第一级和最后一级消耗很大的 GPU 内存，而 95% 的 GPU 内存使用会很少，因此训练将很不高效。

## DP+PP

DeepSpeed 流水线 [并行教程](https://www.deepspeed.ai/tutorials/pipeline/) 中有一张图演示了如何将 DP 与 PP 结合起来，如下所示。

![dp-pp-2d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png)

这里重要的是要了解 DP rank 0 是看不见 GPU2 的， DP rank 1 是看不到 GPU3 的。对于 DP 而言，只有 GPU 0 和 1，并向它们馈送数据。GPU0 使用 PP “秘密地” 将它的一些负载卸载到 GPU2。同样地， GPU1 也会得到 GPU3 的帮助。

由于每个维度至少需要 2 个 GPU，因此这儿至少需要 4 个 GPU。

## DP+PP+TP

为了更高效地训练，可以将 PP、TP 和 DP 相结合，称为 3D 并行，如下图所示。

![dp-pp-tp-3d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png)

此图来自博文《[3D 并行: 扩展到万亿参数模型](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)》), 这也是一篇好文章。

由于每个维度至少需要 2 个 GPU，因此在这里你至少需要 8 个 GPU 才能实现完整的 3D 并行。

## ZeRO DP+PP+TP

DeepSpeed 的主要功能之一是 ZeRO，它是 DP 的超级可伸缩增强版，我们在 [ZeRO 数据并行](#ZeRO-数据并行) 一节中已经讨论过了。通常它是一个独立的功能，不需要 PP 或 TP。但它也可以与 PP、TP 结合使用。

当 ZeRO-DP 与 PP (以及 TP) 结合时，它通常只启用 ZeRO 阶段 1，它只对优化器状态进行分片。ZeRO 阶段 2 还会对梯度进行分片，阶段 3 也对模型权重进行分片。

虽然理论上可以将 ZeRO 阶段 2 与 流水线并行 一起使用，但它会对性能产生不良影响。每个 micro batch 都需要一个额外的 reduce-scatter 通信来在分片之前聚合梯度，这会增加潜在的显著通信开销。根据流水线并行的性质，我们会使用小的 micro batch ，并把重点放在算术强度 (micro batch size) 与最小化流水线气泡 (micro batch 的数量) 两者间折衷。因此，增加的通信开销会损害流水线并行。

此外，由于 PP，层数已经比正常情况下少，因此并不会节省很多内存。PP 已经将梯度大小减少了 `1/PP`，因此在此基础之上的梯度分片和纯 DP 相比节省不了多少内存。

ZeRO 阶段 3 也可用于训练这种规模的模型，但是，它需要的通信量比 DeepSpeed 3D 并行更多。一年前，在对我们的环境进行仔细评估后，我们发现 Megatron-DeepSpeed 3D 并行性表现最佳。此后，ZeRO 阶段 3 的性能有了显著提高，如果我们今天要对其进行重新评估，也许我们会选择阶段 3。

## BF16Optimizer

用 FP16 训练巨型 LLM 模型是一个禁忌。

我们已经通过花费几个月的时间 [训练 104B 模型](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide) 自证了这一点，你可以从 [Tensorboard](https://huggingface.co/bigscience/tr8-104B-logs/tensorboard) 发现，彻头彻尾地失败了。在与不断发散的 lm-loss 作斗争的过程中，我们学到了很多:

![104B-fail](https://huggingface.co/blog/assets/86_bloom_megatron_deepspeed/104b-lm-loss.png)

我们也从 Megatron-LM 和 DeepSpeed 团队那里得到了相同的建议，在他们训得 [530B 模型](https://arxiv.org/abs/2201.11990) 后。最近发布的 [OPT-175B](https://arxiv.org/abs/2205.01068) 也报告说他们在 FP16 上训练得非常艰难。

所以早在一月份，我们就知道我们要在支持 BF16 格式的 A100 上进行训练。Olatunji Ruwase 开发了一个用来训练 BLOOM 的 `BF16Optimizer`。

如果您不熟悉这种数据格式，请查看它的 [位布局](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#bfloat16_floating-point_format)。BF16 格式的关键是它的指数位数与 FP32 相同，因此不会溢出，但 FP16 经常溢出！FP16 的最大数值范围为 64k，您只能进行较小数的乘法。例如你可以做 `250*250=62500`，但如果你尝试 `255*255=65025`，你就会溢出，这是导致训练出现问题的主要原因。这意味着你的权重必须保持很小。一种称为损失缩放 (loss scaling) 的技术有助于缓解这个问题，但是当模型变得非常大时，FP16 较小的数值范围仍然是一个问题。

BF16 没有这个问题，你可以很容易地做 `10_000*10_000=100_000_000`, 完全没问题。

当然，由于 BF16 和 FP16 的大小相同，均为 2 个字节，因此，没有免费的午餐，当使用 BF16 时，代价就是它的精度非常差。然而，你应该还记得我们在训练时采用的随机梯度下降法及其变体，该方法有点像蹒跚而行，如果你这步没有找到完美的方向其实没关系，你会在接下来的步骤中纠正自己。

无论使用 BF16 还是 FP16，都有一个权重副本始终在 FP32 中  —— 这是由优化器更新的内容。因此 16 位格式仅用于计算，优化器以全精度更新 FP32 权重，然后将它们转换为 16 位格式以用于下一次迭代。

所有 PyTorch 组件都已更新，以确保它们在 FP32 中执行任何累加，因此不会发生精度损失。

一个关键问题是梯度累积，它是流水线并行的主要特征之一，因为每个 micro batch 处理的梯度都会累积。在 FP32 中实现梯度累积以保证训练的精确性至关重要，这正是 `BF16Optimizer` 所做的。

除了其他改进之外，我们认为使用 BF16 混合精度训练将潜在的噩梦变成了一个相对平稳的过程，这可以从以下 lm 损失图中看出:

![176B - 损失](https://huggingface.co/blog/assets/86_bloom_megatron_deepspeed/176b-lm-loss.png)

## CUDA 融合核函数

GPU 主要做两件事。它可以将数据写到显存或从显存读数据，并对这些数据执行计算。当 GPU 忙于读写数据时， GPU 的计算单元就会空闲。如果我们想有效地利用 GPU，我们希望将空闲时间降至最低。

核函数是一组实现特定 PyTorch 操作的指令。例如，当你调用 `torch.add` 时，它会通过一个 [PyTorch 调度器](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)，它会根据输入张量及其他变量的取值来决定它应该运行哪些代码，最后运行它。CUDA 核函数使用 CUDA 来实现这些代码，因此只能在 NVIDIA GPU 上运行。

现在，当使用 GPU 计算 `c = torch.add (a, b); e = torch.max ([c,d])` 时，一般情况下，PyTorch 将执行的操作是启动两个单独的核函数，一个执行 `a` 和  `b` 的加法，另一个执行取 `c` 和 `d` 两者的最大值。在这种情况下，GPU 从其显存中获取 `a` 和 `b`，执行加法运算，然后将结果写回显存。然后它获取 `c` 和 `d` 并执行 max 操作，然后再次将结果写回显存。

如果我们要融合这两个操作，即将它们放入一个 “融合核函数” 中，然后启动那个内核，我们不会将中间结果 `c` 写到显存中，而是将其保留在 GPU 寄存器中，并且仅需要获取 `d` 来完成最后的计算。这节省了大量开销并防止 GPU 空闲，因此整个操作会更加高效。

融合核函数就是这样。它们主要将多个离散的计算和进出显存的数据移动替换为有很少数据移动的融合计算。此外，一些融合核函数会对操作进行数学变换，以便可以更快地执行某些计算组合。

为了快速高效地训练 BLOOM，有必要使用 Megatron-LM 提供的几个自定义 CUDA 融合核函数。特别地，有一个 LayerNorm 的融合核函数以及用于融合缩放、掩码和 softmax 这些操作的各种组合的核函数。Bias Add 也通过 PyTorch 的 JIT 功能与 GeLU 融合。这些操作都是瓶颈在内存的，因此将它们融合在一起以达到最大化每次显存读取后的计算量非常重要。因此，例如，在执行瓶颈在内存的 GeLU 操作时同时执行 Bias Add，运行时间并不会增加。这些核函数都可以在 [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM) 代码库 中找到。

## 数据集

Megatron-LM 的另一个重要特性是高效的数据加载器。在首次训练启动前，每个数据集中的每个样本都被分成固定序列长度 (BLOOM 为 2048) 的样本，并创建索引以对每个样本进行编号。基于训练超参，我们会确定每个数据集所需要参与的 epoch 数，并基于此创建一个有序的样本索引列表，然后打乱它。举个例子，如果一个数据集中有 10 个样本并应参与 2 个 epoch 的训练，则系统首先按 `[0, ..., 9, 0, ..., 9]` 顺序排好样本索引，然后打乱该顺序为数据集创建最终的全局顺序。请注意，这意味着训练不会简单地遍历整个数据集然后重复，你有可能在看到另一个样本之前看到同一个样本两次，但在训练结束时模型将只看到每个样本两次。这有助于确保整个训练过程中的训练曲线平滑。这些索引，包括每个样本在原始数据集中的偏移量，被保存到一个文件中，以避免每次开始训练时都重新计算它们。最后，可以将其中几个数据集以不同的权重混合到训练最终使用的数据中。

## 嵌入 LayerNorm

在我们努力阻止 104B 模型发散的过程中，我们发现在第一个层词嵌入层之后添加一个额外的 LayerNorm 可以使训练更加稳定。

该洞察来自对 bitsandbytes 的实验，[bitsandbytes](https://github.com/facebookresearch/bitsandbytes) 有一个 `StableEmbedding` 操作，它是一个带有 LayerNorm 的普通嵌入，其使用均匀 xavier 函数来初始化。

## 位置编码

基于论文 [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)，我们还用 AliBi 替换了普通的位置嵌入，它允许外推比训练模型的输入序列更长的输入序列。因此，即使我们训练时使用长度为 2048 的序列，模型也可以在推理过程中处理更长的序列。

## 训练中的困难

随着架构、硬件和软件的就位，我们得以在 2022 年 3 月上旬开始训练。然而，从那时起，事情其实并非一帆风顺。在本节中，我们将讨论我们遇到的一些主要障碍。

在训练开始之前，有很多问题需要弄清楚。特别是，我们发现了几个问题，这些问题只有在我们开始在 48 个节点上进行训练后才会出现，而不会在小规模时出现。例如，需要设 `CUDA_LAUNCH_BLOCKING=1` 来防止框架挂起，我们需要将优化器组分成更小的组，否则框架会再次挂起。你可以在 [训前编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md) 中详细了解这些内容。

训练期间遇到的主要问题类型是硬件故障。由于这是一个拥有大约 400 个 GPU 的新集群，平均每周我们会遇到 1-2 个 GPU 故障。我们每 3 小时 (100 次迭代) 保存一个检查点。因此，我们每周因硬件崩溃平均损失 1.5 小时的训练成果。Jean Zay 系统管理员随后将更换有故障的 GPU 并恢复节点。与此同时，我们有备用节点可供使用。

我们还遇到过多次导致 5-10 小时停机的各种其他问题，其中一些与 PyTorch 中的死锁错误有关，另一些则是由于磁盘空间不足。如果您对具体细节有兴趣，请参阅 [训练编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md)。

在对训练这个模型进行可行性分析时，所有这些停机时间都被计划在内了，我们也据此选择了合适的模型大小和我们希望模型消耗的数据量。因此，即使存在这些停机问题，我们还是成功地在预计时间内完成了训练。如前所述，它需要大约 100 万个计算时才能完成。

另一个问题是 SLURM 并非设计为供一组人使用。SLURM 作业由单个用户拥有，如果他们不在身边，则该组的其他成员无法对正在运行的作业执行任何操作。我们制定了一个终止方案，允许组中的其他用户终止当前进程，而不需要启动该进程的用户在场。这在 90% 的问题上都很有效。如果 SLURM 设计者读到这篇文章，请添加一个 Unix 组的概念，这样一个 SLURM 作业就可以由一个组拥有。

由于训练是全天候 24/7 进行的，我们需要有人随叫随到 - 但由于我们在欧洲和加拿大西海岸都有人，因此不需要有人携带传呼机，我们能很好地互相备份。当然，周末的训练也得有人看着。我们自动化了大部分事情，包括自动从硬件崩溃中恢复，但有时仍需要人工干预。

## 结论

训练中最困难和最紧张的部分是训练开始前的 2 个月。我们承受着尽快开始训练的巨大压力，因为资源分配的时间有限，我们直到最后一刻才接触到 A100。所以这是一个非常困难的时期，考虑到 `BF16Optimizer` 是在最后一刻编写出来的，我们需要调试它并修复各种 bug。正如上一节所述，我们发现了新问题，这些问题只有在我们开始在 48 个节点上进行训练后才会出现，并且不会在小规模时出现。

但是一旦我们把这些整理完，训练本身出奇的顺利，没有出现大的问题。大多数时候，我们只有一个人看着，只有少数几个人参与故障排除。我们得到了 Jean Zay 管理部门的大力支持，他们迅速解决了训练期间出现的大部分需求。

总的来说，这是一次超级紧张但回报颇丰的经历。

训练大型语言模型仍然是一项具有挑战性的任务，但我们希望通过公开构建和共享这项技术，其他人可以借鉴我们的经验。

## 资源

### 重要链接

- [主训练文档](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md)
- [Tensorboard](https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard)
- [训练用的 slurm 脚本](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/tr11-176B-ml.slurm)
- [训练记录](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md)

### 论文与文章

我们不可能在本文中详细解释所有内容，因此如果此处介绍的技术激起你的好奇心，使你想了解更多信息，请阅读以下论文:

Megatron-LM:

- [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473).
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)

DeepSpeed:

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [DeepSpeed: Extreme-scale model training for everyone](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

Megatron-LM 和 Deepspeeed 联合:

- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990).

ALiBi:

-  [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
- [What Language Model to Train if You Have One Million GPU Hours?](https://openreview.net/forum?id=rI7BL3fHIZq) - 你会在那里找到最终使得我们选择 ALiBi 的实验。

BitsNBytes:

- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861) (我们使用了该论文中的嵌入 LaynerNorm，但是论文的其他部分及其技术也很妙，我们没用 8 位优化器的唯一原因是我们已经使用 DeepSpeed-ZeRO 节省了优化器内存)。

## 博文致谢

非常感谢以下这些人，他们提出了很好的问题并帮助提高了文章的可读性 (按字母序):

* Britney Muller,
* Douwe Kiela,
* Jared Casper,
* Jeff Rasley,
* Julien Launay,
* Leandro von Werra,
* Omar Sanseviero,
* Stefan Schweter and
* Thomas Wang.

本文图表主要由 Chunte Lee 创作。
