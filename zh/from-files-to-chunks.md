---
title: "从文件到块：提高 Hugging Face 存储效率"
thumbnail: /blog/assets/from-files-to-chunks/thumbnail.png
authors:
  - user: jsulz
  - user: erinys
translators:
  - user: smartisan
  - user: zhongdongy
    proofreader: true
---

# 从文件到块: 提高 Hugging Face 存储效率

Hugging Face 在 [Git LFS 仓库](https://huggingface.co/docs/hub/en/repositories-getting-started#requirements) 中存储了超过 [30 PB 的模型、数据集和 Spaces](https://huggingface.co/spaces/xet-team/lfs-analysis)。由于 Git 在文件级别进行存储和版本控制，任何文件的修改都需要重新上传整个文件。这在 Hub 上会产生高昂的成本，因为平均每个 Parquet 和 CSV 文件大小在 200-300 MB 之间，Safetensor 文件约 1 GB，而 GGUF 文件甚至可能超过 8 GB。设想一下，仅仅修改 GGUF 文件中的一行元数据，就需要等待数 GB 大小的文件重新上传。除了耗费用户时间和传输成本外，Git LFS 还需要保存文件的两个完整版本，这进一步增加了存储开销。

下图展示了 Hub 上各类仓库 (模型、数据集和 Spaces) 中 LFS 存储容量在 2022 年 3 月至 2024 年 9 月期间的增长趋势:

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/from-files-to-chunks/lfs-analysis-min.png" alt="LFS Storage Growth" width=90%>
</p>

Hugging Face 的 Xet 团队正在采用一种创新的存储方案: 将文件分块存储。通过只传输发生变化的数据块，我们可以显著提升存储效率和迭代速度，同时确保用户能可靠地访问不断演进的数据集和模型。下面让我们详细了解其工作原理。

## 基于内容的分块原理

我们采用的分块方法称为基于内容的分块 (Content-Defined Chunking，CDC)。与将文件视为不可分割的整体不同，CDC 根据文件内容本身来确定边界，将文件划分为大小可变的数据块。为了计算这些块的边界，我们使用 [滚动哈希算法](https://en.wikipedia.org/wiki/Rolling_hash) 来扫描文件的字节序列。

让我们通过一个简单的例子来说明:

```bash
transformerstransformerstransformers
```

这里我们用文本来演示，但实际上这个过程适用于任何字节序列。

滚动哈希算法通过在数据上滑动固定大小的窗口来计算哈希值。比如，当窗口长度为 4 时，算法会依次计算 `tran` 、 `rans` 、 `ansf` 等字符序列的哈希值，直到处理完整个文件。

当某个位置的哈希值满足预设条件时，就会在该处设置块的边界。例如，可以设置如下条件:

```python
hash(data) % 2^12 == 0
```

如果序列 `mers` 的哈希值满足这个条件，那么文件就会被分成三个块:

```bash
transformers | transformers | transformers
```

系统会计算这些块的哈希值，建立块哈希值到实际内容的映射，并最终将它们存储在基于内容寻址的存储系统 (Content-Addressed Storage，CAS) 中。由于这三个块完全相同，CAS 只需要存储一个块的实际内容，从而自动实现了数据去重。🪄

## 处理插入和删除操作

当文件内容发生变化时，CDC 的优势就体现出来了: 它能够精确定位变化的部分，高效处理插入和删除操作。让我们看一个具体示例，在原文件中插入 `super` 后:

```bash
transformerstransformerssupertransformers
```

使用相同的边界条件重新应用滚动哈希算法，新的分块结果如下:

```bash
transformers | transformers | supertransformers
```

前两个块的内容系统中已经存在，无需重新存储。只有 `supertransformers` 是新的数据块，因此保存这个更新版本只需要上传和存储这一个新块即可。

为了验证这种优化方案在实际应用中的效果，我们将 XetHub 上基于 CDC 的存储实现与 Git LFS 进行了对比测试。在三个迭代开发场景中，我们发现存储和传输性能始终提升了 50%。其中一个典型案例是 [CORD-19 数据集](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html)——这是一个在 2020 年至 2022 年间持续更新的 COVID-19 研究论文集合，共有 50 次增量更新。下表对比了两种存储方案的性能指标:

| 指标 | 基于 Git LFS 的仓库 | 基于 Xet 的仓库 |
| --------------------- | ------------------------- | --------------------- |
| 平均下载时间 | 51 分钟 | 19 分钟 |
| 平均上传时间 | 47 分钟 | 24 分钟 |
| 存储占用 | 8.9 GB | 3.52 GB |

通过只传输和保存变化的数据块，再结合改进的压缩算法和优化的网络请求，基于 Xet 的 CDC 方案显著缩短了上传和下载时间，同时大幅降低了存储多个版本所需的空间。想深入了解测试细节？请查看我们的 [完整基准测试报告](https://xethub.com/blog/benchmarking-the-modern-development-experience)。

## CDC 技术对 Hub 的影响

那么，CDC 如何应用于 Hugging Face Hub 上的各类文件呢？为了直观展示 CDC 在文件集合上的存储节省潜力，我们开发了一个简单的 [重复数据删除估算工具](https://github.com/huggingface/dedupe_estimator)。我们用这个工具分析了 [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) 仓库中 `model.safetensors` 文件的两个版本，得到了以下结果:

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/from-files-to-chunks/safetensors_dedupe_image.png" alt="Safetensors Deduplication" width=40%>
</p>

图中的绿色区域表示两个版本之间内容的重叠部分，这意味着我们可以在单个文件内部以及不同版本之间进行有效的数据去重。

|           | Git LFS 存储占用 | Xet 存储占用 |
| --------- | ------------------------ | --------------------------- |
| 版本 1 | 664 MB                   | 509 MB                      |
| 版本 2 | 548 MB                   | 136 MB                      |
| 总计     | 1.2 GB                   | 645 MB                      |

在这个案例中，采用基于 Xet 的存储方案不仅大大缩短了第二个版本的上传和下载时间，还将总存储空间减少了 53%。通过进一步的压缩优化，我们预计还能额外节省 10% 的空间。

我们对 Hub 上的仓库进行的初步研究显示，CDC 技术对某些类型的数据特别有效。例如，微调模型通常只修改部分参数，大部分模型权重在不同版本间保持不变，这使它们非常适合使用数据去重技术。同样，模型检查点文件也是理想的应用场景，因为相邻检查点之间的变化往往很小。这两类文件都展现出 30-85% 的去重比率。考虑到 PyTorch 模型检查点在 Hub 上占用了约 200 TB 的存储空间，如果能达到 50% 的去重率，我们可以立即节省 100 TB 的存储空间，并在未来每月减少 7-8 TB 的增长。

除了降低存储成本，块级数据去重还能显著提升数据传输效率，因为只需要传输实际发生变化的数据块。这对于需要频繁处理多个模型版本或数据集版本的团队来说尤其重要，可以大大减少等待时间，提高工作效率。

目前，我们团队正在开发 Hub 的基于 Xet 存储的概念验证系统，计划在 2025 年初推出首批基于 Xet 的仓库。欢迎 [关注我们的团队](https://huggingface.co/xet-team)，了解更多技术进展。我们将持续分享在全球分布式仓库扩展 CDC、优化网络性能、保护数据隐私以及并行化分块算法等方面的研究成果。