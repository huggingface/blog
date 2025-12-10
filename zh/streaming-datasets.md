---
title: "流式数据集：效率提升 100 倍"
thumbnail: /blog/assets/streaming_datasets/streaming_datasets.png
authors:
- user: andito
- user: lhoestq
- user: burtenshaw
- user: pcuenq
- user: merve
translators:
- user: chenglu
---

# 流式数据集：效率提升 100 倍！

## 快速了解（TLDR）

> 现在只需一行代码，就能通过 `load_dataset('dataset', streaming=True)` 以流式方式加载数据集，无需下载！
>
> 无需复杂配置、不占磁盘空间、不再担心 “磁盘已满” 或 429 请求过多错误，立即开始训练 TB 级数据集！
> 性能非常强劲：在 64×H100、256 个并发 worker 环境下，流式加载速度甚至超过本地 SSD！
> 我们优化后的流式系统：请求数减少 100 倍 → 数据解析速度提升 10 倍 → 样本处理速度翻倍 → 即使在 256 个并发 worker 下也 0 崩溃。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/streaming-dark.gif" width="800" height="auto" alt="数据流式加载可视化">


在机器学习中，特别是在处理 TB 级别的数据时，数据加载一直是个大难题。我们自己在训练 [SmolLM3](https://huggingface.co/blog/smollm3) 时也深有体会，有段时间每次训练前都得等上 3 小时下载数据。

虽然 `datasets` 库早就支持流式加载，但在大规模训练中依然面临瓶颈。今天，这一切都变了 🔥。我们花了几个月优化后端，全面提升流式数据集的速度与效率。

那我们到底做了哪些优化？⤵️


## 一样简单的 API，更强大的性能

首先最重要的一点：**改进后的接口依然兼容原来的用法**。你只需要加上 `streaming=True`，就能流式加载 Hugging Face 上的任意数据集，依旧简单直接。🚀

```python
from datasets import load_dataset

# 以流式方式加载数据集，而非先下载
dataset = load_dataset("HuggingFaceM4/FineVisionMax", split="train", streaming=True)
# 获取第一个样本
print(next(iter(dataset)))
```

全球成千上万的 AI 开发者每天都在使用 `datasets`，现在他们无需改动任何代码，就能直接享受到更高的性能。


## 问题挑战：大规模流式加载

流式加载一直是快速了解数据集的好方法，但在训练模型时，大多数人仍然选择将数据预先下载到本地，或使用 S3 等云存储——我们在训练 [SmolVLM](https://huggingface.co/blog/smolvlm2) 时也是这么做的。

我们希望改变这种情况，于是在开发 [nanoVLM](https://github.com/huggingface/nanoVLM) 时，尝试直接从 Hugging Face Hub 进行流式读取。

但很快就遇到一个严重问题：**一次测试运行在不到一分钟的时间内发出了超过 10 万个请求，结果我们的 IP 被 Hub 屏蔽了！**😅

问题的根源在于：**每个 `DataLoader` 的 worker 都在独立初始化数据集**，这导致大量冗余请求，形成了“请求风暴”，其中大部分其实是没必要的。

于是我们对启动逻辑进行了深度优化，**最终将启动请求量减少了 100 倍**。总体性能提升如下：

* **数据文件解析速度：提升 10 倍**
* **启动请求效率：提高最多 100 倍**
* **流式速度：提升最多 2 倍**
* **在途请求效率：提升最多 2 倍**


## 技术揭秘：我们具体改了什么？

我们主要优化了两个阶段：**启动阶段** 和 **流式加载阶段**。


### 1. 启动优化 ⚡️

初始的数据文件解析阶段会触发大量请求。我们进行了以下两项关键优化：

* **持久化数据文件缓存**：现在所有 `DataLoader` worker 会共享数据文件列表缓存。第一个 worker 从 Hub 获取文件列表，其余 worker 直接从本地缓存中读取，从而**几乎完全消除启动阶段的请求**，大幅缩短加载时间，彻底告别“请求风暴”。

* **优化文件解析逻辑**：我们精简了初始 worker 向 Hub 请求文件列表的 API 调用数量，将多个请求进行打包处理，进一步**降低启动延迟**。

### 2. 流式加载优化 🏎️

为了提升训练过程中的流式吞吐量，我们新增了两个关键功能：

* **Parquet 数据预取（Prefetching）**：我们为 Parquet 格式的数据集启用了预取功能。这意味着，在模型处理当前数据块的同时，`datasets` 库会在后台**提前加载下一块数据**。这样可以让整个数据管道始终保持“满负荷”，确保 GPU 不会因等待数据而处于空闲状态，大大提升训练效率。

* **可配置缓冲机制（Buffering）**：针对高级用户，我们开放了缓冲区的配置参数，支持自定义设置**预取数量和数据块大小**，方便根据自身硬件和网络情况进行 I/O 优化。


以下是如何将默认流式请求大小从 32MiB 提升到 128MiB，并启用预取的示例代码：

```python
import pyarrow
import pyarrow.dataset

fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
    cache_options=pyarrow.CacheOptions(
        prefetch_limit=1,
        range_size_limit=128 << 20
    ),
)
ds = load_dataset(parquet_dataset_id, streaming=True, fragment_scan_options=fragment_scan_options)
```

通过这些优化，你的数据加载速度可以提升一倍，训练效率更高！


## 为什么比 S3 还快？背后是 Xet 技术

Hugging Face 使用了 **Xet 存储系统**：这是一种去重式存储方案，上传和下载速度极快。与传统远程存储不同，Xet 会跳过重复数据，只传输独特内容。

比如，在 Hugging Face 上传大型数据集时，Xet 的去重机制大幅减少了数据传输量，上传更快；数据一上传完，就可以立即开始流式读取。

对于 Parquet 文件，Xet 利用 [Parquet 内容定义切块（CDC）](https://huggingface.co/blog/parquet-cdc) 来实现去重，进一步加快传输速度。

此外，我们还推出了 `pyspark_huggingface` 包，支持 Spark 直接读写 HF 数据集，内置对 Parquet CDC 和 Xet 的支持，大幅加快大数据处理。

---

## 想自定义流式管道？可以！

有些数据格式 `datasets` 库还不支持，或者你希望获得更高的控制权，我们也提供了强大的自定义流式能力。

`huggingface_hub` 库中的 [HfFileSystem](https://huggingface.co/docs/huggingface_hub/guides/hf_file_system) 可高效读取远程数据集文件：

```python
from huggingface_hub import HfFileSystem

path = f"hf://datasets/{dataset_id}/{path_in_repo}"
with HfFileSystem().open(path) as f:
    # 使用 .read() 或 .readline() 流式读取数据
    # 也支持 .seek() 进行随机访问
```

将 `HfFileSystem` 传入 PyTorch 的 `DataLoader` 时，会**复用 `.ls()` 和 `.glob()` 的缓存结果**，从而**避免在列举数据文件时产生额外的网络请求**，进一步提升流式加载的效率和稳定性。


## 极限测试：我们把它用在 nanoVLM 上了！

目前我们正在使用这些流式优化功能训练下一代 SmolVLM 模型。得益于新改进，流式加载比我们集群上的多层硬盘系统还要快，**几乎等同于从本地 SSD 读取数据的速度**！

过去，为了避免慢速网络，我们还要把数据拷贝到本地 SSD——整个过程花费 3 小时。而现在，直接流式加载，训练马上开始！

更多细节请看：[nanoVLM GitHub](https://github.com/huggingface/nanoVLM)


## 快速上手，立见成效

这些强大的新功能已经集成到 `datasets` 和 `huggingface_hub` 库中。想要体验全新的流式加载性能，只需升级你的库版本，并查阅[官方文档](https://huggingface.co/docs/datasets/stream)即可开始使用：

```bash
pip install --upgrade datasets huggingface_hub
```

为庆祝这一更新，我们已将 FineVision 所有数据源合并并预先打乱成一个统一数据集：[FineVisionMax](https://huggingface.co/datasets/HuggingFaceM4/FineVisionMax)。你可以直接用它来训练 VLM 模型，无需手动处理多个数据集！

```python
from datasets import load_dataset

# 以流式方式加载数据集，而非先下载
dataset = load_dataset("HuggingFaceM4/FineVisionMax", split="train", streaming=True)
# 获取第一个样本
print(next(iter(dataset)))
```

想了解我们是如何大规模运行的？欢迎查看：[nanoVLM 项目](https://github.com/huggingface/nanoVLM)

祝你流式加载愉快！🤗
