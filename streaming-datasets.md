---
title: Streaming datasets: 100x More Efficient
thumbnail: /blog/assets/streaming_datasets/streaming_datasets.png
authors:
- user: andito
- user: lhoestq
- user: burtenshaw
- user: pcuenq
---


## TLDR

> We boosted `load_dataset('dataset', streaming=True)`, streaming datasets without downloading them with one line of code!
> 
> Start training on multi-TB datasets immediately, without complex setups, downloading, no "disk out of space", or 429 “stop requesting!” errors.   
> It's super fast! Outrunning our local SSDs when training on 64xH100 with 256 workers downloading data.
> We've improved streaming to have 100x fewer requests, → 10× faster data resolution → 2x sample/sec, → 0 worker crashes at 256 concurrent workers.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/streaming-dark.gif" width="800" height="auto" alt="Visualization of a dataset being streamed">

## Streaming datasets: 100x More Efficient

Loading data, especially at the terabyte scale, is a major pain in any machine learning workflow. We suffered this while training [SmolLM3](https://huggingface.co/blog/smollm3), at one point we had to wait 3 hours before each run to download enough data. 

Streaming has always been possible in the `datasets` library, but large scale training with massive datasets remained a challenge. That changes today 🔥. We spent a few months improving the backend, focusing on streaming datasets to make it faster and more efficient.

What did we do exactly? ⤵️

## Streaming: The Same Easy API

First things first: our changes are backwards compatible. You can still stream any dataset from the Hub with the same simple `streaming=True` flag. It's as easy as ever. 🚀

```python
from datasets import load_dataset

# Stream a dataset instead of downloading it
dataset = load_dataset("HuggingFaceM4/FineVisionMax", split="train", streaming=True)
# Get the first example
print(next(iter(dataset)))
```

Thousands of AI developers around the world use `datasets` daily; they should just get improved performance with zero extra work.

## The Challenge: Streaming at Scale

Streaming was a lifesaver to quickly understand a dataset, but to train models, people were usually downloading the data locally, or using a cloud storage service such as S3. That's what we were doing for training [SmolVLM](https://huggingface.co/blog/smolvlm2), we had all of our data on S3 and were streaming directly from it.

But we wanted to change that. We started experimenting with streaming directly from the Hub when we were developing [nanoVLM](https://github.com/huggingface/nanoVLM) and started uncovering the issues: our test run generated over 100,000 requests in under a minute, which got our IP blocked by the Hub! 😅 The issue comes from every `DataLoader` worker initializing the dataset independently. As we dug deeper, we found that this creates a storm of redundant requests, many of which are unnecessary. Our changes ultimately reduced startup requests by a factor of 100. In total, our improvements delivered:

- Data files resolution time: 10x faster
- Startup requests: Up to 100x more efficient
- Streaming speed: Up to 2x faster
- In-flight requests: Up to 2x more efficient

## Under the Hood: What We Improved

So, what changed? We focused on two phases: startup and streaming.

**1. Startup⚡️**
The initial resolution of data files was creating a ton of requests. We made two major changes:
- Persistent Data Files Cache: We are now caching the list of data files across all DataLoader workers. The first worker resolves the file list from the Hub. All others workers read directly from this local cache, virtually eliminating startup requests and slashing resolution time. No more request storms!
- Optimized Resolution Logic: We also minimized the number of API calls required for that initial worker to fetch the file list. We now bundle the necessary requests as efficiently as possible, reducing latency even further. 

**2. Streaming 🏎️**
To improve throughput during streaming itself, we've introduced two new features:
- Prefetching for Parquet: We enabled prefetching for Parquet datasets. This means that while your model is processing the current chunk of data, the datasets library is already fetching the next chunk in the background. This keeps the data pipeline full and ensures your GPU is never left waiting for data.
- Configurable Buffering: Advanced users can now fine-tune streaming performance for their specific hardware and network setup. We've exposed options to configure the buffer's block size and the prefetch volume, giving you maximum control to optimize I/O.

This is how we can increase the minimum request size when streaming from 32MiB (default) to 128MiB and configure prefetching:

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

Together, these improvements can double your data throughput, allowing you to train faster and more efficiently.

## How are we faster than plain S3: Xet

Hugging Face uses Xet: a dedupe-based storage which enables fast deduped uploads and downloads. Unlike traditional remote storage, data transfers are faster on Xet because duplicated data is only transferred once. For example: uploading a large scale dataset to Hugging Face leverages Xet which accelerates uploads. Once the dataset is uploaded, it can be streamed right away.

Deduplication for Parquet is enabled through [Parquet Content Defined Chunking (CDC)](https://huggingface.co/blog/parquet-cdc). Thanks to Parquet CDC and Xet deduplication, uploading datasets on Hugging Face is faster than on any traditional remote storage.

This is supported by our `pyspark_huggingface` package, a Spark Data Source to read/write HF datasets. It includes Parquet CDC and Xet support, accelerating data transfers on HF dramatically.

## Need a custom streaming pipeline ?

Some data file formats are not supported in `datasets`, and sometimes there is a need for more control, so we made it easy to build custom streaming pipelines. This has been battle-tested in the LeRobot library to sample video frames, and in the `WebDataset` library to stream TAR archives.

We improved the [HfFileSystem](https://huggingface.co/docs/huggingface_hub/guides/hf_file_system) in the `huggingface_hub` library to efficiently read files from remote Hugging Face dataset repositories and stream data:

```python
from huggingface_hub import HfFileSystem

path = f"hf://datasets/{dataset_id}/{path_in_repo}"
with HfFileSystem().open(path) as f:
    # loop with .read() or .readline() to stream data
    # or do random access with .seek()
```

Passing a `HfFileSystem` to a torch `DataLoader` reuses the cached results from `.ls()` and `.glob()` which eliminates the need for additional requests when listing data files.


## Push streaming to the limit

We're now using these streaming enhancements in nanoVLM to train the next generation of SmolVLMs. With these tweaks, we achieve better performance from streaming than from training on our cluster's hierarchical hard disk setup. In fact, streaming is now as fast as reading the data from local SSDs! Previously, transferring data to local SSDs was the process that used to delay our trainings by three-hours. For more details, check out our GitHub.


## Get Started and See the Difference

These powerful new features landed in the datasets and huggingface_hub libraries. To take advantage of them, simply update your libraries and check out [the documentation](https://huggingface.co/docs/datasets/stream):

```Bash
pip install --upgrade datasets huggingface_hub
```

To celebrate this, we preconcatenated and shuffled all the data sources in FineVision into [FineVisionMax](https://huggingface.co/datasets/HuggingFaceM4/FineVisionMax). You can use this single combined dataset to train your VLM – no need to handle multiple datasets manually!

```python
from datasets import load_dataset

# Stream a dataset instead of downloading it
dataset = load_dataset("HuggingFaceM4/FineVisionMax", split="train", streaming=True)
# Get the first example
print(next(iter(dataset)))
```

And you can see how we do it at scale in [nanoVLM](https://github.com/huggingface/nanoVLM)!

Happy streaming! 🤗

