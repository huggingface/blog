---
title: "Data Analytics with Apache Arrow and Hugging Face Datasets" 
thumbnail: /blog/assets/arrow-datasets/bla.png
authors:
- user: cakiki
- user: lhoestq
- user: mariosasko
---

# Data Analytics with Apache Arrow and Hugging Face Datasets

<!-- {blog_metadata} -->
<!-- {authors} -->

Hugging Face Datasets is a library that was designed to complement machine learning training workflows. It enables users to share, download, process, and format datasets to get them ready for training with any of the major machine learning frameworks, across all data modalities: audio, video, text, and tabular. In the Hugging Face ecosystem, the 🤗 `datasets` library is typically the first entrypoint of a machine learning project: it is therefore especially crucial it be performant and efficient.

| ![HF Libraries](./assets/arrow-datasets/hf-libraries.png) |
|:--:|
| <i>Hugging Face libraries mapped to the steps of a typical machine learning workflow. Source: <a href="https://github.com/nlp-with-transformers" rel="noopener" target="_blank" >Natural Language Processing with Transformers</a></i>|

One of the many strengths of 🤗 `datasets` is its ability to [quickly and efficiently](https://huggingface.co/docs/datasets/about_arrow) handle data that is too large fit into system memory. This is made possible by using the Apache [Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html) as its underlying memory model. This post will showcase the synergies between the two projects, and how using the Arrow ecosystem can supercharge your library, giving you access to more features than you initially bargained for. This post will show you how you can use 🤗 `datasets` for out-of-core data analytics to better understand your data, ahead of using it to train a model.

## Apache Arrow
According to the [official website](https://arrow.apache.org/), Apache Arrow is: (1) a column-oriented **standardized memory format**, (2) a set of **libraries** that implement said format and various utilities around it, and (3) an **ecosystem** of projects using them. 🤗 `datasets` is one of many projects of this ecosystem and leverages the Arrow memory format through the use of the [PyArrow library ](https://arrow.apache.org/docs/python/index.html): the official Python API of Apache Arrow. The Arrow format is further described as a:

> language-independent columnar memory format for flat and hierarchical data, organized for efficient analytic operations on modern hardware like CPUs and GPUs. The Arrow memory format also supports zero-copy reads for lightning-fast data access without serialization overhead.

Let's unpack this:
- **language indepedent**: the data representation in memory is the same, no matter the language or implementation.
- **columnar**: data is contiguous in memory and grouped by column, as opposed to by row.
- **efficient analytic operations**: the data is formatted in such a way that allows vectorized operations like [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data).
- **zero-copy reads with serialization overhead**: once data is in memory, moving it between different libraries (e.g. from `pandas` to `polars`), threads, or even different languages (e.g. from `python` to `rust`) is free.
TODO, add: constant time random access + memmaping

As a user, you typically won't need interact with Arrow directly, but rather use [libaries which use Arrow](https://arrow.apache.org/powered_by/) to represent tabular data in memory.

<!-- | ![Serialization](https://arrow.apache.org/img/copy.png) | ![Standardization](https://arrow.apache.org/img/shared.png)
|:--:|:--:|
|<i>TODO</i>|<i>TODO</i>|
<div align="center"> Source: <a href="https://arrow.apache.org/overview/" rel="noopener" target="_blank" >Apache Arrow Overview</a></div> -->

### Hugging Face Datasets
A Hugging Face Dataset is at its core an Arrow Table: a 2-dimensional data structure with a schema and named columns. When you load a dataset for the first time, it is downloaded locally, then serialized into 

### The Arrow Compute API

### Hugging Face Dataset Formats

```python
from datasets import load_dataset
dset = load_dataset("cakiki/stack-smol-xxl", split="train")
print(dset.num_rows, " rows")
print(dset.dataset_size / 1024**3, " GB")
>>> 12,962,249 rows
>>> 66.9516989979893 GB
```

## Mapping Compute Primitives

Scratchpad: synergy between the two libraries. success story of arrow. Come for the memory model / mmapping, stay for the data analytics.
