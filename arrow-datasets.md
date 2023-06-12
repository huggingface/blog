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

- **language indepedent**: the data representation in memory is the same, no matter the programming language or library.
- **columnar**: data is contiguous in memory and grouped by column, as opposed to by row.
- **efficient analytic operations**: the data is formatted in such a way that allows vectorized operations like [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data).
- **zero-copy reads with serialization overhead**: once data is in memory, moving it between different libraries (e.g. from `pandas` to `polars`), threads, or even different languages (e.g. from `python` to `rust`) is free.

Further features of Arrow that make it especially useful in machine learning worflows include:

- **Constant-time random access**: indexing into a specific row is a constant time operation
- **Efficient sequential scans**: the columnar memory layout makes for efficient iterating over the dataset
- **Data can be memory-mapped**: larger-than-memory datasets can be serialized onto disk and memory-mapped from there.
As a user, you typically won't need interact with Arrow directly, but rather use [libaries which use Arrow](https://arrow.apache.org/powered_by/) to represent tabular data in memory. However, as we'll see later in this post, even if the Arrow internals are abstracted away, being aware of them can give users access to more functionality than afforded by a given library.

<!-- | ![Serialization](https://arrow.apache.org/img/copy.png) | ![Standardization](https://arrow.apache.org/img/shared.png)
|:--:|:--:|
|<i>TODO</i>|<i>TODO</i>|
<div align="center"> Source: <a href="https://arrow.apache.org/overview/" rel="noopener" target="_blank" >Apache Arrow Overview</a></div> -->

### Hugging Face Datasets

A Hugging Face `datasets.Dataset` object is at its core a dataframe-like structure called an Arrow Table: a 2-dimensional data structure with a schema and named columns. When you load a dataset for the first time, it is serialized in batches into one ore more `.arrow` memory files in your cache directory, then finally mmapped from those files on disk. To peek behind the scenes of this process let's begin by downloading the dataset we'll be using, a subset of [the Stack](https://huggingface.co/datasets/bigcode/the-stack-dedup), a dataset that was used to train the [StarCoder](https://huggingface.co/bigcode/starcoder) family of code large language models.

```python
from datasets import load_dataset
dset = load_dataset("cakiki/stack-smol-xxl", split="train") # ~107GB of disk space required to run this
print(type(dset))
print(type(dset.data.table)) # dset.data.table is the underlying Arrow Table
print([shard["filename"].split("/")[-1] for shard in dset.cache_files])
print(len(dset.cache_files), "cache files")
>>> <class 'datasets.arrow_dataset.Dataset'> 
>>> <class 'pyarrow.lib.Table'>
>>> ['parquet-train-00000-of-00144.arrow', 'parquet-train-00001-of-00144.arrow', ... ] # These are the serialized files on disk. (Output trimmed) 
>>> 144 cache files
```

Caching onto disk is not limited to loading data, but also transforming it. Indeed, as we'll see in a second, when you operate on a dataset, be it through filtering, mapping, or structural transformations, these are also cached to disk and only need to run once. Before we move on to the next section, let's inspect the code dataset we just loaded:

```python
print(f'{dset.num_rows:,}', "rows")
print(dset.num_columns, "columns")
print(dset.dataset_size / 1024**3, " GB")
print(dset[:4])
>>> 11,658,586 rows
>>> 29 columns
>>> 73.18686429131776 GB
>>> {'hexsha': ['0a84ade1a79baf9000df6f66818fc3a568c24173', '0a84b588e484c6829ab1a6706db302acd8514288', '0a89bca33bdfce1f07390d394915f18e0d901651', '0a96a3eeaa089f8582180f803252e3739ab2661d'], 'size': [10354, 999, 43147, 597], 'ext': ['css', 'css', 'css', 'css'], 'lang': ['CSS', 'CSS', 'CSS', 'CSS'], ...} # By default, slicing a dataset returns a dictionary of the columns as lists of regular python objects. Makes sense since arrow is a columnar format. (Output trimmed)
```

### Hugging Face Dataset Formats


## Mapping Compute Primitives

Scratchpad: synergy between the two libraries. success story of arrow. Come for the memory model / mmapping, stay for the data analytics.

## Further Resources

- 
- 