---
title: "Leveraging the Power of Apache Arrow with Hugging Face Datasets" 
thumbnail: /blog/assets/arrow-datasets/bla.png
authors:
- user: cakiki
- user: lhoestq
- user: 
---

# Leveraging the Power of Apache Arrow with Hugging Face Datasets OR Data Analytics with Apache Arrow and Hugging Face Datasets

<!-- {blog_metadata} -->
<!-- {authors} -->

Your content here [...]

Hugging Face Datasets is a library that was designed to complement machine learning training workflows. It enables users to share, download, preprocess, and format datasets for model training. 

One of the remarkable features of `datasets` is its ability to efficiently handle data that is larger than memory. This is made possible by using Apache Arrow as an underlying memory model. This post will explore the synergies between the two projects, and how using the Arrow ecosystem can supercharge your library, giving you access to more features than you bargained for.

synergy between the two libraries. success story of arrow. Come for the memory model / mmapping, stay for the data analytics.

## Apache Arrow and Arrow Tables

## Hugging Face Dataset Formats

```python
from datasets import load_dataset
dset = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train")
print(dset.num_rows, " rows")
print(dset.dataset_size / 1024**3, " GB")
>>> 12962249 rows
>>> 66.9516989979893 GB
```
## The Arrow Compute API

## Mapping Compute Primitives
