---
title: "Leveraging the Power of Apache Arrow with Hugging Face Datasets" 
thumbnail: /blog/assets/arrow-datasets/bla.png
authors:
- user: cakiki
- user: lhoestq
- user: 
---

# Leveraging the Power of Apache Arrow with Hugging Face Datasets

<!-- {blog_metadata} -->
<!-- {authors} -->

Your content here [...]

Hugging Face Datasets is a datasets library that was designed to complement machine learning training workflows.

## Apache Arrow and Arrow Tables

## Hugging Face Dataset Formats

```python
from datasets import load_dataset
dset = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train")
print(dset.num_rows)
print(dset.dataset_size / 1024**3)
>>> 12962249
>>> 66.9516989979893
```
## The Arrow Compute API

## Mapping Compute Primitives
