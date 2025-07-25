---
title: "Parquet Content-Defined Chunking"
thumbnail: /blog/assets/parquet-cdc/thumbnail.png
authors:
- user: kszucs
---

# Parquet Content-Defined Chunking

Reduce Parquet file upload and download times on Hugging Face Hub by leveraging the new Xet storage layer and Apache Arrow’s Parquet Content-Defined Chunking (CDC) feature enabling more efficient and scalable data workflows.

**TL;DR:** Parquet Content-Defined Chunking (CDC) is now available in PyArrow and Pandas, enabling efficient deduplication of Parquet files on content-addressable storage systems like Hugging Face's Xet storage layer. CDC dramatically reduces data transfer and storage costs by uploading or downloading only the changed data chunks. Enable CDC by passing the `use_content_defined_chunking` argument:

```python
pq.write_table(table, "hf://user/repo/file.parquet", use_content_defined_chunking=True)
df.to_parquet("hf://user/repo/file.parquet", use_content_defined_chunking=True)
```

```python 
import pandas as pd
import pyarrow.parquet as pq

df.to_parquet("hf://datasets/{user}/{repo}/path.parquet", use_content_defined_chunking=True)
pq.write_table(table, "hf://datasets/{user}/{repo}/path.parquet", use_content_defined_chunking=True)
```

## Table of Contents

- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Different Use Cases for Parquet Deduplication](#different-use-cases-for-parquet-deduplication)
    - [1. Re-uploading an Exact Copies of the Table](#1-re-uploading-an-exact-copies-of-the-table)
    - [2. Adding and Removing Columns from the Table](#2-adding-and-removing-columns-from-the-table)
    - [3. Changing Column Types in the Table](#3-changing-column-types-in-the-table)
    - [4. Appending New Rows and Concatenating Tables](#4-appending-new-rows-and-concatenating-tables)
    - [5. Inserting / Deleting Rows in the Table](#5-inserting--deleting-rows-in-the-table)
    - [6. Using Different Row-group Sizes](#6-using-different-row-group-sizes)
    - [7. Using Varying File-Level Splits](#7-using-varying-file-level-splits)
- [Using Parquet CDC feature with Pandas](#using-parquet-cdc-feature-with-pandas)
- [References](#references)
- [Conclusion](#conclusion)

## Introduction

Apache Parquet is a columnar storage format that is widely used in the data engineering community. 

As of today, Hugging Face hosts nearly 21 PB of datasets, with Parquet files alone accounting for over 4 PB of that storage. Optimizing Parquet storage is therefore a high priority.
Hugging Face has introduced a new storage layer called [Xet](https://huggingface.co/blog/xet-on-the-hub) that leverages content-defined chunking to efficiently deduplicate chunks of data reducing storage costs and improving download/upload speeds.

While Xet is format agnostic, Parquet's layout and column-chunk (data page) based compression can produce entirely different byte-level representations for data with minor changes, leading to suboptimal deduplication performance. To address this, the Parquet files should be written in a way that minimizes the byte-level differences between similar data, which is where content-defined chunking (CDC) comes into play.

Let's explore the performance benefits of the new Parquet CDC feature used alongside Hugging Face's Xet storage layer.

## Data Preparation

For demonstration purposes, we will use a manageable sized subset of [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) dataset.


```python
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def shuffle_table(table, seed=40):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(table))
    return table.take(indices)


# download the dataset from Hugging Face Hub into local cache
path = hf_hub_download(
    repo_id="Open-Orca/OpenOrca", 
    filename="3_5M-GPT3_5-Augmented.parquet", 
    repo_type="dataset"
)

# read the cached parquet file into a PyArrow table 
orca = pq.read_table(path, schema=pa.schema([
    pa.field("id", pa.string()),
    pa.field("system_prompt", pa.string()),
    pa.field("question", pa.large_string()),
    pa.field("response", pa.large_string()),
]))

# augment the table with some additional columns
orca = orca.add_column(
    orca.schema.get_field_index("question"),
    "question_length",
    pc.utf8_length(orca["question"])
)
orca = orca.add_column(
    orca.schema.get_field_index("response"),
    "response_length",
    pc.utf8_length(orca["response"])
)

# shuffle the table to make it unique to the Xet storage
orca = shuffle_table(orca)

# limit the table to the first 100,000 rows 
table = orca[:100_000]

# take a look at the first 3 rows of the table
table[:3].to_pandas()
```



|    | id           | system_prompt                                              | question_length | question                                                      | response_length | response                                                      |
|---:|:-------------|:----------------------------------------------------------|----------------:|:--------------------------------------------------------------|----------------:|:--------------------------------------------------------------|
| 0  | cot.64099    | You are an AI assistant that helps people find...          | 241             | Consider the question. What is the euphrates l...             | 1663            | The question is asking what the Euphrates Rive...             |
| 1  | flan.1206442 | You are an AI assistant. You will be given a t...          | 230             | Single/multi-select question: Is it possible t...              | 751             | It is not possible to conclude that the cowboy...              |
| 2  | t0.1170225   | You are an AI assistant. User will you give yo...          | 1484            | Q:I'm taking a test and have to guess the righ...              | 128             | The passage mainly tells us what things are im...              |



### Upload the table as a Parquet file to Hugging Face Hub

Since [pyarrow>=21.0.0](https://github.com/apache/arrow/pull/45089) we can use Hugging Face URIs in the `pyarrow` functions to directly read and write parquet (and other file formats) files to the Hub using the `hf://` URI scheme.


```python
>>> pq.write_table(table, "hf://datasets/kszucs/pq/orca.parquet")
New Data Upload: 100%|███████████████████████████████████████████████| 96.1MB / 96.1MB, 48.0kB/s  
Total Bytes:  96.1M
Total Transfer:  96.1M
```


We can see that the table has been uploaded entirely (total bytes == total transfer) as new data because it is not known to the Xet storage layer yet. Now read it back as a `pyarrow` table:


```python
downloaded_table = pq.read_table("hf://datasets/kszucs/pq/orca.parquet")
assert downloaded_table.equals(table)
```

Note that all `pyarrow` functions that accept a file path also accept a Hugging Face URI, like [pyarrow datasets](https://arrow.apache.org/docs/python/dataset.html), 
[CSV functions](https://arrow.apache.org/docs/python/generated/pyarrow.csv.read_csv.html), [incremental Parquet writer](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html) or reading only the parquet metadata:


```python
pq.read_metadata("hf://datasets/kszucs/pq/orca.parquet")
```




    <pyarrow._parquet.FileMetaData object at 0x16ebfa980>
      created_by: parquet-cpp-arrow version 21.0.0-SNAPSHOT
      num_columns: 6
      num_rows: 100000
      num_row_groups: 1
      format_version: 2.6
      serialized_size: 4143



## Different Use Cases for Parquet Deduplication

To demonstrate the effectiveness of the content-defined chunking feature, we will try out how it performs in case of:
1. Re-uploading exact copies of the table
2. Adding/removing columns from the table
3. Changing column types in the table
4. Appending new rows and concatenating tables
5. Inserting / deleting rows in the table
6. Change row-group size of the table
7. Using Varying File-Level Splits


### 1. Re-uploading an Exact Copies of the Table

While this use case sounds trivial, traditional file systems do not deduplicate files resulting in full re-upload and re-download of the data. In contrast, a system utilizing content-defined chunking can recognize that the file content is identical and avoid unnecessary data transfer.


```python
>>> pq.write_table(table, "hf://datasets/kszucs/pq/orca-copy.parquet")
New Data Upload: |                                                   |  0.00B /  0.00B,  0.00B/s  
Total Bytes:  96.1M
Total Transfer:  0.00
```


We can see that no new data has been uploaded, and the operation was instantaneous. Now let's see what happens if we upload the same file again but to a different repository:



```python
>>> pq.write_table(table, "hf://datasets/kszucs/pq-copy/orca-copy-again.parquet")
New Data Upload: |                                                   |  0.00B /  0.00B,  0.00B/s  
Total Bytes:  96.1M
Total Transfer:  0.00
```


The upload was instantaneous again since deduplication works across repositories as well. This is a key feature of the Xet storage layer, allowing efficient data sharing and collaboration. You can read more about the details and scaling challenges in the [From Chunks to Blocks: Accelerating Uploads and Downloads on the Hub](https://huggingface.co/blog/from-chunks-to-blocks) blog post.

### 2. Adding and Removing Columns from the Table

First write out the original and changed tables to local parquet files to see their sizes:


```python
table_with_new_columns = table.add_column(
    table.schema.get_field_index("response"),
    "response_short",
    pc.utf8_slice_codeunits(table["response"], 0, 10)
)
table_with_removed_columns = table.drop(["response"])
    
pq.write_table(table, "/tmp/original.parquet")
pq.write_table(table_with_new_columns, "/tmp/with-new-columns.parquet")
pq.write_table(table_with_removed_columns, "/tmp/with-removed-columns.parquet")
```


```python
!ls -lah /tmp/*.parquet
```

    -rw-r--r--  1 kszucs  wheel    92M Jul 22 14:47 /tmp/original.parquet
    -rw-r--r--  1 kszucs  wheel    92M Jul 22 14:47 /tmp/with-new-columns.parquet
    -rw-r--r--  1 kszucs  wheel    67M Jul 22 14:47 /tmp/with-removed-columns.parquet


Now upload them to Hugging Face to see how much data is actually transferred:


```python
>>> pq.write_table(table_with_new_columns, "hf://datasets/kszucs/pq/orca-added-columns.parquet")
New Data Upload: 100%|███████████████████████████████████████████████|  575kB /  575kB,  288kB/s  
Total Bytes:  96.6M
Total Transfer:  575k
```


We can see that only the new columns and the new parquet metadata placed in the file's footer were uploaded, while the original data was not transferred again. This is a huge benefit of the Xet storage layer, as it allows us to efficiently add new columns without transferring the entire dataset again.

Same applies to removing columns, as we can see below:


```python
>>> pq.write_table(table_with_removed_columns, "hf://datasets/kszucs/pq/orca-removed-columns.parquet")
New Data Upload: 100%|███████████████████████████████████████████████| 37.7kB / 37.7kB, 27.0kB/s  
Total Bytes:  70.6M
Total Transfer:  37.7k
```


To have a better understanding of what has been uploaded, we can visualize the differences between the two parquet files using the [deduplication estimation tool](https://github.com/huggingface/dataset-dedupe-estimator):


```python
from de import visualize

visualize(table, table_with_new_columns, title="With New Columns", prefix="orca")
```


#### With New Columns

| Compression | Vanilla Parquet |
|:---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-with-new-columns-nocdc.parquet.png) |
| Dedup Stats | 157.4 MB / 313.8 MB = 50%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-with-new-columns-nocdc.parquet.png) |
| Dedup Stats | 96.7 MB / 192.7 MB = 50%|



Adding two new columns mean that we have unseen data pages which must be transferred (highlighted in red), but the rest of the data remains unchanged (highlighted in green), so it is not transferred again. Note the small red area in the footer metadata which almost always changes as we modify the parquet file. The dedup stats show `<deduped size> / <total size> = <dedup ratio>` where smaller ratios mean higher deduplication performance. 

Also visualize the difference after removing a column:


```python
visualize(table, table_with_removed_columns, title="With Removed Columns", prefix="orca")
```


#### With Removed Columns

| Compression | Vanilla Parquet |
|:---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-with-removed-columns-nocdc.parquet.png) |
| Dedup Stats | 156.6 MB / 269.4 MB = 58%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-with-removed-columns-nocdc.parquet.png) |
| Dedup Stats | 96.1 MB / 166.7 MB = 57%|



Since we are removing entire columns we can only see changes in the footer metadata, all the other columns remain unchanged and already existing in the storage layer, so they are not transferred again.

### 3. Changing Column Types in the Table

Another common use case is changing the column types in the table e.g. to reduce the storage size or to optimize the data for specific queries. Let's change the `question_length` column from `int64` data type to `int32` and see how much data is transferred:


```python
# first make the table much smaller by removing the large string columns
# to highlight the differences better
table_without_text = table_with_new_columns.drop(["question", "response"])

# cast the question_length column to int64
table_with_casted_column = table_without_text.set_column(
    table_without_text.schema.get_field_index("question_length"),
    "question_length",
    table_without_text["question_length"].cast("int32")
)
```


```python
>>> pq.write_table(table_with_casted_column, "hf://datasets/kszucs/pq/orca-casted-column.parquet")
New Data Upload: 100%|███████████████████████████████████████████████|  181kB /  181kB,  113kB/s  
Total Bytes:  1.80M
Total Transfer:  181k
```


Again, we can see that only the new column and the updated parquet metadata were uploaded. Now visualize the deduplication heatmap:


```python
visualize(table_without_text, table_with_casted_column, title="With Casted Column", prefix="orca")
```


#### With Casted Column

| Compression | Vanilla Parquet |
|:---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-with-casted-column-nocdc.parquet.png) |
| Dedup Stats | 2.8 MB / 5.3 MB = 52%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-with-casted-column-nocdc.parquet.png) |
| Dedup Stats | 1.9 MB / 3.6 MB = 53%|



The first red region indicates the new column that was added, while the second red region indicates the updated metadata in the footer. The rest of the data remains unchanged and is not transferred again.

### 4. Appending New Rows and Concatenating Tables

We are going to append new rows by concatenating another slice of the original dataset to the table. 


```python
table = orca[:100_000]
next_10k_rows = orca[100_000:110_000]
table_with_appended_rows = pa.concat_tables([table, next_10k_rows])

assert len(table_with_appended_rows) == 110_000
```

Now check that only the new rows are being uploaded since the original data is already known to the Xet storage layer:


```python
>>> pq.write_table(table_with_appended_rows, "hf://datasets/kszucs/pq/orca-appended-rows.parquet")
New Data Upload: 100%|███████████████████████████████████████████████| 10.3MB / 10.3MB, 1.36MB/s  
Total Bytes:  106M
Total Transfer:  10.3M
```


```python
visualize(table, table_with_appended_rows, title="With Appended Rows", prefix="orca")
```


#### With Appended Rows

| Compression | Vanilla Parquet |
|:---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-with-appended-rows-nocdc.parquet.png) |
| Dedup Stats | 173.1 MB / 328.8 MB = 52%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-with-appended-rows-nocdc.parquet.png) |
| Dedup Stats | 106.5 MB / 201.8 MB = 52%|



Since each column gets new data, we can see multiple red regisions. This is due to the actual parquet file specification where whole columns are laid out after each other (within each row group). 

### 5. Inserting / Deleting Rows in the Table

Here comes the difficult part as insertions and deletions are shifting the existing rows which lead to different columns chunks or data pages in the parquet nomenclature. Since each data page is compressed separately, even a single row insertion or deletion can lead to a completely different byte-level representation starting from the edited row(s) to the end of the parquet file. 

This parquet specific problem cannot be solved by the Xet storage layer alone, the parquet file itself needs to be written in a way that minimizes the data page differences even if there are inserted or deleted rows. 

Let's try to use the existing mechanism and see how it performs.


```python
table = orca[:100_000]

# remove 4k rows from two places 
table_with_deleted_rows = pa.concat_tables([
    orca[:15_000], 
    orca[18_000:60_000],
    orca[61_000:100_000]
])

# add 1k rows at the first third of the table
table_with_inserted_rows = pa.concat_tables([
    orca[:10_000],
    orca[100_000:101_000],
    orca[10_000:50_000],
    orca[101_000:103_000],
    orca[50_000:100_000],
])

assert len(table) == 100_000
assert len(table_with_deleted_rows) == 96_000
assert len(table_with_inserted_rows) == 103_000
```


```python
>>> pq.write_table(table_with_inserted_rows, "hf://datasets/kszucs/pq/orca-inserted-rows.parquet")
New Data Upload: 100%|███████████████████████████████████████████████| 89.8MB / 89.8MB, 42.7kB/s  
Total Bytes:  99.1M
Total Transfer:  89.8M
```



```python
>>> pq.write_table(table_with_deleted_rows, "hf://datasets/kszucs/pq/orca-deleted-rows.parquet")
New Data Upload: 100%|███████████████████████████████████████████████| 78.2MB / 78.2MB, 46.5kB/s  
Total Bytes:  92.2M
Total Transfer:  78.2M
```


Also visualize both cases to see the differences:



```python
visualize(table, table_with_deleted_rows, title="Deleted Rows", prefix="orca")
visualize(table, table_with_inserted_rows, title="Inserted Rows", prefix="orca")
```


#### Deleted Rows

| Compression | Vanilla Parquet |
|:---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-deleted-rows-nocdc.parquet.png) |
| Dedup Stats | 185.3 MB / 306.8 MB = 60%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-deleted-rows-nocdc.parquet.png) |
| Dedup Stats | 174.4 MB / 188.3 MB = 92%|




#### Inserted Rows

| Compression | Vanilla Parquet |
|:---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-inserted-rows-nocdc.parquet.png) |
| Dedup Stats | 190.1 MB / 318.0 MB = 59%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-inserted-rows-nocdc.parquet.png) |
| Dedup Stats | 186.2 MB / 195.2 MB = 95%|



We can see that the deduplication performance has dropped significantly (higher ratio), and the deduplication heatmaps show that the compressed parquet files are quite different from each other. This is due to the fact that the inserted and deleted rows have shifted the existing rows, leading to different data pages starting from the edited row(s) to the end of the parquet file. 

We can solve this problem by writing parquet files with a new [pyarrow feature called content-defined chunking (CDC)](https://github.com/apache/arrow/pull/45360). This feature ensures that the columns are consistently getting chunked into data pages based on their content, similarly how the Xet storage layer deduplicates data but applied to the logical values of the columns before any serialization or compression happens. 

The feature can be enabled by passing `use_content_defined_chunking=True` to the `write_parquet` function:

```python
import pyarrow.parquet as pq

pq.write_table(table, "hf://user/repo/filename.parquet", use_content_defined_chunking=True)
```

Pandas also supports the new feature:

```python
df.to_parquet("hf://user/repo/filename.parquet", use_content_defined_chunking=True)
```

Let's visualize the deduplication difference before and after using the Parquet CDC feature:


```python
visualize(table, table_with_deleted_rows, title="With Deleted Rows", prefix="orca", with_cdc=True)
visualize(table, table_with_inserted_rows, title="With Inserted Rows", prefix="orca", with_cdc=True)
```


#### Deleted Rows

| Compression | Vanilla Parquet | CDC Parquet |
|:---:|---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-deleted-rows-nocdc.parquet.png) | ![Parquet none cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-with-deleted-rows-cdc.parquet.png) |
| Dedup Stats | 185.3 MB / 306.8 MB = 60%| 162.9 MB / 307.2 MB = 53%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-deleted-rows-nocdc.parquet.png) | ![Parquet snappy cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-with-deleted-rows-cdc.parquet.png) |
| Dedup Stats | 174.4 MB / 188.3 MB = 92%| 104.3 MB / 188.8 MB = 55%|




#### Inserted Rows

| Compression | Vanilla Parquet | CDC Parquet |
|:---:|---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-inserted-rows-nocdc.parquet.png) | ![Parquet none cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-with-inserted-rows-cdc.parquet.png) |
| Dedup Stats | 190.1 MB / 318.0 MB = 59%| 164.1 MB / 318.4 MB = 51%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-inserted-rows-nocdc.parquet.png) | ![Parquet snappy cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-with-inserted-rows-cdc.parquet.png) |
| Dedup Stats | 186.2 MB / 195.2 MB = 95%| 102.8 MB / 195.7 MB = 52%|



Looks much better! Since the proof of the pudding is in the eating, let's actually upload the tables using the content-defined chunking parquet feature and see how much data is transferred. 

Note that we need to upload the original table first with content-defined chunking enabled:


```python
>>> pq.write_table(table, "hf://datasets/kszucs/pq/orca-cdc.parquet", use_content_defined_chunking=True)
New Data Upload: 100%|███████████████████████████████████████████████| 94.5MB / 94.5MB, 46.5kB/s  
Total Bytes:  96.4M
Total Transfer:  94.5M
```


```python
>>> pq.write_table(
...     table_with_inserted_rows, 
...     "hf://datasets/kszucs/pq/orca-inserted-rows-cdc.parquet", 
...     use_content_defined_chunking=True
... )
New Data Upload: 100%|███████████████████████████████████████████████| 6.00MB / 6.00MB, 1.00MB/s  
Total Bytes:  99.3M
Total Transfer:  6.00M
```



```python
>>> pq.write_table(
...     table_with_deleted_rows, 
...     "hf://datasets/kszucs/pq/orca-deleted-rows-cdc.parquet", 
...     use_content_defined_chunking=True
... )
New Data Upload: 100%|███████████████████████████████████████████████| 7.57MB / 7.57MB, 1.35MB/s  
Total Bytes:  92.4M
Total Transfer:  7.57M
```


The uploaded data is significantly smaller than before, showing much better deduplication performance as highlighted in the heatmaps above.

Important to note that the same performance benefits apply to downloads using the `huggingface_hub.hf_hub_download()` and `datasets.load_dataset()` functions.


### 6. Using Different Row-group Sizes

There are cases depending on the reader/writer constraints where larger or smaller row-group sizes might be beneficial. The parquet writer implementations use fixed-sized row-groups by default, in case of pyarrow the default is 1Mi rows. Dataset writers may change to reduce the row-group size in order to improve random access performance or to reduce the memory footprint of the reader application.

Changing the row-group size will shift rows between row-groups, shifting values between data pages, so we have a similar problem as with inserting or deleting rows. Let's compare the deduplication performance between different row-group sizes using the parquet CDC feature:


```python
from de import visualize

# pick a larger subset of the dataset to have enough rows for the row group size tests
table = orca[2_000_000:3_000_000]

visualize(table, (table, {"row_group_size": 128 * 1024}), title="Small Row Groups", with_cdc=True, prefix="orca")
visualize(table, (table, {"row_group_size": 256 * 1024}), title="Medium Row Groups", with_cdc=True, prefix="orca")
```


#### Small Row Groups

| Compression | Vanilla Parquet | CDC Parquet |
|:---:|---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-small-row-groups-nocdc.parquet.png) | ![Parquet none cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-small-row-groups-cdc.parquet.png) |
| Dedup Stats | 1.6 GB / 3.1 GB = 52%| 1.6 GB / 3.1 GB = 50%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-small-row-groups-nocdc.parquet.png) | ![Parquet snappy cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-small-row-groups-cdc.parquet.png) |
| Dedup Stats | 1.1 GB / 1.9 GB = 59%| 995.0 MB / 1.9 GB = 51%|




#### Medium Row Groups

| Compression | Vanilla Parquet | CDC Parquet |
|:---:|---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-medium-row-groups-nocdc.parquet.png) | ![Parquet none cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-none-medium-row-groups-cdc.parquet.png) |
| Dedup Stats | 1.6 GB / 3.1 GB = 51%| 1.6 GB / 3.1 GB = 50%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-medium-row-groups-nocdc.parquet.png) | ![Parquet snappy cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/orca-snappy-medium-row-groups-cdc.parquet.png) |
| Dedup Stats | 1.1 GB / 1.9 GB = 57%| 976.5 MB / 1.9 GB = 50%|



### 7. Using Varying File-Level Splits

Datasets often split into multiple files to improve parallelism and random access. Parquet CDC combined with the Xet storage layer can efficiently deduplicate data across multiple files even if the data is split at different boundaries. 

Let's write out the dataset with three different file-level splitting then compare the deduplication performance:


```python
from pathlib import Path
from de import estimate


def write_dataset(table, base_dir, num_shards, **kwargs):
    """Simple utility to write a pyarrow table to multiple Parquet files."""
    # ensure that directory exists
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    # split and write the table into multiple files
    rows_per_file = len(table) / num_shards
    for i in range(num_shards):
        start = i * rows_per_file
        end = min((i + 1) * rows_per_file, len(table))
        shard = table.slice(start, end - start)
        path = base_dir / f"part-{i}.parquet"
        pq.write_table(shard, path, **kwargs)


write_dataset(orca, "orca5-cdc", num_shards=5, use_content_defined_chunking=True)
write_dataset(orca, "orca10-cdc", num_shards=10, use_content_defined_chunking=True)
write_dataset(orca, "orca20-cdc", num_shards=20, use_content_defined_chunking=True)

estimate("orca5-cdc/*.parquet", "orca10-cdc/*.parquet", "orca20-cdc/*.parquet")
```

    Total size: 9.3 GB
    Chunk size: 3.2 GB


Even though we uploaded the dataset with three different sharding configurations, the overall upload size would be barely larger than the original dataset size. 


### Using Parquet CDC feature with Pandas

So far we've used PyArrow, let’s explore using the same CDC feature with Pandas by downloading, filtering then uploading the dataset with the content-defined chunking feature enabled:


```python
import pandas as pd

src = "hf://datasets/teknium/OpenHermes-2.5/openhermes2_5.json"
df = pd.read_json(src)
```


```python
>>> dst = "hf://datasets/kszucs/pq/hermes-2.5-cdc.parquet"
>>> df.to_parquet(dst, use_content_defined_chunking=True)
New Data Upload: 100%|███████████████████████████████████████████████|  799MB /  799MB,  197kB/s  
Total Bytes:  799M
Total Transfer:  799M
```


```python
>>> short_df = df[[len(c) < 10 for c in df.conversations]]
>>> short_dst = "hf://datasets/kszucs/pq/hermes-2.5-cdc-short.parquet"
>>> short_df.to_parquet(short_dst, use_content_defined_chunking=True)
New Data Upload: 100%|███████████████████████████████████████████████| 21.9MB / 21.9MB, 45.4kB/s  
Total Bytes:  801M
Total Transfer:  21.9M
```



```python
import pyarrow as pa
from de import visualize

visualize(
    pa.Table.from_pandas(df), 
    pa.Table.from_pandas(short_df),
    title="Hermes 2.5 Short Conversations",
    with_cdc=True,
    prefix="hermes"
)
```


#### Hermes 2.5 Short Conversations

| Compression | Vanilla Parquet | CDC Parquet |
|:---:|---:|---:|
| None | ![Parquet none nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/hermes-none-hermes-2.5-short-conversations-nocdc.parquet.png) | ![Parquet none cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/hermes-none-hermes-2.5-short-conversations-cdc.parquet.png) |
| Dedup Stats | 1.9 GB / 3.2 GB = 58%| 1.6 GB / 3.2 GB = 51%|
| Snappy | ![Parquet snappy nocdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/hermes-snappy-hermes-2.5-short-conversations-nocdc.parquet.png) | ![Parquet snappy cdc](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parquet-cdc/hermes-snappy-hermes-2.5-short-conversations-cdc.parquet.png) |
| Dedup Stats | 1.5 GB / 1.6 GB = 94%| 821.1 MB / 1.6 GB = 51%|



Since Parquet CDC is applied at the parquet data page level (column chunk level), the deduplication performance depends on the filter's selectivity, or rather the distribution of the changes across the dataset. If most of the data pages are affected, then the deduplication ratio will drop significantly.

## References

More details about the feature can be found at:
- [Hugging Face's Xet announcement](https://huggingface.co/blog/xet-on-the-hub)
- [parquet-dedupe-estimator's readme](https://github.com/huggingface/dataset-dedupe-estimator)
- [PyArrow's documentation page](https://arrow.apache.org/docs/python/parquet.html#content-defined-chunking)
- [Pull request implementing Parquet CDC](https://github.com/apache/arrow/pull/45360)

## Conclusion

We explored the performance benefits of the new Parquet content-defined chunking feature used alongside Hugging Face's Xet storage layer. We demonstrated how it can efficiently deduplicate data in various scenarios making parquet operations faster and more storage-efficient. Comparing to traditional cloud storage solutions, the Xet storage layer with Parquet CDC can significantly reduce data transfer times and costs.

Migrate your Hugging Face repositories from Git LFS to Xet to benefit from this here: [https://huggingface.co/join/xet](https://huggingface.co/join/xet)