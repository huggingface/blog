---
title: "Improving Parquet Dedupe on Hugging Face Hub"
thumbnail: /blog/assets/improve_parquet_dedupe/thumbnail.png
authors:
  - user: yuchenglow
  - user: seanses
---

# Improving Parquet Dedupe on Hugging Face Hub

The Xet team at Hugging Face is working on improving the efficiency of the Hub's
storage architecture to make it easier and quicker for users to
store and update data and models. As Hugging Face hosts nearly 11PB of datasets
with Parquet files alone accounting for over 2.2PB of that storage,
optimizing Parquet storage is of pretty high priority. 

Most Parquet files are bulk exports from various data analysis pipelines
or databases, often appearing as full snapshots rather than incremental
updates. Data deduplication becomes critical for efficiency when users want to 
update their datasets on a regular basis. Only by deduplicating can we store 
all versions as compactly as possible, without requiring everything to be uploaded
again on every update. In an ideal case, we should be able to store every version 
of a growing dataset with only a little more space than the size of its largest version.

Our default storage algorithm uses byte-level [Content-Defined Chunking (CDC)](https://joshleeb.com/posts/content-defined-chunking.html), 
which generally dedupes well over insertions and deletions, but the Parquet layout brings some challenges. 
Here we run some experiments to see how some simple modifications behave on
Parquet files, using a 2GB Parquet file with 1,092,000 rows from the
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/data/CC-MAIN-2013-20)
dataset and generating visualizations using our [dedupe
estimator](https://github.com/huggingface/dedupe_estimator).

## Background

Parquet tables work by splitting the table into row groups, each with a fixed
number of rows (for instance 1000 rows). Each column within the row group is
then compressed and stored:

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improve_parquet_dedupe/layout.png" alt="Parquet Layout" width=80%>
</p>

Intuitively, this means that operations which do not mess with the row
grouping, like modifications or appends, should dedupe pretty well. So let's
test this out!

## Append

Here we append 10,000 new rows to the file and compare the results with the
original version. Green represents all deduped blocks, red represents all
new blocks, and shades in between show different levels of deduplication.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improve_parquet_dedupe/1_append.png" alt="Visualization of dedupe from data appends" width=30%>
</p>

We can see that indeed we are able to dedupe nearly the entire file,
but only with changes seen at the end of the file. The new file is 99.1%
deduped, requiring only 20MB of additional storage. This matches our
intuition pretty well.

## Modification

Given the layout, we would expect that row modifications to be pretty
isolated, but this is apparently not the case. Here we make a small
modification to row 10000, and we see that while most of the file does dedupe,
there are many small regularly spaced sections of new data!

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improve_parquet_dedupe/2_mod.png" alt="Visualization of dedupe from data modifications" width=30%>
</p>

A quick scan of the [Parquet file
format](https://parquet.apache.org/docs/file-format/metadata/) suggests
that absolute file offsets are part of the Parquet column headers (see the
structures ColumnChunk and ColumnMetaData)! This means that any
modification is likely to rewrite all the Column headers. So while the
data does dedupe well (it is mostly green), we get new bytes in every
column header. 

In this case, the new file is only 89% deduped, requiring 230MB of additional
storage.

## Deletion

Here we delete a row from the middle of the file (note: insertion should have
similar behavior). As this reorganizes the entire row group layout (each 
row group is 1000 rows), we see that while we dedupe the first half of
the file, the remaining file has completely new blocks. 

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improve_parquet_dedupe/3_delete.png" alt="Visualization of dedupe from data deletion" width=30%>
</p>

This is mostly because the Parquet format compresses each column
aggressively. If we turn off compression we are able to dedupe more
aggressively:

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improve_parquet_dedupe/4_delete_no_compress.png" alt="Visualization of dedupe from data deletion without column compression" width=30%>
</p>

However the file sizes are nearly 2x larger if we store the data
uncompressed. 

Is it possible to have the benefit of dedupe and compression at the same
time?

## Content Defined Row Groups

One potential solution is to use not only byte-level CDC, but also apply it at the row level: 
we split row groups not based on absolute count (1000 rows), but on a hash of a provided
“Key” column. In other words, I split off a row group whenever the hash of
the key column % [target row count] = 0, with some allowances for a minimum
and a maximum row group size. 

I hacked up a quick inefficient experimental demonstration
[here](https://gist.github.com/ylow/db38522fb0ca69bdf1065237222b4d1c).

With this, we are able to efficiently dedupe across compressed Parquet
files even as I delete a row. Here we clearly see a big red block
representing the rewritten row group, followed by a small change for every
column header.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improve_parquet_dedupe/5_content_defined.png" alt="Visualization of dedupe from data deletion with content defined row groups" width=30%>
</p>

# Optimizing Parquet for Dedupe-ability

Based on these experiments, we could consider improving Parquet file
dedupe-ability in a couple of ways:

1. Use relative offsets instead of absolute offsets for file structure
data. This would make the Parquet structures position independent and 
easy to “memcpy” around, although it is an involving file format change that
is probably difficult to do.
2. Support content defined chunking on row groups. The format actually
supports this today as it does not require row groups to be uniformly sized,
so this can be done with minimal blast radius. Only Parquet format writers
would have to be updated.

While we will continue exploring ways to improve Parquet storage performance
(Ex: perhaps we could optionally rewrite Parquet files before uploading?
Strip absolute file offsets on upload and restore on download?), we would 
love to work with the Apache Arrow project to see if there is interest in
implementing some of these ideas in the Parquet / Arrow code base.

In the meantime, we are also exploring the behavior of our data dedupe process
on other common filetypes. Please do try our [dedupe
estimator](https://github.com/huggingface/dedupe_estimator) and tell us about
your findings!

