---
title: "Improving Parquet Dedupe"
thumbnail: /blog/assets/improve_parquet_dedupe/thumbnail.png
authors:
  - user: ylow
---

# Improving Parquet Dedupe

HuggingFace hosts nearly 11PB of datasets, with Parquet files alone accounting
for over 2.2PB of that storage. Since many of these files are bulk exports from
various data analysis pipelines or databases, they often appear as full
snapshots rather than incremental updates.

This is where data deduplication becomes critical. If users want to provide
updated datasets on a regular basis, we would like to be able to store all
versions of the dataset in as compact a way as possible. Dataset updates should
not require everything to be uploaded again. In an ideal case, we should be
able to store every version of a growing dataset, in little more than the size
of the largest version of the dataset.

While we use Content Defined Chunking which allows us to dedupe well over
insertions and deletions, the Parquet layout brings some challenges. Here we
run some dedupe experiments seeing how some simple data modifications behave on
Parquet files. We experiment  on a 2GB parquet file with 1092000 rows from the
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/data/CC-MAIN-2013-20)
dataset, generating visualizations using our [dedupe
estimator](https://github.com/huggingface/dedupe_estimator).

## Background

Parquet tables work by splitting the table into Row Groups : each with a fixed
number of rows (for instance 1000 rows). Each column within the row group is
then compressed and stored:

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improving_parquet_dedupe/layout.png" alt="Parquet Layout" width=90%>
</p>


Intuitively, this means that operations which do not mess with the row
grouping like modifications or appends should dedupe pretty well. So lets
test this out!

## Append

Here we  appending 10,000 new rows to the file and compare with the
original version. Green represents all deduped blocks, red represents all
new blocks, and shades in between are proportionate.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improving_parquet_dedupe/1_append.png" alt="Visualization of dedupe from data appends" width=90%>
</p>

Here we see that indeed we are able to dedupe pretty much the entire file,
but only with changes seen at the end of the file. The new file is 99.1%
deduped: requiring only 20MB of additional storage. This matches our
intuition pretty well here.

## Modification

Given the layout we would expect that row modifications to be pretty
isolated, but this is apparently not the case. Here we make a small
modification to row 10000, and we see while most of the file does dedupe,
there are many small regularly spaced sections of new data!

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improving_parquet_dedupe/2_mod.png" alt="Visualization of dedupe from data modifications" width=90%>
</p>

A quick scan of the [Parquet file
format](https://parquet.apache.org/docs/file-format/metadata/) suggests
that absolute file offsets are part of the Parquet column headers (See the
        structures ColumnChunk and ColumnMetaData) ! This means that any
modification, is likely to rewrite all the Column headers. So while the
data does dedupe well (it is mostly green), we get new bytes in every
column header. 

Here, the new file is only 89% deduped, requiring 230MB of additional
storage.

(Idea for the Parquet format maintainers. Perhaps switch to relative offsets
instead of absolute offsets?)

## Deletion

Here we delete a row from the middle of the file. (Insertion should have
    similar behavior) As this reorganizes the entire row group layout (each
        row group is 1000 rows), we see that we dedupe the first half of
    the file, but the remaining file is has completely new blocks. 

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improving_parquet_dedupe/3_delete.png" alt="Visualization of dedupe from data deletion" width=90%>
</p>

This is mostly because the Parquet format compresses each column
aggressively. If we turn off compression we are able to dedupe more
aggressively.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improving_parquet_dedupe/4_delete_no_compress.png" alt="Visualization of dedupe from data deletion without column compression" width=90%>
</p>


However the file sizes are nearly 2x larger if we store the data
uncompressed. 

Is it possible to have the benefit of dedupe and compression at the same
time?

## Content Defined Row Groups

One solution is to consider content defined chunking. i.e. We split row
groups not based on absolute count (1000 rows), but on a hash of a provided
“Key” column. In other words, I split off a row group whenever the hash of
the key column % [target row count] = 0, with some allowances for a minimum
and a maximum row group size. I hacked up a quick inefficient experimental
demonstration
[here](https://gist.github.com/ylow/db38522fb0ca69bdf1065237222b4d1c).

With this, we are able to efficiently dedupe across compressed Parquet
files even as I delete a row. Here we clearly see a big red block
representing the rewritten row group, followed by a small change for every
column header.


<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/improving_parquet_dedupe/5_content_defined.png" alt="Visualization of dedupe from data deletion with content defined row groups" width=90%>
</p>

# Optimizing Parquet for Dedupe-ability

Based on these experiments we could consider improving Parquet for
dedupe-ability in a couple of ways:

1. Use relative offsets instead of absolute offsets for file structure
data. (This will also make the Parquet structures position independent and
       easy to “memcpy” around). This is an involving file format change
and is probably difficult to do.
2. Support content defined chunking on row groups. As the format actually
supports this today as it does not require row groups to be uniformly sized,
this can be done with minimal blast radius. Only the Parquet format writers
will have to be updated.

While we will continue exploring ways to improve Parquet storage performance
(Ex: perhaps we could optionally rewrite Parquet files before uploading?
Strip absolute file offsets on upload and restore on download?), we will
love to work with the Apache Arrow project to see there is  interest in
implementing some of these ideas in the Parquet / Arrow code base.

In the meantime, we are also exploring the behavior of our data dedupe process
on other common filetypes. Please do try our [dedupe
estimator](https://github.com/huggingface/dedupe_estimator) and let us know
your findings!

