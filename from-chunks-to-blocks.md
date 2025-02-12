---
title: "From Chunks to Blocks: Accelerating Uploads and Downloads on the Hub"
thumbnail: /blog/assets/from-chunks-to-blocks/thumbnail.png
authors:
  - user: jsulz
    org: xet-team
  - user: yuchenglow
    org: xet-team
  - user: znation
    org: xet-team
  - user: saba9
    org: xet-team
---

# From Chunks to Blocks: Accelerating Uploads and Downloads on the Hub

Content-defined chunking (CDC) plays a central role in [enabling deduplication within a Xet-backed repository](https://huggingface.co/blog/from-files-to-chunks). The idea is straightforward: break each file’s data into chunks, store only unique ones, reap the benefits.

In practice, it's more complex. If we focused solely on maximizing deduplication, the design would call for the smallest possible chunk size. By doing that, we’d create significant overheads for the infrastructure and the builders on the Hub.

On Hugging Face's [Xet team](https://huggingface.co/xet-team), we're bringing CDC from theory to production to deliver faster uploads and downloads to AI builders (by a factor of 2-3x in some cases). Our guiding principle is simple: enable rapid experimentation and collaboration for teams building and iterating on models and datasets. This means focusing on more than just deduplication; we’re optimizing how data moves across the network, how it’s stored, and the entire development experience.

## The Realities of Scaling Deduplication

Imagine uploading a 200GB repository to the Hub. Today, [there are a number of ways to do this](https://huggingface.co/docs/huggingface_hub/en/guides/upload), but all use a file-centric approach. To bring faster file transfers to the Hub, we've open-sourced [xet-core](https://github.com/huggingface/xet-core) and `hf_xet`, an integration with [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) which uses a chunk-based approach written in Rust.

If you consider a 200GB repository with unique chunks, that's 3 million entries (at [~64KB per chunk](https://github.com/huggingface/xet-core/blob/main/merkledb/src/constants.rs#L5)) in the content-addressed store (CAS) backing all repositories. If a new version of a model is uploaded or a branch in the repository is created with different data, more unique chunks are added, driving up the entries in the CAS.

With nearly 45PB across 2 million model, dataset, and space repositories on the Hub, a purely chunk-based approach could incur **690 billion chunks**. Managing this volume of content using only chunks is simply not viable due to:

- **Network Overheads**: If each chunk is downloaded or uploaded individually, millions of requests are generated on each upload and download, overwhelming both client and server. Even [batching queries](https://developers.google.com/classroom/best-practices/batch) simply shifts the problem to the storage layer.
- **Infrastructure Overheads**: A naive CAS that tracks chunks individually would require billions of entries, leading to steep monthly bills on services like [DynamoDB](https://aws.amazon.com/pm/dynamodb/) or [S3](https://aws.amazon.com/s3/). At Hugging Face’s scale, this quickly adds up.

In short, network requests balloon, databases struggle to manage the metadata, and the cost of orchestrating each chunk skyrockets all while you wait for your files to transfer.

## Design Principles for Deduplication at Scale

These challenges lead to a key realization:

> **Deduplication is a performance optimization, not the final goal.**

The final goal is to improve the experience of builders iterating and collaborating on models and datasets. The system components from the client to the storage layer do not need to guarantee deduplication. Instead, they leverage deduplication as one tool among many to aid in this.

By loosening the deduplication constraint, we naturally arrive at a second design principle:

> **Avoid communication or storage strategies that scale 1:1 with the number of chunks**.

What does this mean? We scale with **aggregation.**

## Scaling Deduplication with Aggregation

Aggregation takes chunks and groups them, referencing them intelligently in ways that provide clever (and practical) benefits:

- **Blocks**: Instead of transferring and storing chunks, we bundle data together in blocks of [up to 64MB](https://github.com/huggingface/xet-core/blob/main/merkledb/src/constants.rs#L6) after deduplication. Blocks are still content-addressed, but this reduces CAS entries by a factor of 1,000.
- **Shards**: Shards provide the mapping between files and chunks (referencing blocks as they do so). This allows us to identify which parts of a file have changed, referencing shards generated from past uploads. When chunks are already known to exist in the CAS, they’re skipped, slashing unnecessary transfers and queries.

Together, blocks and shards unlock significant benefits. However, when someone uploads a new file, how do we know if a chunk has been uploaded before so we can eliminate an unnecessary request? Performing a network query for every chunk is not scalable and goes against the “no 1:1” principle we mentioned above.

The solution is **key chunks** which are a 0.1% subset of all chunks selected with a simple modulo condition based on the chunk hash. We provide a global index over these key chunks and the shards they are found in, so that when the chunk is queried, the related shard is returned to provide local deduplication. This allows us to leverage the principles of [spatial locality](https://en.wikipedia.org/wiki/Locality_of_reference). If a key chunk is referenced in a shard, it’s likely that other similar chunk references are available in the same shard. This further improves deduplication and reduces network and database requests.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/from-chunks-to-blocks/key-chunks.png" alt="Key chunks" width=90%>
</p>

## Aggregated Deduplication in Practice

The Hub currently stores over 3.5PB of `.gguf` files, most of which are quantized versions of other models on the Hub. Quantized models represent an interesting opportunity for deduplication due to the [nature of quantization](https://huggingface.co/docs/hub/en/gguf) where values are restricted to a smaller integer range and scaled. This restricts the range of values in the weight matrices, naturally leading to more repetition. Additionally, many repositories of quantized models store multiple different variants (e.g., [Q4_K, Q3_K, Q5_K](https://huggingface.co/docs/hub/en/gguf#quantization-types)) with a great deal of overlap.

A good example of this in practice is [bartowski/gemma-2-9b-it-GGUF](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF) which contains 29 quantizations of [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) totalling 191GB. To upload, we use `hf_xet` integrated with `huggingface_hub` to perform chunk-level deduplication locally then aggregate and store data at the block level.

Once uploaded, we can start to see some cool patterns! We’ve included a visualization that shows the deduplication ratio for each block. The darker the block, the more frequently parts of it are referenced across model versions. If you go to the [Space hosting this visualization](https://huggingface.co/spaces/xet-team/quantization-dedup), hovering over any heatmap cell highlights all references to the block in orange across all models while clicking on a cell will select all other files that share blocks:

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/from-chunks-to-blocks/dedupe_viz.gif" alt="Quantization deduplication visualization" width=90%>
</p>

A single block of deduplication might only represent a few MB of savings, but as you can see there are many overlapping blocks! With this many blocks that quickly adds up. Instead of uploading 191GB, the Xet-backed version of the `gemma-2-9b-it-GGUF` repository stores 1515 unique blocks for a total of approximately 97GB to our test CAS environment (a savings of ~94GB).

While the storage improvements are significant, the real benefit is what this means for contributors to the Hub. At 50MB/s, the deduplication optimizations amount to a four hour difference in upload time; a speedup of nearly 2x:

| Repo       | Stored Size | Upload Time @ 50MB/s |
| ---------- | ----------- | -------------------- |
| Original   | 191 GB      | 509 minutes          |
| Xet-backed | 97 GB       | 258 minutes          |

Similarly, local chunk caching significantly speeds up downloads. If a file is changed or a new quantization is added that has significant overlap with the local chunk cache, you won’t have to re-download any chunks that are unchanged. This contrasts to the file-based approach where the entirety of the new or updated file must be downloaded.

Taken together, this demonstrates how local chunk-level deduplication paired with block-level aggregation dramatically streamlines not just storage, but developing on the Hub. By providing this level of efficiency in file transfers, AI builders can move faster, iterate quickly, and worry less about hitting infrastructure bottlenecks. For anyone pushing large files to the Hub (whether you're pushing a new model quantization or an updated version of a training set) this helps you shift focus to building and sharing, rather than waiting and troubleshooting.

We’re fast at work, rolling out the first Xet-backed repositories in the coming weeks and months! As we do that, we will be releasing more updates to bring these speeds to every builder on the Hub to make file transfers feel invisible.

[Follow us](https://huggingface.co/xet-team) on the Hub to learn more about our progress!
