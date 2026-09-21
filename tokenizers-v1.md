---
title: "tokenizers v1: encode, decode and scaling, measured"
thumbnail: /blog/assets/tokenizers-v1/thumbnail.png
authors:
- user: lysandre
---

# tokenizers v1: encode, decode and scaling, measured


<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-header.static.hf.space" width="100%" height="352" frameborder="0" scrolling="no"></iframe>
</figure>

The tokenizer has not historically been the bottleneck within ML workflows. Compute-wise, tokenization is light compared to the heavy modeling happening in the rest of the pipeline. Yet, in some cases, it has rapidly become key to accelerating (or slowing down) your machine learning work.

As models become faster and workloads scale, that balance begins to shift. Training on massive datasets, serving many concurrent requests, or repeatedly processing long inputs can put enough pressure on the tokenizer that it starves the model of data.

This is why we have chosen to heavily focus on performance for the upcoming version 1 of tokenizers. Tokenization should be light and should scale with your workflow. Your GPUs should never sit idle waiting for the CPU to complete its tokenization.

---

In this article, we look at what makes v1 faster than v0.23, often by tens of times.

This work was entirely possible thanks to the rest of the ecosystem. Tokenization is a very active area of open source work, and libraries such as [gigatoken](https://github.com/marcelroed/gigatoken), [tiktoken](https://crates.io/crates/tiktoken-rs), [kitoken](https://crates.io/crates/kitoken), [tokie](https://crates.io/crates/tokie), [fastokens](https://crates.io/crates/fastokens), [wordchipper](https://crates.io/crates/wordchipper) and [ai-tokenizer](https://www.npmjs.com/package/ai-tokenizer), as well as many others, have each pushed on what a fast tokenizer can be. We read that work, and several of the ideas below reached us because another project showed they were worth trying. 

Before this refactor, tokenizers was nowhere near the performance it could have had, so contributing to it may not have seemed worth it. With this refactor, we hope to make clear that we intend tokenizers to be a library worth contributing to.

We also thank IBM, NVIDIA, and the ExecuTorch team for contributing patches and helping us test across a wide range of hardware to broaden platform support.

## Results

We showcase results for the release candidate of tokenizers v1 against other widely used alternatives. We go over single-threaded, multi-threaded, scaling across threads, per-model comparison, per-language comparison, latency, decoding throughput, memory heap, as well as crate size.

We run this from the [tokbench](https://github.com/huggingface/tokbench) repository, and add a command to rerun the benchmarks on your hardware if you would like to do so.

<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-results.static.hf.space" width="100%" height="2615" frameborder="0" scrolling="no"></iframe>
</figure>

## What V1 Is

v1 will produce the same token IDs as v0.23. The goal was to preserve the output, the API, the vocabulary and the merge ranks, and improve everything that **can** be improved. That includes breadth. The library stays general across tokenizer families rather than specialising on BPE, so v1 loads everything v0.23 loaded.

A tokenizer converts text into the list of integers a model reads. tokenizers runs that conversion in four stages. Normalization applies operations such as lowercasing or Unicode normalization to the raw text. Pre-tokenization splits the text into smaller pieces called pre-tokens. The model turns each pre-token into tokens and maps them to IDs in its vocabulary. Post-processing adds any special tokens the model expects.

The model stage is where most of the work described here happens. Eight of the ten model families measured in this article use byte pair encoding, or BPE. BPE starts from the bytes of a pre-token and repeatedly joins the highest ranked adjacent pair until no ranked pair remains. The ranking is learned when the tokenizer is trained and ships with it, so the same text always produces the same IDs. A merge never crosses a pre-token boundary. The other two families use WordPiece and Unigram, the two other model types the library supports.

The [tokenization pipeline](https://huggingface.co/docs/tokenizers/pipeline) page documents the four stages. [Tokenization algorithms](https://huggingface.co/docs/transformers/tokenizer_summary) documents BPE, WordPiece and Unigram.

<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-pipeline.static.hf.space" width="100%" height="341" frameborder="0" scrolling="no"></iframe>
</figure>

Each stage was worked on. These are the changes that mattered:

| change | what it does |
| --- | --- |
| workspace split | one crate became a workspace: `tk-encode` is the required runtime, and `tk-serialize`, `tk-convert` and `tk-train` are linked only when an application needs them |
| no-alloc model | the merge working set lives in a caller-owned scratch buffer; the loop never touches the allocator |
| bitcannon | the split pattern becomes Boolean operations over bitstreams, using SIMD instructions to find splits instead of a regex engine |
| merge-loop rewrite | the pieces being merged form an intrusive doubly-linked list inside one preallocated buffer, so a merge updates two indices instead of moving data |
| word cache | a thread-local memo from pre-token bytes to finished ids, so a repeated word is merged once |
| native parallelism | one shared tokenizer encodes from many threads at once; each thread draws its scratch buffer and word cache from its own sub-pool, so threads no longer queue on a single lock ([#2365](https://github.com/huggingface/tokenizers/pull/2365)) |

### The Split: Bitstreams Instead Of A Regex

BPE models use a regular expression to split the input text into smaller, easier to process chunks called pre-tokens. Merges happen inside a pre-token and never across the boundary between two of them, so this split decides what the rest of the pipeline sees.

That regular expression is a fixed parameter of the model. It ships with the tokenizer and never changes at runtime, so there is no need for a general-purpose regex engine to interpret it on every encode. An equivalent splitting function can be written by hand, once, for the pattern a given model actually uses.

A hand-written function can then use the SIMD instructions (single instruction, multiple data) of a modern CPU, which apply one operation to many bytes at once and suit UTF-8 text well. bitcannon views the input's bytes as parallel streams of bits, so boundaries fall out of boolean operations across whole registers instead of a scan that advances one character at a time. It decides 64 bytes per register operation. The same idea drives [Parabix](https://www.cs.sfu.ca/~ashriram/papers/2012_HPCA_Parabix.pdf) for text processing and [simdjson](https://arxiv.org/abs/1902.08318) for JSON.

This depends on recognising the pattern. A handful of grammars cover most byte-level BPE models, and a tokenizer whose pattern is not among them keeps the regex path and none of this speed-up. That is why the gains above vary as much as they do.

<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-split.static.hf.space" width="100%" height="304" frameborder="0" scrolling="no"></iframe>
</figure>

### The Word Cache

Real text contains many repeated words. Because BPE always produces the same token IDs for a given pre-token, v1 can save the result after processing it once. A thread-local cache maps each pre-token's bytes to its token IDs, allowing later occurrences to skip the merge process.

Naturally, as the input grows, the number of unique words can grow more slowly than the total number of words. Repeated words then account for an increasing share of the input. New words still appear, which accounts for the occasional misses in the animation below.

<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-cache.static.hf.space" width="100%" height="976" frameborder="0" scrolling="no"></iframe>
</figure>

Reproduce the shared-prefix result with:

```bash
tokbench measure prefix-sharing \
  --engine pipeline \
  --engine hf-tokenizers \
  --compare-to pipeline-no-cache \
  --corpus agentic_swe
```

Caching works best when the input contains repeated pre-tokens. Input with few repeated pre-tokens can pay for lookups without receiving many hits.

### The Merge Loop

The next major cost comes from the BPE merge loop. For each pre-token, the loop repeatedly finds the highest-priority adjacent pair and merges it. The previous implementation allocated new memory for every call and built a new priority queue for every pre-token.

v1 reuses a scratch buffer owned by the caller, removing those repeated allocations. It stores symbols in a flat array and links adjacent symbols by their positions in that array, which makes updates during merging cheaper. It also processes a batch of pre-tokens in a single model call.

Each candidate pair is also packed into a single 64-bit value, with the merge rank in the high bits. Comparing two candidates is then just comparing two integers, and "no merge here" is the largest possible value, so the loop finds its next merge without a branch.

## Method

Small differences in benchmark design can produce large differences in tokenizer performance. We used the following rules to keep the comparison consistent across engines.

| rule | why |
| --- | --- |
| one timing loop | every engine runs the identical loop; no per-engine fast path |
| load excluded | vocabulary load is timed separately, never inside encode |
| id-hash verified | FNV-1a over the output ids must match the baseline exactly |
| common cells only | medians are over cells every engine ran and verified |
| complete sweep per process | each repeat starts in a new process and retains every cell |
| physical-core pinning | workers are pinned to eight distinct physical cores, never sibling SMT threads |
| independent Jobs | separate Jobs measure host-to-host variation |

Repeatedly encoding one document can be faster than encoding a stream of distinct documents on the same build. The first approach measures performance when the entire document is already represented in the cache. The second measures performance on new input while allowing previously seen pre-tokens to remain cached.

Both conditions are sometimes described as "warm," even though they measure different workloads. Our headline results use distinct documents, and the complete corpus is too large to fit in the cache. Tokenizer benchmarks should identify which workload they use because the choice can dominate the result.

## What This Adds Up To

Across the ten model families v1's encode path covers, it encodes text **3 to 30 times faster** than v0.23 with one thread on an Apple M4 Max. The low end is t5-base, the high end gpt2. It scales at **76%** of linear across eight workers. Throughout these changes, v1 produces exactly the same token IDs as the released library.

The overall improvement comes from several changes working together: a hand-written splitter in place of a regex engine, a cache that answers a repeated word without merging it again, a merge loop that never touches the allocator, and one model call per batch of pre-tokens instead of one per pre-token. Each reduces the work done at a different point in the pipeline.

The next priority is support for more model families. We will move additional models onto the new merge loop before `1.0.0`.

This post is generated from [tokbench](https://github.com/huggingface/tokbench) results and will be updated as support expands.

## Getting It

A release candidate for v1 is on crates.io. The API you call is the one you already call, so the only thing that changes is which build you install.

It is the ordinary install:

```bash
cargo add tokenizers --pre
```

Training is behind a default-on feature that pulls a C++ dependency with it. If you only need to encode, turn it off to exclude the training implementation:

```bash
cargo add tokenizers --pre --no-default-features --features http
```

Encoding is unchanged: same call, same ids.

```rust
use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("deepseek-ai/DeepSeek-V4-Flash", None)?;

let encoding = tokenizer.encode("The tokenizer is no longer the bottleneck.", false)?; println!("{:?}", encoding.get_ids()); // [671, 17840, 9160, 344, 1119, 5827, 270, 111127, 16] println!("{:?}", encoding.get_tokens()); // ["The", "Ġtoken", "izer", "Ġis", "Ġno", "Ġlonger", "Ġthe", "Ġbottleneck", "."]

Ok(()) } ```

For a batch, `encode_batch` is what scales across cores. It is the call the scaling view above measures.

```rust
let encodings = tokenizer.encode_batch(documents, false)?;
```

Every figure in this post was measured against this crate. The Python bindings wrap the same code and are built from `bindings/python`, but they add per-call overhead that none of these measurements include.

## Progress Towards V1

The benchmarks in this post cover the completed release-candidate work listed first. The remaining sections show what is still required for `1.0.0` and what we plan to explore afterward.

### Release Candidate: Implemented

This work is in the Rust pre-release on crates.io:

```bash
cargo add tokenizers --pre
```

- workspace split: divide the single crate into `tk-encode`, `tk-serialize`, `tk-convert` and `tk-train`, so an application links only what it uses
- bitcannon: replace regex splitting on the encoding path with bitstream operations covering GPT-2, cl100k, o200k, Tekken and DeepSeek. This replaced the finite-state machines that shipped first [#2201](https://github.com/huggingface/tokenizers/pull/2201) [#2317](https://github.com/huggingface/tokenizers/pull/2317)
- WordCache: reuse the token IDs of previously processed pre-tokens [#2262](https://github.com/huggingface/tokenizers/pull/2262), `af5a3e3`
- faster lookup and merging structures: add FlatCache, MPHF RankStore, incremental merging, and BucketVocabStore [#2190](https://github.com/huggingface/tokenizers/pull/2190) [#2188](https://github.com/huggingface/tokenizers/pull/2188)
- reusable model memory: move temporary model state into scratch buffers so tokenization does not allocate on each call [#2175](https://github.com/huggingface/tokenizers/pull/2175) [#2183](https://github.com/huggingface/tokenizers/pull/2183)
- pipeline post-processing: expose post-processing as the `STAGE_POST` pipeline stage [#2182](https://github.com/huggingface/tokenizers/pull/2182)
- batched model calls: process multiple pre-token spans in one call [#2304](https://github.com/huggingface/tokenizers/pull/2304)
- faster decoding: write decoded bytes directly into a reusable buffer, avoid intermediate strings and copies, accelerate token lookup, support buffered streaming, and decode batches in parallel
- `role_to_token` support [#2343](https://github.com/huggingface/tokenizers/pull/2343)
- Node.js bindings [#2281](https://github.com/huggingface/tokenizers/pull/2281)

### 1.0.0

- one encoding implementation: use `tk-encode` during training validation so training and inference cannot produce different tokenization results
- optional offsets and masks: compute this metadata only when requested, keeping it off the token-ID-only path
- rework normalizers
- bitnorm support, building on atomnorm [#2209](https://github.com/huggingface/tokenizers/pull/2209)
- spm precompiled
- simpler Python bindings: reduce locking, wrapper types, and handwritten dispatch code while preserving subclassing, serialization, custom decoders, mutation behavior, and support for free-threaded CPython
- inference-only C and C++ bindings for ExecuTorch and llama.cpp, with possible JVM, Swift, and Go bindings to follow

### After 1.0.0

- tok-devices: explore GPU encoding and batch decoding while keeping text and token IDs on the device. The decoder would upload the vocabulary once, calculate output positions in parallel, and gather the corresponding bytes on the GPU. This would be an optional component intended for large batches, subject to further prototyping and measurement.
