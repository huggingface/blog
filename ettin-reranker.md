---
title: "Introducing the Ettin Reranker Family"
thumbnail: /blog/assets/train-sentence-transformers/st-hf-thumbnail.png
authors:
- user: tomaarsen
---

# Introducing the Ettin Reranker Family

## TL;DR

Today I'm releasing six new [Sentence Transformers](https://sbert.net/) CrossEncoder rerankers, state-of-the-art at their respective sizes, built on top of the [Ettin](https://huggingface.co/collections/jhu-clsp/encoders-vs-decoders-the-ettin-suite) ModernBERT encoders, together with the data and full training recipe that produced them:

- [`cross-encoder/ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1)
- [`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1)
- [`cross-encoder/ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1)
- [`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)
- [`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1)
- [`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1)

The models were trained with a **distillation recipe**: pointwise MSE on [`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) scores over [`cross-encoder/ettin-reranker-v1-data`](https://huggingface.co/datasets/cross-encoder/ettin-reranker-v1-data), which is a subset of [`lightonai/embeddings-pre-training`](https://huggingface.co/datasets/lightonai/embeddings-pre-training) mixed with a reranked subset of [`lightonai/embeddings-fine-tuning`](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning).

![Our six rerankers paired with embeddinggemma-300m on MTEB(eng, v2) Retrieval](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ettin-reranker/mteb_ndcg10_embeddinggemma-300m.png)

*Our six rerankers paired with [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) on MTEB(eng, v2) Retrieval. See [Results](#results) for five more embedder pairings.*

If you're new to rerankers and want the "why" first, jump to [What is a reranker, and why pair one with an embedder?](#what-is-a-reranker-and-why-pair-one-with-an-embedder). If you just want to plug a model in, jump to [Usage](#usage). If you want to train your own, jump to [Training](#training).

> [!TIP]
> I bootstrapped the training recipe below with the new [`train-sentence-transformers` Agent Skill](https://github.com/huggingface/sentence-transformers/tree/main/skills) shipped in [Sentence Transformers v5.5.0](https://github.com/huggingface/sentence-transformers/releases/tag/v5.5.0). Install it with `hf skills add train-sentence-transformers [--global] [--claude]` and ask your AI coding agent (Claude Code, Codex, Cursor, Gemini CLI, ...) to fine-tune a `SentenceTransformer`, `CrossEncoder`, or `SparseEncoder` model on your data.

## Table of contents

- [What is a reranker, and why pair one with an embedder?](#what-is-a-reranker-and-why-pair-one-with-an-embedder)
- [Usage](#usage)
  - [End-to-end retrieve-then-rerank pipeline](#end-to-end-retrieve-then-rerank-pipeline)
- [Architecture Details](#architecture-details)
- [Results](#results)
  - [MTEB(eng, v2) Retrieval](#mteb-eng-v2-retrieval)
  - [Speed](#speed)
- [Training](#training)
  - [Distillation recipe](#distillation-recipe)
  - [Dataset](#dataset)
  - [Training Arguments](#training-arguments)
  - [Evaluation](#evaluation)
  - [Overall Training Script](#overall-training-script)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## What is a reranker, and why pair one with an embedder?

A reranker (a.k.a. pointwise cross-encoder) is a neural model that takes a `(query, document)` pair and outputs a single relevance score. Unlike an embedding model, which encodes the query and document separately and computes their similarity from the two embedding vectors, a reranker lets the two texts attend to each other through every transformer layer. That joint encoding is more accurate but also more expensive: the model has to be run once per `(query, document)` pair rather than once per text.

Because cross-encoders are too expensive to run over a full corpus, the common production pattern is **retrieve-then-rerank**: a fast embedding model retrieves the top-K candidates (cheap), then a cross-encoder re-orders just those K with high accuracy. The total cost stays bounded while the final ranking is much closer to what an exhaustive cross-encoder pass would produce.

![Embedding vs Reranker Models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-reranker/embedding_vs_reranker_model.png)

> [!NOTE]
> Throughout this blogpost I'll use "reranker" and "cross-encoder" interchangeably.

## Usage

The released models are normal Sentence Transformers `CrossEncoder` models, so you can use them with just 3 lines of code:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ettin-reranker-32m-v1")
scores = model.predict([
    ("Where was Apple founded?", "Apple Inc. was founded in Cupertino, California in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne."),
    ("Where was Apple founded?", "The Fuji apple is an apple cultivar developed in the late 1930s and brought to market in 1962."),
])
print(scores)
# [11.393298  2.968891]   <- larger means more relevant
```

For a query and a list of candidates, you can also use `rank` to get back sorted indices and scores:

```python
ranked = model.rank(
    query="Which planet is known as the Red Planet?",
    documents=[
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ],
    top_k=4,
    return_documents=True,
)
for r in ranked:
    print(f"({r['score']:.2f}): {r['text']}")
# (10.82): Mars, known for its reddish appearance, is often referred to as the Red Planet.
# (9.86): Saturn, famous for its rings, is sometimes mistaken for the Red Planet.
# (8.55): Jupiter, the largest planet in our solar system, has a prominent red spot.
# (6.21): Venus is often called Earth's twin because of its similar size and proximity.
```

You can swap [`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1) for any other size to trade quality for speed. All six accept up to 8K tokens of context (useful for long-document reranking) thanks to ModernBERT's long-context pre-training.

It is recommended to install [`kernels`](https://github.com/huggingface/kernels) and set `model_kwargs={"dtype": "bfloat16", "attn_implementation": "flash_attention_2"}` for the highest throughput. See the [Speed](#speed) section below for more details, but in general you can expect a 1.7x-8.3x speedup over default loading depending on model size and sequence length.

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ettin-reranker-32m-v1",
    model_kwargs={"dtype": "bfloat16", "attn_implementation": "flash_attention_2"},
)
```

### End-to-end retrieve-then-rerank pipeline

A complete example with a fast embedder for retrieval and the reranker for the final ordering:

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# Fast retrieval with a static embedder (sub-millisecond on CPU per query)
embedder = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
reranker = CrossEncoder("cross-encoder/ettin-reranker-68m-v1")

corpus = [
    "Apple Inc. was founded in Cupertino, California in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
    "The Fuji apple is an apple cultivar developed in the late 1930s.",
    "Steve Jobs introduced the iPhone in 2007 at Macworld.",
    "Macintosh computers were sold by Apple from 1984 onward.",
    # ... thousands or millions more in production
]
query = "Where was Apple founded?"

# Step 1: encode + retrieve top-100
query_emb = embedder.encode_query(query, convert_to_tensor=True)
corpus_emb = embedder.encode_document(corpus, convert_to_tensor=True)
scores = embedder.similarity(query_emb, corpus_emb)[0]
top_k_idx = scores.topk(min(100, len(corpus))).indices.tolist()

# Step 2: rerank
top_k_docs = [corpus[i] for i in top_k_idx]
ranked = reranker.rank(query, top_k_docs, top_k=5, return_documents=True)
for r in ranked:
    print(f"({r['score']:.2f}): {r['text']}")
# (11.63): Apple Inc. was founded in Cupertino, California in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.
# (4.71): Steve Jobs introduced the iPhone in 2007 at Macworld.
# (1.96): The Fuji apple is an apple cultivar developed in the late 1930s.
# (1.49): Macintosh computers were sold by Apple from 1984 onward.
```

This is the same shape used by most modern search systems. The retriever decides what enters the funnel, the reranker decides what wins.

## Architecture Details

All six rerankers share the same architecture and differ only in their backbone size. The backbone is one of the six [Ettin encoders](https://huggingface.co/blog/ettin) from Johns Hopkins University's Ettin suite. These are ModernBERT-style models with unpadded attention, RoPE positional encodings, GeGLU, and 2T tokens of open-license pre-training, supporting up to 8192 tokens of context.

On top of each encoder, the reranker uses a 4-module classification head that mirrors `ModernBertForSequenceClassification` but is built from Sentence Transformers' modular components. The underlying `Transformer` is a plain `AutoModel` rather than `AutoModelForSequenceClassification`, which lets us use sequence unpadding for variable-length inputs for Flash Attention 2. At medium-document sequence lengths this is a 1.7x-8.3x speedup over fp32+SDPA depending on model size (see [Speed](#speed) for the full benchmark):

```
1. Transformer(FA2)
2. Pooling(cls)
3. Dense(H, H, bias=False, GELU)
4. LayerNorm(H)
5. Dense(H, 1, scores)
```

In my ablations, CLS pooling outperformed mean pooling. That was a little surprising. ModernBERT uses global attention only every third layer and the other two-thirds use local-window attention that cannot reach CLS from distant positions. Empirically, those few global layers carry enough signal to make CLS the better pooling choice.

| Model | Backbone | Hidden size | Layers | Params (head incl.) |
| --- | --- | ---: | ---: | ---: |
| [`cross-encoder/ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1) | [`jhu-clsp/ettin-encoder-17m`](https://huggingface.co/jhu-clsp/ettin-encoder-17m) | 256 | 7 | 17.6M |
| [`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1) | [`jhu-clsp/ettin-encoder-32m`](https://huggingface.co/jhu-clsp/ettin-encoder-32m) | 384 | 10 | 32.8M |
| [`cross-encoder/ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1) | [`jhu-clsp/ettin-encoder-68m`](https://huggingface.co/jhu-clsp/ettin-encoder-68m) | 512 | 19 | 68.6M |
| [`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1) | [`jhu-clsp/ettin-encoder-150m`](https://huggingface.co/jhu-clsp/ettin-encoder-150m) | 768 | 22 | 150.9M |
| [`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1) | [`jhu-clsp/ettin-encoder-400m`](https://huggingface.co/jhu-clsp/ettin-encoder-400m) | 1024 | 28 | 401.6M |
| [`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1) | [`jhu-clsp/ettin-encoder-1b`](https://huggingface.co/jhu-clsp/ettin-encoder-1b) | 1792 | 28 | 1.00B |

All six models are released under the **Apache 2.0** license, matching the Ettin encoders.

## Results

### MTEB(eng, v2) Retrieval

I ran each released model through the full [`MTEB(eng, v2)` Retrieval benchmark](https://github.com/embeddings-benchmark/mteb) (10 tasks, top-100 reranked) using MTEB's [two-stage reranking flow](https://embeddings-benchmark.github.io/mteb/get_started/advanced_usage/two_stage_reranking/), pairing each reranker with six embedding models that span the speed/quality spectrum:

| Embedding Model | Active params | Retriever-only NDCG@10 |
| --- | ---: | ---: |
| [`sentence-transformers/static-retrieval-mrl-en-v1`](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) | 0M | 0.3495 |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 23M | 0.4292 |
| [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33M | 0.5149 |
| [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 137M | 0.5226 |
| [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) | 308M | 0.5463 |
| [`jinaai/jina-embeddings-v5-text-small-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval) | 596M | 0.5980 |

The **dashed retriever-only line** in each chart below is the headline number to beat. Anything below it means the reranker actively hurts the pipeline on average:

| | |
|-|-|
| ![MTEB(eng, v2) Retrieval with static-retrieval-mrl-en-v1 + reranker](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ettin-reranker/mteb_ndcg10_static-retrieval-mrl-en-v1.png) | ![MTEB(eng, v2) Retrieval with all-MiniLM-L6-v2 + reranker](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ettin-reranker/mteb_ndcg10_all-MiniLM-L6-v2.png) |
| ![MTEB(eng, v2) Retrieval with bge-small-en-v1.5 + reranker](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ettin-reranker/mteb_ndcg10_bge-small-en-v1.5.png) | ![MTEB(eng, v2) Retrieval with nomic-embed-text-v1.5 + reranker](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ettin-reranker/mteb_ndcg10_nomic-embed-text-v1.5.png) |
| ![MTEB(eng, v2) Retrieval with embeddinggemma-300m + reranker](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ettin-reranker/mteb_ndcg10_embeddinggemma-300m.png) | ![MTEB(eng, v2) Retrieval with jina-embeddings-v5-text-small-retrieval + reranker](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ettin-reranker/mteb_ndcg10_jina-embeddings-v5-text-small-retrieval.png) | 

<details><summary>Full table of results (click to expand)</summary>

Mean NDCG@10 over the 6 embedder pairings, sorted descending. Our six models are in **bold**, and the teacher [`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) is underlined.

| Reranker | Params | MTEB(eng, v2) Retrieval NDCG@10 |
| --- | ---: | ---: |
| [`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B)<sup>†</sup> | 4.02B | 0.6367 |
| <u>[`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2)</u> | 1.54B | <u>0.6115</u> |
| **[`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1)** | **1.00B** | **0.6114** |
| **[`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1)** | **401M** | **0.6091** |
| **[`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)** | **151M** | **0.5994** |
| [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 596M | 0.5940 |
| [`mixedbread-ai/mxbai-rerank-base-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | 494M | 0.5920 |
| **[`cross-encoder/ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1)** | **68.6M** | **0.5915** |
| [`jinaai/jina-reranker-m0`](https://huggingface.co/jinaai/jina-reranker-m0) | 2.44B | 0.5856 |
| [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | 150M | 0.5843 |
| **[`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1)** | **32.8M** | **0.5779** |
| [`ibm-granite/granite-embedding-reranker-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2) | 150M | 0.5656 |
| **[`cross-encoder/ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1)** | **17.6M** | **0.5576** |
| [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 568M | 0.5526 |
| [`zeroentropy/zerank-2-reranker`](https://huggingface.co/zeroentropy/zerank-2-reranker)<sup>†</sup> | 4.02B | 0.5300 |
| [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large) | 560M | 0.5098 |
| [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22.7M | 0.5082 |
| [`cross-encoder/ms-marco-MiniLM-L12-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) | 33.4M | 0.5066 |
| [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) | 435M | 0.5063 |
| [`cross-encoder/ms-marco-MiniLM-L4-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L4-v2) | 19.2M | 0.4979 |
| [`mixedbread-ai/mxbai-rerank-xsmall-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) | 70.8M | 0.4968 |
| [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) | 278M | 0.4890 |
| [`mixedbread-ai/mxbai-rerank-base-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1) | 184M | 0.4865 |

<sup>†</sup> Capped to `max_seq_length=8192` (the 4B Qwen3-based rerankers don't fit on a single H100 80GB at native context). Native-context evaluation is likely higher.

</details>

<details><summary>Full table of NanoBEIR results (click to expand)</summary>

[NanoBEIR](https://huggingface.co/collections/sentence-transformers/nanobeir-with-bm25-rankings) is a fast 13-dataset subset of [BEIR](https://github.com/beir-cellar/beir) that uses 50 queries per dataset against up to 5000 documents each. NanoBEIR is what `metric_for_best_model` was set to during training (see [Evaluation](#evaluation)), and what I used to guide the experimentation.

| Reranker | Params | NanoBEIR mean NDCG@10 |
| --- | ---: | ---: |
| [`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) | 1.54B | 0.7318 |
| **[`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1)** | **1.00B** | **0.7237** |
| [`jinaai/jina-reranker-m0`](https://huggingface.co/jinaai/jina-reranker-m0) | 2.44B | 0.7197 |
| **[`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1)** | **401M** | **0.7193** |
| [`mixedbread-ai/mxbai-rerank-base-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | 494M | 0.7162 |
| **[`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)** | **151M** | **0.7086** |
| [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | 150M | 0.7017 |
| [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 568M | 0.6971 |
| **[`cross-encoder/ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1)** | **68.6M** | **0.6915** |
| [`ibm-granite/granite-embedding-reranker-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2) | 150M | 0.6909 |
| **[`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1)** | **32.8M** | **0.6825** |
| **[`cross-encoder/ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1)** | **17.6M** | **0.6746** |
| [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) | 435M | 0.6488 |
| [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large) | 560M | 0.6379 |
| [`cross-encoder/ms-marco-MiniLM-L12-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) | 33.4M | 0.6369 |
| [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22.7M | 0.6312 |
| [`cross-encoder/ms-marco-MiniLM-L4-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L4-v2) | 19.2M | 0.6298 |
| [`mixedbread-ai/mxbai-rerank-base-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1) | 184M | 0.6231 |
| [`mixedbread-ai/mxbai-rerank-xsmall-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) | 70.8M | 0.6136 |
| [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) | 278M | 0.6027 |

</details>

The smallest model I'm releasing, our 17M, beats the 33M `ms-marco-MiniLM-L12-v2` by +0.051 NDCG@10 (0.5576 vs 0.5066) on MTEB and +0.038 (0.6746 vs 0.6369) on NanoBEIR at roughly half the parameter count. The 32M beats the 568M `BAAI/bge-reranker-v2-m3` by +0.025 (0.5779 vs 0.5526) on MTEB, a 17x parameter gap. If you've been using one of the legacy MiniLM rerankers as the default in your retrieve-then-rerank stack, swapping in our 17M (or 32M) is a low-risk drop-in replacement, with a noticeable quality bump on both benchmarks.

Moving up the table, our 150M is the strongest reranker I tested in the under-600M range on MTEB, edging out the recent `Qwen/Qwen3-Reranker-0.6B` (596M) by +0.005 (0.5994 vs 0.5940) and beating every BAAI bge-reranker variant by 0.03 to 0.05. The 68M is also worth a mention: at 0.5915 it lands almost exactly on `Qwen3-Reranker-0.6B` (0.5940) while using a ninth of the parameters.

At the top of the released range, our 1B model closely tracks its teacher. It comes within 0.0001 of the 1.54B `mxbai-rerank-large-v2` on MTEB (0.6114 vs 0.6115) and within 0.008 on NanoBEIR, despite distilling from a model 54% larger than itself. The distillation effectively closes the gap to the teacher, which is what I was hoping to see going into this release.

The overall strongest reranker in the comparison is `Qwen/Qwen3-Reranker-4B` at 0.6367 MTEB, +0.025 above our 1B model. Closing that gap from the current recipe would likely require distilling from a stronger teacher (our teacher itself sits below `Qwen3-Reranker-4B`). For most retrieve-then-rerank workloads, our 1B at a quarter of the parameters (see [Speed](#speed)) is a much more practical pick.

### Speed

Quality numbers are only half of what matters for a reranker. The other half is whether its latency fits inside the budget you have between retrieval and showing results to the user. Let me walk through what I measured.

I benchmarked all six released models against thirteen public rerankers (strong baselines up to about 1B parameters) on a single NVIDIA H100 80GB. The queries and documents come from [`sentence-transformers/natural-questions`](https://huggingface.co/datasets/sentence-transformers/natural-questions) at its natural document-length distribution: most NQ answers are short, some are long. Documents are truncated at `max_length=512` to avoid giving the older models an unfair advantage. Each model uses its best supported attention implementation: Flash Attention 2 wherever the architecture supports it (BERT, XLM-RoBERTa, ModernBERT, Qwen2), SDPA where it doesn't, and eager for DeBERTa-v2 (which currently has neither FA2 nor SDPA support in `transformers`).

For every model an auto-batch search starts at batch size 8 and doubles until the GPU runs out of memory. At each batch size I run three timed passes and keep the median throughput, so a single unlucky run doesn't drag the number around. The reported throughput is at whichever batch size won.

**Table 1.** Throughput in pairs per second, all in `bfloat16`. Our six rerankers are in **bold**.

| Model | Params | Attn | pairs / second |
|---|---:|---|---|
| **[`cross-encoder/ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1)** | **17M** | FA2 | **7517** |
| **[`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1)** | **32M** | FA2 | **6602** |
| **[`cross-encoder/ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1)** | **68M** | FA2 | **4913** |
| [`cross-encoder/ms-marco-MiniLM-L4-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L4-v2) | 19M | FA2 | 4029 |
| [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22M | FA2 | 3817 |
| [`cross-encoder/ms-marco-MiniLM-L12-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) | 33M | FA2 | 3311 |
| **[`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)** | **150M** | FA2 | **3237** |
| [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) | 278M | FA2 | 2858 |
| [`mixedbread-ai/mxbai-rerank-xsmall-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) | 70M | eager | 2636 |
| [`mixedbread-ai/mxbai-rerank-base-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1) | 184M | eager | 1953 |
| **[`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1)** | **400M** | FA2 | **1738** |
| [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large) | 560M | FA2 | 1659 |
| [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 568M | FA2 | 1569 |
| [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | 150M | FA2 | 1418 |
| [`ibm-granite/granite-embedding-reranker-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2) | 150M | FA2 | 1404 |
| **[`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1)** | **1B** | FA2 | **928** |
| [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) | 435M | eager | 867 |
| [`mixedbread-ai/mxbai-rerank-base-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | 494M | FA2 | 809 |
| <u>[`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2)</u> | <u>1.5B</u> | FA2 | <u>387</u> |

Our 17M is the fastest reranker in the whole comparison, at 7517 pairs per second. That's almost twice the throughput of `ms-marco-MiniLM-L6-v2` (3817) and faster even than the smaller `ms-marco-MiniLM-L4-v2` (4029). And as you saw in the MTEB table earlier, our 17M is also more accurate than every MiniLM variant. If you're currently running a MiniLM cross-encoder, swapping to our 17M is a one-line change that improves both your latency and search quality.

Our 150M is an even more interesting comparison, because there are two direct architectural peers at exactly 150M parameters: [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) and [`ibm-granite/granite-embedding-reranker-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2). Both are built on the same ModernBERT-base backbone. Our 150M runs at 3237 pairs per second, while the two peers come in at 1418 and 1404 respectively, for a 2.3x speed gap.

All three 150M models use Flash Attention 2, but the two peers load through `AutoModelForSequenceClassification`, which keeps the inputs padded. So attention itself runs the FA2 kernel, but the rest of the model is still doing dense compute on padding tokens that don't contribute anything. Our modular `Transformer` module (see [Architecture Details](#architecture-details) above) propagates unpadded inputs all the way through the model, so every layer only spends compute on real tokens. That's the difference between getting some of FA2's benefit and getting all of it.

At the bottom of the table, our 1B model hits 928 pairs per second, which is 2.4x faster than the 1.54B teacher `mxbai-rerank-large-v2` (387 pairs per second) while matching its MTEB score within 0.0001. The teacher is Qwen2-based with a prompt-template overhead per pair, so the distilled student inherits the teacher's calibration and judgement but skips all the runtime baggage. This is honestly the most satisfying single number in the whole release for me.

One unfortunate note: the DeBERTa-v2-based `mxbai-rerank-{xsmall,base,large}-v1` series ends up much slower than the rest of the table because DeBERTa-v2 currently supports neither Flash Attention 2 nor SDPA in `transformers`. The 70M `mxbai-rerank-xsmall-v1` runs at 2636 pairs per second, about half the throughput of our 68M at almost the same parameter count. The models themselves are perfectly fine, they just don't get to use modern attention kernels.

<details><summary>Same benchmark on a consumer GPU (RTX 3090, 24 GB)</summary>

If you're self-hosting on a consumer card rather than a datacenter GPU, here's the same throughput sweep on an RTX 3090. Same benchmark setup as Table 1: `bfloat16`, best-supported attention per model, three-trial median throughput at the largest batch that fits.

| Model | Params | Best attn | pairs / second |
|---|---:|---|---:|
| **[`cross-encoder/ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1)** | **17M** | FA2 | **9008** |
| [`cross-encoder/ms-marco-MiniLM-L4-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L4-v2) | 19M | FA2 | 5071 |
| **[`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1)** | **32M** | FA2 | **4497** |
| [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22M | FA2 | 4234 |
| [`cross-encoder/ms-marco-MiniLM-L12-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) | 33M | FA2 | 2847 |
| **[`cross-encoder/ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1)** | **68M** | FA2 | **1916** |
| [`mixedbread-ai/mxbai-rerank-xsmall-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) | 70M | eager | 1677 |
| [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) | 278M | FA2 | 1329 |
| **[`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)** | **150M** | FA2 | **982** |
| [`mixedbread-ai/mxbai-rerank-base-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1) | 184M | eager | 772 |
| [`ibm-granite/granite-embedding-reranker-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2) | 150M | FA2 | 598 |
| [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | 150M | FA2 | 586 |
| [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large) | 560M | FA2 | 448 |
| [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 568M | FA2 | 436 |
| **[`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1)** | **400M** | FA2 | **429** |
| [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) | 435M | eager | 266 |
| [`mixedbread-ai/mxbai-rerank-base-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | 494M | FA2 | 221 |
| **[`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1)** | **1B** | FA2 | **189** |
| <u>[`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2)</u> | <u>1.5B</u> | FA2 | <u>69</u> |

Our 17M is still the fastest model in the table at 9008 pairs per second, actually higher than its H100 number, which suggests that at tiny sizes raw compute isn't the bottleneck and the H100's extra muscle doesn't translate. The middle of the table reshuffles a bit, with the MiniLM rerankers overtaking our 32M and 68M, and the 1B slipping behind `mxbai-rerank-base-v2` (189 vs 221 pairs per second). Our 150M model still holds a solid lead over the two 150M ModernBERT-based peers, and the teacher-replacement story still holds, with our 1B at 2.7x the throughput of the 1.5B `mxbai-rerank-large-v2` (189 vs 69 pairs per second).

</details>

<details><summary>Same benchmark on CPU (Intel Core i7-13700K)</summary>

| Model | Params | Best attn | pairs / second |
|---|---:|---|---:|
| **[`cross-encoder/ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1)** | **17M** | SDPA | **267.4** |
| [`cross-encoder/ms-marco-MiniLM-L4-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L4-v2) | 19M | SDPA | 206.2 |
| [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22M | SDPA | 143.9 |
| **[`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1)** | **32M** | SDPA | **92.5** |
| [`cross-encoder/ms-marco-MiniLM-L12-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) | 33M | SDPA | 75.9 |
| [`mixedbread-ai/mxbai-rerank-xsmall-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) | 70M | eager | 38.9 |
| **[`cross-encoder/ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1)** | **68M** | SDPA | **31.2** |
| [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) | 278M | SDPA | 19.2 |
| [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | 150M | SDPA | 14.7 |
| [`ibm-granite/granite-embedding-reranker-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2) | 150M | SDPA | 14.5 |
| **[`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)** | **150M** | SDPA | **14.0** |
| [`mixedbread-ai/mxbai-rerank-base-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1) | 184M | eager | 13.4 |
| [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large) | 560M | SDPA | 6.2 |
| [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 568M | SDPA | 6.0 |
| **[`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1)** | **400M** | SDPA | **5.2** |
| [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) | 435M | eager | 4.3 |
| [`mixedbread-ai/mxbai-rerank-base-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | 494M | SDPA | 3.5 |
| **[`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1)** | **1B** | SDPA | **2.1** |

On CPU, we can't take advantage of bf16, Flash Attention 2, or unpadding, so the latency story is a bit simpler: the higher the parameter count, the slower the model. The 17M model is considerably faster than `ms-marco-MiniLM-L6-v2` (267.4 vs 143.9 pairs per second) and even faster than the smaller `ms-marco-MiniLM-L4-v2` (206.2). As expected, our 150M model lands alongside the two 150M peers (14.0 vs 14.5 and 14.7 pairs per second) now that unpadding no longer applies. If you're CPU-bound, our 17M and 32M are the practical picks.

</details>

To explain where the speed comes from, the next table sweeps `fp32+SDPA`, `bf16+SDPA`, and `bf16+FA2` for our six models using the same bench config. The FA2 column is split in two: one with the inputs still padded (what a wrapped model would see) and one with unpadded inputs (what our modular `Transformer` actually does). The rightmost column is what our models use by default when FA2 is enabled.

**Table 2.** Precision and attention ablation for the six released sizes at `max_length=512` on natural NQ documents. Each cell shows pairs / second with the multiplier relative to `fp32+SDPA` in parentheses, and peak GPU memory on the second line. The rightmost column (in **bold**) is the configuration our models use by default when FA2 is enabled.

| Model | Params | fp32+SDPA | bf16+SDPA | bf16+FA2 w. padding | **bf16+FA2 w.o. padding** |
|---|---|---|---|---|---|
| [`ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1) | 17M | 4402 (1.00x)<br>0.8 GB | 4523 (1.03x)<br>2.2 GB | 3744 (0.85x)<br>1.9 GB | **7517 (1.71x)**<br>**1.4 GB** |
| [`ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1) | 32M | 3307 (1.00x)<br>1.2 GB | 4357 (1.32x)<br>1.6 GB | 3040 (0.92x)<br>2.9 GB | **6602 (2.00x)**<br>**1.1 GB** |
| [`ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1) | 68M | 1364 (1.00x)<br>1.0 GB | 2861 (2.10x)<br>2.2 GB | 2003 (1.47x)<br>2.0 GB | **4913 (3.60x)**<br>**1.5 GB** |
| [`ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1) | 150M | 671 (1.00x)<br>1.6 GB | 1942 (2.90x)<br>1.8 GB | 1396 (2.08x)<br>3.1 GB | **3237 (4.83x)**<br>**1.4 GB** |
| [`ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1) | 400M | 266 (1.00x)<br>2.5 GB | 1113 (4.18x)<br>1.8 GB | 864 (3.25x)<br>2.7 GB | **1738 (6.53x)**<br>**2.2 GB** |
| [`ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1) | 1B | 112 (1.00x)<br>4.6 GB | 630 (5.60x)<br>2.8 GB | 522 (4.64x)<br>3.6 GB | **928 (8.26x)**<br>**4.5 GB** |

The total speedup from `bf16+FA2 w.o. padding` over the `fp32+SDPA` baseline grows sharply with model size, from 1.71x on the 17M to 8.26x on the 1B. Most of that growth comes from `bf16` alone: the `fp32+SDPA` to `bf16+SDPA` step gives the 17M only a 1.03x speedup but gives the 1B a full 5.60x speedup, also due to the lowered memory cost allowing for bigger batch sizes. In short, `bfloat16` is the biggest single contributor to the overall speedup.

Unexpectedly, turning on FA2 while the inputs are still padded is actually slower than `bf16+SDPA` at every size in the release. The FA2 kernel prefers an unpadded format, and when you feed it padded inputs you pay the bookkeeping overhead of converting between formats while still spending compute on the padding tokens themselves. So the `bf16+FA2 w. padding` column is roughly what you'd measure if you swapped `sdpa` for `flash_attention_2` in `model_kwargs` without changing anything else about the model loader. This is the situation that `gte-reranker-modernbert-base` and `granite-embedding-reranker-english-r2` from Table 1 are in.

Lastly, going from `bf16+FA2 w. padding` to `bf16+FA2 w.o. padding` is worth between 1.78x (1B) and 2.45x (68M) of additional throughput, and it also cuts peak memory considerably, allowing for higher batch sizes.

So my recommendation is simple: enable `bf16` and FA2 together. The six Ettin rerankers will use unpadded inputs by default, since that's what the modular `Transformer` module from the [Architecture Details](#architecture-details) section is set up for. The full snippet is the same as in the [Usage](#usage) section above:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ettin-reranker-150m-v1",
    model_kwargs={
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",  # See tip below
    },
)
```

> [!TIP]
> Use `pip install kernels` to install FA2. It ships pre-built kernels for a wide range of GPU architectures, CUDA versions, and operating systems.

One caveat for other CrossEncoders: the full speedup is only available for models built with a modular `Transformer` like the Ettin rerankers. Applying the same two flags to a CrossEncoder that loads through `AutoModelForSequenceClassification` lands you in the slower `bf16+FA2 w. padding` column of Table 2 instead.

## Training

The training script below started as the output of the new [`train-sentence-transformers` Agent Skill](https://github.com/huggingface/sentence-transformers/tree/main/skills), shipped in [Sentence Transformers v5.5.0](https://github.com/huggingface/sentence-transformers/releases/tag/v5.5.0). If you use an AI coding agent (Claude Code, Codex, Cursor, Gemini CLI, ...), you can install the skill and ask it to fine-tune a `SentenceTransformer`, `CrossEncoder`, or `SparseEncoder` model on your data. The skill carries version-aware guidance for base model selection, loss and evaluator choice, hard-negative mining, distillation, LoRA, Matryoshka, multilingual training, and static embeddings, plus template scripts for each model type.

```bash
hf skills add train-sentence-transformers --claude   # symlinks into .claude/skills/
hf skills add train-sentence-transformers --global   # under ~/.agents/skills/
```

A prompt like *"Fine-tune a cross-encoder reranker on `(query, document)` pairs from my dataset, mine hard negatives, and push to my Hub repo"* will produce a runnable script you can then iterate on. That's how I started working on the recipe below.

All six rerankers were trained with the same single-stage recipe. Only the learning rate and the per-device batch size vary per model size. The full training script is ~150 lines and uses one published dataset.

The recipe converged after a single sweep across model sizes. Each size's learning rate was tuned by a small grid search on a ~15% subset of the final training data, and the resulting LRs transferred cleanly to the full-data runs without re-tuning. No per-size tuning beyond LR was needed.

### Distillation recipe

Most published reranker recipes train on human-labeled relevance triples (a query, one positive document, and optionally hard negatives) with a contrastive, pointwise, pairwise, or listwise loss like [`MultipleNegativesRankingLoss`](https://sbert.net/docs/package_reference/cross_encoder/losses.html#multiplenegativesrankingloss), [`BinaryCrossEntropyLoss`](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss), [`RankNetLoss`](https://sbert.net/docs/package_reference/cross_encoder/losses.html#ranknetloss), or [`LambdaLoss`](https://sbert.net/docs/package_reference/cross_encoder/losses.html#lambdaloss), respectively. See my earlier [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker) blogpost, for example.

But this approach has a few practical and theoretical drawbacks. First, positives need to be human-labeled, which is expensive and slow to scale across many domains. Second, the model only ever sees a label for the small subset of `(query, document)` pairs that someone went through. Especially after hard negative mining, you end up with a lot of false negatives, e.g. as shown in [Hard Negatives, Hard Lessons](https://arxiv.org/abs/2505.16967). Third, the binary nature of this labeling doesn't match reality, where some documents are simply more relevant than others.

I took a different route here: pointwise MSE distillation from an existing strong teacher reranker. The setup is simple enough to describe in three lines:

- **Teacher**: [`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) (1.54B parameters).
- **Loss**: [`MSELoss`](https://sbert.net/docs/package_reference/cross_encoder/losses.html#mseloss) on the raw teacher logits (range ~[−12, 22]), i.e. without rescaling.
- **Training data**: ~143M `(query, document, teacher_score)` triples.

### Dataset

I've released the training data as a single Hugging Face dataset, [`cross-encoder/ettin-reranker-v1-data`](https://huggingface.co/datasets/cross-encoder/ettin-reranker-v1-data), assembled from two sources. Each source is kept as its own split so the provenance is transparent:

1. LightOn pre-training data ([`lightonai/embeddings-pre-training`](https://huggingface.co/datasets/lightonai/embeddings-pre-training), non-curated): 32 splits covering broad-domain text similarity signal (MTP, FW-EDU, Reddit, PAQ, S2ORC, Amazon, Wikipedia, MS MARCO, etc.). I limit the number of samples for some of the splits, resulting in ~110M `(query, document, similarity)` triples in total. 
2. Rescored retrieval data from [`lightonai/embeddings-fine-tuning`](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning): 7 splits (`msmarco`, `hotpotqa`, `trivia`, `nq`, `squadv2`, `fiqa`, `fever`). The source dataset has up to 2048 candidate documents per query (initially scored with [`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)), which I rescored with [`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) and uploaded as [`cross-encoder/lightonai-embeddings-fine-tuning-reranked-v1`](https://huggingface.co/datasets/cross-encoder/lightonai-embeddings-fine-tuning-reranked-v1). That dataset subsamples each query's 2048 candidates down to 256 using the [Jang et al.](https://arxiv.org/abs/2604.04734) quantile-anchor recipe (all positives + top-16 hard + ~239 quantile-anchor stratified). For training, I pick 64 of those 256 per query: 32 from the score-sorted head (the positive plus the hardest negatives) and 32 medium-difficulty negatives sampled from a band further down the teacher's ranking. See the [dataset card](https://huggingface.co/datasets/cross-encoder/ettin-reranker-v1-data) for the exact rank positions.

Total: ~143M `(query, document, score)` triples, plus a held-out 5K-row eval split (the tail of `quora`) that drives the in-training eval loss.

### Training Arguments

Most hyperparameters are constant across model sizes:

```python
CrossEncoderTrainingArguments(
    num_train_epochs=1,                    # I chose more data over more epochs
    per_device_train_batch_size=...,       # global_batch_size // world_size (see table below)
    gradient_accumulation_steps=1,
    learning_rate=...,                     # per-size, see table
    warmup_ratio=0.03,                     # ~3% linear warmup, then linear decay (default)
    bf16=True,                             # FA2 + bf16 throughout
    eval_strategy="steps",
    eval_steps=0.05,                       # NanoBEIR every 5% of training
    save_strategy="steps",
    save_steps=0.05,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
    seed=12,
)
```

Only the learning rate and global batch size very per model size.

| Size | Learning rate | Global batch size |
| --- | ---: | ---: |
| 17m  | 2.4e-4 | 1024 |
| 32m  | 1.2e-4 | 512  |
| 68m  | 3e-5   | 256  |
| 150m | 1.5e-5 | 192  |
| 400m | 7e-6   | 256  |
| 1b   | 3e-6   | 512  |

`global_batch_size` is `per_device_batch_size x world_size x gradient_accumulation_steps`. On a single 8-GPU node, the 1024 global batch for 17m means `per_device=128`. On 8 nodes, it means `per_device=8`. The training script computes `per_device_batch_size` from `global_batch_size // world_size` so the same script works at any node count. The global batch size could be made more consistent, but I found that the above values worked well and didn't want to retune them just for the sake of consistency.

### Evaluation

I monitored NanoBEIR mean NDCG@10 during training (eval every 5% of steps) and used it as the `metric_for_best_model` for `load_best_model_at_end`. NanoBEIR is fast, so I could afford it 20 times per training run. After training, I evaluated both the best checkpoint (according to NanoBEIR) and the last checkpoint on the full MTEB(eng, v2) Retrieval benchmark. The final release checkpoint was the one that did best on MTEB. The NanoBEIR-preferred checkpoint won for all sizes except 68m, where the last checkpoint was slightly stronger.

### Overall Training Script

The complete script (what every released model was trained with) is a single file. Only `ENCODER_SIZE` changes per run, and everything else is automatic:

```python
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

from sentence_transformers import CrossEncoder
from sentence_transformers.base.modules import Dense
from sentence_transformers.cross_encoder import (
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import MSELoss
from sentence_transformers.sentence_transformer.modules import LayerNorm, Pooling, Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Per-size config. I swept the learning rates with these global (effective) batch sizes,
# also by incorporating accum_steps
CONFIGS: dict[str, dict] = {
    "17m":  {"base_model_name": "jhu-clsp/ettin-encoder-17m",  "learning_rate": 2.4e-4, "global_batch_size": 1024},
    "32m":  {"base_model_name": "jhu-clsp/ettin-encoder-32m",  "learning_rate": 1.2e-4, "global_batch_size": 512},
    "68m":  {"base_model_name": "jhu-clsp/ettin-encoder-68m",  "learning_rate": 3e-5,   "global_batch_size": 256},
    "150m": {"base_model_name": "jhu-clsp/ettin-encoder-150m", "learning_rate": 1.5e-5, "global_batch_size": 192},
    "400m": {"base_model_name": "jhu-clsp/ettin-encoder-400m", "learning_rate": 7e-6,   "global_batch_size": 256},
    "1b":   {"base_model_name": "jhu-clsp/ettin-encoder-1b",   "learning_rate": 3e-6,   "global_batch_size": 512},
}
ENCODER_SIZE = "17m"

def main() -> None:
    config = CONFIGS[ENCODER_SIZE]
    encoder_id = config["base_model_name"]
    learning_rate = config["learning_rate"]
    global_batch_size = config["global_batch_size"]

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_device_batch_size = global_batch_size // world_size
    dataloader_workers = 0 if world_size > 8 else 4
    run_name = f"ettin-reranker-{ENCODER_SIZE}-lr{learning_rate:.0e}"

    # 1. Load a model to finetune with model card data
    # The model mirrors ModernBertForSequenceClassification, but with a 'headless' Transformer that just loads
    # AutoModel. This allows for unpadding with FA2, which isn't possible with AutoModelForSequenceClassification.
    # This speeds up training considerably, while heavily reducing memory usage.
    torch.manual_seed(12)
    transformer = Transformer(encoder_id, model_kwargs={"attn_implementation": "flash_attention_2"})
    transformer.model.config.num_labels = 1
    embedding_dimension = transformer.get_embedding_dimension()
    pooling = Pooling(embedding_dimension=embedding_dimension, pooling_mode="cls")
    dense_inner = Dense(
        in_features=embedding_dimension, out_features=embedding_dimension, bias=False,
        activation_function=nn.GELU(),
        module_input_name="sentence_embedding", module_output_name="sentence_embedding",
    )
    norm = LayerNorm(dimension=embedding_dimension)
    dense_score = Dense(
        in_features=embedding_dimension, out_features=1, bias=True,
        activation_function=nn.Identity(),
        module_input_name="sentence_embedding", module_output_name="scores",
    )
    model = CrossEncoder(
        modules=[transformer, pooling, dense_inner, norm, dense_score],
        num_labels=1,
        activation_fn=nn.Identity(),
        model_card_data=CrossEncoderModelCardData(
            model_name=f"Ettin Reranker {ENCODER_SIZE} distilled from mxbai-rerank-large-v2",
            language="en",
            license="apache-2.0",
        ),
    )
    actual_attn = getattr(model[0].model.config, "_attn_implementation", None)
    if not (actual_attn and "flash" in actual_attn.lower()):
        logging.warning(f"FA2 may not be active (attn_impl={actual_attn!r}); training will be slower.")

    # 2. Load the dataset. Each config is one source subset (32 lighton + 7 rerank retrieval
    # domains). The held-out eval rows live as the 'validation' split of the 'quora' config.
    dataset_repo = "cross-encoder/ettin-reranker-v1-data"
    train_pieces = []
    eval_dataset = None
    for config_name in get_dataset_config_names(dataset_repo):
        dataset = load_dataset(dataset_repo, config_name)
        train_pieces.append(dataset["train"])
        if "validation" in dataset:
            eval_dataset = dataset["validation"]
    train_dataset = concatenate_datasets(train_pieces)
    print(train_dataset)

    # 3. Define a loss function
    loss = MSELoss(model)

    # 4. Specify training arguments
    args = CrossEncoderTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        bf16=True,
        eval_strategy="steps",
        eval_steps=0.05,
        save_strategy="steps",
        save_steps=0.05,
        save_total_limit=5,
        logging_steps=0.025,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        dataloader_num_workers=dataloader_workers,
        run_name=run_name,
        seed=12,
    )

    # 5. Create an evaluator
    evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq", "fiqa2018", "touche2020", "scifact",
                       "hotpotqa", "arguana", "fever", "dbpedia", "climatefever", "scidocs",
                       "quoraretrieval"],
        batch_size=per_device_batch_size,
        always_rerank_positives=False,
        show_progress_bar=False,
    )

    # 6. Create a trainer
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    # 7. Evaluate before training
    if trainer.is_world_process_zero():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            evaluator(model)

    # 8. Train
    trainer.train()

    # 9. Evaluate the final model
    if trainer.is_world_process_zero():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            evaluator(model)

    # 10. Save the final model
    final_dir = f"models/{run_name}/final"
    model.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
```

For multi-node training (anything past 17m/32m), launch the same script with `torchrun`:

```bash
# Single-node (17m, 32m): defaults work
python train.py

# Multi-node 4n setup for 150m, preserves global_batch_size=192:
torchrun --nproc_per_node=8 --nnodes=4 ... train.py
```

## Conclusion

The ettin-reranker-v1 family, trained with a single simple recipe, is state-of-the-art at every released size up to 1B parameters. Pointwise MSE distillation from a strong teacher onto a broad-domain and retrieval-specific mix scales cleanly from 17M to 1B parameters, with only the learning rate and per-device batch size changing between sizes.

Every ettin-reranker-v1 model beats the `ms-marco-MiniLM-L*-v2` family by a comfortable margin on MTEB and NanoBEIR. `cross-encoder/ettin-reranker-150m-v1` is the strongest mid-tier reranker I tested in the under-600M range, `cross-encoder/ettin-reranker-400m-v1` lands within 0.0024 of the 1.54B teacher's MTEB score, and `cross-encoder/ettin-reranker-1b-v1` matches that teacher within 0.0001.

Everything in one place:

- **Models**:
    - [`cross-encoder/ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1)
    - [`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1)
    - [`cross-encoder/ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1)
    - [`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)
    - [`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1)
    - [`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1)
- **Dataset**: [`cross-encoder/ettin-reranker-v1-data`](https://huggingface.co/datasets/cross-encoder/ettin-reranker-v1-data) with ~143M `(query, document, label)` triples, kept as 39 named splits so the provenance of every row is visible.
- **Training script**: the ~150 lines in [Overall Training Script](#overall-training-script) above, which is the same script used for all six models.

If you build something on top of these, please let me know! I'd genuinely love to see what people do with them, and if you manage to train better rerankers using the released data, even better. The recipe is intentionally simple, partly so that there's plenty of headroom for someone else to improve it. Train a stronger teacher and the same script can keep producing better students.

## Acknowledgements

I'd like to thank the Ettin team (Orion Weller, Kathryn Ricci, Marc Marone, Antoine Chaffin, Dawn Lawrie, and Benjamin Van Durme) for [building the base encoders](https://huggingface.co/blog/ettin) that these rerankers are built on, the LightOn team (Antoine Chaffin, Raphael Sourty, Paulo Moura, and Amélie Chatelain) for [their work on the training data collection](https://huggingface.co/blog/lightonai/denseon-lateon), and the Mixedbread AI team (Xianming Li, Aamir Shakir, Rui Huang, Tsz-fung Andrew Lee, Julius Lipp, Benjamin Clavié, and Jing Li) for [their work on the teacher model](https://arxiv.org/abs/2506.03487).

## Citation

If you use the ettin-reranker-v1 family or any of the released artifacts, please cite this blogpost:

```bibtex
@misc{aarsen2026ettin-reranker,
    title = "Introducing the Ettin Reranker Family",
    author = "Aarsen, Tom",
    year = "2026",
    publisher = "Hugging Face",
    url = "https://huggingface.co/blog/ettin-reranker",
}
```