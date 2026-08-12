---
title: "Multi-Vector (Late Interaction) Embedding Models with Sentence Transformers"
thumbnail: /blog/assets/multi-vector-encoder/st-hf-lighton-thumbnail.png
authors:
- user: tomaarsen
- user: NohTow
  guest: true
  org: lightonai
- user: raphaelsty
  guest: true
  org: lightonai
---

# Multi-Vector (Late Interaction) Embedding Models with Sentence Transformers

[Sentence Transformers](https://sbert.net/) is a Python library for using and training embedding and reranker models for applications like retrieval augmented generation, semantic search, and more. With the v6.0 update, it gains a fourth model type: `MultiVectorEncoder`, for ColBERT-style late interaction retrieval. Any [PyLate](https://github.com/lightonai/pylate) checkpoint and any [Stanford-NLP ColBERT](https://github.com/stanford-futuredata/ColBERT) checkpoint loads straight into it, and [colpali-engine](https://github.com/illuin-tech/colpali) models for visual document retrieval can be used too, through the same familiar API you already use for dense, sparse, and reranker models.

Where a regular embedding model compresses a whole text into one vector, a multi-vector model keeps **one vector per token** and scores query against document with the MaxSim operator. That preserves token-level matching information that a single vector has to average away, which usually means stronger retrieval at the cost of a bigger index. It's also the state of the art for visual document retrieval, where a text query is matched against page images directly, with no OCR step in between.

In this blogpost, we'll show you how to use these models: loading the various checkpoint formats, encoding and scoring, plugging them into a search stack, running them on page images, and keeping the index affordable.

<!--
> [!TIP]
> If you want to train your own multi-vector models, check out the companion blogpost: [Training and Finetuning Multi-Vector Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-multi-vector-encoder).
-->

## Table of Contents

* [What are Multi-Vector Models?](#what-are-multi-vector-models)
    + [The MaxSim Operator](#the-maxsim-operator)
    + [What You Gain, and What It Costs](#what-you-gain-and-what-it-costs)
* [Installation](#installation)
* [Loading a Model](#loading-a-model)
    + [Inspecting What a Checkpoint Configured](#inspecting-what-a-checkpoint-configured)
* [Encoding Queries and Documents](#encoding-queries-and-documents)
* [Scoring with MaxSim](#scoring-with-maxsim)
    + [Score Magnitude and MeanMaxSim](#score-magnitude-and-meanmaxsim)
* [Semantic Search](#semantic-search)
* [Retrieve and Rerank](#retrieve-and-rerank)
* [Indexing](#indexing)
* [Visual Document Retrieval](#visual-document-retrieval)
* [Interpretability](#interpretability)
* [Token Pooling](#token-pooling)
* [Speeding Up Inference](#speeding-up-inference)
* [Evaluating a Model](#evaluating-a-model)
* [Coming from PyLate or colpali-engine](#coming-from-pylate-or-colpali-engine)
* [Supported Models](#supported-models)
* [Additional Resources](#additional-resources)

## What are Multi-Vector Models?

A dense embedding model reads a text and returns a single fixed-size vector. Everything the model noticed has to fit in those 384, 768, or 1024 numbers, and similarity is one dot product between two such summaries. This works remarkably well, but the compression is lossy in a specific way: a rare entity, an exact identifier, or one crucial clause in a long passage all have to compete for room in the same vector.

A multi-vector model (also called a late-interaction or ColBERT-style model, after the [ColBERT paper](https://arxiv.org/abs/2004.12832)) skips that compression. It runs the same transformer, but instead of pooling the token embeddings into one vector, it projects each token embedding down to a small dimension (classically 128) and keeps all of them. A 9-token document becomes a 9x128 matrix, not a 1x128 vector.

The interaction between query and document is then deferred until scoring time, which is where the name "late interaction" comes from. A cross-encoder interacts early (both texts go through the model together, which is accurate but too slow to precompute). A bi-encoder barely interacts at all (one dot product between two finished summaries). Late interaction sits in between: documents are still encoded independently and can be indexed offline, but scoring compares every query token against every document token.

![Dense embedding versus multi-vector late interaction: a dense model encodes each text into one vector and scores with cosine similarity, while a multi-vector model keeps one vector per token and scores every query token against every document token with MaxSim](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/maxsim_explainer.gif)

### The MaxSim Operator

Scoring uses MaxSim: for each query token, take its highest similarity against any document token, then sum those maxima across the query.

$$\text{MaxSim}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} Q_i \cdot D_j$$

Because the token embeddings are L2-normalized, each of those dot products is a cosine similarity in `[-1, 1]`, so the sum lands in `[-|Q|, |Q|]`.

You can read the operator as a soft alignment: every query token points at the one document token that best explains it, and the score is how well the document explains the query overall.

The alignment doesn't have to be lexical, since the token embeddings are contextualized. Encode "Where do penguins live?" against "Penguins inhabit Antarctica." with [`lightonai/mLateOn`](https://huggingface.co/lightonai/mLateOn) and the query token `live` finds its best match on `inhabit` at 0.94, a word it shares no characters with! It isn't one-to-one either, since several query tokens routinely settle on the same document token. But when an exact match does matter to you (a product code, a surname, a function name), MaxSim has a token sitting right there to match it, where a single-vector model had to fold it into an average.

### What You Gain, and What It Costs

You gain retrieval quality, particularly on queries where one specific piece of a document is what makes it relevant, and on out-of-domain data where a dense model's compression was tuned for a different distribution.

The cost is index size. One vector per token instead of one vector per document is a lot more vectors, only partly offset by the smaller dimension. Encoding 4,874 Natural Questions passages with [`lightonai/LateOn`](https://huggingface.co/lightonai/LateOn) produced 608,414 token vectors, an average of 124.8 per passage:

| Representation | Vectors | Dimensions | float32 index |
| --- | ---: | ---: | ---: |
| Dense, [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 4,874 | 384 | 7.5 MB |
| Dense, [`gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 4,874 | 768 | 15.0 MB |
| Multi-vector, `LateOn` | 608,414 | 128 | 311.5 MB |

That's about 42x the storage of the MiniLM index, or 62 KiB per passage. [Token Pooling](#token-pooling) shrinks that index, and [Retrieve and Rerank](#retrieve-and-rerank) avoids building one at all.

With the tradeoff in mind, let's get a model running.

## Installation

Multi-vector models work with a plain install:

```bash
pip install -U sentence-transformers
```

For ColPali-style visual document retrieval, you also need the image dependencies (see [Installation](https://sbert.net/docs/installation.html) for all extras):

```bash
pip install -U "sentence-transformers[image]"
```

> [!NOTE]
> Sentence Transformers v6.0 requires `transformers` v5.x, `torch` 2.2+, and `huggingface-hub` v1.x. If you pin any of those lower, plan the upgrade first. See the [Migration Guide](https://sbert.net/docs/migration_guide.html) for the full list of breaking changes.

## Loading a Model

Loading a multi-vector model looks exactly like loading any other Sentence Transformers model:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")
```

To find models that work, look for the [`sentence-transformers` tag](https://huggingface.co/models?library=sentence-transformers&other=multi-vector) on the Hub. Anything carrying it loads with the line above, whether it started life as a PyLate checkpoint, a Stanford-NLP ColBERT checkpoint, or a ColPali-family model for visual document retrieval. We're working through the ecosystem to get that tag onto every model that works, so the list keeps growing.

Underneath, `MultiVectorEncoder` reads each of the formats these checkpoints have been published in over the years, so PyLate and Stanford-NLP checkpoints load directly even where the tag hasn't been added yet:

```python
from sentence_transformers import MultiVectorEncoder

# Native Sentence Transformers checkpoints. PyLate builds on the same schema,
# so any PyLate checkpoint loads identically
model = MultiVectorEncoder("lightonai/LateOn")
model = MultiVectorEncoder("mixedbread-ai/mxbai-edge-colbert-v0-17m")
model = MultiVectorEncoder("LiquidAI/LFM2-ColBERT-350M")

# Any Stanford-NLP ColBERT checkpoint, detected via the `HF_ColBERT` architecture
# marker. The inline projection weight and the recipe come from `artifact.metadata`
model = MultiVectorEncoder("colbert-ir/colbertv2.0")
model = MultiVectorEncoder("answerdotai/answerai-colbert-small-v1")

# A bare transformer: a fresh random projection is appended, so training is required
model = MultiVectorEncoder("answerdotai/ModernBERT-base")
```

Visual document retrieval models are the exception. ColPali-family checkpoints ship in colpali-engine's own format, which carries no information Sentence Transformers can use, so each one needs a small configuration added to its repository before it loads. Most of that work is done and waiting to be merged. See [Supported Models](#supported-models) for the current state and how to load them today.

### Inspecting What a Checkpoint Configured

Multi-vector models carry a handful of recipe knobs that differ per checkpoint: marker prefixes for queries and documents, length caps, whether queries are padded out with `[MASK]` tokens, and which tokens are skipped when scoring documents. All of them live in the module configs, so `print(model)` shows you exactly what you loaded. Here's the original ColBERTv2 checkpoint, which pads every query to exactly 32 tokens and truncates documents at 180:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("colbert-ir/colbertv2.0")
print(model)
"""
MultiVectorEncoder(
  (0): Transformer({..., 'document_length': 180,
                    'query_expansion': {'strategy': 'fixed', 'attend': False, 'token': None, 'length': 32}})
  (1): Dense({'in_features': 768, 'out_features': 128, 'bias': False, ...})
  (2): MultiVectorMask({'skiplist_words': ['!', '"', '#', ...], 'skiplist_tasks': ['document'], ...})
  (3): Normalize({...})
)
"""
print(model.prompts)
# {'query': '[unused0] ', 'document': '[unused1] '}
```

That's the classic ColBERT pipeline: a `Transformer` producing contextualized token embeddings, a token-level `Dense` projecting each of them to 128 dimensions, a `MultiVectorMask` deciding which tokens count during scoring, and a token-level `Normalize`. Other checkpoints fill in different values. `lightonai/GTE-ModernColBERT-v1` uses the same four modules with `[Q] ` and `[D] ` prompts, no query expansion, and caps of 48 and 300.

You rarely need to touch any of this, since every released checkpoint configures its own. It matters when you build a model from a bare backbone, which is covered in [Creating Custom Models](https://sbert.net/docs/multi_vector_encoder/usage/custom_models.html).

## Encoding Queries and Documents

Multi-vector models are asymmetric: queries and documents go through different prefixes, different length caps, and different scoring masks. Unlike many dense models, where the two are interchangeable, [`encode_query()`](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html#sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_query) and [`encode_document()`](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html#sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_document) are required to get correct embeddings:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("lightonai/mLateOn")

queries = ["What is the capital of France?"]
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)

print(query_embeddings[0].shape)
# (10, 128)
print(document_embeddings[0].shape, document_embeddings[1].shape)
# (10, 128) (10, 128)
```

Note what you get back: a *list* of 2D arrays, one per input, each of shape `(num_tokens, embedding_dim)`. Unlike dense embeddings, you can't stack these into one rectangular array, because every input has its own token count. These two documents happen to tokenize to the same length, but a longer passage would produce a taller matrix.

Each call applies the model's own recipe for you. `encode_query` prepends the query marker, expands the query to a fixed length if the checkpoint asks for it, and caps it at the query length. `encode_document` prepends the document marker, caps at the document length, and drops any skiplisted tokens (punctuation, for most checkpoints) from the scoring mask.

The usual `encode()` arguments all still apply, so `batch_size`, `show_progress_bar`, `convert_to_tensor`, `device`, and multi-process pools work the way you'd expect:

```python
document_embeddings = model.encode_document(
    documents,
    batch_size=64,
    convert_to_tensor=True,
    show_progress_bar=True,
)
```

## Scoring with MaxSim

[`model.similarity()`](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html#sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.similarity) computes the full all-pairs MaxSim matrix:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("lightonai/LateOn")

query_embeddings = model.encode_query(["Which planet is known as the Red Planet?"])
document_embeddings = model.encode_document([
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
])

scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[10.7942, 11.1104, 10.9743, 11.0811]])
```

Mars wins, as it should. Note how close the runners-up are: Saturn also contains the literal phrase "the Red Planet", and Jupiter is a planet with a red spot, so a token-level operator has plenty to latch onto in all three. The ordering is what matters.

Scores often sit this close together, as [GLInt](https://huggingface.co/blog/chungimungi/glint#1-mining-in-maxsim-space) shows by measuring the spread across a full candidate pool. MaxSim takes a *maximum* per query token, so a document will usually give every query token some decent best match, and scores start from a floor. Contextualized token embeddings are also anisotropic, clustering in a narrow cone rather than spreading out, so even arbitrary token pairs tend to score high.

There is also [`model.similarity_pairwise()`](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html#sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.similarity_pairwise), for when you already have matched pairs and just want the pair scores instead of the full similarity matrix:

```python
scores = model.similarity_pairwise(query_embeddings, document_embeddings[:1])
print(scores)
# tensor([10.7942])
```

### Score Magnitude and MeanMaxSim

MaxSim sums over query tokens, so its magnitude scales with how many query tokens there are, which means you can't compare scores across models with different query recipes. LateOn encodes the Red Planet query above as 12 tokens. Run that same query and those same documents through ColBERTv2, which pads and truncates every query to exactly 32 tokens, and the scores land in a completely different range:

```python
model = MultiVectorEncoder("colbert-ir/colbertv2.0")
# ... same encode_query / encode_document / similarity calls ...
print(scores)
# tensor([[12.7970, 27.1945, 23.8495, 24.5656]])
```

Within one model the ordering is all you need, but if you want scores on a bounded scale, switch the model's similarity function to MeanMaxSim, which divides by the query token count. Back on LateOn:

```python
model = MultiVectorEncoder("lightonai/LateOn", similarity_fn_name="meanmaxsim")
# or on an already-loaded model: model.similarity_fn_name = "meanmaxsim"

print(model.similarity(query_embeddings, document_embeddings))
# tensor([[0.8995, 0.9259, 0.9145, 0.9234]])
```

Now every score is an average cosine similarity in `[-1, 1]`, although you'll only see `[0, 1]` in practice.

## Semantic Search

If your corpus is small, exhaustive MaxSim over all of it is the simplest thing that works. Encode the corpus once, then score each query against everything:

```python
import time

from datasets import load_dataset

from sentence_transformers import MultiVectorEncoder

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
# Several questions share an answer passage, so drop repeats but keep the order
corpus = list(dict.fromkeys(dataset["answer"]))  # 5,000 rows -> 4,874 passages

model = MultiVectorEncoder("lightonai/LateOn")
corpus_embeddings = model.encode_document(corpus, convert_to_tensor=True, show_progress_bar=True)

query = "when did richmond last play in a preliminary final"
start = time.perf_counter()
query_embeddings = model.encode_query([query], convert_to_tensor=True)
scores = model.similarity(query_embeddings, corpus_embeddings)[0]  # 98ms
top_scores, top_indices = scores.topk(3)
print(f"Search took {(time.perf_counter() - start) * 1000:.1f}ms")

for score, index in zip(top_scores.tolist(), top_indices.tolist()):
    print(f"{score:.4f}  {corpus[index][:100]}")
"""
Search took 122.7ms
11.9192  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieved
11.7591  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contest
11.6710  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fou
"""
```

Those 4,874 passages encoded in 20 seconds on an RTX 3090, and each search takes about 120ms end to end, most of that the MaxSim scoring against all 608,414 token vectors. This is exact, but it scales linearly in total corpus tokens and keeps every token vector in memory, so reach for it when you have a few thousand documents rather than a few million. The runnable version of this script is [semantic_search.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/applications/semantic_search.py).

Past that size you want a real late-interaction index, which Sentence Transformers doesn't ship. It doesn't need to: these indexes store whatever `encode_document` produced, so you encode here and hand the token embeddings to something built for them. [Indexing](#indexing) has working snippets for four of the options, and the section directly below covers how to skip the index entirely.

## Retrieve and Rerank

You can also get late-interaction quality without maintaining a late-interaction index, by using a multi-vector model as your *reranker*. A fast bi-encoder narrows a large corpus to a handful of candidates, then the multi-vector model rescores only those:

```python
from datasets import load_dataset

from sentence_transformers import MultiVectorEncoder, SentenceTransformer
from sentence_transformers.util import semantic_search

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:50000]")
corpus = list(dict.fromkeys(dataset["answer"]))

# First stage: index the corpus once with a fast bi-encoder
retriever = SentenceTransformer("jinaai/jina-embeddings-v5-text-nano-retrieval")
corpus_embeddings = retriever.encode_document(corpus, convert_to_tensor=True, show_progress_bar=True)

# Second stage: rescore the candidates with MaxSim
reranker = MultiVectorEncoder("perplexity-ai/pplx-embed-v1-late-0.6b", trust_remote_code=True)

# Retrieve the top 50
query = "when did richmond last play in a preliminary final"
hits = semantic_search(retriever.encode_query([query], convert_to_tensor=True), corpus_embeddings, top_k=50)[0]
candidates = [corpus[hit["corpus_id"]] for hit in hits]

# And rerank them with the multi-vector model
query_embeddings = reranker.encode_query([query])
document_embeddings = reranker.encode_document(candidates)
scores = reranker.similarity(query_embeddings, document_embeddings)[0]

for index in scores.argsort(descending=True)[:3].tolist():
    print(f"{scores[index].item():.4f}  {candidates[index][:100]}")
```

Only the 50 candidates are ever encoded as multi-vectors, so your index stays a normal dense index and the token vectors are transient. This is the same role a cross-encoder plays in a retrieve-and-rerank stack, but a multi-vector model is considerably cheaper per candidate. You encode the documents in one batch and score them with a matrix multiplication, instead of one forward pass per query-document pair. The runnable script is [retrieve_rerank.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/applications/retrieve_rerank.py), which prints the timings of both stages.

## Indexing

Several vector databases index and score multi-vectors natively: [Qdrant](https://qdrant.tech/documentation/concepts/vectors/) since v1.10, [Weaviate](https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings) since v1.29, [Vespa](https://blog.vespa.ai/announcing-long-context-colbert-in-vespa/) for years now, [LanceDB](https://docs.lancedb.com/search/multivector-search) since v0.15.0, and [VectorChord](https://docs.vectorchord.ai/vectorchord/usage/indexing-with-maxsim-operators.html), which adds a MaxSim operator to Postgres that plain pgvector doesn't have. [Milvus](https://milvus.io/docs/array-of-structs.md) joined them in v2.6.4, under array-of-structs rather than the unrelated feature it calls multi-vector search. If you would rather not run a server at all, LightOn's [fast-plaid](https://github.com/lightonai/fast-plaid) is a `pip install` away and implements PLAID directly, and [PyLate](https://github.com/lightonai/pylate) wraps it in a fuller retrieval stack.

A few others get you partway. [OpenSearch](https://docs.opensearch.org/latest/search-plugins/search-relevance/rerank-by-field-late-interaction/) and [Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/rank-vectors) can rescore candidates with MaxSim but not retrieve on it, and the Elasticsearch field is additionally in technical preview and Enterprise-tier. [turbopuffer](https://turbopuffer.com/docs/schema) has late-interaction indexing in private beta.

fast-plaid, Qdrant, Weaviate, and Vespa all take exactly what `encode_document` returns, so the code is the same up to the client library. Here's a working snippet for each, run against the 4,874 passages and 608,414 token vectors from the [Semantic Search](#semantic-search) example. Each one carries the ingestion and query times it produced on one machine (RTX 3090, i7-13700K), with no tuning beyond what the code shows, to give a sense of the shape of the work. All four answer the query faster than the 98ms `model.similarity` took in that section, and three of them do it on the CPU, since fast-plaid is the only one here using the GPU.

All four returned the same three passages in the same order as the exhaustive PyTorch MaxSim earlier in this post, and the three databases reproduce its scores to four decimals! That is because their snippets score every document, which is affordable at this size and removes approximation as a variable. fast-plaid is approximate by design, so its scores differ slightly. The notes under each one say what changes when you switch to an approximate index, which is where rankings start to drift.

<details>
<summary><b>fast-plaid</b></summary>

[fast-plaid](https://github.com/lightonai/fast-plaid) is LightOn's Rust implementation of PLAID, the index ColBERT was originally built around. There's no server to start, and it reads the tensors `encode_document` hands back without any conversion.

```python
# pip install sentence-transformers datasets fast-plaid
from datasets import load_dataset
from fast_plaid import search
from sentence_transformers import MultiVectorEncoder

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
corpus = list(dict.fromkeys(dataset["answer"]))
model = MultiVectorEncoder("lightonai/LateOn")
query = "when did richmond last play in a preliminary final"

document_embeddings = model.encode_document(corpus, batch_size=32, convert_to_tensor=True)
query_embedding = model.encode_query(query, convert_to_tensor=True)

fast_plaid = search.FastPlaid(index="natural-questions", device="cuda")

# 4,874 documents (608,414 token vectors) indexed in 5s
fast_plaid.create(documents_embeddings=document_embeddings)

results = fast_plaid.search(queries_embeddings=query_embedding.unsqueeze(0), top_k=3)  # 11ms

for index, score in results[0]:
    print(f"{score:.4f}  {corpus[index][:90]}")
"""
11.8828  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieve
11.7676  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contes
11.6758  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fo
"""
```

This is the only one of the four that is approximate, and it is the one place in this section where the scores do not match the exhaustive MaxSim. PLAID prunes with centroids and stores quantized residuals, so the three scores drift by a few hundredths in both directions against the 11.9192 / 11.7591 / 11.6710 computed earlier. The ranking is unaffected here, and that is the trade PLAID is making: it was designed for corpora far larger than this one, where scanning everything is not an option.

</details>

<details>
<summary><b>Qdrant</b></summary>

[Qdrant](https://qdrant.tech/documentation/concepts/vectors/) needs a server: `docker run -p 6333:6333 qdrant/qdrant`. The client also has a local mode (`QdrantClient(":memory:")`) that needs no server, but it's a pure-Python reimplementation, so use it for trying things out rather than for timing them.

```python
# pip install sentence-transformers datasets qdrant-client
from datasets import load_dataset
from qdrant_client import QdrantClient, models
from sentence_transformers import MultiVectorEncoder

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
corpus = list(dict.fromkeys(dataset["answer"]))
model = MultiVectorEncoder("lightonai/LateOn")
query = "when did richmond last play in a preliminary final"

document_embeddings = model.encode_document(corpus, batch_size=32)
query_embedding = model.encode_query(query)

client = QdrantClient("http://localhost:6333")
client.create_collection(
    collection_name="natural-questions",
    vectors_config=models.VectorParams(
        size=model.get_embedding_dimension(),
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        # MaxSim never walks the HNSW graph, so skip building one
        hnsw_config=models.HnswConfigDiff(m=0),
    ),
)

# 4,874 documents (608,414 token vectors) ingested in 26.3s
client.upload_points(
    collection_name="natural-questions",
    points=[
        models.PointStruct(id=idx, vector=embedding, payload={"text": text})
        for idx, (embedding, text) in enumerate(zip(document_embeddings, corpus))
    ],
    batch_size=64,
)

results = client.query_points(
    collection_name="natural-questions",
    query=query_embedding,
    limit=3,
    with_payload=True,
).points  # 18ms

for result in results:
    print(f"{result.score:.4f}  {result.payload['text'][:90]}")
"""
11.9192  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieve
11.7591  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contes
11.6710  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fo
"""
```

`MAX_SIM` is the only comparator Qdrant offers, and `hnsw_config=HnswConfigDiff(m=0)` is their recommendation for late-interaction fields, since the vectors are used for rescoring rather than graph traversal. Note that Qdrant themselves suggest reserving late interaction for reranking a few hundred candidates rather than scanning a whole collection, which is the [Retrieve and Rerank](#retrieve-and-rerank) pattern. At 4,874 documents the full scan costs 18ms and is exact, but that doesn't extrapolate.

</details>

<details>
<summary><b>Weaviate</b></summary>

[Weaviate](https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings) needs a server too: `docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.34.0`. Multi-vector support needs 1.29 or newer, and the embedded mode isn't available on Windows.

```python
# pip install sentence-transformers datasets weaviate-client
import weaviate
from datasets import load_dataset
from sentence_transformers import MultiVectorEncoder
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
corpus = list(dict.fromkeys(dataset["answer"]))
model = MultiVectorEncoder("lightonai/LateOn")
query = "when did richmond last play in a preliminary final"

document_embeddings = model.encode_document(corpus, batch_size=32)
query_embedding = model.encode_query(query)

client = weaviate.connect_to_local()
collection = client.collections.create(
    "Documents",
    # self_provided turns on MaxSim late interaction
    vector_config=[Configure.MultiVectors.self_provided(name="colbert")],
    properties=[Property(name="text", data_type=DataType.TEXT)],
)

# 4,874 documents (608,414 token vectors) ingested in 41s
with collection.batch.fixed_size(batch_size=64) as batch:
    for text, embedding in zip(corpus, document_embeddings):
        batch.add_object(properties={"text": text}, vector={"colbert": embedding.tolist()})

results = collection.query.near_vector(
    near_vector=query_embedding.tolist(),
    target_vector="colbert",
    limit=3,
    return_metadata=MetadataQuery(distance=True),
)  # 17ms

for result in results.objects:
    # Weaviate reports the MaxSim score as a negated distance
    print(f"{-result.metadata.distance:.4f}  {result.properties['text'][:90]}")
"""
11.9192  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieve
11.7591  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contes
11.6710  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fo
"""

client.close()
```

Defaults are enough here: Weaviate's dynamic `ef` resolves to 100 for a top-3 query, and this ranking is already exact from about 32 upward. That margin is a property of the embeddings rather than of Weaviate, so it's worth confirming on your own model instead of assuming the defaults hold.

Weaviate also supports MUVERA encoding, which made ingestion 3x faster and queries 1.8x faster in our test. It cost far more accuracy than that speed is worth at this size though: the correct third passage didn't appear even in its top 50.

</details>

<details>
<summary><b>Vespa</b></summary>

[Vespa](https://docs.vespa.ai/en/tensor-user-guide.html) also runs in a container, but `pyvespa` starts it for you, so there's no separate `docker run`.

```python
# pip install sentence-transformers datasets pyvespa
from datasets import load_dataset
from sentence_transformers import MultiVectorEncoder
from vespa.deployment import VespaDocker
from vespa.package import (
    ApplicationPackage, Document, Field, FirstPhaseRanking, Function, RankProfile, Schema,
)

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
corpus = list(dict.fromkeys(dataset["answer"]))
model = MultiVectorEncoder("lightonai/LateOn")
query = "when did richmond last play in a preliminary final"

document_embeddings = model.encode_document(corpus, batch_size=32)
query_embedding = model.encode_query(query)

# "dt" is a mapped dimension over the variable token count, "x" the dense 128-dim vector
package = ApplicationPackage(
    name="colbert",
    schema=[
        Schema(
            name="doc",
            document=Document(fields=[
                Field(name="text", type="string", indexing=["summary"]),
                Field(name="colbert", type="tensor<float>(dt{}, x[128])", indexing=["attribute"]),
            ]),
            rank_profiles=[
                RankProfile(
                    name="colbert",
                    inputs=[("query(qt)", "tensor<float>(qt{}, x[128])")],
                    functions=[Function(
                        name="max_sim",  # per query token take the best document token, then sum
                        expression="sum(reduce(sum(query(qt) * attribute(colbert), x), max, dt), qt)",
                    )],
                    first_phase=FirstPhaseRanking(expression="max_sim"),
                )
            ],
        )
    ],
)
app = VespaDocker(port=8080).deploy(application_package=package)  # ~40s to boot

# Vespa reads a mixed tensor as {token index: vector}, for documents and queries alike
def to_tensor(embedding):
    return {str(token): vector for token, vector in enumerate(embedding.tolist())}

# 4,874 documents (608,414 token vectors) ingested in ~80s
app.feed_iterable(
    ({"id": str(idx), "fields": {"text": text, "colbert": to_tensor(embedding)}}
     for idx, (text, embedding) in enumerate(zip(corpus, document_embeddings))),
    schema="doc",
)

response = app.query(body={
    "yql": "select text from doc where true",
    "ranking.profile": "colbert",
    "hits": 3,
    "input.query(qt)": to_tensor(query_embedding),
})  # ~75ms warm, ~115ms on the first call

for hit in response.hits:
    print(f"{hit['relevance']:.4f}  {hit['fields']['text'][:90]}")
"""
11.9192  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieve
11.7591  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contes
11.6710  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fo
"""
```

Vespa asks for the most upfront structure of the four, because you're declaring a ranking pipeline rather than just an index. In exchange you get to write MaxSim out as a tensor expression and see exactly what it computes. This version puts MaxSim in `first-phase` over `where true`, which scores all 4,874 documents and is why the output matches exhaustive MaxSim exactly. It's deliberately not what Vespa recommends at scale: their [ColBERT sample app](https://github.com/vespa-engine/sample-apps/tree/master/colbert) stores int8-binarized vectors and moves MaxSim into `second-phase` to rerank a cheaper first stage.

Moving to that phased setup needs care: `second-phase` rescores only the best 100 candidates by default, and here that window left two of the three correct passages unscored entirely. Raising `rerank-count` to cover your candidate set fixes that, though at this size the phased version still came out slower than simply scanning everything.

</details>

## Visual Document Retrieval

Late interaction is the state of the art for visual document retrieval: matching a text query against page *images*, with charts, tables, and layout intact, and no OCR step. This is what the [ColPali](https://arxiv.org/abs/2407.01449) family of models does, and those checkpoints load and run through the same API, with the `revision` pinning the open pull request that adds this one's Sentence Transformers configuration ([Supported Models](#visual-document-retrieval-models) has the full list). Image documents are passed as URLs, local paths, or PIL images:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("vidore/colqwen2-v1.0", revision="refs/pr/N", trust_remote_code=True)

queries = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
images = [
    "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc1.jpg",
    "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc2.jpg",
    "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc3.jpg",
    "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc4.jpg",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(images)
print(query_embeddings[0].shape, document_embeddings[0].shape)
# (25, 128) (755, 128)

scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[13.7065, 11.3266, 11.2454, 10.2928],
#         [ 7.2340, 15.9825,  6.8053,  6.3357]])
```

Each query retrieves its own page (the diagonal), and the second query separates much more cleanly than the first, since only one of the four pages is about outlay over time.

The code is unchanged. Underneath, the processor handles the visual prompt and the image patches, and MaxSim scores query text tokens against document image patches. A page holds many separate regions, which is exactly what makes late interaction a natural fit here, since a single vector would have to average a chart, a table, and three paragraphs into one summary. That fidelity costs index space, though. The shapes above are 755 token vectors for one page against 25 for the query, where a Natural Questions passage from earlier averaged about 125, so [token pooling](#token-pooling) is worth reaching for earlier here than it is for text.

Page images are the common case, but they're not the only non-text modality. Sentence Transformers accepts text, images, audio, and video, and a checkpoint supports whichever of those its processor does, which `model.modalities` reports. [vidore/colqwen-omni-v0.1](https://huggingface.co/vidore/colqwen-omni-v0.1) is built on Qwen2.5-Omni, for example, so it also embeds audio documents through the same `encode_document` call, passed as file paths, arrays, or `torchcodec` decoder objects. A single document can combine modalities too, by passing a dict like `{"text": ..., "image": ...}` in place of a bare value.

These are VLMs, so plan for the memory they need. [The table in Supported Models](#visual-document-retrieval-models) runs from 252M to 4.4B parameters, and the small end of it stays practical on CPU where the multi-billion ones don't.

## Interpretability

Because MaxSim is a sum of per-query-token maxima, a ranking decomposes exactly: every point of a document's score belongs to one query token and one document token. That lets you answer "why did this rank here?" precisely, rather than by eye.

For image documents, `sentence_transformers.multi_vector_encoder.interpretability` overlays that decomposition onto the page as the standard ColPali heatmap, either aggregated over the query or one map per query token. Asking "How much was spent on water resources and power?" against the outlays page from above, this is where the `water` token went:

![MaxSim heatmap of the query token "water" overlaid on a 1971 US budget outlays page, with the brightest patch on the "Water Resources & Power" bar of the lower chart](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/maxsim_heatmap.png)

[heatmap.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/interpretability/heatmap.py) is the runnable version, including the masking step that lines the document embedding up with the patch grid.

Text documents have no patch grid to overlay, but the same decomposition applies. [text_similarity_map.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/interpretability/text_similarity_map.py) ranks a corpus and then attributes the top hit's score token by token, here on the Natural Questions corpus from earlier with the 32M-parameter [mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m):

```
Query: when did richmond last play in a preliminary final
Top 3 of 4874 documents by exhaustive MaxSim (191.0ms):
  12.3489  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieved since 19
  12.1771  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contested betwee
  12.0591  2018 UEFA Champions League Final The 2018 UEFA Champions League Final was the final match of the 201

  query token       best document token      sim   share
  when              since                 0.9154    7.4%
  did               had                   0.9675    7.8%
  rich              rich                  0.9764    7.9%
  mond              mond                  0.9856    8.0%
  last              to                    0.9249    7.5%
  play              game                  0.9384    7.6%
  in                the                   0.9732    7.9%
  a                 a                     0.9587    7.8%
  preliminary       preliminary           0.9394    7.6%
  final             final                 0.9654    7.8%
  --------------------------------------------------------
  3 special tokens                        2.8038   22.7%
  MaxSim score                           12.3489  100.0%
```

`rich`, `mond`, `preliminary`, and `final` matched themselves, while `when` settled on `since` and `play` on `game`. The special tokens are worth noticing too: three of them contribute 22.7% of the score while carrying none of the query's content. Below this table the script prints the passage itself, with the winning tokens highlighted in place.

## Token Pooling

If the index footprint worries you, the most effective knob is to store fewer token vectors. `HierarchicalTokenPooling` implements the [token pooling](https://www.answer.ai/posts/colbert-pooling.html) technique from Answer.AI: it clusters each document's token vectors with Ward linkage on cosine similarity and replaces each cluster with its mean, keeping roughly `1 / pool_factor` of the tokens. Within one document a lot of token vectors end up close to each other, so much of what you drop is redundancy rather than signal:

```python
from datasets import load_dataset

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
documents = list(dict.fromkeys(dataset["answer"]))

model = MultiVectorEncoder("lightonai/LateOn")

pooling = HierarchicalTokenPooling(pool_factor=2)
document_embeddings = model.encode_document(documents, token_pooling=pooling)
```

There are three places to apply it, depending on when you want to pay for it:

```python
# 1. Per encode call, as above
document_embeddings = model.encode_document(documents, token_pooling=pooling)

# 2. Standalone, on embeddings you already have saved (e.g. list of [num_tokens, num_dims] tensors)
pooled = pooling.pool(document_embeddings)

# 3. Baked into the model, so every consumer of the checkpoint gets pooled documents
model.append(HierarchicalTokenPooling(pool_factor=2))
model.save_pretrained("my-pooled-colbert")
```

By default, pooling applies to documents only, since queries are short and are the side you can't afford to distort. On the Natural Questions corpus from earlier, the reduction tracks `pool_factor` closely, and pooling all 608k token vectors took about 6 seconds:

| `pool_factor` | Token vectors | Reduction | float32 index |
| :---: | ---: | :---: | ---: |
| 1 (off) | 608,414 | 1.00x | 311.5 MB |
| 2 | 305,438 | 1.99x | 156.4 MB |
| 3 | 204,407 | 2.98x | 104.7 MB |
| 4 | 153,936 | 3.95x | 78.8 MB |

A cluster mean is a worse match for a query token than the best of its members was, and the coarser the clusters, the more that shows. The [original experiments](https://www.answer.ai/posts/colbert-pooling.html) measured that cost on BEIR and found very little of it: 100.6% of the unpooled retrieval performance on average at `pool_factor=2`, and about 99% at `pool_factor=3`. Halving your index for free is a good deal, so 2 is a reasonable place to start. How much it costs on your data is corpus-specific though, so measure it with an [evaluator](#evaluating-a-model) before you settle on a factor. The runnable comparison is [token_pooling.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/compression/token_pooling.py).

How far you can push `pool_factor` is also partly a property of the model. LightOn's [hierarchical pooling regularization](https://huggingface.co/blog/lightonai/lateon-hpool-regularization) trains for exactly that, shaping the embedding space so pooling costs less and reporting 99.4% retention at 5x compression. Training with that regularizer isn't in Sentence Transformers yet, but the resulting checkpoints are ordinary PyLate models, so [`lightonai/LateOn-hpool-regularized`](https://huggingface.co/lightonai/LateOn-hpool-regularized) loads and pools like any other.

## Speeding Up Inference

Multi-vector models run through the same backend machinery as the rest of Sentence Transformers, so you get `torch` (default), `onnx`, and `openvino`, alongside half precision, Flash Attention, and `torch.compile`.

On GPU, fp16 with Flash Attention is the best configuration we measured, at 2.44x the throughput of fp32 with no measurable retrieval quality loss. Flash Attention helps multi-vector models more than most, because documents are only truncated and never padded to a shared length, so your batches have widely varying sequence lengths that unpadding can exploit:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder(
    "lightonai/GTE-ModernColBERT-v1",
    model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "float16"},
)
```

<div style="display: flex; flex-wrap: wrap; gap: 16px; justify-content: center;">
  <figure style="flex: 1 1 300px; min-width: 0; margin: 0; text-align: center;">
    <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/mve_backends_benchmark_gpu.png"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/mve_backends_benchmark_gpu.png" alt="Multi-vector backend benchmarks on GPU" style="width: 100%;" /></a>
    <figcaption>GPU</figcaption>
  </figure>
  <figure style="flex: 1 1 300px; min-width: 0; margin: 0; text-align: center;">
    <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/mve_backends_benchmark_cpu.png"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/mve_backends_benchmark_cpu.png" alt="Multi-vector backend benchmarks on CPU" style="width: 100%;" /></a>
    <figcaption>CPU</figcaption>
  </figure>
</div>

> [!WARNING]
> Models with non-attend query expansion (`attend=False`, which covers the Stanford-NLP checkpoints like `colbert-ir/colbertv2.0` and `answerdotai/answerai-colbert-small-v1`) reject Flash Attention at load time. Flash Attention strips `attention_mask=0` positions, so the `[MASK]` expansion tokens that MaxSim scores would never receive an attention update. Use `"sdpa"` for those models.

On CPU, OpenVINO is your better bet where the architecture is supported, and int8 quantization buys a further speedup at a cost of about 0.4% accuracy. See [Speeding up Inference](https://sbert.net/docs/multi_vector_encoder/usage/efficiency.html) for the full benchmark details, the export and quantization helpers, and a flowchart for picking a backend.

## Evaluating a Model

`MultiVectorNanoBEIREvaluator` runs the [NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6) suite of 13 small BEIR subsets with MaxSim scoring, and needs no data preparation on your side:

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator

model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")
evaluator = MultiVectorNanoBEIREvaluator(batch_size=16)
results = evaluator(model)
print(f"{evaluator.primary_metric}: {results[evaluator.primary_metric]:.4f}")
```

This also makes it easy to check the claim from the top of this post. [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) is a late-interaction finetune of [`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base), so running both over five NanoBEIR subsets compares multi-vector against dense at the same backbone and the same parameter count:

| Model | Params | MSMARCO | NFCorpus | NQ | FiQA2018 | SciFact | Mean |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GTE-ModernColBERT-v1** (multi-vector, 128d) | 149M | **0.7089** | **0.3958** | **0.7719** | 0.5664 | **0.8298** | **0.6546** |
| gte-modernbert-base (dense, 768d) | 149M | 0.6446 | 0.3474 | 0.7237 | **0.6198** | 0.8203 | 0.6312 |
| all-MiniLM-L6-v2 (dense, 384d) | 23M | 0.5540 | 0.3323 | 0.5904 | 0.4775 | 0.7265 | 0.5361 |

Late interaction wins on four of the five subsets and on the mean! FiQA2018 (financial question answering) goes the other way, which is the shape of the tradeoff you should expect: a solid gain in retrieval quality for the same model size, paid for in index footprint, rather than a universal win on every dataset.

Alongside NanoBEIR, `MultiVectorInformationRetrievalEvaluator`, `MultiVectorRerankingEvaluator`, `MultiVectorTripletEvaluator`, and `MultiVectorDistillationEvaluator` cover the usual evaluation setups on your own data. They're documented in the [Evaluation API Reference](https://sbert.net/docs/package_reference/multi_vector_encoder/evaluation.html).

## Coming from PyLate or colpali-engine

`MultiVectorEncoder` absorbs the modeling, inference, training, and evaluation of both libraries. Every PyLate checkpoint loads directly, and [Supported Models](#supported-models) lists the colpali-engine checkpoints along with the `revision` to pass where one is still needed. If you're migrating, these are the calls that change:

| PyLate | Sentence Transformers |
|---|---|
| `pylate.models.ColBERT(model_name_or_path=...)` | `MultiVectorEncoder(...)` |
| `model.encode(..., is_query=True)` | `model.encode_query(...)` |
| `model.encode(..., is_query=False)` | `model.encode_document(...)` |
| `pylate.scores.colbert_scores` | `model.similarity` |
| `pylate.indexes.PLAID` / `pylate.retrieve.ColBERT` | no equivalent, keep PyLate's PLAID or see [Indexing](#indexing) |

| colpali-engine | Sentence Transformers |
|---|---|
| `ColQwen2.from_pretrained(...)` + `ColQwen2Processor` | `MultiVectorEncoder(...)` |
| `processor.process_queries(...)` + `model(**batch)` | `model.encode_query(queries)` |
| `processor.process_images(...)` + `model(**batch)` | `model.encode_document(images)` |
| `processor.score_multi_vector(qs, ds)` | `model.similarity(query_embeddings, document_embeddings)` |
| `mask_non_image_embeddings=True` | `MultiVectorMask(keep_only_token_ids=[...])` |
| `HierarchicalTokenPooler` | `HierarchicalTokenPooling` |
| `colpali_engine.interpretability` | `sentence_transformers.multi_vector_encoder.interpretability` |

One difference worth calling out: on a **bare** (non-ColBERT) checkpoint, PyLate's `ColBERT("bert-base-uncased")` applies the classic recipe by default, while `MultiVectorEncoder("bert-base-uncased")` builds a plain stack and leaves the prefixes, query expansion, and skiplist as explicit choices. The training loss and evaluator equivalents, and the data-handling differences, are in the [Migration Guide](https://sbert.net/docs/migration_guide.html#migrating-from-pylate).

Note that save compatibility is one-way in every case: PyLate, Stanford-NLP ColBERT, and colpali-engine checkpoints all load into `MultiVectorEncoder`, but `MultiVectorEncoder.save_pretrained` output isn't loadable by any of them.

## Supported Models

The [`sentence-transformers` tag](https://huggingface.co/models?library=sentence-transformers&other=multi-vector) on the Hub is the list that stays current, and we're working to get it onto every model that works. The tables below are what we test against directly, so treat them as a starting point rather than the full set. For text retrieval in particular, any PyLate or Stanford-NLP ColBERT checkpoint loads whether or not it carries the tag yet.

### Text Retrieval Models

These load with their trained prefix tokens, query expansion, and punctuation skiplist recovered from the saved configuration. Where a `revision` is listed, pass it until the pull request on that repository is merged, after which the plain model name is enough.

| Model | Parameters | Backbone | Notes |
| --- | :---: | --- | --- |
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 149M | gte-modernbert-base | No `revision` needed |
| [lightonai/Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT) | 149M | ModernBERT-base | No `revision` needed |
| [lightonai/Agent-ModernColBERT](https://huggingface.co/lightonai/Agent-ModernColBERT) | 149M | ModernBERT-base | No `revision` needed |
| [lightonai/ColBERT-Zero](https://huggingface.co/lightonai/ColBERT-Zero) | 149M | ModernBERT-base | No `revision` needed |
| [lightonai/LateOn](https://huggingface.co/lightonai/LateOn) | 149M | ModernBERT-base | No `revision` needed |
| [lightonai/LateOn-hpool-regularized](https://huggingface.co/lightonai/LateOn-hpool-regularized) | 149M | ModernBERT-base | No `revision` needed |
| [lightonai/LateOn-Code](https://huggingface.co/lightonai/LateOn-Code) | 149M | ModernBERT-base (code) | No `revision` needed |
| [lightonai/LateOn-Code-edge](https://huggingface.co/lightonai/LateOn-Code-edge) | 17M | ModernBERT (17M, code, 48-dim output) | No `revision` needed |
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | 307M | ModernBERT (multilingual) | No `revision` needed |
| [LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | 353M | LFM2 (350M) | No `revision` needed |
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | 353M | LFM2.5 (350M) | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) | 32M | ModernBERT (32M) | No `revision` needed |
| [mixedbread-ai/mxbai-edge-colbert-v0-17m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m) | 17M | ModernBERT (17M) | No `revision` needed |
| [mixedbread-ai/mxbai-colbert-large-v1](https://huggingface.co/mixedbread-ai/mxbai-colbert-large-v1) | 335M | bert-large-uncased | `revision="refs/pr/N"` |
| [VAGOsolutions/SauerkrautLM-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-EuroColBERT) | 212M | EuroBERT-210m | No `revision` needed |
| [VAGOsolutions/SauerkrautLM-Reason-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Reason-EuroColBERT) | 212M | EuroBERT-210m | No `revision` needed |
| [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) | 559M | XLM-RoBERTa (multilingual) | No `revision` needed, needs `trust_remote_code=True` |
| [antoinelouis/colbert-xm](https://huggingface.co/antoinelouis/colbert-xm) | 853M | X-MOD (multilingual) | No `revision` needed |
| [yjoonjang/colbert-ko-v1](https://huggingface.co/yjoonjang/colbert-ko-v1) | 149M | ModernBERT (Korean) | No `revision` needed |
| [ytu-ce-cosmos/turkish-colbert](https://huggingface.co/ytu-ce-cosmos/turkish-colbert) | 111M | BERT (Turkish, 256-dim output) | No `revision` needed |
| [samheym/GerColBERT](https://huggingface.co/samheym/GerColBERT) | 110M | BERT (German) | No `revision` needed |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | 33M | BERT (33M) | No `revision` needed |
| [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | 110M | bert-base-uncased | No `revision` needed |
| [lightonai/colbertv2.0](https://huggingface.co/lightonai/colbertv2.0) | 110M | bert-base-uncased | No `revision` needed |
| [perplexity-ai/pplx-embed-v1-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-late-0.6b) | 596M | Qwen3-0.6B (bidirectional) | No `revision` needed, needs `trust_remote_code=True` |

### Visual Document Retrieval Models

ColPali-style models embed page images as documents and text as queries. Each one needs a small Sentence Transformers configuration in its repository, and most of those are open pull requests at the time of writing. Where a `revision` is listed, pass it until the pull request is merged, after which the plain model name is enough:

```python
model = MultiVectorEncoder("vidore/colqwen2.5-v0.2", revision="refs/pr/N", trust_remote_code=True)
```

| Model | Parameters | Backbone | Notes |
| --- | :---: | --- | --- |
| [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3) | 2.9B | PaliGemma-3B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2) | 2.9B | PaliGemma-3B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1) | 2.9B | PaliGemma-3B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colpali](https://huggingface.co/vidore/colpali) | 2.9B | PaliGemma-3B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) | 2.2B | Qwen2-VL-2B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1) | 2.2B | Qwen2-VL-2B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) | 3.8B | Qwen2.5-VL-3B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colqwen2.5-v0.1](https://huggingface.co/vidore/colqwen2.5-v0.1) | 3.8B | Qwen2.5-VL-3B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colsmolvlm-v0.1](https://huggingface.co/vidore/colsmolvlm-v0.1) | 2.1B | SmolVLM-Instruct | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colSmol-500M](https://huggingface.co/vidore/colSmol-500M) | 507M | SmolVLM-500M | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colSmol-256M](https://huggingface.co/vidore/colSmol-256M) | 256M | SmolVLM-256M | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colqwen-omni-v0.1](https://huggingface.co/vidore/colqwen-omni-v0.1) | 4.4B | Qwen2.5-Omni-3B (also audio) | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [ModernVBERT/colmodernvbert](https://huggingface.co/ModernVBERT/colmodernvbert) | 252M | ModernVBERT (250M) | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 4.4B | Qwen3-VL-4B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colpali-v1.3-hf](https://huggingface.co/vidore/colpali-v1.3-hf) | 2.9B | PaliGemma-3B | No `revision` needed |
| [vidore/colpali-v1.2-hf](https://huggingface.co/vidore/colpali-v1.2-hf) | 2.9B | PaliGemma-3B | No `revision` needed |
| [vidore/colqwen2-v1.0-hf](https://huggingface.co/vidore/colqwen2-v1.0-hf) | 2.2B | Qwen2-VL-2B | No `revision` needed |

Nearly all of these are LoRA adapter repositories, which is what `trust_remote_code=True` is for: they ship a small modeling file that merges the adapter onto its base at load time. Most of them also have a `-merged` sibling on the Hub (e.g. [vidore/colpali-v1.3-merged](https://huggingface.co/vidore/colpali-v1.3-merged)) with the adapter already folded into the weights, which loads without remote code.

The three `-hf` entries at the bottom are the transformers-native `*ForRetrieval` ports. They load without any configuration, since the projection and normalization live inside the model, but they're separate uploads that lag the originals, so prefer the checkpoint the authors publish and maintain.

## Additional Resources

### Documentation

- [Multi-Vector Encoder > Usage](https://sbert.net/docs/multi_vector_encoder/usage/usage.html)
- [Multi-Vector Encoder > Pretrained Models](https://sbert.net/docs/multi_vector_encoder/pretrained_models.html)
- [Multi-Vector Encoder > Creating Custom Models](https://sbert.net/docs/multi_vector_encoder/usage/custom_models.html)
- [Multi-Vector Encoder > Speeding up Inference](https://sbert.net/docs/multi_vector_encoder/usage/efficiency.html)
- [Multi-Vector Encoder > API Reference](https://sbert.net/docs/package_reference/multi_vector_encoder/index.html)
- [Installation](https://sbert.net/docs/installation.html)
- [Migration Guide](https://sbert.net/docs/migration_guide.html)

### Example Scripts

- [Semantic Search](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/applications/semantic_search.py)
- [Retrieve and Rerank](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/applications/retrieve_rerank.py)
- [Token Pooling](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/compression/token_pooling.py)
- [ColPali Heatmaps](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/interpretability/heatmap.py)
- [Text Similarity Maps](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/interpretability/text_similarity_map.py)
- [NanoBEIR Evaluation](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/evaluation/nano_beir.py)

### Training

To learn how to train or finetune these models on your own data:

<!--
See the companion blogpost: [Training and Finetuning Multi-Vector Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-multi-vector-encoder).
-->

- [Multi-Vector Encoder > Training Overview](https://sbert.net/docs/multi_vector_encoder/training_overview.html)
- [Multi-Vector Encoder > Loss Overview](https://sbert.net/docs/multi_vector_encoder/loss_overview.html)
- [Multi-Vector Encoder > Training Examples](https://sbert.net/docs/multi_vector_encoder/training/examples.html)

### Hugging Face Hub

- [Multi-vector models on the Hub](https://huggingface.co/models?library=sentence-transformers&other=multi-vector)
- [Sentence Transformers datasets on the Hub](https://huggingface.co/datasets?other=sentence-transformers)

### Companion Blogposts

<!--
- [Training and Finetuning Multi-Vector Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-multi-vector-encoder): the direct training companion to this post.
-->
- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the general training guide for text-only dense embedding models.
- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): Cross Encoder training, the other way to add a precise second stage.
- [Training and Finetuning Sparse Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sparse-encoder): SPLADE and other sparse encoders, which combine well with late interaction in hybrid search.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): single-vector multimodal models, the dense counterpart to ColPali-style retrieval.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): includes a Visual Document Retrieval walkthrough with single-vector models.
- [🪆 Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): shrink dense embeddings by dimension, the way token pooling shrinks multi-vector ones by count.