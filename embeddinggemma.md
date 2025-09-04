---
title: "Welcome EmbeddingGemma, Google's new efficient embedding model"
thumbnail: /blog/assets/embeddinggemma/thumbnail.png
authors:
- user: tomaarsen
- user: Xenova
- user: alvarobartt
- user: ariG23498
- user: pcuenq
- user: sergiopaniego
---

<style>
    .centered {
        display: block;
        margin: 0 auto;
        text-align: center;
    }

    .mobile { display: none; }
    .desktop { display: block; }

    @media (max-width: 768px) {
        .mobile { display: block; }
        .desktop { display: none; }
    }
</style>

# Welcome EmbeddingGemma, Google's new efficient embedding model

## TL;DR

Today, Google releases [EmbeddingGemma](https://huggingface.co/collections/google/embeddinggemma-68b9ae3a72a82f0562a80dc4), a state-of-the-art multilingual embedding model perfect for on-device use cases. Designed for speed and efficiency, the model features a compact size of **308M parameters** and a **2K context window**, unlocking new possibilities for mobile RAG pipelines, agents, and more. EmbeddingGemma is trained to support over **100 languages** and is the highest-ranking text-only multilingual embedding model under 500M on the Massive Text Embedding Benchmark (MTEB) at the time of writing.

## Table of Contents

- [Introduction](#introduction)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [Usage](#usage)
    - [Sentence Transformers](#sentence-transformers)
        - [Retrieval](#retrieval)
    - [LangChain](#langchain)
    - [LlamaIndex](#llamaindex)
    - [Haystack](#haystack)
    - [txtai](#txtai)
    - [Transformers.js](#transformersjs)
    - [Text Embeddings Inference](#text-embeddings-inference)
    - [ONNX Runtime](#onnx-runtime)
- [Finetuning](#finetuning)
    - [Full Finetuning Script](#full-finetuning-script)
    - [Training](#training)
    - [Finetuned Evaluation](#finetuned-evaluation)
- [Further Reading](#further-reading)

## Introduction

[Text embeddings](https://sbert.net/) have become the backbone of modern natural‑language applications, turning words, sentences, and documents into dense vectors that capture meaning, sentiment, and intent. These vectors enable fast similarity search, clustering, classification, and retrieval across massive corpora, powering everything from recommendation engines and semantic search to retrieval-augmented generation and code‑search tools. Embedding models that calculate these embeddings are widely used, with well [over 200 million monthly downloads on Hugging Face](https://huggingface.co/models?library=sentence-transformers&sort=downloads).

Building on this foundation, Google DeepMind’s **EmbeddingGemma** arrives as the newest, most capable small multilingual embedding model yet. With just 308M parameters, a 2k‑token context window, and support for over 100 languages, EmbeddingGemma delivers state‑of‑the‑art performance on the Massive Multilingual Text Embedding Benchmark (MMTEB) while staying under 200 MB of RAM when quantized.

The various design choices result in a very practical, open-source tool for computing high-quality multilingual embeddings on everyday devices.

In this blogpost, we describe the EmbeddingGemma architecture and training, and show you how to use the model with various frameworks like Sentence Transformers, LangChain, LlamaIndex, Haystack, txtai, Transformers.js, Text Embedding Inference, and ONNX.

Afterwards, we demonstrate how to finetune EmbeddingGemma on your domain for even stronger performance. In our example, we finetune EmbeddingGemma on the Medical Instruction and Retrieval Dataset (MIRIAD). The resulting model, [sentence-transformers/embeddinggemma-300m-medical](https://huggingface.co/sentence-transformers/embeddinggemma-300m-medical), achieves state-of-the-art performance on our task: retrieving passages of scientific medical papers in response to detailed medical questions. It even [outperforms models twice as big](#finetuned-evaluation) on this task.

## Architecture

EmbeddingGemma builds on the [Gemma3](https://huggingface.co/blog/gemma3) transformers backbone, but modified to use bi-directional attention instead of causal (one-way) attention. This means that earlier tokens in the sequence can attend to later tokens, effectively turning the architecture from a decoder into an encoder. Encoder models can outperform LLMs, which are decoders, on embedding tasks like retrieval ([Weller et al., 2025](https://arxiv.org/abs/2507.11412)). With this backbone, the model can process a sizable 2048 tokens at once, sufficient for typical retrieval inputs, especially given that larger inputs often result in information loss in the text embeddings.

Beyond the new Gemma3-based encoder backbone, which produces token embeddings, a mean pooling layer converts these token embeddings into text embeddings. Lastly, two dense layers transform the text embeddings into their final form, a 768-dimensional vector.

The EmbeddingGemma model has been trained with [Matryoshka Representation Learning (MRL)](https://huggingface.co/blog/matryoshka), allowing you to truncate the 768‑dimensional output to 512, 256, or 128 dimensions on demand. This results in faster downstream processing and lower memory and disk space utilization. See the [Sentence Transformers usage](#sentence-transformers) for a snippet showing how to perform this truncation.

The model has been trained using a carefully curated, multilingual corpus totalling approximately 320 billion tokens. The proprietary dataset is a blend of publicly available web text, code and technical documentation, and synthetic task‑specific examples. It has been filtered to avoid Child Sexual Abuse Material (CSAM), sensitive data, and low-quality or unsafe content.

## Evaluation

EmbeddingGemma was benchmarked on the MMTEB (Multilingual, v2) and MTEB (English, v2) suites, which span a wide range of tasks, domains, and languages. Despite its modest 308M‑parameter size, the model consistently beats comparable baselines while keeping a very small memory footprint.

<table>
    <tr>
        <th>MTEB (Multilingual, v2) Performance</th>
        <th>MTEB (English, v2) Performance</th>
    </tr>
    <tr>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/embeddinggemma/embeddinggemma-300m-mteb-multilingual.png">
        </td>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/embeddinggemma/embeddinggemma-300m-mteb-eng.png">
        </td>
    </tr>
</table>

The results will be listed on the official [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard). We exclude any model that has been trained on more than 20% of the MTEB data, to mitigate potential over‑fitting.

## Demo

<div class="desktop centered">
    <iframe src="https://webml-community-semantic-galaxy.static.hf.space/" width="100%" height="600"></iframe>
    <p class="centered">
        <em>
            The <a href="https://huggingface.co/spaces/webml-community/semantic-galaxy" target="_blank">demo</a> can
            also be experienced in full screen.
        </em>
    </p>
</div>

<div class="mobile centered">
    <video controls width="800">
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/embeddinggemma/semantic_galaxy.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <p class="centered">
        <em>
            Experience the <a href="https://huggingface.co/spaces/webml-community/semantic-galaxy"
                target="_blank">demo</a> yourself on a Desktop device.
        </em>
    </p>
</div>

## Usage

EmbeddingGemma is integrated with many popular tools, making it easy to incorporate into your existing workflows and applications. The model has been integrated in Sentence Transformers, and thus also in projects that use Sentence Transformers behind the scenes, such as LangChain, LlamaIndex, Haystack, and txtai. See the examples below to get started with your preferred framework.

For production deployments, you can use [Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/en/index) (TEI) to serve the model efficiently on various hardware configurations, and you can use [Transformers.js](https://huggingface.co/docs/transformers.js/index) for use in web applications.

Regardless of your framework choice, you should be mindful of the **prompts**. For embedding models, prompts are prepended to the input text to allow the model to distinguish between different tasks. EmbeddingGemma was trained with these prompt names and prompts, so they should also be included when using the model:

* `query`: `"task: search result | query: "`,
* `document`: `"title: none | text: "`,
* `BitextMining`: `"task: search result | query: "`,
* `Clustering`: `"task: clustering | query: "`,
* `Classification`: `"task: classification | query: "`,
* `InstructionRetrieval`: `"task: code retrieval | query: "`,
* `MultilabelClassification`: `"task: classification | query: "`,
* `PairClassification`: `"task: sentence similarity | query: "`,
* `Reranking`: `"task: search result | query: "`,
* `Retrieval-query`: `"task: search result | query: "`,
* `Retrieval-document`: `"title: none | text: "`,
* `STS`: `"task: sentence similarity | query: "`,
* `Summarization`: `"task: summarization | query: "`

In Sentence Transformers, the `query` and `document` prompts are used automatically when calling `model.encode_query` and `model.encode_document`, but for other frameworks you might have to: $
1) specify prompt names (e.g. "Reranking"),
2) specify prompt strings (e.g. "task: search result | query: "), or
3) manually prepend the prompts to your input text.

The following example scripts will demonstrate this with various frameworks.

### Sentence Transformers

You will need to install the following packages:

```shell
pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
pip install sentence-transformers>=5.0.0
```

#### Retrieval

Inference using Sentence Transformers is rather simple, see this example for semantic search:

```py
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (768,) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.3011, 0.6359, 0.4930, 0.4889]])

# Convert similarities to a ranking
ranking = similarities.argsort(descending=True)[0]
print(ranking)
# tensor([1, 2, 3, 0])
```

* [Sentence Transformers `encode_query` method documentation](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode_query)
* [Sentence Transformers `encode_document` method documentation](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode_document)
* [Sentence Transformers `similarity` method documentation](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity)

<details><summary>Click to see non-retrieval code</summary>

If you’re not looking to use this model for Information Retrieval, then you’re likely best off using the most general `encode` method together with the model prompt that best describes your downstream task out of these options:

* `BitextMining`: Find translated sentence pairs in two languages.
* `Clustering`: Find similar texts to group them together.
* `Classification`: Assign predefined labels to texts.
* `InstructionRetrieval`: Retrieve relevant code snippets based on natural language instructions.
* `MultilabelClassification`: Assign multiple labels to texts.
* `PairClassification`: Assign predefined labels to texts.
* `Reranking`: Reorder search results based on relevance.
* `Retrieval-query`: Retrieve documents based on a query.
* `Retrieval-document`: Retrieve documents based on their content.
* `STS`: Compute semantic textual similarity between texts.
* `Summarization`: Generate concise summaries of texts.

```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Let's inspect the configured prompts
print(model.prompts)
# {
#     "query": "task: search result | query: ",
#     "document": "title: none | text: ",
#     "BitextMining": "task: search result | query: ",
#     "Clustering": "task: clustering | query: ",
#     "Classification": "task: classification | query: ",
#     "InstructionRetrieval": "task: code retrieval | query: ",
#     "MultilabelClassification": "task: classification | query: ",
#     "PairClassification": "task: sentence similarity | query: ",
#     "Reranking": "task: search result | query: ",
#     "Retrieval-query": "task: search result | query: ",
#     "Retrieval-document": "title: none | text: ",
#     "STS": "task: sentence similarity | query: ",
#     "Summarization": "task: summarization | query: ",
# }

# Compute semantic textual similarity using texts, so let's use the STS prompt
texts = [
    "The weather is beautiful today.",
    "It's a lovely day outside.",
    "The stock market crashed yesterday.",
    "I enjoy programming with Python."
]
embeddings = model.encode(texts, prompt_name="STS")
print(embeddings.shape)
# (4, 768)

# Compute similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
"""
tensor([[1.0000, 0.9305, 0.4660, 0.4326],
        [0.9305, 1.0000, 0.4227, 0.4434],
        [0.4660, 0.4227, 1.0000, 0.2638],
        [0.4326, 0.4434, 0.2638, 1.0000]])
"""
```

* [Sentence Transformers `encode` method documentation](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode)
* [Sentence Transformers `similarity` method documentation](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity)

</details>

<details><summary>Click to see how to truncate embedding dimensionality for faster and cheaper search</summary>

Because `google/embeddinggemma-300m` was trained with MRL, the embeddings generated by this model can be truncated to lower dimensionalities without considerably hurting the evaluation performance. Embeddings with lower dimensionalities are both cheaper to store on disk and in memory, as well as faster for downstream tasks like retrieval, clustering, or classification.

In Sentence Transformers, you can set a lower dimensionality using the `truncate_dim` parameter on either the `SentenceTransformer` initialization or when calling `model.encode`/`model.encode_query`/`model.encode_document`:

```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m", truncate_dim=256)

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (256,) (4, 256)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.4016, 0.6715, 0.5283, 0.5261]])

# Convert similarities to a ranking
ranking = similarities.argsort(descending=True)[0]
print(ranking)
# tensor([1, 2, 3, 0])
```

Note that the ranking is preserved despite using 3x smaller embeddings compared to the full-sized embeddings.

* [Sentence Transformers Matryoshka Embeddings documentation](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html)

</details>

### LangChain

If you prefer, you can also use the LangChain `HuggingFaceEmbeddings`, which uses Sentence Transformers behind the scenes. Note that you'll have to tell LangChain to use the prompts called "query" and "document" for queries and documents, respectively. This example involves a simple information retrieval setup, but the same embedding model can be used in more complex scenarios.

You will need to install the following packages:

```
pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
pip install sentence-transformers
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install faiss-cpu
```

```py
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Download the model from the 🤗 Hub. Also specify to use the "query" and "document" prompts
# as defined in the model configuration, as LangChain doesn't automatically use them.
# See https://huggingface.co/google/embeddinggemma-300m/blob/main/config_sentence_transformers.json
embedder = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
    query_encode_kwargs={"prompt_name": "query"},
    encode_kwargs={"prompt_name": "document"}
)

data = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# Create documents for the vector store
documents = [Document(page_content=text, metadata={"id": i}) for i, text in enumerate(data)]

# Create vector store using FAISS. Setting distance_strategy to "MAX_INNER_PRODUCT" uses
# FAISS' FlatIndexIP behind the scenes, which is optimized for inner product search. This
# is what the model was trained for
vector_store = FAISS.from_documents(documents, embedder, distance_strategy="MAX_INNER_PRODUCT")

# Search for top 3 similar documents
query = "Which planet is known as the Red Planet?"
results = vector_store.similarity_search_with_score(query, k=3)

# Print results
for doc, score in results:
    print(f"Text: {doc.page_content} (score: {score:.4f})")
"""
Text: Mars, known for its reddish appearance, is often referred to as the Red Planet. (score: 0.6359)
Text: Jupiter, the largest planet in our solar system, has a prominent red spot. (score: 0.4930)
Text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet. (score: 0.4889)
"""
```

* [LangChain HuggingFaceEmbeddings documentation](https://python.langchain.com/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html)

### LlamaIndex

EmbeddingGemma is also supported in LlamaIndex as it uses Sentence Transformers under the hood. For the correct behaviour, you need to specify the query and document prompts as defined in the model configuration. Otherwise, your performance will be suboptimal. This script shows a rudimentary example of using EmbeddingGemma with LlamaIndex, but you can use the `HuggingFaceEmbedding` class in more difficult settings also.

You will need to install the following packages:

```
pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
pip install sentence-transformers
pip install llama-index
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-faiss
```

```py
import faiss
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# Download from the 🤗 Hub. Also specify the query and document prompts as
# defined in the model configuration, as LlamaIndex doesn't automatically load them.
# See https://huggingface.co/google/embeddinggemma-300m/blob/main/config_sentence_transformers.json
embeddings = HuggingFaceEmbedding(
    model_name="google/embeddinggemma-300m",
    query_instruction="task: search result | query: ",
    text_instruction="title: none | text: ",
)

data = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# Create a sample vector store
store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(768))
store.add([TextNode(id=i, text=text, embedding=embeddings.get_text_embedding(text)) for i, text in enumerate(data)])

# Search for top k similar documents
query = "Which planet is known as the Red Planet?"
query_embedding = embeddings.get_query_embedding(query)
results = store.query(VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=3))

# Print results
for idx, score in zip(results.ids, results.similarities):
    print(f"Text: {data[int(idx)]} (score: {score:.4f})")
"""
Text: Mars, known for its reddish appearance, is often referred to as the Red Planet. (score: 0.6359)
Text: Jupiter, the largest planet in our solar system, has a prominent red spot. (score: 0.4930)
Text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet. (score: 0.4889)
"""
```

* [LlamaIndex HuggingFaceEmbedding documentation](https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/)

### Haystack

EmbeddingGemma can also be used with Haystack, a framework for building production-ready search and language applications. Like LangChain and LlamaIndex, Haystack uses Sentence Transformers behind the scenes and requires you to specify the appropriate prompts. The following example shows how to set up a basic retrieval pipeline using EmbeddingGemma with Haystack.

You will need to install the following packages:

```
pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
pip install sentence-transformers
pip install haystack-ai
```

```py
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Initialize the document store
document_store = InMemoryDocumentStore()

# Initialize the document and query embedders
document_embedder = SentenceTransformersDocumentEmbedder(
    model="google/embeddinggemma-300m", encode_kwargs={"prompt_name": "document"}
)
query_embedder = SentenceTransformersTextEmbedder(
    model="google/embeddinggemma-300m", encode_kwargs={"prompt_name": "query"}
)
document_embedder.warm_up()
query_embedder.warm_up()

data = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

# Convert to Haystack documents and write to document store
documents = [Document(content=text, id=str(i)) for i, text in enumerate(data)]
documents_with_embeddings = document_embedder.run(documents=documents)["documents"]
document_store.write_documents(documents_with_embeddings)

# Create a query pipeline using a query embedder and compatible retriever
query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", query_embedder)
query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=3))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

# Search for top 3 similar documents
query = "Which planet is known as the Red Planet?"
results = query_pipeline.run({"text_embedder": {"text": query}})

# Print results
for document in results["retriever"]["documents"]:
    print(f"Text: {document.content} (score: {document.score:.4f})")
"""
Text: Mars, known for its reddish appearance, is often referred to as the Red Planet. (score: 0.6359)
Text: Jupiter, the largest planet in our solar system, has a prominent red spot. (score: 0.4930)
Text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet. (score: 0.4889)
"""
```

* [Haystack InMemoryEmbeddingRetriever documentation](https://docs.haystack.deepset.ai/docs/inmemoryembeddingretriever)

### txtai

txtai is also compatible with EmbeddingGemma. Like other frameworks, txtai utilizes Sentence Transformers under the hood and needs the appropriate prompts for optimal performance with EmbeddingGemma. The following example demonstrates how to set up a basic retrieval system with txtai.

You will need to install the following packages:

```
pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
pip install sentence-transformers
pip install txtai
```

```py
from txtai import Embeddings

# Download from the 🤗 Hub. Also specify the query and document prompts as
# defined in the model configuration, as txtai doesn't automatically load them.
# See https://huggingface.co/google/embeddinggemma-300m/blob/main/config_sentence_transformers.json
embeddings = Embeddings(
    path="google/embeddinggemma-300m",
    method="sentence-transformers",
    instructions={
        "query": "task: search result | query: ",
        "data": "title: none | text: ",
    }
)

data = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# Create a sample vector store
embeddings.index(data)

# Search for top k similar documents
query = "Which planet is known as the Red Planet?"
results = embeddings.search(query, 3)

# Print results
for idx, score in results:
    print(f"Text: {data[int(idx)]} (score: {score:.4f})")
"""
Text: Mars, known for its reddish appearance, is often referred to as the Red Planet. (score: 0.6359)
Text: Jupiter, the largest planet in our solar system, has a prominent red spot. (score: 0.4930)
Text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet. (score: 0.4889)
"""
```

* [Haystack InMemoryEmbeddingRetriever documentation](https://docs.haystack.deepset.ai/docs/inmemoryembeddingretriever)

### Transformers.js

You can even run EmbeddingGemma 100% locally in your browser with [Transformers.js](https://huggingface.co/docs/transformers.js/en/index)! If you haven't already, you can install the library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:

```shell
npm i @huggingface/transformers
```

You can then compute embeddings as follows:

```javascript
import { AutoModel, AutoTokenizer, matmul } from "@huggingface/transformers";

// Download from the 🤗 Hub
const model_id = "onnx-community/embeddinggemma-300m-ONNX";
const tokenizer = await AutoTokenizer.from_pretrained(model_id);
const model = await AutoModel.from_pretrained(model_id, {
  dtype: "fp32", // Options: "fp32" | "q8" | "q4"
});

// Run inference with queries and documents
const prefixes = {
  query: "task: search result | query: ",
  document: "title: none | text: ",
};
const query = prefixes.query + "Which planet is known as the Red Planet?";
const documents = [
  "Venus is often called Earth's twin because of its similar size and proximity.",
  "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
  "Jupiter, the largest planet in our solar system, has a prominent red spot.",
  "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
].map((x) => prefixes.document + x);

const inputs = await tokenizer([query, ...documents], { padding: true });
const { sentence_embedding } = await model(inputs);

// Compute similarities to determine a ranking
const scores = await matmul(sentence_embedding, sentence_embedding.transpose(1, 0));
const similarities = scores.tolist()[0].slice(1);
console.log(similarities);
// [ 0.30109718441963196, 0.6358831524848938, 0.4930494725704193, 0.48887503147125244 ]

// Convert similarities to a ranking
const ranking = similarities.map((score, index) => ({ index, score })).sort((a, b) => b.score - a.score);
console.log(ranking);
// [
//   { index: 1, score: 0.6358831524848938 },
//   { index: 2, score: 0.4930494725704193 },
//   { index: 3, score: 0.48887503147125244 },
//   { index: 0, score: 0.30109718441963196 }
// ]
```

### Text Embeddings Inference

You can easily deploy EmbeddingGemma for both development and production using Text Embeddings Inference (TEI) version [1.8.1](https://github.com/huggingface/text-embeddings-inference/releases/tag/v1.8.1) or later.	

- CPU:

```shell
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.1 --model-id google/embeddinggemma-300m --dtype float32
```

- CPU with ONNX Runtime:

```shell
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.1 --model-id onnx-community/embeddinggemma-300m-ONNX --dtype float32 --pooling mean
```

- NVIDIA CUDA:

```shell
docker run --gpus all --shm-size 1g -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cuda-1.8.1 --model-id google/embeddinggemma-300m --dtype float32
```

> [!TIP]
> If you run the Docker container with the `cuda-1.8.1` tag, it includes support for multiple GPU architectures: Turing, Ampere, Ada Lovelace, and Hopper. For a lighter image tailored to just your GPU, you can instead use a specific tag such as `turing-1.8.1`, `1.8.1` and `86-1.8.1` (Ampere), `89-1.8.1` (Ada Lovelace), or `hopper-1.8.1`.

Once deployed, regardless of the device or runtime, you can leverage the `/v1/embeddings` endpoint based on the [OpenAI Embeddings API Specification](https://platform.openai.com/docs/api-reference/embeddings/create) to generate embeddings.

```shell
curl http://0.0.0.0:8080/v1/embeddings -H "Content-Type: application/json" -d '{"model":"google/embeddinggemma-300m","input":["task: search result | query: Which planet is known as the Red Planet?","task: search result | query: Where did Amelia Earhart first fly?"]}'
```

Alternatively, you can also leverage the `/embed` endpoint from the [Text Embeddings Inference Embeddings API](https://huggingface.github.io/text-embeddings-inference/), which supports the `prompt_name` parameter, meaning there’s no need to manually prepend the prompt to the inputs but select it via `prompt_name` instead.

```shell
curl http://0.0.0.0:8080/embed -H "Content-Type: application/json" -d '{"inputs":["Which planet is known as the Red Planet?","Where did Amelia Earthart first fly?"],"prompt_name":"query","normalize":true}'
```

> [!TIP]
> Additionally, note that since `google/embeddinggemma-300m` was trained with [Matryoshka Representation Learning (MRL)](https://huggingface.co/blog/matryoshka), you can also leverage the `dimensions` parameter, on both `/v1/embeddings` and `/embed`, to truncate the embeddings to lower dimensionalities (512, 256, and 128) without hurting the evaluation performance.

### ONNX Runtime

You can also run the model directly with [ONNX Runtime](https://onnxruntime.ai/), making it highly portable and cross-platform compatible. The example below shows usage in Python, but the same approach can be applied in other languages (Java, C#, C++, etc.) as well.

```py
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from transformers import AutoTokenizer

# Download from the 🤗 Hub
model_id = "onnx-community/embeddinggemma-300m-ONNX"
model_path = hf_hub_download(model_id, subfolder="onnx", filename="model.onnx") # Download graph
hf_hub_download(model_id, subfolder="onnx", filename="model.onnx_data") # Download weights
session = ort.InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Run inference with queries and documents
prefixes = {
  "query": "task: search result | query: ",
  "document": "title: none | text: ",
}
query = prefixes["query"] + "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
documents = [prefixes["document"] + x for x in documents]

inputs = tokenizer([query] + documents, padding=True, return_tensors="np")

_, sentence_embedding = session.run(None, inputs.data)
print(sentence_embedding.shape)  # (5, 768)

# Compute similarities to determine a ranking
query_embeddings = sentence_embedding[0]
document_embeddings = sentence_embedding[1:]
similarities = query_embeddings @ document_embeddings.T
print(similarities)  # [0.30109745 0.635883 0.49304956 0.48887485]

# Convert similarities to a ranking
ranking = similarities.argsort()[::-1]
print(ranking)  # [1 2 3 0]
```

## Finetuning

As with all models compatible with the Sentence Transformers library, EmbeddingGemma can be easily fine-tuned on your specific dataset. To showcase this, we'll be finetuning `google/embeddinggemma-300m` on the [Medical Instruction and RetrIeval Dataset (MIRIAD)](https://huggingface.co/datasets/miriad/miriad-4.4M) dataset, such that our finetuned model becomes particularly adept at finding passages up to 1000 tokens from scientific medical papers given detailed medical questions. These passages can be used as crucial context for a generative model to answer questions more effectively.

Below, you can explore each key component of the finetuning process using expandable tabs. Each tab contains the relevant code and a detailed explanation.

<div>
<details open>
<summary style="font-size:1.2em; font-weight:bold;">Model</summary>

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData

model = SentenceTransformer(
    "google/embeddinggemma-300m",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="EmbeddingGemma-300m trained on the Medical Instruction and RetrIeval Dataset (MIRIAD)",
    ),
)
# SentenceTransformer(
#   (0): Transformer({'max_seq_length': 1024, 'do_lower_case': False, 'architecture': 'Gemma3TextModel'})
#   (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
#   (2): Dense({'in_features': 768, 'out_features': 3072, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
#   (3): Dense({'in_features': 3072, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
#   (4): Normalize()
# )
```

This code loads the EmbeddingGemma model from Hugging Face, with optional model card metadata for documentation and sharing. The `SentenceTransformer` class loads the model weights and configuration, while the `model_card_data` argument attaches metadata useful for inclusion in the automatically generated model card.

* Documentation: [Sentence Transformers > Training Overview > Model](https://sbert.net/docs/sentence_transformer/training_overview.html#model)

</details>

<details>
<summary style="font-size:1.2em; font-weight:bold;">Dataset</summary>

```python
from datasets import load_dataset

train_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="train").select(range(100_000))
eval_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="eval").select(range(1_000))
test_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="test").select(range(1_000))
# Dataset({
#     features: ['question', 'passage_text'],
#     num_rows: 100000
# })
# Dataset({
#     features: ['question', 'passage_text'],
#     num_rows: 1000
# })
# Dataset({
#     features: ['question', 'passage_text'],
#     num_rows: 1000
# })
```

This code loads the [MIRIAD dataset](https://huggingface.co/datasets/miriad/miriad-4.4M), or rather, a [copy](https://huggingface.co/datasets/tomaarsen/miriad-4.4M-split) that has been divided into train, eval, and test splits. Using a large, high-quality dataset ensures the model learns meaningful representations, while subsetting allows for faster experimentation. The `load_dataset` function fetches the dataset from Hugging Face Datasets, and the `.select()` method limits the number of samples for each split.

* Documentation: [Sentence Transformers > Training Overview > Dataset](https://sbert.net/docs/sentence_transformer/training_overview.html#dataset)

</details>

<details>
<summary style="font-size:1.2em; font-weight:bold;">Loss Function</summary>

```python
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
```

This code defines the loss function for training, using [Cached Multiple Negatives Ranking Loss (CMNRL)](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.losses.CachedMultipleNegativesRankingLoss). CMNRL is effective for retrieval tasks, as it uses in-batch negatives to efficiently train the model to distinguish between correct and incorrect pairs. The loss takes question-answer pairs and treats other answers in the batch as negatives, maximizing the distance between unrelated pairs in the embedding space. The `mini_batch_size` parameter controls the memory usage, but does not affect the training dynamics.

It's recommended to use this loss with a large `per_device_train_batch_size` in `SentenceTransformerTrainingArguments` and a low `mini_batch_size` in `CachedMultipleNegativesRankingLoss` for a strong training signal with low memory usage. Additionally, the [`NO_DUPLICATES` batch sampler](https://sbert.net/docs/package_reference/sentence_transformer/sampler.html#sentence_transformers.training_args.BatchSamplers) is recommended to avoid accidental false negatives.

* Documentation: [Sentence Transformers > Training Overview > Loss Function](https://sbert.net/docs/sentence_transformer/training_overview.html#loss-function)

</details>

<details>
<summary style="font-size:1.2em; font-weight:bold;">Training Arguments</summary>

```python
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainingArguments

run_name = "embeddinggemma-300m-medical-100k"
args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{run_name}",
    num_train_epochs=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if your GPU can't run FP16
    bf16=False,  # Set to True if your GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    prompts={
        "question": model.prompts["query"],
        "passage_text": model.prompts["document"],
    },
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=20,
    run_name=run_name,
)
```

This code sets up all hyperparameters and configuration for training, evaluation, and logging. Proper training arguments are crucial for efficient, stable, and reproducible training. The arguments control batch sizes, learning rate, mixed precision, evaluation and saving frequency, and more. Notably, the `prompts` dictionary maps dataset columns to prompts used by the model to distinguish queries from documents.

* Documentation: [Sentence Transformers > Training Overview > Training Arguments](https://sbert.net/docs/sentence_transformer/training_overview.html#training-arguments)

</details>

<details>
<summary style="font-size:1.2em; font-weight:bold;">Evaluator</summary>

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

queries = dict(enumerate(eval_dataset["question"]))
corpus = dict(enumerate(eval_dataset["passage_text"] + train_dataset["passage_text"][:30_000]))
relevant_docs = {idx: [idx] for idx in queries}
dev_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="miriad-eval-1kq-31kd",
    show_progress_bar=True,
)
dev_evaluator(model)
```

This code sets up an evaluator for information retrieval, using queries and a corpus to measure model performance. Evaluation during training helps monitor progress and avoid overfitting. The evaluator computes retrieval metrics (NDCG, MRR, Recall, Precision, MAP, etc.) by checking if the model retrieves the correct passages for each query. It can be run before, during, and after training, and the results will be logged and incorporated in the automatically generated model card.

Note that this snippet in particular uses all (1k) evaluation questions against a corpus of all (1k) evaluation passages and 30k training passages, for a total of 31k documents. Evaluating only against evaluation passages is too simple for the model.

* Documentation: [Sentence Transformers > Training Overview > Evaluator](https://sbert.net/docs/sentence_transformer/training_overview.html#evaluator)

</details>

<details>
<summary style="font-size:1.2em; font-weight:bold;">Trainer</summary>

```python
from sentence_transformers import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()
```

This code initializes and runs the training loop, coordinating all components.

* Documentation: [Sentence Transformers > Training Overview > Trainer](https://sbert.net/docs/sentence_transformer/training_overview.html#trainer)

</details>
</div>

### Full Finetuning Script

Below is the complete script, combining all components above:

```python
import logging
import traceback

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "google/embeddinggemma-300m",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="EmbeddingGemma-300m trained on the Medical Instruction and RetrIeval Dataset (MIRIAD)",
    ),
)

# 3. Load a dataset to finetune on
train_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="train").select(range(100_000))
eval_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="eval").select(range(1_000))
test_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="test").select(range(1_000))

# 4. Define a loss function. CachedMultipleNegativesRankingLoss (CMNRL) is a special variant of MNRL (a.k.a. InfoNCE),
# which take question-answer pairs (or triplets, etc.) as input. It will take answers from other questions in the batch
# as wrong answers, reducing the distance between the question and the true answer while increasing the distance to the
# wrong answers, in the embedding space.
# The (C)MNRL losses benefit from larger `per_device_train_batch_size` in the Training Arguments, as they can leverage
# more in-batch negative samples. At the same time, the `mini_batch_size` does not affect training performance, but it
# does limit the memory usage. A good trick is setting a high `per_device_train_batch_size` while keeping
# `mini_batch_size` small.
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)

# 5. (Optional) Specify training arguments
run_name = "embeddinggemma-300m-medical-100k"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    prompts={  # Map training column names to model prompts
        "question": model.prompts["query"],
        "passage_text": model.prompts["document"],
    },
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=20,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator using the evaluation queries and 31k answers & evaluate the base model
queries = dict(enumerate(eval_dataset["question"]))
corpus = dict(enumerate(eval_dataset["passage_text"] + train_dataset["passage_text"][:30_000]))
relevant_docs = {idx: [idx] for idx in queries}
dev_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="miriad-eval-1kq-31kd",  # 1k questions, 31k passages
    show_progress_bar=True,
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the evaluation set once more, this will also log the results
# and include them in the model card
dev_evaluator(model)

queries = dict(enumerate(test_dataset["question"]))
corpus = dict(enumerate(test_dataset["passage_text"] + train_dataset["passage_text"][:30_000]))
relevant_docs = {idx: [idx] for idx in queries}
test_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="miriad-test-1kq-31kd",  # 1k questions, 31k passages
    show_progress_bar=True,
)
test_evaluator(model)

# 8. Save the trained model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

# 9. (Optional) Push it to the Hugging Face Hub
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{run_name}')`."
    )
```


### Training

We ran the full training script on an RTX 3090 with 24GB of VRAM, and the completed training and evaluating scripts took 5.5 hours. If desired, you can further reduce the memory footprint by reducing `mini_batch_size` on the `CachedMultipleNegativesRankingLoss` and `batch_size` on the `InformationRetrievalEvaluator` instances. See here the logs from our training run:

| Epoch  | Step | Training Loss | Validation Loss | miriad-eval-1kq-31kd_cosine_ndcg@10 | miriad-test-1kq-31kd_cosine_ndcg@10 |
|:------:|:----:|:-------------:|:---------------:|:-----------------------------------:|:-----------------------------------:|
| -1     | -1   | -             | -               | 0.8474                              | 0.8340                              |
| 0.0256 | 20   | 0.1019        | -               | -                                   | -                                   |
| 0.0512 | 40   | 0.0444        | -               | -                                   | -                                   |
| 0.0767 | 60   | 0.0408        | -               | -                                   | -                                   |
| 0.1023 | 80   | 0.0462        | -               | -                                   | -                                   |
| 0.1279 | 100  | 0.0542        | 0.0525          | 0.8616                              | -                                   |
| 0.1535 | 120  | 0.0454        | -               | -                                   | -                                   |
| 0.1790 | 140  | 0.0403        | -               | -                                   | -                                   |
| 0.2046 | 160  | 0.0463        | -               | -                                   | -                                   |
| 0.2302 | 180  | 0.0508        | -               | -                                   | -                                   |
| 0.2558 | 200  | 0.0497        | 0.0449          | 0.8643                              | -                                   |
| 0.2813 | 220  | 0.0451        | -               | -                                   | -                                   |
| 0.3069 | 240  | 0.0445        | -               | -                                   | -                                   |
| 0.3325 | 260  | 0.0489        | -               | -                                   | -                                   |
| 0.3581 | 280  | 0.0452        | -               | -                                   | -                                   |
| 0.3836 | 300  | 0.0461        | 0.0406          | 0.8832                              | -                                   |
| 0.4092 | 320  | 0.0415        | -               | -                                   | -                                   |
| 0.4348 | 340  | 0.04          | -               | -                                   | -                                   |
| 0.4604 | 360  | 0.0399        | -               | -                                   | -                                   |
| 0.4859 | 380  | 0.0423        | -               | -                                   | -                                   |
| 0.5115 | 400  | 0.0352        | 0.0316          | 0.8823                              | -                                   |
| 0.5371 | 420  | 0.0408        | -               | -                                   | -                                   |
| 0.5627 | 440  | 0.0356        | -               | -                                   | -                                   |
| 0.5882 | 460  | 0.0371        | -               | -                                   | -                                   |
| 0.6138 | 480  | 0.0276        | -               | -                                   | -                                   |
| 0.6394 | 500  | 0.028         | 0.0280          | 0.8807                              | -                                   |
| 0.6650 | 520  | 0.0302        | -               | -                                   | -                                   |
| 0.6905 | 540  | 0.0345        | -               | -                                   | -                                   |
| 0.7161 | 560  | 0.0325        | -               | -                                   | -                                   |
| 0.7417 | 580  | 0.033         | -               | -                                   | -                                   |
| 0.7673 | 600  | 0.0314        | 0.0264          | 0.8910                              | -                                   |
| 0.7928 | 620  | 0.033         | -               | -                                   | -                                   |
| 0.8184 | 640  | 0.029         | -               | -                                   | -                                   |
| 0.8440 | 660  | 0.0396        | -               | -                                   | -                                   |
| 0.8696 | 680  | 0.0266        | -               | -                                   | -                                   |
| 0.8951 | 700  | 0.0262        | 0.0240          | 0.8968                              | -                                   |
| 0.9207 | 720  | 0.0262        | -               | -                                   | -                                   |
| 0.9463 | 740  | 0.0327        | -               | -                                   | -                                   |
| 0.9719 | 760  | 0.0293        | -               | -                                   | -                                   |
| 0.9974 | 780  | 0.0304        | -               | -                                   | -                                   |
| -1     | -1   | -             | -               | 0.9026                              | 0.8862                              |


### Finetuned Evaluation

The performance of the base model was already excellent, with a strong 0.8340 NDCG@10 on our MIRIAD test set. Despite that, we were able to increase it considerably on this domain-specific dataset.

| Model                                                                                                                                        | Number of Parameters | NDCG@10 on `miriad-test-1kq-31kd` |
|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------|-----------------------------------|
| [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5)                                                                      | 109M                 | 0.7541                            |
| [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)                                                    | 118M                 | 0.6852                            |
| [`ibm-granite/granite-embedding-125m-english`](https://huggingface.co/ibm-granite/granite-embedding-125m-english)                            | 125M                 | 0.7745                            |
| [`Snowflake/snowflake-arctic-embed-m-long`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long)                                  | 137M                 | 0.7514                            |
| [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)                                                      | 278M                 | 0.7052                            |
| [`Snowflake/snowflake-arctic-embed-m-v2.0`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)                                  | 305M                 | 0.8467                            |
| [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5)                                                                    | 335M                 | 0.7727                            |
| [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)                                            | 335M                 | 0.7851                            |
| [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)                                                    | 560M                 | 0.7318                            |
| [`Snowflake/snowflake-arctic-embed-l-v2.0`](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)                                  | 568M                 | 0.8433                            |
| [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)                                                              | 596M                 | 0.8493                            |
|                                                                                                                                              |                      |                                   |
| [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) (base)                                                     | 268M                 | 0.8340                            |
| [`sentence-transformers/embeddinggemma-300m-medical`](https://huggingface.co/sentence-transformers/embeddinggemma-300m-medical) (fine-tuned) | 268M                 | **0.8862**                        |

Our fine-tuning process achieved a significant improvement of +0.0522 NDCG@10 on the test set, resulting in a model that comfortably outperforms any existing general-purpose embedding model on our specific task, at this model size. Additional time and compute investment would allow for even stronger results, such as [hard negatives mining](https://sbert.net/docs/package_reference/util.html#sentence_transformers.util.mine_hard_negatives) or training with more than 100k data pairs.

## Further Reading

- [Google EmbeddingGemma blogpost](https://developers.googleblog.com/en/introducing-embeddinggemma/)
- [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- [Sentence Transformers documentation](https://sbert.net/)
- [Sentence Transformers > Training Overview documentation](https://sbert.net/docs/sentence_transformer/training_overview.html)
- [Transformers.js documentation](https://huggingface.co/docs/transformers.js/en/index)
- [Text Embeddings Inference (TEI) documentation](https://huggingface.co/docs/text-embeddings-inference/en/index)
