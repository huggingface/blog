---
title: "Multimodal Embedding & Reranker Models with Sentence Transformers"
thumbnail: /blog/assets/train-sentence-transformers/st-hf-thumbnail.png
authors:
- user: tomaarsen
---

# Multimodal Embedding & Reranker Models with Sentence Transformers

[Sentence Transformers](https://sbert.net/) is a Python library for using and training embedding and reranker models for applications like retrieval augmented generation, semantic search, and more. With the v5.4 update, you can now **encode and compare texts, images, audio, and videos** using the same familiar API. In this blogpost, I'll show you how to use these new multimodal capabilities for both embedding and reranking.

Multimodal embedding models map inputs from different modalities into a shared embedding space, while multimodal reranker models score the relevance of mixed-modality pairs. This opens up use cases like visual document retrieval, cross-modal search, and multimodal RAG pipelines.

## Table of Contents

* [What are Multimodal Models?](#what-are-multimodal-models)
* [Installation](#installation)
* [Multimodal Embedding Models](#multimodal-embedding-models)
    + [Loading a Model](#loading-a-model)
    + [Encoding Images](#encoding-images)
    + [Cross-Modal Similarity](#cross-modal-similarity)
    + [Encoding Queries and Documents](#encoding-queries-and-documents)
* [Multimodal Reranker Models](#multimodal-reranker-models)
    + [Ranking Mixed-Modality Documents](#ranking-mixed-modality-documents)
    + [Predicting Pair Scores](#predicting-pair-scores)
* [Retrieve and Rerank](#retrieve-and-rerank)
* [Input Formats and Configuration](#input-formats-and-configuration)
    + [Supported Input Types](#supported-input-types)
    + [Checking Modality Support](#checking-modality-support)
    + [Processor and Model kwargs](#processor-and-model-kwargs)
* [Supported Models](#supported-models)
* [Additional Resources](#additional-resources)

## What are Multimodal Models?

Traditional embedding models convert text into [fixed-size vectors](https://huggingface.co/blog/matryoshka#understanding-embeddings). Multimodal embedding models extend this by mapping inputs from different modalities (text, images, audio, or video) into a shared embedding space. This means you can compare a text query against image documents (or vice versa) using the same similarity functions you're already familiar with.

Similarly, traditional reranker (Cross Encoder) models compute relevance scores between pairs of texts. Multimodal rerankers can score pairs where one or both elements are images, combined text-image documents, or other modalities.

For example, you can compare a text query against image documents, find video clips matching a description, or build RAG pipelines that work across modalities.

## Installation

Multimodal models require some extra dependencies. Install the extras for the modalities you need (see [Installation](https://sbert.net/docs/installation.html) for more details):

```bash
# For image support
pip install -U "sentence-transformers[image]"

# For audio support
pip install -U "sentence-transformers[audio]"

# For video support
pip install -U "sentence-transformers[video]"

# Mix and match as needed
pip install -U "sentence-transformers[image,video,train]"
```

> [!NOTE]
> VLM-based models like Qwen3-VL-2B require a GPU with at least ~8 GB of VRAM. For the 8B variants, expect ~20 GB. If you don't have a local GPU, consider using a cloud GPU service or Google Colab. On CPU, these models will be extremely slow; text-only or CLIP models are better suited for CPU inference.

## Multimodal Embedding Models

### Loading a Model

Loading a multimodal embedding model works exactly like loading a text-only model:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", revision="refs/pr/23")
```

> [!NOTE]
> The `revision` argument is required for now because the integration pull requests for these models are still pending. Once they're merged, you'll be able to load them without specifying a revision.

The model automatically detects which modalities it supports, so there's nothing extra to configure. See [Processor and Model kwargs](#processor-and-model-kwargs) if you want to control things like image resolution or model precision.

### Encoding Images

With a multimodal model loaded, [`model.encode()`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode) accepts images alongside text. Images can be provided as URLs, local file paths, or PIL Image objects (see [Supported Input Types](#supported-input-types) for all accepted formats):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", revision="refs/pr/23")

# Encode images from URLs
img_embeddings = model.encode([
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
])
print(img_embeddings.shape)
# (2, 2048)
```

### Cross-Modal Similarity

You can compute similarities between text embeddings and image embeddings, since the model maps both into the same space:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", revision="refs/pr/23")

# Encode images
img_embeddings = model.encode([
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
])

# Encode text queries (one matching + one hard negative per image)
text_embeddings = model.encode([
    "A green car parked in front of a yellow building",
    "A red car driving on a highway",
    "A bee on a pink flower",
    "A wasp on a wooden table",
])

# Compute cross-modal similarities
similarities = model.similarity(text_embeddings, img_embeddings)
print(similarities)
# tensor([[0.5115, 0.1078],
#         [0.1999, 0.1108],
#         [0.1255, 0.6749],
#         [0.1283, 0.2704]])
```

As expected, "A green car parked in front of a yellow building" is most similar to the car image (0.51), and "A bee on a pink flower" is most similar to the bee image (0.67). The hard negatives ("A red car driving on a highway", "A wasp on a wooden table") correctly receive lower scores.

You might notice that even the best matching scores (0.51, 0.67) aren't very close to 1.0. This is due to the [modality gap](https://arxiv.org/abs/2203.02053): embeddings from different modalities tend to cluster in separate regions of the space. Cross-modal similarities are typically lower than within-modal ones (e.g., text-to-text), but the relative ordering is preserved, so retrieval still works well.

### Encoding Queries and Documents

For retrieval tasks, [`encode_query()`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode_query) and [`encode_document()`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode_document) are the recommended methods. Many retrieval models prepend different instruction prompts depending on whether the input is a query or a document, similar to how chat models might apply different system prompts depending on the goal. Model authors can specify their prompts in the model config, and `encode_query()` / `encode_document()` automatically load and apply the correct one:

- `encode_query()` uses the model's `"query"` prompt (if available) and sets `task="query"`.
- `encode_document()` uses the first available prompt from `"document"`, `"passage"`, or `"corpus"`, and sets `task="document"`.

Under the hood, both are thin wrappers around `encode()`, they just handle prompt selection for you. Here's what cross-modal retrieval looks like:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", revision="refs/pr/23")

# Encode text queries with the query prompt
query_embeddings = model.encode_query([
    "Find me a photo of a vehicle parked near a building",
    "Show me an image of a pollinating insect",
])

# Encode document screenshots with the document prompt
doc_embeddings = model.encode_document([
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
])

# Compute similarities
similarities = model.similarity(query_embeddings, doc_embeddings)
print(similarities)
# tensor([[0.3907, 0.1490],
#         [0.1235, 0.4872]])
```

These methods accept the same input types as `encode()` (images, URLs, multimodal dicts, etc.) and pass through the same parameters. For models without specialized query/document prompts, they behave identically to `encode()`.

## Multimodal Reranker Models

Multimodal reranker (CrossEncoder) models score the relevance between pairs of inputs, where each element can be text, an image, audio, video, or a combination. They tend to outperform embedding models in terms of quality, but are slower since they process each pair individually. The currently available pretrained multimodal rerankers focus on text and image inputs, but the architecture supports any modality that the underlying model can handle.

### Ranking Mixed-Modality Documents

The [`rank()`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.rank) method scores and ranks a list of documents against a query, supporting mixed modalities:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("Qwen/Qwen3-VL-Reranker-2B", revision="refs/pr/11")

query = "A green car parked in front of a yellow building"
documents = [
    # Image documents (URL or local file path)
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    # Text document
    "A vintage Volkswagen Beetle painted in bright green sits in a driveway.",
    # Combined text + image document
    {
        "text": "A car in a European city",
        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    },
]

rankings = model.rank(query, documents)
for rank in rankings:
    print(f"{rank['score']:.4f}\t(document {rank['corpus_id']})")
"""
0.9375  (document 0)
0.5000  (document 3)
-1.2500 (document 2)
-2.4375 (document 1)
"""
```

The reranker correctly identifies the car image (document 0) as the most relevant result, followed by the combined text+image document about a car in a European city (document 3). The bee image (document 1) scores lowest.
Keep in mind that the [modality gap](https://arxiv.org/abs/2203.02053) can influence absolute scores: text-image pair scores may occupy a different range than text-text or image-image pair scores.

You can also check which modalities a reranker supports using [`modalities`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.modalities) and [`supports()`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.supports), just like with embedding models:

```python
print(model.modalities)
# ['text', 'image', 'video', 'message']

print(model.supports("image"))
# True

# Check if the model supports a specific pair of modalities
print(model.supports(("image", "text")))
# True
```

### Predicting Pair Scores

You can also use [`predict()`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.predict) to get raw relevance scores for specific pairs of inputs:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("jinaai/jina-reranker-m0", trust_remote_code=True)

scores = model.predict([
    ("A green car", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"),
    ("A bee on a flower", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"),
    ("A green car", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"),
])
print(scores)
# [0.9389156  0.96922314 0.46063158]
```

## Retrieve and Rerank

A common pattern is to use an embedding model for fast initial retrieval, then refine the top results with a reranker:

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# Step 1: Retrieve with an embedding model
embedder = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", revision="refs/pr/23")

query = "revenue growth chart"
query_embedding = embedder.encode_query(query)

# Pre-compute corpus embeddings (do this once, then store them)
document_screenshots = [
    "path/to/doc1.png",
    "path/to/doc2.png",
    # ... potentially millions of document screenshots
]
corpus_embeddings = embedder.encode_document(document_screenshots, show_progress_bar=True)

# Simple cosine similarity retrieval, viable as long as embeddings fit in memory
similarities = embedder.similarity(query_embedding, corpus_embeddings)
top_k_indices = similarities.argsort(descending=True)[0][:10]

# Step 2: Rerank the top-k results with a reranker model
reranker = CrossEncoder("nvidia/llama-nemotron-rerank-vl-1b-v2", trust_remote_code=True)

top_k_documents = [document_screenshots[i] for i in top_k_indices]
rankings = reranker.rank(query, top_k_documents)
for rank in rankings:
    print(f"{rank['score']:.4f}\t{top_k_documents[rank['corpus_id']]}")
```

Since the corpus embeddings are pre-computed, the initial retrieval is fast even over millions of documents. The reranker then provides more accurate scoring over the smaller candidate set.

## Input Formats and Configuration

### Supported Input Types

Multimodal models accept a variety of input formats. Here's a summary of what you can pass to [`model.encode()`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode):

| Modality | Accepted Formats |
| --- | --- |
| **Text** | - Strings |
| **Image** | - `PIL.Image.Image` objects<br>- File paths (e.g. `"./photo.jpg"`)<br>- URLs (e.g. `"https://.../image.jpg"`)<br>- Numpy arrays, torch tensors |
| **Audio** | - File paths (e.g. `"./audio.wav"`)<br>- URLs (e.g. `"https://.../audio.wav"`)<br>- Numpy/torch arrays<br>- Dicts with `"array"` and `"sampling_rate"` keys<br>- `torchcodec.AudioDecoder` instances |
| **Video** | - File paths (e.g. `"./video.mp4"`)<br>- URLs (e.g. `"https://.../video.mp4"`)<br>- Numpy/torch arrays<br>- Dicts with `"array"` and `"video_metadata"` keys<br>- `torchcodec.VideoDecoder` instances |
| **Multimodal** | - Dicts mapping modality names to values,<br>e.g. `{"text": "a caption", "image": "https://.../image.jpg"}`<br>Valid keys: `"text"`, `"image"`, `"audio"`, `"video"` |
| **Message** | - List of message dicts with `"role"` and `"content"` keys,<br>e.g. `[{"role": "user", "content": [...]}]` |

### Checking Modality Support

You can check which modalities a model supports using the [`modalities`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.modalities) property and [`supports()`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.supports) method:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", revision="refs/pr/23")

# List all supported modalities
print(model.modalities)
# ['text', 'image', 'video', 'message']

# Check for a specific modality
print(model.supports("image"))
# True
print(model.supports("audio"))
# False
```

The `"message"` modality indicates that the model accepts chat-style message inputs with interleaved content. In practice, you rarely need to use this directly. When you pass strings, URLs, or multimodal dicts, the model converts them to the appropriate message format internally. Sentence Transformers supports two message formats:

- **Structured** (most VLMs, e.g. Qwen3-VL): Content is a list of typed dicts, e.g. `[{"type": "text", "text": "..."}, {"type": "image", "image": ...}]`
- **Flat** (e.g. Deepseek-V3): Content is a direct value, e.g. `"some text"`

The format is auto-detected from the model's chat template.

Since all inputs get converted into the same message format internally, you can mix input types in a single [`encode()`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode) call:

```python
embeddings = model.encode([
    # A text input
    "A green car parked in front of a yellow building",
    # An image input (URL)
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    # A combined text + image input
    {
        "text": "A car in a European city",
        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    },
])
```

<details>
<summary>Click here if you need to pass raw message inputs</summary>

If a model doesn't follow either format and you need full control, you can pass raw message dicts with `role` and `content` keys directly:

```python
embeddings = model.encode([
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"},
                {"type": "text", "text": "Describe this vehicle."},
            ],
        }
    ],
])
```

This bypasses the automatic format conversion and passes the messages directly to the processor's `apply_chat_template()`.

</details>

### Processor and Model kwargs

You may want to control image resolution bounds or model precision. Use `processor_kwargs` and `model_kwargs` when loading the model:

```python
model = SentenceTransformer(
    "Qwen/Qwen3-VL-Embedding-2B",
    model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"},
    processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 600 * 600},
    revision="refs/pr/23",
)
```

- `processor_kwargs` controls how inputs are preprocessed (e.g., image resolution bounds). Higher `max_pixels` means higher quality but more memory and compute. These are passed directly to `AutoProcessor.from_pretrained(...)`.
- `model_kwargs` controls how the underlying model is loaded (e.g., precision, attention implementation). These are passed directly to the appropriate `AutoModel.from_pretrained(...)` call (e.g., `AutoModel`, `AutoModelForCausalLM`, `AutoModelForSequenceClassification`, etc., depending on the configuration of the model modules).

See the [SentenceTransformer](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) API Reference documentation for more details on these kwargs.

> [!NOTE]
> In Sentence Transformers v5.4, `tokenizer_kwargs` has been renamed to `processor_kwargs` to reflect that multimodal models use processors rather than just tokenizers. The old name is still accepted but deprecated.

## Supported Models

Here are the multimodal models supported in v5.4, also available in the [v5.4 integrations collection](https://huggingface.co/collections/sentence-transformers/sentence-transformers-v54-integrations):

### Supported Multimodal Embedding Models

| Model | Parameters | Modalities | Revision | 
| --- | :---: | --- | --- |
| [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | 2B | Text, Image, Video | `revision="refs/pr/23"` |
| [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | 8B | Text, Image, Video | `revision="refs/pr/11"` |
| [nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) | 1.7B | Text, Image | No `revision` needed |
| [nvidia/omni-embed-nemotron-3b](https://huggingface.co/nvidia/omni-embed-nemotron-3b) | 4.7B | Text, Image | No `revision` needed |

### Supported Multimodal Reranker Models

| Model | Parameters | Modalities | Revision | 
| --- | :---: | --- | --- |
| [Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | 2B | Text, Image, Video | `revision="refs/pr/11"` |
| [Qwen/Qwen3-VL-Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) | 8B | Text, Image, Video | `revision="refs/pr/9"` |
| [nvidia/llama-nemotron-rerank-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) | 2B | Text, Image | No `revision` needed |
| [jinaai/jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0) | 2B | Text, Image | No `revision` needed |

### Text-Only Reranker Models (also new in v5.4)

| Model | Parameters | Revision |
| --- | :---: | --- |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 0.6B | `revision="refs/pr/24"` |
| [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | 4B | `revision="refs/pr/11"` |
| [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | 8B | `revision="refs/pr/11"` |
| [mixedbread-ai/mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | 0.5B | No `revision` needed |
| [mixedbread-ai/mxbai-rerank-large-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) | 2B | No `revision` needed |

<details>
<summary>Click here for a text-only reranker usage example</summary>

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v2")

query = "How do I bake sourdough bread?"
documents = [
    "Sourdough bread requires a starter made from flour and water, fermented over several days.",
    "The history of bread dates back to ancient Egypt around 8000 BCE.",
    "To bake sourdough, mix your starter with flour, water, and salt, then let it rise overnight.",
    "Rye bread is a popular alternative to wheat-based breads in Northern Europe.",
]

pairs = [(query, doc) for doc in documents]
scores = model.predict(pairs)
print(scores)
# [ 7.3077507 -2.6217823  8.724761  -2.2488995]

rankings = model.rank(query, documents)
for rank in rankings:
    print(f"{rank['score']:.4f}\t{documents[rank['corpus_id']]}")
# 8.7248  To bake sourdough, mix your starter with flour, water, and salt, then let it rise overnight.
# 7.3078  Sourdough bread requires a starter made from flour and water, fermented over several days.
# -2.2489 Rye bread is a popular alternative to wheat-based breads in Northern Europe.
# -2.6218 The history of bread dates back to ancient Egypt around 8000 BCE.
```

</details>

### CLIP Models

The older CLIP models continue to be supported:

| Model | ImageNet Zero-Shot Top-1 Accuracy | Notes |
| --- | :---: | --- |
| [sentence-transformers/clip-ViT-L-14](https://huggingface.co/sentence-transformers/clip-ViT-L-14) | 75.4 | |
| [sentence-transformers/clip-ViT-B-16](https://huggingface.co/sentence-transformers/clip-ViT-B-16) | 68.1 | |
| [sentence-transformers/clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32) | 63.3 | |
| [sentence-transformers/clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) | N/A | Multilingual text encoder, 50+ languages |

These simple CLIP models still work well on lower-resource hardware.

<details>
<summary>Click here for a CLIP usage example</summary>

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/clip-ViT-L-14")

images = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    "https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg"
]
texts = ["A green car", "A bee on a flower", "Some cats on a couch", "One cat sitting in the window"]

image_embeddings = model.encode(images)
text_embeddings = model.encode(texts)
print(image_embeddings.shape, text_embeddings.shape)
# (3, 768) (4, 768)

similarities = model.similarity(image_embeddings, text_embeddings)
print(similarities)
# tensor([[0.2208, 0.1042, 0.0617, 0.0907],  First image (car) is most similar to "A green car"
#         [0.1205, 0.2303, 0.0632, 0.0917],  Second image (bee) is most similar to "A bee on a flower"
#         [0.1107, 0.0196, 0.2425, 0.1162]]) Third image (multiple cats) is most similar to "Some cats on a couch"
```

</details>

## Additional Resources

### Documentation

- [Sentence Transformer > Usage](https://sbert.net/docs/sentence_transformer/usage/usage.html)
- [Sentence Transformer > Pretrained Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
- [Cross Encoder > Usage](https://sbert.net/docs/cross_encoder/usage/usage.html)
- [Cross Encoder > Pretrained Models](https://sbert.net/docs/cross_encoder/pretrained_models.html)
- [Installation](https://sbert.net/docs/installation.html)

### Training

<!-- To learn how to finetune these multimodal models on your own data, see the companion blogpost: [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers). -->

I'll be releasing a blogpost for training and finetuning multimodal models in the coming weeks, so stay tuned! In the meantime, you can experiment with inference on the pretrained models, or try to experiment with training using the training documentation:

- [Sentence Transformer > Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html)
- [Sentence Transformer > Training Examples](https://sbert.net/docs/sentence_transformer/training/examples.html)
- [Cross Encoder > Training Overview](https://sbert.net/docs/cross_encoder/training_overview.html)
- [Cross Encoder > Training Examples](https://sbert.net/docs/cross_encoder/training/examples.html)
- [Sparse Encoder > Training Overview](https://sbert.net/docs/sparse_encoder/training_overview.html)
- [Sparse Encoder > Training Examples](https://sbert.net/docs/sparse_encoder/training/examples.html)

### Hugging Face Hub

- [Sentence Transformers models on the Hub](https://huggingface.co/models?library=sentence-transformers)
- [Sentence Transformers datasets on the Hub](https://huggingface.co/datasets?other=sentence-transformers)
- [v5.4 Integrations Collection](https://huggingface.co/collections/sentence-transformers/sentence-transformers-v54-integrations)
