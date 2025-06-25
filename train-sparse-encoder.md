---
title: "Training and Finetuning Sparse Embedding Models with Sentence Transformers v5"
thumbnail: /blog/assets/train-sentence-transformers/st-hf-thumbnail.png
authors:
- user: tomaarsen
- user: arthurbresnu
---

# Training and Finetuning Sparse Embedding Models with Sentence Transformers v5

[Sentence Transformers](https://sbert.net/) is a Python library for using and training embedding and reranker models for a wide range of applications, such as retrieval augmented generation, semantic search, semantic textual similarity, paraphrase mining, and more. The last few major versions have introduced significant improvements to training:

* v3.0: (improved) Sentence Transformer (Dense Embedding) model training
* v4.0: (improved) Cross Encoder (Reranker) model training
* v5.0: (new) Sparse Embedding model training

In this blogpost, I'll show you how to use it to finetune a sparse encoder/embedding model and explain why you might want to do so. This results in [tomaarsen/inference-free-splade-distilbert-base-uncased-nq](https://huggingface.co/tomaarsen/inference-free-splade-distilbert-base-uncased-nq), a cheap model that works especially well in hybrid search or retrieve and rerank scenarios.

Finetuning sparse embedding models involves several components: the model, datasets, loss functions, training arguments, evaluators, and the trainer class. I'll have a look at each of these components, accompanied by practical examples of how they can be used for finetuning strong sparse embedding models.

In addition to training your own models, you can choose from a wide range of pretrained sparse encoders available on the [Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers&other=sparse). To help navigate this growing space, we’ve curated a [SPLADE Models collection](https://huggingface.co/collections/arthurbresnu/splade-models-6850b6f81b29d17efc0171de) highlighting some of the most relevant models.  
We list the most prominent ones along with their benchmark results in [Pretrained Models](https://sbert.net/docs/sparse_encoder/pretrained_models.html) in the documentation.


## Table of Contents

- [What are Sparse Embedding models?](#what-are-sparse-embedding-models)
  * [Query and Document Expansion](#query-and-document-expansion)
  * [Why Use Sparse Embedding Models?](#why-use-sparse-embedding-models)
- [Why Finetune?](#why-finetune)
- [Training Components](#training-components)
- [Model](#model)
  * [Splade](#splade)
  * [Inference-free Splade](#inference-free-splade)
  * [Contrastive Sparse Representation (CSR)](#contrastive-sparse-representation-csr)
  * [Architecture Picker Guide](#architecture-picker-guide)
- [Dataset](#dataset)
  * [Data on the Hugging Face Hub](#data-on-the-hugging-face-hub)
  * [Local Data (CSV, JSON, Parquet, Arrow, SQL)](#local-data-csv-json-parquet-arrow-sql)
  * [Local Data that requires pre-processing](#local-data-that-requires-pre-processing)
  * [Dataset Format](#dataset-format)
- [Loss Function](#loss-function)
- [Training Arguments](#training-arguments)
- [Evaluator](#evaluator)
  * [SparseNanoBEIREvaluator](#sparsenanobeirevaluator)
  * [SparseEmbeddingSimilarityEvaluator with STSb](#sparseembeddingsimilarityevaluator-with-stsb)
  * [SparseTripletEvaluator with AllNLI](#sparsetripletevaluator-with-allnli)
- [Trainer](#trainer)
  * [Callbacks](#callbacks)
  * [Multi-Dataset Training](#multi-dataset-training)
- [Evaluation](#evaluation)
- [Training Tips](#training-tips)
- [Vector Database Integration](#vector-database-integration)
- [Additional Resources](#additional-resources)
  * [Training Examples](#training-examples)
  * [Documentation](#documentation)

## What are Sparse Embedding models?

The more broader term "embedding models" refer to models that convert some input, usually text, into a vector representation (embedding) that captures the semantic meaning of the input. Unlike with the raw inputs, you can perform mathematical operations on these embeddings, resulting in similarity scores that can be used for various tasks, such as search, clustering, or classification.

With dense embedding models, i.e. the common variety, the embeddings are typically low-dimensional vectors (e.g., 384, 768, or 1024 dimensions) where most values are non-zero. Sparse embedding models, on the other hand, produce high-dimensional vectors (e.g., 30,000+ dimensions) where most values are zero. Usually, each active dimension (i.e. the dimension with a non-zero value) in a sparse embedding corresponds to a specific token in the model's vocabulary, allowing for interpretability.

Let's have a look at [naver/splade-v3](https://huggingface.co/naver/splade-v3), a state-of-the-art sparse embedding model, as an example:

```python
from sentence_transformers import SparseEncoder

# Download from the 🤗 Hub
model = SparseEncoder("naver/splade-v3")

# Run inference
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# (3, 30522)

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[   32.4323,     5.8528,     0.0258],
#         [    5.8528,    26.6649,     0.0302],
#         [    0.0258,     0.0302,    24.0839]])

# Let's decode our embeddings to be able to interpret them
decoded = model.decode(embeddings, top_k=10)
for decoded, sentence in zip(decoded, sentences):
    print(f"Sentence: {sentence}")
    print(f"Decoded: {decoded}")
    print()
```

```
Sentence: The weather is lovely today.
Decoded: [('weather', 2.754288673400879), ('today', 2.610959529876709), ('lovely', 2.431990623474121), ('currently', 1.5520408153533936), ('beautiful', 1.5046082735061646), ('cool', 1.4664798974990845), ('pretty', 0.8986214995384216), ('yesterday', 0.8603134155273438), ('nice', 0.8322536945343018), ('summer', 0.7702118158340454)]

Sentence: It's so sunny outside!
Decoded: [('outside', 2.6939032077789307), ('sunny', 2.535827398300171), ('so', 2.0600898265838623), ('out', 1.5397940874099731), ('weather', 1.1198079586029053), ('very', 0.9873268604278564), ('cool', 0.9406591057777405), ('it', 0.9026399254798889), ('summer', 0.684999406337738), ('sun', 0.6520509123802185)]

Sentence: He drove to the stadium.
Decoded: [('stadium', 2.7872302532196045), ('drove', 1.8208855390548706), ('driving', 1.6665740013122559), ('drive', 1.5565159320831299), ('he', 1.4721972942352295), ('stadiums', 1.449463129043579), ('to', 1.0441515445709229), ('car', 0.7002660632133484), ('visit', 0.5118278861045837), ('football', 0.502326250076294)]
```

In this example, the embeddings are 30,522-dimensional vectors, where each dimension corresponds to a token in the model's vocabulary. The `decode` method returned the top 10 tokens with the highest values in the embedding, allowing us to interpret which tokens contribute most to the embedding. 

We can even determine the intersection or overlap between embeddings, very useful for determining why two texts are deemed similar or dissimilar:

```python
# Let's also compute the intersection/overlap of the first two embeddings
intersection_embedding = model.intersection(embeddings[0], embeddings[1])
decoded_intersection = model.decode(intersection_embedding)
print(decoded_intersection)
```
```
Decoded: [('weather', 3.0842742919921875), ('cool', 1.379457712173462), ('summer', 0.5275946259498596), ('comfort', 0.3239051103591919), ('sally', 0.22571465373039246), ('julian', 0.14787325263023376), ('nature', 0.08582140505313873), ('beauty', 0.0588383711874485), ('mood', 0.018594780936837196), ('nathan', 0.000752730411477387)]
```

### Query and Document Expansion

A key component of neural sparse embedding models is **query/document expansion**. Unlike traditional lexical methods like BM25, which only match exact tokens, neural sparse models automatically expand the original text with semantically related terms:

- **Traditional, Lexical (e.g. BM25):** Only matches on exact tokens in the text
- **Neural Sparse Models:** Automatically expand with related terms

For example, in the code output above, the sentence "The weather is lovely today" is expanded to include terms like "beautiful", "cool", "pretty", and "nice" which weren't in the original text. Similarly, "It's so sunny outside!" is expanded to include "weather", "summer", and "sun".

This expansion allows neural sparse models to match semantically related content or synonyms even without exact token matches, handle misspellings, and overcome vocabulary mismatch problems. This is why neural sparse models like SPLADE often outperform traditional lexical search methods while maintaining the efficiency benefits of sparse representations.

However, expansion has its risks. For example, query expansion for "What is the weather on Tuesday?" will likely also expand to "monday", "wednesday", etc., which may not be desired.

### Why Use Sparse Embedding Models?

In short, neural sparse embedding models fall in a valuable niche between traditional lexical methods like BM25 and dense embedding models like Sentence Transformers. They have the following advantages:

- **Hybrid potential:** Very effectively combined with dense models, which may struggle with searches where lexical matches are important
- **Interpretability:** You can see exactly which tokens contribute to a match
- **Performance:** Competitive or better than dense models in many retrieval tasks

Throughout this blogpost, I'll use "sparse embedding model" and "sparse encoder model" interchangeably.

## Why Finetune?

The majority of (neural) sparse embedding models employ the aforementioned query/document expansion so that you can match texts with nearly identical meaning, even if they don't share any words. In short, the model has to recognize synonyms so those tokens can be placed in the final embedding.

Most out-of-the-box sparse embedding models will easily recognize that "supermarket", "food", and "market" are useful expansions of a text containing "grocery", but for example:

- "The patient complained of severe cephalalgia."

expands to:

```
'##lal', 'severe', '##pha', 'ce', '##gia', 'patient', 'complaint', 'patients', 'complained', 'warning', 'suffered', 'had', 'disease', 'complain', 'diagnosis', 'syndrome', 'mild', 'pain', 'hospital', 'injury'
```

whereas we wish for it to expand to "headache", the common word for "cephalalgia". This example expands to many domains, e.g. not recognizing that "Java" is a programming language, that "Audi" makes cars, or that "NVIDIA" is a company that makes graphics cards.

Through finetuning, the model can learn to focus exclusively on the domain and/or language that matters to you.

## Training Components

Training Sentence Transformer models involves the following components:

1. [**Model**](#model): The model to train or finetune, which can be a pre-trained Sparse Encoder model or a base model.
2. [**Dataset**](#dataset): The data used for training and evaluation.
3. [**Loss Function**](#loss-function): A function that quantifies the model's performance and guides the optimization process.
4. [**Training Arguments**](#training-arguments) (optional): Parameters that influence training performance and tracking/debugging.
5. [**Evaluator**](#evaluator) (optional): A tool for evaluating the model before, during, or after training.
6. [**Trainer**](#trainer): Brings together the model, dataset, loss function, and other components for training.

Now, let's dive into each of these components in more detail.

## Model

Sparse Encoder models consist of a sequence of [Modules](https://sbert.net/docs/package_reference/sentence_transformer/models.html), [Sparse Encoder specific Modules](https://sbert.net/docs/package_reference/sparse_encoder/models.html) or [Custom Modules](https://sbert.net/docs/sentence_transformer/usage/custom_models.html#advanced-custom-modules), allowing for a lot of flexibility. If you want to further finetune a Sparse Encoder model (e.g. it has a [modules.json file](https://huggingface.co/naver/splade-cocondenser-ensembledistil/tree/main/modules.json)), then you don't have to worry about which modules are used:

```python
from sentence_transformers import SparseEncoder

model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
```

But if instead you want to train from another checkpoint, or from scratch, then these are the most common architectures you can use:

### Splade

Splade models use the [`MLMTransformer`](https://sbert.net/docs/package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.MLMTransformer) followed by a [`SpladePooling`](https://sbert.net/docs/package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SpladePooling) modules. The former loads a pretrained [Masked Language Modeling transformer model](https://huggingface.co/models?pipeline_tag=fill-mask) (e.g. [BERT](https://huggingface.co/google-bert/bert-base-uncased), [RoBERTa](https://huggingface.co/FacebookAI/roberta-base), [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased), [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base), etc.) and the latter pools the output of the MLMHead to produce a single sparse embedding of the size of the vocabulary.

```python
from sentence_transformers import models, SparseEncoder
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling

# Initialize MLM Transformer (use a fill-mask model)
mlm_transformer = MLMTransformer("google-bert/bert-base-uncased")

# Initialize SpladePooling module
splade_pooling = SpladePooling(pooling_strategy="max")

# Create the Splade model
model = SparseEncoder(modules=[mlm_transformer, splade_pooling])
```

This architecture is the default if you provide a fill-mask model architecture to SparseEncoder, so it's easier to use the shortcut:

```python
from sentence_transformers import SparseEncoder

model = SparseEncoder("google-bert/bert-base-uncased")
# SparseEncoder(
#   (0): MLMTransformer({'max_seq_length': 512, 'do_lower_case': False, 'arhitecture': 'BertForMaskedLM'})
#   (1): SpladePooling({'pooling_strategy': 'max', 'activation_function': 'relu', 'word_embedding_dimension': None})
# )
```

### Inference-free Splade

Inference-free Splade uses a [`Router`](https://sbert.net/docs/package_reference/sentence_transformer/models.html#sentence_transformers.models.Router) module with different modules for queries and documents. Usually for this type of architecture, the documents part is a traditional Splade architecture (a [`MLMTransformer`](https://sbert.net/docs/package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.MLMTransformer) followed by a [`SpladePooling`](https://sbert.net/docs/package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SpladePooling) module) and the query part is an [`SparseStaticEmbedding`](https://sbert.net/docs/package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SparseStaticEmbedding) module, which just returns a pre-computed score for every token in the query.

```python
from sentence_transformers import SparseEncoder
from sentence_transformers.models import Router
from sentence_transformers.sparse_encoder.models import SparseStaticEmbedding, MLMTransformer, SpladePooling

# Initialize MLM Transformer for document encoding
doc_encoder = MLMTransformer("google-bert/bert-base-uncased")

# Create a router model with different paths for queries and documents
router = Router.for_query_document(
    query_modules=[SparseStaticEmbedding(tokenizer=doc_encoder.tokenizer, frozen=False)],
    # Document path: full MLM transformer + pooling
    document_modules=[doc_encoder, SpladePooling("max")],
)

# Create the inference-free model
model = SparseEncoder(modules=[router], similarity_fn_name="dot")
# SparseEncoder(
#   (0): Router(
#     (query_0_SparseStaticEmbedding): SparseStaticEmbedding ({'frozen': False}, dim:30522, tokenizer: BertTokenizerFast)
#     (document_0_MLMTransformer): MLMTransformer({'max_seq_length': 512, 'do_lower_case': False, 'arhitecture': 'BertForMaskedLM'})
#     (document_1_SpladePooling): SpladePooling({'pooling_strategy': 'max', 'activation_function': 'relu', 'word_embedding_dimension': None})
#   )
# )
```

This architecture allows for fast query-time processing using the lightweight SparseStaticEmbedding approach, that can be trained and seen as a linear weights, while documents are processed with the full MLM transformer and SpladePooling.

> [!TIP]
> Inference-free Splade is particularly useful for search applications where query latency is critical, as it shifts the computational complexity to the document indexing phase which can be done offline.

> [!NOTE]
> When training models with the `Router` module, you must use the `router_mapping` argument in the `SparseEncoderTrainingArguments` to map the training dataset columns to the correct route ("query" or "document"). For example, if your dataset(s) have `["question", "answer"]` columns, then you can use the following mapping:
>
> ```python
> args = SparseEncoderTrainingArguments(
>     ...,
>     router_mapping={
>         "question": "query",
>         "answer": "document",
>     }
> )
> ```
>
> Additionally, it is recommended to use a much higher learning rate for the SparseStaticEmbedding module than for the rest of the model. For this, you should use the `learning_rate_mapping` argument in the `SparseEncoderTrainingArguments` to map parameter patterns to their learning rates. For example, if you want to use a learning rate of `1e-3` for the SparseStaticEmbedding module and `2e-5` for the rest of the model, you can do this:
>
> ```python
> args = SparseEncoderTrainingArguments(
>     ...,
>     learning_rate=2e-5,
>     learning_rate_mapping={
>         r"SparseStaticEmbedding\.*": 1e-3,
>     }
> )
> ```

### Contrastive Sparse Representation (CSR)

Contrastive Sparse Representation (CSR) models, introduced in [Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation](https://arxiv.org/pdf/2503.01776), apply a [`CSRSparsity`](https://sbert.net/docs/package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.CSRSparsity) module on top of a dense Sentence Transformer model, which usually consist of a [`Transformer`](https://sbert.net/docs/package_reference/sentence_transformer/models.html#sentence_transformers.models.Transformer) followed by a [`Pooling`](https://sbert.net/docs/package_reference/sentence_transformer/models.html#sentence_transformers.models.Pooling) module. You can initialize one from scratch like so:

```python
from sentence_transformers import models, SparseEncoder
from sentence_transformers.sparse_encoder.models import CSRSparsity

# Initialize transformer (can be any dense encoder model)
transformer = models.Transformer("google-bert/bert-base-uncased")

# Initialize pooling
pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")

# Initialize CSRSparsity module
csr_sparsity = CSRSparsity(
    input_dim=transformer.get_word_embedding_dimension(),
    hidden_dim=4 * transformer.get_word_embedding_dimension(),
    k=256,  # Number of top values to keep
    k_aux=512,  # Number of top values for auxiliary loss
)
# Create the CSR model
model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])
```

Or if your base model is 1) a dense Sentence Transformer model or 2) a non-MLM Transformer model (those are loaded as Splade models by default), then this shortcut will automatically initialize the CSR model for you:

```python
from sentence_transformers import SparseEncoder

model = SparseEncoder("mixedbread-ai/mxbai-embed-large-v1")
# SparseEncoder(
#   (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'arhitecture': 'BertModel'})
#   (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
#   (2): CSRSparsity({'input_dim': 1024, 'hidden_dim': 4096, 'k': 256, 'k_aux': 512, 'normalize': False, 'dead_threshold': 30})
# )
```

> [!WARNING]
> Unlike (Inference-free) Splade models, sparse embeddings by CSR models don't have the same size as the vocabulary of the base model. This means you can't directly interpret which words are activated in your embedding like you can with Splade models, where each dimension corresponds to a specific token in the vocabulary.
>
> Beyond that, CSR models are most effective on dense encoder models that use high-dimensional representations (e.g. 1024-4096 dimensions).

### Architecture Picker Guide

If you're unsure which architecture to use, here's a quick guide:

* Do you want to sparsify an existing Dense Embedding model? If yes, use [**CSR**](#contrastive-sparse-representation-csr).
* Do you want your query inference to be instantaneous at the cost of slight performance? If yes, use [**Inference-free SPLADE**](#inference-free-splade).
* Otherwise, use [**SPLADE**](#splade).

## Dataset

The [`SparseEncoderTrainer`](https://sbert.net/docs/package_reference/sparse_encoder/trainer.html#sentence_transformers.sparse_encoder.SparseEncoderTrainer) uses [`datasets.Dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) or [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict) instances for training and evaluation. You can load data from the Hugging Face Datasets Hub or use local data in various formats such as CSV, JSON, Parquet, Arrow, or SQL.

**Note:** Lots of public datasets that work out of the box with Sentence Transformers have been tagged with `sentence-transformers` on the Hugging Face Hub, so you can easily find them on [https://huggingface.co/datasets?other=sentence-transformers](https://huggingface.co/datasets?other=sentence-transformers). Consider browsing through these to find ready-to-go datasets that might be useful for your tasks, domains, or languages.

### Data on the Hugging Face Hub

You can use the [`load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset) function to load data from datasets in the [Hugging Face Hub](https://huggingface.co/datasets)

```python
from datasets import load_dataset

train_dataset = load_dataset("sentence-transformers/natural-questions", split="train")

print(train_dataset)
"""
Dataset({
    features: ['query', 'answer'],
    num_rows: 100231
})
"""
```

Some datasets, like [`nthakur/swim-ir-monolingual`](https://huggingface.co/datasets/nthakur/swim-ir-monolingual), have multiple subsets with different data formats. You need to specify the subset name along with the dataset name, e.g. `dataset = load_dataset("nthakur/swim-ir-monolingual", "de", split="train")`.

### Local Data (CSV, JSON, Parquet, Arrow, SQL)

You can also use [`load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset) for loading local data in certain file formats:

```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files="my_file.csv")
# or
dataset = load_dataset("json", data_files="my_file.json")
```

### Local Data that requires pre-processing

You can use [`datasets.Dataset.from_dict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.from_dict) if your local data requires pre-processing. This allows you to initialize your dataset with a dictionary of lists:

```python
from datasets import Dataset

queries = []
documents = []
# Open a file, perform preprocessing, filtering, cleaning, etc.
# and append to the lists

dataset = Dataset.from_dict({
    "query": queries,
    "document": documents,
})
```

Each key in the dictionary becomes a column in the resulting dataset.

### Dataset Format

It's crucial to ensure that your dataset format matches your chosen [loss function](#loss-function). This involves checking two things:

1. If your loss function requires a *Label* (as indicated in the [Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html) table), your dataset must have a column named **"label"** or **"score"**.
2. All columns other than **"label"** or **"score"** are considered *Inputs* (as indicated in the [Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html) table). The number of these columns must match the number of valid inputs for your chosen loss function. The names of the columns don't matter, **only their order matters**.

For example, if your loss function accepts `(anchor, positive, negative) triplets`, then your first, second, and third dataset columns correspond with `anchor`, `positive`, and `negative`, respectively. This means that your first and second column must contain texts that should embed closely, and that your first and third column must contain texts that should embed far apart. That is why depending on your loss function, your dataset column order matters.

Consider a dataset with columns `["text1", "text2", "label"]`, where the `"label"` column contains floating point similarity scores. This dataset can be used with `SparseCoSENTLoss`, `SparseAnglELoss`, and `SparseCosineSimilarityLoss` because:

1. The dataset has a "label" column, which is required by these loss functions.
2. The dataset has 2 non-label columns, matching the number of inputs required by these loss functions.

If the columns in your dataset are not ordered correctly, use [`Dataset.select_columns`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.select_columns) to reorder them. Additionally, remove any extraneous columns (e.g., `sample_id`, `metadata`, `source`, `type`) using [`Dataset.remove_columns`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.remove_columns), as they will be treated as inputs otherwise.

## Loss Function

Loss functions measure how well a model performs on a given batch of data and guide the optimization process. The choice of loss function depends on your available data and target task. Refer to the [Loss Overview](https://sbert.net/docs/sparse_encoder/loss_overview.html) for a comprehensive list of options.

> [!NOTE]
> To train a `SparseEncoder`, you either need a [`SpladeLoss`](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#sentence_transformers.sparse_encoder.losses.SpladeLoss) or [`CSRLoss`](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#sentence_transformers.sparse_encoder.losses.CSRLoss), depending on the architecture. These are wrapper losses that add sparsity regularization on top of a main loss function, which must be provided as a parameter. The only loss that can be used independently is `SparseMSELoss`, as it performs embedding-level distillation, ensuring sparsity by directly copying the teacher's sparse embedding.

Most loss functions can be initialized with just the `SparseEncoder` that you're training, alongside some optional parameters, e.g.:

```python
from datasets import load_dataset
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import SpladeLoss, SparseMultipleNegativesRankingLoss

# Load a model to train/finetune
model = SparseEncoder("distilbert/distilbert-base-uncased")

# Initialize the SpladeLoss with a SparseMultipleNegativesRankingLoss
# This loss requires pairs of related texts or triplets
loss = SpladeLoss(
    model=model,
    loss=SparseMultipleNegativesRankingLoss(model=model),
    query_regularizer_weight=5e-5,  # Weight for query loss
    corpus_regularizer_weight=3e-5,
) 

# Load an example training dataset that works with our loss function:
train_dataset = load_dataset("sentence-transformers/natural-questions", split="train")
print(train_dataset)
"""
Dataset({
    features: ['query', 'answer'],
    num_rows: 100231
})
"""
```

**Documentation**
- [`SpladeLoss`](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#sentence_transformers.sparse_encoder.losses.SpladeLoss)
- [`CSRLoss`](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#sentence_transformers.sparse_encoder.losses.CSRLoss)
- `sentence_transformers.sparse_encoder.losses.CSRLoss`
- [Losses API Reference](https://sbert.net/docs/package_reference/sparse_encoder/losses.html)

## Training Arguments

The [`SparseEncoderTrainingArguments`](https://sbert.net/docs/package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments) class allows you to specify parameters that influence training performance and tracking/debugging. While optional, experimenting with these arguments can help improve training efficiency and provide insights into the training process.

In the Sentence Transformers documentation, I've outlined some of the most useful training arguments. I would recommend reading it in [Training Overview > Training Arguments](https://sbert.net/docs/sparse_encoder/training_overview.html#training-arguments).

Here's an example of how to initialize [`SparseEncoderTrainingArguments`](https://sbert.net/docs/package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments):

```python
from sentence_transformers import SparseEncoderTrainingArguments

args = SparseEncoderTrainingArguments(
    # Required parameter:
    output_dir="models/splade-distilbert-base-uncased-nq",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if your GPU can't handle FP16
    bf16=False,  # Set to True if your GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="splade-distilbert-base-uncased-nq",  # Used in W&B if `wandb` is installed
)
```

Note that `eval_strategy` was introduced in `transformers` version `4.41.0`. Prior versions should use `evaluation_strategy` instead.

## Evaluator

You can provide the [`SparseEncoderTrainer`](https://sbert.net/docs/package_reference/sparse_encoder/trainer.html#sentence_transformers.sparse_encoder.SparseEncoderTrainer) with an `eval_dataset` to get the evaluation loss during training, but it may be useful to get more concrete metrics during training, too. For this, you can use evaluators to assess the model's performance with useful metrics before, during, or after training. You can use both an `eval_dataset` and an evaluator, one or the other, or neither. They evaluate based on the `eval_strategy` and `eval_steps` [Training Arguments](https://sbert.net/docs/sparse_encoder/training_overview.html#training-arguments).

Here are the implemented Evaluators that come with Sentence Transformers for Sparse Encoder models:

| Evaluator | Required Data |
|-----------|---------------|
| [`SparseBinaryClassificationEvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseBinaryClassificationEvaluator) | Pairs with class labels. |
| [`SparseEmbeddingSimilarityEvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator) | Pairs with similarity scores. |
| [`SparseInformationRetrievalEvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator) | Queries (qid => question), Corpus (cid => document), and relevant documents (qid => set[cid]). |
| [`SparseNanoBEIREvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator) | No data required. |
| [`SparseMSEEvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseMSEEvaluator) | Source sentences to embed with a teacher model and target sentences to embed with the student model. Can be the same texts. |
| [`SparseRerankingEvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseRerankingEvaluator) | List of `{'query': '...', 'positive': [...], 'negative': [...]}` dictionaries. |
| [`SparseTranslationEvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseTranslationEvaluator) | Pairs of sentences in two separate languages. |
| [`SparseTripletEvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator) | (anchor, positive, negative) pairs. |

Additionally, [`SequentialEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sequentialevaluator) should be used to combine multiple evaluators into one Evaluator that can be passed to the [`SparseEncoderTrainer`](https://sbert.net/docs/package_reference/sparse_encoder/trainer.html#sentence_transformers.sparse_encoder.SparseEncoderTrainer).

Sometimes you don't have the required evaluation data to prepare one of these evaluators on your own, but you still want to track how well the model performs on some common benchmarks. In that case, you can use these evaluators with data from Hugging Face.

### SparseNanoBEIREvaluator

**Documentation**
- [`sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator)

```python
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

# Initialize the evaluator. Unlike most other evaluators, this one loads the relevant datasets
# directly from Hugging Face, so there's no mandatory arguments
dev_evaluator = SparseNanoBEIREvaluator()
# You can run evaluation like so:
# results = dev_evaluator(model)
```

### SparseEmbeddingSimilarityEvaluator with STSb

**Documentation**
- [sentence-transformers/stsb](https://huggingface.co/datasets/sentence-transformers/stsb)
- [`sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator)
- [`sentence_transformers.SimilarityFunction`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SimilarityFunction)

```python
from datasets import load_dataset
from sentence_transformers.evaluation import SimilarityFunction
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator

# Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

# Initialize the evaluator
dev_evaluator = SparseEmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
# You can run evaluation like so:
# results = dev_evaluator(model)
```

### SparseTripletEvaluator with AllNLI

**Documentation**
- [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli)
- [`sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator`](https://sbert.net/docs/package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator)
- [`sentence_transformers.SimilarityFunction`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SimilarityFunction)

```python
from datasets import load_dataset
from sentence_transformers.evaluation import SimilarityFunction
from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator

# Load triplets from the AllNLI dataset (https://huggingface.co/datasets/sentence-transformers/all-nli)
max_samples = 1000
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split=f"dev[:{max_samples}]")

# Initialize the evaluator
dev_evaluator = SparseTripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    main_distance_function=SimilarityFunction.DOT,
    name="all-nli-dev",
)
# You can run evaluation like so:
# results = dev_evaluator(model)
```

> [!TIP]
> When evaluating frequently during training with a small `eval_steps`, consider using a tiny `eval_dataset` to minimize evaluation overhead. If you're concerned about the evaluation set size, a 90-1-9 train-eval-test split can provide a balance, reserving a reasonably sized test set for final evaluations. After training, you can assess your model's performance using `trainer.evaluate(test_dataset)` for test loss or initialize a testing evaluator with `test_evaluator(model)` for detailed test metrics.
>
> If you evaluate after training, but before saving the model, your automatically generated model card will still include the test results.

> [!WARNING]
> When using [Distributed Training](https://sbert.net/docs/sentence_transformer/training/distributed.html), the evaluator only runs on the first device, unlike the training and evaluation datasets, which are shared across all devices.

## Trainer

The [`SparseEncoderTrainer`](https://sbert.net/docs/package_reference/sparse_encoder/trainer.html#sentence_transformers.sparse_encoder.SparseEncoderTrainer) is where all previous components come together. We only have to specify the trainer with the model, training arguments (optional), training dataset, evaluation dataset (optional), loss function, evaluator (optional) and we can start training. Let’s have a look at a script where all of these components come together:

```python
import logging

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.models import Router
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.sparse_encoder.models import SparseStaticEmbedding, MLMTransformer, SpladePooling
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# 1. Load a model to finetune with 2. (Optional) model card data
mlm_transformer = MLMTransformer("distilbert/distilbert-base-uncased", tokenizer_args={"model_max_length": 512})
splade_pooling = SpladePooling(
    pooling_strategy="max", word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension()
)
router = Router.for_query_document(
    query_modules=[SparseStaticEmbedding(tokenizer=mlm_transformer.tokenizer, frozen=False)],
    document_modules=[mlm_transformer, splade_pooling],
)

model = SparseEncoder(
    modules=[router],
    model_card_data=SparseEncoderModelCardData(
        language="en",
        license="apache-2.0",
        model_name="Inference-free SPLADE distilbert-base-uncased trained on Natural-Questions tuples",
    ),
)

# 3. Load a dataset to finetune on
full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]
print(train_dataset)
print(train_dataset[0])

# 4. Define a loss function
loss = SpladeLoss(
    model=model,
    loss=SparseMultipleNegativesRankingLoss(model=model),
    query_regularizer_weight=0,
    corpus_regularizer_weight=3e-3,
)

# 5. (Optional) Specify training arguments
run_name = "inference-free-splade-distilbert-base-uncased-nq"
args = SparseEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    learning_rate_mapping={r"SparseStaticEmbedding\.weight": 1e-3},  # Set a higher learning rate for the SparseStaticEmbedding module
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    router_mapping={"query": "query", "answer": "document"},  # Map the column names to the routes
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=200,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = SparseNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=16)

# 7. Create a trainer & train
trainer = SparseEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 8. Evaluate the model performance again after training
dev_evaluator(model)

# 9. Save the trained model
model.save_pretrained(f"models/{run_name}/final")

# 10. (Optional) Push it to the Hugging Face Hub
model.push_to_hub(run_name)
```

In this example I'm finetuning from [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased), a base model that is not yet a Sparse Encoder model. This requires more training data than finetuning an existing Sparse Encoder model, like [`naver/splade-cocondenser-ensembledistil`](https://huggingface.co/naver/splade-cocondenser-ensembledistil).

After running this script, the [tomaarsen/inference-free-splade-distilbert-base-uncased-nq](https://huggingface.co/tomaarsen/inference-free-splade-distilbert-base-uncased-nq) model was uploaded for me. The model scores 0.5241 NDCG@10 on NanoMSMARCO, 0.3299 NDCG@10 on NanoNFCorpus and 0.5357 NDCG@10 NanoNQ, which is a good result for an inference-free distilbert-based model trained on just 100k pairs from the Natural Questions dataset. 

The model uses an average of 184 active dimensions in the sparse embeddings for the documents, compared to 7.7 active dimensions for the queries (i.e. the average number of tokens in the query). This corresponds to a sparsity of 99.39% and 99.97%, respectively.

All of this information is stored in the automatically generated model card, including the base model, language, license, evaluation results, training & evaluation dataset info, hyperparameters, training logs, and more. Without any effort, your uploaded models should contain all the information that your potential users would need to determine whether your model is suitable for them.

### Callbacks

The Sentence Transformers trainer supports various [`transformers.TrainerCallback`](https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback) subclasses, including:

- [`WandbCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.WandbCallback) for logging training metrics to W&B if `wandb` is installed
- [`TensorBoardCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.TensorBoardCallback) for logging training metrics to TensorBoard if `tensorboard` is accessible
- [`CodeCarbonCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.CodeCarbonCallback) for tracking carbon emissions during training if `codecarbon` is installed

These are automatically used without you having to specify anything, as long as the required dependency is installed.

Refer to the [Transformers Callbacks documentation](https://huggingface.co/docs/transformers/en/main_classes/callback) for more information on these callbacks and how to create your own.

### Multi-Dataset Training

Top-performing models are often trained using multiple datasets simultaneously. The [`SparseEncoderTrainer`](https://sbert.net/docs/package_reference/sparse_encoder/trainer.html#sentence_transformers.sparse_encoder.SparseEncoderTrainer) simplifies this process by allowing you to train with multiple datasets without converting them to the same format. You can even apply different loss functions to each dataset. Here are the steps for multi-dataset training:

1. Use a dictionary of [`datasets.Dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) instances (or a [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict)) as the `train_dataset` and `eval_dataset`.
2. (Optional) Use a dictionary of loss functions mapping dataset names to losses if you want to use different losses for different datasets.

Each training/evaluation batch will contain samples from only one of the datasets. The order in which batches are sampled from the multiple datasets is determined by the [`MultiDatasetBatchSamplers`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.MultiDatasetBatchSamplers) enum, which can be passed to the [`SparseEncoderTrainingArguments`](https://sbert.net/docs/package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments) via `multi_dataset_batch_sampler`. The valid options are:

- `MultiDatasetBatchSamplers.ROUND_ROBIN`: Samples from each dataset in a round-robin fashion until one is exhausted. This strategy may not use all samples from each dataset, but it ensures equal sampling from each dataset.
- `MultiDatasetBatchSamplers.PROPORTIONAL` (default): Samples from each dataset proportionally to its size. This strategy ensures that all samples from each dataset are used, and larger datasets are sampled from more frequently.

## Evaluation

Let's evaluate our newly trained inference-free SPLADE model using the NanoMSMARCO dataset, and see how it compares to dense retrieval approaches. We'll also explore hybrid retrieval methods that combine sparse and dense vectors, as well as reranking to further improve search quality.

After running a slightly modified version of our [hybrid_search.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/sparse_encoder/applications/retrieve_rerank/hybrid_search.py) script, we get the following results for the NanoMSMARCO dataset, using these models:
* **Sparse**: [`tomaarsen/inference-free-splade-distilbert-base-uncased-nq`](https://huggingface.co/tomaarsen/inference-free-splade-distilbert-base-uncased-nq) (the model we just trained)
* **Dense**: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* **Reranker**: [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

| Sparse | Dense | Reranker | NDCG@10 | MRR@10 | MAP   |
|--------|-------|----------|---------|--------|-------|
| x      |       |          | 52.41   | 43.06  | 44.20 |
|        | x     |          | 55.40   | 47.96  | 49.08 |
| x      | x     |          | 62.22   | 53.02  | 53.44 |
| x      |       | x        | 66.31   | 59.45  | 60.36 |
|        | x     | x        | 66.28   | 59.43  | 60.34 |
| x      | x     | x        | 66.28   | 59.43  | 60.34 |

The Sparse and Dense rankings can be combined using Reciprocal Rank Fusion (RRF), which is a simple way to combine the results of multiple rankings. If a Reranker is applied, it will rerank the results of the prior retrieval step.

The results indicate that for this dataset, combining Dense and Sparse rankings is very performant, resulting in 12.3% and 18.7% increases over the Dense and Sparse baselines, respectively. In short, combining Sparse and Dense retrieval methods is a very effective way to improve search performance.

Furthermore, applying a reranker on any of the rankings improved the performance to approximately 66.3 NDCG@10, showing that either Sparse, Dense, or Hybrid (Dense + Sparse) found the relevant documents in their top 100, which the reranker then ranked to the top 10. So, replacing a Dense -> Reranker pipeline with a Sparse -> Reranker pipeline might improve both latency and costs:
* Sparse embeddings can be cheaper to store, e.g. our model only uses ~180 active dimensions for MS MARCO documents instead of the common 1024 dimensions for dense models.
* Some Sparse Encoders allow for inference-free query processing, allowing for a near-instant first-stage retrieval, akin to lexical solutions like BM25.

## Training Tips

Sparse Encoder models have a few quirks that you should be aware of when training them:

1. Sparse Encoder models should not be evaluated solely using the evaluation scores, but also with the sparsity of the embeddings. After all, a low sparsity means that the model embeddings are expensive to store and slow to retrieve.
2. The stronger Sparse Encoder models are trained almost exclusively with distillation from a stronger teacher model (e.g. a [CrossEncoder model](https://sbert.net/docs/cross_encoder/usage/usage.html)), instead of training directly from text pairs or triplets. See for example the [SPLADE-v3 paper](https://arxiv.org/abs/2403.06789), which uses [`SparseDistillKLDivLoss`](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#sentence_transformers.sparse_encoder.losses.SparseDistillKLDivLoss) and [`SparseMarginMSELoss`](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss) for distillation. We don't cover this in detail in this blog as it requires more data preparation, but a distillation setup should be seriously considered.

## Vector Database Integration

After training sparse embedding models, the next crucial step is deploying them effectively in production environments. Vector databases provide the essential infrastructure for storing, indexing, and retrieving sparse embeddings at scale. Popular options include Qdrant, OpenSearch, Elasticsearch, and Seismic, among others.

For comprehensive examples covering vector databases mentioned above, refer to the [semantic search with vector database documentation](https://sbert.net/examples/sparse_encoder/applications/semantic_search/README.html#vector-database-search) or below for the Qdrant example. 

### Qdrant Integration Example

Qdrant offers excellent support for sparse vectors with efficient storage and fast retrieval capabilities. Below is a comprehensive implementation example:

### Prerequisites:
- Qdrant running locally (or accessible), see the [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/) for more details.
- Python Qdrant Client installed:
  ```bash
  pip install qdrant-client
  ```

This example demonstrates how to set up Qdrant for sparse vector search by showing how to efficiently encode and index documents with sparse encoders, formulating search queries with sparse vectors, and providing an interactive query interface. See below: 

```python
import time

from datasets import load_dataset
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.search_engines import semantic_search_qdrant

# 1. Load the natural-questions dataset with 100K answers
dataset = load_dataset("sentence-transformers/natural-questions", split="train")
num_docs = 10_000
corpus = dataset["answer"][:num_docs]

# 2. Come up with some queries
queries = dataset["query"][:2]

# 3. Load the model
sparse_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# 4. Encode the corpus
corpus_embeddings = sparse_model.encode_document(
    corpus, convert_to_sparse_tensor=True, batch_size=16, show_progress_bar=True
)

# Initially, we don't have a qdrant index yet
corpus_index = None
while True:
    # 5. Encode the queries using the full precision
    start_time = time.time()
    query_embeddings = sparse_model.encode_query(queries, convert_to_sparse_tensor=True)
    print(f"Encoding time: {time.time() - start_time:.6f} seconds")

    # 6. Perform semantic search using qdrant
    results, search_time, corpus_index = semantic_search_qdrant(
        query_embeddings,
        corpus_index=corpus_index,
        corpus_embeddings=corpus_embeddings if corpus_index is None else None,
        top_k=5,
        output_index=True,
    )

    # 7. Output the results
    print(f"Search time: {search_time:.6f} seconds")
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        for entry in result:
            print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
        print("")

    # 8. Prompt for more queries
    queries = [input("Please enter a question: ")]
```

## Additional Resources

### Training Examples

The following pages contain training examples with explanations as well as links to code. We recommend that you browse through these to familiarize yourself with the training loop:

- [Model Distillation](https://sbert.net/examples/sparse_encoder/training/distillation/README.html) - Examples to make models smaller, faster and lighter.
- [MS MARCO](https://sbert.net/examples/sparse_encoder/training/ms_marco/README.html) - Example training scripts for training on the MS MARCO information retrieval dataset.
- [Retrievers](https://sbert.net/examples/sparse_encoder/training/retrievers/README.html) - Example training scripts for training on generic information retrieval datasets.
- [Natural Language Inference](https://sbert.net/examples/sparse_encoder/training/nli/README.html) - Natural Language Inference (NLI) data can be quite helpful to pre-train and fine-tune models to create meaningful sparse embeddings.
- [Quora Duplicate Questions](https://sbert.net/examples/sparse_encoder/training/quora_duplicate_questions/README.html) - Quora Duplicate Questions is large set corpus with duplicate questions from the Quora community. The folder contains examples how to train models for duplicate questions mining and for semantic search.
- [STS](https://sbert.net/examples/sparse_encoder/training/sts/README.html) - The most basic method to train models is using Semantic Textual Similarity (STS) data. Here, we use sentence pairs and a score indicating the semantic similarity.

### Documentation

Additionally, the following pages may be useful to learn more about Sentence Transformers:

* [Installation](https://sbert.net/docs/installation.html)
* [Quickstart](https://sbert.net/docs/quickstart.html)
* [Usage](https://sbert.net/docs/sparse_encoder/usage/usage.html)
* [Pretrained Models](https://sbert.net/docs/sparse_encoder/pretrained_models.html)
* [Training Overview](https://sbert.net/docs/sparse_encoder/training_overview.html) (This blogpost is a distillation of the Training Overiew documentation)
* [Dataset Overview](https://sbert.net/docs/sentence_transformer/dataset_overview.html)
* [Loss Overview](https://sbert.net/docs/sparse_encoder/loss_overview.html)
* [API Reference](https://sbert.net/docs/package_reference/sparse_encoder/index.html)

And lastly, here are some advanced pages that might interest you:

* [Hyperparameter Optimization](https://sbert.net/examples/training/hpo/README.html)
* [Distributed Training](https://sbert.net/docs/sentence_transformer/training/distributed.html)