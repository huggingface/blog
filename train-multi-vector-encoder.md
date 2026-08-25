---
title: "Training and Finetuning Multi-Vector Embedding Models with Sentence Transformers"
thumbnail: /blog/assets/train-sentence-transformers/st-hf-thumbnail.png
authors:
- user: tomaarsen
---

# Training and Finetuning Multi-Vector Embedding Models with Sentence Transformers

[Sentence Transformers](https://sbert.net/) is a Python library for using and training embedding and reranker models for a wide range of applications, such as retrieval augmented generation, semantic search, semantic textual similarity, and more. Its v6.0 update introduces a fourth model type: `MultiVectorEncoder`, for ColBERT-style late interaction retrieval, alongside a complete training approach for it. In this blogpost, I'll show you how to use it to finetune a multi-vector model that outperforms general-purpose retrievers on your data. This method can also train strong new multi-vector models from scratch.

Finetuning multi-vector models involves several components: the model itself, datasets, loss functions, training arguments, evaluators, and the trainer class. I'll have a look at each of these components, accompanied by practical examples of how they can be used for finetuning strong multi-vector models.

Lastly, in the [Evaluation](#evaluation) section, I'll show you that my finetuned [multi-vector-encoder/mLateOn-medical](https://huggingface.co/multi-vector-encoder/mLateOn-medical) model, trained in 14.5 hours on a single RTX 3090 alongside this blogpost, easily outperforms every general-purpose retrieval model I could find on my medical retrieval evaluation: dense, sparse, lexical, and multi-vector alike.

![NDCG@10 on MIRIAD versus active parameters: the finetuned mLateOn-medical reaches the top at a fraction of the size of the strongest general-purpose models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-multi-vector-encoder/mve_medical_model_size_ndcg.png)

If you're interested in finetuning dense embedding models, sparse embedding models, or rerankers instead, then consider reading through my prior [Training and Finetuning Embedding Models](https://huggingface.co/blog/train-sentence-transformers), [Training and Finetuning Sparse Embedding Models](https://huggingface.co/blog/train-sparse-encoder), and [Training and Finetuning Reranker Models](https://huggingface.co/blog/train-reranker) blogposts. And if you first want to learn how to *use* multi-vector models, from loading and encoding to indexing in vector databases, see the companion [Multi-Vector (Late Interaction) Embedding Models with Sentence Transformers](https://huggingface.co/blog/multi-vector-encoder) blogpost.

## Table of Contents

- [What are Multi-Vector models?](#what-are-multi-vector-models)
- [Why Finetune?](#why-finetune)
- [Training Components](#training-components)
- [Model](#model)
  * [Finetuning an existing multi-vector model](#finetuning-an-existing-multi-vector-model)
  * [Building one from a base transformer](#building-one-from-a-base-transformer)
  * [Which starting point should you pick?](#which-starting-point-should-you-pick)
- [Dataset](#dataset)
  * [Data on the Hugging Face Hub](#data-on-the-hugging-face-hub)
  * [Local Data](#local-data)
  * [Dataset Format](#dataset-format)
- [Loss Function](#loss-function)
- [Training Arguments](#training-arguments)
- [Evaluator](#evaluator)
- [Trainer](#trainer)
  * [Callbacks](#callbacks)
  * [Multi-Dataset Training](#multi-dataset-training)
- [Evaluation](#evaluation)
  * [Shrinking the index with token pooling](#shrinking-the-index-with-token-pooling)
- [Additional Resources](#additional-resources)
  * [Training Examples](#training-examples)
  * [Documentation](#documentation)

## What are Multi-Vector models?

A dense embedding model compresses a whole text into a single vector, and similarity is one dot product between two such summaries. A multi-vector model (also called a late-interaction or ColBERT-style model) skips that compression: it keeps **one small vector per token** and scores a query against a document with the MaxSim operator, where every query token finds its best-matching document token and the scores are summed. Token-level matching preserves exactly the fine-grained signals that a single vector has to average away, which usually means stronger retrieval, at the cost of a bigger index.

The companion [Multi-Vector Embedding Models](https://huggingface.co/blog/multi-vector-encoder) blogpost covers the architecture, encoding, scoring, and indexing in detail, so I'll keep this section short and get to the training.

![Dense embedding versus multi-vector late interaction](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/maxsim_explainer.gif)

## Why Finetune?

Finetuning multi-vector models significantly improves their retrieval performance on your specific domain: the vocabulary, the query style, and the notion of relevance all differ between web search, legal discovery, code search, and scientific literature review. Because queries and documents are matched token by token, multi-vector models pick up fine-grained domain signals that single-vector models tend to average away, and they respond very well to even modest amounts of in-domain finetuning data.

There is a second, less obvious reason: **most released retrieval models were configured for short passages**. The classic ColBERT checkpoints truncate documents at 180 or 300 tokens, and many popular dense models at 256 or 512, because their MS MARCO-style training data rarely goes beyond that. If your documents are long, these models silently discard most of every document before scoring it. On my medical evaluation with passages averaging 941 tokens, I measured that this truncation costs up to 0.24 NDCG@10, considerably more than any difference between model architectures. When you train your own model, you configure the document length that *your* data needs.

LightOn ran into this same dynamic with code retrieval: general [LateOn](https://huggingface.co/lightonai/LateOn) wasn't enough, so they trained [LateOn-Code](https://huggingface.co/lightonai/LateOn-Code). Your domain, whether that's medical, legal, financial, or your company's internal documents, is not getting an official model. This blogpost shows you how to build it yourself, in a matter of hours, on a single consumer GPU.

## Training Components

Training MultiVectorEncoder models involves the following components:

1. [**Model**](#model): The model to finetune or the architecture to build fresh.
2. [**Dataset**](#dataset): The data used for training and evaluation.
3. [**Loss Function**](#loss-function): A function that measures the model's performance and guides the optimization process.
4. [**Training Arguments**](#training-arguments) (optional): Parameters that impact training performance, tracking, and debugging.
5. [**Evaluator**](#evaluator) (optional): A class for evaluating the model before, during, or after training.
6. [**Trainer**](#trainer): Brings together all training components.

Let's take a closer look at each component.

## Model

Multi-vector training gives you a real choice of starting point, and it matters more than you might expect.

### Finetuning an existing multi-vector model

If you want to further finetune an existing multi-vector model, you don't have to worry about the architecture at all:

```python
from sentence_transformers import MultiVectorEncoder

# Loading in fp32 is preferred for training if your memory can handle it
model = MultiVectorEncoder(
    "lightonai/mLateOn-unsupervised",
    model_kwargs={"torch_dtype": "float32"},
    processor_kwargs={"model_max_length": 8192},  # the tokenizer-level token limit
)
```

The checkpoint brings its own recipe along: its query and document marker tokens, its projection head, its scoring skiplist. For finetuning, you generally want to keep all of that and change only what your data demands. The first thing to check is the length configuration: many released checkpoints cap documents at 180 to 512 tokens (see [Why Finetune?](#why-finetune)), and my medical passages run to 1,400 tokens. The mLateOn family already serves the backbone's full 8192 token context, but if your starting checkpoint carries caps, lift them:

```python
# Let the model read full documents instead of the caps it was trained with,
# e.g. GTE-ModernColBERT-v1 ships with query_length=48 and document_length=300
model[0].query_length = None
model[0].document_length = None
```

With the per-task caps unset, truncation falls back to the tokenizer's `model_max_length`, which is why I configure that limit at load time above.

I made one more change: a punctuation skiplist, which excludes punctuation tokens from document-side scoring and storage. In a 4-way ablation (none, punctuation, stopwords, both) it modestly won on quality, and it shrinks the document index by 9.6% on this data for free:

```python
import string

# model[2] is the MultiVectorMask module
model[2].skiplist_words = list(string.punctuation)
model[2].resolve_with_tokenizer(model.tokenizer)  # token ids are cached, so re-resolve after changing
```

### Building one from a base transformer

You can also point `MultiVectorEncoder` at any base transformer, and a fresh, randomly initialized token-level projection is appended for you:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("answerdotai/ModernBERT-base", model_kwargs={"torch_dtype": "float32"})
# MultiVectorEncoder(
#   (0): Transformer({..., 'architecture': 'ModernBertModel'})
#   (1): Dense({'in_features': 768, 'out_features': 128, 'bias': False, ...})
#   (2): MultiVectorMask({'skiplist_words': [], 'skiplist_tasks': ['document'], ...})
#   (3): Normalize({...})
# )
```

That's the classic ColBERT pipeline: a `Transformer` producing contextualized token embeddings, a token-level `Dense` projecting each of them down to 128 dimensions, a `MultiVectorMask` deciding which tokens count during scoring, and a token-level `Normalize`. The projection starts random, so training is required before this model is useful. Interestingly, this works with strong dense embedding backbones too: a fresh projection on [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) reached within 0.03 of the existing-checkpoint starting points in my experiments, from nothing but the projection and 25k training pairs.

The classic ColBERT tokenization tricks (`[MASK]` query expansion, `[Q]` / `[D]` prefix tokens, a document length cap, a punctuation skiplist) are all off by default and configurable. See [Creating Custom Models](https://sbert.net/docs/multi_vector_encoder/usage/custom_models.html) for the full set. For what it's worth, I tested `[MASK]` query expansion in four configurations for my domain finetune and none of them made a measurable difference, so don't feel obliged to reach for the classic recipe.

### Which starting point should you pick?

I measured this directly while preparing this blogpost. I took six starting points and trained each with the identical recipe on 25k medical question-passage pairs from [MIRIAD](https://huggingface.co/datasets/tomaarsen/miriad-4.4M-split), then evaluated on 1,000 held-out questions against a 50,000 passage corpus:

| Starting point | Zero-shot NDCG@10 | After 25k pairs | Delta |
|---|---:|---:|---:|
| [lightonai/mLateOn-unsupervised](https://huggingface.co/lightonai/mLateOn-unsupervised) | 0.9087 | **0.9398** | **+0.0311** |
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | 0.9277 | 0.9319 | +0.0042 |
| [lightonai/LateOn-unsupervised](https://huggingface.co/lightonai/LateOn-unsupervised) | 0.9026 | **0.9206** | **+0.0180** |
| [lightonai/LateOn](https://huggingface.co/lightonai/LateOn) | 0.9185 | 0.9105 | -0.0080 |
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 0.9198 | 0.9007 | -0.0191 |
| Fresh head on [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | - | 0.9177 | - |

The result surprised me, and it replicated across two model families: **the `-unsupervised` checkpoints adapt to a new domain far better than their finished siblings**, overtaking them despite starting lower. These checkpoints sit after large-scale contrastive pretraining but before supervised finetuning on general retrieval, so they carry all the late-interaction structure with none of the general-purpose tuning that domain training then has to undo. The finished checkpoints, by contrast, barely moved or even regressed, at every learning rate I tried.

My recommendation: **if the model family you like publishes a pre-supervised checkpoint, start there.** If not, a fresh projection on a strong retrieval-pretrained backbone is a close runner-up. Continuing from a fully finished checkpoint is the weakest option for domain adaptation, despite being the most natural-feeling one.

## Dataset

The [`MultiVectorEncoderTrainer`](https://sbert.net/docs/package_reference/multi_vector_encoder/trainer.html) uses [`datasets.Dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) or [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict) instances for training and evaluation. You can load data from the [Hugging Face Datasets Hub](https://huggingface.co/datasets) or use local data in whatever format you prefer (e.g. CSV, JSON, Parquet, Arrow, or SQL).

**Note:** Lots of public datasets that work out of the box with Sentence Transformers have been tagged with `sentence-transformers` on the Hugging Face Hub, so you can easily find them on [https://huggingface.co/datasets?other=sentence-transformers](https://huggingface.co/datasets?other=sentence-transformers). Consider browsing through these to find ready-to-go datasets that might be useful for your tasks, domains, or languages.

### Data on the Hugging Face Hub

You can use the [`load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset) function to load data from datasets on the Hub:

```python
from datasets import load_dataset

train_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="train")

print(train_dataset)
"""
Dataset({
    features: ['question', 'passage_text'],
    num_rows: 4467542
})
"""
```

This is the dataset I'll train on in this blogpost: 4.4 million medical questions from [MIRIAD](https://huggingface.co/datasets/miriad/miriad-4.4M), each paired with the source passage that contains its answer (averaging 941 tokens). Simple (query, relevant passage) pairs like these are the easiest retrieval training data to collect for your own domain, and as you'll see, they're all you need.

### Local Data

You can also use [`load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset) for loading local data in common file formats:

```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files="my_file.csv")
# or
dataset = load_dataset("json", data_files="my_file.json")
```

And if your local data requires pre-processing, you can use [`datasets.Dataset.from_dict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.from_dict) to initialize your dataset with a dictionary of lists:

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

### Dataset Format

It is important that your dataset format matches your loss function (or that you choose a loss function that matches your dataset format). Verifying whether a dataset format works with a loss function involves two steps:

1. If your loss function requires a *Label* according to the [Loss Overview](https://sbert.net/docs/multi_vector_encoder/loss_overview.html) table, then your dataset must have a **column named "label" or "score"**. This column is automatically taken as the label.
2. All columns not named "label" or "score" are considered *Inputs* according to the [Loss Overview](https://sbert.net/docs/multi_vector_encoder/loss_overview.html) table. The number of remaining columns must match the number of valid inputs for your chosen loss. The names of these columns are **irrelevant**, only the **order matters**.

There are two multi-vector specific conventions on top of this:

- **Positional query and document assignment**: the first column is embedded as the *query* and all following columns as *documents*, regardless of the column names. This default can be overridden per column via the standard `router_mapping` training argument.
- **Knowledge distillation format**: one column per candidate document, i.e. `(query, document_1, ..., document_N, scores)` where `scores` is a list of N teacher scores per row. For KD datasets that store query and document *IDs* alongside separate text datasets (e.g. [lightonai/ms-marco-en-bge](https://huggingface.co/datasets/lightonai/ms-marco-en-bge)), you can use [`resolve_ids`](https://sbert.net/docs/package_reference/util.html#sentence_transformers.util.dataset.resolve_ids) to resolve the IDs to texts on the fly.

## Loss Function

Loss functions quantify how well a model performs for a given batch of data, allowing an optimizer to update the model weights to produce more favourable (i.e., lower) loss values. The right loss function for your task depends on the data you have and what you're trying to achieve. You can find a full list of options in the [Loss Overview](https://sbert.net/docs/multi_vector_encoder/loss_overview.html).

For the common case of question-answer or question-passage pairs, the workhorse is in-batch negatives training with [`MultiVectorMultipleNegativesRankingLoss`](https://sbert.net/docs/package_reference/multi_vector_encoder/losses.html#multivectormultiplenegativesrankingloss): every other document in the batch acts as a negative for each query. Bigger batches mean more negatives and stronger training, so in practice you'll want its GradCache variant, [`CachedMultiVectorMultipleNegativesRankingLoss`](https://sbert.net/docs/package_reference/multi_vector_encoder/losses.html#cachedmultivectormultiplenegativesrankingloss), which decouples the effective batch size from what fits on your GPU:

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.losses import CachedMultiVectorMultipleNegativesRankingLoss

model = MultiVectorEncoder("lightonai/mLateOn-unsupervised", model_kwargs={"torch_dtype": "float32"})

loss = CachedMultiVectorMultipleNegativesRankingLoss(
    model=model,
    mini_batch_size=16,  # how many documents to encode per chunk: bounds memory, not quality
)
```

The `mini_batch_size` parameter bounds the memory: documents are encoded in chunks of this size, while the effective contrastive batch size (128 in my run below, and in my ablations bigger batches bought nothing further) stays a free choice. GradCache guarantees identical results regardless of the chunk size, so lower it for smaller GPUs at only a wall-clock cost. When your document lengths vary a lot, consider its sibling `mini_batch_num_tokens`, which packs each chunk to a total token budget instead of a document count, so a chunk of unusually long documents can never spike your memory (my `mini_batch_size=16` at roughly 940 tokens per document corresponds to `mini_batch_num_tokens=15_000`).

One multi-vector specific trap: the contrastive losses default to `scale=1.0`, unlike the dense embedding equivalent which defaults to `scale=20.0`. That 20.0 exists because a cosine similarity is a single value in [-1, 1], too narrow a range for a sharp softmax. A MaxSim score instead sums one best-match similarity per query token, so it already spans roughly [0, query_length]: a 32-token query can score up to 32. So, don't copy `scale=20.0` over from a dense training script: it would saturate the softmax and kill your gradients.

For distillation from a stronger teacher, which is how the strongest general-purpose late-interaction models are trained, see [`MultiVectorDistillKLDivLoss`](https://sbert.net/docs/package_reference/multi_vector_encoder/losses.html#multivectordistillkldivloss) and the Knowledge Distillation tab in the [Training Overview](https://sbert.net/docs/multi_vector_encoder/training_overview.html#trainer) documentation.

## Training Arguments

You can customize the training process using the [`MultiVectorEncoderTrainingArguments`](https://sbert.net/docs/package_reference/multi_vector_encoder/training_args.html) class. This class lets you adjust parameters that can impact training speed and help you understand what's happening during training.

For more information on the most useful training arguments, check out the [Multi-Vector Encoder > Training Overview > Training Arguments](https://sbert.net/docs/multi_vector_encoder/training_overview.html#training-arguments). It's worth reading to get the most out of your training.

Here's an example, using the values from my actual training run:

```python
from sentence_transformers import MultiVectorEncoderTrainingArguments
from sentence_transformers.base.sampler import BatchSamplers

args = MultiVectorEncoderTrainingArguments(
    # Required parameter:
    output_dir="models/mLateOn-medical",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=128,  # the effective contrastive batch, thanks to GradCache
    per_device_eval_batch_size=16,
    learning_rate=1e-4,
    warmup_steps=0.05,
    prompts={"question": "[Q] ", "passage_text": "[D] "},  # the checkpoint's markers, keyed by training column
    fp16=False,  # Set to True if you have a GPU that supports FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # in-batch negatives benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.05,
    logging_steps=0.01,
    run_name="mLateOn-medical",  # Will be used in e.g. Trackio, W&B, etc.
)
```

A few of these deserve a comment:

- **`prompts`**: training does not automatically apply the prompts stored in the model, so map them onto your training columns explicitly. Here that is the checkpoint's `[Q] ` marker for the question column and `[D] ` for the passage column, keeping training consistent with inference.
- **`max_length` (deliberately not set)**: this argument caps tokenization during *training only*, for when you want cheaper training than the model's full serving length. I measured what that shortcut costs on this data: training at 512 tokens lost about 0.015 NDCG@10 for about 2x the speed, and the deficit did not shrink with more data, because the model simply never sees what got cut off. Leave it unset so training matches inference, unless you need the speedup more than the quality.
- **`learning_rate=1e-4`**: after a sweep from 5e-6 to 2e-4, I had the best luck with this higher-than-usual learning rate.

## Evaluator

To track your model's performance during training, you can pass an `eval_dataset` to the trainer for evaluation loss, but concrete retrieval metrics are much more informative. Sentence Transformers includes the following built-in evaluators for multi-vector models:

| Evaluator | Required Data |
| --- | --- |
| [`MultiVectorInformationRetrievalEvaluator`](https://sbert.net/docs/package_reference/multi_vector_encoder/evaluation.html#multivectorinformationretrievalevaluator) | Queries, corpus, and relevant document mappings |
| [`MultiVectorNanoBEIREvaluator`](https://sbert.net/docs/package_reference/multi_vector_encoder/evaluation.html#multivectornanobeirevaluator) | No data required |
| [`MultiVectorTripletEvaluator`](https://sbert.net/docs/package_reference/multi_vector_encoder/evaluation.html#multivectortripletevaluator) | (anchor, positive, negative) triplets |
| [`MultiVectorRerankingEvaluator`](https://sbert.net/docs/package_reference/multi_vector_encoder/evaluation.html#multivectorrerankingevaluator) | List of `{'query': '...', 'positive': [...], 'negative': [...]}` dictionaries |
| [`MultiVectorDistillationEvaluator`](https://sbert.net/docs/package_reference/multi_vector_encoder/evaluation.html#multivectordistillationevaluator) | Queries with candidate documents and teacher scores |

For domain finetuning, the [`MultiVectorInformationRetrievalEvaluator`](https://sbert.net/docs/package_reference/multi_vector_encoder/evaluation.html#multivectorinformationretrievalevaluator) built from your own held-out data is the one that matters. One tip on constructing it: **the corpus should be hard enough that models can be told apart**. In my case the MIRIAD questions are generated from their own source passages, which makes retrieval unusually easy: against just the 10k gold passages, nearly every model scored above 0.97 NDCG@10. If your evaluation saturates like that, add *distractor* passages (I use deduplicated passages from the training split) until the scores spread out:

```python
from datasets import load_dataset
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator

dataset = load_dataset("tomaarsen/miriad-4.4M-split")

# Gold: 1,000 evaluation questions, each mapping to its own passage, with the
# eval split's full ~10k unique passages as the initial corpus
corpus = {}
queries = {}
relevant_docs = {}
passage_to_id = {}
for idx, row in enumerate(dataset["eval"]):
    if row["passage_text"] not in passage_to_id:
        passage_to_id[row["passage_text"]] = f"p{len(passage_to_id)}"
        corpus[passage_to_id[row["passage_text"]]] = row["passage_text"]
    if idx < 1_000:
        queries[f"q{idx}"] = row["question"]
        relevant_docs[f"q{idx}"] = {passage_to_id[row["passage_text"]]}

# Distractors: unique train passages that make the haystack realistic
seen = set(passage_to_id)
for row in dataset["train"]:
    if len(corpus) >= 200_000:
        break
    if row["passage_text"] not in seen:
        seen.add(row["passage_text"])
        corpus[f"d{len(corpus)}"] = row["passage_text"]

evaluator = MultiVectorInformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="miriad-dev",
    batch_size=16,
)
# results = evaluator(model)
```

## Trainer

The [`MultiVectorEncoderTrainer`](https://sbert.net/docs/package_reference/multi_vector_encoder/trainer.html) is where all previous components come together. Here is the complete script that trained [multi-vector-encoder/mLateOn-medical](https://huggingface.co/multi-vector-encoder/mLateOn-medical), the model from the introduction:

```python
import logging
import string
import traceback

from datasets import load_dataset

from sentence_transformers import (
    MultiVectorEncoder,
    MultiVectorEncoderModelCardData,
    MultiVectorEncoderTrainer,
    MultiVectorEncoderTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator
from sentence_transformers.multi_vector_encoder.losses import CachedMultiVectorMultipleNegativesRankingLoss

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    # 1. Load the starting checkpoint: contrastively pretrained, not yet supervised
    # Loading in fp32 is preferred for training if your memory can handle it
    model = MultiVectorEncoder(
        "lightonai/mLateOn-unsupervised",
        model_kwargs={"torch_dtype": "float32"},
        processor_kwargs={"model_max_length": 8192},
        model_card_data=MultiVectorEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="mLateOn finetuned on MIRIAD medical retrieval",
        ),
    )

    # 2. Lift the per-task length caps so training and inference see full medical passages
    model[0].query_length = None
    model[0].document_length = None

    # 3. Skip punctuation tokens during scoring: a small quality win and a 9.6% smaller index
    model[2].skiplist_words = list(string.punctuation)
    model[2].resolve_with_tokenizer(model.tokenizer)

    # 4. Load 1 million medical question-passage pairs
    train_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="train").select(range(1_000_000))

    # 5. In-batch negatives with GradCache: large effective batch, memory-bounded chunks
    loss = CachedMultiVectorMultipleNegativesRankingLoss(model=model, mini_batch_size=16)

    # 6. A light dev evaluator to watch progress during training: 500 held-out questions
    # against the eval split's ~10k unique passages. The full 200k protocol runs afterwards.
    eval_split = load_dataset("tomaarsen/miriad-4.4M-split", split="eval")
    corpus, queries, relevant_docs, passage_to_id = {}, {}, {}, {}
    for idx, row in enumerate(eval_split):
        if row["passage_text"] not in passage_to_id:
            passage_to_id[row["passage_text"]] = f"p{len(passage_to_id)}"
            corpus[passage_to_id[row["passage_text"]]] = row["passage_text"]
        if idx < 500:
            queries[f"q{idx}"] = row["question"]
            relevant_docs[f"q{idx}"] = {passage_to_id[row["passage_text"]]}
    dev_evaluator = MultiVectorInformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs, name="miriad-dev", batch_size=16
    )

    # 7. Training arguments, as discussed above
    run_name = "mLateOn-medical"
    args = MultiVectorEncoderTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=16,
        learning_rate=1e-4,
        warmup_steps=0.05,
        prompts={"question": "[Q] ", "passage_text": "[D] "},
        fp16=False,  # Set to True if you have a GPU that supports FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.05,
        logging_steps=0.01,
        run_name=run_name,
    )

    # 8. Create a trainer & train
    trainer = MultiVectorEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 9. Save the trained model
    model.save_pretrained(f"models/{run_name}/final")

    # 10. (Optional) Push it to the Hugging Face Hub
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
```

That's the whole recipe: a pre-supervised checkpoint, a million domain pairs, in-batch negatives, full document length, and a higher-than-usual learning rate. The run took 14.5 hours on my single RTX 3090 at a peak of 17.5 GB VRAM, and every one of those choices was the winner of a measured comparison rather than a guess.

For readers on smaller budgets: my scaling experiments put 100k pairs (75 minutes of training) within 0.012 NDCG@10 of the full million-pair run. Most of the gain comes in the first hour.

### Callbacks

The MultiVectorEncoder trainer supports various [`transformers.TrainerCallback`](https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback) subclasses, including:

- [`WandbCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.WandbCallback) for logging training metrics to W&B if `wandb` is installed
- [`TensorBoardCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.TensorBoardCallback) for logging training metrics to TensorBoard if `tensorboard` is accessible
- [`CodeCarbonCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.CodeCarbonCallback) for tracking carbon emissions during training if `codecarbon` is installed

Enable these via the `report_to` training argument, e.g. `report_to=["wandb", "codecarbon"]`, with the required dependencies installed. It defaults to `"none"`, and `report_to="all"` activates every integration whose dependency is installed.

Refer to the [Transformers Callbacks documentation](https://huggingface.co/docs/transformers/en/main_classes/callback) for more information on these callbacks and how to create your own.

### Multi-Dataset Training

Typically, top-performing general-purpose models are trained on multiple datasets simultaneously. However, this approach can be challenging due to the varying formats of each dataset. Fortunately, the [`MultiVectorEncoderTrainer`](https://sbert.net/docs/package_reference/multi_vector_encoder/trainer.html) allows you to train on multiple datasets without requiring a uniform format. Additionally, it provides the flexibility to apply different loss functions to each dataset. Here are the steps to train with multiple datasets at once:

- Use a dictionary of [`datasets.Dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) instances (or a [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict)) as the `train_dataset` (and optionally also `eval_dataset`).
- (Optional) Use a dictionary of loss functions mapping dataset names to losses. Only required if you wish to use different loss functions for different datasets.

Each training/evaluation batch will only contain samples from one of the datasets. The order in which batches are sampled from the multiple datasets is defined by the [`MultiDatasetBatchSamplers`](https://sbert.net/docs/package_reference/sentence_transformer/sampler.html#sentence_transformers.training_args.MultiDatasetBatchSamplers) enum, which can be passed to the [`MultiVectorEncoderTrainingArguments`](https://sbert.net/docs/package_reference/multi_vector_encoder/training_args.html) via `multi_dataset_batch_sampler`. Valid options are:

- `MultiDatasetBatchSamplers.ROUND_ROBIN`: Round-robin sampling from each dataset until one is exhausted. With this strategy, it's likely that not all samples from each dataset are used, but each dataset is sampled from equally.
- `MultiDatasetBatchSamplers.PROPORTIONAL` (default): Sample from each dataset in proportion to its size. With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.

## Evaluation

To find out where the finetuned model stands, I evaluated it against over 40 retrieval model configurations across four architecture families on the MIRIAD evaluation set, built exactly as in the [Evaluator](#evaluator) section above: 1,000 held-out medical questions searching **200,000 unique passages** (the 10k gold passages hidden among 190k deduplicated distractors from the training split).

![NDCG@10 versus active parameters on the MIRIAD 200k benchmark, with an arrow marking the finetuning jump from mLateOn-unsupervised to mLateOn-medical](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-multi-vector-encoder/mve_medical_model_size_ndcg.png)

The headline results, with the full table in the collapsible below:

| Model | Family | NDCG@10 |
|---|---|---:|
| [**multi-vector-encoder/mLateOn-medical (ours)**](https://huggingface.co/multi-vector-encoder/mLateOn-medical) | **Multi-vector, finetuned** | **0.9139** |
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | Multi-vector, zero-shot | 0.8520 |
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) (cap lifted) | Multi-vector, zero-shot | 0.8502 |
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Dense, zero-shot | 0.7817 |
| [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) | Dense, zero-shot | 0.7563 |
| BM25 | Lexical | 0.7501 |
| [naver/splade-v3](https://huggingface.co/naver/splade-v3) | Sparse, zero-shot | 0.6853 |

The finetuned model tops the table, beating the strongest zero-shot model of any architecture by +0.062 NDCG@10. In other words, the strongest zero-shot model returns the right passage as the very first hit for 75.8% of the queries, while the finetuned model does so for 84.9%, cutting the rank-1 error by more than a third.

The architecture pattern is just as clear: the top of the table is exclusively late interaction. On long documents, one vector per token beats one vector per document, even at matched training and matched backbones. DenseOn and LateOn share training data and architecture except for the head, and the late-interaction sibling wins by +0.12, with the multilingual pair (mDenseOn and mLateOn) replicating this at +0.13 and the pplx-embed pair at +0.03 once its length cap is lifted. Scale doesn't rescue single vectors either: [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), the strongest dense model with roughly 33x the active (non-embedding) parameters of mine, still stops 0.13 short, and the 8B version scores lower than the 4B.

BM25 also performs surprisingly well, beating every sparse model, every truncation-capped multi-vector model, and all but three dense models: the multi-billion [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) and [8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), and [voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano), which reads its full 32k token context to edge past by just 0.006. Don't expect that to transfer to your own data though: MIRIAD's questions are generated from the passages, so the lexical overlap between a query and its gold passage is far larger than in typical retrieval, and BM25's unlimited context length lets it use every one of those overlapping words while most neural checkpoints truncate. A BM25 baseline is cheap and always worth running, just don't count on this margin.

The full field at a glance, sorted by score and colored by architecture family:

![Sorted NDCG@10 on the MIRIAD 200k benchmark for every evaluated model, colored by architecture family](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-multi-vector-encoder/mve_medical_ndcg_by_model.png)

<details><summary>Click to see the full evaluation table</summary>

| Model | Family | NDCG@10 | acc@1 |
|---|---|---:|---:|
| [**multi-vector-encoder/mLateOn-medical (ours)**](https://huggingface.co/multi-vector-encoder/mLateOn-medical) | Multi-vector, finetuned | **0.9139** | 0.849 |
| [tomaarsen/multivector-gte-modernbert-base-miriad](https://huggingface.co/tomaarsen/multivector-gte-modernbert-base-miriad) (ours, previous) | Multi-vector, finetuned | 0.9063 | 0.838 |
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | Multi-vector | 0.8520 | 0.758 |
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) @1024 | Multi-vector | 0.8502 | 0.763 |
| [lightonai/LateOn](https://huggingface.co/lightonai/LateOn) @1024 | Multi-vector | 0.8485 | 0.760 |
| [lightonai/mLateOn-unsupervised](https://huggingface.co/lightonai/mLateOn-unsupervised) | Multi-vector | 0.8304 | 0.733 |
| [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) @1024 | Multi-vector | 0.8186 | 0.727 |
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Dense | 0.7817 | 0.669 |
| [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Dense | 0.7747 | 0.654 |
| [perplexity-ai/pplx-embed-v1-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-late-0.6b) @1024 | Multi-vector | 0.7702 | 0.632 |
| [lightonai/ColBERT-Zero](https://huggingface.co/lightonai/ColBERT-Zero) | Multi-vector | 0.7613 | 0.675 |
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | Multi-vector | 0.7582 | 0.664 |
| [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) | Dense | 0.7563 | 0.638 |
| BM25 | Lexical | 0.7501 | 0.641 |
| [jinaai/jina-embeddings-v5-text-small-retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval) | Dense | 0.7470 | 0.620 |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Dense | 0.7408 | 0.620 |
| [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) | Dense | 0.7384 | 0.615 |
| [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) | Multi-vector | 0.7350 | 0.639 |
| [mixedbread-ai/mxbai-edge-colbert-v0-17m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m) | Multi-vector | 0.7271 | 0.631 |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) @512 | Multi-vector | 0.7264 | 0.615 |
| [lightonai/DenseOn](https://huggingface.co/lightonai/DenseOn) @1024 | Dense | 0.7239 | 0.597 |
| [lightonai/mDenseOn](https://huggingface.co/lightonai/mDenseOn) @1024 | Dense | 0.7227 | 0.585 |
| [jinaai/jina-embeddings-v5-text-nano-retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval) | Dense | 0.7206 | 0.587 |
| [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | Dense | 0.7126 | 0.572 |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | Dense | 0.7102 | 0.582 |
| [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) | Dense | 0.7068 | 0.568 |
| [perplexity-ai/pplx-embed-v1-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-late-0.6b) | Multi-vector | 0.7008 | 0.570 |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | Dense | 0.7000 | 0.563 |
| [lightonai/DenseOn](https://huggingface.co/lightonai/DenseOn) | Dense | 0.6943 | 0.570 |
| [naver/splade-v3](https://huggingface.co/naver/splade-v3) | Sparse | 0.6853 | 0.574 |
| [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) | Dense | 0.6813 | 0.546 |
| [naver/splade-v3-distilbert](https://huggingface.co/naver/splade-v3-distilbert) | Sparse | 0.6806 | 0.567 |
| [codefuse-ai/F2LLM-v2-0.6B](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B) | Dense | 0.6799 | 0.536 |
| [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) @512 | Multi-vector | 0.6785 | 0.571 |
| [prithivida/Splade_PP_en_v1](https://huggingface.co/prithivida/Splade_PP_en_v1) | Sparse | 0.6755 | 0.577 |
| [lightonai/LateOn](https://huggingface.co/lightonai/LateOn) | Multi-vector | 0.6713 | 0.561 |
| [tomaarsen/embeddinggemma-300m-miriad-unsloth](https://huggingface.co/tomaarsen/embeddinggemma-300m-miriad-unsloth) | Dense, finetuned | 0.6705 | 0.530 |
| [lightonai/LateOn-regularized](https://huggingface.co/lightonai/LateOn-regularized) | Multi-vector | 0.6673 | 0.554 |
| [lightonai/LateOn-unsupervised](https://huggingface.co/lightonai/LateOn-unsupervised) | Multi-vector | 0.6672 | 0.553 |
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | Multi-vector | 0.6612 | 0.555 |
| [opensearch-project/opensearch-neural-sparse-encoding-v2-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v2-distill) | Sparse | 0.6518 | 0.531 |
| [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (prompted) | Dense | 0.6387 | 0.498 |
| [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | Dense | 0.6355 | 0.502 |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | Dense | 0.6308 | 0.498 |
| [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) @1024 | Multi-vector | 0.6218 | 0.504 |
| [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Dense | 0.6203 | 0.487 |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | Multi-vector | 0.6184 | 0.514 |
| [tomaarsen/splade-modernbert-base-miriad](https://huggingface.co/tomaarsen/splade-modernbert-base-miriad) | Sparse, finetuned | 0.6142 | 0.473 |
| [NeuML/biomedbert-base-colbert](https://huggingface.co/NeuML/biomedbert-base-colbert) | Multi-vector | 0.5963 | 0.463 |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | Dense | 0.5930 | 0.454 |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | Dense | 0.5881 | 0.457 |
| [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Dense | 0.5159 | 0.396 |
| [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) | Multi-vector | 0.4992 | 0.401 |
| [mixedbread-ai/mxbai-colbert-large-v1](https://huggingface.co/mixedbread-ai/mxbai-colbert-large-v1) | Multi-vector | 0.4690 | 0.358 |
| [sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) | Dense | 0.4614 | 0.323 |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Dense | 0.4458 | 0.321 |
| [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | Multi-vector | 0.4347 | 0.346 |

Models marked `@N` are evaluated with their document length cap lifted to N tokens, since their native caps (180 to 512 tokens) would otherwise truncate the 941-token average passages. For every multi-vector model this lift was worth +0.08 to +0.24 NDCG@10 over the as-served row, and even the dense DenseOn gained +0.03 from the same treatment.

</details>

Note that this does not mean that [multi-vector-encoder/mLateOn-medical](https://huggingface.co/multi-vector-encoder/mLateOn-medical) is the strongest model on *all* domains: it's simply the strongest in *my* domain. This is totally fine, as I just need this model to work well on my data.

Don't underestimate the power of finetuning multi-vector models on your domain. Fourteen and a half hours on a single consumer GPU produced a model that no general-purpose retriever comes close to on this data, and the recipe is a single script with no teacher model and no mined negatives!

### Shrinking the index with token pooling

The fair objection to multi-vector retrieval is index size, and this domain is close to the worst case for it. Storing one vector per token, my model needs about 878 vectors per passage, so the 200,000-passage corpus takes roughly 45 GB at fp16, where a dense model needs well under 1 GB. Document length is what makes that gap so wide: the Natural Questions passages in the [companion post](https://huggingface.co/blog/multi-vector-encoder) average about 125 token vectors each, seven times fewer, so a corpus of short passages starts from a far smaller index than this one does. The [`HierarchicalTokenPooling`](https://sbert.net/docs/package_reference/multi_vector_encoder/modules.html#hierarchicaltokenpooling) module compresses exactly this: it clusters each document's token embeddings and stores the cluster means, keeping roughly `1 / pool_factor` of the vectors:

```python
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

pooling = HierarchicalTokenPooling(pool_factor=4)
document_embeddings = model.encode_document(passages, token_pooling=pooling)
```

I measured it post-hoc on the finished model, with no pooling-aware training, and on long documents it is remarkably cheap:

![Index size for the 200,000-passage corpus versus NDCG@10, with the token pooling trajectory sweeping the multi-vector index into dense-model territory](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-multi-vector-encoder/mve_medical_index_size_ndcg.png)

(Every size in that figure is measured rather than nominal: I encoded a sample of the corpus with each model at its evaluated configuration and counted the vectors it actually stores, at fp16, with the sparse and BM25 rows counting one id plus one weight per active dimension.)

Halving the index costs 0.0033 NDCG@10 and leaves rank-1 accuracy untouched, a quarter of the index still scores 0.8991, and even at pool factor 10, with the index down to 4.5 GB and inside dense-model territory, the model scores 0.8765: still ahead of every other model in the benchmark. If index size has kept you away from late interaction, this curve is the answer.

## Additional Resources

### Training Examples

These pages have training examples with explanations as well as links to training scripts. You can use them to get familiar with the multi-vector training loop:

* [MIRIAD](https://sbert.net/examples/multi_vector_encoder/training/miriad/README.html): domain-specific training on medical retrieval, an earlier and simpler cousin of this blogpost's recipe
* [MS MARCO](https://sbert.net/examples/multi_vector_encoder/training/msmarco/README.html): contrastive and knowledge distillation recipes
* [Multimodal](https://sbert.net/examples/multi_vector_encoder/training/multimodal/README.html): ColPali-style visual document retrieval training
* [PEFT Adapters](https://sbert.net/examples/multi_vector_encoder/training/peft/README.html): parameter-efficient finetuning with LoRA

### Documentation

For further learning, you may also want to explore the following resources on Sentence Transformers:

* [Installation](https://sbert.net/docs/installation.html)
* [Quickstart](https://sbert.net/docs/quickstart.html)
* [Usage](https://sbert.net/docs/multi_vector_encoder/usage/usage.html)
* [Creating Custom Models](https://sbert.net/docs/multi_vector_encoder/usage/custom_models.html)
* [Pretrained Models](https://sbert.net/docs/multi_vector_encoder/pretrained_models.html)
* [Training Overview](https://sbert.net/docs/multi_vector_encoder/training_overview.html) (This blogpost is a distillation of the Training Overview documentation)
* [Loss Overview](https://sbert.net/docs/multi_vector_encoder/loss_overview.html)
* [API Reference](https://sbert.net/docs/package_reference/multi_vector_encoder/index.html)

And here is an advanced page that might interest you:

* [Distributed Training](https://sbert.net/docs/sentence_transformer/training/distributed.html)

And the companion blogpost, covering everything about *using* these models:

* [Multi-Vector (Late Interaction) Embedding Models with Sentence Transformers](https://huggingface.co/blog/multi-vector-encoder)
