---
title: "Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers"
thumbnail: /blog/assets/train-sentence-transformers/st-hf-thumbnail.png
authors:
- user: tomaarsen
---

# Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers

[Sentence Transformers](https://sbert.net/) is a Python library for using and training embedding and reranker models for applications like retrieval augmented generation, semantic search, and more. In my [previous blogpost](https://huggingface.co/blog/multimodal-sentence-transformers), I introduced the new multimodal capabilities, showing how to use embedding and reranker models that handle text, images, audio, and video. In this blogpost, I'll show you how to **train or finetune** these multimodal models on your own data.

As a practical example, I'll walk through finetuning [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) for Visual Document Retrieval (VDR), the task of retrieving relevant document pages (as images, with charts, tables, and layout intact) for a given text query. The resulting [`tomaarsen/Qwen3-VL-Embedding-2B-vdr`](https://huggingface.co/tomaarsen/Qwen3-VL-Embedding-2B-vdr) demonstrates how much performance you can gain by finetuning on your own domain. On my evaluation data, the finetuned model achieves an NDCG@10 of 0.947 compared to the base model's 0.888, and outperforms all existing VDR models I tested against, including models up to 4x its size.

![Model size vs NDCG for VDR models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multimodal-sentence-transformers/vdr_plot.png)

> [!TIP]
> If you're new to multimodal models in Sentence Transformers, I recommend reading [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers) first. For training text-only embedding, reranker, or sparse embedding models, see the [Prior Blogposts](#prior-blogposts) section at the end.

## Table of Contents

* [Why Finetune?](#why-finetune)
* [Training Components](#training-components)
* [Model](#model)
* [Dataset](#dataset)
    + [Visual Document Retrieval Dataset](#visual-document-retrieval-dataset)
    + [Dataset Format](#dataset-format)
* [Loss Function](#loss-function)
    + [CachedMultipleNegativesRankingLoss](#cachedmultiplenegativesrankingloss)
    + [MatryoshkaLoss](#matryoshkaloss)
* [Training Arguments](#training-arguments)
* [Evaluator](#evaluator)
* [Trainer](#trainer)
* [Results](#results)
    + [Model Size vs NDCG@10](#model-size-vs-ndcg10)
    + [Matryoshka Dimensions vs NDCG@10](#matryoshka-dimensions-vs-ndcg10)
* [Training Multimodal Reranker Models](#training-multimodal-reranker-models)
* [Additional Resources](#additional-resources)
    + [Prior Blogposts](#prior-blogposts)
    + [Training Examples](#training-examples)
    + [Documentation](#documentation)

## Why Finetune?

General-purpose multimodal embedding models like [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) are trained on diverse data to perform well across a wide range of languages and tasks: image-text matching, visual question answering, document understanding, and more. But this generality means the model is rarely the best choice for any specific task.

Consider Visual Document Retrieval: given a text query like "What was the company's Q3 revenue?", the model must find the most relevant document screenshot from a corpus of thousands. This requires understanding document layouts, charts, tables, and text, which is a very different skill from e.g. matching pictures of shoes with product descriptions.

By finetuning on domain-specific data, the model can learn these specialized patterns. In my experiment, finetuning improved NDCG@10 from 0.888 to 0.947, ahead of every recent multimodal model I tested, including ones up to 4x larger.

## Training Components

Training multimodal Sentence Transformer models involves the same components as training text-only models:

1. [**Model**](#model): The multimodal model to train or finetune.
2. [**Dataset**](#dataset): The data used for training and evaluation.
3. [**Loss Function**](#loss-function): A function that quantifies the model's performance and guides the optimization process.
4. [**Training Arguments**](#training-arguments) (optional): Parameters that influence training performance and tracking/debugging.
5. [**Evaluator**](#evaluator) (optional): A tool for evaluating the model before, during, or after training.
6. [**Trainer**](#trainer): Brings together the model, dataset, loss function, and other components for training.

The multimodal training pipeline uses the same [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/trainer.html#sentence_transformers.sentence_transformer.trainer.SentenceTransformerTrainer) as text-only training. The key difference is that your datasets contain images (or other modalities) alongside text, and the model's processor handles the image preprocessing automatically.

Let's walk through each component, using Visual Document Retrieval (matching text queries to document screenshots) as a running example.

## Model

The most common approach is to finetune an existing multimodal embedding model, or to start from a Vision-Language Model (VLM) checkpoint. The [`Transformer`](https://sbert.net/docs/package_reference/base/modules.html#sentence_transformers.base.modules.Transformer) module automatically detects supported modalities from the model's processor.

To finetune an existing multimodal embedding model (e.g. one that already has a `modules.json` file), you can pass `processor_kwargs` and `model_kwargs` to control preprocessing and model loading respectively. `processor_kwargs` are passed directly to [`AutoProcessor.from_pretrained(...)`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor.from_pretrained) (e.g., image resolution bounds: higher `max_pixels` means higher quality but more memory), while `model_kwargs` are passed to the appropriate [`AutoModel.from_pretrained(...)`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel.from_pretrained) call (e.g., precision, attention implementation):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Qwen/Qwen3-VL-Embedding-2B",
    model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"},
    processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 600 * 600},
)
```

You can also start from a fresh VLM checkpoint that hasn't been trained for embeddings yet. Sentence Transformers will attempt to recognize the architecture, infer the supported modalities from the processor, and set up the appropriate forward method and pooling. If the automatic detection doesn't work perfectly for a particular model, the configuration in the saved `sentence_bert_config.json` can be edited to adjust modality settings, forward methods, and output handling:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-VL-2B")
```

In both cases, the [`Transformer`](https://sbert.net/docs/package_reference/base/modules.html#sentence_transformers.base.modules.Transformer) module inspects the processor to determine which modalities are available, and [`Pooling`](https://sbert.net/docs/package_reference/sentence_transformer/modules.html#sentence_transformers.sentence_transformer.modules.Pooling) is added automatically if needed. You can verify the supported modalities:

```python
print(model.modalities)
# ['text', 'image', 'video', 'message']

print(model.supports("image"))
# True
```

<details>
<summary>Alternative: Building multimodal models with Router</summary>

Instead of using a single VLM backbone, you can compose separate encoders for different modalities using the [`Router`](https://sbert.net/docs/package_reference/base/modules.html#sentence_transformers.base.modules.Router) module. This lets you combine any existing encoders and route inputs to the appropriate one based on detected modality:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Dense, Pooling, Router, Transformer

# Create separate encoders for different modalities
text_encoder = Transformer("sentence-transformers/all-MiniLM-L6-v2")
text_pooling = Pooling(text_encoder.get_embedding_dimension(), pooling_mode="mean")
text_projection = Dense(text_encoder.get_embedding_dimension(), 768)

# SigLIP outputs pooled embeddings directly, so no separate Pooling module is needed
image_encoder = Transformer("google/siglip2-base-patch16-224")

# Route inputs based on modality
router = Router(
    sub_modules={
        "text": [text_encoder, text_pooling, text_projection],
        "image": [image_encoder],
    },
)

model = SentenceTransformer(modules=[router])
```

> [!WARNING]
> Since Router-based multimodal models use separate encoders per modality, their embedding spaces are initially unaligned. Training is required to align the spaces for meaningful cross-modal similarity. The `Dense` projection layer shown above helps map embeddings from different encoders into a shared space.

This approach is useful when you want to use lightweight, specialized encoders rather than a large VLM. You can also combine Router-based multimodality with task-based routing (e.g. different encoders for queries vs. documents) using `route_mappings`. See the [`Router`](https://sbert.net/docs/package_reference/base/modules.html#sentence_transformers.base.modules.Router) documentation for advanced routing scenarios.

</details>

## Dataset

### Visual Document Retrieval Dataset

For this example, I use the [`tomaarsen/llamaindex-vdr-en-train-preprocessed`](https://huggingface.co/datasets/tomaarsen/llamaindex-vdr-en-train-preprocessed) dataset, a preprocessed English subset of [`llamaindex/vdr-multilingual-train`](https://huggingface.co/datasets/llamaindex/vdr-multilingual-train). The source dataset was released alongside the [Visual Document Retrieval Goes Multilingual](https://huggingface.co/blog/vdr-2b-multilingual) blogpost by LlamaIndex, and consists of ~500k multilingual query-image samples collected from public internet PDFs, with queries synthetically generated using VLMs (gemini-1.5-pro and Qwen2-VL-72B). 
My preprocessed version filters to the 53,512 English samples and resolves 4 of the 16 ID-based hard negatives per sample into actual document screenshot images, so it can be used directly for training without further preprocessing:

```python
from datasets import load_dataset

train_dataset = load_dataset("tomaarsen/llamaindex-vdr-en-train-preprocessed", "train", split="train")
train_dataset = train_dataset.select_columns(["query", "image", "negative_0"])
eval_dataset = load_dataset("tomaarsen/llamaindex-vdr-en-train-preprocessed", "eval", split="train")
```

The `train` config contains the first 10,000 samples, and the `eval` config contains the next 300 samples (a `full` config with all 53,512 samples is also available). For training, I select `query`, `image`, and `negative_0` to form (anchor, positive, hard negative) triplets. Including additional hard negatives would likely improve the training signal, but each extra negative also increases memory usage and training time, so I stick with one. For evaluation, I keep all four hard negatives per query to build a more challenging retrieval corpus (more on that in the [Evaluator](#evaluator) section).

### Dataset Format

Just like text-only training, the dataset format must match your chosen [loss function](#loss-function). The rules are the same:

1. If your loss function requires a *Label*, your dataset must have a column named **"label"** or **"score"**.
2. All columns other than **"label"** or **"score"** are considered *Inputs*. The number of these columns must match the number of valid inputs for your chosen loss function. Beyond the label column, the column names don't matter, only the order does.

For multimodal datasets, the inputs can contain:
- **Text**: strings.
- **Image**: PIL images, file paths, URLs, or numpy/torch arrays.
- **Audio**: file paths, numpy/torch arrays, dicts with `"array"` and `"sampling_rate"` keys, or (if `torchcodec` is installed) `torchcodec.AudioDecoder` instances.
- **Video**: file paths, numpy/torch arrays, dicts with `"array"` and `"video_metadata"` keys, or (if `torchcodec` is installed) `torchcodec.VideoDecoder` instances.
- **Multimodal dicts**: a dict mapping modality names to values, e.g. `{"text": ..., "image": ...}`. The keys must be `"text"`, `"image"`, `"audio"`, or `"video"`.

The data collator automatically calls `model.preprocess()`, which detects the modality of each input and applies the appropriate preprocessing. No manual tokenization or image processing is needed.

> [!TIP]
> Many Hugging Face datasets that work out of the box with Sentence Transformers have been tagged with `sentence-transformers`, allowing you to easily find them at [https://huggingface.co/datasets?other=sentence-transformers](https://huggingface.co/datasets?other=sentence-transformers).

## Loss Function

### CachedMultipleNegativesRankingLoss

For this training, I use [`CachedMultipleNegativesRankingLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss), a common choice for retrieval tasks. It accepts (query, positive) pairs with any number of additional hard negative columns, from 0 up to n, as long as each sample has the same number of negatives.
During training, the loss pushes each query's similarity to its positive *up* and its similarity to every negative *down*. The negatives come from two sources:

1. **Hard negatives**: the negative column(s) explicitly supplied in the dataset (just `negative_0` in our triplet setup).
2. **In-batch negatives**: the positives and hard negatives from every *other* sample in the same batch, reused as additional negatives for this query at no extra cost.

More negatives per query means a stronger training signal, so a larger batch size directly improves training quality. Beyond that, the "cached" variant of the loss uses gradient caching to make large effective batch sizes feasible even when GPU memory is limited.

The `mini_batch_size` parameter controls how many samples are processed at once during the cached forward passes. For large multimodal models, setting this to a small value (e.g., 1) is important to avoid out-of-memory errors without sacrificing the benefits of large effective batch sizes:

```python
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss

loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=1)
```

### MatryoshkaLoss

To produce embeddings that work well at multiple dimensionalities, I wrap the base loss with [`MatryoshkaLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss). This trains the model so that truncating the embedding to a smaller number of dimensions still yields good performance:

```python
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss

loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=1)
loss = MatryoshkaLoss(model, loss, matryoshka_dims=[2048, 1536, 1024, 512, 256, 128, 64])
```

This is especially useful for multimodal models, where embeddings can be large (2048 dimensions for Qwen3-VL). With Matryoshka training, you can use truncated embeddings (e.g., 256 or 128 dimensions) at deployment time for faster search with minimal quality loss. As I'll show in the [Results](#results) section, the finetuned model achieves near-peak performance even at 512 dimensions.

## Training Arguments

The [`SentenceTransformerTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentencetransformertrainingarguments) class lets you control training hyperparameters. Here's the configuration used for the VDR finetuning:

```python
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments, BatchSamplers

run_name = "Qwen3-VL-Embedding-2B-vdr"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.05,
    run_name=run_name,
)
```

A few things to note for (multimodal) training:

- `bf16=True`: bfloat16 is generally preferred over float16 due to better numerical stability.
- `batch_sampler=BatchSamplers.NO_DUPLICATES`: When using `MultipleNegativesRankingLoss` or its cached variant, having no duplicate samples in a batch ensures that every in-batch negative is a truly different sample.
- `per_device_train_batch_size=64`: This may seem large for a 2B parameter VLM, but `CachedMultipleNegativesRankingLoss` with `mini_batch_size=1` handles the memory constraints through gradient caching.
- `eval_steps`, `save_steps`, and `logging_steps`: Setting these to a fraction (e.g., 0.1) means evaluation, saving, and logging will happen every 10% of an epoch, which is useful for monitoring training progress.

## Evaluator

To track retrieval performance before, during, and after training, I use the [`InformationRetrievalEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#informationretrievalevaluator). It computes standard retrieval metrics like NDCG@10, MAP, and Recall@k:

```python
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

# Build the evaluation data from the eval dataset.
# Queries and corpus use integer IDs: query 0's relevant document is corpus 0.
eval_queries = {qid: sample["query"] for qid, sample in enumerate(eval_dataset)}
eval_corpus = {did: sample["image"] for did, sample in enumerate(eval_dataset)}
num_eval = len(eval_dataset)

# Add hard negatives to the corpus with offset IDs (num_eval, 2*num_eval, ...)
# so they don't collide with the positive document IDs (0..num_eval-1).
negative_columns = ["negative_0", "negative_1", "negative_2", "negative_3"]
for neg_idx, neg_col in enumerate(negative_columns):
    for did, sample in enumerate(eval_dataset):
        eval_corpus[num_eval * (neg_idx + 1) + did] = sample[neg_col]

# Each query's relevant document is the positive at the same index
eval_relevant_docs = {idx: [idx] for idx in range(len(eval_dataset))}

eval_evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    batch_size=1,
    show_progress_bar=True,
    name="vdr-eval-hard",
)
```

The evaluator takes text queries, a corpus of images (including hard negatives), and a mapping of which documents are relevant to which queries. Note that the corpus contains a mix of positive and hard negative document screenshots, making this a challenging evaluation. Using `batch_size=1` prevents out-of-memory issues during evaluation of the large VLM.

## Trainer

The [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/trainer.html#sentence_transformers.sentence_transformer.trainer.SentenceTransformerTrainer) brings everything together. Here's the complete training script:

```python
from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.sentence_transformer.model_card import SentenceTransformerModelCardData
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

# 1. Load a model to finetune with (optional) model card data
model = SentenceTransformer(
    "Qwen/Qwen3-VL-Embedding-2B",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="Qwen3-VL-Embedding-2B model trained on Visual Document Retrieval query-document screenshot pairs",
    ),
    model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"},
    # Control image resolution: lower values save memory, higher values preserve detail
    processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 600 * 600},
)

# 2. Load a dataset to finetune on: (query, positive, negative_0) triplets for training,
# all 4 hard negatives retained for evaluation
train_dataset = load_dataset("tomaarsen/llamaindex-vdr-en-train-preprocessed", "train", split="train")
train_dataset = train_dataset.select_columns(["query", "image", "negative_0"])
eval_dataset = load_dataset("tomaarsen/llamaindex-vdr-en-train-preprocessed", "eval", split="train")

# 3. Define a loss function
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=1)
loss = MatryoshkaLoss(model, loss, matryoshka_dims=[2048, 1536, 1024, 512, 256, 128, 64])

# 4. (Optional) Specify training arguments
run_name = "Qwen3-VL-Embedding-2B-vdr"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # BF16 is preferred over FP16 for VLMs due to better numerical stability
    bf16=True,  # Set to True if your GPU supports BF16 (most modern GPUs do)
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.05,
    run_name=run_name,  # Used in e.g. Trackio if installed
    # report_to=["codecarbon", "trackio"],  # Uncomment to enable logging (pip install codecarbon trackio)
)

# 5. (Optional) Create an evaluator & evaluate the base model
eval_queries = {qid: sample["query"] for qid, sample in enumerate(eval_dataset)}
eval_corpus = {did: sample["image"] for did, sample in enumerate(eval_dataset)}
num_eval = len(eval_dataset)
negative_columns = ["negative_0", "negative_1", "negative_2", "negative_3"]
for neg_idx, neg_col in enumerate(negative_columns):
    for did, sample in enumerate(eval_dataset):
        eval_corpus[num_eval * (neg_idx + 1) + did] = sample[neg_col]
eval_relevant_docs = {idx: [idx] for idx in range(len(eval_dataset))}

eval_evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    batch_size=1,
    show_progress_bar=True,
    name="vdr-eval-hard",
)
eval_evaluator(model)

# 6. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=eval_evaluator,
)
trainer.train()

# 7. (Optional) Evaluate at each Matryoshka dimension
eval_evaluator(model)
for dim in [2048, 1536, 1024, 512, 256, 128, 64]:
    dim_evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant_docs,
        truncate_dim=dim,
        batch_size=1,
        show_progress_bar=True,
        name=f"vdr-eval-hard-{dim}d",
    )
    dim_evaluator(model)

# 8. Save the trained model
model.save_pretrained(f"models/{run_name}/final")

# 9. (Optional) Push it to the Hugging Face Hub
# This pushes to your personal namespace, e.g. {your_username}/Qwen3-VL-Embedding-2B-vdr
model.push_to_hub("Qwen3-VL-Embedding-2B-vdr")
```

The training script is nearly identical to a text-only training script. The only differences are:

1. Model loading: We pass `model_kwargs` for precision and attention implementation, and `processor_kwargs` for image resolution bounds.
2. Loss function: We use `CachedMultipleNegativesRankingLoss` with `mini_batch_size=1` to handle the large VLM without running out of memory.
3. Evaluator: The evaluator uses images in the corpus and text as queries, enabling cross-modal retrieval evaluation.

Everything else (the trainer, training arguments, dataset loading) works exactly the same as text-only training.

## Results

### Model Size vs NDCG@10

After training for just 1 epoch, the finetuned [tomaarsen/Qwen3-VL-Embedding-2B-vdr](https://huggingface.co/tomaarsen/Qwen3-VL-Embedding-2B-vdr) model achieves an NDCG@10 of **0.947** on the evaluation set (300 queries, 1500 corpus documents, cosine similarity). This is a significant improvement over the base [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) model's 0.888, and outperforms all existing VDR models:

![Model size vs NDCG for VDR models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multimodal-sentence-transformers/vdr_plot.png)

<details>
<summary>Full NDCG@10 numbers by model (20 models)</summary>

| Model | Parameters | NDCG@10 |
| :--- | :---: | :---: |
| **tomaarsen/Qwen3-VL-Embedding-2B-vdr** | **2.1B** | **0.947** |
| Qwen/Qwen3-VL-Embedding-8B | 8.1B | 0.923 |
| nvidia/omni-embed-nemotron-3b | 4.7B | 0.915 |
| nvidia/llama-nemotron-embed-vl-1b-v2 | 1.7B | 0.912 |
| nomic-ai/nomic-embed-multimodal-7b | 8.3B | 0.912 |
| llamaindex/vdr-2b-multi-v1 | 2.2B | 0.912 |
| llamaindex/vdr-2b-v1 | 2.2B | 0.911 |
| nomic-ai/nomic-embed-multimodal-3b | 3.8B | 0.899 |
| Qwen/Qwen3-VL-Embedding-2B | 2.1B | 0.888 |
| LCO-Embedding/LCO-Embedding-Omni-7B | 8.9B | 0.888 |
| LCO-Embedding/LCO-Embedding-Omni-3B | 4.7B | 0.860 |
| BAAI/BGE-VL-v1.5-zs | 7.6B | 0.800 |
| BAAI/BGE-VL-v1.5-mmeb | 7.6B | 0.797 |
| BAAI/BGE-VL-MLLM-S2 | 7.6B | 0.792 |
| BidirLM/BidirLM-Omni-2.5B-Embedding | 2.5B | 0.775 |
| royokong/e5-v | 8.4B | 0.767 |
| BAAI/BGE-VL-MLLM-S1 | 7.6B | 0.710 |
| sentence-transformers/clip-ViT-L-14 | 428M | 0.611 |
| BAAI/BGE-VL-large | 428M | 0.467 |
| BAAI/BGE-VL-base | 150M | 0.335 |

</details>


The finetuned 2B model outperforms even the 8B Qwen3-VL-Embedding model, demonstrating the power of task-specific finetuning. Finetuning on your own domain is often worth considering, even when a larger general-purpose model is available!

### Matryoshka Dimensions vs NDCG@10

The comparison above uses full-size 2048-dim embeddings. Thanks to the Matryoshka training, the finetuned model also holds up well when truncated to fewer dimensions, letting you trade off embedding size and retrieval quality at deployment time:

![MRL dimensions vs NDCG@10](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multimodal-sentence-transformers/vdr_plot_mrl.png)

> [!NOTE]
> The finetuned model's peak is at the full 2048 dimensions (0.948), but it stays within 0.3% of peak all the way down to 512 (4x smaller), and retains over 92% of peak even at 64 (32x smaller). Matryoshka training concentrates the most important information in the earlier dimensions, so moderate truncation costs very little performance.

<details>
<summary>Full NDCG@10 numbers by dimension</summary>

| Dimensions | Base NDCG@10 | Finetuned NDCG@10 |
| :---: | :---: | :---: |
| 2048 (full) | 0.8961 (100%) | 0.9480 (100%) |
| 1536 | 0.8940 (99.8%) | 0.9439 (99.6%) |
| 1024 | 0.8941 (99.8%) | 0.9464 (99.8%) |
| 512 | 0.8760 (97.8%) | 0.9451 (99.7%) |
| 256 | 0.8347 (93.2%) | 0.9372 (98.9%) |
| 128 | 0.7888 (88.0%) | 0.9058 (95.5%) |
| 64 | 0.6852 (76.5%) | 0.8758 (92.4%) |

</details>

The gap between 1024 and 2048 dimensions is small (0.946 vs. 0.948), so I've saved the model with `truncate_dim=1024` set in its configuration. This means that `SentenceTransformer("tomaarsen/Qwen3-VL-Embedding-2B-vdr")` produces 1024-dimensional embeddings by default, halving the storage footprint compared to the full 2048. If you want a different dimensionality, pass `truncate_dim=N` when loading to override it.

## Training Multimodal Reranker Models

You can also finetune multimodal Cross Encoder (reranker) models using the same training infrastructure. The key difference is using [`CrossEncoderTrainer`](https://sbert.net/docs/package_reference/cross_encoder/trainer.html#sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer) and Cross Encoder-specific loss functions. This section provides a brief overview; see the [full training examples](https://github.com/huggingface/sentence-transformers/tree/main/examples/cross_encoder/training/multimodal) for complete, runnable scripts with dataset preparation and evaluation.

Here's a simplified example based on the [doodles training script](https://github.com/huggingface/sentence-transformers/blob/main/examples/cross_encoder/training/multimodal/training_doodles_any_to_any.py), which trains a reranker to match images with text captions:

```python
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.modules import LogitScore, Transformer
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

# 1. Build the model from modules
transformer = Transformer(
    "Qwen/Qwen3.5-0.8B",
    transformer_task="any-to-any",
    model_kwargs={"torch_dtype": "bfloat16", "device_map": "auto", "attn_implementation": "flash_attention_2"},
    processing_kwargs={"chat_template": {"add_generation_prompt": True}},
)

# Extend chat template to support "query" and "document" roles
transformer.processor.chat_template = transformer.processor.chat_template.replace(
    'message.role == "user"', 'message.role in ["user", "query", "document"]'
)

# LogitScore: score = log(P("1")) - log(P("0"))
score_head = LogitScore(
    true_token_id=transformer.tokenizer.convert_tokens_to_ids("1"),
    false_token_id=transformer.tokenizer.convert_tokens_to_ids("0"),
)

model = CrossEncoder(
    modules=[transformer, score_head],
    num_labels=1,
    prompts={
        "image_to_text": "Given the image, judge whether the text matches it. Respond with 1 if they match, 0 if they don't.",
        "text_to_image": "Given the text, judge whether the image matches it. Respond with 1 if they match, 0 if they don't.",
    },
)

# 2. Define the loss
loss = BinaryCrossEntropyLoss(model)

# 3. Multi-dataset training with separate directions
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset={"image_to_text": train_image_to_text, "text_to_image": train_text_to_image},
    eval_dataset={"image_to_text": eval_image_to_text, "text_to_image": eval_text_to_image},
    loss=loss,
    evaluator=[image_to_text_evaluator, text_to_image_evaluator],
)
trainer.train()
```

There are multiple valid architectural choices for multimodal rerankers, including:

1. Any-to-Any + LogitScore: Uses the multimodal language model to generate a token, then computes the log-odds of "1" vs "0".
2. Feature Extraction + Pooling + Dense: Uses only the multimodal base model, and extracts the last token's hidden state and projects it to a score via a Dense layer, avoiding the language modeling head computation.

Both approaches are demonstrated in the [multimodal cross encoder training examples](https://github.com/huggingface/sentence-transformers/tree/main/examples/cross_encoder/training/multimodal).

The two scripts linked above split the training data into two datasets, one per direction (image-to-text and text-to-image), with a task-specific prompt for each that tells the model how to score in that direction. Each positive pair is then expanded with randomly sampled negatives so the loss sees a balanced mix of matches and non-matches.

## Additional Resources

### Prior Blogposts

* [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): Multimodal inference
* [Training and Finetuning Embedding Models with Sentence Transformers v3](https://huggingface.co/blog/train-sentence-transformers): Training embedding models
* [Training and Finetuning Reranker Models with Sentence Transformers v4](https://huggingface.co/blog/train-reranker): Training reranker models
* [Training and Finetuning Sparse Embedding Models with Sentence Transformers v5](https://huggingface.co/blog/train-sparse-encoder): Training sparse embedding models

### Training Examples

The Sentence Transformers repository includes several multimodal training examples:

* [Visual Document Retrieval](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/training/multimodal/training_visual_document_retrieval.py): The training script used in this blogpost to finetune a VLM-based embedding model for document screenshot retrieval
* [Multimodal Reranker (Any-to-Any)](https://github.com/huggingface/sentence-transformers/blob/main/examples/cross_encoder/training/multimodal/training_doodles_any_to_any.py): Train a multimodal reranker using LogitScore
* [Multimodal Reranker (Feature Extraction)](https://github.com/huggingface/sentence-transformers/blob/main/examples/cross_encoder/training/multimodal/training_doodles_feature_extraction.py): Train a multimodal reranker using Pooling + Dense

### Documentation

Additionally, the following pages may be useful to learn more about training with Sentence Transformers:

* [Sentence Transformer > Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html)
* [Sentence Transformer > Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html)
* [Cross Encoder > Training Overview](https://sbert.net/docs/cross_encoder/training_overview.html)
* [Cross Encoder > Loss Overview](https://sbert.net/docs/cross_encoder/loss_overview.html)
* [Dataset Overview](https://sbert.net/docs/sentence_transformer/dataset_overview.html)
* [API Reference](https://sbert.net/docs/package_reference/sentence_transformer/index.html)
