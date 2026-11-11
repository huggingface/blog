---
title: "Token Pooling-Aware Training for Multi-Vector Embedding Models"
thumbnail: /blog/assets/train-sentence-transformers/st-hf-thumbnail.png
authors:
- user: tomaarsen
---

# Token Pooling-Aware Training for Multi-Vector Embedding Models

Multi-vector embedding models give each query token its own opportunity to find a match in a document. This makes them wonderfully expressive for search, but those document vectors add up quickly. A corpus with millions of documents can contain hundreds of millions of vectors!

Token pooling makes these representations smaller by merging similar token vectors. The more aggressively I compress them, however, the easier it becomes to lose something useful. Pooling-aware training teaches the encoder to preserve the information that matters when its vectors are merged, helping you retrieve better from the same storage budget.

In this post, I'll explain how pooling works, train a model for compressed retrieval with Sentence Transformers, and explore the resulting quality and storage tradeoff. With `pool_factor=4`, my normally trained model retains 94.6% of its unpooled NanoBEIR nDCG@10, with a **4x** reduction in document vectors. Token pooling-aware training raises that retention to 97.8% at the same storage budget! Let's see how.

> [!NOTE]
> This post builds on pooling-aware training studied in Stefan Josef's [Learn to Pool](https://stefan-jo.github.io/learn-to-pool/downloads/paper.pdf) and [LightOn's pooling regularization study](https://huggingface.co/blog/lightonai/lateon-hpool-regularization), which also explores targeted training and adaptive budgets. Here, I'll focus on understanding the method and putting it to work with a short, reproducible Sentence Transformers example.

![Mean NanoBEIR nDCG@10 versus stored document vectors for pool factors 2, 3, 4, and 6 and unpooled retrieval, comparing ordinary and pooling-aware training.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/token-pooling/mve_token_pooling_results.png)

<p align="center"><em><small>Pooling-aware training improves retrieval quality at the same vector budget. Details below.</small></em></p>

## Table of Contents

- [Why Multi-Vector Retrieval Needs So Many Vectors](#why-multi-vector-retrieval-needs-so-many-vectors)
- [Token Pooling](#token-pooling)
  * [Choosing a Pooling Budget](#choosing-a-pooling-budget)
  * [Calibrating Adaptive Pooling](#calibrating-adaptive-pooling)
- [Token Pooling-Aware Training](#token-pooling-aware-training)
  * [A Loss Wrapper in Sentence Transformers](#a-loss-wrapper-in-sentence-transformers)
  * [Optional Kernel Acceleration](#optional-kernel-acceleration)
- [Using the Trained Model](#using-the-trained-model)
- [Evaluation Results](#evaluation-results)
- [Combining Compression Techniques](#combining-compression-techniques)
- [References](#references)
- [Code and Models](#code-and-models)

## Why Multi-Vector Retrieval Needs So Many Vectors

A single-vector embedding model compresses an entire document into one vector. [Multi-vector models](https://huggingface.co/blog/multi-vector-encoder) such as ColBERT instead keep a collection of contextual token vectors, allowing different parts of a query to match different parts of a document.

For example, a query about a laptop with a long battery life can find evidence about the laptop in one part of a document and battery endurance in another. With MaxSim scoring, each query vector selects its best matching document vector, and those scores are added together:

$$
\operatorname{MaxSim}(Q, D) = \sum_{q \in Q} \max_{d \in D} q^\top d
$$

Even with the 2-bit or 4-bit residual quantization supported by [Fast-PLAID](https://github.com/lightonai/fast-plaid), storing hundreds of vectors per document adds up. At 4 bits per residual coordinate, 200 vectors of 128 dimensions require 12.8 KB of quantized residuals per document, or 12.8 GB across a million documents. Pooling each document to 50 vectors reduces that to 3.2 GB, at the same dimensionality and precision.

The useful observation is that many document vectors can be similar. Several tokens might describe closely related concepts, creating an opportunity to represent them together.

## Token Pooling

[HierarchicalTokenPooling](https://sbert.net/docs/package_reference/multi_vector_encoder/modules.html#hierarchicaltokenpooling) groups similar token embeddings using [Ward hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering). Starting with one group per token vector, it repeatedly merges the two groups that cause the smallest increase in how spread out the vectors are within their groups. This spread is measured by the sum of squared distances from each vector to its group's mean, so Ward's method favors compact groups of similar vectors.

These merges build a hierarchy of progressively larger groups, letting us choose a level of compression that fits our budget. We then represent each group by its mean and normalize the resulting vectors again, ready for MaxSim scoring.

You can make compression proportional to document length:

```python
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

pooling = HierarchicalTokenPooling(pool_factor=4)
```

With `pool_factor=4`, documents retain approximately one quarter of their original vectors, so longer documents can keep proportionally more capacity. By default, the first token is protected from merging and pooling applies only to document embeddings, leaving query embeddings unchanged.

### Choosing a Pooling Budget

`HierarchicalTokenPooling` offers three alternative controls:

| Setting | What it expresses |
| --- | --- |
| `max_tokens=16` | A maximum of 16 vectors per document, including protected tokens. |
| `pool_factor=4` | Approximately one quarter of each document's original vectors. |
| `distance_threshold=...` | A shared Ward merge-distance threshold, allowing vector counts to vary with embedding geometry. |

This distinction matters when document lengths vary. With `pool_factor=4`, a longer document can retain approximately one quarter of its vectors, allowing its representation to grow with its content. With `distance_threshold`, the vector count can grow when more distinct embeddings need to be preserved.

### Calibrating Adaptive Pooling

Adaptive pooling uses a shared distance threshold to merge similar embeddings more aggressively while preserving more vectors for documents with distinct content. It can be used with any multi-vector model, _including one trained without a pooling-aware objective_.

Rather than choosing a threshold manually, you can calibrate it for a target average vector count. Given a representative sample of your corpus in `calibration_documents`:

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

model = MultiVectorEncoder("lightonai/LateOn")
calibration_embeddings = model.encode_document(calibration_documents)
threshold = HierarchicalTokenPooling.calibrate_distance_threshold(calibration_embeddings, target_tokens=16)
pooling = HierarchicalTokenPooling(distance_threshold=threshold)

# Apply pooling, targeting an average of 16 vectors per document
document_embeddings = model.encode_document(documents_to_index, token_pooling=pooling)
```

Calibration only needs document embeddings and returns a threshold you can reuse when indexing. The target is an average, so individual documents can receive more or fewer than 16 vectors. The achieved average can also change on another corpus.

## Token Pooling-Aware Training

Pooling can already be useful on an ordinary multi-vector model. The difficulty appears when a document contains a distinction that matters for retrieval, but merging its vectors makes that distinction harder to recover. The original training objective never required the model to work with these compressed representations. But, you can teach it to do exactly that.

If you've used [Matryoshka embeddings](https://huggingface.co/blog/matryoshka), the training principle will feel familiar. Matryoshka applies a retrieval objective at several embedding widths, teaching smaller prefixes to remain useful. Pooling-aware training applies the objective to representations with fewer token vectors.

For my example, each training batch contributes two losses:

1. Retrieval using the original document embeddings.
2. Retrieval after compressing documents to approximately one quarter of their original vectors.

I average them:

$$
\mathcal{L} = \tfrac{1}{2}\mathcal{L}_{\mathrm{unpooled}} + \tfrac{1}{2}\mathcal{L}_{\mathrm{pooled}}
$$

The first term teaches the model to retrieve with its full representations. The second teaches it to arrange its token embeddings so that useful matches survive pooling. Both objectives update the same encoder, giving it feedback about retrieval before and after compression.

Ward clustering makes discrete decisions about which tokens belong together. During training, `HierarchicalTokenPooling` uses a straight-through estimator to pass gradients from pooled vectors back to their member tokens, without differentiating the cluster assignments. The encoder can then learn from retrieval errors made with its pooled representations.

### A Loss Wrapper in Sentence Transformers

[`MultiVectorTokenPoolingLoss`](https://sbert.net/docs/package_reference/multi_vector_encoder/losses.html#multivectortokenpoolingloss) wraps an existing multi-vector retrieval loss and applies it using your chosen poolers:

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.losses import (
    CachedMultiVectorMultipleNegativesRankingLoss,
    MultiVectorTokenPoolingLoss,
)
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

model = MultiVectorEncoder("answerdotai/ModernBERT-base", similarity_fn_name="maxsim")
pooling = HierarchicalTokenPooling(pool_factor=4)

base_loss = CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=64)
loss = MultiVectorTokenPoolingLoss(
    model,
    base_loss,
    token_poolings=[None, pooling],
    weights=[0.5, 0.5],
)
```

`CachedMultiVectorMultipleNegativesRankingLoss` supplies the retrieval objective. It teaches each query to score its relevant document above the negative documents in the batch. By default, it uses `model.similarity`, using the model's similarity setting (MaxSim in this example) and kernel settings. A model configured with `similarity_fn_name="meanmaxsim"` would also use MeanMaxSim during training.

Gradient caching allows the encoder to process small mini-batches while retaining a larger set of in-batch negatives. My training script uses mini-batches of 64 with a training batch size of 256.

`MultiVectorTokenPoolingLoss` adds the compression objective. Here, `None` means that one variant uses the original embeddings, while `pooling` supplies the factor-4 variant. The two weights average their losses. Pooling variants reuse the computed token embeddings, so each additional pooler adds clustering and scoring work without requiring a separate encoder pass for that variant.

The poolers can also be custom `BaseTokenPooling` implementations that preserve gradients through their input embeddings. You can adapt the training objective to the representation you want to deploy.

Replace the factor-4 pooler with your chosen deployment configuration, or include several poolers and choose weights for their losses.

The complete [training script](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py) includes the dataset, trainer, validation loss, and small retrieval evaluations during training. From a checkout of the repository, run both variants locally with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv run examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py --baseline
uv run examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py
```

The script declares its training and kernel dependencies, so uv installs them for you. Compatible kernels are enabled automatically.

You can also train on a cloud GPU with [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-quickstart). Install the CLI and log in with a token that can run Jobs and upload models:

```bash
pip install -U huggingface_hub
hf auth login

hf jobs uv run \
    --flavor a10g-small \
    --timeout 4h \
    --secrets HF_TOKEN \
    -- https://raw.githubusercontent.com/huggingface/sentence-transformers/main/examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py
```

This runs the pooling-aware variant on an A10G GPU with a four-hour time limit. Passing `HF_TOKEN` as a secret lets the script upload the trained model to your Hub account when training finishes. To run the ordinary-loss baseline as well, repeat the command with `--baseline` after the script URL.

### Optional Kernel Acceleration

You can accelerate MaxSim scoring, Ward clustering, and supported backbone operations with [kernels](https://sbert.net/docs/multi_vector_encoder/usage/kernels.html). Install the training and kernel dependencies:

```bash
pip install -U "sentence-transformers[train,kernels]"
```

In the training example above, replace the model and pooler initialization before constructing the loss:

```python
model = MultiVectorEncoder(
    "answerdotai/ModernBERT-base",
    device="cuda",
    similarity_fn_name="maxsim",
    use_kernels=True,
)
pooling = HierarchicalTokenPooling(pool_factor=4)
```

Hierarchical poolers inherit the model's kernel settings by default, so you only need to enable kernels on the model.

Sentence Transformers selects compatible kernels for your setup and uses the existing implementation wherever a replacement isn't available. For example, the [default MaxSim kernel](https://huggingface.co/kernels/sentence-transformers/late-interaction-kernels) scores queries against documents without building the full token similarity matrix in memory, but it does require CUDA and Triton.

The default [Ward clustering kernel](https://huggingface.co/kernels/sentence-transformers/ward-clustering) considerably speeds up hierarchical clustering by running on CUDA instead of the CPU. We're talking about 30-150x faster than SciPy depending on the setting.

## Using the Trained Model

Let's load the pooling-aware model and retrieve with compressed document embeddings. Kernel settings aren't saved with the model, so add `use_kernels=True` when loading if you'd like to use them for inference too:

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

model = MultiVectorEncoder("tomaarsen/mve-ModernBERT-base-token-pooling")
pooling = HierarchicalTokenPooling(pool_factor=4)

queries = ["Which planet is known as the Red Planet?"]
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents, token_pooling=pooling)

print([len(embedding) for embedding in document_embeddings])
# [5, 6, 5]
print(model.similarity(query_embeddings, document_embeddings))
# tensor([[16.0037, 25.9879, 22.8159]], device='cuda:0')

# For comparison, the same model without token pooling:
# [18, 21, 19]
# tensor([[16.6115, 27.6060, 24.1766]], device='cuda:0')
```

The pooler compresses the document embeddings after the encoder's forward pipeline, reducing the total from 58 vectors to 16 while leaving query embeddings unchanged. Mars still receives the highest similarity score, followed by Jupiter and Venus, so the ranking is preserved despite storing far fewer document vectors.

You can change `pool_factor` at inference to explore a different storage budget with the same trained model. For repeated use, you can also pass `token_pooling=pooling` when initializing `MultiVectorEncoder` to make it the runtime default. The pooler's `tasks` still control which inputs are compressed (e.g. only documents).

## Evaluation Results

To investigate, I trained two models from `answerdotai/ModernBERT-base`, each with a fresh 128-dimensional token projection:

- [tomaarsen/mve-ModernBERT-base-token-pooling-baseline](https://huggingface.co/tomaarsen/mve-ModernBERT-base-token-pooling-baseline), using only the cached contrastive loss.
- [tomaarsen/mve-ModernBERT-base-token-pooling](https://huggingface.co/tomaarsen/mve-ModernBERT-base-token-pooling), using the cached contrastive loss wrapped with the unpooled and factor-4 objectives above.

Both runs use one epoch over the same 120,000 MS MARCO triplets, seed 42, and matching initialization and optimization settings. The training token cap is 300, and short queries are expanded to at least 32 token positions to provide additional contextual query vectors. Only the loss differs between the two experiments, and I didn't run a hyperparameter search.

I evaluated both models on NanoBEIR@13, reporting mean nDCG@10 (%). The figure compares ordinary and pooling-aware training with pool factors of 2, 3, 4, and 6, alongside unpooled retrieval. The table also includes fixed caps and adaptive thresholds, showing how the training benefits transfer to other pooling settings.

![Mean NanoBEIR nDCG@10 versus stored document vectors for pool factors 2, 3, 4, and 6 and unpooled retrieval, comparing ordinary and pooling-aware training.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/token-pooling/mve_token_pooling_results.png)

| Pooling at inference | Mean vectors | Ordinary model | Pooling-aware model | Gain (points) |
| --- | ---: | ---: | ---: | ---: |
| Unpooled | 193.22 | 51.84 | 52.39 | +0.55 |
| `pool_factor=2` | 96.86 | 50.10 | 51.74 | +1.64 |
| `pool_factor=3` | 64.74 | 49.70 | 51.45 | +1.76 |
| `pool_factor=4` | 48.68 | 49.03 | 51.23 | +2.20 |
| `pool_factor=5` | 39.04 | 48.13 | 50.28 | +2.15 |
| `pool_factor=6` | 32.62 | 47.07 | 49.85 | +2.78 |
| `max_tokens=64` | 54.76 | 49.27 | 50.89 | +1.63 |
| `max_tokens=32` | 29.58 | 48.25 | 50.12 | +1.87 |
| `max_tokens=16` | 15.60 | 46.61 | 49.26 | +2.65 |
| `max_tokens=8` | 7.99 | 43.12 | 46.61 | +3.48 |
| `max_tokens=4` | 4.00 | 37.22 | 43.00 | +5.78 |
| `distance_threshold` (calibrated) | 54.76 | 50.10 | 51.24 | +1.14 |
| `distance_threshold` (calibrated) | 29.58 | 48.38 | 50.96 | +2.58 |
| `distance_threshold` (calibrated) | 15.60 | 47.23 | 49.63 | +2.41 |

For adaptive pooling, I calibrated a separate threshold for each encoder and NanoBEIR corpus, using its unpooled document embeddings to match the average vector counts under caps of 64, 32, and 16. Short documents bring those averages below their respective caps.

At the trained `pool_factor=4`, nDCG@10 improves from 49.03 to 51.23, while storing 74.8% fewer document vectors than unpooled retrieval. This increases score retention from 94.6% to 97.8%, where retention is the pooled mean nDCG@10 divided by each model's own unpooled mean. Unpooled retrieval also improves, from 51.84 to 52.39.

The gains extend to fixed caps and adaptive pooling, even though training used only the full representation and proportional pooling. At an average of 29.58 vectors per document, adaptive pooling retains 97.3% of the pooling-aware model's unpooled score while storing 84.7% fewer document vectors. The normally trained model retains 93.3% at the same budget.

## Combining Compression Techniques

If index size is your main concern, I would reach for these options in this order:

1. Embedding quantization to 4, 2, or 1 bit(s), for example via [Fast-PLAID](https://github.com/lightonai/fast-plaid), to reduce storage per coordinate.
2. Token pooling to reduce the number of document vectors, optionally with pooling-aware fine-tuning as shown in this blog.
3. [Matryoshka token embeddings](https://sbert.net/examples/multi_vector_encoder/training/matryoshka/README.html) to reduce the dimensionality of each token vector, optionally with fine-tuning to preserve retrieval quality at smaller widths.

If the model's default dimensionality is larger than 128, I would consider Matryoshka truncation and fine-tuning earlier as well. These approaches can be combined, so evaluate the resulting index size and retrieval quality with your search engine and corpus.

`MultiVectorMatryoshkaLoss` trains token embeddings to remain useful after truncation. You can wrap it with `MultiVectorTokenPoolingLoss` to train for both smaller dimensions and fewer document vectors, reusing the encoder's embeddings across the compression settings.

Try the [complete training example](https://sbert.net/examples/multi_vector_encoder/training/token_pooling/README.html), choose a budget that matters for your application, and compare it with a matched ordinary-loss run. The accompanying [evaluation script](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/training/token_pooling/evaluate_token_pooling.py) measures unpooled retrieval, caps of 64, 32, 16, 8, and 4 vectors, and pool factors of 2 through 6, so you can plot the quality and storage tradeoff for your own model.

With a few extra lines in the training objective, a smaller document representation becomes something the model actively learns to use. That opens up an exciting amount of room to make multi-vector retrieval more practical!

## References

- Stefan Josef. [Learn to Pool: Lightweight Fine-Tuning for Flexible Multi-Vector Compression](https://stefan-jo.github.io/learn-to-pool).
- Benjamin Clavié, Antoine Chaffin, and Griffin Adams. [Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling](https://huggingface.co/papers/2409.14683).
- Antoine Chaffin. [After the party comes the free lunch: regularizing ColBERT models to enhance pooling capabilities and reduce index footprint](https://huggingface.co/blog/lightonai/lateon-hpool-regularization).
- Luyu Gao, Yunyi Zhang, Jiawei Han, and Jamie Callan. [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://huggingface.co/papers/2101.06983).
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka).

## Code and Models

- [Token pooling-aware training example and README](https://sbert.net/examples/multi_vector_encoder/training/token_pooling/README.html)
- [Complete training script](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py)
- [NanoBEIR evaluation script](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/training/token_pooling/evaluate_token_pooling.py)
- Example checkpoints:
    - [tomaarsen/mve-ModernBERT-base-token-pooling](https://huggingface.co/tomaarsen/mve-ModernBERT-base-token-pooling)
    - [tomaarsen/mve-ModernBERT-base-token-pooling-baseline](https://huggingface.co/tomaarsen/mve-ModernBERT-base-token-pooling-baseline)
- [Pretrained multi-vector models compatible with Sentence Transformers](https://huggingface.co/collections/sentence-transformers/multivectorencoder-integrations)
- [Matryoshka-aware training for multi-vector models](https://sbert.net/examples/multi_vector_encoder/training/matryoshka/README.html)
- [Fast-PLAID](https://github.com/lightonai/fast-plaid)
