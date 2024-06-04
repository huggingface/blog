---
title: "用 Sentence Transformers v3 训练和微调嵌入模型"
thumbnail: /blog/assets/train-sentence-transformers/st-hf-thumbnail.png
authors:
- user: tomaarsen
translators:
- user: innovation64
- user: zhongdongy
  proofreader: true
---

# 用 Sentence Transformers v3 训练和微调嵌入模型

[Sentence Transformers](https://sbert.net/) 是一个 Python 库，用于使用和训练各种应用的嵌入模型，例如检索增强生成 (RAG)、语义搜索、语义文本相似度、释义挖掘 (paraphrase mining) 等等。其 3.0 版本的更新是该工程自创建以来最大的一次，引入了一种新的训练方法。在这篇博客中，我将向你展示如何使用它来微调 Sentence Transformer 模型，以提高它们在特定任务上的性能。你也可以使用这种方法从头开始训练新的 Sentence Transformer 模型。

现在，微调 Sentence Transformers 涉及几个组成部分，包括数据集、损失函数、训练参数、评估器以及新的训练器本身。我将详细讲解每个组成部分，并提供如何使用它们来训练有效模型的示例。

## 目录

- [为什么进行微调？](#为什么进行微调)
- [训练组件](#训练组件)
- [数据集](#数据集)
  - [Hugging Face Hub 上的数据](#hugging-face-hub-上的数据)
  - [本地数据 (CSV, JSON, Parquet, Arrow, SQL)](#本地数据-csv-json-parquet-arrow-sql)
  - [需要预处理的本地数据](#需要预处理的本地数据)
  - [数据集格式](#数据集格式)

- [损失函数](#损失函数)
- [训练参数](#训练参数)
- [评估器](#评估器)
  - [使用 STSb 的 Embedding Similarity Evaluator](#使用-stsb-的-embedding-similarity-evaluator)
  - [使用 AllNLI 的 Triplet Evaluator](#使用-allnli-的-triplet-evaluator)

- [训练器](#训练器)
  - [回调函数](#回调函数)

- [多数据集训练](#多数据集训练)
- [弃用](#弃用)
- [附加资源](#附加资源)
  - [训练示例](#训练示例)
  - [文档](#文档)

## 为什么进行微调？

微调 Sentence Transformer 模型可以显著提高它们在特定任务上的性能。这是因为每个任务都需要独特的相似性概念。让我们以几个新闻文章标题为例:

- “Apple 发布新款 iPad”
- “NVIDIA 正在为下一代 GPU 做准备 “

根据用例的不同，我们可能希望这些文本具有相似或不相似的嵌入。例如，一个针对新闻文章的分类模型可能会将这些文本视为相似，因为它们都属于技术类别。另一方面，一个语义文本相似度或检索模型应该将它们视为不相似，因为它们具有不同的含义。

## 训练组件

训练 Sentence Transformer 模型涉及以下组件:

1. [ **数据集** ](#数据集): 用于训练和评估的数据。
2. [ **损失函数** ](#损失函数): 一个量化模型性能并指导优化过程的函数。
3. [ **训练参数** ](#训练参数) (可选): 影响训练性能和跟踪/调试的参数。
4. [ **评估器** ](#评估器) (可选): 一个在训练前、中或后评估模型的工具。
5. [ **训练器** ](#训练器): 将模型、数据集、损失函数和其他组件整合在一起进行训练。

现在，让我们更详细地了解这些组件。

## 数据集

[`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) 使用 [`datasets.Dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) 或 [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict) 实例进行训练和评估。你可以从 Hugging Face 数据集中心加载数据，或使用各种格式的本地数据，如 CSV、JSON、Parquet、Arrow 或 SQL。

注意: 许多开箱即用的 Sentence Transformers 的 Hugging Face 数据集已经标记为 `sentence-transformers` ，你可以通过浏览 [https://huggingface.co/datasets?other=sentence-transformers](https://huggingface.co/datasets?other=sentence-transformers) 轻松找到它们。我们强烈建议你浏览这些数据集，以找到可能对你任务有用的训练数据集。

### Hugging Face Hub 上的数据

要从 Hugging Face Hub 中的数据集加载数据，请使用 [`load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset) 函数:

```python
from datasets import load_dataset

train_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev")

print(train_dataset)
"""
Dataset({
    features: ['premise', 'hypothesis', 'label'],
    num_rows: 942069
})
"""
```

一些数据集，如 [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli)，具有多个子集，不同的数据格式。你需要指定子集名称以及数据集名称。

### 本地数据 (CSV, JSON, Parquet, Arrow, SQL)

如果你有常见文件格式的本地数据，你也可以使用 [`load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset) 轻松加载:

```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files="my_file.csv")
# or
dataset = load_dataset("json", data_files="my_file.json")
```

### 需要预处理的本地数据

如果你的本地数据需要预处理，你可以使用 [`datasets.Dataset.from_dict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.from_dict) 用列表字典初始化你的数据集:

```python
from datasets import Dataset

anchors = []
positives = []
# Open a file, perform preprocessing, filtering, cleaning, etc.
# and append to the lists

dataset = Dataset.from_dict({
    "anchor": anchors,
    "positive": positives,
})
```

字典中的每个键都成为结果数据集中的列。

### 数据集格式

确保你的数据集格式与你选择的 [损失函数](#损失函数) 相匹配至关重要。这包括检查两件事:

1. 如果你的损失函数需要 _标签_ (如 [损失概览](https://sbert.net/docs/sentence_transformer/loss_overview.html) 表中所指示)，你的数据集必须有一个名为**“label” **或**“score”**的列。
2. 除 **“label”** 或 **“score”** 之外的所有列都被视为 _输入_ (如 [损失概览](https://sbert.net/docs/sentence_transformer/loss_overview.html) 表中所指示)。这些列的数量必须与你选择的损失函数的有效输入数量相匹配。列的名称无关紧要， **只有它们的顺序重要**。

例如，如果你的损失函数接受 `(anchor, positive, negative)` 三元组，那么你的数据集的第一、第二和第三列分别对应于 `anchor` 、 `positive` 和 `negative` 。这意味着你的第一和第二列必须包含应该紧密嵌入的文本，而你的第一和第三列必须包含应该远距离嵌入的文本。这就是为什么根据你的损失函数，你的数据集列顺序很重要的原因。
考虑一个带有 `["text1", "text2", "label"]` 列的数据集，其中 `"label"` 列包含浮点数相似性得分。这个数据集可以用 `CoSENTLoss` 、 `AnglELoss` 和 `CosineSimilarityLoss` ，因为:

1. 数据集有一个“label”列，这是这些损失函数所必需的。
2. 数据集有 2 个非标签列，与这些损失函数所需的输入数量相匹配。

如果你的数据集中的列没有正确排序，请使用 [`Dataset.select_columns`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.select_columns) 来重新排序。此外，使用 [`Dataset.remove_columns`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.remove_columns) 移除任何多余的列 (例如， `sample_id` 、 `metadata` 、 `source` 、 `type` )，因为否则它们将被视为输入。

## 损失函数

损失函数衡量模型在给定数据批次上的表现，并指导优化过程。损失函数的选择取决于你可用的数据和目标任务。请参阅 [损失概览](https://sbert.net/docs/sentence_transformer/loss_overview.html) 以获取完整的选择列表。

大多数损失函数可以使用你正在训练的 `SentenceTransformer` `model` 来初始化:

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss

# Load a model to train/finetune
model = SentenceTransformer("FacebookAI/xlm-roberta-base")

# Initialize the CoSENTLoss
# This loss requires pairs of text and a floating point similarity score as a label
loss = CoSENTLoss(model)

# Load an example training dataset that works with our loss function:
train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
"""
Dataset({
    features: ['sentence1', 'sentence2', 'label'],
    num_rows: 942069
})
"""
```

## 训练参数

[`SentenceTransformersTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentencetransformertrainingarguments) 类允许你指定影响训练性能和跟踪/调试的参数。虽然这些参数是可选的，但实验这些参数可以帮助提高训练效率，并为训练过程提供洞察。

在 Sentence Transformers 的文档中，我概述了一些最有用的训练参数。我建议你阅读 [训练概览 > 训练参数](https://sbert.net/docs/sentence_transformer/training_overview.html#training-arguments) 部分。

以下是如何初始化 [`SentenceTransformersTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentencetransformertrainingarguments) 的示例:

```python
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    fp16=True, # Set to False if your GPU can't handle FP16
    bf16=False, # Set to True if your GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES, # Losses using "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet", # Used in W&B if `wandb` is installed
)
```

注意 `eval_strategy` 是在 `transformers` 版本 `4.41.0` 中引入的。之前的版本应该使用 `evaluation_strategy` 代替。

## 评估器

你可以为 [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) 提供一个 `eval_dataset` 以便在训练过程中获取评估损失，但在训练过程中获取更具体的指标也可能很有用。为此，你可以使用评估器来在训练前、中或后评估模型的性能，并使用有用的指标。你可以同时使用 `eval_dataset` 和评估器，或者只使用其中一个，或者都不使用。它们根据 `eval_strategy` 和 `eval_steps` [训练参数](#training-arguments) 进行评估。

以下是 Sentence Tranformers 随附的已实现的评估器:

| 评估器 | 所需数据 |
| --- | --- |
| [`BinaryClassificationEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#binaryclassificationevaluator) | 带有类别标签的句子对 |
| [`EmbeddingSimilarityEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#embeddingsimilarityevaluator) | 带有相似性得分的句子对 |
| [`InformationRetrievalEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#informationretrievalevaluator) | 查询（qid => 问题），语料库 (cid => 文档)，以及相关文档 (qid => 集合[cid]) |
| [`MSEEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#mseevaluator) | 需要由教师模型嵌入的源句子和需要由学生模型嵌入的目标句子。可以是相同的文本。 |
| [`ParaphraseMiningEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#paraphraseminingevaluator) | ID 到句子的映射以及带有重复句子 ID 的句子对。 |
| [`RerankingEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#rerankingevaluator) | {'query': '..', 'positive': [...], 'negative': [...]} 字典的列表。 |
| [`TranslationEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#translationevaluator) | 两种不同语言的句子对。 |
| [`TripletEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#tripletevaluator) | (锚点，正面，负面) 三元组。 |

此外，你可以使用 [`SequentialEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sequentialevaluator) 将多个评估器组合成一个，然后将其传递给 [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer)。

如果你没有必要的评估数据但仍然想跟踪模型在常见基准上的性能，你可以使用 Hugging Face 上的数据与这些评估器一起使用。

### 使用 STSb 的 Embedding Similarity Evaluator

STS 基准测试 (也称为 STSb) 是一种常用的基准数据集，用于衡量模型对短文本 (如 “A man is feeding a mouse to a snake.”) 的语义文本相似性的理解。

你可以自由浏览 Hugging Face 上的 [sentence-transformers/stsb](https://huggingface.co/datasets/sentence-transformers/stsb) 数据集。

```python
from datasets import load_dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

# Load the STSB dataset
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

# Initialize the evaluator
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
# Run evaluation manually:
# print(dev_evaluator(model))

# Later, you can provide this evaluator to the trainer to get results during training
```

### 使用 AllNLI 的 Triplet Evaluator

AllNLI 是 [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) 和 [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) 数据集的合并，这两个数据集都是用于自然语言推理的。这个任务的传统目的是确定两段文本是否是蕴含、矛盾还是两者都不是。它后来被采用用于训练嵌入模型，因为蕴含和矛盾的句子构成了有用的 `(anchor, positive, negative)` 三元组: 这是训练嵌入模型的一种常见格式。

在这个片段中，它被用来评估模型认为锚文本和蕴含文本比锚文本和矛盾文本更相似的频率。一个示例文本是 “An older man is drinking orange juice at a restaurant.”。

你可以自由浏览 Hugging Face 上的 [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) 数据集。

```python
from datasets import load_dataset
from sentence_transformers.evaluation import TripletEvaluator, SimilarityFunction

# Load triplets from the AllNLI dataset
max_samples = 1000
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split=f"dev[:{max_samples}]")

# Initialize the evaluator
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    main_distance_function=SimilarityFunction.COSINE,
    name=f"all-nli-{max_samples}-dev",
)
# Run evaluation manually:
# print(dev_evaluator(model))

# Later, you can provide this evaluator to the trainer to get results during training
```

## 训练器

[`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) 将模型、数据集、损失函数和其他组件整合在一起进行训练:

```python
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "microsoft/mpnet-base",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MPNet base trained on AllNLI triplets",
    )
)

# 3. Load a dataset to finetune on
dataset = load_dataset("sentence-transformers/all-nli", "triplet")
train_dataset = dataset["train"].select(range(100_000))
eval_dataset = dataset["dev"]
test_dataset = dataset["test"]

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    fp16=True, # Set to False if GPU can't handle FP16
    bf16=False, # Set to True if GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES, # MultipleNegativesRankingLoss benefits from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet", # Used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    name="all-nli-dev",
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

# (Optional) Evaluate the trained model on the test set, after training completes
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="all-nli-test",
)
test_evaluator(model)

# 8. Save the trained model
model.save_pretrained("models/mpnet-base-all-nli-triplet/final")

# 9. (Optional) Push it to the Hugging Face Hub
model.push_to_hub("mpnet-base-all-nli-triplet")
```

在这个示例中，我从一个尚未成为 Sentence Transformer 模型的基础模型 [`microsoft/mpnet-base`](https://huggingface.co/microsoft/mpnet-base) 开始进行微调。这需要比微调现有的 Sentence Transformer 模型，如 [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)，更多的训练数据。

运行此脚本后，[tomaarsen/mpnet-base-all-nli-triplet](https://huggingface.co/tomaarsen/mpnet-base-all-nli-triplet) 模型被上传了。使用余弦相似度的三元组准确性，即 `cosine_similarity(anchor, positive) > cosine_similarity(anchor, negative)` 的百分比为开发集上的 90.04% 和测试集上的 91.5% ！作为参考，[`microsoft/mpnet-base`](https://huggingface.co/microsoft/mpnet-base) 模型在训练前在开发集上的得分为 68.32%。

所有这些信息都被自动生成的模型卡存储，包括基础模型、语言、许可证、评估结果、训练和评估数据集信息、超参数、训练日志等。无需任何努力，你上传的模型应该包含潜在用户判断你的模型是否适合他们的所有信息。

### 回调函数

Sentence Transformers 训练器支持各种 [`transformers.TrainerCallback`](https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback) 子类，包括:

- [`WandbCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.WandbCallback): 如果已安装 `wandb` ，则将训练指标记录到 W&B
- [`TensorBoardCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.TensorBoardCallback): 如果可访问 `tensorboard` ，则将训练指标记录到 TensorBoard
- [`CodeCarbonCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.CodeCarbonCallback): 如果已安装 `codecarbon` ，则跟踪训练期间的碳排放

这些回调函数会自动使用，无需你进行任何指定，只要安装了所需的依赖项即可。

有关这些回调函数的更多信息以及如何创建你自己的回调函数，请参阅 [Transformers 回调文档](https://huggingface.co/docs/transformers/en/main_classes/callback)。

## 多数据集训练

通常情况下，表现最好的模型是通过同时使用多个数据集进行训练的。[`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) 通过允许你使用多个数据集进行训练，而不需要将它们转换为相同的格式，简化了这一过程。你甚至可以为每个数据集应用不同的损失函数。以下是多数据集训练的步骤:

1. 使用一个 [`datasets.Dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) 实例的字典 (或 [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict)) 作为 `train_dataset` 和 `eval_dataset` 。
2. (可选) 如果你希望为不同的数据集使用不同的损失函数，请使用一个损失函数的字典，其中数据集名称映射到损失。

每个训练/评估批次将仅包含来自一个数据集的样本。从多个数据集中采样批次的顺序由 [`MultiDatasetBatchSamplers`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.MultiDatasetBatchSamplers) 枚举确定，该枚举可以通过 `multi_dataset_batch_sampler` 传递给 [`SentenceTransformersTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentencetransformertrainingarguments)。有效的选项包括:

- `MultiDatasetBatchSamplers.ROUND_ROBIN` : 以轮询方式从每个数据集采样，直到一个数据集用尽。这种策略可能不会使用每个数据集中的所有样本，但它确保了每个数据集的平等采样。
- `MultiDatasetBatchSamplers.PROPORTIONAL` (默认): 按比例从每个数据集采样。这种策略确保了每个数据集中的所有样本都被使用，并且较大的数据集被更频繁地采样。

多任务训练已被证明是高度有效的。例如，[Huang et al. 2024](https://arxiv.org/pdf/2405.06932) 使用了 [`MultipleNegativesRankingLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss)、[`CoSENTLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss) 和 [`MultipleNegativesRankingLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) 的一个变体 (不包含批次内的负样本，仅包含硬负样本)，以在中国取得最先进的表现。他们还应用了 [`MatryoshkaLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) 以使模型能够产生 [Matryoshka Embeddings](https://huggingface.co/blog/matryoshka)。

以下是多数据集训练的一个示例:

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss, SoftmaxLoss

# 1. Load a model to finetune
model = SentenceTransformer("bert-base-uncased")

# 2. Loadseveral Datasets to train with
# (anchor, positive)
all_nli_pair_train = load_dataset("sentence-transformers/all-nli", "pair", split="train[:10000]")
# (premise, hypothesis) + label
all_nli_pair_class_train = load_dataset("sentence-transformers/all-nli", "pair-class", split="train[:10000]")
# (sentence1, sentence2) + score
all_nli_pair_score_train = load_dataset("sentence-transformers/all-nli", "pair-score", split="train[:10000]")
# (anchor, positive, negative)
all_nli_triplet_train = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:10000]")
# (sentence1, sentence2) + score
stsb_pair_score_train = load_dataset("sentence-transformers/stsb", split="train[:10000]")
# (anchor, positive)
quora_pair_train = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[:10000]")
# (query, answer)
natural_questions_train = load_dataset("sentence-transformers/natural-questions", split="train[:10000]")

# Combine all datasets into a dictionary with dataset names to datasets
train_dataset = {
    "all-nli-pair": all_nli_pair_train,
    "all-nli-pair-class": all_nli_pair_class_train,
    "all-nli-pair-score": all_nli_pair_score_train,
    "all-nli-triplet": all_nli_triplet_train,
    "stsb": stsb_pair_score_train,
    "quora": quora_pair_train,
    "natural-questions": natural_questions_train,
}

# 3. Load several Datasets to evaluate with
# (anchor, positive, negative)
all_nli_triplet_dev = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
# (sentence1, sentence2, score)
stsb_pair_score_dev = load_dataset("sentence-transformers/stsb", split="validation")
# (anchor, positive)
quora_pair_dev = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[10000:11000]")
# (query, answer)
natural_questions_dev = load_dataset("sentence-transformers/natural-questions", split="train[10000:11000]")

# Use a dictionary for the evaluation dataset too, or just use one dataset or none at all
eval_dataset = {
    "all-nli-triplet": all_nli_triplet_dev,
    "stsb": stsb_pair_score_dev,
    "quora": quora_pair_dev,
    "natural-questions": natural_questions_dev,
}

# 4. Load several loss functions to train with
# (anchor, positive), (anchor, positive, negative)
mnrl_loss = MultipleNegativesRankingLoss(model)
# (sentence_A, sentence_B) + class
softmax_loss = SoftmaxLoss(model)
# (sentence_A, sentence_B) + score
cosent_loss = CoSENTLoss(model)

# Create a mapping with dataset names to loss functions, so the trainer knows which loss to apply where
# Note: You can also just use one loss if all your training/evaluation datasets use the same loss
losses = {
    "all-nli-pair": mnrl_loss,
    "all-nli-pair-class": softmax_loss,
    "all-nli-pair-score": cosent_loss,
    "all-nli-triplet": mnrl_loss,
    "stsb": cosent_loss,
    "quora": mnrl_loss,
    "natural-questions": mnrl_loss,
}

# 5. Define a simple trainer, although it's recommended to use one with args & evaluators
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=losses,
)
trainer.train()

# 6. Save the trained model and optionally push it to the Hugging Face Hub
model.save_pretrained("bert-base-all-nli-stsb-quora-nq")
model.push_to_hub("bert-base-all-nli-stsb-quora-nq")
```

## 弃用

在 Sentence Transformer v3 发布之前，所有模型都会使用 [`SentenceTransformer.fit`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.fit) 方法进行训练。从 v3.0 开始，该方法将使用 [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/trainer.html#sentence_transformers.trainer.SentenceTransformerTrainer) 作为后端。这意味着你的旧训练代码仍然应该可以工作，甚至可以升级到新的特性，如多 GPU 训练、损失记录等。然而，新的训练方法更加强大，因此建议使用新的方法编写新的训练脚本。

## 附加资源

### 训练示例

以下页面包含带有解释的训练示例以及代码链接。我们建议你浏览这些页面以熟悉训练循环:

- [语义文本相似度](https://sbert.net/examples/training/sts/README.html)
- [自然语言推理](https://sbert.net/examples/training/nli/README.html)
- [释义](https://sbert.net/examples/training/paraphrases/README.html)
- [Quora 重复问题](https://sbert.net/examples/training/quora_duplicate_questions/README.html)
- [Matryoshka Embeddings](https://sbert.net/examples/training/matryoshka/README.html)
- [自适应层模型](https://sbert.net/examples/training/adaptive_layer/README.html)
- [多语言模型](https://sbert.net/examples/training/multilingual/README.html)
- [模型蒸馏](https://sbert.net/examples/training/distillation/README.html)
- [增强的句子转换器](https://sbert.net/examples/training/data_augmentation/README.html)

### 文档

此外，以下页面可能有助于你了解 Sentence Transformers 的更多信息:

- [安装](https://sbert.net/docs/installation.html)
- [快速入门](https://sbert.net/docs/quickstart.html)
- [使用](https://sbert.net/docs/sentence_transformer/usage/usage.html)
- [预训练模型](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
- [训练概览](https://sbert.net/docs/sentence_transformer/training_overview.html) (本博客是训练概览文档的提炼)
- [数据集概览](https://sbert.net/docs/sentence_transformer/dataset_overview.html)
- [损失概览](https://sbert.net/docs/sentence_transformer/loss_overview.html)
- [API 参考](https://sbert.net/docs/package_reference/sentence_transformer/index.html)

最后，以下是一些高级页面，你可能会感兴趣:

- [超参数优化](https://sbert.net/examples/training/hpo/README.html)
- [分布式训练](https://sbert.net/docs/sentence_transformer/training/distributed.html)