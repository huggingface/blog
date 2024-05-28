---
title: "Training and Finetuning Embedding Models with Sentence Transformers v3"
thumbnail: /blog/assets/train-sentence-transformers/st-hf-thumbnail.png
authors:
- user: tomaarsen
---

# Training and Finetuning Embedding Models with Sentence Transformers v3

[Sentence Transformers](https://sbert.net/) is a Python library for using and training embedding models for a wide range of applications, such as retrieval augmented generation, semantic search, semantic textual similarity, paraphrase mining, and more. Its v3.0 update is the largest since the project's inception, introducing a new training approach. In this blogpost, I'll show you how to use it to finetune Sentence Transformer models to improve their performance on specific tasks. You can also use this method to train new Sentence Transformer models from scratch.

Finetuning Sentence Transformers now involves several components, including datasets, loss functions, training arguments, evaluators, and the new trainer itself. I'll go through each of these components in detail and provide examples of how to use them to train effective models.

## Why Finetune?

Finetuning Sentence Transformer models can significantly enhance their performance on specific tasks. This is because each task requires a unique notion of similarity. Let's consider a couple of news article headlines as an example:

- "Apple launches the new iPad"
- "NVIDIA is gearing up for the next GPU generation"

Depending on the use case, we might want similar or dissimilar embeddings for these texts. For instance, a classification model for news articles could treat these texts as similar since they both belong to the Technology category. On the other hand, a semantic textual similarity or retrieval model should consider them dissimilar due to their distinct meanings.

## Training Components

Training Sentence Transformer models involves the following components:

1. [**Dataset**](#dataset): The data used for training and evaluation.
2. [**Loss Function**](#loss-function): A function that quantifies the model's performance and guides the optimization process.
3. [**Training Arguments**](#training-arguments) (optional): Parameters that influence training performance and tracking/debugging.
4. [**Evaluator**](#evaluator) (optional): A tool for evaluating the model before, during, or after training.
5. [**Trainer**](#trainer): Brings together the model, dataset, loss function, and other components for training.

Now, let's dive into each of these components in more detail.

## Dataset

The [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) uses [`datasets.Dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) or [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict) instances for training and evaluation. You can load data from the Hugging Face Datasets Hub or use local data in various formats such as CSV, JSON, Parquet, Arrow, or SQL.

Note: Many Hugging Face datasets that work out of the box with Sentence Transformers have been tagged with `sentence-transformers`, allowing you to easily find them by browsing to [https://huggingface.co/datasets?other=sentence-transformers](https://huggingface.co/datasets?other=sentence-transformers). We strongly recommend that you browse these datasets to find training datasets that might be useful for your tasks.

### Data on Hugging Face Hub

To load data from datasets in the Hugging Face Hub, use the [`load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset) function:

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

Some datasets, like [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli), have multiple subsets with different data formats. You need to specify the subset name along with the dataset name.

### Local Data (CSV, JSON, Parquet, Arrow, SQL)

If you have local data in common file formats, you can easily load it using [`load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset) too:

```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files="my_file.csv")
# or
dataset = load_dataset("json", data_files="my_file.json")
```

### Local Data that requires pre-processing

If your local data requires pre-processing, you can use [`datasets.Dataset.from_dict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.from_dict) to initialize your dataset with a dictionary of lists:

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

Each key in the dictionary becomes a column in the resulting dataset.

### Dataset Format

It's crucial to ensure that your dataset format matches your chosen [loss function](#loss-function). This involves checking two things:

1. If your loss function requires a *Label* (as indicated in the [Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html) table), your dataset must have a column named **"label"** or **"score"**.
2. All columns other than **"label"** or **"score"** are considered *Inputs* (as indicated in the [Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html) table). The number of these columns must match the number of valid inputs for your chosen loss function. The names of the columns don't matter, **only their order matters**.
    
For example, if your loss function accepts `(anchor, positive, negative) triplets`, then your first, second, and third dataset columns correspond with `anchor`, `positive`, and `negative`, respectively. This means that your first and second column must contain texts that should embed closely, and that your first and third column must contain texts that should embed far apart. That is why depending on your loss function, your dataset column order matters.

Consider a dataset with columns `["text1", "text2", "label"]`, where the `"label"` column contains floating point similarity scores. This dataset can be used with `CoSENTLoss`, `AnglELoss`, and `CosineSimilarityLoss` because:

1. The dataset has a "label" column, which is required by these loss functions.
2. The dataset has 2 non-label columns, matching the number of inputs required by these loss functions.

If the columns in your dataset are not ordered correctly, use [`Dataset.select_columns`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.select_columns) to reorder them. Additionally, remove any extraneous columns (e.g., `sample_id`, `metadata`, `source`, `type`) using [`Dataset.remove_columns`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.remove_columns), as they will be treated as inputs otherwise.

## Loss Function

Loss functions measure how well a model performs on a given batch of data and guide the optimization process. The choice of loss function depends on your available data and target task. Refer to the [Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html) for a comprehensive list of options.

Most loss functions can be initialized with just the `SentenceTransformer` `model` that you're training:

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

## Training Arguments

The [`SentenceTransformersTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentencetransformertrainingarguments) class allows you to specify parameters that influence training performance and tracking/debugging. While optional, experimenting with these arguments can help improve training efficiency and provide insights into the training process.

In the Sentence Transformers documentation, I've outlined some of the most useful training arguments. I would recommend reading it in [Training Overview > Training Arguments](https://sbert.net/docs/sentence_transformer/training_overview.html#training-arguments).

Here's an example of how to initialize [`SentenceTransformersTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentencetransformertrainingarguments):

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
    run_name="mpnet-base-all-nli-triplet",  # Used in W&B if `wandb` is installed
```

Note that `eval_strategy` was introduced in `transformers` version `4.41.0`. Prior versions should use `evaluation_strategy` instead.

## Evaluator

You can provide the [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) with an `eval_dataset` to get the evaluation loss during training, but it may be useful to get more concrete metrics during training, too. For this, you can use evaluators to assess the model's performance with useful metrics before, during, or after training. You can both an `eval_dataset` and an evaluator, one or the other, or neither. They evaluate based on the `eval_strategy` and `eval_steps` [Training Arguments](#training-arguments).

Here are the implemented Evaluators that come with Sentence Tranformers:

| Evaluator | Required Data |
| --- | --- |
| [`BinaryClassificationEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#binaryclassificationevaluator) | Pairs with class labels |
| [`EmbeddingSimilarityEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#embeddingsimilarityevaluator) | Pairs with similarity scores |
| [`InformationRetrievalEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#informationretrievalevaluator) | Queries (qid => question), Corpus (cid => document), and relevant documents (qid => set[cid]) |
| [`MSEEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#mseevaluator) | Source sentences to embed with a teacher model and target sentences to embed with the student model. Can be the same texts. |
| [`ParaphraseMiningEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#paraphraseminingevaluator) | Mapping of IDs to sentences & pairs with IDs of duplicate sentences. |
| [`RerankingEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#rerankingevaluator) | List of {'query': '..', 'positive': [...], 'negative': [...]} dictionaries. |
| [`TranslationEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#translationevaluator) | Pairs of sentences in two separate languages. |
| [`TripletEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#tripletevaluator) | (anchor, positive, negative) pairs. |

Additionally, you can use [`SequentialEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sequentialevaluator) to combine multiple evaluators into one, which can then be passed to the [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer).

If you don't have the necessary evaluation data but still want to track the model's performance on common benchmarks, you can use these evaluators with data from Hugging Face:

### EmbeddingSimilarityEvaluator with STSb

The STS Benchmark (a.k.a. STSb) is a commonly used benchmarking dataset to measure the model's understanding of semantic textual similarity of short texts like "A man is feeding a mouse to a snake.".

Feel free to browse the [sentence-transformers/stsb](https://huggingface.co/datasets/sentence-transformers/stsb) dataset on Hugging Face.

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
# Run evaluation:
# dev_evaluator(model)
```

### TripletEvaluator with AllNLI

AllNLI is a concatenation of the [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) datasets, both of which are datasets for Natural Language Inference. This task is traditionally for determining whether two texts are an entailment, contradiction, or neither. It has since been adopted for training embedding models, as the entailing and contradictory sentences make for useful `(anchor, positive, negative)` triplets: a common format for training embedding models. 

In this snippet, it is used to evaluate how frequently the model considers the anchor text and the entailing text to be more similar than the anchor text and the contradictory text. An example text is "An older man is drinking orange juice at a restaurant.".

Feel free to browse the [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) dataset on Hugging Face.

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
# Run evaluation:
# dev_evaluator(model)
```

## Trainer

The [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) brings together the model, dataset, loss function, and other components for training:

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
    fp16=True,  # Set to False if GPU can't handle FP16
    bf16=False,  # Set to True if GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet",  # Used in W&B if `wandb` is installed
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

In this example I'm finetuning from [`microsoft/mpnet-base`](https://huggingface.co/microsoft/mpnet-base), a base model that is not yet a Sentence Transformer model. This requires more training data than finetuning an existing Sentence Transformer model, like [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).

After running this script, the [tomaarsen/mpnet-base-all-nli-triplet](https://huggingface.co/tomaarsen/mpnet-base-all-nli-triplet) model was uploaded for me. The triplet accuracy using cosine similarity, i.e. what percentage of the time `cosine_similarity(anchor, positive) > cosine_similarity(anchor, negative)` is 90.04% for the development set and 91.5% for the testing set! For reference, the [`microsoft/mpnet-base`](https://huggingface.co/microsoft/mpnet-base) model scored only 68.32% on the dev set before training.

All of this information is stored in the automatically generated model card, including the base model, maximum sequence length, output dimensionality, similarity function, training dataset, language, license, usage, evaluation results, training & evaluation dataset (dataset name, size, approximate statistics, examples), hyperparameters, training logs, environmental impact, training hardware, framework versions, and citations. Without any effort, your uploaded models should contain all the information that your potential users would need to determine whether your model is suitable for them.

### Callbacks

The Sentence Transformers trainer supports various [`transformers.TrainerCallback`](https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback) subclasses, including:

- [`WandbCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.WandbCallback) for logging training metrics to W&B if `wandb` is installed
- [`TensorBoardCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.TensorBoardCallback) for logging training metrics to TensorBoard if `tensorboard` is accessible
- [`CodeCarbonCallback`](https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.integrations.CodeCarbonCallback) for tracking carbon emissions during training if `codecarbon` is installed

Refer to the [Transformers Callbacks documentation](https://huggingface.co/docs/transformers/en/main_classes/callback) for more information on these callbacks and how to create your own.

## Multi-Dataset Training

Top-performing models are often trained using multiple datasets simultaneously. The [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) simplifies this process by allowing you to train with multiple datasets without converting them to the same format. You can even apply different loss functions to each dataset. Here are the steps for multi-dataset training:

1. Use a dictionary of [`datasets.Dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) instances (or a [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict)) as the `train_dataset` and `eval_dataset`.
2. (Optional) Use a dictionary of loss functions mapping dataset names to losses if you want to use different losses for different datasets.

Each training/evaluation batch will contain samples from only one of the datasets. The order in which batches are sampled from the multiple datasets is determined by the [`MultiDatasetBatchSamplers`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.MultiDatasetBatchSamplers) enum, which can be passed to the [`SentenceTransformersTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentencetransformertrainingarguments) via `multi_dataset_batch_sampler`. The valid options are:

- `MultiDatasetBatchSamplers.ROUND_ROBIN`: Samples from each dataset in a round-robin fashion until one is exhausted. This strategy may not use all samples from each dataset, but it ensures equal sampling from each dataset.
- `MultiDatasetBatchSamplers.PROPORTIONAL` (default): Samples from each dataset proportionally to its size. This strategy ensures that all samples from each dataset are used, and larger datasets are sampled from more frequently.

Multi-task training has proven to be highly effective. For instance, [Huang et al. 2024](https://arxiv.org/pdf/2405.06932) employed [`MultipleNegativesRankingLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss), [`CoSENTLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss), and a variation of [`MultipleNegativesRankingLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) without in-batch negatives and only hard negatives to achieve state-of-the-art performance on Chinese. They also applied [`MatryoshkaLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) to enable the model to produce [Matryoshka Embeddings](https://huggingface.co/blog/matryoshka).

Here's an example of multi-dataset training:

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

## Deprecation

Prior to the Sentence Transformer v3 release, all models would be trained using the [`SentenceTransformer.fit`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.fit) method. Rather than deprecating this method, starting from v3.0, this method will use the [`SentenceTransformerTrainer`](https://sbert.net/docs/package_reference/sentence_transformer/trainer.html#sentence_transformers.trainer.SentenceTransformerTrainer) behind the scenes. This means that your old training code should still work, and should even be upgraded with the new features such as multi-gpu training, loss logging, etc. That said, the new training approach is much more powerful, so it is **recommended** to write new training scripts using the new approach.

## Training Examples

The following pages contain training examples with explanations as well as links to code. We recommend that you browse through these to familiarize yourself with the training loop:

* [Semantic Textual Similarity](https://sbert.net/examples/training/sts/README.html)
* [Natural Language Inference](https://sbert.net/examples/training/nli/README.html)
* [Paraphrases](https://sbert.net/examples/training/paraphrases/README.html)
* [Quora Duplicate Questions](https://sbert.net/examples/training/quora_duplicate_questions/README.html)
* [Matryoshka Embeddings](https://sbert.net/examples/training/matryoshka/README.html)
* [Adaptive Layer Models](https://sbert.net/examples/training/adaptive_layer/README.html)
* [Multilingual Models](https://sbert.net/examples/training/multilingual/README.html)
* [Model Distillation](https://sbert.net/examples/training/distillation/README.html)
* [Augmented Sentence Transformers](https://sbert.net/examples/training/data_augmentation/README.html)

## Documentation

Additionally, the following pages may be useful to learn more about Sentence Transformers:

* [Installation](https://sbert.net/docs/installation.html)
* [Quickstart](https://sbert.net/docs/quickstart.html)
* [Usage](https://sbert.net/docs/sentence_transformer/usage/usage.html)
* [Pretrained Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
* [Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html) (This blogpost is a distillation of the Training Overiew documentation)
* [Dataset Overview](https://sbert.net/docs/sentence_transformer/dataset_overview.html)
* [Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html)
* [API Reference](https://sbert.net/docs/package_reference/sentence_transformer/index.html)

And lastly, here are some advanced pages that might interest you:

* [Hyperparameter Optimization](https://sbert.net/examples/training/hpo/README.html)
* [Distributed Training](https://sbert.net/docs/sentence_transformer/training/distributed.html)