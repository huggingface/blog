---
title: "mmBERT: ModernBERT goes Multilingual" 
thumbnail: /blog/assets/mmbert/thumbnail.png
authors:
- user: mmarone
  guest: true
  org: jhu-clsp
- user: orionweller
  guest: true
  org: jhu-clsp
- user: will-fleshman
  guest: true
  org: jhu-clsp
- user: eugene-yang
  guest: true
  org: jhu-clsp
- user: dlawrie
  guest: true
  org: jhu-clsp
- user: vandurme
  guest: true
  org: jhu-clsp
---

# mmBERT: ModernBERT goes Multilingual

## TL;DR
This blog post introduces [mmBERT](https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder-68b725831d7c6e3acc435ed4), a state-of-the-art massively multilingual encoder model trained on 3T+ tokens of text in over 1800 languages. It shows significant performance and speed improvements over previous multilingual models, being the first to improve upon XLM-R, while also developing new strategies for effectively learning low-resource languages. mmBERT builds upon ModernBERT for a blazingly fast architecture, and adds novel components to enable efficient multilingual learning.

If you are interested in trying out the models yourself, some example boilerplate is available [at the end of this blogpost!](#usage-examples)

## Training Data
<figure class="image text-center" id="figure1">
  <img src="https://github.com/JHU-CLSP/mmBERT/blob/main/assets/data_dist.jpg?raw=true" alt="Distribution of each phase of pre-training">
  <figcaption> Figure 1: the training data is progressively annealed to include more languages and more uniform sampling throughout training.</figcaption>
</figure>

mmBERT was trained on a carefully curated multilingual dataset totaling over 3T tokens across three distinct training phases. The foundation of our training data consists of three primary open-source and high-quality web crawls that enable both multilingual coverage and data quality:

**DCLM and Filtered DCLM** provides the highest quality English content available, serving as the backbone for strong English performance (with the filtered data coming from [Dolmino](https://huggingface.co/datasets/allenai/dolmino-mix-1124)). This dataset represents state-of-the-art web filtering techniques and forms a crucial component. Due to the high quality of this data, we use a signficantly higher proportion of English than previous generation multilingual encoder models (up to 18%).

**FineWeb2** delivers broad [multilingual web content](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) covering over 1,800 languages. This dataset enables our extensive multilingual coverage while maintaining reasonable quality standards across diverse language families and scripts.

**FineWeb2-HQ** consists of a [filtered subset of FineWeb2](https://huggingface.co/datasets/epfml/FineWeb2-HQ) focusing on 20 high-resource languages. This filtered version provides higher-quality multilingual content that bridges the gap between English-only filtered data and broad multilingual coverage.

The training data also incorporates specialized corpora from [Dolma](https://arxiv.org/abs/2402.00159), [MegaWika v2](https://arxiv.org/abs/2508.03828), [ProLong](https://arxiv.org/abs/2410.02660) and more: code repositories (StarCoder, ProLong), academic content (ArXiv, PeS2o), reference materials (Wikipedia, textbooks), and community discussions (StackExchange), along with instruction and mathematical datasets.

The key innovation in our data approach is the **progressive language inclusion strategy** shown in [Figure 1](#figure1). At each phase we progressively sample from a *flatter* distribution (i.e. closer to uniform), while also adding new languages. This means that high resource languages like Russian start off with a high percentage of the data (i.e. 9%) and then in the last phase of training end around half of that. We start with 60 high-resource languages during pre-training, expand to 110 languages during mid-training, and finally include all 1,833 languages from FineWeb2 during the decay phase. This allows us to maximize the impact of limited low-resource language data without excess reptitions and while maintaining high overall data quality.

## Training Recipe and Novel Components

mmBERT builds upon the [ModernBERT](https://huggingface.co/blog/modernbert) architecture but introduces several key innovations for multilingual learning:

### Architecture
We use the same core architecture as ModernBERT-base with 22 layers and 1152 intermediate dimensions, but switch to the Gemma 2 tokenizer to better handle multilingual text. The base model has 110M non-embedding parameters (307M total due to the larger vocabulary), while the small variant has 42M non-embedding parameters (140M total).

### Three-Phase Training Approach
Our training follows a carefully designed three-phase schedule:

1. **Pre-training (2.3T tokens)**: Warmup and stable learning rate phase using 60 languages with 30% mask rate
2. **Mid-training (600B tokens)**: Context extension to 8192 tokens, higher-quality data, expanded to 110 languages with 15% mask rate  
3. **Decay phase (100B tokens)**: Inverse square root learning rate decay, all 1,833 languages included with 5% mask rate

### Novel Training Techniques

**Inverse Mask Ratio Schedule**: Instead of using a fixed masking rate, we progressively reduce the mask ratio from 30% → 15% → 5% across training phases. This allows the model to learn basic representations with higher masking early on, then focus on more nuanced understanding with lower masking rates.

**Annealed Language Learning**: We dynamically adjust the temperature for multilingual data sampling from τ=0.7 → 0.5 → 0.3. This creates a progression from high-resource language bias toward more uniform sampling, enabling the model to build a strong multilingual foundation before learning low-resource languages. 

**Progressive Language Addition**: Rather than training on all languages simultaneously, we strategically add languages at each phase (60 → 110 → 1,833). This maximizes learning efficiency by avoiding excessive epochs on limited low-resource data while still achieving strong performance.

**Model Merging**: We train three different variants during the decay phase (English-focused, 110-language, and all-language) and use TIES merging to combine their strengths into the final model.

## Results

### Natural Language Understanding (NLU)

<figure class="image text-center" id="table1">
  <img src="https://github.com/JHU-CLSP/mmBERT/blob/main/assets/glue.jpeg?raw=true" alt="GLUE performance">
  <figcaption>Table 1: Performance on GLUE (English)</figcaption>
</figure>

**English Performance**: On the English GLUE benchmark ([Table 1](#table1)), mmBERT base achieves strong performance, substantially outperforming other multilingual models like XLM-R (multilingual RoBERTa) base and mGTE base, while remaining competitive to English-only models despite less than 25% of the mmBERT training data being English.

<figure class="image text-center" id="table2">
  <img src="https://github.com/JHU-CLSP/mmBERT/blob/main/assets/xtreme.jpeg?raw=true" alt="XTREME performance">
  <figcaption>Table 2: Performance on XTREME (Multilingual)</figcaption>
</figure>

**Multilingual Performance**: mmBERT shows significant improvements on XTREME benchmark compared to XLM-R as demonstrated in [Table 2](#table2). Notable gains include strong performance on XNLI classification, substantial improvements in question answering tasks like TyDiQA, and competitive results across PAWS-X and XCOPA for cross-lingual understanding.

The model performs well across most categories, with the exception of some structured prediction tasks like NER and POS tagging, likely due to tokenizer differences that affect word boundary detection. On these categories, it performs about the same as the previous generation, but can be applied to more languages.

### Retrieval Performance

<figure class="image text-center" id="table3">
  <img src="https://github.com/JHU-CLSP/mmBERT/blob/main/assets/mteb-english.jpeg?raw=true" alt="MTEB v2 Eng performance">
  <figcaption>Table 3: Performance on MTEB v2 English</figcaption>
</figure>

**English Retrieval**: Even though mmBERT is designed for massively multilingual settings, in the MTEB v2 English benchmarks ([Table 3](#table3)), mmBERT shows significant gains over previous multilingual models and even ties the capabilities of English-only models like ModernBERT!

<figure class="image text-center" id="table4">
  <img src="https://github.com/JHU-CLSP/mmBERT/blob/main/assets/mteb-multilingual.jpeg?raw=true" alt="MTEB v2 Multilingual performance">
  <figcaption>Table 4: Performance on MTEB v2 Multilingual</figcaption>
</figure>

**Multilingual Retrieval**: mmBERT shows consistent improvements on MTEB v2 multilingual benchmarks compared to other models ([Table 4](#table4)).

<figure class="image text-center" id="table5">
  <img src="https://github.com/JHU-CLSP/mmBERT/blob/main/assets/coir.jpeg?raw=true" alt="CoIR performance">
  <figcaption>Table 5: Performance on CoIR code benchmark</figcaption>
</figure>

**Code Retrieval**: Due to the modern tokenizer (based on Gemma 2) mmBERT also shows strong coding performance ([Table 5](#table5)), making mmBERT suitable for any type of textual data. The only model that outperforms it is EuroBERT, which was able to use the non-publicly accessible Stack v2 dataset.

## Learning Languages in the Decay Phase

One of mmBERT's most significant novel features is demonstrating that low-resource languages can be effectively learned during the short decay phase of training. We validated this approach by testing on languages only introduced during the final 100B token decay phase.

<figure class="image text-center" id="figure2">
  <img src="https://github.com/JHU-CLSP/mmBERT/blob/main/assets/low_resource_merge.jpg?raw=true" alt="Improvements from adding new languages in the decay phase">
  <figcaption> Figure 2: adding more than 1700 languages in the decay phase allows for rapid learning, which we keep through model merging.</figcaption>
</figure>


**Dramatic Performance Gains**: Testing on TiQuaD (Tigray) and FoQA (Faroese), we observed substantial improvements when these languages were included in the decay phase, as shown in [Figure 2](#figure2). The results demonstrate the effectiveness of our progressive language learning approach.

**Competitive with Large Models**: Despite only seeing these languages in the final training phase, mmBERT achieves performance levels that exceed much larger models. On Faroese question answering where LLMs have been benchmarked, mmBERT outperforms Google Gemini 2.5 Pro and OpenAI o3.

**Rapid Learning Mechanism**: The success of decay-phase language learning stems from the model's ability to leverage its strong multilingual foundation built during earlier phases. When exposed to new languages, the model can quickly adapt existing cross-lingual representations rather than learning from scratch.

**Model Merging Benefits**: The final mmBERT models successfully retain most of the decay-phase improvements while benefiting from the English-focused and high-resource variants through TIES merging.
## Efficiency Improvements

mmBERT delivers substantial efficiency gains over previous multilingual encoder models through architectural improvements inherited from ModernBERT:

<figure class="image text-center" id="figure3">
  <img src="https://github.com/JHU-CLSP/mmBERT/blob/main/assets/inference_times.jpg?raw=true" alt="mmBERT is much more efficient than previous multilingual models">
  <figcaption> Figure 3: mmBERT is significantly more efficient than previous multilingual models, up to 2-4x as much!</figcaption>
</figure>

**Throughput Performance**: mmBERT processes text significantly faster than existing multilingual models across various sequence lengths, as demonstrated in [Figure 3](#figure3). Both the small and base models show substantial speed improvements over previous multilingual encoders.

**Modern Architecture Benefits**: The efficiency gains come from two main technical improvements:
- **Flash Attention 2**: Optimized attention computation for better memory usage and speed
- **Unpadding techniques**: Elimination of unnecessary padding tokens during processing

**Sequence Length Scaling**: Unlike older models limited to 512 tokens, mmBERT handles up to 8,192 tokens efficiently while maintaining high throughput. This makes it suitable for longer document processing tasks that are increasingly common in multilingual applications.

**Energy Efficiency**: The combination of better throughput and modern architecture results in lower computational costs for inference, making mmBERT more practical for production deployments where multilingual support is needed at scale.

These efficiency improvements make mmBERT not just more accurate than previous multilingual encoders, but also significantly more practical for real usage.

## Usage Examples
You can use these models with just a few lines of code!

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
model = AutoModelForMaskedLM.from_pretrained("jhu-clsp/mmBERT-base")

def predict_masked_token(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    mask_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
    predictions = outputs.logits[mask_indices]
    top_tokens, top_indices = torch.topk(predictions, 5, dim=-1)
    return [tokenizer.decode(token) for token in top_indices[0]]

# Works across languages
texts = [
    "The capital of France is <mask>.",
    "La capital de España es <mask>.",
    "Die Hauptstadt von Deutschland ist <mask>.",
]

for text in texts:
    predictions = predict_masked_token(text)
    print(f"Text: {text}")
    print(f"Predictions: {predictions}\n")
```

## Fine-tuning Examples

### Encoders
<details><summary>Click to see how to finetune this into a dense embedding model using Sentence Transformers</summary> 

```python
import argparse

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

def main():
    # parse the lr & model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-small")
    args = parser.parse_args()
    lr = args.lr
    model_name = args.model_name
    model_shortname = model_name.split("/")[-1]

    # 1. Load a model to finetune
    model = SentenceTransformer(model_name)

    # 2. Load a dataset to finetune on
    dataset = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"].select(range(1_250_000))
    eval_dataset = dataset_dict["test"]

    # 3. Define a loss function
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)  # Increase mini_batch_size if you have enough VRAM

    run_name = f"{model_shortname}-DPR-{lr}"
    # 4. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"output/{model_shortname}/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
        warmup_ratio=0.05,
        fp16=False,  # Set to False if GPU can't handle FP16
        bf16=True,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicates
        learning_rate=lr,
        # Optional tracking/debugging parameters:
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=500,
        run_name=run_name,  # Used in `wandb`, `tensorboard`, `neptune`, etc. if installed
    )

    # 5. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )
    dev_evaluator(model)

    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. (Optional) Evaluate the trained model on the evaluator after training
    dev_evaluator(model)

    # 8. Save the model
    model.save_pretrained(f"output/{model_shortname}/{run_name}/final")

    # 9. (Optional) Push it to the Hugging Face Hub
    model.push_to_hub(run_name, private=False)

if __name__ == "__main__":
    main()
```
</details>


<details><summary>Click to see how to finetune this into a multi-vector embedding model with PyLate</summary>

```python
from datasets import load_dataset
from pylate import losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

def main():
    # Load the datasets required for knowledge distillation (train, queries, documents)
    train = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="train",
    )

    queries = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="queries",
    )

    documents = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="documents",
    )

    # Set the transformation to load the documents/queries texts using the corresponding ids on the fly
    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents).transform,
    )

    # Define the base model, training parameters, and output directory
    num_train_epochs = 1
    lr = 8e-5
    batch_size = 16
    accum_steps = 1
    model_name = "jhu-clsp/mmBERT-small"
    model_shortname = model_name.split("/")[-1]

    # Set the run name for logging and output directory
    run_name = f"{model_shortname}-colbert-KD-{lr}"
    output_dir = f"output/{model_shortname}/{run_name}"

    # Initialize the ColBERT model from the base model
    model = models.ColBERT(model_name_or_path=model_name)

    # Configure the training arguments (e.g., epochs, batch size, learning rate)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        run_name=run_name,
        logging_steps=10,
        learning_rate=lr,
        gradient_accumulation_steps=accum_steps,
        warmup_ratio=0.05,
    )

    # Use the Distillation loss function for training
    train_loss = losses.Distillation(model=model)

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    # Start the training process
    trainer.train()

    model.save_pretrained(f"{output_dir}/final")

if __name__ == "__main__":
    main()

```
</details>

<details><summary>Click to see how to finetune this into a sparse retrieval model using Sentence Transformers</summary>

```python
import logging

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# 1. Load a model to finetune with 2. (Optional) model card data
model = SparseEncoder(
    "jhu-clsp/mmBERT-small",
    model_card_data=SparseEncoderModelCardData(
        language="en",
        license="apache-2.0",
    )
)

# 3. Load a dataset to finetune on
full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

# 4. Define a loss function
loss = SpladeLoss(
    model=model,
    loss=SparseMultipleNegativesRankingLoss(model=model),
    query_regularizer_weight=5e-5,
    document_regularizer_weight=3e-5,
)

# 5. (Optional) Specify training arguments
run_name = "splade-distilbert-base-uncased-nq"
args = SparseEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
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
</details>

<details><summary>Click to see how to finetune this into a reranker model using Sentence Transformers</summary>

```python
import logging
import traceback

import torch
from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderNanoBEIREvaluator,
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.util import mine_hard_negatives

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name = "jhu-clsp/mmBERT-small"

    train_batch_size = 64
    num_epochs = 1
    num_hard_negatives = 5  # How many hard negatives should be mined for each question-answer pair

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    model = CrossEncoder(
        model_name,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
        ),
    )
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2a. Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
    logging.info("Read the gooaq training dataset")
    full_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 2b. Modify our training dataset to include hard negatives using a very efficient embedding model
    embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
    hard_train_dataset = mine_hard_negatives(
        train_dataset,
        embedding_model,
        num_negatives=num_hard_negatives,  # How many negatives per question-answer pair
        margin=0,  # Similarity between query and negative samples should be x lower than query-positive similarity
        range_min=0,  # Skip the x most similar samples
        range_max=100,  # Consider only the x most similar samples
        sampling_strategy="top",  # Sample the top negatives from the range
        batch_size=4096,  # Use a batch size of 4096 for the embedding model
        output_format="labeled-pair",  # The output format is (query, passage, label), as required by BinaryCrossEntropyLoss
        use_faiss=True,
    )
    logging.info(hard_train_dataset)

    # 2c. (Optionally) Save the hard training dataset to disk
    # hard_train_dataset.save_to_disk("gooaq-hard-train")
    # Load again with:
    # hard_train_dataset = load_from_disk("gooaq-hard-train")

    # 3. Define our training loss.
    # pos_weight is recommended to be set as the ratio between positives to negatives, a.k.a. `num_hard_negatives`
    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(num_hard_negatives))

    # 4a. Define evaluators. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking
    nano_beir_evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"],
        batch_size=train_batch_size,
    )

    # 4b. Define a reranking evaluator by mining hard negatives given query-answer pairs
    # We include the positive answer in the list of negatives, so the evaluator can use the performance of the
    # embedding model as a baseline.
    hard_eval_dataset = mine_hard_negatives(
        eval_dataset,
        embedding_model,
        corpus=full_dataset["answer"],  # Use the full dataset as the corpus
        num_negatives=30,  # How many documents to rerank
        batch_size=4096,
        include_positives=True,
        output_format="n-tuple",
        use_faiss=True,
    )
    logging.info(hard_eval_dataset)
    reranking_evaluator = CrossEncoderRerankingEvaluator(
        samples=[
            {
                "query": sample["question"],
                "positive": [sample["answer"]],
                "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
            }
            for sample in hard_eval_dataset
        ],
        batch_size=train_batch_size,
        name="gooaq-dev",
        # Realistic setting: only rerank the positives that the retriever found
        # Set to True to rerank *all* positives
        always_rerank_positives=False,
    )

    # 4c. Combine the evaluators & run the base model on them
    evaluator = SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])
    evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-gooaq-bce"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_gooaq-dev_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=200,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=hard_train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()

```
</details>

## Model Family and Links

**Standard Models:**
- [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) (140M params total, 42M non-embed)
- [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (307M params total, 110M non-embed)

**Research Resources:**
- [🤗 mmBERT Model Collection](https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder-68b725831d7c6e3acc435ed4)
- [📝 Paper](https://arxiv.org/abs/2509.06888)  
- [🗂️ Training Data](https://huggingface.co/datasets/jhu-clsp/mmbert-pretrain-p1-fineweb2-langs) (3T+ tokens, fully open)
- [💻 GitHub Repository](https://github.com/jhu-clsp/mmBERT)
- [📊 Training Checkpoints](https://huggingface.co/jhu-clsp/mmBERT-checkpoints) for studying training or continued pre-training
