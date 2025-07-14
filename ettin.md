---
title: "Ettin Suite: SoTA Paired Encoders and Decoders" 
thumbnail: /blog/assets/ettin/thumbnail.png
authors:
- user: orionweller
  guest: true
  org: jhu-clsp
- user: kdricci
  guest: true
  org: jhu-clsp
- user: mmarone
  guest: true
  org: jhu-clsp
- user: NohTow
  guest: true
  org: lightonai
- user: dlawrie
  guest: true
  org: jhu-clsp
- user: vandurme
  guest: true
  org: jhu-clsp
---

# Ettin Suite: SoTA Paired Encoders and Decoders

## TL;DR

What would happen if you took the ModernBERT recipe and applied it to a decoder-only model? Turns out, a state-of-the-art decoder language model that beats Llama 3.2 1B and SmolLM2! 

We introduce a new open-data training recipe to reproduce the encoder-only ModernBERT model (and actually beat it!). We then apply the exact same recipe to decoder-only models. For the first time, we have two state-of-the-art models trained in the same setup but with two different training objectives: masked language modeling (MLM), and causal language modeling (CLM).

This blog post introduces [Ettin](https://huggingface.co/collections/jhu-clsp/encoders-vs-decoders-the-ettin-suite-686303e16142257eed8e6aeb), the first suite of SoTA **paired encoder-only and decoder-only models** (17M-1B params) trained with identical data (2T tokens), architecture, and training recipes. Ettin enables true apples-to-apples comparisons between architectures and delivers **state-of-the-art performance for open-data models** in both categories. We then further explore whether it is possible to get a competitive encoder starting from the decoder and vice-versa.

![Attention patterns comparison between encoder and decoder models](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder/blob/main/assets/attention_masks.png?raw=true)

## Encoders vs Decoders: The Architecture Divide

The LLM community has largely converged on decoder-only models like GPT, Llama, and Qwen. Their generative capabilities are impressive, but this focus is detracting attention from other categories, such as encoder-only models like BERT.

However, encoder BERT-like models remain the workhorses of production systems for classification, retrieval, and embedding tasks. They're faster, more memory-efficient, and often more accurate for discriminative tasks. The key difference lies in their attention patterns:

- **Encoder models** use bidirectional attention, allowing each token to "see" all other tokens in the sequence (fully visible)
- **Decoder models** use causal attention, where tokens can only "see" previous tokens to enable autoregressive generation

While decoder models have seen rapid innovation, encoder model development had stagnated – until recently, with efforts like [ModernBERT](https://huggingface.co/blog/modernbert) modernizing them. But which architecture is better? Previous comparisons between encoders and decoders used different datasets, architectures, and training recipes, so it was hard to tell.

Named after the two-headed Norse giant, Ettin provides a **controlled comparison** by training with both architectures on identical data, identical model shapes, and identical training recipes. They only differ in attention patterns and training objectives!

## Training Recipe: Modern Techniques for Both Architectures
We build on the ModernBERT recipe, which borrowed modern techniques from decoder-only models and brought them to encoder training. This provides a strong base for training both architectures.

### Sizes
We train six different sizes, ranging from 17M to 1B parameters. This allows us to test the effects of scale, and provides a wide variety of models for you to use!
No matter if you need a blazing fast on-device model or a powerful but slower model, we got you covered!

![Sizes of Ettin models](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder/blob/main/assets/sizes_small.png?raw=true)

### Three-Phase Training Process

We use a comprehensive three-phase training approach to maximize performance:

**Phase 1 - Pre-training (1.7T tokens)**: We start with a diverse mixture of high-quality data sources, training on shorter contexts (1024 tokens) to establish strong foundational knowledge.

**Phase 2 - Context Extension (250B tokens)**: We increase context length to 8K tokens using higher-quality filtered data, allowing models to understand longer documents and more complex relationships.

**Phase 3 - Decay (100B tokens)**: We finish with premium data sources including scientific papers, textbooks, and curated content while gradually reducing the learning rate.

### Modern Architecture Components
Our encoder models gain all the benefits of ModernBERT's speed, allowing them to be significantly faster than the previous generations of encoders.

### Data Sources and Quality

Unlike ModernBERT, **all our training data is public and reproducible**:

![Data used to train Ettin models](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder/blob/main/assets/training_data.jpg?raw=true)

You can continue to train these models on new data or propose a new recipe to further improve results!

## Encoder Results: Beating ModernBERT

Our encoder models **outperform ModernBERT** across all tasks and model sizes, while using completely open training data. Since we provide a large range of sizes, you can now use ModernBERT-style models in smaller sizes (great for on-device or for fast-inference), or power up with a 1B-sized encoder that crushes the competition.

![Encoder performance comparison showing Ettin models beating ModernBERT](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder/blob/main/assets/encoder_results.jpg?raw=true)


## Decoder Results: Beating Llama 3.2 and SmolLM2

Applying the same recipe to decoder models yields equally impressive results, with our models **outperforming or matching established baselines** such as Llama 3.2 and SmolLM2:

![Decoder performance comparison showing Ettin models beating Llama 3.2 and SmolLM2](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder/blob/main/assets/decoder_results.jpg?raw=true)

The gains are particularly strong on knowledge-intensive tasks like SciQ, reflecting the benefits of our high-quality training data mixture. These results demonstrate that our training recipe creates genuinely strong models in both architectural paradigms.

## Fair Fight: Encoders vs Decoders on Even Ground

For the first time, we can fairly compare encoder and decoder architectures trained with identical data and recipes. The results reveal fundamental architectural advantages that persist even when all other factors are controlled:

![Encoder vs decoder comparison across model sizes and tasks](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder/blob/main/assets/enc_vs_dec.jpg?raw=true)

### Architecture-Specific Advantages Persist

The results show clear patterns:

**Encoders dominate classification and retrieval**: On MNLI classification, even a 150M encoder (89.2) outperforms a 400M decoder (88.2). For retrieval tasks, the gap is smaller but still noticable - especially when decoders are not trained with MNTP.

**Decoders excel at generation**: On generative tasks, decoders maintain consistent advantages, with the performance gap actually widening at larger model sizes.

**Size doesn't always matter**: A 400M encoder beats a 1B decoder on classification tasks, while a 400M decoder beats a 1B encoder on generation tasks.

### Cross-Objective Training Falls Short

Due to the lack of new encoder models, works like [LLM2Vec](https://arxiv.org/abs/2404.05961) have proposed to continue pre-training decoders with MLM. We can now test the effectiveness of this strategy!

We switched the objective and continued to train our models with the opposite objective for 50B additional tokens. This is what we found:

- **Encoder-from-decoder**: Still generally trails native encoders on classification/retrieval
- **Decoder-from-encoder**: Are significantly worse than native decoders, especially at larger scales. This may be because the encoders were trained with MLM instead of MNTP (masked next token prediction) as proposed by LLM2Vec (and used in our encoder from decoder recipe).

This suggests the architecture choice matters fundamentally, not just the training objective.

## Beyond Performance: Understanding Model Behavior

With identical training data, we can study how different objectives affect learning. For example, analyzing gender bias using the WinoGender benchmark reveals:

- **Encoder models** prefer gender-neutral pronouns more often (60%+ neutral vs 30%+ for decoders)
- **Both architectures** show male bias, but decoders slightly more so
- **Cross-objective training** affects bias patterns in measurable ways

This opens doors for systematic studies of how training objectives influence model behavior beyond just accuracy metrics.

## Usage Examples
You can use these models with just a few lines of code!

### Encoders

```python
from transformers import AutoTokenizer, AutoModel

# Load encoder for classification/embeddings
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
model = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-150m")

def predict_masked_token(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions for [MASK] tokens
    mask_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
    predictions = outputs.logits[mask_indices]
    
    # Get top 5 predictions
    top_tokens = torch.topk(predictions, 5, dim=-1)
    return [tokenizer.decode(token) for token in top_tokens.indices[0]]

# Example
masked_text = "The capital of France is [MASK]."
predictions = predict_masked_token(masked_text)
print(f"Predictions: {predictions}")
```

**For classification and retrieval tasks, use encoder models:** You may want to use a fine-tuned version for these tasks as well.

### Decoders

**For text generation tasks, use decoder models:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load decoder for generation
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-150m")
model = AutoModelForCausalLM.from_pretrained("jhu-clsp/ettin-decoder-150m")

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    parser.add_argument("--model_name", type=str, default="jhu-clsp/ettin-encoder-150m")
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
    model_name = "jhu-clsp/ettin-encoder-150m"
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
    "jhu-clsp/ettin-encoder-150m",
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
    model_name = "jhu-clsp/ettin-encoder-150m"

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

### Decoders

<details>
<summary>Click to expand decoder training code</summary>

# Full training
python trl/scripts/sft.py \
    --model_name_or_path jhu-clsp/ettin-decoder-17m \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir ettin-decoder-17m \
    --push_to_hub


# LoRA
python trl/scripts/sft.py \
    --model_name_or_path jhu-clsp/ettin-decoder-17m \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir ettin-decoder-17m \
    --push_to_hub

with sft.py:
```python
import argparse

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    clone_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model_kwargs.pop("use_cache", None)  # Image models do not support cache
        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )

    # Set default chat template if needed
    if tokenizer.chat_template is None:
        # TODO: source should be passed as an argument
        model, tokenizer = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)

```
</details>

## Model Family and Links

The complete Ettin suite includes models at six different scales (for both encoders and decoders):

**Standard Models:**
- [ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m) / [ettin-decoder-17m](https://huggingface.co/jhu-clsp/ettin-decoder-17m) (17M params)
- [ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-32m) / [ettin-decoder-17m](https://huggingface.co/jhu-clsp/ettin-decoder-32m) (32M params)
- [ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-68m) / [ettin-decoder-17m](https://huggingface.co/jhu-clsp/ettin-decoder-68m) (68M params)

- [ettin-encoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-150m) / [ettin-decoder-150m](https://huggingface.co/jhu-clsp/ettin-decoder-150m) (150M params)  
- [ettin-encoder-400m](https://huggingface.co/jhu-clsp/ettin-encoder-400m) / [ettin-decoder-400m](https://huggingface.co/jhu-clsp/ettin-decoder-400m) (400M params)
- [ettin-encoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-1b) / [ettin-decoder-1b](https://huggingface.co/jhu-clsp/ettin-decoder-1b) (1B params)

**Research Resources:**
- [🤗 Ettin Model Collection](https://huggingface.co/collections/jhu-clsp/encoders-vs-decoders-the-ettin-suite-686303e16142257eed8e6aeb)
- [📝 Paper](https://github.com/jhu-clsp/ettin-encoder-vs-decoder)  
- [🗂️ Training Data](https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data) (2T+ tokens, fully open)
- [💻 GitHub Repository](https://github.com/jhu-clsp/ettin-encoder-vs-decoder)
- [📊 250+ Training Checkpoints](https://huggingface.co/datasets/jhu-clsp/ettin-checkpoints) for studying training dynamics or knowledge learning
