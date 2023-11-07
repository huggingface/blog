---
title: "Comparing the Performance of LLMs: A Deep Dive into Roberta, Llama 2, and Mistral for Disaster Tweets Analysis with Lora" 
thumbnail: /blog/assets/Lora-for-sequence-classification-with-Roberta-Llama-Mistral/Thumbnail.png
authors:
- user: mehdiiraqui 
  guest: true
---

# Comparing the Performance of LLMs: A Deep Dive into Roberta, Llama 2, and Mistral for Disaster Tweets Analysis with Lora
<!-- TOC -->

- [Comparing the Performance of LLMs: A Deep Dive into Roberta, Llama 2, and Mistral for Disaster Tweets Analysis with LoRA](#comparing-the-performance-of-llms-a-deep-dive-into-roberta-llama-2-and-mistral-for-disaster-tweets-analysis-with-lora)
    - [Introduction](#introduction)
    - [Hardware Used](#hardware-used)
    - [Goals](#goals)
    - [Dependencies](#dependencies)
    - [Pre-trained Models](#pre-trained-models)
        - [RoBERTa](#roberta)
        - [Llama 2](#llama-2)
        - [Mistral 7B](#mistral-7b)
    - [LoRA](#lora)
    - [Setup](#setup)
    - [Data preparation](#data-preparation)
        - [Data loading](#data-loading)
        - [Data Processing](#data-processing)
    - [Models](#models)
        - [RoBERTa](#roberta)
            - [Load RoBERTA Checkpoints for the Classification Task](#load-roberta-checkpoints-for-the-classification-task)
            - [LoRA setup for RoBERTa classifier](#lora-setup-for-roberta-classifier)
        - [Mistral](#mistral)
            - [Load checkpoints for the classfication model](#load-checkpoints-for-the-classfication-model)
            - [LoRA setup for Mistral 7B classifier](#lora-setup-for-mistral-7b-classifier)
        - [Llama 2](#llama-2)
            - [Load checkpoints for the classification mode](#load-checkpoints-for-the-classfication-mode)
            - [LoRA setup for Llama 2 classifier](#lora-setup-for-llama-2-classifier)
    - [Setup the trainer](#setup-the-trainer)
        - [Evaluation Metrics](#evaluation-metrics)
        - [Custom Trainer for Weighted Loss](#custom-trainer-for-weighted-loss)
        - [Trainer Setup](#trainer-setup)
            - [RoBERTa](#roberta)
            - [Mistral-7B](#mistral-7b)
            - [Llama 2](#llama-2)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Results](#results)
    - [Conclusion](#conclusion)
    - [Resources](#resources)

<!-- /TOC -->



## Introduction 

In the fast-moving world of Natural Language Processing (NLP), we often find ourselves comparing different language models to see which one works best for specific tasks. This blog post is all about comparing three models: RoBERTa, Mistral-7b, and Llama-2-7b. We used them to tackle a common problem - classifying tweets about disasters. It is important to note that Mistral and Llama 2 are large models with 7 billion parameters. In contrast, RoBERTa-large (355M parameters) is a relatively smaller model used as a baseline for the comparison study.

In this blog, we used PEFT (Parameter-Efficient Fine-Tuning) technique: LoRA (Low-Rank Adaptation of Large Language Models) for fine-tuning the pre-trained model on the sequence classification task. LoRa is designed to significantly reduce the number of trainable parameters while maintaining strong downstream task performance. 

The main objective of this blog post is to implement LoRA fine-tuning for sequence classification tasks using three pre-trained models from Hugging Face: [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), and [roberta-large](https://huggingface.co/roberta-large)

## Hardware Used 

- Number of nodes: 1 
- Number of GPUs per node: 1
- GPU type: A6000 
- GPU memory: 48GB


## Goals

- Implement fine-tuning of pre-trained LLMs using LoRA PEFT methods.
- Learn how to use the HuggingFace APIs ([transformers](https://huggingface.co/docs/transformers/index), [peft](https://huggingface.co/docs/peft/index), and [datasets](https://huggingface.co/docs/datasets/index)).
- Setup the hyperparameter tuning and experiment logging using [Weights & Biases](https://wandb.ai).



## Dependencies

```bash
datasets
evaluate
peft
scikit-learn
torch
transformers
wandb 
```
Note: For reproducing the reported results, please check the pinned versions in the [wandb reports](#resources).


## Pre-trained Models

### [RoBERTa](https://arxiv.org/abs/1907.11692)

RoBERTa (Robustly Optimized BERT Approach) is an advanced variant of the BERT model proposed by Meta AI research team. BERT is a transformer-based language model using self-attention mechanisms for contextual word representations and trained with a masked language model objective. Note that BERT is an encoder only model used for natural language understanding tasks (such as sequence classification and token classification).

RoBERTa is a popular model to fine-tune and appropriate as a baseline for our experiments. For more information, you can check the Hugging Face model [card](https://huggingface.co/docs/transformers/model_doc/roberta).


### [Llama 2](https://arxiv.org/abs/2307.09288)

Llama 2 models, which stands for Large Language Model Meta AI, belong to the family of large language models (LLMs) introduced by Meta AI. The Llama 2 models vary in size, with parameter counts ranging from 7 billion to 65 billion.

Llama 2 is an auto-regressive language model, based on the transformer decoder architecture. To generate text, Llama 2 processes a sequence of words as input and iteratively predicts the next token using a sliding window.
Llama 2 architecture is slightly different from models like GPT-3. For instance, Llama 2 employs the SwiGLU activation function rather than ReLU and opts for rotary positional embeddings in place of absolute learnable positional embeddings. 
 
The recently released Llama 2 introduced architectural refinements to better leverage very long sequences by extending the context length to up to 4096 tokens, and using grouped-query attention (GQA) decoding. 

### [Mistral 7B](https://arxiv.org/abs/2310.06825)

Mistral 7B v0.1, with 7.3 billion parameters, is the first LLM introduced by Mistral AI.
The main novel techniques used in Mistral 7B's architecture are: 
- Sliding Window Attention: Replace the full attention (square compute cost) with a sliding window based attention where each token can attend to at most 4,096 tokens from the previous layer (linear compute cost). This mechanism enables Mistral 7B to handle longer sequences, where higher layers can access historical information beyond the window size of 4,096 tokens. 
- Grouped-query Attention: used in Llama 2 as well, the technique optimizes the inference process (reduce processing time) by caching the key and value vectors for previously decoded tokens in the sequence.  

## [LoRA](https://arxiv.org/abs/2106.09685)

PEFT, Parameter Efficient Fine-Tuning, is a collection of techniques (p-tuning, prefix-tuning, IA3, Adapters, and LoRa) designed to fine-tune large models using a much smaller set of training parameters while preserving the performance levels typically achieved through full fine-tuning. 

LoRA, Low-Rank Adaptation, is a PEFT method that shares similarities with Adapter layers. Its primary objective is to reduce the model's trainable parameters. LoRA's operation involves 
learning a low rank update matrix while keeping the pre-trained weights frozen.

![image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/Lora-for-sequence-classification-with-Roberta-Llama-Mistral/lora.png)

## Setup

RoBERTa has a limitatiom of maximum sequence length of 512, so we set the `MAX_LEN=512` for all models to ensure a fair comparison.

```python
MAX_LEN = 512 
roberta_checkpoint = "roberta-large"
mistral_checkpoint = "mistralai/Mistral-7B-v0.1"
llama_checkpoint = "meta-llama/Llama-2-7b-hf"
```

## Data preparation
### Data loading

We will load the dataset from Hugging Face:
```python
from datasets import load_dataset
dataset = load_dataset("mehdiiraqui/twitter_disaster")
```
 Now, let's split the dataset into training and validation datasets. Then add the test set:

```python
from datasets import Dataset
# Split the dataset into training and validation datasets
data = dataset['train'].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
data['val'] = data.pop("test")
# Convert the test dataframe to HuggingFace dataset and add it into the first dataset
data['test'] = dataset['test']
```

Here's an overview of the dataset:

```bash
DatasetDict({
    train: Dataset({
        features: ['id', 'keyword', 'location', 'text', 'target'],
        num_rows: 6090
    })
    val: Dataset({
        features: ['id', 'keyword', 'location', 'text', 'target'],
        num_rows: 1523
    })
    test: Dataset({
        features: ['id', 'keyword', 'location', 'text', 'target'],
        num_rows: 3263
    })
})
```

Let's check the data distribution:

```python
import pandas as pd

data['train'].to_pandas().info()
data['test'].to_pandas().info()
```

- Train dataset

```<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7613 entries, 0 to 7612
Data columns (total 5 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   id        7613 non-null   int64 
 1   keyword   7552 non-null   object
 2   location  5080 non-null   object
 3   text      7613 non-null   object
 4   target    7613 non-null   int64 
dtypes: int64(2), object(3)
memory usage: 297.5+ KB
```


- Test dataset
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3263 entries, 0 to 3262
Data columns (total 5 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   id        3263 non-null   int64 
 1   keyword   3237 non-null   object
 2   location  2158 non-null   object
 3   text      3263 non-null   object
 4   target    3263 non-null   int64 
dtypes: int64(2), object(3)
memory usage: 127.6+ KB
```

**Target distribution in the train dataset**
```
target
0    4342
1    3271
Name: count, dtype: int64
```

As the classes are not balanced, we will compute the positive and negative weights and use them for loss calculation later:
```python
pos_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().target.value_counts()[1])
neg_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().target.value_counts()[0])
```

The final weights are: 
```
POS_WEIGHT, NEG_WEIGHT = (1.1637114032405993, 0.8766697374481806)
```

Then, we compute the maximum length of the column text:
```python
# Number of Characters
max_char = data['train'].to_pandas()['text'].str.len().max()
# Number of Words
max_words = data['train'].to_pandas()['text'].str.split().str.len().max()
```

```
The maximum number of characters is 152.
The maximum number of words is 31.
```

### Data Processing

Let's take a look to one row example of training data: 

```python
data['train'][0]
```

```
{'id': 5285,
 'keyword': 'fear',
 'location': 'Thibodaux, LA',
 'text': 'my worst fear. https://t.co/iH8UDz8mq3',
 'target': 0}
```

The data comprises a keyword, a location and the text of the tweet. For the sake of simplicity, we select the `text` feature as the only input to the LLM. 

At this stage, we prepared the train, validation, and test sets in the HuggingFace format expected by the pre-trained LLMs. The next step is to define the tokenized dataset for training using the appropriate tokenizer to transform the `text` feature into two Tensors of sequence of token ids and attention masks. As each model has its specific tokenizer, we will need to define three different datasets. 


We start by defining the RoBERTa dataloader: 

-  Load the tokenizer: 
```python
from transformers import AutoTokenizer
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_checkpoint, add_prefix_space=True)
```
**Note:** The RoBERTa tokenizer has been trained to treat spaces as part of the token. As a result, the first word of the sentence is encoded differently if it is not preceded by a white space. To ensure the first word includes a space, we set `add_prefix_space=True`. Also, to maintain consistent pre-processing for all three models, we set the parameter to 'True' for Llama 2 and Mistral 7b.

- Define the preprocessing function for converting one row of the dataframe:
```python
def roberta_preprocessing_function(examples):
    return roberta_tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)
```

By applying the preprocessing function to the first example of our training dataset, we have the tokenized inputs (`input_ids`) and the attention mask:
```python
roberta_preprocessing_function(data['train'][0])
```
```
{'input_ids': [0, 127, 2373, 2490, 4, 1205, 640, 90, 4, 876, 73, 118, 725, 398, 13083, 329, 398, 119, 1343, 246, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

- Now, let's  apply the preprocessing function to the entire dataset: 

```python
col_to_delete = ['id', 'keyword','location', 'text']
# Apply the preprocessing function and remove the undesired columns
roberta_tokenized_datasets = data.map(roberta_preprocessing_function, batched=True, remove_columns=col_to_delete)
# Rename the target to label as for HugginFace standards
roberta_tokenized_datasets = roberta_tokenized_datasets.rename_column("target", "label")
# Set to torch format
roberta_tokenized_datasets.set_format("torch")
```

**Note:** we deleted the undesired columns from our data: id, keyword, location and text. We have deleted the text because we have already converted it into the inputs ids and the attention mask:


We can have a look into our tokenized training dataset:
```python
roberta_tokenized_datasets['train'][0]
```

```
{'label': tensor(0),
 'input_ids': tensor([    0,   127,  2373,  2490,     4,  1205,   640,    90,     4,   876,
            73,   118,   725,   398, 13083,   329,   398,   119,  1343,   246,
             2]),
 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
```


- For generating the training batches, we also need to pad the rows of a given batch to the maximum length found in the batch. For that, we will use the `DataCollatorWithPadding` class:

```python
# Data collator for padding a batch of examples to the maximum length seen in the batch
from transformers import DataCollatorWithPadding
roberta_data_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)
```


You can follow the same steps for preparing the data for Mistral 7B and Llama 2 models: 

**Note** that Llama 2 and Mistral 7B don't have a default `pad_token_id`. So, we use the `eos_token_id` for padding as well.

- Mistral 7B: 

```python
# Load Mistral 7B Tokenizer
from transformers import AutoTokenizer, DataCollatorWithPadding
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_checkpoint, add_prefix_space=True)
mistral_tokenizer.pad_token_id = mistral_tokenizer.eos_token_id
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token

def mistral_preprocessing_function(examples):
    return mistral_tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

mistral_tokenized_datasets = data.map(mistral_preprocessing_function, batched=True, remove_columns=col_to_delete)
mistral_tokenized_datasets = mistral_tokenized_datasets.rename_column("target", "label")
mistral_tokenized_datasets.set_format("torch")

# Data collator for padding a batch of examples to the maximum length seen in the batch
mistral_data_collator = DataCollatorWithPadding(tokenizer=mistral_tokenizer)
```

- Llama 2:
```python
# Load Llama 2 Tokenizer
from transformers import AutoTokenizer, DataCollatorWithPadding
llama_tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint, add_prefix_space=True)
llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
llama_tokenizer.pad_token = llama_tokenizer.eos_token

def llama_preprocessing_function(examples):
    return llama_tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

llama_tokenized_datasets = data.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
llama_tokenized_datasets = llama_tokenized_datasets.rename_column("target", "label")
llama_tokenized_datasets.set_format("torch")

# Data collator for padding a batch of examples to the maximum length seen in the batch
llama_data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)
```

Now that we have prepared the tokenized datasets, the next section will showcase how to load the pre-trained LLMs checkpoints and how to set the LoRa weights. 

## Models

### RoBERTa

#### Load RoBERTa Checkpoints for the Classification Task

We load the pre-trained RoBERTa model with a sequence classification head using the Hugging Face `AutoModelForSequenceClassification` class:

```python
from transformers import AutoModelForSequenceClassification 
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_checkpoint, num_labels=2)
```


####  LoRA setup for RoBERTa classifier

We import LoRa configuration and set some parameters for RoBERTa classifier:
- TaskType: Sequence classification
- r(rank): Rank for our decomposition matrices
- lora_alpha: Alpha parameter to scale the learned weights. LoRA paper advises fixing alpha at 16
- lora_dropout: Dropout probability of the LoRA layers
- bias: Whether to add bias term to LoRa layers

The code below uses the values recommended by the [Lora paper](https://arxiv.org/abs/2106.09685). [Later in this post](#hyperparameter-tuning) we will perform hyperparameter tuning of these parameters using `wandb`.

```python
from peft import get_peft_model, LoraConfig, TaskType
roberta_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=2, lora_alpha=16, lora_dropout=0.1, bias="none",
)
roberta_model = get_peft_model(roberta_model, roberta_peft_config)
roberta_model.print_trainable_parameters()
```

We can see that the number of trainable parameters represents only 0.64% of the RoBERTa model parameters:
```bash
trainable params: 2,299,908 || all params: 356,610,052 || trainable%: 0.6449363911929212
```


### Mistral

#### Load checkpoints for the classfication model

Let's load the pre-trained Mistral-7B model with a sequence classification head:

```python
from transformers import AutoModelForSequenceClassification
import torch
mistral_model =  AutoModelForSequenceClassification.from_pretrained(
  pretrained_model_name_or_path=mistral_checkpoint,
  num_labels=2,
  device_map="auto"
)
```
For Mistral 7B, we have to add the padding token id as it is not defined by default.

```python
mistral_model.config.pad_token_id = mistral_model.config.eos_token_id
```

####  LoRa setup for Mistral 7B classifier

For Mistral 7B model, we need to specify the `target_modules` (the query and value vectors from the attention modules):

```python
from peft import get_peft_model, LoraConfig, TaskType

mistral_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=2, lora_alpha=16, lora_dropout=0.1, bias="none", 
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

mistral_model = get_peft_model(mistral_model, mistral_peft_config)
mistral_model.print_trainable_parameters()
```

The number of trainable parameters reprents only 0.024% of the Mistral model parameters:
```
trainable params: 1,720,320 || all params: 7,112,380,416 || trainable%: 0.02418768259540745
```

### Llama 2

#### Load checkpoints for the classfication mode

Let's load pre-trained Llama 2 model with a sequence classification header. 

```python
from transformers import AutoModelForSequenceClassification
import torch
llama_model =  AutoModelForSequenceClassification.from_pretrained(
  pretrained_model_name_or_path=llama_checkpoint,
  num_labels=2,
  device_map="auto",
  offload_folder="offload",
  trust_remote_code=True
)
```

For Llama 2, we have to add the padding token id as it is not defined by default.

```python
llama_model.config.pad_token_id = llama_model.config.eos_token_id
```

#### LoRa setup for Llama 2 classifier

We define LoRa for Llama 2 with the same parameters as for Mistral:

```python
from peft import get_peft_model, LoraConfig, TaskType
llama_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.05, bias="none", 
    target_modules=[
        "q_proj",
        "v_proj",  
    ],
)

llama_model = get_peft_model(llama_model, llama_peft_config)
llama_model.print_trainable_parameters()

```

The number of trainable parameters reprents only 0.12% of the Llama 2 model parameters:
```
trainable params: 8,404,992 || all params: 6,615,748,608 || trainable%: 0.1270452143516515
```

At this point, we defined the tokenized dataset for training as well as the LLMs setup with LoRa layers. The following section will introduce how to launch training using the HuggingFace `Trainer` class. 


## Setup the trainer

### Evaluation Metrics

First, we define the performance metrics we will use to compare the three models: F1 score, recall, precision and accuracy:

```python
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}
```

### Custom Trainer for Weighted Loss 
As mentioned at the beginning of this post, we have an imbalanced distribution between positive and negative classes. We need to train our models with a weighted cross-entropy loss to account for that. The `Trainer` class doesn't support providing a custom loss as it expects to get the loss directly from the model's outputs. 

So, we need to define our custom `WeightedCELossTrainer` that overrides the `compute_loss` method to calculate the weighted cross-entropy loss based on the model's predictions and the input labels: 

```python
from transformers import Trainer

class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weights, pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```


### Trainer Setup

Let's set the training arguments and the trainer for the three models.

#### RoBERTa 
First important step is to move the models to the GPU device for training.

```python
roberta_model = roberta_model.cuda()
roberta_model.device()
```
It will print the following: 
```
device(type='cuda', index=0)
```

Then, we set the training arguments:

```python
from transformers import TrainingArguments

lr = 1e-4
batch_size = 8
num_epochs = 5

training_args = TrainingArguments(
    output_dir="roberta-large-lora-token-classification",
    learning_rate=lr,
    lr_scheduler_type= "constant",
    warmup_ratio= 0.1,
    max_grad_norm= 0.3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    fp16=False,
    gradient_checkpointing=True,
)
```

Finally, we define the RoBERTa trainer by providing the model, the training arguments and the tokenized datasets:

```python
roberta_trainer = WeightedCELossTrainer(
    model=mistral_model,
    args=training_args,
    train_dataset=mistral_tokenized_datasets['train'],
    eval_dataset=mistral_tokenized_datasets["val"],
    data_collator=mistral_data_collator,
    compute_metrics=compute_metrics
)
```

#### Mistral-7B

Similar to RoBERTa, we initialize the `WeightedCELossTrainer` as follows: 

```python
from transformers import TrainingArguments, Trainer

mistral_model = mistral_model.cuda()

lr = 1e-4
batch_size = 8
num_epochs = 5

training_args = TrainingArguments(
    output_dir="mistral-lora-token-classification",
    learning_rate=lr,
    lr_scheduler_type= "constant",
    warmup_ratio= 0.1,
    max_grad_norm= 0.3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    fp16=True,
    gradient_checkpointing=True,
)


mistral_trainer = WeightedCELossTrainer(
    model=mistral_model,
    args=training_args,
    train_dataset=mistral_tokenized_datasets['train'],
    eval_dataset=mistral_tokenized_datasets["val"],
    data_collator=mistral_data_collator,
    compute_metrics=compute_metrics
)
```

**Note** that we needed to enable half-precision training by setting `fp16` to `True`. The main reason is that Mistral-7B is large, and its weights cannot fit into one GPU memory (48GB) with full float32 precision. 

#### Llama 2

Similar to Mistral 7B, we define the trainer as follows: 

```python
from transformers import TrainingArguments, Trainer

llama_model = llama_model.cuda()

lr = 1e-4
batch_size = 8
num_epochs = 5
training_args = TrainingArguments(
    output_dir="llama-lora-token-classification",
    learning_rate=lr,
    lr_scheduler_type= "constant",
    warmup_ratio= 0.1,
    max_grad_norm= 0.3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    fp16=True,
    gradient_checkpointing=True,
)



llama_trainer = WeightedCELossTrainer(
    model=llama_model,
    args=training_args,
    train_dataset=llama_tokenized_datasets['train'],
    eval_dataset=llama_tokenized_datasets["val"],
    data_collator=llama_data_collator,
    compute_metrics=compute_metrics
)
```




## Hyperparameter Tuning

We have used Wandb Sweep API to run hyperparameter tunning with Bayesian search strategy (30 runs). The hyperparameters tuned are the following.


| method | metric              | lora_alpha                                | lora_bias                 | lora_dropout            | lora_rank                                          | lr                          | max_length                |
|--------|---------------------|-------------------------------------------|---------------------------|-------------------------|----------------------------------------------------|-----------------------------|---------------------------|
| bayes  | goal: maximize      | distribution: categorical                 | distribution: categorical | distribution: uniform   | distribution: categorical                          | distribution: uniform       | distribution: categorical |
|        | name: eval/f1-score | values:   <br>-16     <br>-32     <br>-64 | values: None              | -max: 0.1   <br>-min: 0 | values:     <br>-4   <br>-8    <br>-16     <br>-32 | -max: 2e-04<br>-min: 1e-05 | values: 512               |           |


 For more information, you can check the Wandb experiment report in the [resources sections](#resources).



## Results

| Models  | F1 score | Training time  | Memory consumption           | Number of trainable parameters |
|---------|----------|----------------|------------------------------|--------------------------------|
| RoBERTa | 0.8077   | 538 seconds    | GPU1: 9.1 Gb<br>GPU2: 8.3 Gb | 0.64%                          |
| Mistral 7B | 0.7364   | 2030 seconds   | GPU1: 29.6 Gb<br>GPU2: 29.5 Gb | 0.024%                         |
| Llama 2  | 0.7638   | 2052 seconds   | GPU1: 35 Gb <br>GPU2: 33.9 Gb | 0.12%                          |


## Conclusion

In this blog post, we compared the performance of three large language models (LLMs) - RoBERTa, Mistral 7b, and Llama 2 - for disaster tweet classification using LoRa. From the performance results, we can see that RoBERTa is outperforming Mistral 7B and Llama 2 by a large margin. This raises the question about whether we really need a complex and large LLM for tasks like short-sequence binary classification? 

One learning we can draw from this study is that one should account for the specific project requirements, available resources, and performance needs to choose the LLMs model to use. 

Also, for relatively *simple* prediction tasks with short sequences base models such as RoBERTa remain competitive. 

Finally, we showcase that LoRa method can be applied to both encoder (RoBERTa) and decoder (Llama 2 and Mistral 7B) models. 

## Resources 

1. You can find the code script in the following [Github project](https://github.com/mehdiir/Roberta-Llama-Mistral/).

2. You can check the hyper-param search results in the following Weight&Bias reports:
    - [RoBERTa](https://api.wandb.ai/links/mehdi-iraqui/505c22j1)
    - [Mistral 7B](https://api.wandb.ai/links/mehdi-iraqui/24vveyxp)
    - [Llama 2](https://api.wandb.ai/links/mehdi-iraqui/qq8beod0)
