---
title: "Improving Hugging Face Training Efficiency Through Packing with Flash Attention 2" 
thumbnail: /blog/assets/packing-with-FA2/thumbnail.png
authors:
- user: RQlee
  guest: true
  org: ibm
- user: ArthurZ
- user: achikundu
  guest: true
  org: ibm
- user: lwtr
  guest: true
  org: ibm
- user: rganti
  guest: true
  org: ibm
- user: mayank-mishra
  guest: true
  org: ibm
---


## TL;DR
Training with packed instruction tuning examples (without padding) is now compatible with Flash Attention 2 in Hugging Face, thanks to a [recent PR](https://github.com/huggingface/transformers/pull/31629) and the new [DataCollatorWithFlattening](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithFlattening)

It can provide up to 2x improvement in training throughput while maintaining convergence quality. Read on for the details! 

## Introduction
Padding input sequences in mini-batches is a usual method to collate inputs during training. However, this introduces inefficiencies because of the irrelevant padding tokens. Packing examples without padding, and using the token position information, is a more efficient alternative. However, previous implementations of packing did not consider example boundaries when using Flash Attention 2, resulting in undesired cross-example attention that reduce quality and convergence.

Hugging Face Transformers now addresses this with a new feature that maintains boundary awareness during packing, alongside the introduction of a new data collator, `DataCollatorWithFlattening`.

By selecting `DataCollatorWithFlattening`, Hugging Face `Trainer` users can now seamlessly concatenate sequences into a single tensor while accounting for sequence boundaries during Flash Attention 2 computations. This is achieved through the `flash_attn_varlen_func`, which calculates the cumulative sequence lengths in each mini-batch (`cu_seqlens`).

The same feature is available to Hugging Face `SFTTrainer` users in the `TRL` library by setting a new flag, `padding_free=True`, when calling the data collator `DataCollatorForCompletionOnlyLM`.

## Up to 2x throughput increase 

We see significant improvement in training throughput using this feature with the new `DataCollatorWithFlattening`. The figure below shows the throughput measured in tokens/second during training. In this example, the throughput is the per-GPU average over 8 A100-80 GPU over one epoch of a 20K randomly selected sample from two different instruct tuning datasets, FLAN and OrcaMath. 

![throughput](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/thruput.png)

FLAN has short sequences on average but a large variance in sequence length, so example lengths in each batch may vary widely. This means that padded FLAN batches may incur a significant overhead in unused padding tokens. Training on the FLAN dataset shows a significant benefit using the new `DataCollatorWithFlattening` in terms of increased throughput. We see a 2x throughput increase on the models shown here: llama2-7B, mistral-7B, and granite-8B-code. 

OrcaMath has longer examples and a lower variance in example length. As such, the improvement from packing is lower. Our experiments show a 1.4x increase in throughput when training using this form of packing on the OrcaMath dataset across these three models.

![memory](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/memory.png)

Memory usage also improves through packing with the new `DataCollatorWithFlattening`. The following figure shows the peak memory usage of the same three models training on the same two datasets. Peak memory is reduced by 20% on the FLAN dataset, which benefits considerably from packing. 

Peak memory reduction is 6% on the OrcaMath dataset with its more homogeneous example lengths.

Packing examples, when it reduces the number of optimization steps, may harm training convergence. The new feature, however, retains the minibatches and, hence, the same number of optimization steps as would be used with padded examples. Thus, there is no impact on train convergence, as we see in the next figure, which shows identical validation loss of the same three models training on the same two datasets, whether the models are trained with packing using the new `DataCollatorWithFlattening` or with padding.

![ValLoss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/ValLoss.png)

## How it works 

Consider a batch of data with a batchsize = 4 where the four sequences are as follows:

![batch](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/four_sequences.png)

After concatenating the examples, the padding-free collator returns the `input_ids`, `labels`, and `position_ids` of each example. Hence, the collator provides, for this batch of data,  

![example](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/input_ids_labels_position_ids.png)

The modifications required are lightweight and are limited to providing the `position_ids` to Flash Attention 2. 

This relies, however, on the model exposing `position_ids`. As of the time of writing, 14 models expose them and are supported by the solution. Specifically, Llama 2 and 3, Mistral, Mixtral, Granite, DBRX, Falcon, Gemma, OLMo, Phi 1, 2, and 3, phi3, Qwen 2 and 2 MoE, StableLM, and StarCoder 2 are all supported by the solution.

## Getting started
Reaping the benefits of packing with `position_ids` is easy. 

If you are using Hugging Face `Trainer` from `Transformers`, only two steps are required:

1) Instantiate the model with Flash Attention 2
2) Use the new `DataCollatorWithFlattening`

If you are using Hugging Face `SFTTrainer` from `TRL` with `DataCollatorForCompletionOnlyLM`, then the two required steps are:

1) Instantiate the model with Flash Attention 2
2) Set `padding_free=True`  when calling `DataCollatorForCompletionOnlyLM` as follows:
`collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, padding_free=True)`
   
## How to use it

For `Trainer` users, the example below illustrates how to use the new feature. 

```Python
# Example using DataCollatorWithFlattening
 
import torch

# load model as usual
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "instructlab/merlinite-7b-lab",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# read dataset as usual
from datasets import load_dataset
train_dataset = load_dataset("json", data_files="path/to/my/dataset")["train"]

# use DataCollatorWithFlattening
from transformers import DataCollatorWithFlattening
data_collator = DataCollatorWithFlattening()

# train
from transformers import TrainingArguments, Trainer
train_args = TrainingArguments(output_dir="/save/path")
trainer = Trainer(
    args=train_args,
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator
)
trainer.train()
```

For `TRL` users, the example below shows how to use the new feature with `SFTTrainer`.

```Python
# SFTTrainer example using DataCollatorForCompletionOnlyLM

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

model = AutoModelForCausalLM.from_pretrained(
    "instructlab/merlinite-7b-lab",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("instructlab/merlinite-7b-lab")
tokenizer.pad_token = tokenizer.eos_token 

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, padding_free=True)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./tmp",
        gradient_checkpointing=True,
        per_device_train_batch_size=8
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
```


## Conclusions

Packing instruction tuning examples, instead of padding, is now fully compatible with Flash Attention 2, thanks to a recent PR and the new `DataCollatorWithFlattening`. The method is compatible with models that use `position_ids`. Benefits can be seen in throughput and peak memory usage during training, with no degradation in training convergence. Actual throughput and memory improvement depends on the model and the distribution of example lengths in the training data. Training with data that has a wide variation of example lengths will see the greatest benefit, with respect to padding, by using the `DataCollatorWithFlattening`. The same feature is available to `SFTTrainer` users in the `TRL` library by setting a new flag, `padding_free=True`, when calling `DataCollatorForCompletionOnlyLM`.

For a more detailed analysis, have a look at the paper at https://huggingface.co/papers/2407.09105
