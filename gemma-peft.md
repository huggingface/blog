---
title: Fine-Tuning Gemma Models in Hugging Face
authors:
- user: svaibhav
  guest: true
- user: alanwaketan
  guest: true
- user: ybelkada
- user: ArthurZ
---

# Fine-Tuning Gemma Models in Hugging Face

## Introduction

We recently announced that [Gemma](https://huggingface.co/blog/gemma), the open weights language model from Google Deepmind, is available for the broader open-source community via Hugging Face. It’s available in 2 billion and 7 billion parameter sizes with pretrained and instruction-tuned flavors. It’s available on Hugging Face, supported in TGI, and easily accessible for deployment and fine-tuning in the Vertex Model Garden and Google Kubernetes Engine.

<div class="flex items-center justify-center">
<img src="/blog/assets/gemma-peft/Gemma-peft.png" alt="Gemma Deploy">
</div>



The Gemma family of models also happens to be well suited for prototyping and experimentation using the free GPU resource via Colab. In this post we will briefly review how you can do Parameter Efficient FineTuning (PEFT) for Gemma models, using the Hugging Face Transformers and PEFT libraries on GPUs and Cloud TPUs for anyone who wants to fine-tune Gemma models on their own dataset.



## Why PEFT?

The default (full weight) training for language models, even for modest sizes, tends to be memory and compute-intensive. On one hand, it can be prohibitive for users relying on openly available compute platforms such as a Colab or Kaggle notebook for learning and experimentation. And on the other hand even for enterprise users, the cost of adapting these models for different domains is an important metric to optimize. PEFT, or parameter-efficient fine tuning is a popular technique to accomplish this at low cost. 

# PyTorch on GPU and TPU
Gemma models in HuggingFace transformers are well optimized for both PyTorch and PyTorch/XLA. This enables both TPU and GPU users to access and experiment with Gemma models as needed. Together with the Gemma release, we also improved the [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/) experience for PyTorch/XLA in Hugging Face. This [FSDP via SPMD](https://github.com/pytorch/xla/issues/6379) integration also allows other Hugging Face models to take advantage of TPU acceleration via PyTorch/XLA. In this post we will focus on PEFT, more specifically, Low-Rank Adaptation (LoRA), for Gemma models. For a more comprehensive set of LoRA techniques we encourage readers to review the [Scaling Down to Scale Up, from Lialin et al](https://arxiv.org/pdf/2303.15647.pdf) and [this](https://pytorch.org/blog/finetune-llms/) excellent blog post by Belkada et al. 

## Low-Rank Adaptation for Large Language Models

Low-Rank Adaptation (LoRA) is one of the parameter-efficient fine-tuning techniques for large language models (LLMs). It incorporates a fraction of total model parameters to be fine-tuned by freezing the original model and only training an adapter layer which is decomposed into low-rank matrices. PEFT library provides an easy abstraction that allows users to select particular layers from the model where adapter weights are applied.

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
```

In this snippet we refer to all `nn.Linear` layers as the target layers to be adapted.

In the following example, we will leverage [QLoRA, from Dettmers et al.](https://arxiv.org/abs/2305.14314) in order to quantize the base model in 4-bit precision for a more memory efficient fine-tuning protocol. A model can be loaded with QLoRA by first installing a `bitsandbytes` library on your environment, then passing a `BitsAndBytesConfig` object to `from_pretrained` when loading the model.

## Before we begin

In order to access Gemma model artifacts, the users are required to accept the consent form here ([link](https://huggingface.co/google/gemma-7b-it)).
Now let’s get started with an implementation.

## Learning to quote

Assuming that you have submitted the consent form, you can access the model artifact from the [Hugging Face models](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b).

We start by downloading the model and tokenizer into the respective objects. Here we are also including `BitsAndBytesConfig` for weight only quantization.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer

model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])
```

Now we test the model before starting the finetuning, using a famous quote:


```python
text = "Quote: Imagination is more"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

The model does a reasonable completion with some extra tokens:
```
Quote: Imagination is more important than knowledge. Knowledge is limited. Imagination encircles the world.

-Albert Einstein

I
```

But this is not exactly the format we would love the answer to be. Let’s see if we can use fine-tuning to teach the model to generate the answer in the following format.

```
Quote: Imagination is more important than knowledge. Knowledge is limited. Imagination encircles the world.

Author: Albert Einstein

```
To begin with, let's select an English quotes dataset.
```python
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
```

Now let’s finetune this model using the LoRA config stated above:

```python
import transformers
from trl import SFTTrainer

def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"
    return [text]

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
trainer.train()
```

Finally we are ready to test the model once more with the same prompt we used earlier:

```python
text = "Quote: Imagination is"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

This time we get the response in the format we like:


```
Quote: Imagination is more important than knowledge. Knowledge is limited. Imagination encircles the world.

Author: Albert Einstein
```


## Accelerate with FSDP via SPMD on TPU

As mentioned earlier, Hugging Face `transformers` now support PyTorch/XLA’s latest FSDP implementation. This can greatly accelerate the fine-tuning speed. To enable that, one just needs to add an FSDP config to the `transformers.Trainer`:

```python
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Set up the FSDP config. To enable FSDP via SPMD, set xla_fsdp_v2 to True.
fsdp_config = {"fsdp_transformer_layer_cls_to_wrap": [
        "GemmaDecoderLayer"
    ],
    "xla": True,
    "xla_fsdp_v2": True,
    "xla_fsdp_grad_ckpt": True}

# Finally, set up the trainer and train the model.
trainer = Trainer(
    model=model,
    train_dataset=data,
    args=TrainingArguments(
        per_device_train_batch_size=64,  # This is actually the global batch size for SPMD.
        num_train_epochs=100,
        max_steps=-1,
        output_dir="./output",
        optim="adafactor",
        logging_steps=1,
        dataloader_drop_last = True,  # Required for SPMD.
        fsdp="full_shard",
        fsdp_config=fsdp_config,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
```

## Next Steps

We walked through this simple example adapted from the source notebook to illustrate the LoRA finetuning method applied to Gemma models. The full colab for GPU can be found [here](https://huggingface.co/google/gemma-7b/blob/main/examples/notebook_sft_peft.ipynb), and the full script for TPU can be found [here](https://huggingface.co/google/gemma-7b/blob/main/examples/example_fsdp.py). We are excited about the endless possibilities for research and learning thanks to this recent addition to our open source ecosystem. We encourage users to also visit the [Gemma documentation](https://huggingface.co/docs/transformers/v4.38.0/en/model_doc/gemma), as well as our launch blog for more examples to train, finetune and deploy Gemma models. 


