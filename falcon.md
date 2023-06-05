---
title: "The Falcon has landed in the Hugging Face ecosystem" 
thumbnail: /blog/assets/147_falcon/falcon_thumbnail.png
authors:
- user: slippylolo
- user: lvwerra
- user: TODO
---

# The Falcon has landed in the Hugging Face ecosystem

<!-- {blog_metadata} -->
<!-- {authors} -->

## Introduction

Falcon is a new family of state-of-the-art large language models created by the [Technology Innovation Institute](https://www.tii.ae/) in Abu Dhabi, released under the Apache 2.0 license. **Notably, [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b) is the first “truly open” model with capabilities rivaling many current closed-source models**. This is fantastic news for practitioners, enthusiasts, and industry, as it opens the door for many exciting use cases.

In this blog, we will be taking a deep dive into the Falcon models: first discussing what makes them unique and then **showcasing how easy it is to build on top of them (inference, quantization, finetuning, and more) with tools from the 🤗 Hugging Face ecosystem**. 

## Table of Content

- [The Falcon models](#the-falcon-models)
- [Demo](#demo)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Fine-tuning with PEFT](#fine-tuning-with-peft)
- [Conclusion](#conclusion)

## The Falcon models

The Falcon family is composed of two base models: [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b) and its little brother [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b). **The 40B parameters model currently tops the charts of the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), while the 7B model is the best in its weight class**.

Falcon-40B requires ~90GB of GPU memory —that’s a lot, but still less than the requirements for LLaMA-65B, which Falcon outperforms. On the other hand, Falcon-7B only needs ~15GB, making inference and finetuning accessible even on consumer hardware. *(Later in this blog, we will discuss how we can leverage quantization to make Falcon-40B accessible even on cheaper GPUs!)* 

TII has also made available so-called instruct versions of the two models, [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) and [Falcon-40B-Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct). These experimental variants have been further finetuned on instructions and conversational data; they thus lend better to popular assistant-style tasks. If you are just looking to quickly play with the models they are your best shot. It’s also possible to build your own custom instruct version, based on the plethora of datasets built by the community—keep reading for a step-by-step tutorial! 

Falcon-7B and Falcon-40B have been trained on 1.5 trillion and 1 trillion tokens respectively, in line with modern models optimising for inference. **The key ingredient for the high quality of the Falcon models is their training data, predominantly based (>80%) on [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)—a novel massive web dataset based on CommonCrawl**. Instead of gathering scattered curated sources, TII has focused on scaling and improving the quality of web data, leveraging large-scale deduplication and strict filtering to match the quality of other corpora. The Falcon models still include some curated sources in their training (such as conversational data from Reddit), but significantly less so than has been common for state-of-the-art LLMs like GPT-3 or PaLM. The best part? TII has publicly released a 600 billion tokens extract of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) for the community to use in their own LLMs!   

Another interesting feature of the Falcon models is their use of [**multiquery attention**](https://arxiv.org/abs/1911.02150). The vanilla multihead attention scheme has one query, key, and value per head; multiquery instead shares one key and value across all heads. This trick doesn’t significantly influence pretraining, but it greatly [improves the scalability of inference](https://arxiv.org/abs/2211.05102): indeed, **the K,V-cache kept during autoregressive decoding is now significantly smaller** (10-100 times depending on the specific of the architecture), reducing memory costs and enabling novel optimizations such as statefulness.

| Model | License | Commercial use? | Pretraining length [tokens] | Pretraining compute [PF-days] | Leaderboard score | K,V-cache size for a 2.048 context |
| --- | --- | --- | --- | --- | --- | --- |
| StableLM-Alpha-7B | CC-BY-SA-4.0 | ✅ | 1,500B | 700 | 38.3* | 800MB |
| LLaMA-7B | LLaMA license | ❌ | 1,000B | 500 | 47.6 | 1,100MB |
| MPT-7B | Apache 2.0 | ✅ | 1,000B | 500 | 48.6 | 1,100MB |
| Falcon-7B | LLaMA license | ❌ | 1,500B | 700 | 48.8 | 20MB |
| LLaMA-33B | LLaMA license | ❌ | 1,500B | 3200 | 56.9 | 3,300MB |
| LLaMA-65B | LLaMA license | ❌ | 1,500B | 6300 | 58.3 | 5,400MB |
| Falcon-40B | Apache 2.0 | ✅ | 1,000B | 2800 | 60.4 | 240MB |

**score from the base version not available, we report the tuned version instead.*


# Demo

You can easily try the Big Falcon Model (40 billion parameters!) in [this Space](https://huggingface.co/spaces/tiiuae/falcon-chat) or in the playground embedded below:

[TODO: verify this actually works:]

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.32.0/gradio.js"> </script>
<gradio-app space="tiiuae/falcon-chat"></gradio-app>

Under the hood, this playground uses 🤗 `text-generation-inference` ([repo](https://github.com/huggingface/text-generation-inference)), a scalable Rust, Python, and gRPC server for text generation. It's the same technology that powers [HuggingChat](https://huggingface.co/chat/).

We've also built a Core ML version of the 7B instruct model, and this is how it runs on an M1 Macbook Pro:

[TODO: embed video]

The video shows a lightweight app that leverages a Swift library for the heavy lifting: model loading, tokenization, input preparation, generation, and decoding. We are busy building this library to empower developers to integrate powerful LLMs in all types of applications without having to reinvent the wheel. It's still a bit rough, but we can't wait to share it with you. Meanwhile, you can download the [Core ML weights](https://huggingface.co/tiiuae/falcon-7b-instruct/tree/main/coreml/text-generation) from the repo and explore them yourself!


# Inference

You can use the familiar transformers APIs to run the models on your own hardware, but you need to pay attention to a couple of details:

- The models were trained using the `bfloat16` datatype, so we recommend you use the same. This requires a recent version of CUDA and works best on modern cards. You may also try to run inference using `float16`, but keep in mind that the models were evaluated using `bfloat16`.
- You need to allow remote code execution. This is because the models use a new architecture that is not part of `transformers` yet - instead, the code necessary is provided by the model authors in the repo. Specifically, these are the files whose code will be used if you allow remote execution (using `falcon-7b-instruct` as an example): [configuration_RW.py](https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/configuration_RW.py), [modelling_RW.py](https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/modelling_RW.py).

With these considerations, you can use the transformers `pipeline` API to load the 7B instruction model like this:

```python
from transformers import AutoTokenizer
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

```

And then, you'd run text generation using code like the following:

```python
sequences = pipeline(
   "Has humankind ever set foot on the Moon?",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```

And you may get something like the following:

```
Valencia, city of the sun
The city that glitters like a star
A city of a thousand colors
Where the night is illuminated by stars
Valencia, the city of my heart
Where the past is kept in a golden chest

```

### Inference of Falcon 40B

Running the 40B model is challenging because of its size: it doesn't fit in a single A100 with 80 GB of RAM. Loading in 8-bit mode, it is possible to run in about 45 GB of RAM, which fits in an A6000 (48 GB) but not in the 40 GB version of the A100. This is how you'd do it:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
```

Note, however, that mixed 8-bit inference will use `torch.float16` instead of `torch.bfloat16`, so make sure you test the results thoroughly.

If you have multiple cards and `accelerate` installed, you can take advantage of `device_map="auto"` to automatically distribute the model layers across various cards. It can even offload some layers to the CPU if necessary, but this will impact inference speed.

There's also the possibility to use [4-bit loading](https://huggingface.co/blog/4bit-transformers-bitsandbytes) using the latest version of `bitsandbytes`, `transformers` and `accelerate`. In this case, the 40B model takes ~27 GB of RAM to run. Unfortunately, this is slightly more than the memory available in cards such as 3090 or 4090, but it's enough to run on 30 or 40 GB cards.

### Text Generation Inference

## Evaluation

## Fine-tuning with PEFT

Training 10B+ sized models can be technically and computationally challenging. In this section we look at the tools available in the Hugging Face ecosystem to efficiently train extremely large models on simple hardware and show how to fine-tune the Falcon-7b on a single NVIDIA T4 (16GB - Google Colab)..

Let's see how we can train Falcon on the [Guanaco dataset](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) a filtered subset of the [Open Assistant dataset](https://huggingface.co/datasets/OpenAssistant/oasst1). With the [PEFT library](https://github.com/huggingface/peft) we can use the recent [QLoRA](https://arxiv.org/abs/2305.14314) approach to fine-tune adapters that are placed on top of the frozen 4-bit model. You can learn more about the integration of 4-bit quantized models [in this blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

Only a tiny fraction of the model is trainable when using Low Rank Adapters (LoRA) such that the number of learned parameters as well as the size of the trained artifact is dramatically reduced. As shown in the screenshot below, the saved model has only 65MB for the 7B parameters model (15GB in float16).

![repo-screenshot.png]()

More specifically, after selecting the target modules to adapt (in practice the query / key layers of the attention module), little trainable linear layers are attached close to these modules as illustrated below). The hidden states produced by these modules are then added to the original states to get the final hidden state.

![lora-animated.gif]()

Once trained, there is no need to save the entire model as the base model is kept frozen. Also it is possible to keep the model in any arbitrary dtype (int8, fp4, fp16, etc.) as long as the output hidden states from these modules are casted to the same dtype as the ones from the adapters - this is the case for bitsandbytes modules (`Linear8bitLt` and `Linear4bit` ) that returns hidden states with the same dtype as the original unquantized module.

We fine-tuned the two variants of Falcon model (7B and 40B) on the Guanaco dataset. For the 7b model, we fine-tuned the model on a single NVIDIA-T4 16GB and we fine-tuned the 40B model on a single NVIDIA A100 80GB. We used 4bit quantized base models and QLoRA method, as well as [the recent `SFTTrainer` from TRL library.](https://huggingface.co/docs/trl/main/en/sft_trainer) 

The full script to reproduce our experiments is available [here](https://gist.github.com/pacman100/1731b41f7a90a87b457e8c5415ff1c14), but only few lines of code are required to quickly run the `SFTTrainer`

```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()
```

Check out the original qlora repository: https://github.com/artidoro/qlora/blob/main/qlora.py for further work about evaluating the trained models.

### Fine-tuning Resources
- **[Colab notebook to fine-tune Falcon-7B on Guanaco dataset using 4bit and PEFT](https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing)** 
- **[Training code](https://gist.github.com/pacman100/1731b41f7a90a87b457e8c5415ff1c14)** 
- **[40B model adatpers]((https://huggingface.co/smangrul/falcon-40B-int4-peft-lora-sfttrainer))** ([logs](https://wandb.ai/smangrul/huggingface/runs/3hpqq08s/workspace?workspace=user-younesbelkada))
- **[7B model adapters](https://huggingface.co/ybelkada/falcon-7b-guanaco-lora)** ([logs](https://wandb.ai/younesbelkada/huggingface/runs/2x4zi72j?workspace=user-younesbelkada)) 

## Conclusion

