---
title: "Faster fine-tuning using TRL & Unsloth"
thumbnail: /blog/assets/hf_unsloth/thumbnail.png
authors:
- user: danielhanchen
  guest: true
---

# Make LLM Fine-tuning 2x faster with unsloth and 🤗 TRL

Pulling your hair out because LLM finetuning is taking forever? In this blogpost, we introduce a lightweight tool developed by the community to make LLM finetuning go super fast!

Before diving into this blogpost, we suggest you to first have a look at our QLoRA blogpost, and be familiar with LLM fine-tuning using the 🤗 PEFT library.

## Unsloth - 2x faster, -40% memory usage, 0% accuracy degradation

The Unsloth library is a lightweight library for faster LLM fine-tuning which is fully compatible with the Hugging Face ecosystem (Hub, transformers, PEFT, TRL). The library is actively developed by the Unsloth team (Daniel and Michael) and the open source community. The library supports most NVIDIA GPUs from V100s all the way to H100s and can be used with the entire trainer suite from the TRL library (SFTTrainer, DPOTrainer, PPOTrainer). At this time of writing, Unsloth supports Llama (CodeLlama, Yi, etc) and Mistral architectures.

Unsloth works by overwriting some parts of the modeling code of models with optimized operations. By manually deriving backpropagation steps and rewriting all Pytorch modules into Triton kernels, Unsloth can both reduce memory usage and make finetuning faster. Also, there are no approximations during finetuning, so the accuracy degradation is 0%.

## Benchmarking

| Model          | Dataset   | GPU  | PEFT + TRL Time | Unsloth Time | PEFT + TRL VRAM | Unsloth VRAM |
|----------------|-----------|------|-----------------|--------------|-----------------|--------------|
| Llama-2 7b     | OASST     | T4   | 2222            | 1355 (1.64x) | 10.4GB          | 8.4GB        |
| Llama-2 7b     | Alpaca    | T4   | 1468            | 942 (1.56x)  | 7.1GB           | 6.4GB        |
| Mistral 7b     | Slim Orca | A100 | 1571            | 842 (1.87x)  | 19.4GB          | 12.4GB       |
| Code Llama 34b | Slim Orca | A100 | 1982            | 1042 (1.9x)  | 33.22GB         | 27.4GB       |

Unsloth was benchmarked on over 59 runs using 4 datasets and on Tesla T4 and A100 instances via Google Colab. QLoRA was applied to all linear layers (attention and MLP) with a rank of 16, and gradient checkpointing was on. By testing against the latest Transformers branch, which now has SDPA natively integrated, Unsloth is around 2x faster. All 59 notebooks are provided for full reproducibility, and more details are in Unsloth’s benchmarking details [here](https://unsloth.ai/blog/mistral-benchmark)

## How do I use Unsloth?

Just load your model with `FastLanguageModel.from_pretrained`! On the latest Transformers branch, you can now load pre-quantized models in 4bit! For example, you can use `unsloth/llama-2-7b-bnb-4bit`, `unsloth/llama-2-13b-bnb-4bit`, `unsloth/mistral-7b-bnb-4bit` or `unsloth/codellama-34b-bnb-4bit` to load a pre-quantized 4bit model. Currently, Unsloth only supports Llama and Mistral type architectures. Make a Github issue if you want other model support!

You will need to provide your intended maximum sequence length to `from_pretrained`. Otherwise the API is pretty much similar to Transformers’ `from_pretrained`, except that `FastLanguageModel.from_pretrained` also returns the model tokenizer.

```python
from unsloth import FastLanguageModel

# Load Llama model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-bnb-4bit", # Supports Llama, Mistral - replace this!
    max_seq_length = 2048, # Supports RoPE Scaling internally, so choose any!
    load_in_4bit = True,
)
```

Once the model has been loaded, use `FastLanguageModel.get_peft_model` to attach adapters in order to perform QLoRA fine-tuning.

```python
# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",      # Currently only supports bias = "none"
    use_gradient_checkpointing = True,
)
```

Once adapters are attached, you can use the model directly within any class from the HF ecosystem, such as the `SFTTrainer` from TRL!

## Unsloth + TRL integration

To use Unsloth with the TRL library, simply pass the Unsloth model into `SFTTrainer` or `DPOTrainer`! The trained model is fully compatible with the Hugging Face ecosystem, so you can push the final model on the Hub and use transformers for inference out of the box!

## Reproducible notebooks

We are sharing below fully reproducible notebooks for anyone that wants to try out Unsloth with SFTTrainer on a free-tier Google Colab instance.

Llama 7b Free Tesla T4 colab example: https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing

Mistral 7b Free Tesla T4 colab example: 
https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing

CodeLlama 34b A100 colab example:
https://colab.research.google.com/drive/1y7A0AxE3y8gdj4AVkl2aZX47Xu3P1wJT?usp=sharing
