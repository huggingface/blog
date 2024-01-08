---
title: "Make LLM Fine-tuning 2x faster with Unsloth and 🤗 TRL"
thumbnail: /blog/assets/hf_unsloth/thumbnail.png
authors:
- user: danielhanchen
  guest: true
---

# Make LLM Fine-tuning 2x faster with Unsloth and 🤗 TRL

Pulling your hair out because LLM fine-tuning is taking forever? In this post, we introduce a lightweight tool developed by the community to make LLM fine-tuning go super fast!

Before diving into Unsloth, it may be helpful to read our [QLoRA blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes), or be familiar with LLM fine-tuning using the 🤗 PEFT library.

## Unsloth - 2x faster, -40% memory usage, 0% accuracy degradation

The [Unsloth library](https://github.com/unslothai/unsloth) is a lightweight library for faster LLM fine-tuning which is fully compatible with the Hugging Face ecosystem (Hub, transformers, PEFT, TRL). The library is actively developed by the Unsloth team ([Daniel](https://huggingface.co/danielhanchen) and [Michael](https://github.com/shimmyshimmer)) and the open source community. The library supports most NVIDIA GPUs –from GTX 1070 all the way up to H100s–, and can be used with the entire trainer suite from the TRL library (SFTTrainer, DPOTrainer, PPOTrainer). At the time of writing, Unsloth supports the Llama (CodeLlama, Yi, etc) and Mistral architectures.

Unsloth works by overwriting some parts of the modeling code with optimized operations. By manually deriving backpropagation steps and rewriting all Pytorch modules into Triton kernels, Unsloth can both reduce memory usage and make fine-tuning faster. Crucially, accuracy degradation is 0% with respect to normal QLoRA, because no approximations are made in the optimized code.

## Benchmarking

| Model           | Dataset   | GPU  | TRL + SDPA Time (s) | Unsloth (s)  | Speedup | TRL VRAM | Unsloth VRAM |
|-----------------|-----------|------|---------------------|--------------|---------|----------|--------------|
| Llama-2 7b      | OASST     | T4   | 2222                | 1355         | 1.64x   | 10.4GB   | 8.4GB        |
| Llama-2 7b      | Alpaca    | T4   | 1468                | 942          | 1.56x   | 7.1GB    | 6.4GB        |
| Tiny Llama 1.1b | Alpaca    | T4   | 13090               | 6996         | 1.87x   | 8.57GB   | 3.7GB        |
| Mistral 7b      | Slim Orca | A100 | 1571                | 842          | 1.87x   | 19.4GB   | 12.4GB       |
| Code Llama 34b  | Slim Orca | A100 | 1982                | 1042         | 1.9x    | 33.2GB   | 27.4GB       |

Unsloth was benchmarked across 59 runs using 4 datasets on Tesla T4 and A100 Google Colab instances. QLoRA was applied to all linear layers (attention and MLP) with a rank of 16, and gradient checkpointing was on. By testing against the latest Transformers branch, which now has SDPA natively integrated, Unsloth is about 2x faster on the latest hardware. All 59 notebooks are provided for full reproducibility, and more details are in Unsloth’s benchmarking details [here](https://unsloth.ai/blog/mistral-benchmark)

## How do I use Unsloth?

Just load your model with `FastLanguageModel.from_pretrained`! Currently, Unsloth supports Llama and Mistral type architectures (Yi, Deepseek, TinyLlama, Llamafied Qwen). Make a Github issue if you want others! Also, on the latest Transformers branch, you can now load pre-quantized 4bit models directly! This makes downloading models 4x faster, and reduces memory fragmentation by around 500MB, which allows you to fit larger batches! We have a few pre-quantized models for your convenience, including `unsloth/llama-2-7b-bnb-4bit`, `unsloth/llama-2-13b-bnb-4bit`, `unsloth/mistral-7b-bnb-4bit` and `unsloth/codellama-34b-bnb-4bit`.

You will need to provide your intended maximum sequence length to `from_pretrained`. Unsloth internally performs RoPE Scaling, so larger maximum sequence lengths are automatically supported. Otherwise the API is pretty much the same as transformers’ `from_pretrained`, except that `FastLanguageModel.from_pretrained` also returns the model tokenizer for convenience.

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
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",      # Currently only supports bias = "none"
    use_gradient_checkpointing = True,
)
```

Once adapters are attached, you can use the model directly within any class from the HF ecosystem, such as the `SFTTrainer` from TRL!

## Unsloth + TRL integration

To use Unsloth with the TRL library, simply pass the Unsloth model into `SFTTrainer` or `DPOTrainer`! The trained model is fully compatible with the Hugging Face ecosystem, so you can push the final model to the Hub and use transformers for inference out of the box!

```python
import torch

from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

from unsloth import FastLanguageModel

max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
# Get dataset
dataset = load_dataset("imdb", split="train")

# Load Llama model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-bnb-4bit", # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 4,
      warmup_steps = 10,
      max_steps = 60,
      fp16 = not torch.cuda.is_bf16_supported(),
      bf16 = torch.cuda.is_bf16_supported(),
      logging_steps = 1,
      output_dir = "outputs",
      optim = "adamw_8bit",
      seed = 3407,
  ),
)
trainer.train()
```

## Reproducible notebooks

We are sharing below fully reproducible notebooks for anyone that wants to try out Unsloth with SFTTrainer on a free-tier Google Colab instance.

Llama 7b Free Tesla T4 colab example [here](https://huggingface.co/datasets/unsloth/notebooks/blob/main/Alpaca_%2B_Llama_7b_full_example.ipynb)

Mistral 7b Free Tesla T4 colab example [here](https://huggingface.co/datasets/unsloth/notebooks/blob/main/Alpaca_%2B_Mistral_7b_full_example.ipynb)

CodeLlama 34b A100 colab example [here](https://huggingface.co/datasets/unsloth/notebooks/blob/main/Alpaca_%2B_Codellama_34b_full_example.ipynb)

Zephyr DPO replication T4 colab example [here](https://huggingface.co/datasets/unsloth/notebooks/blob/main/DPO_Zephyr_Unsloth_Example.ipynb)
