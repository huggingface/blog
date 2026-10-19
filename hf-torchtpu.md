---
title: "Hugging Face ❤️ TorchTPU: TPUs, the Native Way"
thumbnail: /blog/assets/hf-torchtpu/thumbnail.png
authors:
- user: tengomucho
- user: Jingya
- user: pagezyhf
- user: jeffboudier
- user: regisss
---

# Hugging Face ❤️ TorchTPU: TPUs, Natively

So far, TPUs meant leaving your tools: porting PyTorch code to *torch_xla* or rewriting it in JAX. `torch_tpu` is a native PyTorch backend: TPU is just another *device*. It is also natively supported by Transformers, Diffusers, Kernels and the whole Hugging Face ecosystem, enabling users to benefit from the speed of TPUs and to share their models on the HF Hub easily.

## TorchTPU

[TorchTPU](https://github.com/google-pytorch/torch_tpu) enables users to run models with PyTorch on TPU devices through a new TPU backend. It features PyTorch Pallas kernels tailored for TPUs, support for distributed training and an integration with *torch.compile*.

<!-- TODO: Link to Google's blog post? -->
<!-- TODO: Command to install TorchTPU? -->

## Native support in the HF ecosystem

### Where were we

Until now, running PyTorch on TPUs meant going through PyTorch/XLA (`torch_xla`). It worked, but it came with some friction: lazy tensor execution, graph recompilations, and debugging that could be hard to reason about. To smooth this out for our users, we built Optimum TPU, which provided inference and training support for Hugging Face models on TPUs. With native support now arriving through TorchTPU, Optimum TPU is deprecated.

### Transformers

Transformers v5 has been designed around one core principle: being the source of truth for model definitions so that they can be run anywhere. Thus, deploying your favourite model on TPU to generate text is as easy as it gets: you only need to pass `device_map="tpu"` when instantiating your model. For instance:

```py
model_id = "Qwen/Qwen3.5-9B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="tpu", # will load model directly on the TPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("The future of AI is", return_tensors="pt").to("tpu")
out = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

The model can work in eager mode by default, but you can also use `torch.compile` to get better performance and use it in a distributed setup.

### Diffusers

Diffusers now runs on Google Cloud TPUs through TorchTPU, with support for both eager and compiled modes. Once TorchTPU is installed, `pipe.to("tpu")` is the only change most pipelines need under compiled mode. We’ve also added TPU support for tensor parallelism in Diffusers, allowing models that are too large for a single chip to be distributed across multiple chips.

```py
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    dtype=torch.bfloat16,
)
pipe.to("tpu")
pipe.transformer.compile(backend="tpu", fullgraph=True, dynamic=False)
```

### Kernels

The HF Kernels hub contains ready-to-use and optimized TPU kernels that you can access [here](https://huggingface.co/kernels?sort=trending&search=tpu). You can use them with the Kernels library to speed your models up.
TPU kernels can be written with [Pallas](https://docs.jax.dev/en/latest/pallas/tpu/), which gives low-level control over TPU instructions. They could even be used to boost your inference with the kernels available for [Tokamax](https://github.com/openxla/tokamax), like with this ragged-dot kernel example:

```py
repo_id = "tengomucho/ragged-dot-tpu"
kernel = get_kernel(repo_id, version=1, trust_remote_code=[repo_id])
ALL_EXPERTS_FUNCTIONS.register("tokamax_ragged_dot", kernel.experts_forward)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    experts_implementation="tokamax_ragged_dot",
)
```

<!-- TODO: add image (upload to huggingface/documentation-images under blog/hf-torchtpu/) -->

## CI Dashboard

The whole Transformers CI is run on TPUs to make sure we offer the best possible support. You can see more details in our CI dashboard: <!-- TODO: add link -->

The whole Hugging Face ecosystem now natively supports TPU through TorchTPU. Transformers v5 has been designed around one core principle: being the source of truth for model definitions so that they can be run anywhere. Thus, deploying your favourite model on TPU to generate text is as easy as it gets:

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B",
    device_map="tpu", # will load model directly on the TPU
)
```

You only need to pass the TPU device to `device_map`.
The Diffusers integration has the same objective: image and video generation on TPU is possible with just a few lines of code. This includes tensor parallelism, which will enable users to fully benefit from the power of TPUs.
