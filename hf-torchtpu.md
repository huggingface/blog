---
title: "Hugging Face ❤️ TorchTPU: TPUs, Natively"
thumbnail: /blog/assets/hf-torchtpu/thumbnail.png
authors:
- user: tengomucho
- user: Jingya
- user: pagezyhf
- user: jeffboudier
- user: regisss
---

# Hugging Face ❤️ TorchTPU: TPUs, Natively

If you ever tried to run PyTorch on a TPU, you probably remember the detour. You either ported your code to `torch_xla`, or you ended up learning JAX on a Friday night. Neither was the plan.

That detour is gone. [TorchTPU](https://github.com/google-pytorch/torch_tpu) is a native PyTorch backend, so a TPU is now just another device. It also works across the Hugging Face ecosystem. Transformers, Diffusers and Kernels all support it. Pick a model from the Hub, send it to `"tpu"`, and you're done.

## Wait, what's a TPU?

TPU stands for [Tensor Processing Unit](https://cloud.google.com/tpu). TPUs are Google's own AI accelerators. They are chips built for one job: the matrix math that neural networks spend most of their time on. Google has been designing them for about ten years, and uses them to train and serve its own models, including Gemini.

You don't need to work at Google to use one. TPUs are available on [Google Cloud](https://docs.cloud.google.com/tpu/docs/intro-to-tpu), from a single chip up to pods of thousands of chips. They shine when you need a lot of compute, for training or for serving at scale.

The catch has always been software. The TPU stack grew up around TensorFlow and JAX, and PyTorch users had to work harder to get in.

## The old way

Until now, running PyTorch on TPUs meant going through PyTorch/XLA (`torch_xla`). It worked, but it had its quirks. Tensors were lazy (the tensors, not you), so your code didn't always run when you thought it did. Small changes could trigger graph recompilations, and debugging was hard to reason about.

To make life easier, we built Optimum TPU. It gave Hugging Face models inference and training support on TPUs. Now that TorchTPU brings native support, Optimum TPU is deprecated. It did its job, and it has earned its retirement.

## Enter TorchTPU

[TorchTPU](https://github.com/google-pytorch/torch_tpu) adds TPU as a new backend to PyTorch. You write regular PyTorch code. It comes with Pallas kernels tuned for TPUs, support for distributed training, and a `torch.compile` integration for when you want more speed. Installing it on a TPU VM takes one command:

```shell
pip install torch-tpu --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index gives you the CPU build of `torch`. Without it, pip pulls the much bigger CUDA build, which you don't need on a TPU.

## Your favourite libraries, now on TPU

### Transformers

Transformers v5 was built around one idea: be the source of truth for model definitions, so they can run anywhere. "Anywhere" now includes TPUs. To load a model on a TPU, pass `device_map="tpu"`. That's it.

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

The model runs in eager mode by default. You can also use `torch.compile` for better performance, or run it in a distributed setup.

### Diffusers

Diffusers runs on TPUs too, in both eager and compiled modes. Once TorchTPU is installed, moving a pipeline to TPU takes one line: `pipe.to("tpu")`. Here is Stable Diffusion 3.5 in compiled mode:

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

Model too big for one chip? We added TPU support for tensor parallelism in Diffusers, so you can split it across several chips.

### Kernels

The [Kernels Hub](https://huggingface.co/kernels?sort=trending&search=tpu) has ready-to-use TPU kernels. Load one with the Kernels library and give your model a speed boost.

TPU kernels are written in [Pallas](https://docs.jax.dev/en/latest/pallas/tpu/), which gives you low-level control over the hardware. You can also bring in kernels from the [Tokamax](https://github.com/openxla/tokamax) library. Here is a ragged-dot kernel from Tokamax plugged into a Mixture-of-Experts model:

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

## We test it so you don't have to

Adding TPU support is the easy part. Keeping it from breaking is harder. So we run the Transformers model test suite on TPU, and the results land in our CI dashboard, right next to the other accelerators we test on.

<!-- TODO: upload ../tpu-test/pytorch-na-2026/assets/tpu-tcid.png to huggingface/documentation-images under blog/hf-torchtpu/ -->
![Transformers CI dashboard with TPU test results next to other accelerators](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-torchtpu/tpu-tcid.png)
*The Transformers CI dashboard, with TPU results next to other accelerators. Work in progress!*

It's not all green yet, and we're fine with that. Of the tests that run on TPU, 96.5% pass. Every failure we find becomes a small repro script and an issue on the TorchTPU repo. So far we have reported 34 issues, and 19 are already fixed. Running on TPU also caught a few bugs in Transformers itself, like in `StaticCache` and the Qwen3 tensor-parallel plans. We fixed those upstream.

If something breaks on TPU, we want to find out before you do.

## Get started

TPUs are no longer a side quest for PyTorch users. Install TorchTPU, grab a model from the Hub, and send it to `"tpu"`.

- [TorchTPU on GitHub](https://github.com/google-pytorch/torch_tpu)
- [TPU kernels on the Hub](https://huggingface.co/kernels?sort=trending&search=tpu)
- [Cloud TPU documentation](https://docs.cloud.google.com/tpu/docs)

We can't wait to see what you build. If you hit a problem, open an issue and tell us about it.
