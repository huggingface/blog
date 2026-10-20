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

<img src="https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/banner.jpg" alt="The Hugging Face mascot and a smiling PyTorch logo cheering over a glowing chip, on a manga-style background." width="624" style="max-width: 100%; height: auto;">

If you ever tried to run PyTorch on a TPU, you probably ported your code to `torch_xla`, or you ended up having to learn JAX. With [TorchTPU](https://github.com/google-pytorch/torch_tpu), a native PyTorch backend, TPU is now just another device. It opens up native support of TPU in the Hugging Face ecosystem. Transformers, Diffusers and Kernels run natively on TPU. Pick a model from the Hub, send it to `"tpu"`, and you're done.

## Wait, what's a TPU? 🤔

TPU stands for [Tensor Processing Unit](https://cloud.google.com/tpu). TPUs are Google's own AI accelerators. They are chips built for one job: the matrix math that neural networks spend most of their time on. Google has been designing them for about ten years, and uses them to train and serve its own models, including Gemini.

You don't need to work at Google to use one. TPUs are available on [Google Cloud](https://docs.cloud.google.com/tpu/docs/intro-to-tpu), from a single chip up to pods of thousands of chips. They shine when you need a lot of compute, for training or for serving at scale.

The catch has always been software. The TPU stack grew up around TensorFlow and JAX, and PyTorch users had to work harder to get in.

## The old way

Until now, running PyTorch on TPUs meant going through PyTorch/XLA (`torch_xla`). It worked, but it had its quirks. Tensors were lazy (the tensors, not you), so your code didn't always run when you thought it did. Small changes could trigger graph recompilations, and debugging was hard to reason about.

To make life easier, we built Optimum TPU. It gave Hugging Face models inference and training support on TPUs. Now that TorchTPU brings native support, Optimum TPU is deprecated. It did its job, and it has earned its retirement. 🫡

## Enter TorchTPU 🚀

[TorchTPU](https://github.com/google-pytorch/torch_tpu) adds TPU as a new backend to PyTorch. You write regular PyTorch code. It comes with Pallas kernels tuned for TPUs, support for distributed training, and a `torch.compile` integration for when you want more speed. Installing it on a TPU VM takes one command:

```shell
pip install torch-tpu --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index gives you the CPU build of `torch`. Without it, pip pulls the much bigger CUDA build, which you don't need on a TPU.

## Your favourite libraries, now on TPU

### Transformers

Transformers v5 was built around one idea: be the source of truth for model definitions, so they can run anywhere. "Anywhere" now includes TPUs. To load a model on a TPU, pass `device_map="tpu"`. That's it.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

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

### Images? Videos? Diffusers paints them on TPUs 🎨

TPU is now a first-class device in Diffusers. For prototyping or debugging, eager mode only needs `pipe.to("tpu")`. For deployment, compiled mode adds one `compile` call. Your pipelines and code stay the same, and there's no TPU fork to maintain.

```py
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", dtype=torch.bfloat16
)
pipe.to("tpu")  # eager mode: that's the whole change
# optional: compiled mode
pipe.transformer.compile(backend="tpu", fullgraph=True, dynamic=False)

image = pipe("a corgi astronaut planting a flag on a pizza moon").images[0]
```

The first call compiles, and every call after that with the same image size reuses the graph.

Models too big for one chip? Pass a `TensorParallelConfig` to `from_pretrained` and the transformer is sharded across chips as it loads, so the full model never has to fit in one place. Many popular pipelines support it out of the box, and the ones that don't are just a `tp_plan` away. The [Diffusers TPU guide](https://huggingface.co/docs/diffusers/main/optimization/tpu) has the full recipe.

Here are some examples run on a Cloud TPU v6e-4:

<sub><i>TP=4 (each model split across four chips) · bf16 · batch 1</i></sub>

<table>
  <tr>
    <th><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMax-H3</a> (video + audio)</th>
    <th colspan="2"><a href="https://huggingface.co/Qwen/Qwen-Image-2.1">Qwen-Image-2.1</a></th>
  </tr>
  <tr>
    <td><video controls src="https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/videos/minimax-h3-hf-trireme.mp4"></video></td>
    <td><img src="https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/qwen-image-21-capybara-1024.jpg" alt="A capybara wearing a wizard hat, reading a book by candlelight, oil painting"></td>
    <td><img src="https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/qwen-image-21-neon-2048.jpg" alt="Neon shop sign reading QWEN IMAGE 2.1"></td>
  </tr>
  <tr>
    <td><sub><i>"The fluffy plush Hugging Face mascot from the picture, a round yellow smiling face with rosy cheeks and two little open hands, stands at the prow of an ancient Greek trireme […] Sound of crashing waves, creaking timber, wind in the rigging, distant seagulls, and a low epic orchestral swell."</i> · 1344×768, 5.2 s, 49 steps · 188 s compiled, 262 s eager</sub></td>
    <td><sub><i>"A capybara wearing a wizard hat, reading a book by candlelight, oil painting"</i> · 1024×1024, 40 steps · 3.5 s compiled, 8.0 s eager</sub></td>
    <td><sub><i>"A neon shop sign that reads 'QWEN IMAGE 2.1', rainy night, reflections on wet pavement"</i> · 2048×2048, 40 steps</sub></td>
  </tr>
</table>

### Kernels

The [Kernels Hub](https://huggingface.co/kernels?sort=trending&search=tpu) has ready-to-use TPU kernels. Load one with the Kernels library and give your model a speed boost.

TPU kernels are written in [Pallas](https://docs.jax.dev/en/latest/pallas/tpu/), which gives you low-level control over the hardware. You can also bring in kernels from the [Tokamax](https://github.com/openxla/tokamax) library. Here is a ragged-dot kernel from Tokamax plugged into a Mixture-of-Experts model:

```py
from kernels import get_kernel
from transformers import AutoModelForCausalLM
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

repo_id = "tengomucho/ragged-dot-tpu"
kernel = get_kernel(repo_id, version=1, trust_remote_code=[repo_id])
ALL_EXPERTS_FUNCTIONS.register("tokamax_ragged_dot", kernel.experts_forward)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    experts_implementation="tokamax_ragged_dot",
)
```

The bigger the batch, the bigger the win. At batch size 64, decoding is 21.5× faster on Qwen3-30B-A3B and 12.8× faster on Gemma 4 26B-A4B.

![Decode throughput with the Hub ragged-dot kernel versus the default grouped_mm, for Qwen3-30B-A3B and Gemma 4 26B-A4B](https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/kernel-ragged-dot-speedup.png)
*Decode throughput on a TPU v6e-8, Hub kernel vs. the default implementation.*

## We test it so you don't have to 🧪

Adding TPU support is the easy part. Keeping it from breaking is harder. So we run the Transformers model test suite on TPU, and the results land in our CI dashboard, right next to the other accelerators we test on.

![Transformers CI dashboard with TPU test results next to other accelerators](https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/tpu-ci-dashboard.png)
*The Transformers CI dashboard, with TPU results next to other accelerators. Work in progress!*

It's not all green yet, and we're fine with that. Of the tests that run on TPU, 96.5% pass. Every failure we find becomes a small repro script and an issue on the TorchTPU repo. So far we have reported 34 issues, and 19 are already fixed. Running on TPU also caught a few bugs in Transformers itself, like in `StaticCache` and the Qwen3 tensor-parallel plans. We fixed those upstream.

If something breaks on TPU, we want to find out before you do.

## Get started

TPUs are no longer a side quest for PyTorch users. Install TorchTPU, grab a model from the Hub, and send it to `"tpu"`.

- [TorchTPU on GitHub](https://github.com/google-pytorch/torch_tpu)
- [TPU kernels on the Hub](https://huggingface.co/kernels?sort=trending&search=tpu)
- [Cloud TPU documentation](https://docs.cloud.google.com/tpu/docs)

We can't wait to see what you build 🤗. If you hit a problem, open an issue and tell us about it.
