---
title: 🧨 Stable Diffusion  in JAX / Flax !
thumbnail: /blog/assets/108_stable_diffusion_jax/thumbnail.png
authors:
- user: pcuenq
- user: patrickvonplaten
---

# Stable Diffusion XL 🤝 JAX + TPUv5e

<!-- {blog_metadata} -->
<!-- {authors} -->

Running large image-generation models, such as Stable Diffusion XL in production can be challenging due to large memory requirements as well as an increased inference time.
Stable Diffusion XL counts roughly 3.5 billion parameters and therefore requires a significant amount of memory (up to 7 GB in float16 or bfloa16 precision) which makes it difficult to 
deploy in practice.

Google's recently release [TPUv5e chip](https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga) offers almost the same computing power in terms of TFLOPl compared to 
its predecessor while being significantly cheaper.

🧨 Diffusers JAX integration offers a convenient way to run SDXL on TPU via the XLA, the Accelerated Linear Algebra compiler.

In this blog post,
- 1. We state a couple of reasons why JAX + TPU + Diffusers is a powerful framework to run SDXL
- 2. Explain how you can write a simple image generation pipeline with Diffusers and JAX
- 3. Show benchmarks comparing different TPU settings

# Why JAX + TPUv5e for SDXL ?

The advantage of JAX + TPUv5e boils down essentially to two factors:

- JIT-compilation

A notable feature of JAX, is its just-in-time (jit) compilation. JIT-compilation allows the JIT-compiler to trace a function at the first run so that it can generate highly optimized TPU binaries that are re-used for subsequent calls.
The catch of JAX JIT-compilation is that it requires all input, intermediate out output shapes to be **static** meaning that they can be know before runtime. Additionally, every time we change the input or output shapes
we trigger a costly recompilation of the JIT-compiler which decreases performance.
Writing static functions often poses problems in text-generation where the output sequence length is dependent on the text-input and thus cannot be known before-hand.
In contrast, in image generation input and output shapes (*e.g. image shapes) are static and known before runtime which therefore makes it quite easy to write highly optimized text-generation inference functions in JAX.
Also input and output shapes are do not have to be changed between function calls in text-to-image generation therefore allowing us to prevent costly recompilations

- High-performance throughput for high batch sizes

Workloads can be scaled across multiple devices using JAX's [pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html), which expresses single-program multiple-data (SPMD) programs. Applying pmap to a function will compile a function with XLA, then execute in parallel on XLA devices. 
For text-to-image generation workloads this means that increasing the number of images rendered simultaneously is straightforward to implement and doesn't compromise performance.
Additionally TPUv5e chips come in multiple flavors such as 1,4 and 8-core setups which allow you to tailor the number of chips for your use case

## How to write a image generation pipeline in JAX

TODO: ...


## Benchmark

TODO: ....


## ... (more???)

