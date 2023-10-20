---
title: 🧨 Accelerating Stable Diffusion XL Inference with JAX on Cloud TPU v5e
thumbnail: /blog/assets/sdxl-jax/thumbnail.jpg
authors:
- user: pcuenq
- user: jffacevedo
  guest: true
- user: alexspiridonov
  guest: true
- user: pmotter
  guest: true
- user: yyetim
  guest: true
- user: svaibhav
  guest: true
- user: vjsingh
  guest: true
- user: patrickvonplaten
---

# Accelerating Stable Diffusion XL Inference with JAX on Cloud TPU v5e


Generative AI models, such as Stable Diffusion XL (SDXL), enable the creation of high-quality, realistic content with wide-ranging applications. However, harnessing the power of such models presents significant challenges and computational costs. SDXL is a large image generation model whose UNet component is about three times as large as the one in the previous version of the model. Deploying a model like this in production is challenging due to the increased memory requirements, as well as increased inference times. Today, we are thrilled to announce that Hugging Face Diffusers now supports serving SDXL using JAX on Cloud TPUs, enabling high-performance, cost-efficient inference.

[Google Cloud TPUs](https://cloud.google.com/tpu) are custom-designed AI accelerators, which are optimized for training and inference of large AI models, including state-of-the-art LLMs and generative AI models such as SDXL. The new [Cloud TPU v5e](https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga) is purpose-built to bring the cost-efficiency and performance required for large-scale AI [training](https://cloud.google.com/blog/products/compute/using-cloud-tpu-multislice-to-scale-ai-workloads) and [inference](https://cloud.google.com/blog/products/compute/how-cloud-tpu-v5e-accelerates-large-scale-ai-inference). At less than half the cost of TPU v4, TPU v5e makes it possible for more organizations to train and deploy AI models.

🧨 Diffusers JAX integration offers a convenient way to run SDXL on TPU via [XLA](https://github.com/openxla/xla), and we built a demo to showcase it. You can try it out in [this Space](https://huggingface.co/spaces/google/sdxl) or in the playground embedded below:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.45.1/gradio.js"> </script>
<gradio-app theme_mode="light" space="google/sdxl"></gradio-app>

Under the hood, this demo runs on several TPU v5e-4 instances (each instance has 4 TPU chips) and takes advantage of parallelization to serve four large 1024×1024 images in about 4 seconds. This time includes format conversions, communications time, and frontend processing; the actual generation time is about 2.3s, as we'll see below!

In this blog post,
1. [We describe why JAX + TPU + Diffusers is a powerful framework to run SDXL](#why-jax--tpu-v5e-for-sdxl)
2. [Explain how you can write a simple image generation pipeline with Diffusers and JAX](#how-to-write-an-image-generation-pipeline-in-jax)
3. [Show benchmarks comparing different TPU settings](#benchmark)

## Why JAX + TPU v5e for SDXL?

Serving SDXL with JAX on Cloud TPU v5e with high performance and cost-efficiency is possible thanks to the combination of purpose-built TPU hardware and a software stack optimized for performance. Below we highlight two key factors: JAX just-in-time (jit) compilation and XLA compiler-driven parallelism with JAX pmap.

#### JIT compilation

A notable feature of JAX is its [just-in-time (jit) compilation](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html). The JIT compiler traces code during the first run and generates highly optimized TPU binaries that are re-used in subsequent calls.
The catch of this process is that it requires all input, intermediate, and output shapes to be **static**, meaning that they must be known in advance. Every time we change the shapes
a new and costly compilation process will be triggered again. JIT compilation is ideal for services that can be designed around static shapes: compilation runs once, and then we take advantage of super-fast inference times.

Image generation is well-suited for JIT compilation. If we always generate the same number of images and they have the same size, then the output shapes are constant and known in advance. The text inputs are also constant: by design, Stable Diffusion and SDXL use fixed-shape embedding vectors (with padding) to represent the prompts typed by the user. Therefore, we can write JAX code that relies on fixed shapes, and that can be greatly optimized!

#### High-performance throughput for high batch sizes

Workloads can be scaled across multiple devices using JAX's [pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html), which expresses single-program multiple-data (SPMD) programs. Applying pmap to a function will compile a function with XLA, then execute it in parallel on various XLA devices. 
For text-to-image generation workloads this means that increasing the number of images rendered simultaneously is straightforward to implement and doesn't compromise performance. For example, running SDXL on a TPU with 8 chips will generate 8 images in the same time it takes for 1 chip to create a single image.

TPU v5e instances come in multiple shapes, including 1, 4 and 8-chip shapes, all the way up to 256 chips (a full TPU v5e pod), with ultra-fast ICI links between chips. This allows you to choose the TPU shape that best suits your use case and easily take advantage of the parallelism that JAX and TPUs provide.

## How to write an image generation pipeline in JAX

We'll go step by step over the code you need to write to run inference super-fast using JAX! First, let's import the dependencies.

```python
# Show best practices for SDXL JAX
import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from diffusers import FlaxStableDiffusionXLPipeline
import time
```

We'll now load the base SDXL model and the rest of the components required for inference. The diffusers pipeline takes care of downloading and caching everything for us. Adhering to JAX's functional approach, the model's parameters are returned separately and will have to be passed to the pipeline during inference:

```python
pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", split_head_dim=True
)
```

Model parameters are downloaded in 32-bit precision by default. To save memory and run computation faster we'll convert them to `bfloat16`, an efficient 16-bit representation. However, there's a caveat: for best results, we have to keep the _scheduler state_ in `float32`, otherwise precision errors accumulate and result in low-quality or even black images.

```python
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state
```

We are now ready to set up our prompt and the rest of the pipeline inputs.

```python
default_prompt = "high-quality photo of a baby dolphin ​​playing in a pool and wearing a party hat"
default_neg_prompt = "illustration, low-quality"
default_seed = 33
default_guidance_scale = 5.0
default_num_steps = 25
```

The prompts have to be supplied as tensors to the pipeline, and they always have to have the same dimensions across invocations. This allows the inference call to be compiled. The pipeline `prepare_inputs` method performs all the necessary steps for us, so we'll create a helper function to prepare both our prompt and negative prompt as tensors. We'll use it later from our `generate` function:

```python
def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids
```

To take advantage of parallelization, we'll replicate the inputs across devices. A Cloud TPU v5e-4 has 4 chips, so by replicating the inputs we get each chip to generate a different image, in parallel. We need to be careful to supply a different random seed to each chip so the 4 images are different:

```python
NUM_DEVICES = jax.device_count()

# Model parameters don't change during inference,
# so we only need to replicate them once.
p_params = replicate(params)

def replicate_all(prompt_ids, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng
```

We are now ready to put everything together in a generate function:

```python
def generate(
    prompt,
    negative_prompt,
    seed=default_seed,
    guidance_scale=default_guidance_scale,
    num_inference_steps=default_num_steps,
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    images = pipeline(
        prompt_ids,
        p_params,
        rng,
        num_inference_steps=num_inference_steps,
        neg_prompt_ids=neg_prompt_ids,
        guidance_scale=guidance_scale,
        jit=True,
    ).images

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1], ) + images.shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))
```

`jit=True` indicates that we want the pipeline call to be compiled. This will happen the first time we call `generate`, and it will be very slow – JAX needs to trace the operations, optimize them, and convert them to low-level primitives. We'll run a first generation to complete this process and warm things up:

```python
start = time.time()
print(f"Compiling ...")
generate(default_prompt, default_neg_prompt)
print(f"Compiled in {time.time() - start}")
```

This took about three minutes the first time we ran it.
But once the code has been compiled, inference will be super fast. Let's try again!

```python
start = time.time()
prompt = "llama in ancient Greece, oil on canvas"
neg_prompt = "cartoon, illustration, animation"
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")
```

It now took about 2s to generate the 4 images!

## Benchmark

The following measures were obtained running SDXL 1.0 base for 20 steps, with the default Euler Discrete scheduler. We compare Cloud TPU v5e with TPUv4 for the same batch sizes. Do note that, due to parallelism, a TPU v5e-4 like the ones we use in our demo will generate **4 images** when using a batch size of 1 (or 8 images with a batch size of 2). Similarly, a TPU v5e-8 will generate 8 images when using a batch size of 1.

The Cloud TPU tests were run using Python 3.10 and jax version 0.4.16. These are the same specs used in our [demo Space](https://huggingface.co/spaces/google/sdxl).

|                 | Batch Size | Latency  | Perf/$ |
|-----------------|------------|:--------:|--------|
| TPU v5e-4 (JAX) | 4          | 2.33s    | 21.46  |
|                 | 8          | 4.99s    | 20.04  |
| TPU v4-8 (JAX)  | 4          | 2.16s    | 9.05   |
|                 | 8          | 4.17     | 8.98   |

TPU v5e achieves up to 2.4x greater perf/$ on SDXL compared to TPU v4, demonstrating the cost-efficiency of the latest TPU generation.

To measure inference performance, we use the industry-standard metric of throughput. First, we measure latency per image when the model has been compiled and loaded. Then, we calculate throughput by dividing batch size over latency per chip. As a result, throughput measures how the model is performing in production environments regardless of how many chips are used. We then divide throughput by the list price to get performance per dollar.

## How does the demo work?

The [demo we showed before](https://huggingface.co/spaces/google/sdxl) was built using a script that essentially follows the code we posted in this blog post. It runs on a few Cloud TPU v5e devices with 4 chips each, and there's a simple load-balancing server that routes user requests to backend servers randomly. When you enter a prompt in the demo, your request will be assigned to one of the backend servers, and you'll receive the 4 images it generates.

This is a simple solution based on several pre-allocated TPU instances. In a future post, we'll cover how to create dynamic solutions that adapt to load using GKE.

All the code for the demo is open-source and available in Hugging Face Diffusers today. We are excited to see what you build with Diffusers + JAX + Cloud TPUs!
