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

Google's [Cloud TPU v5e](https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga), which was recently announced, offers almost the same computing power in terms of TFLOP compared to 
its predecessor, while being significantly cheaper.

🧨 Diffusers JAX integration offers a convenient way to run SDXL on TPU via XLA. You can try it out in [this demo Space](https://huggingface.co/spaces/google/sdxl) or in the playground embedded below:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.37.0/gradio.js"> </script>
<gradio-app theme_mode="light" space="google/sdxl"></gradio-app>

Under the hood, this demo runs on several TPU v5e-4 instances, and takes advantage of parallelization to create 4 large 1024×1024 images in about 4 seconds, including all communications overhead!

In this blog post,
- 1. We state a couple of reasons why JAX + TPU + Diffusers is a powerful framework to run SDXL
- 2. Explain how you can write a simple image generation pipeline with Diffusers and JAX
- 3. Show benchmarks comparing different TPU settings

## Why JAX + TPUv5e for SDXL ?

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

## How to write an image generation pipeline in JAX

We'll go step by step over the code you need to write to run inference super-fast using JAX! First, let's import the dependencies.

```Python
# Show best practices for SDXL JAX
import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from diffusers import FlaxStableDiffusionXLPipeline
import time
```

We'll now load the the base SDXL model and the rest of components required for inference. The diffusers pipeline takes care of downloading and caching everything for us. Adhering to JAX's functional approach, the model's parameters are returned seperatetely and will have to be passed to the pipeline during inference:

```Python
pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", split_head_dim=True
)
```

Model parameters are downloaded in 32-bit by default. To save memory and run computation faster we'll convert them to `bfloat16`, an efficient 16-bit representation. However, there's a caveat: for best results we have to keep the _scheduler state_ in `float32`, otherwise precision errors accumulate and result in low-quality or even black images.

```Python
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state
```

We are now ready to set up our prompt and the rest of the pipeline inputs.

```Python
default_prompt = "high-quality photo of a baby dolphin ​​playing in a pool and wearing a party hat"
default_neg_prompt = "illustration, low-quality"
default_seed = 33
default_guidance_scale = 5.0
default_num_steps = 25
```

The prompts have to be supplied as tensors to the pipeline, and they always have to have the same dimensions across invocations. This allows the inference call to be compiled. The pipeline `prepare_inputs` method performs all the necessary steps for us, so we'll create a helper function to prepare both our prompt and negative prompt as tensors. We'll use it later from our `generate` function:

```Python
def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids
```

To take advantage of parallelization, we'll replicate the inputs across devices. A Cloud TPU v5e-4 has 4 cores, so by replicating the inputs we get each core to generate a different image, in parallel. We need to be careful to supply a different random seed to each core so the 4 images are different:

```Python
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

```Python
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

`jit=True` indicates that we want the pipeline call to be compiled. This will happen the first time we call `generate`, and it will be very slow – JAX needs to trace the operations, optimize them and convert to low-level primitives. We'll run a first generation to complete this process and warm things up:

```Python
start = time.time()
print(f"Compiling ...")
generate(default_prompt, default_neg_prompt)
print(f"Compiled in {time.time() - start}")
```

TODO: put the output time here

Once the code has been compiled, inference will be super fast. Let's try again!

```Python
start = time.time()
prompt = "llama in ancient Greece, oil on canvas"
neg_prompt = "cartoon, illustration, animation"
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")
```

TODO: put the output

## Benchmark

The following measures were obtained running SDXL 1.0 base for 25 steps, with the default Euler Discrete scheduler. We used Python 3.10 and jax version 0.4.16. These are the same specs used in our [demo Space](#).

## ... (more???)

