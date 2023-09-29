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

TODO: Intro

## Why JAX + TPUv5e for SDXL?

TODO: ...

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

We are now ready to setup our prompt and the rest of the pipeline inputs.

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

# Initial random seed of 'key'
rng = jax.random.PRNGKey(seed)

def replicate_all(prompt_ids, neg_prompt_ids, rng):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng
```

We are now ready to put everything together in a generate function:

```Python
def generate(
    prompt,
    negative_prompt,
    rng=rng,
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
    return pipeline.numpy_to_pil(np.array(images)), rng
```

`jit=True` indicates that we want the pipeline call to be compiled. This will happen the first time we call `generate`, and it will be very slow – JAX needs to trace the operations, optimize them and convert to low-level primitives. We'll run a first generation to complete this process and warm things up:

```Python
start = time.time()
print(f"Compiling ...")
generate(default_prompt, default_neg_prompt, rng)
print(f"Compiled in {time.time() - start}")
```

TODO: put the output here

Once the code has been compiled, inference will be super fast. Let's try again!

```Python
start = time.time()
prompt = "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
neg_prompt = "cartoon, illustration, animation. face. male, female"
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")
```

TODO: put the output

## Benchmark

TODO: ....


## ... (more???)
