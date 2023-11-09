---
title: "SDXL in 4 steps with Latent Consistency LoRAs"
thumbnail: /blog/assets/lcm_sdxl/thumbnail.jpg
authors:
- user: pcuenq
- user: valhalla
- user: sayakpaul
- user: multimodalart
- user: SimianLuo
  guest: true
- user: dg845
  guest: true
---

# SDXL in 4 steps with Latent Consistency LoRAs

Latent Consistency Models (LCM) are a way to decrease the number of steps required to generate an image with Stable Diffusion (or SDXL) by _distilling_ the original model into another version that requires fewer steps (4 to 8 instead of the original 25 to 50). Distillation is a type of training procedure that attempts to replicate the outputs from a source model using a new one. The distilled model may be designed to be smaller (that’s the case, for instance, of DistilBERT or the recently-released DistilWhisper or SSD-1B) or, in this case, require fewer steps to run. It’s usually a lengthy and costly process that requires huge amounts of data, patience, and a few GPUs.

Well, that was the status quo before today!

We are delighted to announce a new method that can essentially make Stable Diffusion and SDXL faster, as if they had been distilled using the LCM process!

## Contents

- [Method Overview](#method-overview)
- [Why does this matter](#why-does-this-matter)
- [Fast Inference with SDXL LCM LoRAs](#fast-inference-with-sdxl-lcm-loras)
  - [Quality Comparison](#quality-comparison)
  - [Guidance Scale and Negative Prompts](#guidance-scale-and-negative-prompts)
  - [Quality vs base SDXL](#quality-vs-base-sdxl)
- [Benchmarks](#benchmarks)
- [LCM LoRAs and Models Released Today](#lcm-loras-and-models-released-today)
- [Bonus: Use LCM LoRAs with regular SDXL LoRAs](#bonus-use-lcm-loras-with-regular-sdxl-loras)
- [How to train LCM LoRAs](#how-to-train-lcm-loras)
- [Resources](#resources)
- [Credits](#credits)

## Method Overview

So, what’s the trick? The central idea is to use a LoRA trained on an LCM model and apply it to a non-LCM model, together with the noise scheduler of LCM models. If you are itching to see how this looks in practice, just jump to the [next section](#fast-inference-with-sdxl-lcm-loras) to play with the inference code. If you want to train your own LoRAs, this is the process you’d use:

1. Select an available LCM model from the Hub. For example, for SDXL we recommend you start with ….
2. Train a LoRA with your desired characteristics (style, dataset) on the LCM model. LoRA is a type of performance-efficient fine-tuning, or PEFT, that is much cheaper to accomplish than full model fine-tuning. For additional details on LoRA training, please check ….
3. Use the LoRA with the standard SDXL diffusion model and the LCM scheduler and bingo, you get high-quality inference in just a few steps.

## Why does this matter?

Fast inference of Stable Diffusion and SDXL enables new use-cases and workflows. To name a few:

- **Accessibility**: generative tools can be used effectively by more people, even if they don’t have access to the latest hardware.
- **Faster iteration**: get more images and multiple variants in a fraction of the time! This is great for artists and researchers; whether for personal or commercial use.
- Production workloads may be possible on different accelerators, including CPUs.
- Cheaper image generation services.

To gauge the speed difference we are talking about, generating a single 1024x1024 image on an M1 Mac with SDXL (base) takes about a minute. Using the LCM LoRA, we get great results in just ~6s (4 steps). This is an order of magnitude faster, and not having to wait for results is a game-changer. Using a 4090, we get almost instant response (less than 1s). This unlocks the use of SDXL in applications where real-time events are a requirement. 

## Fast Inference with SDXL LCM LoRAs

The version of `diffusers` released today makes it very easy to use LCM LoRAs:

```py
from diffusers import DiffusionPipeline, LCMScheduler
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")

pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device="cuda", dtype=torch.float16)

prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
images = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=1,
).images[0]
```

Note how the code:
- Instantiates a standard diffusion pipeline with the SDXL 1.0 base model.
- Applies the LCM LoRA.
- Changes the scheduler to the LCMScheduler, which is the one used in latent consistency models.
- That’s it!

This would result in the following full-resolution image:

TODO: image

### Quality Comparison

Let’s see how the number of steps impacts generation quality. The following code will generate images with 1 to 8 total inference steps:

```py
images = []
for steps in range(8):
    generator = torch.Generator(device=pipe.device).manual_seed(1337)
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps+1,
        guidance_scale=1,
        generator=generator,
    ).images[0]
    images.append(image)
```

These are the 8 images displayed in a grid:

TODO: grid

As expected, using just **1** step produces an approximate shape without discernible features and lacking texture. However, results quickly improve, and they are usually very satisfactory in just 4 to 6 steps. Personally, I find the 8-step image in the previous test to be a bit too saturated and “cartoony” for my taste, so I’d probably choose between the ones with 5 and 6 steps in this example. Generation is so fast that you can create a bunch of different variants using just 4 steps, and then select the ones you like and iterate using a couple more steps and refined prompts as necessary.

### Guidance Scale and Negative Prompts

Note that in the previous examples we used a `guidance_scale` of `1`, which effectively disables it. This works well for most prompts, and it’s fastest, but ignores negative prompts. You can also explore using negative prompts by providing a guidance scale between `1` and `2` – we found that larger values don’t work.

### Quality vs base SDXL

How does this compare against the standard SDXL pipeline, in terms of quality? Let’s see an example!

We can quickly revert our pipeline to a standard SDXL pipeline by unloading the LoRA weights and switching to the default scheduler:

```py
from diffusers import EulerDiscreteScheduler

pipe.unload_lora_weights()
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
```

Then we can run inference as usual for SDXL. We’ll gather results using varying number of steps:

```py
images = []
for steps in (1, 4, 8, 15, 20, 25, 30, 50):
    generator = torch.Generator(device=pipe.device).manual_seed(1337)
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    images.append(image)
```

TODO: grid

As you can see, images in this example are pretty much useless until ~20 steps (second row), and quality still increases niteceably with more steps. The details in the final image are amazing, but it took 50 steps to get there.

## Benchmarks

This section is not meant to be exhaustive, but illustrative of the generation speed we achieve on various computers. Let us stress again how liberating it is to explore image generation so easily.

| Hardware                               | SDXL LoRA LCM (4 steps) | SDXL standard (25 steps) |
|----------------------------------------|-------------------------|--------------------------|
| Mac, M1 Max                            | 6.5s                    | 64s                      |
| 2080 Ti                                | 4.7s                    | 10.2s                    |
| 3090                                   | 1.4s                    | 7s                       |
| 4090                                   | 0.7s                    | 3.4s                     |
| T4 (Google Colab Free Tier)            | 8.4s                    | 26.5s                    |
| A100 (80 GB)                           | 1.2s                    | 3.8s                     |
| Intel i9-10980XE CPU (1/36 cores used) | 29s                     | 219s                     |

These tests were run with a batch size of 1 in all cases. For cards with a lot of capacity, such as A100, performance increases significantly when generating multiple images at once, which is usually the case for production workloads.

## LCM LoRAs and Models Released Today

- [`latent-consistency/lcm-lora-sdxl`](https://huggingface.co/latent-consistency/lcm-lora-sdxl). LCM LoRA for [SDXL 1.0 base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), as seen in the examples above.
- [`latent-consistency/lcm-lora-sdv1-5`](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5). LCM LoRA for [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5).
- [`latent-consistency/lcm-lora-ssd-1b`](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b). LCM LoRA for [`segmind/SSD-1B`](https://huggingface.co/segmind/SSD-1B), a distilled SDXL model that's 50% smaller and 60% faster than the original SDXL.

- [`latent-consistency/lcm-sdxl`](https://huggingface.co/latent-consistency/lcm-sdxl). Full fine-tuned consistency model derived from [SDXL 1.0 base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
- [`latent-consistency/lcm-ssd-1b`](https://huggingface.co/latent-consistency/lcm-ssd-1b). Full fine-tuned consistency model derived from [`segmind/SSD-1B`](https://huggingface.co/segmind/SSD-1B).

## Bonus: Use LCM LoRAs with regular SDXL LoRAs

Using the [diffusers + PEFT integration](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference), you can combine LCM LoRAs with regular SDXL LoRAs, giving them the superpower to run LCM inference in only 4 steps.

Here we are going to combine `CiroN2022/toy_face` LoRA with the LCM LoRA:

```py
from diffusers import DiffusionPipeline, LCMScheduler
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(lcm_lora_id)
pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")

pipe.set_adapters(["lora", "toy"], adapter_weights=[1.0, 0.8])
pipe.to(device="cuda", dtype=torch.float16)

prompt = "a toy_face man"
negative_prompt = "blurry, low quality, render, 3D, oversaturated"
images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=4,
    guidance_scale=0.5,
).images[0]
images
```

TODO: image

How about any SD1.5 base model? It also works!

Let's use the [`collage-diffusion`](https://huggingface.co/wavymulder/collage-diffusion) model to test. Note that this is a fine-tuned model, not a LoRA:

```py
from diffusers import DiffusionPipeline, LCMScheduler
import torch

model_id = "wavymulder/collage-diffusion"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lcm_lora_id)
pipe.to(device="cuda", dtype=torch.float16)

prompt = "collage style kid sits looking at the night sky, full of stars"
negative_prompt = "blurry, low quality, render, 3D, oversaturated"

generator = torch.Generator(device=pipe.device).manual_seed(1337)
images = pipe(
    prompt=prompt,
    generator=generator,
    negative_prompt=negative_prompt,
    num_inference_steps=4,
    guidance_scale=0.2,
).images[0]
images
```

TODO: image

https://huggingface.co/plasmo/woolitize

```py
from diffusers import DiffusionPipeline, LCMScheduler
import torch

model_id = "plasmo/woolitize"
lcm_lora_id = "lcm-sd/lcm-sd1.5-lora"

pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lcm_lora_id, weight_name="lcm_sd_lora.safetensors", adapter_name="lcm")
pipe.to(device="cuda", dtype=torch.float16)

prompt = "woolitize portrait of Bill Gates"
negative_prompt = "blurry, low quality, render, 3D, oversaturated"

generator = torch.Generator(device=pipe.device).manual_seed(12312)
images = pipe(
    prompt=prompt,
    generator=generator,
    negative_prompt=negative_prompt,
    num_inference_steps=4,
    guidance_scale=0.3,
).images[0]
images
```

TODO: image

## How to train LCM LoRAs

## Resources

## Credits

