---
title: "SDXL in 4 steps with Latent Consistency LoRAs"
thumbnail: /blog/assets/lcm_sdxl/lcm_thumbnail.png
authors:
- user: pcuenq
- user: valhalla
- user: SimianLuo
  guest: true
- user: dg845
  guest: true
- user: tyq1024
  guest: true
- user: sayakpaul
- user: multimodalart
---

# SDXL in 4 steps with Latent Consistency LoRAs

[Latent Consistency Models (LCM)](https://huggingface.co/papers/2310.04378) are a way to decrease the number of steps required to generate an image with Stable Diffusion (or SDXL) by _distilling_ the original model into another version that requires fewer steps (4 to 8 instead of the original 25 to 50). Distillation is a type of training procedure that attempts to replicate the outputs from a source model using a new one. The distilled model may be designed to be smaller (that’s the case of [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) or the recently-released [Distil-Whisper](https://github.com/huggingface/distil-whisper)) or, in this case, require fewer steps to run. It’s usually a lengthy and costly process that requires huge amounts of data, patience, and a few GPUs.

Well, that was the status quo before today!

We are delighted to announce a new method that can essentially make Stable Diffusion and SDXL faster, as if they had been distilled using the LCM process! How does it sound to run _any_ SDXL model in about 1 second instead of 7 on a 3090, or 10x faster on Mac? Read on for details!

## Contents

- [Method Overview](#method-overview)
- [Why does this matter](#why-does-this-matter)
- [Fast Inference with SDXL LCM LoRAs](#fast-inference-with-sdxl-lcm-loras)
  - [Quality Comparison](#quality-comparison)
  - [Guidance Scale and Negative Prompts](#guidance-scale-and-negative-prompts)
  - [Quality vs base SDXL](#quality-vs-base-sdxl)
  - [LCM LoRAs with other Models](#lcm-loras-with-other-models)
  - [Full Diffusers Integration](#full-diffusers-integration)
- [Benchmarks](#benchmarks)
- [LCM LoRAs and Models Released Today](#lcm-loras-and-models-released-today)
- [Bonus: Combine LCM LoRAs with regular SDXL LoRAs](#bonus-combine-lcm-loras-with-regular-sdxl-loras)
- [How to train LCM LoRAs](#how-to-train-lcm-loras)
- [Resources](#resources)
- [Credits](#credits)

## Method Overview

So, what’s the trick? 
For latent consistency distillation, each model needs to be distilled separately. The core idea with LCM LoRA is to train just a small number of adapters, [known as LoRA layers](https://huggingface.co/docs/peft/conceptual_guides/lora), instead of the full model. The resulting LoRAs can then be applied to any fine-tuned version of the model without having to distil them separately. If you are itching to see how this looks in practice, just jump to the [next section](#fast-inference-with-sdxl-lcm-loras) to play with the inference code. If you want to train your own LoRAs, this is the process you’d use:

1. Select an available teacher model from the Hub. For example, you can use [SDXL (base)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), or any fine-tuned or dreamboothed version you like.
2. [Train a LCM LoRA](#how-to-train-lcm-models-and-loras) on the model. LoRA is a type of performance-efficient fine-tuning, or PEFT, that is much cheaper to accomplish than full model fine-tuning. For additional details on PEFT, please check [this blog post](https://huggingface.co/blog/peft) or [the diffusers LoRA documentation](https://huggingface.co/docs/diffusers/training/lora).
3. Use the LoRA with any SDXL diffusion model and the LCM scheduler; bingo! You get high-quality inference in just a few steps.

For more details on the process, please [download our paper](https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/LCM-LoRA-Technical-Report.pdf).

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

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/lcm-lora/lcm-1.jpg?download=true" alt="SDXL in 4 steps with LCM LoRA"><br>
    <em>Image generated with SDXL in 4 steps using an LCM LoRA.</em>
</p>

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

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/lcm-lora/lcm-grid.jpg?download=true" alt="LCM LoRA generations with 1 to 8 steps"><br>
    <em>LCM LoRA generations with 1 to 8 steps.</em>
</p>

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

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/lcm-lora/lcm-sdxl-grid.jpg?download=true" alt="SDXL results for various inference steps"><br>
    <em>SDXL pipeline results (same prompt and random seed), using 1, 4, 8, 15, 20, 25, 30, and 50 steps.</em>
</p>

As you can see, images in this example are pretty much useless until ~20 steps (second row), and quality still increases noticeably with more steps. The details in the final image are amazing, but it took 50 steps to get there.

### LCM LoRAs with other models

This technique also works for any other fine-tuned SDXL or Stable Diffusion model. To demonstrate, let's see how to run inference on [`collage-diffusion`](https://huggingface.co/wavymulder/collage-diffusion), a model fine-tuned from [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) using Dreambooth.

The code is similar to the one we saw in the previous examples. We load the fine-tuned model, and then the LCM LoRA suitable for Stable Diffusion v1.5.

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

generator = torch.Generator(device=pipe.device).manual_seed(1337)
images = pipe(
    prompt=prompt,
    generator=generator,
    negative_prompt=negative_prompt,
    num_inference_steps=4,
    guidance_scale=1,
).images[0]
images
```

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/lcm-lora/collage.png?download=true" alt="LCM LoRA technique with a Dreambooth Stable Diffusion v1.5 model, allowing 4-step inference."><br>
    <em>LCM LoRA technique with a Dreambooth Stable Diffusion v1.5 model, allowing 4-step inference.</em>
</p>

### Full Diffusers Integration

The integration of LCM in `diffusers` makes it possible to take advantage of many features and workflows that are part of the diffusers toolbox. For example:

- Out of the box `mps` support for Macs with Apple Silicon.
- Memory and performance optimizations like flash attention or `torch.compile()`.
- Additional memory saving strategies for low-RAM environments, including model offload.
- Workflows like ControlNet or image-to-image.
- Training and fine-tuning scripts.


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

These tests were run with a batch size of 1 in all cases, using [this script](https://huggingface.co/datasets/pcuenq/gists/blob/main/sayak_lcm_benchmark.py) by [Sayak Paul](https://huggingface.co/sayakpaul).

For cards with a lot of capacity, such as A100, performance increases significantly when generating multiple images at once, which is usually the case for production workloads.

## LCM LoRAs and Models Released Today

- [Latent Consistency Models LoRAs Collection](https://huggingface.co/collections/latent-consistency/latent-consistency-models-loras-654cdd24e111e16f0865fba6)
    - [`latent-consistency/lcm-lora-sdxl`](https://huggingface.co/latent-consistency/lcm-lora-sdxl). LCM LoRA for [SDXL 1.0 base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), as seen in the examples above.
    - [`latent-consistency/lcm-lora-sdv1-5`](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5). LCM LoRA for [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5).
    - [`latent-consistency/lcm-lora-ssd-1b`](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b). LCM LoRA for [`segmind/SSD-1B`](https://huggingface.co/segmind/SSD-1B), a distilled SDXL model that's 50% smaller and 60% faster than the original SDXL.

- [`latent-consistency/lcm-sdxl`](https://huggingface.co/latent-consistency/lcm-sdxl). Full fine-tuned consistency model derived from [SDXL 1.0 base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
- [`latent-consistency/lcm-ssd-1b`](https://huggingface.co/latent-consistency/lcm-ssd-1b). Full fine-tuned consistency model derived from [`segmind/SSD-1B`](https://huggingface.co/segmind/SSD-1B).

## Bonus: Combine LCM LoRAs with regular SDXL LoRAs

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

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/lcm-lora/lcm-toy.png?download=true" alt="Combining LoRAs for fast inference"><br>
    <em>Standard and LCM LoRAs combined for fast (4 step) inference.</em>
</p>

Need ideas to explore some LoRAs? Check out our experimental [LoRA the Explorer (LCM version)](https://huggingface.co/spaces/latent-consistency/lcm-LoraTheExplorer) Space to test amazing creations by the community and get inspired!


## How to Train LCM Models and LoRAs

As part of the `diffusers` release today, we are providing training and fine-tuning scripts developed in collaboration with the LCM team authors. They allow users to:

- Perform full-model distillation of Stable Diffusion or SDXL models on large datasets such as Laion.
- Train LCM LoRAs, which is a much easier process. As we've shown in this post, it also makes it possible to run fast inference with Stable Diffusion, without having to go through distillation training.

For more details, please check the instructions for [SDXL](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/README_sdxl.md) or [Stable Diffusion](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/README.md) in the repo.

We hope these scripts inspire the community to try their own fine-tunes. Please, do let us know if you use them for your projects!

## Resources

- Latent Consistency Models [project page](https://latent-consistency-models.github.io), [paper](https://huggingface.co/papers/2310.04378).
- [LCM LoRAs](https://huggingface.co/collections/latent-consistency/latent-consistency-models-loras-654cdd24e111e16f0865fba6)
    - [For SDXL](https://huggingface.co/latent-consistency/lcm-lora-sdxl).
    - [For Stable Diffusion v1.5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5).
    - [For Segmind's SSD-1B](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b).
    - [Technical Report](https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/LCM-LoRA-Technical-Report.pdf).
- Demos
    - [SDXL in 4 steps with Latent Consistency LoRAs](https://huggingface.co/spaces/latent-consistency/lcm-lora-for-sdxl)
    - [Near real-time video stream](https://huggingface.co/spaces/latent-consistency/Real-Time-LCM-ControlNet-Lora-SD1.5)

- [LoRA the Explorer (experimental LCM version)](https://huggingface.co/spaces/latent-consistency/lcm-LoraTheExplorer)
- PEFT: [intro](https://huggingface.co/blog/peft), [repo](https://github.com/huggingface/peft)
- Training scripts
    - [For Stable Diffusion 1.5](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/README.md)
    - [For SDXL](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/README_sdxl.md)

## Credits

The amazing work on Latent Consistency Models was performed by the [LCM Team](https://latent-consistency-models.github.io), please make sure to check out their code, report and paper. This project is a collaboration between the [diffusers team](https://github.com/huggingface/diffusers), the LCM team, and community contributor [Daniel Gu](https://huggingface.co/dg845). We believe it's a testament to the enabling power of open source AI, the cornerstone that allows researchers, practitioners and tinkerers to explore new ideas and collaborate. We'd also like to thank [`@madebyollin`](https://huggingface.co/madebyollin) for their continued contributions to the community, including the `float16` autoencoder we use in our training scripts.
