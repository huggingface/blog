---
title: "Introducing Würtschen: Fast Diffusion for Image Generation" 
thumbnail: /blog/assets/wuertschen/thumbnail.png
authors:
- user: dome272
  guest: true
- user: babbleberns
  guest: true
- user: kashif
- user: sayakpaul
- user: pcuenq
---

# Introducing Würtschen: Fast Diffusion for Image Generation

<!-- {blog_metadata} -->
<!-- {authors} -->

![Collage of images created with Würtschen](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/wuertschen/collage_compressed.jpg)

## What is Würtschen?

Würstchen is a diffusion model, whose text-conditional model works in a highly compressed latent space of images. Why is this important? Compressing data can reduce computational costs for both training and inference by magnitudes. Training on 1024x1024 images is way more expensive than training on 32x32. Usually, other works make use of a relatively small compression, in the range of 4x - 8x spatial compression. Würstchen takes this to an extreme. Through its novel design, it achieves a 42x spatial compression. This was unseen before because common methods fail to faithfully reconstruct detailed images after 16x spatial compression. Würstchen employs a two-stage compression, what we call Stage A and Stage B. Stage A is a VQGAN, and Stage B is a Diffusion Autoencoder (more details can be found in the  **[paper](https://arxiv.org/abs/2306.00637)**). A third model, Stage C, is learned in that highly compressed latent space. This training requires fractions of the compute used for current top-performing models, while also allowing cheaper and faster inference.

![Würstchen images with Prompts](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/wuertschen/generated_images.jpg)

## Why another text-to-image model?

Well, this one is pretty fast. Würstchen’s biggest benefits come from the fact that it can generate images much faster than models like Stable Diffusion XL while using much less memory. So for all the consumer graphic cards, who don’t have A100 lying around, this will come in handy. Here is a comparison with SDXL over different batch sizes.

![Inference Speed Plots](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/wuertschen/inference_speed_v2.jpg)

Here is also a real-time comparison between both models and their inference speeds.

However, probably the most significant benefit of Würstchen comes with the reduced training costs. Würstchen v1, which works at 512x512, required only 9.000 GPU hours of training. Comparing this to what Stable Diffusion 1.4 needed, namely 150,000 GPU hours, shows that this 16x reduction in costs not only benefits researchers when conducting new experiments, it also opens the doors for more companies to train such models. Würstchen v2 used 24,602 GPU hours. With resolutions going up to 1536, this is still 6x cheaper than SD1.4 which was only trained at 512x512.

![Inference Speed Plots](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/wuertschen/compute_comparison.jpg)

You can also find a detailed explanation video here:

<iframe width="708" height="398" src="https://www.youtube.com/embed/ogJsCPqgFMk" title="Efficient Text-to-Image Training (16x cheaper than Stable Diffusion) | Paper Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## How to use Würstchen?

The model is available through the Diffusers Library.

```Python
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipeline = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to("cuda")

caption = "Anthropomorphic cat dressed as a firefighter"
images = pipeline(
	caption,
	height=1024,
	width=1536,
	prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
	prior_guidance_scale=4.0,
	num_images_per_prompt=4,
).images
```

![Anthropomorphic cat dressed as a fire-fighter](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/wuertschen/Anthropomorphic_cat_dressed_as_a_fire_fighter.jpg)

More [speed-ups](https://huggingface.co/docs/diffusers/optimization/torch2.0) can be achieved by using `torch.compile`:

```Python
pipeline.prior_prior = torch.compile(pipeline.prior_prior , mode="reduce-overhead", fullgraph=True)
pipeline.decoder = torch.compile(pipeline.decoder, mode="reduce-overhead", fullgraph=True)
```

or by using [xFormers](https://facebookresearch.github.io/xformers/)'s memory efficient attention:

```Python
pipeline.enable_xformers_memory_efficient_attention()
```
