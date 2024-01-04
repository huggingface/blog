---
title: "Welcome aMUSEd: Efficient Text-to-Image Generation"
thumbnail: /blog/assets/amused/thumbnail.png
authors:
- user: Isamu136
  guest: true
- user: valhalla
- user: williamberman
- user: sayakpaul
---


# Welcome aMUSEd: Efficient Text-to-Image Generation

![amused_grid](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/amused/main_image_grid.jpeg)

We’re excited to present an efficient non-diffusion text-to-image model named **aMUSEd**. It’s called so because it’s a open reproduction of [Google's MUSE](https://muse-model.github.io/). aMUSEd’s generation quality is not the best and we’re releasing a research preview with a permissive license. 

In contrast to the commonly used latent diffusion approach [(Rombach et al. (2022))](https://arxiv.org/abs/2112.10752), aMUSEd employs a Masked Image Model (MIM) methodology. This not only requires fewer inference steps, as noted by [Chang et al. (2023)](https://arxiv.org/abs/2301.00704), but also enhances the model's interpretability.

Just as MUSE, aMUSEd demonstrates an exceptional ability for style transfer using a single image, a feature explored in depth by [Sohn et al. (2023)](https://arxiv.org/abs/2306.00983). This aspect could potentially open new avenues in personalized and style-specific image generation.

In this blog post, we will give you some internals of aMUSEd, show how you can use it for different tasks, including text-to-image, and show how to fine-tune it. Along the way, we will provide all the important resources related to aMUSEd, including its training code. Let’s get started 🚀

## Table of contents

* [How does it work?](#how-does-it-work)
* [Using in `diffusers`](#using-amused-in-🧨-diffusers)
* [Fine-tuning aMUSEd](#fine-tuning-amused)
* [Limitations](#limitations)
* [Resources](#resources)

We have built a demo for readers to play with aMUSEd. You can try it out in [this Space](https://huggingface.co/spaces/amused/amused) or in the playground embedded below:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.45.1/gradio.js"> </script>
<gradio-app theme_mode="light" space="amused/amused"></gradio-app>


## How does it work?

aMUSEd is based on ***Masked Image Modeling***. It makes for a compelling use case for the community to explore components that are known to work in language modeling in the context of image generation. 

The figure below presents a pictorial overview of how aMUSEd works. 

![amused_architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/amused/amused_architecture.png)

During ***training***:

- input images are tokenized using a VQGAN to obtain image tokens
- the image tokens are then masked according to a cosine masking schedule.
- the masked tokens (conditioned on the prompt embeddings computed using a [CLIP-L/14 text encoder](https://huggingface.co/openai/clip-vit-large-patch14) are passed to a [U-ViT](https://arxiv.org/abs/2301.11093) model that predicts the masked patches

During ***inference***:

- input prompt is embedded using the [CLIP-L/14 text encoder](https://huggingface.co/openai/clip-vit-large-patch14).
- iterate till `N` steps are reached:
    - start with randomly masked tokens and pass them to the U-ViT model along with the prompt embeddings
    - predict the masked tokens and only keep a certain percentage of the most confident predictions based on the `N` and mask schedule. Mask the remaining ones and pass them off to the U-ViT model
- pass the final output to the VQGAN decoder to obtain the final image

As mentioned at the beginning, aMUSEd borrows a lot of similarities from MUSE. However, there are some notable differences:

- aMUSEd doesn’t follow a two-stage approach for predicting the final masked patches.
- Instead of using T5 for text conditioning, CLIP L/14 is used for computing the text embeddings.
- Following Stable Diffusion XL (SDXL), additional conditioning, such as image size and cropping, is passed to the U-ViT. This is referred to as “micro-conditioning”.

To learn more about aMUSEd, we recommend reading the technical report [here](https://huggingface.co/papers/2401.01808). 

## Using aMUSEd in 🧨 diffusers

aMUSEd comes fully integrated into 🧨 diffusers. To use it, we first need to install the libraries: 

```bash
pip install -U diffusers accelerate transformers -q
```

Let’s start with text-to-image generation:

```python
import torch
from diffusers import AmusedPipeline

pipe = AmusedPipeline.from_pretrained(
    "amused/amused-512", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "A mecha robot in a favela in expressionist style"
negative_prompt = "low quality, ugly"

image = pipe(prompt, negative_prompt=negative_prompt, generator=torch.manual_seed(0)).images[0]
image
```

![text2image_512.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/amused/text2image_512.png)

We can study how `num_inference_steps` affects the quality of the images under a fixed seed:

```python
from diffusers.utils import make_image_grid 

images = []
for step in [5, 10, 15]:
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=step, generator=torch.manual_seed(0)).images[0]
    images.append(image)

grid = make_image_grid(images, rows=1, cols=3)
grid
```

![image_grid_t2i_amused.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/amused/image_grid_t2i_amused.png)

Crucially, because of its small size (only ~800M parameters, including the text encoder and VQ-GAN), aMUSEd is very fast. The figure below provides a comparative study of the inference latencies of different models, including aMUSEd:

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/amused/amused_speed_comparison.png" alt="Speed Comparison">
  <figcaption>Tuples, besides the model names, have the following format: (timesteps, resolution). Benchmark conducted on A100. More details are in the technical report.</figcaption>
</figure>


As a direct byproduct of its pre-training objective, aMUSEd can do image inpainting zero-shot, unlike other models such as SDXL. 

```python
import torch
from diffusers import AmusedInpaintPipeline
from diffusers.utils import load_image
from PIL import Image

pipe = AmusedInpaintPipeline.from_pretrained(
    "amused/amused-512", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a man with glasses"
input_image = (
    load_image(
        "https://huggingface.co/amused/amused-512/resolve/main/assets/inpainting_256_orig.png"
    )
    .resize((512, 512))
    .convert("RGB")
)
mask = (
    load_image(
        "https://huggingface.co/amused/amused-512/resolve/main/assets/inpainting_256_mask.png"
    )
    .resize((512, 512))
    .convert("L")
)   

image = pipe(prompt, input_image, mask, generator=torch.manual_seed(3)).images[0]
```

![inpainting_grid_amused.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/amused/inpainting_grid_amused.png)

aMUSEd is the first non-diffusion system within `diffusers`. Its iterative scheduling approach for predicting the masked patches made it a good candidate for `diffusers`. We are excited to see how the community leverages it. 

We encourage you to check out the technical report to learn about all the tasks we explored with aMUSEd. 

## Fine-tuning aMUSEd

We provide a simple [training script](https://github.com/huggingface/diffusers/blob/main/examples/amused/train_amused.py) for fine-tuning aMUSEd on custom datasets. With the 8-bit Adam optimizer and float16 precision, it's possible to fine-tune aMUSEd with just under 11GBs of GPU VRAM. With LoRA, the memory requirements get further reduced to just 7GBs. 

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/amused/finetuned_amused_result.png" alt="Fine-tuned result.">
  <figcaption>a pixel art character with square red glasses</figcaption>
</figure>

aMUSEd comes with an OpenRAIL license, and hence, it’s commercially friendly to adapt. Refer to [this directory](https://github.com/huggingface/diffusers/tree/main/examples/amused) for more details on fine-tuning. 

## Limitations

aMUSEd is not a state-of-the-art image generation regarding image quality. We released aMUSEd to encourage the community to explore non-diffusion frameworks such as MIM for image generation. We believe MIM’s potential is underexplored, given its benefits:

- Inference efficiency
- Smaller size, enabling on-device applications
- Task transfer without requiring expensive fine-tuning
- Advantages of well-established components from the language modeling world 

_(Note that the original work on MUSE is close-sourced)_

For a detailed description of the quantitative evaluation of aMUSEd, refer to the technical report. 

We hope that the community will find the resources useful and feel motivated to improve the state of MIM for image generation. 

## Resources

**Papers**:

- [*Muse:* Text-To-Image Generation via Masked Generative Transformers](https://muse-model.github.io/)
- [aMUSEd: An Open MUSE Reproduction](https://huggingface.co/papers/2401.01808)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) (T5)
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (CLIP)
- [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)
- [Simple diffusion: End-to-end diffusion for high resolution images](https://arxiv.org/abs/2301.11093) (U-ViT)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

**Code + misc**:

- [aMUSEd training code](https://github.com/huggingface/amused)
- [aMUSEd documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/amused)
- [aMUSEd fine-tuning code](https://github.com/huggingface/diffusers/tree/main/examples/amused)
- [aMUSEd models](https://huggingface.co/amused)

## Acknowledgements

Suraj led training. William led data and supported training. Patrick von Platen supported both training and data and provided general guidance. Robin Rombach did the VQGAN training and provided general guidance. Isamu Isozaki helped with insightful discussions and made code contributions.

Thanks to Patrick von Platen and Pedro Cuenca for their reviews on the blog post draft.
