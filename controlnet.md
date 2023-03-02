---
title: "Welcoming ControlNet to Diffusers 🧨" 
thumbnail: /blog/assets/controlnet/thumbnail.png <!-- UPDATE -->
authors:
- user: takuma104
  guest: true
- user: patrickvonplaten
- user: YiYiXu
---

# Welcoming ControlNet to Diffusers 🧨

<!-- {blog_metadata} -->
<!-- {authors} -->

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

<a target="_blank" href="https://colab.research.google.com/github/nateraw/huggingface-hub-examples/blob/main/vit_image_classification_explained.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> <!-- Update the Colab Notebook link-->

Today we’re thrilled to announce that Diffusers now officially supports ControlNet!

The amount of controlled generation the original Stable Diffusion models offer has been far from desired. For example, you won’t be able to generate images conditioned on a semantic segmentation map along with the input text prompt. ControlNet provides a minimal interface to tackle this problem allowing users to customize the generation process up to a great extent. With ControlNet, users can easily condition the generation with different spatial contexts such as a depth map, a segmentation map, a scribble, keypoints, and so on! 

In this blog post, we first introduce the [`StableDiffusionControlNetPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/controlnet) and then show how it can suit various controlled outputs. 

Let’s get controlling! 

***[Include a collage of cool examples?]***

## ControlNet: TL;DR

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) by Lvmin Zhang and Maneesh Agrawala. It introduces a framework that allows for supporting various spatial contexts that can serve as additional conditionings to Diffusion models such as Stable Diffusion. It is comprised of the following steps:

1. Cloning the pre-trained parameters of a Diffusion model such as Stable Diffusion (referred to as “trainable copy”) while also maintaining the pre-trained parameters separately (”locked copy”). It is done so that the locked parameter copy can preserve the vast knowledge learned from a large dataset, whereas the trainable copy is employed to learn task-specific aspects. 
2. The trainable and locked copies of the parameters are connected via “zero convolution” layers which are optimized as a part of the ControlNet framework.  

A sample from the training set for ControlNet-like training looks like this (additional conditioning is via edge maps):

<div align="center">
    <table>
    <tr>
        <th>Original Image</th>
        <th>Conditioning</th>
        <th>Prompt</th>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/original_bird.png" width=300/></td>
        <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/input_canny.png" width=300/></td>
        <td>"bird"</td>
    </tr>
    </table>
</div>

Similarly, if we were to condition ControlNet with semantic segmentation maps, a training sample would been like so:

<div align="center">
    <table>
    <tr>
        <th>Original Image</th>
        <th>Conditioning</th>
        <th>Prompt</th>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/original_house.png" width=300/></td>
        <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/segmentation_map.png" width=300/></td>
        <td>"big house"</td>
    </tr>
    </table>
</div>

## The `StableDiffusionControlNetPipeline`

To experiment with ControlNet, we expose `StableDiffusionControlNetPipeline` similar to
the [other pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview) we offer from Diffusers. Central to the `StableDiffusionControlNetPipeline` is the `controlnet` parameter which lets you provide a particular `ControlNet` instance trained
on specific spatial contexts as conditioning. 

We will explore different use cases with the `StableDiffusionControlNetPipeline` in this blog post, but for starters, let’s load a `StableDiffusionControlNetPipeline` that can generate images conditioned on edge maps. First, we obtain an edge map using OpenCV:

```python
from PIL import Image
import cv2

low_threshold = 100
high_threshold = 200
canny = cv2.Canny(
    np.asarray(your_image), low_threshold, high_threshold
)[:,:,None]
canny = Image.fromarray(np.concatenate([canny] * 3, axis=-1))
```

The figure below shows an example edge map of an input image extracted using the above code:

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/input_canny.png" width="500" />
</p>

Next, we load the `StableDiffusionControlNetPipeline` with a `ControlNet` model that was specifically trained to generate images conditioned on edge maps like above:

```python
import torch 

weight_dtype = torch.float16
device = "cuda"

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-canny",torch_dtype=weight_dtype
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet,
    torch_dtype=weight_dtype
).to(device)
```

Now, time for some birds 🐦

```python
prompt = "bird"
generator = torch.manual_seed(0)
image = pipe(prompt, canny, generator=generator, height=768, width=512).images[0]
```

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/bird_of_my_life.png" width="500" />
</p>

And some more 🦅 🕊️

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/birds_of_my_life.png" width="500" />
</p>

That was enough birds. In the next section, let's explore different use cases
where ControlNet is useful.

## Support Multiple Use Cases with `StableDiffusionControlNetPipeline`

***FILL ME***

## Conclusion

We have been playing a lot with `StableDiffusionControlNetPipeline`, and our experience has been fun so far! 

We’re excited to see what the community builds on top of this pipeline. If you want to check out other pipelines and techniques supported in Diffusers that allow for controlled generation, check out our [official documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlling_generation).