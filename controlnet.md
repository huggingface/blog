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

The amount of controlled generation the original Stable Diffusion models offer has been far from desired. For example, you won’t be able to generate images conditioned on a semantic segmentation map along with the input text prompt. ControlNet provides a minimal interface to tackle this problem allowing users to customize the generation process up to a great extent. With ControlNet, users can easily condition the generation with different spatial contexts such as a depth map, a segmentation map, a scribble, keypoints, and so on! Sky is the limit 

You can make your favorite commic character coming into life 


In this blog post, we first introduce the [`StableDiffusionControlNetPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/controlnet) and then show how it can suit various controlled outputs. 

Let’s get controlling! 

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/collage.png" width=600/>
</p>

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
the [other pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview) we offer from Diffusers. Central to the `StableDiffusionControlNetPipeline` is the `controlnet` parameter which lets you provide a particular [`ControlNetModel`](https://huggingface.co/docs/diffusers/main/en/api/models#diffusers.ControlNetModel) instance trained on specific spatial contexts as conditioning. 

We will explore different use cases with the `StableDiffusionControlNetPipeline` in this blog post. The first ControlNet model we are going to walk you through is the Canny model - this is one of the most popular models that generated some of the most amazing images you see on the internet.

Before you begin, make sure you have all the necessary libraries installed:

```python
!pip install -qqq git+https://github.com/huggingface/diffusers.git transformers
!pip install git+https://github.com/huggingface/accelerate
```

You will also to install opencv and controlnet-aux

```python
!pip install opencv-contrib-python
!pip install controlnet_aux
```

we will use the famous painting "Girl With A Pearl" for this example, let's download it and take a look

```python
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image
```

<p align="center">
<img src="https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_6_output_0.jpeg" width=600/>
</p>

Next, We will put the image through the canny pre-processor

```python
import cv2
from PIL import Image
import numpy as np

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
```

as you can see, it is essentially edge detection

```python
canny_image
```

<p align="center">
<img src="https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_10_output_0.jpeg" width=600/>
</p>

Now, we load the official Stable Diffusion 1.5 Model as well as the ControlNet model for canny edges.

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
```

To speed-up things and reduce memory, let’s enable model offloading and use the fast UniPCMultistepScheduler.

```python
from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# this command loads the individual model components on GPU on-demand.
pipe.enable_model_cpu_offload()
```

Now we are ready to run the ControlNet pipeline!


You should provide a prompt to guide the image generation process, just like what you would normally do with your Stable Diffusion img2img Pipeline. However, ControlNet will give you a lot more control over the generated image because you will be able to control the exact composition generated image with the canny edge image we just created.


It will be fun to see some images where contemporary celebrities posing for this exact same painting from the 17th century. And it's really easy to do that with ControlNet, all we have to do is to include the names of these celebrities in the prompt!


```python
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


prompt = ", best quality, extremely detailed"
prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(4)]

output = pipe(
    prompt,
    canny_image,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    generator=generator,
)

image_grid(output.images, 2, 2)
```

<p align="center">
<img src="https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_16_output_1.jpeg" width=600/>
</p>

You can combine ControlNet combines with fine-tuning too! For example, you can fine-tune a model on yourself with dreambooth, and use it to render yourself into different scenes.


In this post, we are going to use our beloved Mr Potato Head as an example to show you how to use ControlNet with dreambooth.


we can use the same ContrlNet, however instead of using the Stable Diffusion 1.5, we are going to load the [Mr Potato Head model](https://huggingface.co/sd-dreambooth-library/mr-potato-head) into our pipeline - Mr Potato Head is a Stable Diffusion model fine-tuned with Mr Potato Head concept using Dreambooth 🥔

```python
model_id = "sd-dreambooth-library/mr-potato-head"
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

Now let's make Mr Potato posing for Johannes Vermeer 😆

```python
generator = torch.Generator(device="cpu").manual_seed(2)
prompt = "a photo of sks mr potato head, best quality, extremely detailed"
output = pipe(
    prompt,
    canny_image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    generator=generator,
)
```

I think we can all agree that Mr Potato Head is not the best candidate but he tried his best and did a pretty good job in capture some of the essence! 

```python
output.images[0]
```

<p align="center">
<img src="https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_22_output_0.jpeg" width=600/>
</p>

Another really cool application of ControlNet is that you can take a pose from one image and reuse it to generate a different image with the exact same pose. So in this next example, we are going to teach a cute anime girl how to do yoga using [Open Pose ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-openpose)!


First, we will need to get some images of people doing yoga


```python
urls = "yoga1.jpeg", "yoga2.jpeg", "yoga3.jpeg", "yoga4.jpeg"
imgs = [load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/" + url) for url in urls]
```

```python
image_grid(imgs, 2, 2)
```

<p align="center">
<img src="https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_25_output_0.jpeg" width=600/>
</p>

Now let's extract yoga poses using the OpenPose pre-processors

```python
from controlnet_aux import OpenposeDetector

model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
```


```python
poses = [model(img) for img in imgs]
image_grid(poses, 2, 2)
```


<p align="center">
<img src="https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_28_output_0.jpeg" width=600/>
</p>


To use these yoga poses to generate new images, let's create a [Open Pose ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-openpose). And we are going to get our cute anime girl from [the anything4 model](https://huggingface.co/andite/anything-v4.0). So, let's create the pipeline and the Open Pose ControlNet.


```python
controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
)

model_id = "andite/anything-v4.0"
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

Now it's yoga time! 

```python
generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(4)]
prompt = " best quality, extremely detailed"
output = pipe(
    [prompt] * 4,
    poses,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    # negative_prompt="art, painting, rendering, 2d, unreal engine, oversaturated, high contrast",
    generator=generator,
)
```
```python
image_grid(output.images, 2, 2)
```

<p align="center">
<img src="https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_33_output_0.jpeg"/>
</p>


## Conclusion

We have been playing a lot with `StableDiffusionControlNetPipeline`, and our experience has been fun so far! 

We’re excited to see what the community builds on top of this pipeline. If you want to check out other pipelines and techniques supported in Diffusers that allow for controlled generation, check out our [official documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlling_generation).