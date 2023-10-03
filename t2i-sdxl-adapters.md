---
title: "Efficient Controllable Generation for SDXL with T2I-Adapters"
thumbnail: /blog/assets/t2i-sdxl-adapters/thumbnail.png
authors:
- user: Adapter
  guest: true
- user: valhalla
- user: sayakpaul
- user: Xintao
  guest: true
- user: hysts
---

# Efficient Controllable Generation for SDXL with T2I-Adapters


<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/hf_tencent.png" height=180/>
</p>

[T2I-Adapter](https://huggingface.co/papers/2302.08453) is an efficient plug-and-play model that provides extra guidance to pre-trained text-to-image models while freezing the original large text-to-image models. T2I-Adapter aligns internal knowledge in T2I models with external control signals. We can train various adapters according to different conditions and achieve rich control and editing effects.

As a contemporaneous work, [ControlNet](https://hf.co/papers/2302.05543) has a similar function and is widely used. However, it can be **computationally expensive** to run. This is because, during each denoising step of the reverse diffusion process, both the ControlNet and UNet need to be run. In addition, ControlNet emphasizes the importance of copying the UNet encoder as a control model, resulting in a larger parameter number. Thus, the generation is bottlenecked by the size of the ControlNet (the larger, the slower the process becomes). 

T2I-Adapters provide a competitive advantage to ControlNets in this matter. T2I-Adapters are smaller in size, and unlike ControlNets, T2I-Adapters are run just once for the entire course of the denoising process. 

| **Model Type** | **Model Parameters** | **Storage (fp16)** |
| --- | --- | --- |
| [ControlNet-SDXL](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) | 1251 M | 2.5 GB |
| [ControlLoRA](https://huggingface.co/stabilityai/control-lora) (with rank 128) | 197.78 M (84.19% reduction)  | 396 MB (84.53% reduction) |
| [T2I-Adapter-SDXL](https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0) | 79 M (**_93.69% reduction_**) | 158 MB (**_94% reduction_**) |

Over the past few weeks, the Diffusers team and the T2I-Adapter authors have been collaborating to bring the support of T2I-Adapters for [Stable Diffusion XL (SDXL)](https://huggingface.co/papers/2307.01952) in [`diffusers`](https://github.com/huggingface/diffusers). In this blog post, we share our findings from training T2I-Adapters on SDXL from scratch, some appealing results, and, of course, the T2I-Adapter checkpoints on various conditionings (sketch, canny, lineart, depth, and openpose)!

![Collage of the results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/results_collage.png)

Compared to previous versions of T2I-Adapter (SD-1.4/1.5), [T2I-Adapter-SDXL](https://github.com/TencentARC/T2I-Adapter) still uses the original recipe, driving 2.6B SDXL with a 79M Adapter! T2I-Adapter-SDXL maintains powerful control capabilities while inheriting the high-quality generation of SDXL!

## Training T2I-Adapter-SDXL with `diffusers`

We built our training script on [this official example](https://github.com/huggingface/diffusers/blob/main/examples/t2i_adapter/README_sdxl.md) provided by `diffusers`. 

Most of the T2I-Adapter models we mention in this blog post were trained on 3M high-resolution image-text pairs from LAION-Aesthetics V2 with the following settings: 

- Training steps: 20000-35000
- Batch size: Data parallel with a single GPU batch size of 16 for a total batch size of 128.
- Learning rate: Constant learning rate of 1e-5.
- Mixed precision: fp16

We encourage the community to use our scripts to train custom and powerful T2I-Adapters, striking a competitive trade-off between speed, memory, and quality. 

## Using T2I-Adapter-SDXL in `diffusers`

Here, we take the lineart condition as an example to demonstrate the usage of [T2I-Adapter-SDXL](https://github.com/TencentARC/T2I-Adapter/tree/XL). To get started, first install the required dependencies:

```bash
pip install -U git+https://github.com/huggingface/diffusers.git
pip install -U controlnet_aux==0.0.7 # for conditioning models and detectors
pip install transformers accelerate 
```

The generation process of the T2I-Adapter-SDXL mainly consists of the following two steps:

1. Condition images are first prepared into the appropriate *control image* format.
2. The *control image* and *prompt* are passed to the [`StableDiffusionXLAdapterPipeline`](https://github.com/huggingface/diffusers/blob/0ec7a02b6a609a31b442cdf18962d7238c5be25d/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L126).

Let's have a look at a simple example using the [Lineart Adapter](https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0). We start by initializing the T2I-Adapter pipeline for SDXL and the lineart detector. 

```python
import torch
from controlnet_aux.lineart import LineartDetector
from diffusers import (AutoencoderKL, EulerAncestralDiscreteScheduler,
                       StableDiffusionXLAdapterPipeline, T2IAdapter)
from diffusers.utils import load_image, make_image_grid

# load adapter
adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
).to("cuda")

# load pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler"
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id,
    vae=vae,
    adapter=adapter,
    scheduler=euler_a,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# load lineart detector
line_detector = LineartDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
```

Then, load an image to detect lineart:

```python
url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_lin.jpg"
image = load_image(url)
image = line_detector(image, detect_resolution=384, image_resolution=1024)
```

![Lineart Dragon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/lineart_dragon.png)

Then we generate: 

```python
prompt = "Ice dragon roar, 4k photo"
negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
gen_images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    num_inference_steps=30,
    adapter_conditioning_scale=0.8,
    guidance_scale=7.5,
).images[0]
gen_images.save("out_lin.png")
```

![Lineart Generated Dragon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/lineart_generated_dragon.png)

There are two important arguments to understand that help you control the amount of conditioning.

1. `adapter_conditioning_scale`
    
    This argument controls how much influence the conditioning should have on the input. High values mean a higher conditioning effect and vice-versa. 
    
2. `adapter_conditioning_factor`
    
    This argument controls how many initial generation steps should have the conditioning applied. The value should be set between 0-1 (default is 1). The value of `adapter_conditioning_factor=1` means the adapter should be applied to all timesteps, while the `adapter_conditioning_factor=0.5` means it will only applied for the first 50% of the steps.

For more details, we welcome you to check the [official documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/adapter). 

## Try out the Demo

You can easily try T2I-Adapter-SDXL in [this Space](https://huggingface.co/spaces/TencentARC/T2I-Adapter-SDXL) or in the playground embedded below:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.43.1/gradio.js"></script>
<gradio-app src="https://tencentarc-t2i-adapter-sdxl.hf.space"></gradio-app>

You can also try out [Doodly](https://huggingface.co/spaces/TencentARC/T2I-Adapter-SDXL-Sketch), built using the sketch model that turns your doodles into realistic images (with language supervision):

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.43.1/gradio.js"></script>
<gradio-app src="https://tencentarc-t2i-adapter-sdxl-sketch.hf.space"></gradio-app>

## More Results

Below, we present results obtained from using different kinds of conditions. We also supplement the results with links to their corresponding pre-trained checkpoints. Their model cards contain more details on how they were trained, along with example usage. 

### Lineart Guided

![Lineart guided more results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/lineart_guided.png)
*Model from [`TencentARC/t2i-adapter-lineart-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0)*

### Sketch Guided

![Sketch guided results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/sketch_guided.png)
*Model from [`TencentARC/t2i-adapter-sketch-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-sketch-sdxl-1.0)*

### Canny Guided

![Sketch guided results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/canny_guided.png)
*Model from [`TencentARC/t2i-adapter-canny-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0)*

### Depth Guided

![Depth guided results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/depth_guided.png)
*Depth guided models from [`TencentARC/t2i-adapter-depth-midas-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-depth-midas-sdxl-1.0) and [`TencentARC/t2i-adapter-depth-zoe-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-depth-zoe-sdxl-1.0) respectively*

### OpenPose Guided

![OpenPose guided results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/pose_guided.png)
*Model from [`TencentARC/t2i-adapter-openpose-sdxl-1.0`](https://hf.co/TencentARC/t2i-adapter-openpose-sdxl-1.0)*

---

*Acknowledgements: Immense thanks to [William Berman](https://twitter.com/williamLberman) for helping us train the models and sharing his insights.*
