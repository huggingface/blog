---
title: "Remote VAEs for decoding with HF endpoints 🤗"
thumbnail: /blog/assets/remote_vae/thumbnail.png
authors:
- user: hlky
- user: sayakpaul
---

# Remote VAEs for decoding with HF endpoints 🤗

When operating with latent-space diffusion models for high-resolution image and video synthesis, the VAE decoder can consume quite a bit more memory. This makes it hard for the users to run these models on consumer GPUs without going through latency sacrifices and others alike. 

For example, with offloading, there is a device transfer overhead, causing delays in the overall inference latency. Tiling is another solution that lets us operate on so-called “tiles” of inputs. However, it can have a negative impact on the quality of the final image. 

Therefore, we want to pilot an idea with the community — delegating the decoding process to a remote endpoint. 

**Table of contents**:

- [Getting started](#getting-started)
    - Basic example
    - Generation
    - Queueing
- [Available VAEs](#available-vaes)
- [Advantages of using a remote VAE](#advantages-of-using-a-remote-vae)
- [Provide feedback](#provide-feedback)

## Getting started

Below, we cover three use cases where we think this remote VAE inference would be beneficial.

### Basic example

Here, we show how to use the remote VAE on random tensors.

<details><summary>Code</summary>
<p>

```python
import io
import requests
import torch
from base64 import b64encode
from PIL import Image
from safetensors.torch import _tobytes

ENDPOINT = "https://lqmfdhmzmy4dw51z.us-east-1.aws.endpoints.huggingface.cloud/"

def remote_decode(latent: torch.Tensor) -> Image.Image:
    shape = list(latent.shape)
    dtype = str(latent.dtype).split(".")[-1]
    tensor_data = b64encode(_tobytes(latent, "inputs")).decode("utf-8")
    parameters = {"shape": shape, "dtype": dtype}
    data = {"inputs": tensor_data, "parameters": parameters}
    headers = {"Content-Type": "application/json", "Accept": "image/jpeg"}
    response = requests.post(ENDPOINT, json=data, headers=headers)
    if not response.ok:
        raise RuntimeError(response.json())
    image = Image.open(io.BytesIO(response.content))
    return image

image = remote_decode(torch.randn([1, 4, 64, 64]))
```

</p>
</details>

<figure class="image flex flex-col items-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/output.png"/>
</figure>

Usage for Flux is slightly different. Flux latents are packed so we need to send the `height` and `width`.

<details><summary>Code</summary>
<p>

```python
import io
import requests
import torch
from base64 import b64encode
from PIL import Image
from safetensors.torch import _tobytes

ENDPOINT = "https://zy1z7fzxpgtltg06.us-east-1.aws.endpoints.huggingface.cloud"

def remote_decode(latent: torch.Tensor, height: int, width: int) -> Image.Image:
    shape = list(latent.shape)
    dtype = str(latent.dtype).split(".")[-1]
    tensor_data = b64encode(_tobytes(latent, "inputs")).decode("utf-8")
    parameters = {"shape": shape, "dtype": dtype, "height": height, "width": width}
    data = {"inputs": tensor_data, "parameters": parameters}
    headers = {"Content-Type": "application/json", "Accept": "image/jpeg"}
    response = requests.post(ENDPOINT, json=data, headers=headers)
    if not response.ok:
        raise RuntimeError(response.json())
    image = Image.open(io.BytesIO(response.content))
    return image

image = remote_decode(torch.randn([1, 4096, 64]), height=1024, width=1024)

```

</p>
</details>

<figure class="image flex flex-col items-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/flux_random_latent.png"/>
</figure>

### Generation

But we want to use the VAE on an actual pipeline to get an actual image, not random noise. The example below shows how to do it with SD v1.5. 

<details><summary>Details</summary>
<p>

```python
from diffusers import StableDiffusionPipeline
import io
import requests
import torch
from base64 import b64encode
from PIL import Image
from safetensors.torch import _tobytes

ENDPOINT = "https://lqmfdhmzmy4dw51z.us-east-1.aws.endpoints.huggingface.cloud/"

def remote_decode(latent: torch.Tensor) -> Image.Image:
    shape = list(latent.shape)
    dtype = str(latent.dtype).split(".")[-1]
    tensor_data = b64encode(_tobytes(latent, "inputs")).decode("utf-8")
    parameters = {"shape": shape, "dtype": dtype}
    data = {"inputs": tensor_data, "parameters": parameters}
    headers = {"Content-Type": "application/json", "Accept": "image/jpeg"}
    response = requests.post(ENDPOINT, json=data, headers=headers)
    if not response.ok:
        raise RuntimeError(response.json())
    image = Image.open(io.BytesIO(response.content))
    return image

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    vae=None,
).to("cuda")

prompt = "Strawberry ice cream, in a stylish modern glass, coconut, splashing milk cream and honey, in a gradient purple background, fluid motion, dynamic movement, cinematic lighting, Mysterious"

latent = pipe(
    prompt=prompt,
    output_type="latent",
).images
image = remote_decode(latent)
image.save("test.jpg")

```

</p>
</details>

<figure class="image flex flex-col items-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/test.jpg"/>
</figure>

Here’s another example with Flux.

<details><summary>Code</summary>
<p>

```python
from diffusers import FluxPipeline
import io
import requests
import torch
from base64 import b64encode
from PIL import Image
from safetensors.torch import _tobytes

ENDPOINT = "https://zy1z7fzxpgtltg06.us-east-1.aws.endpoints.huggingface.cloud"

def remote_decode(latent: torch.Tensor, height: int, width: int) -> Image.Image:
    shape = list(latent.shape)
    dtype = str(latent.dtype).split(".")[-1]
    tensor_data = b64encode(_tobytes(latent, "inputs")).decode("utf-8")
    parameters = {"shape": shape, "dtype": dtype, "height": height, "width": width}
    data = {"inputs": tensor_data, "parameters": parameters}
    headers = {"Content-Type": "application/json", "Accept": "image/jpeg"}
    response = requests.post(ENDPOINT, json=data, headers=headers)
    if not response.ok:
        raise RuntimeError(response.json())
    image = Image.open(io.BytesIO(response.content))
    return image

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    vae=None,
).to("cuda")

prompt = "Strawberry ice cream, in a stylish modern glass, coconut, splashing milk cream and honey, in a gradient purple background, fluid motion, dynamic movement, cinematic lighting, Mysterious"

latent = pipe(
    prompt=prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    output_type="latent",
).images
image = remote_decode(latent, height=1024, width=1024)
image.save("test.jpg")
```

</p>
</details>

<figure class="image flex flex-col items-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/test_1.jpg"/>
</figure>

### Queueing

One of the great benefits of using a remote VAE is that we can queue multiple generation requests. While the current latent is being processed for decoding, we can already queue another one. This helps improve concurrency. 


<details><summary>Code</summary>
<p>

```python
import io
import queue
import requests
import threading
import torch
from base64 import b64encode
from IPython.display import display
from PIL import Image
from safetensors.torch import _tobytes
from diffusers import StableDiffusionPipeline

ENDPOINT = "https://lqmfdhmzmy4dw51z.us-east-1.aws.endpoints.huggingface.cloud"

def remote_decode(latent: torch.Tensor) -> Image.Image:
    shape = list(latent.shape)
    dtype = str(latent.dtype).split(".")[-1]
    tensor_data = b64encode(_tobytes(latent, "inputs")).decode("utf-8")
    parameters = {"shape": shape, "dtype": dtype}
    data = {"inputs": tensor_data, "parameters": parameters}
    headers = {"Content-Type": "application/json", "Accept": "image/jpeg"}
    response = requests.post(ENDPOINT, json=data, headers=headers)
    return Image.open(io.BytesIO(response.content))

def decode_worker(q: queue.Queue):
    while True:
        item = q.get()
        if item is None:
            break
        image = remote_decode(latent=item)
        display(image)
        q.task_done()

q = queue.Queue()
thread = threading.Thread(target=decode_worker, args=(q,), daemon=True)
thread.start()

def decode(latent: torch.Tensor):
    q.put(latent)

prompts = [
    "Blueberry ice cream, in a stylish modern glass , ice cubes, nuts, mint leaves, splashing milk cream, in a gradient purple background, fluid motion, dynamic movement, cinematic lighting, Mysterious",
    "Lemonade in a glass, mint leaves, in an aqua and white background, flowers, ice cubes, halo, fluid motion, dynamic movement, soft lighting, digital painting, rule of third's composition, Art by Greg rutkowski, Coby whitmore",
    "Comic book art, beautiful, vintage, pastel neon colors, extremely detailed pupils, delicate features, light on face, slight smile, Artgerm, Mary Blair, Edmund Dulac, long dark locks, bangs, glowing, fashionable style, fairytale ambience, hot pink.",
    "Masterpiece, vanilla cone ice cream garnished with chocolate syrup, crushed nuts, choco flakes, in a brown background, gold, cinematic lighting, Art by WLOP",
    "A bowl of milk, falling cornflakes, berries, blueberries, in a white background, soft lighting, intricate details, rule of third's, octane render, volumetric lighting",
    "Cold Coffee with cream, crushed almonds, in a glass, choco flakes, ice cubes, wet, in a wooden background, cinematic lighting, hyper realistic painting, art by Carne Griffiths, octane render, volumetric lighting, fluid motion, dynamic movement, muted colors,",
]

pipe = StableDiffusionPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    torch_dtype=torch.float16,
    vae=None,
).to("cuda")

pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

_ = pipe(
    prompt=prompts[0],
    output_type="latent",
)

for prompt in prompts:
    latent = pipe(
        prompt=prompt,
        output_type="latent",
    ).images
    decode(latent)

q.put(None)
thread.join()
```

</p>
</details>


<figure class="image flex flex-col items-center text-center m-0 w-full">
   <video
      alt="queue.mp4"
      autoplay loop autobuffer muted playsinline
    >
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/queue.mp4" type="video/mp4">
  </video>
</figure>


## Available VAEs

|   | **Endpoint** | **Model** |
|:-:|:-----------:|:--------:|
| **Stable Diffusion v1** | [https://lqmfdhmzmy4dw51z.us-east-1.aws.endpoints.huggingface.cloud](https://lqmfdhmzmy4dw51z.us-east-1.aws.endpoints.huggingface.cloud) | [`stabilityai/sd-vae-ft-mse`](https://hf.co/stabilityai/sd-vae-ft-mse) |
| **Stable Diffusion XL** | [https://m5fxqwyk0r3uu79o.us-east-1.aws.endpoints.huggingface.cloud](https://m5fxqwyk0r3uu79o.us-east-1.aws.endpoints.huggingface.cloud) | [`madebyollin/sdxl-vae-fp16-fix`](https://hf.co/madebyollin/sdxl-vae-fp16-fix) |
| **Flux** | [https://zy1z7fzxpgtltg06.us-east-1.aws.endpoints.huggingface.cloud](https://zy1z7fzxpgtltg06.us-east-1.aws.endpoints.huggingface.cloud) | [`black-forest-labs/FLUX.1-schnell`](https://hf.co/black-forest-labs/FLUX.1-schnell) |


## Advantages of using a remote VAE

These tables demonstrate the VRAM requirements with different GPUs. Memory usage % determines whether users of a certain GPU will need to offload. Offload times vary with CPU, RAM and HDD/NVMe. Tiled decoding increases inference time.

<details><summary>v1</summary>
<p>

| GPU | Resolution | Time (seconds) | Memory (%) | Tiled Time (secs) | Tiled Memory (%) |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 4090 | 512x512 | 0.031 | 5.60% | 0.031 (0%) | 5.60% |
| NVIDIA GeForce RTX 4090 | 1024x1024 | 0.148 | 20.00% | 0.301 (+103%) | 5.60% |
| NVIDIA GeForce RTX 4080 | 512x512 | 0.05 | 8.40% | 0.050 (0%) | 8.40% |
| NVIDIA GeForce RTX 4080 | 1024x1024 | 0.224 | 30.00% | 0.356 (+59%) | 8.40% |
| NVIDIA GeForce RTX 4070 Ti | 512x512 | 0.066 | 11.30% | 0.066 (0%) | 11.30% |
| NVIDIA GeForce RTX 4070 Ti | 1024x1024 | 0.284 | 40.50% | 0.454 (+60%) | 11.40% |
| NVIDIA GeForce RTX 3090 | 512x512 | 0.062 | 5.20% | 0.062 (0%) | 5.20% |
| NVIDIA GeForce RTX 3090 | 1024x1024 | 0.253 | 18.50% | 0.464 (+83%) | 5.20% |
| NVIDIA GeForce RTX 3080 | 512x512 | 0.07 | 12.80% | 0.070 (0%) | 12.80% |
| NVIDIA GeForce RTX 3080 | 1024x1024 | 0.286 | 45.30% | 0.466 (+63%) | 12.90% |
| NVIDIA GeForce RTX 3070 | 512x512 | 0.102 | 15.90% | 0.102 (0%) | 15.90% |
| NVIDIA GeForce RTX 3070 | 1024x1024 | 0.421 | 56.30% | 0.746 (+77%) | 16.00% |

</p>
</details>

<details><summary>Details</summary>
<p>

| GPU | Resolution | Time (seconds) | Memory Consumed (%) | Tiled Time (seconds) | Tiled Memory (%) |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 4090 | 512x512 | 0.057 | 10.00% | 0.057 (0%) | 10.00% |
| NVIDIA GeForce RTX 4090 | 1024x1024 | 0.256 | 35.50% | 0.257 (+0.4%) | 35.50% |
| NVIDIA GeForce RTX 4080 | 512x512 | 0.092 | 15.00% | 0.092 (0%) | 15.00% |
| NVIDIA GeForce RTX 4080 | 1024x1024 | 0.406 | 53.30% | 0.406 (0%) | 53.30% |
| NVIDIA GeForce RTX 4070 Ti | 512x512 | 0.121 | 20.20% | 0.120 (-0.8%) | 20.20% |
| NVIDIA GeForce RTX 4070 Ti | 1024x1024 | 0.519 | 72.00% | 0.519 (0%) | 72.00% |
| NVIDIA GeForce RTX 3090 | 512x512 | 0.107 | 10.50% | 0.107 (0%) | 10.50% |
| NVIDIA GeForce RTX 3090 | 1024x1024 | 0.459 | 38.00% | 0.460 (+0.2%) | 38.00% |
| NVIDIA GeForce RTX 3080 | 512x512 | 0.121 | 25.60% | 0.121 (0%) | 25.60% |
| NVIDIA GeForce RTX 3080 | 1024x1024 | 0.524 | 93.00% | 0.524 (0%) | 93.00% |
| NVIDIA GeForce RTX 3070 | 512x512 | 0.183 | 31.80% | 0.183 (0%) | 31.80% |
| NVIDIA GeForce RTX 3070 | 1024x1024 | 0.794 | 96.40% | 0.794 (0%) | 96.40% |

</p>
</details>

## Provide feedback

If you like the idea and feature, please help us with your feedback on how we can make this better and whether you’d be interested in having this kind of feature more natively integrated into the Hugging Face ecosystem. If this pilot goes well, we plan on creating optimized VAE endpoints for more models, including the ones that can generate high-resolution videos!

### Steps:

1. Open an issue on Diffusers through [this link](https://github.com/huggingface/diffusers/issues/new?template=remote-vae-pilot-feedback.yml). 
2. Answer the questions and provide any extra info you want. 
3. Hit submit!
