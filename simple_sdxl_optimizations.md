---
title: "Exploring simple optimizations for SDXL"
thumbnail: /blog/assets/simple_sdxl_optimizations/thumbnail.png
authors:
- user: sayakpaul
- user: stevhliu
---

# Exploring simple optimizations for SDXL

<a target="_blank" href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/exploring_simple%20optimizations_for_sdxl.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[Stable Diffusion XL (SDXL)](https://huggingface.co/papers/2307.01952) is the latest latent diffusion model by Stability AI for generating high-quality super realistic images. It overcomes challenges of previous Stable Diffusion models like getting hands and text right as well as spatially correct compositions. In addition, SDXL is also more context aware and requires fewer words in its prompt to generate better looking images. 

However, all of these improvements come at the expense of a significantly larger model. How much larger? The base SDXL model has 3.5B parameters (the UNet, in particular), which is approximately 3x larger than the previous Stable Diffusion model.

To explore how we can optimize SDXL for inference speed and memory use, we ran some tests on an A100 GPU (40 GB). For each inference run, we generate 4 images and repeat it 3 times. While computing the inference latency, we only consider the final iteration out of the 3 iterations. 

So if you run SDXL out-of-the-box as is with full precision and use the default attention mechanism, it’ll consume 28GB of memory and take 72.2 seconds!

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to("cuda")
pipe.unet.set_default_attn_processor()
```

This isn’t very practical and can slow you down because you’re often generating more than 4 images. And if you don’t have a more powerful GPU, you’ll run into that frustrating out-of-memory error message. So how can we optimize SDXL to increase inference speed and reduce its memory-usage? 

In 🤗 Diffusers, we have a bunch of optimization tricks and techniques to help you run memory-intensive models like SDXL and we'll show you how! The two things we’ll focus on are *inference speed* and *memory*.

<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 5px;">
🧠 The techniques discussed in this post are applicable to all the <a href=https://huggingface.co/docs/diffusers/main/en/using-diffusers/pipeline_overview>pipelines</a>.
</div>

## Inference speed

Diffusion is a random process, so there's no guarantee you'll get an image you’ll like. Often times, you’ll need to run inference multiple times and iterate, and that’s why optimizing for speed is crucial. This section focuses on using lower precision weights and incorporating memory-efficient attention and `torch.compile` from PyTorch 2.0 to boost speed and reduce inference time.

### Lower precision

Model weights are stored at a certain *precision* which is expressed as a floating point data type. The standard floating point data type is float32 (fp32), which can accurately represent a wide range of floating numbers. For inference, you often don’t need to be as precise so you should use float16 (fp16) which captures a narrower range of floating numbers. This means fp16 only takes half the amount of memory to store compared to fp32, and is twice as fast because it is easier to calculate. In addition, modern GPU cards have optimized hardware to run fp16 calculations, making it even faster.

With 🤗 Diffusers, you can use fp16 for inference by specifying the `torch.dtype` parameter to convert the weights when the model is loaded:

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipe.unet.set_default_attn_processor()
```

Compared to a completely unoptimized SDXL pipeline, using fp16 takes 21.7GB of memory and only 14.8 seconds. You’re almost speeding up inference by a full minute!

### Memory-efficient attention

The attention blocks used in transformers modules can be a huge bottleneck, because memory increases _quadratically_ as input sequences get longer. This can quickly take up a ton of memory and leave you with an out-of-memory error message. 😬

Memory-efficient attention algorithms seek to reduce the memory burden of calculating attention, whether it is by exploiting sparsity or tiling. These optimized algorithms used to be mostly available as third-party libraries that needed to be installed separately. But starting with PyTorch 2.0, this is no longer the case. PyTorch 2 introduced [scaled dot product attention (SDPA)](https://pytorch.org/blog/accelerated-diffusers-pt-20/), which offers fused implementations of [Flash Attention](https://huggingface.co/papers/2205.14135), [memory-efficient attention](https://huggingface.co/papers/2112.05682) (xFormers), and a PyTorch implementation in C++. SDPA is probably the easiest way to speed up inference: if you’re using PyTorch ≥ 2.0 with 🤗 Diffusers, it is automatically enabled by default!

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
```

Compared to a completely unoptimized SDXL pipeline, using fp16 and SDPA takes the same amount of memory and the inference time improves to 11.4 seconds. Let’s use this as the new baseline we’ll compare the other optimizations to.

### torch.compile

PyTorch 2.0 also introduced the `torch.compile` API for just-in-time (JIT) compilation of your PyTorch code into more optimized kernels for inference. Unlike other compiler solutions, `torch.compile` requires minimal changes to your existing code and it is as easy as wrapping your model with the function.

With the `mode` parameter, you can optimize for memory overhead or inference speed during compilation, which gives you way more flexibility.

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

Compared to the previous baseline (fp16 + SDPA), wrapping the UNet with `torch.compile` improves inference time to 10.2 seconds. 

<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 5px;">
⚠️ The first time you compile a model is slower, but once the model is compiled, all subsequent calls to it are much faster!
</div>

## Model memory footprint

Models today are growing larger and larger, making it a challenge to fit them into memory. This section focuses on how you can reduce the memory footprint of these enormous models so you can run them on consumer GPUs. These techniques include CPU offloading, decoding latents into images over several steps rather than all at once, and using a distilled version of the autoencoder.

### Model CPU offloading

Model offloading saves memory by loading the UNet into the GPU memory while the other components of the diffusion model (text encoders, VAE) are loaded onto the CPU. This way, the UNet can run for multiple iterations on the GPU until it is no longer needed.

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_model_cpu_offload()
```

Compared to the baseline, it now takes 20.2GB of memory which saves you 1.5GB of memory.

### Sequential CPU offloading

Another type of offloading which can save you more memory at the expense of slower inference is sequential CPU offloading. Rather than offloading an entire model - like the UNet - model weights stored in different UNet submodules are offloaded to the CPU and only loaded onto the GPU right before the forward pass. Essentially, you’re only loading parts of the model each time which allows you to save even more memory. The only downside is that it is significantly slower because you’re loading and offloading submodules many times.

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_sequential_cpu_offload()
```

Compared to the baseline, this takes 19.9GB of memory but the inference time increases to 67 seconds.

## Slicing

In SDXL, a variational encoder (VAE) decodes the refined latents (predicted by the UNet) into realistic images. The memory requirement of this step scales with the number of images being predicted (the batch size). Depending on the image resolution and the available GPU VRAM, it can be quite memory-intensive. 

This is where “slicing” is useful. The input tensor to be decoded is split into slices and the computation to decode it is completed over several steps. This saves memory and allows larger batch sizes.

```python
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_vae_slicing()
```

With sliced computations, we reduce the memory to 15.4GB. If we add sequential CPU offloading, it is further reduced to 11.45GB which lets you generate 4 images (1024x1024) per prompt. However, with sequential offloading, the inference latency also increases. 

## Caching computations

Any text-conditioned image generation model typically uses a text encoder to compute embeddings from the input prompt. SDXL uses *two* text encoders! This contributes quite a bit to the inference latency. However, since these embeddings remain unchanged throughout the reverse diffusion process, we can precompute them and reuse them as we go. This way, after computing the text embeddings, we can remove the text encoders from memory. 

First, load the text encoders and their corresponding tokenizers and compute the embeddings from the input prompt:

```python
tokenizers = [tokenizer, tokenizer_2]
text_encoders = [text_encoder, text_encoder_2]

(
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds
) = encode_prompt(tokenizers, text_encoders, prompt)
```

Next, flush the GPU memory to remove the text encoders:

```jsx
del text_encoder, text_encoder_2, tokenizer, tokenizer_2
flush()
```

Now the embeddings are good to go straight to the SDXL pipeline:

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    torch_dtype=torch.float16,
).to("cuda")

call_args = dict(
		prompt_embeds=prompt_embeds,
		negative_prompt_embeds=negative_prompt_embeds,
		pooled_prompt_embeds=pooled_prompt_embeds,
		negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
		num_images_per_prompt=num_images_per_prompt,
		num_inference_steps=num_inference_steps,
)
image = pipe(**call_args).images[0]
```

Combined with SDPA and fp16, we can reduce the memory to 21.9GB. Other techniques discussed above for optimizing memory can also be used with cached computations. 

## Tiny Autoencoder

As previously mentioned, a VAE decodes latents into images. Naturally, this step is directly bottlenecked by the size of the VAE. So, let’s just use a smaller autoencoder! The [Tiny Autoencoder by `madebyollin`](https://github.com/madebyollin/taesd), available [the Hub](https://huggingface.co/madebyollin/taesdxl) is just 10MB and it is distilled from the original VAE used by SDXL. 

```python
from diffusers import AutoencoderTiny

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
pipe.to("cuda")
```

With this setup, we reduce the memory requirement to 15.6GB while reducing the inference latency at the same time. 

<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 5px;">
⚠️ The Tiny Autoencoder can omit some of the more fine-grained details from images, which is why the Tiny AutoEncoder is more appropriate for image previews.
</div>

## Conclusion

To conclude and summarize the savings from our optimizations:

<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 5px;">
⚠️ While profiling GPUs to measure the trade-off between inference latency and memory requirements, it is important to be aware of the hardware being used. The above findings may not translate equally from hardware to hardware. For example, `torch.compile` only seems to benefit modern GPUs, at least for SDXL.
</div>

| **Technique** | **Memory (GB)** | **Inference latency (ms)** |
| --- | --- | --- |
| unoptimized pipeline | 28.09 | 72200.5 |
| fp16 | 21.72 | 14800.9 |
| **fp16 + SDPA (default)** | **21.72** | **11413.0** |
| default + `torch.compile` | 21.73 | 10296.7 |
| default + model CPU offload | 20.21 | 16082.2 |
| default + sequential CPU offload | 19.91 | 67034.0 |
| **default + VAE slicing** | **15.40** | **11232.2** |
| default + VAE slicing + sequential CPU offload | 11.47 | 66869.2 |
| default + precomputed text embeddings | 21.85 | 11909.0 |
| default + Tiny Autoencoder | 15.48 | 10449.7 |

We hope these optimizations make it a breeze to run your favorite pipelines. Try these techniques out and share your images with us! 🤗

---

**Acknowledgements**: Thank you to [Pedro Cuenca](https://twitter.com/pcuenq?lang=en) for his helpful reviews on the draft.