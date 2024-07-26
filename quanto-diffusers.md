---
title: "Memory-efficient Diffusion Transformers with Quanto and Diffusers"
thumbnail: /blog/assets/quanto-diffusers/thumbnail.png
authors:
- user: sayakpaul
- user: dacorvo
---

# Memory-efficient Diffusion Transformers with Quanto and Diffusers

Over the past few months, we have seen an emergence in the use of Transformer-based diffusion backbones for high-resolution text-to-image (T2I) generation. Thanks to the nature of Transformers, these backbones show good scalability. For example, they can vary from 0.6B parameters all the way up to 8B parameters. 

With bigger scales arises the problem of memory management. The problem intensifies because a diffusion pipeline consists of several components: - a text encoder, a diffusion backbone, and a decoder (for latent-space diffusion). Modern diffusion pipelines use multiple text encoders. Stable Diffusion 3 uses three as an example. It takes 18.765 GB of GPU memory to run inference in the FP16 precision. 

These high memory requirements can make the field exclusive for those with only consumer GPUs. In this post, we discuss improving the memory efficiency of Transformer-based diffusion pipelines in Diffusers with Quanto. We will do this with the help of Quanto's quantization utilities. 

### Table of contents

- [Preliminaries](#preliminaries)
- [Quantizing a `DiffusionPipeline` with Quanto](#quantizing-a-diffusionpipeline-with-quanto)
- [Generality of the observations](#generality-of-the-observations)
- [Misc findings](#misc-findings)
  - [`bfloat16` as the main compute data-type](#bfloat16-is-usually-better-on-h100)
  - [The promise of `qint8`](#the-promise-of-qint8)
  - [INT4](#how-about-int4)
- [Bonus - saving and loading Diffusers models in Quanto](#bonus---saving-and-loading-diffusers-models-in-quanto)
- [Tips](#tips)
- [Conclusion](#conclusion)

## Preliminaries

For a detailed introduction to Quanto, please refer to [this post](https://huggingface.co/blog/quanto-introduction). In short, it is a quantization toolkit in PyTorch. 

For the benchmarking purposes, we use an H100 GPU with the following environment: 

- CUDA 12.2
- PyTorch 2.4.1
- Diffusers (installed from [this commit](https://github.com/huggingface/diffusers/commit/bce9105ac79636f68dcfdcfc9481b89533db65e5))
- Quanto (installed from [this commit](https://github.com/huggingface/optimum-quanto/commit/285862b4377aa757342ed810cd60949596b4872b))

Unless otherwise specified, we default to perform computations in FP16. Because the VAEs usually lead to numerical instability problems, we decided not to quantize the VAE at all. Our benchmarking code can be found [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/blob/main/quanto-exps-2/benchmark.py). 

At the time of this writing, we have the following Transformer-based diffusion pipelines for text-to-image generation in Diffusers:

- [PixArt-Alpha](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart) and [PixArt-Sigma](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart_sigma)
- [Stable Diffusion 3](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3)
- [Hunyuan DiT](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuandit)
- [Lumina](https://huggingface.co/docs/diffusers/main/en/api/pipelines/lumina)
- [Aura Flow](https://huggingface.co/docs/diffusers/main/en/api/pipelines/aura_flow)

We also have [Latte](https://huggingface.co/docs/diffusers/main/en/api/pipelines/latte), which is a Transformer-based text-to-video generation pipeline. For brevity, for this post, we keep our study limited to the following three: PixArt-Sigma, Stable Diffusion 3, and Aura Flow. The table below shows the parameter counts of their diffusion backbones:

| **Model** | **Checkpoint** | **# Params (Billion)** |
| --- | --- | --- |
| PixArt | https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS | 0.611 |
| Stable Diffusion 3 | https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers | 2.028 |
| Aura Flow | https://huggingface.co/fal/AuraFlow/ | 6.843 |

It’s worth keeping in mind that this post primarily focuses on memory efficiency at a slight or negligible cost of inference latency. 

## Quantizing a `DiffusionPipeline` with Quanto

Quantizing a model with Quanto is straightforward. 

```python
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import PixArtSigmaPipeline
import torch

pipeline = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
).to("cuda")

quantize(pipeline.transformer, weights=qfloat8)
freeze(pipeline.transformer)
```

We call `quantize()` on the model to be quantized, specifying what we want to quantize. In the above case, we are just quantizing the parameters, leaving the activations as is. We’re quantizing to the FP8 data-type. We finally call `freeze()` to replace the original parameters with the quantized parameters. 

We can then call this `pipeline` normally:

```python
image = pipeline("ghibli style, a fantasy landscape with castles").images[0]
```

<table>
    <tr>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptpixart-bs1-dtypefp16-qtypenone-qte0.png" alt="Image 1">
            <caption>FP16</caption>
        </td>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptpixart-bs1-dtypefp16-qtypefp8-qte0.png" alt="Image 2">
            <caption>Diffusion Transformer in FP8</caption>
        </td>
    </tr>
</table>

We notice almost no quality degradation in the FP8 output but it leads to memory savings: 

| **Batch Size** | **Quantization** | **Memory (GB)** | **Latency (Seconds)** |
| --- | --- | --- | --- |
| 1 | None | 12.086 | 1.2 |
| 1 | FP8 | **11.547** | 1.540 |
| 4 | None | 12.087 | 4.482 |
| 4 | FP8 | **11.548** | 5.109 |

We can quantize the text encoder in the same way:

```python
quantize(pipeline.text_encoder, weights=qfloat8)
freeze(pipeline.text_encoder)
```

Quantizing the text encoder additionally leads to further memory improvements: 

| **Batch Size** | **Quantization** | **Quantize TE** | **Memory (GB)** | **Latency (Seconds)** |
| --- | --- | --- | --- | --- |
| 1 | FP8 | False | 11.547 | 1.540 |
| 1 | FP8 | True | **5.363** | 1.601 |
| 4 | FP8 | False | 11.548 | 5.109 |
| 4 | FP8 | True | **5.364** | 5.141 |

Text encoder quantization doesn't lead to quality degradation in this case:

![ckpt@pixart-bs@1-dtype@fp16-qtype@fp8-qte@1.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptpixart-bs1-dtypefp16-qtypefp8-qte1.png)

## Generality of the observations

These observations generally hold for the models we considered except for Stable Diffusion 3. For it to work with quantization, we can

- Only quantize the first text encoder ([`CLIPTextModelWithProjection`](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTextModelWithProjection)) or
- Only quantize the third text encoder ([`T5EncoderModel`](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel)) or
- Quantize the first and third text encoders

The table below gives an idea about the expected memory savings for various text encoder quantization combinations: 

| **Batch Size** | **Quantization** | **Quantize \nTE 1** | **Quantize \nTE 2** | **Quantize \nTE 3** | **Memory (GB)** | **Latency (Seconds)** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | FP8 | 1 | 1 | 1 | 8.200 | 2.858 |
| 1 ✅ | FP8 | 0 | 0 | 1 | 8.294 | 2.781 |
| 1 | FP8 | 1 | 1 | 0 | 14.384 | 2.833 |
| 1 | FP8 | 0 | 1 | 0 | 14.475 | 2.818 |
| 1 ✅ | FP8 | 1 | 0 | 0 | 14.384 | 2.730 |
| 1  | FP8 | 0 | 1 | 1 | 8.325 | 2.875 |
| 1 ✅ | FP8 | 1 | 0 | 1 | 8.204 | 2.789 |
| 1 | None | - | - | - | 16.403 | 2.118 |

<table>
    <tr>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptsd3-bs1-dtypefp16-qtypefp8-qte1-first1.png" alt="Image 1">
            <caption>Quantized Text Encoder: 1</caption>
        </td>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptsd3-bs1-dtypefp16-qtypefp8-qte1-third1.png" alt="Image 2">
            <caption>Quantized Text Encoder: 3</caption>
        </td>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptsd3-bs1-dtypefp16-qtypefp8-qte1-first1-third1%201.png" alt="Image 3">
            <caption>Quantized Text Encoders: 1 and 3</caption>
        </td>
    </tr>
</table>


## Misc findings

### `bfloat16` is usually better on H100

`bfloat16` as the computation data-type can be slightly better both in terms of memory and inference latency for supported GPU architectures. The table below presents some numbers for PixArt: 

| **Batch Size** | **Precision** | **Quantization** | **Memory (GB)** | **Latency \n(Seconds)** | **Quantize \nTE** |
| --- | --- | --- | --- | --- | --- |
| 1 | FP16 | INT8 | 5.363 | 1.538 | True |
| 1 | BF16 | INT8 | 5.364 | **1.454** | True |
| 1 | FP16 | FP8 | 5.363 | 1.601 | True |
| 1 | BF16 | FP8 | 5.363 | **1.495** | True |

### The promise of `qint8`

We found quantizing with `qint8`  is generally better in terms of inference latency. This effect gets more pronounced when we horizontally fuse the attention QKV projections (calling `fuse_qkv_projections()` in Diffusers), thereby thickening the dimensions of the int8 kernels to speed up computation. We present some evidence below for PixArt: 

| **Batch Size** | **Quantization** | **Memory \n(GB)** | **Latency \n(Seconds)** | **Quantize \nTE** | **QKV \nProjection** |
| --- | --- | --- | --- | --- | --- |
| 1 | INT8 | 5.363 | 1.538 | True | False |
| 1 | INT8 | 5.536 | **1.504** | True | True |
| 4 | INT8 | 5.365 | 5.129 | True | False |
| 4 | INT8 | 5.538 | **4.989** | True | True |

### How about INT4?

We additionally experimented with `qint4` when using `bfloat16`. This is only applicable to `bfloat16` on H100 because other configurations are not supported yet. With `qint4`, we can expect to see more improvements in memory consumption at the cost of increased inference latency (PixArt-Sigma numbers below):

| **Batch Size** | **Quantize TE** | **Memory (GB)** | **Latency (Seconds)** |
| --- | --- | --- | --- |
| 1 | No | 9.380 | 7.431 |
| 1 | Yes | **3.058**	 | 7.604 |

Note, however, that due to the aggressive discretization of INT4, the end results can take a hit. This is why, for Transformer-based models in general, we usually leave the final projection layer out of quantization. In Quanto, we do this by:

```python
quantize(pipeline.transformer, weights=qint4, exclude="proj_out")
freeze(pipeline.transformer)
```

`"proj_out"` corresponds to the final layer in `pipeline.transformer`. The table below presents results for various settings:

<table>
    <tr>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckpt%40pixart-bs%401-dtype%40bf16-qtype%40int4-qte%400-fuse%400.png" alt="Image 1">
            <caption>Quantize TE: No, Layer exclusion: None</caption>
        </td>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckpt%40pixart-bs%401-dtype%40bf16-qtype%40int4-qte%400-fuse%400-exclude%40proj_out.png" alt="Image 2">
            <caption>Quantize TE: No, Layer exclusion: "proj_out"</caption>
        </td>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckpt%40pixart-bs%401-dtype%40bf16-qtype%40int4-qte%401-fuse%400.png" alt="Image 3">
            <caption>Quantize TE: Yes, Layer exclusion: None</caption>
        </td>
        <td>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckpt%40pixart-bs%401-dtype%40bf16-qtype%40int4-qte%401-fuse%400-exclude%40proj_out.png" alt="Image 4">
            <caption>Quantize TE: Yes, Layer exclusion: "proj_out"</caption>
        </td>
    </tr>
</table>

To recover the lost image quality, a common practice is to perform quantization-aware training, which is supported in Quanto.

All the results of our experiments for this post can be found [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/tree/main/quanto-exps-2). 

## Bonus - saving and loading Diffusers models in Quanto

We added the support to save and load quantized models from Diffusers in Quanto. This means we can save a model like `PixArtTransformer2DModel` using a quantization precision of our choice:

```python
from diffusers import PixArtTransformer2DModel
from optimum.quanto import QuantizedPixArtTransformer2DModel, qfloat8

model = PixArtTransformer2DModel.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="transformer")
qmodel = QuantizedPixArtTransformer2DModel.quantize(model, weights=qfloat8)
qmodel.save_pretrained("pixart-sigma-fp8")
```

The resulting checkpoint is ***587MB*** in size as opposed to the original 2.44GB. We can then load it:

```python
from diffusers import PixArtTransformer2DModel
import torch

transformer = QuantizedPixArtTransformer2DModel.from_pretrained("pixart-sigma-fp8") 
transformer.to(device="cuda", dtype=torch.float16)
```

And use it in a `DiffusionPipeline`:

```python
from diffusers import DiffusionPipeline

pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
    transformer=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.transformer = transformer

prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(prompt).images[0]
```

In the future, we can expect to pass the `transformer` directly when initializing the pipeline so that this works:

```diff
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
-    transformer=None,
+    transformer=transformer,
    torch_dtype=torch.float16,
).to("cuda")
```

If you want more models from Diffusers supported in Quanto for saving and loading, please open an issue [here](https://github.com/huggingface/optimum-quanto/issues/new) and mention `@sayakpaul`.  

## Tips

- Based on the requirements, you may want to apply different kinds of quantization to the different models involved in a `DiffusionPipeline`. For example, FP8 for the text encoder and INT8 for the diffusion transformer. Thanks to the flexibility of Diffusers and Quanto, this can be done seamlessly.
- To optimize for your use cases, you can even combine quantization with other [memory optimization techniques](https://huggingface.co/docs/diffusers/main/en/optimization/memory) in Diffusers, such as `enable_model_cpu_offload()`.

## Conclusion

In this post, we showed how to quantize Transformer models from Diffusers and optimize their memory consumption. The effects of quantization become more visible when we additionally quantize the text encoders involved in the mix. We hope you will apply some of the workflows to your projects and benefit from them 🤗