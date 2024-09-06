---
title: "基于 Quanto 和 Diffusers 的内存高效 transformer 扩散模型"
thumbnail: /blog/assets/quanto-diffusers/thumbnail.png
authors:
- user: sayakpaul
- user: dacorvo
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 基于 Quanto 和 Diffusers 的内存高效 transformer 扩散模型

过去的几个月，我们目睹了使用基于 transformer 模型作为扩散模型的主干网络来进行高分辨率文生图 (text-to-image，T2I) 的趋势。和一开始的许多扩散模型普遍使用 UNet 架构不同，这些模型使用 transformer 架构作为扩散过程的主模型。由于 transformer 的性质，这些主干网络表现出了良好的可扩展性，模型参数量可从 0.6B 扩展至 8B。

随着模型越变越大，内存需求也随之增加。对扩散模型而言，这个问题愈加严重，因为扩散流水线通常由多个模型串成: 文本编码器、扩散主干模型和图像解码器。此外，最新的扩散流水线通常使用多个文本编码器 - 如: Stable Diffusion 3 有 3 个文本编码器。使用 FP16 精度对 SD3 进行推理需要 18.765GB 的 GPU 显存。

这么高的内存要求使得很难将这些模型运行在消费级 GPU 上，因而减缓了技术采纳速度并使针对这些模型的实验变得更加困难。本文，我们展示了如何使用 Diffusers 库中的 Quanto 量化工具脚本来提高基于 transformer 的扩散流水线的内存效率。

### 目录

- [基于 Quanto 和 Diffusers 的内存高效 transformer 扩散模型](#基于-quanto-和-diffusers-的内存高效-transformer-扩散模型)
  - [目录](#目录)
  - [基础知识](#基础知识)
  - [用 Quanto 量化 `DiffusionPipeline` ](#用-quanto-量化-diffusionpipeline)
  - [上述攻略通用吗？](#上述攻略通用吗)
  - [其他发现](#其他发现)
    - [在 H100 上 `bfloat16` 通常表现更好](#在-h100-上-bfloat16-常表现更好)
    - [`qint8` 的前途](#qint8-的前途)
    - [INT4 咋样？](#int4-咋样)


  - [加个鸡腿 - 在 Quanto 中保存和加载 Diffusers 模型](#加个鸡腿—在-quanto-中保存和加载-diffusers-模型)
  - [小诀窍](#小诀窍)
  - [总结](#总结)

## 基础知识

你可参考 [这篇文章](https://huggingface.co/blog/zh/quanto-introduction) 以获取 Quanto 的详细介绍。简单来说，Quanto 是一个基于 PyTorch 的量化工具包。它是 [Hugging Face Optimum](https://github.com/huggingface/optimum) 的一部分，Optimum 提供了一套硬件感知的优化工具。

模型量化是 LLM 从业者必备的工具，但在扩散模型中并不算常用。Quanto 可以帮助弥补这一差距，其可以在几乎不伤害生成质量的情况下节省内存。

我们基于 H100 GPU 配置进行基准测试，软件环境如下:

- CUDA 12.2
- PyTorch 2.4.0
- Diffusers (从源代码安装，参考 [此提交](https://github.com/huggingface/diffusers/commit/bce9105ac79636f68dcfdcfc9481b89533db65e5))
- Quanto (从源代码安装，参考 [此提交](https://github.com/huggingface/optimum-quanto/commit/285862b4377aa757342ed810cd60949596b4872b))

除非另有说明，我们默认使用 FP16 进行计算。我们不对 VAE 进行量化以防止数值不稳定问题。你可于 [此处](https://huggingface.co/datasets/sayakpaul/sample-datasets/blob/main/quanto-exps-2/benchmark.py) 找到我们的基准测试代码。

截至本文撰写时，以下基于 transformer 的扩散模型流水线可用于 Diffusers 中的文生图任务:

- [PixArt-Alpha](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart) 及 [PixArt-Sigma](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart_sigma)
- [Stable Diffusion 3](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3)
- [Hunyuan DiT](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuandit)
- [Lumina](https://huggingface.co/docs/diffusers/main/en/api/pipelines/lumina)
- [Aura Flow](https://huggingface.co/docs/diffusers/main/en/api/pipelines/aura_flow)

另外还有一个基于 transformer 的文生视频流水线: [Latte](https://huggingface.co/docs/diffusers/main/en/api/pipelines/latte)。

为简化起见，我们的研究仅限于以下三个流水线: PixArt-Sigma、Stable Diffusion 3 以及 Aura Flow。下表显示了它们各自的扩散主干网络的参数量:

|     **模型**     |   **Checkpoint** | **参数量（Billion）** |
|:-----------------:|:--------------------------------------------------------:|:----------------------:|
|      PixArt       | https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS |         0.611         |
| Stable Diffusion 3| https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers |         2.028         |
|     Aura Flow     |        https://huggingface.co/fal/AuraFlow/              |         6.843         |

<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 5px;">
请记住，本文主要关注内存效率，因为量化对推理延迟的影响很小或几乎可以忽略不计。
</div>

## 用 Quanto 量化 `DiffusionPipeline`

使用 Quanto 量化模型非常简单。

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

我们对需量化的模块调用 `quantize()` ，以指定我们要量化的部分。上例中，我们仅量化参数，保持激活不变，量化数据类型为 FP8。最后，调用 `freeze()` 以用量化参数替换原始参数。

然后，我们就可以如常调用这个 `pipeline` 了:

```python
image = pipeline("ghibli style, a fantasy landscape with castles").images[0]
```

<table>
<tr style="text-align: center;">
    <th>FP16</th>
    <th> 将 transformer 扩散主干网络量化为 FP8</th>
</tr>
<tr>
    <td><img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptpixart-bs1-dtypefp16-qtypenone-qte0.png" width=512 alt="FP16 image."/></td>
    <td><img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptpixart-bs1-dtypefp16-qtypefp8-qte0.png" width=512 alt="FP8 quantized image."/></td>
</tr>
</table>

我们注意到使用 FP8 可以节省显存，且几乎不影响生成质量; 我们也看到量化模型的延迟稍有变长:

| **Batch Size** | **量化** | **内存 (GB)** | **延迟 (秒)** |
|:--------------:|:----------------:|:---------------:|:--------------------:|
|       1        |       无       |      12.086     |         1.200        |
|       1        |       FP8        |     **11.547**  |         1.540        |
|       4        |       无       |      12.087     |         4.482        |
|       4        |       FP8        |     **11.548**  |         5.109        |

我们可以用相同的方式量化文本编码器:

```python
quantize(pipeline.text_encoder, weights=qfloat8)
freeze(pipeline.text_encoder)
```

文本编码器也是一个 transformer 模型，我们也可以对其进行量化。同时量化文本编码器和扩散主干网络可以带来更大的显存节省:

| **Batch Size** | **量化** | **是否量化文本编码器** | **显存 (GB)** | **延迟 (秒)** |
|:--------------:|:----------------:|:---------------:|:---------------:|:--------------------:|
|       1        |       FP8        |      否      |      11.547     |         1.540        |
|       1        |       FP8        |       是      |     **5.363**   |         1.601        |
|       4        |       FP8        |      否      |      11.548     |         5.109        |
|       4        |       FP8        |       是      |     **5.364**   |         5.141        |

量化文本编码器后生成质量与之前的情况非常相似:

![ckpt@pixart-bs@1-dtype@fp16-qtype@fp8-qte@1.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptpixart-bs1-dtypefp16-qtypefp8-qte1.png)

## 上述攻略通用吗？

将文本编码器与扩散主干网络一起量化普遍适用于我们尝试的很多模型。但 Stable Diffusion 3 是个特例，因为它使用了三个不同的文本编码器。我们发现 _ 第二个 _ 文本编码器量化效果不佳，因此我们推荐以下替代方案:

- 仅量化第一个文本编码器 ([`CLIPTextModelWithProjection`](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTextModelWithProjection)) 或
- 仅量化第三个文本编码器 ([`T5EncoderModel`](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel)) 或
- 同时量化第一个和第三个文本编码器

下表给出了各文本编码器量化方案的预期内存节省情况 (扩散 transformer 在所有情况下均被量化):

| **Batch Size** | **量化** | **量化文本编码器 1** | **量化文本编码器 2** | **量化文本编码器 3** | **显存 (GB)** | **延迟 (秒)** |
|:--------------:|:----------------:|:-----------------:|:-----------------:|:-----------------:|:---------------:|:--------------------:|
|       1        |       FP8        |         1         |         1         |         1         |      8.200      |         2.858        |
|      1 ✅       |       FP8        |         0         |         0         |         1         |      8.294      |         2.781        |
|       1        |       FP8        |         1         |         1         |         0         |     14.384      |         2.833        |
|       1        |       FP8        |         0         |         1         |         0         |     14.475      |         2.818        |
|      1 ✅       |       FP8        |         1         |         0         |         0         |     14.384      |         2.730        |
|       1        |       FP8        |         0         |         1         |         1         |      8.325      |         2.875        |
|      1 ✅       |       FP8        |         1         |         0         |         1         |      8.204      |         2.789        |
|       1        |       无       |         -         |         -         |         -         |     16.403      |         2.118        |

<table>
<tr style="text-align: center;">
    <th> 量化文本编码器: 1</th>
    <th> 量化文本编码器: 3</th>
    <th> 量化文本编码器: 1 和 3</th>
</tr>
<tr>
    <td><img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptsd3-bs1-dtypefp16-qtypefp8-qte1-first1.png" width=300 alt="Image with quantized text encoder 1."/></td>
    <td><img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptsd3-bs1-dtypefp16-qtypefp8-qte1-third1.png" width=300 alt="Image with quantized text encoder 3."/></td>
    <td><img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckptsd3-bs1-dtypefp16-qtypefp8-qte1-first1-third1%201.png" width=300 alt="Image with quantized text encoders 1 and 3."/></td>
</tr>
</table>

## 其他发现

### 在 H100 上 `bfloat16` 通常表现更好

对于支持 `bfloat16` 的 GPU 架构 (如 H100 或 4090)，使用 `bfloat16` 速度更快。下表列出了在我们的 H100 参考硬件上测得的 PixArt 的一些数字: **Batch Size** **精度** **量化** **显存 (GB)** **延迟 (秒)** **是否量化文本编码器**

| **Batch Size** | **精度** | **量化** | **显存（GB）**  | **延迟（秒）** | **是否量化文本编码器** |
|:--------------:|:-------------:|:----------------:|:---------------:|:--------------------:|:---------------:|
|       1        |      FP16     |       INT8       |      5.363      |         1.538        |       是      |
|       1        |      BF16     |       INT8       |      5.364      |        **1.454**     |       是      |
|       1        |      FP16     |       FP8        |      5.363      |         1.601        |       是      |
|       1        |      BF16     |       FP8        |      5.363      |        **1.495**     |       是      |

### `qint8` 的前途

我们发现使用 `qint8` (而非 `qfloat8` ) 进行量化，推理延迟通常更好。当我们对注意力 QKV 投影进行水平融合 (在 Diffusers 中调用 `fuse_qkv_projections()` ) 时，效果会更加明显，因为水平融合会增大 int8 算子的计算维度从而实现更大的加速。我们基于 PixArt 测得了以下数据以证明我们的发现:

| **Batch Size** | **量化** | **显存 (GB)** | **延迟 (秒)** | **是否量化文本编码器** | **QKV 融合** |
|:--------------:|:----------------:|:---------------:|:--------------------:|:---------------:|:------------------:|
|       1        |       INT8       |      5.363      |         1.538        |       是      |       否        |
|       1        |       INT8       |      5.536      |        **1.504**     |       是      |       是         |
|       4        |       INT8       |      5.365      |         5.129        |       是      |       否        |
|       4        |       INT8       |      5.538      |        **4.989**     |       是      |       是         |

### INT4 咋样？

在使用 `bfloat16` 时，我们还尝试了 `qint4` 。目前我们仅支持 H100 上的 `bfloat16` 的 `qint4` 量化，其他情况尚未支持。通过 `qint4` ，我们期望看到内存消耗进一步降低，但代价是推理延迟变长。延迟增加的原因是硬件尚不支持 int4 计算 - 因此权重使用 4 位，但计算仍然以 `bfloat16` 完成。下表展示了 PixArt-Sigma 的结果:

| **Batch Size** | **是否量化文本编码器** | **显存 (GB)** | **延迟 (秒)** |
|:--------------:|:---------------:|:---------------:|:--------------------:|
|       1        |       否        |      9.380      |         7.431        |
|       1        |       是       |     **3.058**   |         7.604        |

但请注意，由于 INT4 量化比较激进，最终结果可能会受到影响。所以，一般对于基于 transformer 的模型，我们通常不量化最后一个投影层。在 Quanto 中，我们做法如下:

```python
quantize(pipeline.transformer, weights=qint4, exclude="proj_out")
freeze(pipeline.transformer)
```

`"proj_out"` 对应于 `pipeline.transformer` 的最后一层。下表列出了各种设置的结果:

<table>
<tr style="text-align: center;">
    <th> 量化文本编码器: 否 , 不量化的层: 无 </th>
    <th> 量化文本编码器: 否 , 不量化的层: "proj_out"</th>
    <th> 量化文本编码器: 是 , 不量化的层: 无 </th>
    <th> 量化文本编码器: 是 , 不量化的层: "proj_out"</th>
</tr>
<tr>
    <td><img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckpt%40pixart-bs%401-dtype%40bf16-qtype%40int4-qte%400-fuse%400.png" width=300 alt="Image 1 without text encoder quantization."/></td>
    <td><img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckpt%40pixart-bs%401-dtype%40bf16-qtype%40int4-qte%400-fuse%400-exclude%40proj_out.png" width=300 alt="Image 2 without text encoder quantization but with proj_out excluded in diffusion transformer quantization."/></td>
    <td><img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckpt%40pixart-bs%401-dtype%40bf16-qtype%40int4-qte%401-fuse%400.png" width=300 alt="Image 3 with text encoder quantization."/></td>
    <td><img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quanto-diffusers/ckpt%40pixart-bs%401-dtype%40bf16-qtype%40int4-qte%401-fuse%400-exclude%40proj_out.png" width=300 alt="Image 3 with text encoder quantization but with proj_out excluded in diffusion transformer quantization.."/></td>
</tr>
</table>

为了恢复损失的图像质量，常见的做法是进行量化感知训练，Quanto 也支持这种训练。这项技术超出了本文的范围，如果你有兴趣，请随时与我们联系！

本文的所有实验结果都可以在 [这里](https://huggingface.co/datasets/sayakpaul/sample-datasets/tree/main/quanto-exps-2) 找到。

## 加个鸡腿 - 在 Quanto 中保存和加载 Diffusers 模型

以下代码可用于对 Diffusers 模型进行量化并保存量化后的模型:

```python
from diffusers import PixArtTransformer2DModel
from optimum.quanto import QuantizedPixArtTransformer2DModel, qfloat8

model = PixArtTransformer2DModel.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="transformer")
qmodel = QuantizedPixArtTransformer2DModel.quantize(model, weights=qfloat8)
qmodel.save_pretrained("pixart-sigma-fp8")
```

此代码生成的 checkpoint 大小为 _**587MB**_  ，而不是原本的 2.44GB。然后我们可以加载它:

```python
from optimum.quanto import QuantizedPixArtTransformer2DModel
import torch

transformer = QuantizedPixArtTransformer2DModel.from_pretrained("pixart-sigma-fp8")
transformer.to(device="cuda", dtype=torch.float16)
```

最后，在 `DiffusionPipeline` 中使用它:

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    transformer=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.transformer = transformer

prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(prompt).images[0]
```

将来，我们计划支持在初始化流水线时直接传入 `transformer` 就可以工作:

```diff
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
- transformer=None,
+ transformer=transformer,
    torch_dtype=torch.float16,
).to("cuda")
```

`QuantizedPixArtTransformer2DModel` 实现可参考 [此处](https://github.com/huggingface/optimum-quanto/blob/601dc193ce0ed381c479fde54a81ba546bdf64d1/optimum/quanto/models/diffusers_models.py#L184)。如果你希望 Quanto 支持对更多的 Diffusers 模型进行保存和加载，请在 [此处](https://github.com/huggingface/optimum-quanto/issues/new) 提出需求并 `@sayakpaul` 。

## 小诀窍

- 根据应用场景的不同，你可能希望对流水线中不同的模块使用不同类型的量化。例如，你可以对文本编码器进行 FP8 量化，而对 transformer 扩散模型进行 INT8 量化。由于 Diffusers 和 Quanto 的灵活性，你可以轻松实现这类方案。
- 为了优化你的用例，你甚至可以将量化与 Diffuser 中的其他 [内存优化技术]((https://huggingface.co/docs/diffusers/main/en/optimization/memory)) 结合起来，如 `enable_model_cpu_offload() ` 。

## 总结

本文，我们展示了如何量化 Diffusers 中的 transformer 模型并优化其内存消耗。当我们同时对文本编码器进行量化时，效果变得更加明显。我们希望大家能将这些工作流应用到你的项目中并从中受益🤗。

感谢 [Pedro Cuenca](https://github.com/pcuenca) 对本文的细致审阅。