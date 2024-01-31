---
title: "在免费版 Google Colab 上使用 🧨 diffusers 运行 IF"
thumbnail: /blog/assets/if/thumbnail.jpg
authors:
- user: shonenkov
  guest: true
- user: Gugutse
  guest: true
- user: ZeroShot-AI
  guest: true
- user: williamberman
- user: patrickvonplaten
- user: multimodalart
translators:
- user: SuSung-boy
---

# 在免费版 Google Colab 上使用 🧨 diffusers 运行 IF

<a target="_blank" href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/deepfloyd_if_free_tier_google_colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


**本文简介**: 本文展示了如何在免费版 Google Colab 上使用 🧨 diffusers 运行最强大的开源文本生成图片模型之一 **IF**。

您也可以直接访问 IF 的 [Hugging Face Space](https://huggingface.co/spaces/DeepFloyd/IF) 页面来探索模型强大的性能。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/nabla.jpg" alt="if-collage"><br>
    <em>压缩的生成图片样例，选自官方 <a href="https://github.com/deep-floyd/IF/blob/release/pics/nabla.jpg">IF GitHub 库</a></em>
</p>

## 介绍

IF 是一类像素级的文生图模型，由 [DeepFloyd](https://github.com/deep-floyd/IF) 于 2023 年 4 月下旬发布。IF 的模型架构受 Google 的闭源模型 [Imagen](https://imagen.research.google/) 的强烈启发。

与现有的文本生成图片模型（如 Stable Diffusion）相比，IF 有两个明显的优势：

- IF 模型直接在 “像素空间”（即未降维、未压缩的图片）中计算生成，而非需要迭代去噪的隐空间（如 [Stable Diffusion](http://hf.co/blog/stable_diffusion)）。
- IF 模型基于 [T5-XXL](https://huggingface.co/google/t5-v1_1-xxl) 文本编码器的输出进行训练。T5-XXL 是一个比 Stable DIffusion 中的 [CLIP](https://openai.com/research/clip) 更强大的文本编码器。

因此，IF 更擅长生成具有高频细节（例如人脸和手部）的图片，并且 IF 是 **第一个能够在图片中生成可靠文字** 的开源图片生成模型。

不过，在具有上述两个优势（像素空间计算、使用更优文本编码器）的同时，IF 模型也存在明显的不足，那就是参数量更加庞大。IF 模型的文本编码器 T5、文本生成图片网络 UNet、超分辨率模型 upscaler UNet 的参数量分别为 4.5B、4.3B、1.2B，而 [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) 模型的文本编码器 CLIP 和去噪网络 UNet 的参数量仅为 400M 和 900M。

尽管如此，我们仍然可以在消费级 GPU 上运行 IF 模型，不过这需要一些优化技巧来降低显存占用。不用担心，我们将在本篇博客中详细介绍如何使用 🧨 diffusers 库来实现这些技巧。

在本文后面的 1.) 中，我们将介绍如何使用 IF 模型进行文本生成图片；在 2.) 和 3.) 中，我们将介绍 IF 模型的 Img2Img 和 Inpainting (图片修复) 能力。

💡 **注意**：本文为保证 IF 模型可以在免费版 Google Colab 上成功运行，采用了多模型组件顺序在 GPU 上加载卸载的技巧，以放慢生成速度为代价换取显存占用降低。如果您有条件使用更高端的 GPU 如 A100，我们建议您把所有的模型组件都加载并保留在 GPU 上，以获得最快的图片生成速度，代码详情见 [IF 的官方示例](https://huggingface.co/spaces/DeepFloyd/IF)。

💡 **注意**：本文为保证读者在阅读时图片加载得更快，对文中的一些高分辨率图片进行了压缩。在您自行使用官方模型尝试生成时，图片质量将会更高！

让我们开始 IF 之旅吧！🚀

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/meme.png"><br>
    <em>IF 模型生成含文字的图片的强大能力</em>
</p>

## 本文目录

* [接受许可证](#接受许可证)
* [优化 IF 模型以在有限的硬件条件下运行](#优化-if-模型以在有限的硬件条件下运行)
* [可用资源](#可用资源)
* [安装依赖](#安装依赖)
* [文本生成图片](#1-文本生成图片)
* [Img2Img](#2-img2img)
* [Inpainting](#3-inpainting)

## 接受许可证

在您使用 IF 模型之前，您需要接受它的使用条件。 为此：

- 1. 确保已开通 [Hugging Face 帐户](https://huggingface.co/join) 并登录
- 2. 接受 [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) 模型卡的许可证。在 Stage1 模型卡上接受许可证会自动接受其他 IF 模型许可证。
- 3. 确保在本地已安装 `huggingface_hub` 库并登录

```sh
pip install huggingface_hub --upgrade
```

在 Python shell 中运行登录函数

```py
from huggingface_hub import login

login()
```

输入您的 [Hugging Face Hub 访问令牌](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens)。

## 优化 IF 模型以在有限的硬件条件下运行

**最先进的机器学习技术不应该只掌握在少数精英手里。** 要使机器学习更 “普惠大众” 就意味着模型能够在消费级硬件上运行，而不是仅支持在最新型最高端的硬件上运行。

深度学习开放社区创造了众多世界一流的工具，来支持在消费级硬件上运行资源密集型模型。例如:

- [🤗 accelerate](https://github.com/huggingface/accelerate) 提供用于处理 [大模型](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) 的实用工具。
- [🤗 safetensors](https://github.com/huggingface/safetensors) 在保证模型保存的安全性的同时，还能显著加快大模型的加载速度。
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 使所有的 PyTorch 模型都可以采用 8 位量化。

Diffusers 库无缝集成了上述库，只需调用一个简单的 API 即可实现大模型的优化。

免费版 Google Colab 既受 CPU RAM 限制（13GB RAM），又受 GPU VRAM 限制（免费版 T4 为 15GB RAM），无法直接运行整个 IF 模型（>10B）。

我们先来看看运行完整 float32 精度的 IF 模型时，各个组件所需的内存占用：

- [T5-XXL 文本编码器](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0/tree/main/text_encoder): 20GB
- [Stage1 UNet](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0/tree/main/unet): 17.2GB
- [Stage2 超分辨率 UNet](https://huggingface.co/DeepFloyd/IF-II-L-v1.0/blob/main/pytorch_model.bin): 2.5 GB
- [Stage 3 x4-upscaler 超分辨率模型](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler): 3.4GB

可见我们无法以 float32 精度运行 IF 模型，因为 T5 和 Stage1 UNet 权重所需的内存占用均超出了免费版 CPU RAM 的可用范围。

很容易想到，我们可以通过降低模型运行的位精度来减少内存占用。如果以 float16 精度来运行 IF 模型，则 T5、Stage1 UNet、Stage2 UNet 所需的内存占用分别下降至 11GB、8.6GB、1.25GB。对于免费版 GPU 的 15GB RAM 限制，float16 精度已经满足运行条件，不过在实际加载 T5 模型时，我们很可能仍然会遇到 CPU 内存溢出错误，因为 CPU 的一部分内存会被其他进程占用。

因此我们继续降低位精度，实际上仅降低 T5 的精度就可以了。这里我们使用 `bitsandbytes` 库将 T5 量化到 8 位精度，最终可以将 T5 权重的内存占用降低至 [8GB](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0/blob/main/text_encoder/model.8bit.safetensors)。

好了，现在 IF 模型的每个组件的 CPU 和 GPU 内存占用都各自符合免费版 Google Colab 的限制，接下来我们只需要确保在运行每个组件的时候，CPU 和 GPU 内存不会被其他组件或者进程占用就可以了。

Diffusers 库支持模块化地独立加载单个组件，也就是说我们可以只加载文本编码器 T5，而不加载文本生成图片模型 UNet，反之亦然。这种模块化加载的技巧可以确保在运行多个组件的管线时，每个组件仅在需要计算时才被加载，可以有效避免同时加载时导致的 CPU 和 GPU 内存溢出。

来实操代码试一试吧！🚀

![t2i_64](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/t2i_64.png)

## 可用资源

免费版 Google Colab 的 CPU RAM 可用资源约 13GB：

``` python
!grep MemTotal /proc/meminfo
```

```bash
MemTotal:       13297192 kB
```

免费版 GPU 型号为 NVIDIA T4，其 VRAM 可用资源约 15GB:

``` python
!nvidia-smi
```

```bash
Sun Apr 23 23:14:19 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   72C    P0    32W /  70W |   1335MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                                
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

## 安装依赖

本文使用的优化技巧需要安装最新版本的依赖项。如果您在运行代码时遇到问题，请首先仔细检查依赖项的安装版本。

``` python
! pip install --upgrade \
  diffusers~=0.16 \
  transformers~=4.28 \
  safetensors~=0.3 \
  sentencepiece~=0.1 \
  accelerate~=0.18 \
  bitsandbytes~=0.38 \
  torch~=2.0 -q
```

## 1. 文本生成图片

这一部分我们将分步介绍如何使用 Diffusers 运行 IF 模型来完成文本到图片的生成。对于接下来使用的 API 和优化技巧，文中仅作简要的解释，如果您想深入了解更多原理或者细节，可以前往 [Diffusers](https://huggingface.co/docs/diffusers/index)，[Transformers](https://huggingface.co/docs/transformers/index)，[Accelerate](https://huggingface.co/docs/accelerate/index)，以及 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 的官方文档查看。

### 1.1 加载文本编码器

首先我们使用 Transformers 库加载 8 位量化后的文本编码器 T5。Transformers 库直接支持 [bitsandbytes](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-8bit) 量化，可以通过 `load_in_8bit` 参数来标识是否加载 8 位量化模型。

设置参数 `variant="8bit"` 来下载预量化版的权重。

Transformers 还支持模块化地独立加载单个模型的某些层！`device_map` 参数可以指定单个模型的权重在不同 GPU 设备上加载或者卸载的映射策略，在不需要参与计算时甚至可以卸载到 CPU 或者磁盘上。这里我们设置 `device_map` 参数为 `"auto"`，让 transformers 库自动创建设备映射。更多相关信息，请查看 [transformers 文档](https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map)。

``` python
from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    subfolder="text_encoder", 
    device_map="auto", 
    load_in_8bit=True, 
    variant="8bit"
)
```

### 1.2 创建 prompt embeddings

Diffusers API 中的 `DiffusionPipeline` 类及其子类专门用于访问扩散模型。`DiffusionPipeline` 中的每个实例都包含一套独立的方法和默认的模型。我们可以通过 `from_pretrained` 方法来覆盖默认实例中的模型，只需将目标模型实例作为关键字参数传给 `from_pretrained`。

上文说过，我们在加载文本编码器 T5 的时候无需加载扩散模型组件 UNet，因此这里我们需要用 `None` 来覆盖 `DiffusionPipeline` 的实例中的 UNet 部分，此时将 `from_pretrained` 方法的 `unet` 参数设为 `None` 即可实现。

``` python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", 
    text_encoder=text_encoder, # 传入前面加载的 8 位量化文本编码器实例
    unet=None, 
    device_map="auto"
)
```

IF 模型还有一个超分辨率管线。为了后面能够方便地加载运行，我们这里把 prompt embeddings 保存下来，后面就可以直接输入给超分辨率管线，而不需要再经过文本编码器了。

接下来就可以开始输入 prompt 了。为了凸显 IF 模型能够生成带文字的图片的优势，这里要在 Stable Diffusion 中生成 [宇航员骑马](https://huggingface.co/blog/stable_diffusion) (an astronaut just riding a
horse) 的图片示例的基础上, 增加一个带有文字的指示牌！

我们给出一个合适的 prompt：

``` python
prompt = "a photograph of an astronaut riding a horse holding a sign that says Pixel's in space"
```

然后输入给 8 位量化的 T5 模型，生成 prompt 的 embeddings：

``` python
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
```

### 1.3 释放内存

当 prompt embeddings 创建完成之后，我们就不再需要文本编码器了。但目前 T5 仍然存在于 GPU 内存中，因此我们需要释放 T5 占用的内存，以便加载 UNet。

释放 PyTorch 内存并非易事。我们必须对所有指向实际分配到 GPU 上的 Python 对象实施垃圾回收。

为此，我们首先使用 Python 关键字 `del` 来删除掉所有引用的已分配到 GPU 内存上的 Python 对象。

``` python
del text_encoder
del pipe
```

不过仅删除 Python 对象仍然不够，因为垃圾回收机制实际上是在释放 GPU 完成之后才完成的。

然后，我们调用 `torch.cuda.empty_cache()` 方法来释放缓存。实际上该方法也并非绝对必要，因为缓存中的 cuda 内存也能够立即用于进一步分配，不过它可以帮我们在 Colab UI 中验证是否有足够的内存可用。

这里我们编写一个辅助函数 `flush()` 来刷新内存。

``` python
import gc
import torch

def flush():
    gc.collect()
    torch.cuda.empty_cache()
```

运行 `flush()`。

``` python
flush()
```

### 1.4 Stage1：核心扩散过程

好了，现在已经有足够的 GPU 内存可用，我们就能重新加载一个只包含 UNet 部分的 `DiffusionPipeline` 了，因为接下来我们只需要运行核心扩散过程部分。

按照上文中对 UNet 内存占用的计算，IF 模型的 UNet 部分权重能够以 float16 精度加载，设置 `variant` 和 `torch_dtype` 参数即可实现。

``` python
pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", 
    text_encoder=None, 
    variant="fp16", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
```

一般情况下，我们会直接将 prompt 传入 `DiffusionPipeline.__call__` 函数。不过我们这里已经计算出了 prompt embeddings，因此只需传入 embeddings 即可。

Stage1 的 UNet 接收 embeddings 作为输入运行完成后，我们还需要继续运行 Stage2 的超分辨率组件，因此我们需要保存模型的原始输出 (即 PyTorch tensors) 来输入到 Stage2，而不是 PIL 图片。这里设置参数 `output_type="pt"` 可以将 Stage1 输出的 PyTorch tensors 保留在 GPU 上。

我们来定义一个随机生成器，并运行 Stage1 的扩散过程。

``` python
generator = torch.Generator().manual_seed(1)
image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds, 
    output_type="pt",
    generator=generator,
).images
```

虽然运行结果是原始的 PyTorch tensors，我们仍然可以手动将其转换为 PIL 图片，起码先瞧一瞧生成图片的大概样子嘛。Stage1 的输出可以转换为一张 64x64 的图片。

``` python
from diffusers.utils import pt_to_pil

pil_image = pt_to_pil(image)
pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

pil_image[0]
```

![t2i_64](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/t2i_64.png)

Stage1 完成之后，我们同样删除 Python 指针，释放 CPU 和 GPU 内存。

``` python
del pipe
flush()
```

### 1.5 Stage2：超分辨率 64x64 到 256x256

IF 模型包含多个独立的超分辨率组件。

对于每个超分辨率扩散过程组件，我们都使用单独的管线来运行。

在加载超分辨率管线时需要传入文本参数。如果需要，它也是可以同时加载文本编码器，来从 prompt 开始运行的。不过更一般的做法是从第一个 IF 管线中计算得到的 prompt embeddings 开始，此时要把 `text_encoder` 参数设为 `None`。

创建一个超分辨率 UNet 管线。

``` python
pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", 
    text_encoder=None, # 未用到文本编码器 => 节省内存!
    variant="fp16", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
```

将 Stage1 输出的 Pytorch tensors 和 T5 输出的 embeddings 输入给 Stage2 并运行。

``` python
image = pipe(
    image=image, 
    prompt_embeds=prompt_embeds, 
    negative_prompt_embeds=negative_embeds, 
    output_type="pt",
    generator=generator,
).images
```

我们同样可以转换为 PIL 图片来查看中间结果。

``` python
pil_image = pt_to_pil(image)
pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

pil_image[0]
```

![t2i_upscaled](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/t2i_upscaled.png)

再一次，删除 Python 指针，释放内存。

``` python
del pipe
flush()
```

### 1.6 Stage3：超分辨率 256x256 到 1024x1024

IF 模型的第 2 个超分辨率组件是 Stability AI 之前发布的 [x4 Upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)。

我们创建相应的管线，并设置参数 `device_map="auto"` 直接加载到 GPU 上。

``` python
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
```

🧨 diffusers 可以使得独立开发的扩散模型非常简便地组合使用，因为 diffusers 中的管线可以链接在一起。比如这里我们可以设置参数 `image=image` 来将先前输出的 PyTorch tensors 输入给 Stage3 管线。

💡 **注意**：x4 Upscaler 并非使用 T5，而使用它 [自己的文本编码器](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/main/text_encoder)。因此，我们不能使用 1.2 中创建的 prompt embeddings，必须传入原始 prompt。

``` python
pil_image = pipe(prompt, generator=generator, image=image).images
```

IF 模型管线在生成图片时默认会在右下角添加 IF 水印。由于 Stage3 使用的 x4 upscaler 管线并非属于 IF (实际上属于 Stable Diffusion)，因此经过超分辨率生成的图片也不会带有 IF 水印。

不过我们可以手动添加水印。

``` python
from diffusers.pipelines.deepfloyd_if import IFWatermarker

watermarker = IFWatermarker.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="watermarker")
watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)
```

查看 Stage3 的输出图片。

``` python
pil_image[0]
```

![t2i_upscaled_2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/t2i_upscaled_2.png)

看！免费版 Google Colab 上运行 IF 模型生成精美的 1024x1024 图片了！

至此，我们已经展示了使用 🧨 diffusers 来分解和模块化加载资源密集型扩散模型的全部内容，是不是非常简单！

💡 **注意**：我们不建议在生产流程中使用上述以放慢推理速度为代价来换取低内存消耗的设置：8 位量化、模型权重的解耦和重分配、磁盘卸载等，尤其是需要重复使用某个扩散模型组件的时候。在实际生产中，我们还是建议您使用 40GB VRAM 的 A100，以确保所有的模型组件可以同时加载到 GPU 上。如果您条件满足，可以参考 Hugging Face 上的 [**官方 IF 示例**](https://huggingface.co/spaces/DeepFloyd/IF) 设置。

## 2. Img2Img

在 1.) 中加载的文本生成图片的 IF 模型各个组件的预训练权重，也同样可用于文本引导的图片生成图片，也叫 Img2Img，还能用于 Inpainting (图片修复)，我们将在 3.) 中介绍。Img2Img 和 Inpainting 的核心扩散过程，除了初始噪声是图片之外，其余均与文本生成图片的扩散过程相同。

这里我们创建 Img2Img 管线 `IFImg2ImgPipeline` 和超分辨率管线
`IFImg2ImgSuperResolution`，并加载和 1.) 中各个组件相同的预训练权重。

内存优化的 API 也都相同！

同样地释放内存。

``` python
del pipe
flush()
```

对于 Img2Img，我们需要一张初始图片。

这一部分，我们将使用在外网著名的 “Slaps Roof of Car” meme (可以理解为汽车推销员表情包制作模板)。首先从网上下载这张图片。

``` python
import requests

url = "https://i.kym-cdn.com/entries/icons/original/000/026/561/car.jpg"
response = requests.get(url)
```

然后使用 PIL 图像库加载图片。

``` python
from PIL import Image
from io import BytesIO

original_image = Image.open(BytesIO(response.content)).convert("RGB")
original_image = original_image.resize((768, 512))
original_image
```

![iv_sample](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/iv_sample.png)

Img2Img 管线可以接收 PIL 图像对象或原始 tensors 对象作为输入。点击 [此处](https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/if#diffusers.IFImg2ImgPipeline.__call__) 可跳转文档页面查看更详细的输入参数说明。

### 2.1 文本编码器

Img2Img 可以由文本引导。这里我们也尝试给出一个合适的 prompt 并使用文本编码器 T5 创建其 embeddings。

首先再次加载 8 位量化的文本编码器。

``` python
from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    subfolder="text_encoder", 
    device_map="auto", 
    load_in_8bit=True, 
    variant="8bit"
)
```

对于 Img2Img，我们需要使用 [`IFImg2ImgPipeline`](https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/if#diffusers.IFImg2ImgPipeline) 类来加载预训练权重，而不能使用 1.) 中的 `DiffusionPipeline` 类。这是因为当使用 `from_pretrained()` 方法加载 IF 模型（或其他扩散模型）的预训练权重时，会返回 **默认的文本生成图片** 管线 [`IFPipeline`](https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/if#diffusers.IFPipeline)。因此，要加载 Img2Img 或 Depth2Img 等非默认形式的管线，必须指定明确的类名。

``` python
from diffusers import IFImg2ImgPipeline

pipe = IFImg2ImgPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", 
    text_encoder=text_encoder, 
    unet=None, 
    device_map="auto"
)
```

我们来把汽车推销员变得动漫风一些，对应的 prompt 为：

``` python
prompt = "anime style"
```

同样地，使用 T5 来创建 prompt embeddings。

``` python
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
```

释放 CPU 和 GPU 内存。

同样先删除 Python 指针，

``` python
del text_encoder
del pipe
```

再刷新内存。

``` python
flush()
```

### 2.2 Stage1：核心扩散过程

接下来也是一样，我们在管线中只加载 Stage1 UNet 部分权重。

``` python
pipe = IFImg2ImgPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", 
    text_encoder=None, 
    variant="fp16", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
```

运行 Img2Img Stage1 管线需要原始图片和 prompt embeddings 作为输入。

我们可以选择使用 `strength` 参数来配置 Img2Img 的变化程度。`strength` 参数直接控制了添加的噪声强度，该值越高，生成图片偏离原始图片的程度就越大。

``` python
generator = torch.Generator().manual_seed(0)
image = pipe(
    image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds, 
    output_type="pt",
    generator=generator,
).images
```

我们再次查看一下生成的 64x64 图片。

``` python
pil_image = pt_to_pil(image)
pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

pil_image[0]
```

![iv_sample_1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/iv_sample_1.png)

看起来不错！我们可以继续释放内存，并进行超分辨率放大图片了。

``` python
del pipe
flush()
```

### 2.3 Stage2: 超分辨率

对于超分辨率，我们使用 `IFImg2ImgSuperResolutionPipeline` 类，并加载与 1.5 中相同的预训练权重。

``` python
from diffusers import IFImg2ImgSuperResolutionPipeline

pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", 
    text_encoder=None, 
    variant="fp16", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
```
💡 **注意**：Img2Img 超分辨率管线不仅需要 Stage1 输出的生成图片，还需要原始图片作为输入。

实际上我们还可以在 Stage2 输出的图片基础上继续使用 Stable Diffusion x4 upscaler 进行二次超分辨率。不过这里没有展示，如果需要，请使用 1.6 中的代码片段进行尝试。

``` python
image = pipe(
    image=image,
    original_image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds, 
    generator=generator,
).images[0]
image
```

![iv_sample_2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/iv_sample_2.png)

好了！Img2Img 的全部内容也介绍完毕。我们继续释放内存，然后介绍最后一个 Inpainting 管线。

``` python
del pipe
flush()
```

## 3. Inpainting

IF 模型的 Inpainting 管线大体上与 Img2Img 相同，只不过仅对图片的部分指定区域进行去噪和生成。

我们首先用图片 mask 来指定一个待修复区域。

让我们来展示一下 IF 模型 “生成带文字的图片” 这项令人惊叹的能力！我们来找一张带标语的图片，然后用 IF 模型替换标语的文字内容。

首先下载图片

``` python
import requests

url = "https://i.imgflip.com/5j6x75.jpg"
response = requests.get(url)
```

并将其转换为 PIL 图片对象。

``` python
from PIL import Image
from io import BytesIO

original_image = Image.open(BytesIO(response.content)).convert("RGB")
original_image = original_image.resize((512, 768))
original_image
```

![inpainting_sample](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/inpainting_sample.png)

我们指定标语牌区域为 mask 待修复区域，让 IF 模型替换该区域的文字内容。

为方便起见，我们已经预生成了 mask 图片并将其加载到 HF 数据集中了。

下载 mask 图片。

``` python
from huggingface_hub import hf_hub_download

mask_image = hf_hub_download("diffusers/docs-images", repo_type="dataset", filename="if/sign_man_mask.png")
mask_image = Image.open(mask_image)

mask_image
```

![masking_sample](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/masking_sample.png)

💡 **注意**：您也可以自行手动创建灰度 mask 图片。下面是一个创建 mask 图片的代码例子。

``` python
from PIL import Image
import numpy as np

height = 64
width = 64

example_mask = np.zeros((height, width), dtype=np.int8)

# 设置待修复区域的 mask 像素值为 255
example_mask[20:30, 30:40] = 255

# 确保 PIL 的 mask 图片模式为 'L'
# 'L' 代表单通道灰度图
example_mask = Image.fromarray(example_mask, mode='L')

example_mask
```

![masking_by_hand](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/masking_by_hand.png)

好了，我们可以开始修复图片了🎨🖌 

### 3.1. 文本编码器

我们同样先加载文本编码器。

``` python
from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    subfolder="text_encoder", 
    device_map="auto", 
    load_in_8bit=True, 
    variant="8bit"
)
```

再创建一个 inpainting 管线，这次使用 `IFInpaintingPipeline` 类并初始化文本编码器预训练权重。

``` python
from diffusers import IFInpaintingPipeline

pipe = IFInpaintingPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", 
    text_encoder=text_encoder, 
    unet=None, 
    device_map="auto"
)
```

我们来让图片中的这位男士为 “just stack more layers” 作个代言！

*注：外网中的一个梗，每当现有神经网络解决不了现有问题时，就会有 Just Stack More Layers！ ......*

``` python
prompt = 'the text, "just stack more layers"'
```

给定 prompt 之后，接着创建 embeddings。

``` python
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
```

然后再次释放内存。

``` python
del text_encoder
del pipe
flush()
```

### 3.2 Stage1: 核心扩散过程 

同样地，我们只加载 Stage1 UNet 的预训练权重。

``` python
pipe = IFInpaintingPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", 
    text_encoder=None, 
    variant="fp16", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
```

这里，我们需要传入原始图片、mask 图片和 prompt embeddings。

``` python
image = pipe(
    image=original_image,
    mask_image=mask_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds, 
    output_type="pt",
    generator=generator,
).images
```

可视化查看一下中间输出。

``` python
pil_image = pt_to_pil(image)
pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

pil_image[0]
```

![inpainted_output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/inpainted_output.png)

看起来不错！标语牌上的文字内容非常连贯！

我们继续释放内存，做超分辨率放大图片。

``` python
del pipe
flush()
```

### 3.3 Stage2: 超分辨率

对于超分辨率，使用 `IFInpaintingSuperResolutionPipeline` 类来加载预训练权重。

``` python
from diffusers import IFInpaintingSuperResolutionPipeline

pipe = IFInpaintingSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", 
    text_encoder=None, 
    variant="fp16", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
```

IF 模型的 inpainting 超分辨率管线需要接收 Stage1 输出的图片、原始图片、mask 图片、以及 prompt embeddings 作为输入。

让我们运行最后的超分辨率管线。

``` python
image = pipe(
    image=image,
    original_image=original_image,
    mask_image=mask_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds, 
    generator=generator,
).images[0]
image
```

![inpainted_final_output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/if/inpainted_final_output.png)

非常好！IF 模型生成的单词没有出现任何拼写错误！

## 总结

运行完整的 float32 精度的 IF 模型共需要至少 40GB 内存。本文展示了如何仅使用开源库来使 IF 模型能够在免费版 Google Colab 上运行并生成图片。

机器学习领域的生态如此壮大主要受益于各种工具和模型的开源共享。本文涉及到的模型来自于 DeepFloyd, StabilityAI, 以及 [Google](https://huggingface.co/google)，涉及到的库有 Diffusers, Transformers, Accelerate, 和 bitsandbytes 等，它们同样来自于不同组织的无数贡献者。

非常感谢 DeepFloyd 团队创建和开源 IF 模型，以及为良好的机器学习生态做出的贡献🤗。

