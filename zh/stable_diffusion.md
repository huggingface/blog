---
title: 使用Diffusers来实现Stable Diffusion 🧨
thumbnail: /blog/assets/98_stable_diffusion/thumbnail.png
authors:
- user: valhalla
- user: pcuenq
- user: natolambert
- user: patrickvonplaten
translators:
- user: tunglinwu
---

# 使用Diffusers来实现Stable Diffusion 🧨


<a target="_blank" href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**实现Stable Diffusion的效果** 🎨 *...借由 🧨 Diffusers*

Stable Diffusion 是一种文本到图像的潜在扩散模型，由 [CompVis](https://github.com/CompVis)、[Stability AI](https://stability.ai/) 和 [LAION](https://laion.ai/) 的研究人员和工程师创建。它是在 [LAION-5B](https://laion.ai/blog/laion-5b/) 数据库的一个子集上使用 512x512 图像训练的。*LAION-5B* 是目前存在的最大、可自由访问的多模态数据集。

在这篇文章中，我们将展示如何使用 [🧨 Diffusers 库](https://github.com/huggingface/diffusers)中的 Stable Diffusion 模型，解释模型的工作原理，并深入探讨 `diffusers` 如何让用户定制图像生成流水线。

**注意**: 强烈建议您对扩散模型有基本的了解。如果您对扩散模型完全陌生，我们建议阅读以下博客文章之一：
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Getting started with 🧨 Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)

现在，让我们开始生成一些图像吧 🎨。

## 运行 Stable Diffusion

### 许可证

在使用模型之前，您需要接受该模型的[许可证](https://huggingface.co/spaces/CompVis/stable-diffusion-license)，以便下载和使用权重。**注意：现在不再需要通过 UI 显式接受许可证**。

该许可证旨在减轻如此强大的机器学习系统可能带来的潜在有害影响。我们请求用户**完整且仔细地阅读许可证**。以下是摘要：
1. 您不能故意使用模型生成或分享非法或有害的输出或内容。
2. 我们对您生成的输出不主张任何权利，您可以自由使用这些输出，并对其使用负责，且不得违反许可证中规定的条款。
3. 您可以重新分发权重，并将模型用于商业用途和/或作为服务使用。如果这样做，请注意，您必须包括与许可证中相同的使用限制，并向所有用户提供 CreativeML OpenRAIL-M 的副本。


### 使用方法

首先，您应该安装 `diffusers==0.10.2` 以运行以下代码片段：

```bash
pip install diffusers==0.10.2 transformers scipy ftfy accelerate
```

在这篇文章中，我们将使用模型版本 [`v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4)，但您也可以使用其他版本的模型，如 1.5、2 和 2.1，只需做最小的代码修改。

Stable Diffusion 模型可以使用 [`StableDiffusionPipeline`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) 流水线在推理中运行，仅需几行代码即可。流水线设置了从文本生成图像所需的一切，只需一个简单的 `from_pretrained` 函数调用。

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
```

如果有 GPU 可用，咱们把它移过去吧！

```python
pipe.to("cuda")
```

**注意**: 如果您受限于 GPU 内存且 GPU RAM 少于 10GB，请确保加载 `StableDiffusionPipeline` 时使用 float16 精度，而不是上述的默认 float32 精度。

您可以通过加载 `fp16` 分支的权重并告诉 `diffusers` 权重为 float16 精度来实现：

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
```

要运行流水线，只需定义提示词并调用 `pipe`。

```python
prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt).images[0]

# 您可以用以下代码保存图像
# image.save(f"astronaut_rides_horse.png")
```

结果如下所示

![png](assets/98_stable_diffusion/stable_diffusion_12_1.png)
    
每次运行上面的代码都会生成不同的图像。

如果您某个时候得到了黑色图像，可能是因为模型内置的内容过滤器可能检测到不适合的内容。如果您认为不该是这样，可以尝试调整提示词或使用不同的种子。事实上，模型预测结果中包含是否检测到不适合内容的信息。让我们看看它们是什么样子：

```python
result = pipe(prompt)
print(result)
```

```json
{
    'images': [<PIL.Image.Image image mode=RGB size=512x512>],
    'nsfw_content_detected': [False]
}
```

如果您想要确定性的输出，可以设定一个随机种子并将生成器传递给流水线。每次使用相同种子的生成器时，您将得到相同的图像输出。


```python
import torch

generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]

# 您可以用以下代码保存图像
# image.save(f"astronaut_rides_horse.png")
```

结果如下所示

![png](assets/98_stable_diffusion/stable_diffusion_14_1.png)

您可以使用 `num_inference_steps` 参数更改推理步骤的数量。

通常，步骤越多，结果越好，但是步骤越多，生成所需的时间也越长。Stable Diffusion 在相对较少的步骤下表现得很好，所以我们建议使用默认的 `50` 步推理步骤。如果您想要更快的结果，可以使用更少的步骤。如果您想要可能更高质量的结果，可以使用更大的步骤数。

让我们尝试以更少的去噪步骤运行流水线。

```python
import torch

generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=15, generator=generator).images[0]

# 您可以用以下代码保存图像
# image.save(f"astronaut_rides_horse.png")
```

![png](assets/98_stable_diffusion/stable_diffusion_16_1.png)

注意图像的结构虽然相同，但宇航员的宇航服和马的整体形态出现了问题。这表明，仅使用15次去噪步骤显著降低了生成结果的质量。正如之前提到的，通常50次去噪步骤足以生成高质量图像。

除了`num_inference_steps`参数之外，我们在之前的所有示例中还使用了另一个名为`guidance_scale`的函数参数。`guidance_scale`是一种增强生成结果与条件信号（在本例中为文本）的符合度以及整体样本质量的方法。它也被称为[无分类器指导](https://arxiv.org/abs/2207.12598)，简单来说，它强制生成结果更好地匹配提示词，可能会以图像质量或多样性为代价。对于稳定扩散，`7`到`8.5`之间的值通常是较好的选择。默认情况下，管道使用`guidance_scale`为7.5。

如果使用非常大的值，图像可能看起来很好，但多样性会减少。你可以在本文的[此部分](#writing-your-own-inference-pipeline)了解此参数的技术细节。

接下来，我们看看如何一次生成同一提示的多张图像。首先，我们将创建一个`image_grid`函数，以帮助我们在网格中将它们美观地可视化。

```python
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
```

我们可以通过使用一个包含重复多次的相同提示词的列表来生成多张图像。我们将这个列表传递给管道，而不是之前使用的字符串。


```python
num_images = 3
prompt = ["a photograph of an astronaut riding a horse"] * num_images

images = pipe(prompt).images

grid = image_grid(images, rows=1, cols=3)

# 您可以用以下代码保存图像
# grid.save(f"astronaut_rides_horse.png")
```

![png](assets/98_stable_diffusion/stable_diffusion_22_1.png)

默认情况下，Stable Diffusion生成的图像为`512 × 512`像素。通过使用`height`和`width`参数，非常容易覆盖默认值以创建纵向或横向比例的矩形图像。

在选择图像尺寸时，我们建议以下几点：
- 确保`height`和`width`都是8的倍数。
- 尺寸低于512可能会导致图像质量降低。
- 在两个方向上超过512会导致图像区域重复（全局一致性丧失）。
- 创建非正方形图像的最佳方法是一个维度使用`512`，另一个维度使用大于`512`的值。

让我们运行一个示例：

```python
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt, height=512, width=768).images[0]

# 您可以用以下代码保存图像
# image.save(f"astronaut_rides_horse.png")
```

![png](assets/98_stable_diffusion/stable_diffusion_26_1.png)
    

## Stable Diffusion 是如何工作的？

在看到Stable Diffusion可以生成的高质量图像后，让我们尝试更好地理解模型的工作原理。

Stable Diffusion基于一种特殊类型的扩散模型，称为**潜在扩散(Latent Diffusion)**，该模型在[基于潜在扩散模型的高分辨率图像合成](https://arxiv.org/abs/2112.10752)中提出。

一般来说，扩散模型是通过一步步去噪高斯噪声，从而得到目标样本（例如*图像*）的机器学习系统。有关它们如何工作的更详细概述，请查看[此Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)。

扩散模型已被证明在生成图像数据方面达到了最先进的水平。但扩散模型的一个缺点是逆向去噪过程非常慢，因为它是重复的、序列化的。此外，这些模型消耗大量内存，因为它们在像素空间中操作，而在生成高分辨率图像时，像素空间变得非常庞大。因此，训练这些模型和进行推理都非常具有挑战性。

潜在扩散通过在低维的*潜在*空间上应用扩散过程来减少内存和计算复杂度，而不是使用实际的像素空间。这是标准扩散模型与潜在扩散模型之间的关键区别：**在潜在扩散中，模型被训练生成图像的潜在（压缩）表示。**

潜在扩散中有三个主要组件：

1. 一个自编码器（VAE）。
2. 一个[U-Net](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb#scrollTo=wW8o1Wp0zRkq)。
3. 一个文本编码器，例如[CLIP的文本编码器](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)。

**1. 自编码器（VAE）**

VAE模型有两个部分，一个编码器和一个解码器。编码器用于将图像转换为低维的潜在表示，这将作为*U-Net*模型的输入。
解码器则将潜在表示转化为图像。

在潜在扩散*训练*期间，编码器用于获取图像的潜在表示（_潜在变量_），用于正向扩散过程，在每一步中加入更多的噪声。在*推理*期间，通过逆向扩散过程生成的去噪潜在变量由VAE解码器转换回图像。正如我们将看到的，在推理期间我们**只需要VAE解码器**。

**2. U-Net**

U-Net的结构包括一个编码器部分和一个解码器部分，两者都由ResNet块组成。
编码器将图像表示压缩为较低分辨率的图像表示，而解码器将较低分辨率的图像表示解码回原始的较高分辨率图像表示，假定其噪声较少。
更具体地说，U-Net的输出预测了可以用来计算预测的去噪图像表示的噪声残差。

为了防止U-Net在下采样时丢失重要信息，通常会在编码器的下采样ResNet块和解码器的上采样ResNet块之间添加捷径连接。
此外，Stable Diffusion的U-Net能够通过交叉注意力层将其输出与文本嵌入进行条件化。交叉注意力层通常在编码器和解码器部分的ResNet块之间添加。

**3. 文本编码器**

文本编码器负责将输入提示，例如"An astronaut riding a horse"转换为U-Net可以理解的嵌入空间。它通常是一个简单的*基于变换器(transformer-based)的*编码器，用于将输入标记序列映射为一系列潜在的文本嵌入。

受[Imagen](https://imagen.research.google/)启发，Stable Diffusion在训练期间**不会**训练文本编码器，而是直接使用已经训练好的CLIP文本编码器，[CLIPTextModel](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)。

**为什么潜在扩散快且高效？**

由于潜在扩散在低维空间中操作，相比于像素空间的扩散模型，它极大地减少了内存和计算需求。例如，Stable Diffusion中使用的自编码器的缩减因子为8。这意味着形状为`(3, 512, 512)`的图像在潜在空间中变为`(4, 64, 64)`，这意味着空间压缩比为`8 × 8 = 64`。

这就是为什么即使在16GB的Colab GPU上，也能如此快速地生成`512 × 512`的图像的原因！


**推理中的稳定扩散**

将所有部分结合起来，我们现在来仔细看看模型在推理中的工作原理，并通过展示逻辑流程来进行说明

<p align="center">
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/stable_diffusion.png" alt="sd-pipeline" width="500"/>
</p>

稳定扩散模型同时接受一个潜在种子和文本提示作为输入。然后使用潜在种子生成大小为\\( 64 \times 64 \\)的随机潜在图像表示，而文本提示则通过CLIP的文本编码器转换为大小为\\( 77 \times 768 \\)的文本嵌入。

接下来，U-Net模型在文本嵌入的条件下，逐步对随机潜在图像表示进行*去噪*。U-Net的输出——即噪声残差——通过调度算法计算出去噪后的潜在图像表示。可以使用多种不同的调度算法来进行此计算，每种算法各有优缺点。对于稳定扩散，我们推荐使用以下几种调度器之一：

- [PNDM调度器](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py)（默认使用）
- [DDIM调度器](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py)
- [K-LMS调度器](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py)

关于调度算法如何工作的理论超出了本笔记本的范围，但简而言之，应该记住它们是根据前一个噪声表示和预测的噪声残差来计算预测的去噪图像表示的。有关更多信息，我们建议参考[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)。

*去噪*过程重复*约*50次，以逐步获得更好的潜在图像表示。
一旦完成，潜在图像表示将由变分自动编码器的解码器部分进行解码。

在对潜在扩散和稳定扩散进行简要介绍后，我们来看如何高级使用🤗 Hugging Face `diffusers`库！

## 编写自己的推理管道

最后，我们展示如何使用`diffusers`创建自定义的扩散管道。
编写自定义推理管道是`diffusers`库的高级用法，可以用于替换某些组件，例如上面提到的VAE或调度器。

例如，我们将展示如何使用不同的调度器，即[Katherine Crowson's](https://github.com/crowsonkb) K-LMS调度器，该调度器已在[此PR](https://github.com/huggingface/diffusers/pull/185)中添加。

[预训练模型](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main)包含设置完整扩散管道所需的所有组件。它们存储在以下文件夹中：
- `text_encoder`: 稳定扩散使用CLIP，但其他扩散模型可能使用其他编码器，如`BERT`。
- `tokenizer`: 必须与`text_encoder`模型所使用的分词器相匹配。
- `scheduler`: 在训练期间用于逐渐向图像添加噪声的调度算法。
- `unet`: 用于生成输入的潜在表示的模型。
- `vae`: 我们将用来将潜在表示解码为真实图像的自动编码器模块。

我们可以通过引用保存它们的文件夹来加载组件，使用`from_pretrained`中的`subfolder`参数。

```python
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# 1. 加载自动编码器模型，将用来将潜在表示解码为图像空间。
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. 加载分词器和文本编码器，以对文本进行分词和编码。
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. 用于生成潜在变量的UNet模型。
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
```

我们加载带有适配参数的[K-LMS调度器](https://github.com/huggingface/diffusers/blob/71ba8aec55b52a7ba5a1ff1db1265ffdd3c65ea2/src/diffusers/schedulers/scheduling_lms_discrete.py#L26)而不是加载预定义的调度器。

```python
from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
```

接下来，将模型移动到GPU上。

```python
torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device) 
```

现在我们定义生成图像时要使用的参数。

请注意，`guidance_scale`与[Imagen论文](https://arxiv.org/pdf/2205.11487.pdf)中的方程(2)中的指导权重`w`类似。`guidance_scale == 1`表示不进行分类器自由指导。这里我们将其设置为7.5，就像之前一样。

与之前的例子相比，我们将`num_inference_steps`设置为100，以获得更清晰的图像。

```python
prompt = ["a photograph of an astronaut riding a horse"]

height = 512 # 稳定扩散的默认高度
width = 512 # 稳定扩散的默认宽度

num_inference_steps = 100 # 去噪步骤数

guidance_scale = 7.5 # 分类器自由指导的比例

generator = torch.manual_seed(0) # 用于创建初始潜在噪声的种子生成器

batch_size = len(prompt)
```

首先，我们为传递的提示获取`text_embeddings`。这些嵌入将用于条件UNet模型，并引导图像生成接近输入提示。

```python
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
```

我们还将为分类器自由指导获取无条件的文本嵌入，即填充标记（空文本）的嵌入。它们需要具有与条件`text_embeddings`相同的形状（`batch_size`和`seq_length`）。

```python
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   
```

对于分类器自由指导，我们需要执行两次前向传递：一次使用条件输入（`text_embeddings`），另一次使用无条件嵌入（`uncond_embeddings`）。实际上，我们可以将两者连接成一个批次，以避免进行两次前向传递。


```python
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
```

接下来，生成初始随机噪声。

```python
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)
```

如果此时检查`latents`，我们会发现它们的形状为`torch.Size([1, 4, 64, 64])`，比我们要生成的图像小得多。稍后模型将把这种潜在表示（纯噪声）转换为`512 × 512`图像。

接下来，我们使用所选的`num_inference_steps`初始化调度器。
这将计算去噪过程中使用的`sigma`和确切时间步值。

```python
scheduler.set_timesteps(num_inference_steps)
```

K-LMS调度器需要将`latents`乘以其`sigma`值。让我们在此进行操作：


```python
latents = latents * scheduler.init_noise_sigma
```

我们已准备好编写去噪循环。


```python
from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
# 如果我们正在进行分类器自由指导，则扩展潜在变量，以避免进行两次前向传递。
latent_model_input = torch.cat([latents] * 2)

latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

# 预测噪声残差
with torch.no_grad():
noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

# 进行分类器自由指导
noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

# 计算去噪图像的隐空间表示
latents = scheduler.step(noise_pred, t, latents).prev_sample 
```

代码执行后，潜在变量`latents`应该不再只是噪声，而是去噪后潜在图像的表示。

在去噪循环中，我们需要从潜在空间解码图像。


```python
# 将潜在变量缩放回去。
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample
```

最后，将解码的图像转换为像素值，并显示它们。

```python
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]
```

![png](assets/98_stable_diffusion/stable_diffusion_k_lms.png)

我们已经从使用 🤗 Hugging Face Diffusers 的 Stable Diffusion 基础应用，逐步深入到了更高级的用法，并尝试介绍现代扩散系统的各个组成部分。如果你对这个主题感兴趣并想了解更多内容，我们推荐以下资源：

- 我们的 [Colab notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) 提供了有关 Stable Diffusion 的实践练习。
- [Diffusers 入门指南](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) 的 notebook，概述了扩散系统的基本知识。
- [Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) 博客文章。
- 我们的 [GitHub 代码](https://github.com/huggingface/diffusers)，如果你觉得 `diffusers` 对你有帮助，我们会很高兴收到你的 ⭐ ！

### Citation:
```
@article{patil2022stable,
  author = {Patil, Suraj and Cuenca, Pedro and Lambert, Nathan and von Platen, Patrick},
  title = {Stable Diffusion with 🧨 Diffusers},
  journal = {Hugging Face Blog},
  year = {2022},
  note = {[https://huggingface.co/blog/rlhf](https://huggingface.co/blog/stable_diffusion)},
}
```
