---
title: "欢迎 Stable Diffusion 3 加入 🧨 Diffusers"
thumbnail: /blog/assets/sd3/thumbnail.png
authors:
- user: dn6
- user: YiYiXu
- user: sayakpaul
- user: OzzyGT
- user: kashif
- user: multimodalart
translators:
- user: hugging-hoi2022
- user: zhongdongy
  proofreader: true
---

# 欢迎 Stable Diffusion 3 加入 🧨 Diffusers

作为 Stability AI 的 Stable Diffusion 家族最新的模型，[Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3-research-paper) (SD3) 现已登陆 Hugging Face Hub，并且可用在 🧨 Diffusers 中使用了。

当前放出的模型版本是 Stable Diffusion 3 Medium，有二十亿 (2B) 的参数量。

针对当前发布版本，我们提供了:

1. Hub 上可供下载的模型
2. Diffusers 的代码集成
3. SD3 的 Dreambooth 和 LoRA 训练脚本

## 目录

- [SD3 新特性](#SD3 新特性)
- [在 Diffusers 中使用 SD3](#在 Diffusers 中使用 SD3)
- [对 SD3 进行内存优化以适配各种硬件](#对 SD3 进行内存优化)
- [性能优化与推理加速](#SD3 性能优化)
- [SD3 微调和 LoRA 创建](#使用 DreamBooth 和 LoRA 进行微调)

## SD3 新特性

### 模型

作为一个隐变量扩散模型，SD3 包含了三个不同的文本编码器 ([CLIP L/14](https://huggingface.co/openai/clip-vit-large-patch14)、[OpenCLIP bigG/14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 和 [T5-v1.1-XXL](https://huggingface.co/google/t5-v1_1-xxl)) 、一个新提出的多模态 Diffusion Transformer (MMDiT) 模型，以及一个 16 通道的 AutoEncoder 模型 (与 [Stable Diffusion XL](https://arxiv.org/abs/2307.01952) 中的类似)。

SD3 以序列 Embedding 的形式处理文本输入和视觉隐空间特征。位置编码 (Positional Encoding) 是施加在隐空间特征的 2x2 patch 上的，随后被展开成 patch 的 Enbedding 序列。这一序列和文本的特征序列一起，被送入 MMDiT 的各个模块中去。两种特征序列被转化成相同特征维度，拼接在一起，然后送入一系列注意力机制模块和多层感知机 (MLP) 里。

为应对两种模态间的差异，MMDiT 模块使用两组不同的权重去转换文本和图像序列的特征维度。两个序列之后会在注意力操作之前被合并在一起。这种设计使得两种表征能在自己的特征空间里工作，同时也使得它们之间可以通过注意力机制 [1] 从对方的特征中提取有用的信息。这种文本和图像间双向的信息流动有别于以前的文生图模型，后者的文本信息是通过 cross-attention 送入模型的，且不同层输入的文本特征均是文本编码器的输出，不随深度的变化而改变。

此外，SD3 还在时间步 (timestep) 这一条件信息上加入了汇合过的文本特征，这些文本特征来自使用的两个 CLIP 模型。这些汇合过的文本特征被拼接在一起，然后加到时间步的 Embedding 上，再送入每个 MMDiT 模块。

### 使用 Rectified Flow Matching 训练

除了结构上的创新，SD3 也使用了 [conditional flow-matching](https://arxiv.org/html/2403.03206v1#S2) 作为训练目标函数来训练模型。这一方法中，前向加噪过程被定义为一个 [rectified flow](https://arxiv.org/html/2403.03206v1#S3)，以一条直线连接数据分布和噪声分布。

采样过程也变得更简单了，当采样步数减少的时候，模型性能也很稳定。为此，我们也引入了新的 scheduler ( `FlowMatchEulerDiscreteScheduler` )，集成了 rectified flow-matching 的运算公式以及欧拉方法 (Euler Method) 的采样步骤。同时还提出了一个与生成分辨率相关的 `shift` 参数。对于高分辨率，增大 `shift` 的值可以更好地处理 noise scaling。针对 2B 模型，我们建议设置 `shift=3.0` 。

如想快速尝试 SD3，可以使用下面的一个基于 Gradio 的应用:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.36.1/gradio.js"> </script>
<gradio-app theme_mode="light" space="stabilityai/stable-diffusion-3-medium"></gradio-app>

## 在 Diffusers 中使用 SD3

如想在 diffusers 中使用 SD3，首先请确保安装的 diffusers 是最新版本:

```python
pip install --upgrade diffusers
```

使用模型前，你需要先到 [Stable Diffusion 3 Medium 在 Hugging Face 的页面](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)，填写表格并同意相关内容。一切就绪后，你需要登录你的 huggingface 账号:

```bash
huggingface-cli login
```

下面程序将会下载 SD3 的 2B 参数模型，并使用 `fp16` 精度。Stability AI 原本发布的模型精度就是 `fp16` ，这也是推荐的模型推理精度。

### 文生图

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
	"A cat holding a sign that says hello world",
	negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image
```

![hello_world_cat](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/sd3/hello_world_cat.png)

### 图生图

```python
import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipe(prompt, image=init_image).images[0]
image
```

![wizard_cat](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/sd3/wizard_cat.png)

相关的 SD3 文档可在 [这里](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3) 查看。

## 对 SD3 进行内存优化

SD3 使用了三个文本编码器，其中一个是 [T5-XXL model](https://huggingface.co/google/t5-v1_1-xxl)，是一个很大的模型。这使得在显存小于 24GB 的 GPU 上跑模型非常困难，即使使用的是 `fp16` 精度。

对此，diffusers 集成了一些内存优化手段，来让 SD3 能在更多的 GPU 上跑起来。

### 使用 Model Offloading 推理

Diffusers 上一个最常用的内存优化手段就是 model offloading。它使得你可以在推理时，把一些当前不需要的模型组件卸载到 CPU 上，以此节省 GPU 显存。但这会引入少量的推理时长增长。在推理时，model offloading 只会将模型当前需要参与计算的部分放在 GPU 上，而把剩余部分放在 CPU 上。

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

prompt = "smiling cartoon dog sits at a table, coffee mug on hand, as a room goes up in flames. “This is fine,” the dog assures himself."
image = pipe(prompt).images[0]
```

### 不使用 T5 模型进行推理

[推理时移除掉 4.7B 参数量的 T5-XXL 文本编码器](https://arxiv.org/html/2403.03206v1#S5.F9) 可以很大程度地减少内存需求，带来的性能损失却很小。

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", text_encoder_3=None, tokenizer_3=None, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "smiling cartoon dog sits at a table, coffee mug on hand, as a room goes up in flames. “This is fine,” the dog assures himself."
image = pipe("").images[0]
```

## 使用量化版的 T5-XXL 模型

使用 `bitsandbytes` 这个库，你也可以加载 8 比特量化版的 T5-XXL 模型，进一步减少显存需求。

```python
import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig

# Make sure you have `bitsandbytes` installed.
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
text_encoder = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_3",
    quantization_config=quantization_config,
)
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    text_encoder_3=text_encoder,
    device_map="balanced",
    torch_dtype=torch.float16
)
```

_完整代码在 [这里](https://gist.github.com/sayakpaul/82acb5976509851f2db1a83456e504f1)。_

### 显存优化小结

所有的基准测试都用了 2B 参数量的 SD3 模型，测试在一个 A100-80G 上进行，使用 `fp16` 精度推理，PyTorch 版本为 2.3。

我们对每个推理调用跑十次，记录平均峰值显存用量和 20 步采样的平均时长。

## SD3 性能优化

为加速推理，我们可以使用 `torch.compile()` 来获取优化过的 `vae` 和 `transformer` 部分的计算图。

```python
import torch
from diffusers import StableDiffusion3Pipeline

torch.set_float32_matmul_precision("high")

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
).to("cuda")
pipe.set_progress_bar_config(disable=True)

pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# Warm Up
prompt = "a photo of a cat holding a sign that says hello world",
for _ in range(3):
 _ = pipe(prompt=prompt, generator=torch.manual_seed(1))

# Run Inference
image = pipe(prompt=prompt, generator=torch.manual_seed(1)).images[0]
image.save("sd3_hello_world.png")
```

_完整代码可参考 [这里](https://gist.github.com/sayakpaul/508d89d7aad4f454900813da5d42ca97)。_

我们测量了使用过 `torch.compile()` 的 SD3 的推理速度 (在 A100-80G 上，使用 `fp16` 推理，PyTorch 版本为 2.3)。我们针对每个生成任务跑 10 遍，每次推理使用 20 步采样。平均推理耗时是 **0.585 秒**， _这比 eager execution 模式下快了四倍_ 。

## 使用 DreamBooth 和 LoRA 进行微调

最后，我们还提供了使用 [LoRA](https://huggingface.co/blog/lora) 的 [DreamBooth](https://dreambooth.github.io/) 代码，用于微调 SD3。这一程序不仅能微调模型，还能作为一个参考，如果你想使用 rectified flow 来训练模型。当然，热门的 rectified flow 实现代码还有 [minRF](https://github.com/cloneofsimo/minRF/)。

如果需要使用该程序，首先需要确保各项设置都已完成，同时准备好一个数据集 (比如 [这个](https://huggingface.co/datasets/diffusers/dog-example))。你需要安装 `peft` 和 `bitsandbytes` ，然后再开始运行训练程序:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth-sd3-lora"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --instance_data_dir=${INSTANCE_DIR} \
  --output_dir=/raid/.cache/${OUTPUT_DIR} \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --weighting_scheme="logit_normal" \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

## 声明

感谢 Stability AI 团队开发并开源了 Stable Diffusion 3 并让我们提早体验，也感谢 [Linoy](https://huggingface.co/linoyts) 对撰写此文的帮助。