---
title: "在 SDXL 上用 T2I-Adapter 实现高效可控的文生图"
thumbnail: /blog/assets/t2i-sdxl-adapters/thumbnail.png
authors:
- user: Adapter
  guest: true
- user: valhalla
- user: sayakpaul
- user: Xintao
  guest: true
- user: hysts
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 在 SDXL 上用 T2I-Adapter 实现高效可控的文生图


<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/hf_tencent.png" height=180/>
</p>

[T2I-Adapter](https://huggingface.co/papers/2302.08453) 是一种高效的即插即用模型，其能对冻结的预训练大型文生图模型提供额外引导。T2I-Adapter 将 T2I 模型中的内部知识与外部控制信号结合起来。我们可以根据不同的情况训练各种适配器，实现丰富的控制和编辑效果。

同期的 [ControlNet](https://hf.co/papers/2302.05543) 也有类似的功能且已有广泛的应用。然而，其运行所需的 **计算成本比较高**。这是因为其反向扩散过程的每个去噪步都需要运行 ControlNet 和 UNet。另外，对 ControlNet 而言，复制 UNet 编码器作为控制模型的一部分对效果非常重要，这也导致了控制模型参数量的进一步增大。因此，ControlNet 的模型大小成了生成速度的瓶颈 (模型越大，生成得越慢)。

在这方面，T2I-Adapters 相较 ControlNets 而言颇有优势。T2I-Adapter 的尺寸较小，而且，与 ControlNet 不同，T2I-Adapter 可以在整个去噪过程中仅运行一次。 

| **模型** | **参数量** | **所需存储空间（fp16）** |
| --- | --- | --- |
| [ControlNet-SDXL](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) | 1251 M | 2.5 GB |
| [ControlLoRA](https://huggingface.co/stabilityai/control-lora) (rank = 128) | 197.78 M (参数量减少 84.19%)  | 396 MB (所需空间减少 84.53%) |
| [T2I-Adapter-SDXL](https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0) | 79 M (**_参数量减少 93.69%_**) | 158 MB (**_所需空间减少 94%_**) |

在过去的几周里，Diffusers 团队和 T2I-Adapter 作者紧密合作，在 [`diffusers`](https://github.com/huggingface/diffusers) 库上为 [Stable Diffusion XL (SDXL)](https://huggingface.co/papers/2307.01952) 增加 T2I-Adapter 的支持。本文，我们将分享我们在从头开始训练基于 SDXL 的 T2I-Adapter 过程中的发现、漂亮的结果，以及各种条件 (草图、canny、线稿图、深度图以及 OpenPose 骨骼图) 下的 T2I-Adapter checkpoint！

![结果合辑](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/results_collage.png)

与之前版本的 T2I-Adapter (SD-1.4/1.5) 相比，[T2I-Adapter-SDXL](https://github.com/TencentARC/T2I-Adapter) 还是原来的配方，不一样之处在于，用一个 79M 的适配器去驱动 2.6B 的大模型 SDXL！ T2I-Adapter-SDXL 在继承 SDXL 的高品质生成能力的同时，保留了强大的控制能力！

## 用 `diffusers` 训练 T2I-Adapter-SDXL

我们基于 `diffusers` 提供的 [这个官方示例](https://github.com/huggingface/diffusers/blob/main/examples/t2i_adapter/README_sdxl.md) 构建了我们的训练脚本。

本文中提到的大多数 T2I-Adapter 模型都是在 LAION-Aesthetics V2 的 3M 高分辨率 `图文对` 上训练的，配置如下:

- 训练步数: 20000-35000
- batch size: 采用数据并行，单 GPU batch size 为 16，总 batch size 为 128
- 学习率: 1e-5 的恒定学习率
- 混合精度: fp16

我们鼓励社区使用我们的脚本来训练自己的强大的 T2I-Adapter，并对速度、内存和生成的图像质量进行折衷以获得竞争优势。

## 在 `diffusers` 中使用 T2I-Adapter-SDXL

这里以线稿图为控制条件来演示 [T2I-Adapter-SDXL](https://github.com/TencentARC/T2I-Adapter/tree/XL) 的使用。首先，安装所需的依赖项:

```bash
pip install -U git+https://github.com/huggingface/diffusers.git
pip install -U controlnet_aux==0.0.7 # for conditioning models and detectors
pip install transformers accelerate
```

T2I-Adapter-SDXL 的生成过程主要包含以下两个步骤:

1. 首先将条件图像转换为符合要求的 _控制图像_ 格式。
2. 然后将 _控制图像_ 和 _提示_ 传给 [`StableDiffusionXLAdapterPipeline`](https://github.com/huggingface/diffusers/blob/0ec7a02b6a609a31b442cdf18962d7238c5be25d/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L126)。

我们看一个使用 [Lineart Adapter](https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0) 的简单示例。我们首先初始化 SDXL 的 T2I-Adapter 流水线以及线稿检测器。

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

然后，加载图像并生成其线稿图:

```python
url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_lin.jpg"
image = load_image(url)
image = line_detector(image, detect_resolution=384, image_resolution=1024)
```

![龙的线稿图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/lineart_dragon.png)

然后生成:

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

![用线稿图生成出来的龙](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/lineart_generated_dragon.png)

理解下述两个重要的参数，可以帮助你调节控制程度。

1. `adapter_conditioning_scale`

    该参数调节控制图像对输入的影响程度。越大代表控制越强，反之亦然。

2. `adapter_conditioning_factor`

    该参数调节适配器需应用于生成过程总步数的前面多少步，取值范围在 0-1 之间 (默认值为 1)。 `adapter_conditioning_factor=1` 表示适配器需应用于所有步，而 `adapter_conditioning_factor=0.5` 则表示它仅应用于前 50% 步。

更多详情，请查看 [官方文档](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/adapter)。

## 试玩演示应用

你可以在 [这儿](https://huggingface.co/spaces/TencentARC/T2I-Adapter-SDXL) 或下述嵌入的游乐场中轻松试玩 T2I-Adapter-SDXL:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.43.1/gradio.js"></script>
<gradio-app src="https://tencentarc-t2i-adapter-sdxl.hf.space"></gradio-app>

你还可以试试 [Doodly](https://huggingface.co/spaces/TencentARC/T2I-Adapter-SDXL-Sketch)，它用的是草图版模型，可以在文本监督的配合下，把你的涂鸦变成逼真的图像:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.43.1/gradio.js"></script>
<gradio-app src="https://tencentarc-t2i-adapter-sdxl-sketch.hf.space"></gradio-app>

## 更多结果

下面，我们展示了使用不同控制图像作为条件获得的结果。除此以外，我们还分享了相应的预训练 checkpoint 的链接。如果想知道有关如何训练这些模型的更多详细信息及其示例用法，可以参考各自模型的模型卡。

### 使用线稿图引导图像生成

![线稿图的更多结果](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/lineart_guided.png)
_模型见 [`TencentARC/t2i-adapter-lineart-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0)_

### 使用草图引导图像生成

![草图的结果](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/sketch_guided.png)
_模型见 [`TencentARC/t2i-adapter-sketch-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-sketch-sdxl-1.0)_

### 使用 Canny 检测器检测出的边缘图引导图像生成

![Canny 边缘图的结果](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/canny_guided.png)
_模型见 [`TencentARC/t2i-adapter-canny-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0)_

### 使用深度图引导图像生成

![深度图的结果](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/depth_guided.png)
_模型分别见 [`TencentARC/t2i-adapter-depth-midas-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-depth-midas-sdxl-1.0) 及 [`TencentARC/t2i-adapter-depth-zoe-sdxl-1.0`](https://huggingface.co/TencentARC/t2i-adapter-depth-zoe-sdxl-1.0)_

### 使用 OpenPose 骨骼图引导图像生成

![OpenPose 骨骼图的结果](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/t2i-adapters-sdxl/pose_guided.png)
_模型见 [`TencentARC/t2i-adapter-openpose-sdxl-1.0`](https://hf.co/TencentARC/t2i-adapter-openpose-sdxl-1.0)_

---

_致谢: 非常感谢 [William Berman](https://twitter.com/williamLberman) 帮助我们训练模型并分享他的见解。_