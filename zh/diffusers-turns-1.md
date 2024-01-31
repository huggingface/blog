---
title: 🤗 Diffusers 一岁啦 !
thumbnail: /blog/assets/diffusers-turns-1/diffusers-turns-1.png
authors:
- user: stevhliu
- user: sayakpaul
- user: pcuenq
translators:
- user: vermillion
- user: zhongdongy
  proofreader: true
---

# 🤗 Diffusers 一岁啦 !


十分高兴 🤗 Diffusers 迎来它的一岁生日！这是令人激动的一年，感谢社区和开源贡献者，我们对我们的工作感到十分骄傲和自豪。去年，文本到图像的模型，如 DALL-E 2, Imagen, 和 Stable Diffusion 以其从文本生成逼真的图像的能力，吸引了全世界的关注，也带动了对生成式 AI 的大量兴趣和开发工作。但是这些强大的工作不易获取。

在 Hugging Face, 我们的使命是一起通过相互合作和帮助，构建一个开放和有道德的 AI 未来，让机器学习民主化。我们的使命促使我们创造了 🤗 Diffusers 库，让 _每个人_ 能实验，研究，或者尝试文本到图像的生成模型。这便是我们设计这个模块化的库的初衷，你可以个性化扩散模型的某个部分，或者仅仅是开箱即用。

作为 🤗 Diffusers 的第一个版本，下面是在社区的帮助下，我们加入的最值得一提的特性。我们对作社区的一员，提高功能性，推动扩散模型不局限于文本到图像的生成，感到骄傲和感激。

**目录**

- [提高逼真性](#提高逼真性)
- [视频生成](#)
- [文本到 3D 模型生成]()
- [图像编辑]()
- [加速扩散模型]()
- [种族偏见和安全性]()
- [对 LoRA 的支持]()
- [基于 Torch 2.0 的优化]()
- [社区贡献]()
- [基于 🤗 Diffusers 的产品]()
- [展望]()

## 提高逼真性

众所周知，生成模型能生成逼真的图像，但如果你凑近看，绝对能发现某些瑕疵，比如多余的手指。今年，DeepFloyd IF 和 Stability AI SDXL 模型给出了让生成图像更逼真的方法。

[DeepFloyd IF](https://stability.ai/blog/deepfloyd-if-text-to-image-model) - 一个分步生成图片的模块化扩散模型 (比如，一个图片被三倍地上采样以提高分辨率)，不像 Stable Diffusion，IF 模型直接在像素层次上操作，并采用一个大语言模型来编码文本。

[Stable Diffusion XL (SDXL)](https://stability.ai/blog/sdxl-09-stable-diffusion) - Stability AI 的最前沿的 Stable Diffusion 模型，和之前的 Stable Diffusion 2 相比，参数量显著地增加了。它能生成超真实的图片，先用一个基础模型让图像很接近输入提示词，然后用一个改善模型专门提高细节和高频率的内容。

现在就去查阅 DeepFloyd IF 的 [文档](https://huggingface.co/docs/diffusers/v0.18.2/en/api/pipelines/if#texttoimage-generation) 和 SDXL 的 [文档](https://huggingface.co/docs/diffusers/v0.18.2/en/api/pipelines/stable_diffusion/stable_diffusion_xl)，然后生成你自己的图片吧！

## 视频生成

文本到图像很酷，但文本到视频更酷！我们现在能支持两种文本到视频的方法: [VideoFusion](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video) 和 [Text2Video-Zero](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video_zero)。

如果你对文本到图像的流程熟悉，那么文本到视频也一样:

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=24).frames
video_path = export_to_video(video_frames)
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/darthvader_cerpense.gif" alt="Generated video of Darth Vader surfing."/>
</div>

我们期待文生视频能在 🤗 Diffusers 的第二年迎来革命，也十分激动能看到社区在此之上的工作，进一步推进视频生成领域的进步！

## 文本到 3D

除了文本到视频，我们也提供了文本到 3D 的生成模型，多亏了 OpenAI 的 [Shap-E](https://hf.co/papers/2305.02463) 模型。Shap-E 在大量 3D 和文本的数据对上以编码的形式训练，在编码器的输出层条件化了一个扩散模型。你用它可以为游戏，内部设计和建筑生成 3D 资产。

现在就尝试 [`ShapEPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/shap_e#diffusers.ShapEPipeline) 和 [`ShapEImg2ImgPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/shap_e#diffusers.ShapEImg2ImgPipeline) 吧。

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/cake_out.gif" alt="3D render of a birthday cupcake generated using SHAP-E."/>
</div>

## 图像编辑

图像编辑是在时尚，材料设计和摄影领域最实用的功能之一。而图片编辑的可能性被扩散模型进一步增加。

在 🤗 Diffusers 中，我们提供了许多 [流水线](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlling_generation) 用来做图像编辑。有些图像编辑流水线能根据你的提示词从心所欲地修改图像，从图片中移除某个概念，甚至有流水线综合了很多创造高质量图片 (如全景图) 的生成方法。用 🤗 Diffusers，你现在就可以体验未来的图片编辑技术！

## 更快的扩散模型

众所周知，扩散模型以其迭代的过程而耗时。利用 OpenAI 的 [Consistency Models](https://huggingface.co/papers/2303.01469)，图像生成流程的速度有显著提高。生成单张 256x256 分辨率的图片，现在在一张 CPU 上只要 3/4 秒！你可以在 🤗 Diffusers 上尝试 [`ConsistencyModelPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/consistency_models)。

在更快的扩散模型之外，我们也提供许多面向更快推理的技术，比如 [PyTorch 2.0 的 `scaled_dot_product_attention()` (SDPA) 和 `torch.compile()`](https://pytorch.org/blog/accelerated-diffusers-pt-20), sliced attention, feed-forward chunking, VAE tiling, CPU and model offloading, 以及更多。这些优化节约内存，加快生成，允许你能在客户端 GPU 上运行。当你用 🤗 Diffusers 部署一个模型，所有的优化都即刻支持！

除此外，我们也支持具体的硬件格式如 ONNX，Pytorch 中 Apple 芯片的 `mps` 设备，Core ML 以及其他的。

欲了解更多关于 🤗 Diffusers 的优化，请查看 [文档](https://huggingface.co/docs/diffusers/optimization/opt_overview)！

## 道德和安全

生成模型很酷，但是它们也很容易生成有害的和 NSFW 内容，为了帮助用户负责和有道德地使用这些模型，我们添加了 [`safety_checker`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) 模块来标记生成内容中不合适的。模型的创造者可以决定是加入留该模块。

另外，生成模型也能生成误导性的信息，今年早些时候，[Balenciaga Pope](https://www.theverge.com/2023/3/27/23657927/ai-pope-image-fake-midjourney-computer-generated-aesthetic)
以画面真实如病毒般传播，虽然是虚假的。这呼吁了我们区分生成的和真实的内容的重要性。这便是我们对 SDXL 模型的生成内容添加一个不可见水印的原因，以帮助用户更好地辨别。

这些特性的开发都是由我们的 [ethical charter](https://huggingface.co/docs/diffusers/main/en/conceptual/ethical_guidelines) 主持，你能在我们的文档中看到。

## 对 LoRA 的支持

对扩散模型的微调是昂贵，且超出客户端 GPU 能力的。我们添加了低秩适应 (Low-Rank Adaptation, [LoRA](https://huggingface.co/papers/2106.09685)，是一种参数高效的微调策略) 技术来填补此空缺，你可以更快速地以更少内存地微调扩散模型。最终的模型参数和原模型相比也十分轻量，所以你可以容易地分享你的个性化模型。欲了解更多，请参阅我们的 [文档](https://huggingface.co/docs/diffusers/main/en/training/lora)，其展示了如何用 LoRA 在 Stable Diffusion 上进行微调。

在 LoRA 之外，我们对个性化的生成也提供了其他的 [训练技术](https://huggingface.co/docs/diffusers/main/en/training/overview)，包括 DreamBooth, textual inversion, custom diffusion 以及更多！

## 面向 Torch 2.0 的优化

PyTorch 2.0 [引入了支持](https://pytorch.org/get-started/pytorch-2.0/#pytorch-2x-faster-more-pythonic-and-as-dynamic-as-ever) `torch.compile()` 和 `scaled_dot_product_attention()` (
一种注意力机制的更高效实现)。🤗 Diffusers 提供了对这些特性的 [支持](https://huggingface.co/docs/diffusers/optimization/torch2.0)，带来了速度的大量提升，有时甚至能快两倍多。

在视觉内容 (图片，视频，三维资产等) 外，我们也提供了音频支持！请查阅 [文档](https://huggingface.co/docs/diffusers/using-diffusers/audio) 以了解更多。

## 社区的亮点

过去一年中，最令人愉悦的经历，便是看到社区如何把 🤗 Diffusers 融入到他们的项目中。从使用 LoRA 到更快的文本到图像的生成模型，到实现最前沿的绘画工具，这里是几个我们最喜欢的项目:

<div class="mx-auto max-w-screen-xl py-8">
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">
      我们构建 Core ML Stable Diffusion，让它对开发者而言，在他们的 iOS, iPadOS 和 macOS 应用中，以 Apple Silicon 最高的效率，更容易添加最前沿的生成式 AI 能力。我们在 🤗 Diffusers 的基础上构建，而不是从头开始，因为不论想法新旧，🤗 Diffusers 能持续快速地跟进领域的发展，并且做到位的改进。</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/10639145?s=200&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Atila Orhon</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">🤗 Diffusers 对我深入了解 Stable Diffusion 模型而言十分友好。🤗 Diffusers 的实现最独特之处是，它不是来自科研阶段的代码，而主要由速度驱动。科研时的代码总是写的很糟糕，难于理解 (缺少规范书写，断言，设计和记号不一致)，在 🤗 Diffusers 上在数小时内实现我的想法，犹如呼吸一般简单。没有它，我估计会花更多的时间才开始 hack 代码。规范的文档和例子也十分有帮助。
      </p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/35953539?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Simo</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">
      BentoML 是一个统一的框架，对构建，装载，和量化产品级 AI 应用，涉及传统的机器学习，预训练 AI 模型，生成式和大语言模型。所有的 Hugging Face 的 Diffusers 模型和管线都能无缝地整合进 BentoML 的应用中，让模型的运行能在最合适的硬件并按需实现自主规模缩放。
      </p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/49176046?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">BentoML</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Invoke AI 是一个开源的生成式 AI 工具，用来助力专业创作，从游戏设计和摄像到建筑和产品设计。Invoke 最近开放了 invoke.ai，允许用户以最新的开源研究成果助力，在任意电脑上生成资产。
      </p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/113954515?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">InvokeAI</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">TaskMatrix 连接大语言模型和一系列视觉模型，助力聊天同时发送送和接受图片。</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/6154722?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Chenfei Wu</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Lama Cleaner 是一个强大的图像绘画工具，用 Stable Diffusion 的技术移除不想要的物体、瑕疵、或者人物。它也可以擦除和替换图像中的任意东西。</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://github.com/Sanster/lama-cleaner/raw/main/assets/logo.png" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Qing</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Grounded-SAM 结合了一个强大的零样本检测器 Grounding-DINO 和 Segment-Anything-Model (SAM) 来构建一个强大的流水线，以用文本输入检测和分割任意物体。当和 🤗 Diffusers 绘画模型结合起来时，Grounded-SAM 能做高可控的图像编辑人物，包括替换特定的物体，绘画背景等等。
      </p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/113572103?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Tianhe Ren</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Stable-Dreamfusion 结合 🤗 Diffusers 中方便的 2D 扩散模型来复现最近文本到 3D 和图像到 3D 的方法。</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/25863658?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">kiui</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">MMagic (Multimodal Advanced, Generative, and Intelligent Creation) 是一个先进并且易于理解的生成式 AI 工具箱，提供最前沿的 AI 模型 (比如 🤗 Diffusers 的扩散模型和 GAN 模型)，用来合成，编辑和改善图像和视频。在 MMagic 中，用户可以用丰富的部件来个性化他们的模型，就像玩乐高一样，并且很容易地管理训练的过程。
      </p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/10245193?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">mmagic</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Tune-A-Video，由 Jay Zhangjie Wu 和他来自 Show Lab 的团队开发，是第一个用单个文本-视频对实现微调预训练文本到图像的扩散模型，它能够在改变视频内容的同时保持内容的运动状态。</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/101181824?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Jay Zhangjie Wu</p>
      </div>
    </div>
  </div>
</div>

同时我们也和 Google Cloud 合作 (他们慷慨地提供了计算资源) 来提供技术性的指导和监督，以帮助社区用 TPU 来训练扩散模型 (请参考 [比赛](http://opensource.googleblog.com/2023/06/controlling-stable-diffusion-with-jax-diffusers-and-cloud-tpus.html) )。有很多很酷的模型，比如这个 [demo](https://huggingface.co/spaces/mfidabel/controlnet-segment-anything) 结合了 ControlNet 和 Segment Anything。

<div class="flex justify-center">
  <img src="https://github.com/mfidabel/JAX_SPRINT_2023/blob/8632f0fde7388d7a4fc57225c96ef3b8411b3648/EX_1.gif?raw=true" alt="ControlNet and SegmentAnything demo of a hot air balloon in various styles">
</div>

最后，我们十分高兴收到超过 300 个贡献者对我们的代码的改进，以保证我们能以最开放的形式合作。这是一些来自我们社区的贡献:

- [Model editing](https://github.com/huggingface/diffusers/pull/2721) by [@bahjat-kawar](https://github.com/bahjat-kawar), 一个修改模型隐式假设的流水线。
- [LDM3D](https://github.com/huggingface/diffusers/pull/3668) by [@estelleafl](https://github.com/estelleafl), 一个生成 3D 图片的扩散模型。
- [DPMSolver](https://github.com/huggingface/diffusers/pull/3314) by [@LuChengTHU](https://github.com/LuChengTHU), 显著地提高推理速度。
- [Custom Diffusion](https://github.com/huggingface/diffusers/pull/3031) by [@nupurkmr9](https://github.com/nupurkmr9), 一项用同一物体的少量图片生成个性化图片的技术。

除此之外，由衷地感谢如下贡献者，为我们实现了 Diffusers 中最有用的功能。

- [@takuma104](https://github.com/huggingface/diffusers/commits?author=takuma104)
- [@nipunjindal](https://github.com/huggingface/diffusers/commits?author=nipunjindal)
- [@isamu-isozaki](https://github.com/huggingface/diffusers/commits?author=isamu-isozaki)
- [@piEsposito](https://github.com/huggingface/diffusers/commits?author=piEsposito)
- [@Birch-san](https://github.com/huggingface/diffusers/commits?author=Birch-san)
- [@LuChengTHU](https://github.com/huggingface/diffusers/commits?author=LuChengTHU)
- [@duongna21](https://github.com/huggingface/diffusers/commits?author=duongna21)
- [@clarencechen](https://github.com/huggingface/diffusers/commits?author=clarencechen)
- [@dg845](https://github.com/huggingface/diffusers/commits?author=dg845)
- [@Abhinay1997](https://github.com/huggingface/diffusers/commits?author=Abhinay1997)
- [@camenduru](https://github.com/huggingface/diffusers/commits?author=camenduru)
- [@ayushtues](https://github.com/huggingface/diffusers/commits?author=ayushtues)

## 用 🤗 Diffusers 做产品

在过去一年中，我们看到了许多公司在 🤗 Diffusers 的基础上构建他们的产品。这是几个吸引到我们关注的产品:

- [PlaiDay](http://plailabs.com/): “PlaiDay 是一个生成式 AI 产品，人们可以合作，创造和连接。我们的平台解锁了人脑的无限创造力，为表达提供了一个安全，有趣的画板。”
- [Previs One](https://previs.framer.wiki/): “Previs One 是一个面向电影故事板和预可视化的扩散模型 - 它能如同导演般理解电影和电视的合成规则。”
- [Zust.AI](https://zust.ai/): “我们利用生成式 AI 来为品牌和市场营销创造工作室级别的图像产品。”
- [Dashtoon](https://dashtoon.com/): “Dashtoon 在构建一个创造和消耗视觉内容的平台。我们有多个流水线配置多个 LoRA，多个 Control-Net，甚至多个 Diffusers 模型。Diffusers 已经让产品设计师和 ML 设计师之间的鸿沟十分小了，这让 dashtoon 能更加重视用户的价值。”
- [Virtual Staging AI](https://www.virtualstagingai.app/): “用生成模型做家具，来填满空荡荡的房间吧。”
- [Hexo.AI](https://www.hexo.ai/): “Hexo AI 帮助品牌在市场上得到更高的 ROI，通过个性化的市场规模。Hexo 在构建一个专门的生成引擎，通过引入用户数据，生成全部个性化的创造。”

如果你在用 🤗 Diffusers 构建产品，我们十分乐意讨论如何让我们的库更加好！欢迎通过 [patrick@hf.co](mailto:patrick@hf.co) 或者 [patrick@hf.co](sayak@hf.co) 来联系我们。

## 展望

作为我们的一周年庆，我们对社区和开源贡献者十分感激，他们帮我们在如此短的时间如此多的事情。我们十分开心，将在今年秋天的 ICCV 2023 展示一个 🤗 Diffusers 的 demo - 如果你参加，请过来看我们的表演！我们将持续发展和提高我们的库，让它对每个人而言更加容易使用。我们也十分激动能看到社区用我们的工具和资源做的下一步创造。感谢你们作为我们目前旅途中的一员，我们期待继续一起为机器学习的民主化做贡献！🥳
❤️ Diffusers 团队

---

**致谢**: 感谢 [Omar Sanseviero](https://huggingface.co/osanseviero), [Patrick von Platen](https://huggingface.co/patrickvonplaten), [Giada Pistilli](https://huggingface.co/giadap) 的审核，以及 [Chunte Lee](https://huggingface.co/Chunte) 设计的 thumbnail。