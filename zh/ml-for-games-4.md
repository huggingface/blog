---
title: "制作 2D 素材｜基于 AI 5 天创建一个农场游戏，第 4 天"
thumbnail: /blog/assets/124_ml-for-games/thumbnail4.png
authors:
- user: dylanebert
translators:
- user: SuSung-boy
- user: zhongdongy
  proofreader: true
---

# 制作 2D 素材｜基于 AI 5 天创建一个农场游戏，第 4 天


**欢迎使用 AI 进行游戏开发！** 在本系列中，我们将使用 AI 工具在 5 天内创建一个功能完备的农场游戏。到本系列结束时，您将了解到如何将多种 AI 工具整合到游戏开发流程中。本系列文章将向您展示如何将 AI 工具用于:

1. 美术风格
2. 游戏设计
3. 3D 素材
4. 2D 素材
5. 剧情

想快速观看视频的版本？你可以在 [这里](https://www.tiktok.com/@individualkex/video/7192994527312137518) 观看。不过如果你想要了解技术细节，请继续阅读吧！

**注意:** 本教程面向熟悉 Unity 开发和 C# 语言的读者。如果您不熟悉这些技术，请先查看 [Unity for Beginners](https://www.tiktok.com/@individualkex/video/7086863567412038954) 系列后再继续阅读。

## 第 4 天：2D 素材

本教程系列的 [第 3 部分](https://huggingface.co/blog/zh/ml-for-games-3) 讨论到现阶段 **文本-3D** 技术应用到游戏开发中并不可行。不过对于 2D 来说，情况就大相径庭了。

在这一部分中，我们将探讨如何使用 AI 制作 2D 素材。

### 前言

本部分教程将介绍如何将 Stable Diffusion 工具嵌入到传统 2D 素材制作流程中，来帮助从业者使用 AI 制作 2D 素材。此教程适用于具有一定图片编辑和 2D 游戏素材制作知识基础的读者，同时对游戏或者 AI 领域的初学者和资深从业者也会有所帮助。

必要条件：
- 图片编辑软件。可以根据您的使用习惯偏好选择，如 [Photoshop](https://www.adobe.com/products/photoshop.html) 或 [GIMP](https://www.gimp.org/) (免费)。
- Stable Diffusion。可以参照 [第 1 部分](https://huggingface.co/blog/ml-for-games-1#setting-up-stable-diffusion) 的说明设置 Stable Diffusion。

### Image2Image

诸如 [Diffusion models](https://en.wikipedia.org/wiki/Diffusion_model) 之类的扩散模型生成图片的过程是从初始噪声开始，通过不断去噪来重建图片，同时在去噪过程中可以添加额外的指导条件来引导生成图片的某种特性，这个条件可以是文本、轮廓、位置等。基于扩散模型的 Image2Image 生成图片的过程也一样，但并非从初始噪声开始，而是输入真实图片，这样最终生成的图片将会与输入图片有一定的相似性。

Image2Image 中的一个比较重要的参数是 **去噪强度** (denoising strength)，它可以控制生成图片与输入图片的差异程度。去噪强度为 0 会生成与输入图片完全一致的图片，去噪强度为 1 则截然不同。去噪强度也可以理解为 **创造性**。例如：给定一张圆形图案的输入图片，添加文本提示语 “月亮”，对去噪强度设置不同的参数值，Image2Image 可以生成不同创造性的图片，示意图如下。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/moons.png" alt="Denoising Strength 示例">
</div>

基于 Stable Diffusion 的 Image2Image 方法并非代替了传统美术作品绘图流程，而是作为一种工具辅助使用。具体来说，您可以先手动绘制图片，然后将其输入给 Image2Image，调整相关参数后得到生成图片，然后继续将生成的图片输入给 Image2Image 进行多次迭代，直到生成一张满意的图片。以本系列的农场游戏为例，我会在接下来的部分说明具体细节。

### 示例：玉米

在这一小节中，我会介绍使用 Image2Image 为农场游戏的农作物玉米生成图标的完整流程。首先需要确定整体构图，我简单勾勒了一张非常粗糙的玉米图标草图。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/corn1.png" alt="Corn 1">
</div>

接下来，我输入以下提示语：

> corn, james gilleard, atey ghailan, pixar concept artists, stardew valley, animal crossing
>
> 注：corn：玉米；james gilleard：未来主义插画艺术家；atey ghailan：现拳头游戏概念艺术家；pixar concept artists：皮克斯动画概念艺术；stardew valley：星露谷物语，一款像素风农场游戏；animal crossing：动物之森，任天堂游戏

同时设置去噪强度为 0.8，确保扩散模型生成的图片在保持原始构图的同时兼顾更多的创造性。从多次随机生成的图片中，我挑选了一张喜欢的，如下所示。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/corn2.png" alt="Corn 2">
</div>

生成的图片不需要很完美，因为通常会多次迭代来不断修复不完美的部分。对于上面挑选的图片，我觉得整体风格很不错，不过玉米叶部分稍微有些复杂，所以我使用 PhotoShop 做了一些修改。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/corn3.png" alt="Corn 3">
</div>

请注意，这里我仅在 PhotoShop 中用笔刷非常粗略地涂掉了要改的部分，然后把它输入到 Image2Image 中，让 Stable Diffusion 自行填充这部分的细节。由于这次输入图片的大部分信息需要被保留下来，因此我设置去噪强度为 0.6，得到了一张 *还不错* 的图片。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/corn4.png" alt="Corn 4">
</div>

接着我在 PhotoShop 中又做了一些修改：简化了底部的线条以及去除了顶部的新芽，再一次输入 Stable Diffusion 迭代，并且删除了背景，最终的玉米图标如下图所示。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/corn5.png" alt="Corn 5">
</div>

瞧！不到 10 分钟，一个玉米图标游戏素材就制作完成了！其实您可以花更多时间来打磨一个更好的作品。如想了解如何制作更加精致的游戏素材，可以前往观看详细演示视频。

### 示例：镰刀

很多时候，您可能需要对扩散模型进行 负面引导 才能生成期望的图片。下图毫无疑问可以用作镰刀图标，但这些简单的图片却需要大量迭代次数才能生成。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/scythe.png" alt="Scythe">
</div>

原因可能是这样：扩散模型使用的训练图片基本都是网络上的，而网络上关于镰刀的图片大部分是 **武器**，只有小部分是 *农具*，这就导致模型生成的镰刀图片会偏离 农具。一种解决方法是改善提示语：以增加 负面提示语 的方式引导模型避开相应的结果。上述示例中，除了输入 **镰刀，农具** 之外，在负面提示语一栏输入 **武器** 就能奏效。当然，也不只有这一种解决方法。

[Dreambooth](https://dreambooth.github.io/)、[textual inversion](https://textual-inversion.github.io/) 和 [LoRA](https://huggingface.co/blog/lora) 技术用于定制个人专属的扩散模型，可以使模型生成更加明确的图片。在 2D 生成领域，这些技术会越来越重要，不过具体技术细节不在本教程范围之内，这里就不展开了。

[layer.ai](https://layer.ai/) 和 [scenario.gg](https://www.scenario.gg/) 等是专门提供游戏素材生成的服务商，可以使游戏从业者在游戏开发过程中生成的游戏素材保持风格一致，他们的底层技术很可能就是 dreambooth 或 textual inversion。在新兴的开发游戏素材生成工具包赛道，是这些技术成为主流？还是会再出现其他技术？让我们拭目以待！

如果您对 Dreambooth 的工作流程细节感兴趣，可以查看 [博客文章](https://huggingface.co/blog/dreambooth) 阅读相关信息，也可以进入 Hugging Face 的 Dreambooth Training [Space](https://huggingface.co/spaces/multimodalart/dreambooth-training) 应用体验整个流程。

点击 [这里](https://huggingface.co/blog/zh/ml-for-games-5) 继续阅读第五部分，我们一起进入 **AI 设计游戏剧情**。
