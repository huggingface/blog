---
title: "AI 制作 3D 素材｜基于 AI 5 天创建一个农场游戏，第 3 天"
thumbnail: /blog/assets/124_ml-for-games/thumbnail3.png
authors:
- user: dylanebert
translators:
- user: SuSung-boy
- user: zhongdongy
  proofreader: true
---

# AI 制作 3D 素材｜基于 AI 5 天创建一个农场游戏，第 3 天


**欢迎使用 AI 进行游戏开发**！在本系列中，我们将使用 AI 工具在 5 天内创建一个功能完备的农场游戏。到本系列结束时，您将了解到如何将多种 AI 工具整合到游戏开发流程中。本文将向您展示如何将 AI 工具用于:

1. 美术风格
2. 游戏设计
3. 3D 素材
4. 2D 素材
5. 剧情

想快速观看视频的版本？你可以在 [这里](https://www.tiktok.com/@individualkex/video/7190364745495678254) 观看。不过如果你想要了解技术细节，请继续阅读吧！

**注意**： 本教程面向熟悉 Unity 开发和 C# 语言的读者。如果您不熟悉这些技术，请先查看 [Unity for Beginners](https://www.tiktok.com/@individualkex/video/7086863567412038954) 系列后再继续阅读。

## 第 3 天：3D 素材

本教程系列的 [第 2 部分](https://huggingface.co/blog/zh/ml-for-games-2) 介绍了 **使用 AI 进行游戏设计**。更具体地说，我们提问 ChatGPT 进行头脑风暴，进而设计农场游戏所需的功能组件。

在这一部分中，我们将探讨如何使用 AI 制作 3D 素材。先说结论：*不可行*。因为现阶段的文本-3D 技术水平还没有发展到可用于游戏开发的程度。不过 AI 领域在迅速变革，可能很快就有突破。如想了解 [文本-3D 现阶段进展](#文本-3D 现阶段进展)，[现阶段不可行的原因](#现阶段不可行的原因)，以及 [文本-3D 的未来发展](#文本-3D 的未来发展)，请继续往下阅读。

### 文本-3D 现阶段进展

我们在 [第 1 部分](https://huggingface.co/blog/zh/ml-for-games-1) 中介绍了使用 Stable Diffusion 帮助确立游戏美术风格，这类 文本-图像 的工具在游戏开发流程中表现非常震撼。同时游戏开发中也有 3D 建模需求，那么从文本生成 3D 模型的文本-3D 工具表现如何？下面总结了此领域的近期进展：

- [DreamFusion](https://dreamfusion3d.github.io/) 使用 diffusion 技术从 2D 图像生成 3D 模型。
- [CLIPMatrix](https://arxiv.org/abs/2109.12922) 和 [CLIP-Mesh-SMPLX](https://github.com/NasirKhalid24/CLIP-Mesh-SMPLX) 可以直接生成 3D 纹理网格。
- [CLIP-Forge](https://github.com/autodeskailab/clip-forge) 可以从文本生成体素 (体积像素，3 维空间最小分割单元，类似图片的像素) 3D 模型。
- [CLIP-NeRF](https://github.com/cassiePython/CLIPNeRF) 可以输入文本或者图像来驱动 NeRF 生成新的 3D 模型。
- [Point-E](https://huggingface.co/spaces/openai/point-e) 和 [Pulsar+CLIP](https://colab.research.google.com/drive/1IvV3HGoNjRoyAKIX-aqSWa-t70PW3nPs) 可以用文本生成 3D 点云。
- [Dream Textures](https://github.com/carson-katri/dream-textures/releases/tag/0.0.9) 使用了 文本-图像 技术，可以在 Blender (三维图形图像软件) 中自动对场景纹理贴图。

除 CLIPMatrix 和 CLIP-Mesh-SMPLX 之外，上述大部分方法或基于 [视图合成](https://en.wikipedia.org/wiki/View_synthesis) (view synthesis) 生成 3D 对象，或生成特定主体的新视角，这就是 [NeRFs](https://developer.nvidia.com/blog/getting-started-with-nvidia-instant-nerfs/) (Neural Radiance Fields，神经辐射场) 背后的思想。NeRF 使用神经网络来做视图合成，这与传统 3D 渲染方法 (网格、UV 映射、摄影测量等) 有较大差异。

<figure class="image text-center">
  <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2022/05/Excavator_NeRF.gif" alt="NeRF">
  <figcaption>使用 NeRF 做视图合成</figcaption>
</figure>

那么，这些技术为游戏开发者带来了多少可能性? 我认为 *现阶段* 是零，实际上它还没有发展到可用于游戏开发的程度。下面我会说明原因。

### 现阶段不可行的原因

**注意:** 此部分面向熟悉传统 3D 渲染技术 (如 [网格](https://en.wikipedia.org/wiki/Polygon_mesh)，[UV 映射](https://en.wikipedia.org/wiki/UV_mapping)，和 [摄影测量](https://en.wikipedia.org/wiki/Photogrammetry)) 的读者。

网格是大部分 3D 世界的运行基石。诸如 NeRFs 的视图合成技术虽然效果非常惊艳，但现阶段却难以兼容网格。不过 [NeRFs 转换为网格方向的工作已经在进行中](https://github.com/NVlabs/instant-ngp)，这部分的工作与 [摄影测量](https://en.wikipedia.org/wiki/Photogrammetry) 有些类似，摄影测量是对现实世界特定对象采集多张图像并组合起来，进而制作网格化的 3D 模型素材。

<figure class="image text-center">
  <img src="https://github.com/NVlabs/instant-ngp/raw/master/docs/assets_readme/testbed.png" alt="NeRF-to-mesh">
  <figcaption>NVlabs instant-ngp, 支持 NeRF-网格 转换。</figcaption>
</figure>

既然基于神经网络的 文本-NeRF-网格和摄影测量的采图-组合-网格两者的 3D 化流程有相似之处，同样他们也具有相似的局限性：生成的 3D 网格素材不能直接在游戏中使用，而需要大量的专业知识和额外工作才能使用。因此我认为，NeRF-网格可能是一个有用的工具，但现阶段并未显示出 文本-3D 的变革潜力。

还拿摄影测量类比，目前 NeRF-网格 最适合的场景同样是创建超高保真模型素材，但实际上这需要大量的人工后处理工作，因此这项技术用在 5 天创建一个农场游戏系列中没有太大意义。为保证游戏开发顺利进行，对于需要有差异性的多种农作物 3D 模型，我决定仅使用颜色不同的立方体加以区分。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/cubes.png" alt="Stable Diffusion Space 应用">
</figure>

不过 AI 领域的变革非常迅速，可能很快就会出现可行的解决方案。在下文中，我将讨论 文本-3D 的一些发展方向。

### 文本-3D 的未来发展

虽然 文本-3D 领域最近取得了长足进步，但与现阶段 文本-2D 的影响力相比，仍有显著的差距。对于如何缩小这个差距，我这里推测两个可能的方向：

1. 改进 NeRF-网格 和网格生成 (将连续的几何空间细分为离散的网格拓扑单元) 技术。如上文提到的，现阶段 NeRF 生成的 3D 模型需要大量额外的工作才能作为游戏素材使用，虽然这种方法在创建高保真模型素材时非常有效，但它是以大量时间开销为代价的。如果您跟我一样使用 low-poly (低多边形) 美术风格来开发游戏，那么对于从零开始制作 3D 素材，您可能会偏好更低耗时的方案。
2. 更新渲染技术：允许 NeRF 直接在引擎中渲染。虽然没有官方公告，不过从 [Nvidia Omniverse](https://www.nvidia.com/en-us/omniverse/) 和 [Google DreamFusion3d](https://dreamfusion3d.github.io/) 推测，有许多开发者正在为此努力。

时间会给我们答案。如果您想跟上最新进展，可以在 [Twitter](https://twitter.com/dylan_ebert_) 上关注我查看相关动态。如果我错过了哪些新进展，也可以随时与我联系！

请继续阅读 [第 4 部分](https://huggingface.co/blog/zh/ml-for-games-4) 的分享，我将为您介绍如何 **使用 AI 制作 2D 素材**。

#### 致谢

感谢 Poli [@multimodalart](https://huggingface.co/multimodalart) 提供的最新开源 文本-3D 信息。
