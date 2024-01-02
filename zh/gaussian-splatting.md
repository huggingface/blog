---
title: "3D 高斯点染简介"
thumbnail: /blog/assets/124_ml-for-games/thumbnail-gaussian-splatting.png
authors:
- user: dylanebert
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 3D 高斯点染简介

3D 高斯点染技术由 [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://huggingface.co/papers/2308.04079) 一文首次提出。作为一种栅格化技术，3D 高斯点染可用于实时且逼真地渲染从一小组图像中学到的场景。本文将详细介绍其工作原理并讨论其对图形学的未来会带来什么影响。

## 什么是 3D 高斯点染？

3D 高斯点染本质上是一种栅格化技术。也就是说:

1. 我们有场景描述数据;
2. 我们会把这些数据在屏幕上渲染出来。

大家可能对计算机图形学中的三角形栅格化比较熟悉，其通过在屏幕上渲染许多三角形的方式来绘制图形。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/triangle.png)

其实 3D 高斯点染与三角形栅格化是一样的，只不过把基本绘图元素从三角形换成了高斯图像。下图给出了高斯图像的一个例子，为清晰起见，我们标出了它的边框。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/single-gaussian.png)

每个高斯图像可由以下参数来描述:

- **位置**: 所在位置 (XYZ)
- **协方差**: 缩放程度 (3x3 矩阵)
- **颜色**: 颜色 (RGB)
- **Alpha**: 透明度 (α)

在实践中，我们通过在屏幕上绘制多个高斯图像，从而画出想要的图像。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/three-gaussians.png)

上图是由 3 个高斯图像渲染出的图像。那么用 700 万个高斯图像可能会渲染出怎样的图像呢？看看下图:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/bicycle.png)

如果这 700 万个高斯图像每个都完全不透明的话，渲染出的图像又会怎么样呢？如下:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/ellipsoids.png)

以上，我们对 3D 高斯点染有了一个初步的认识。接下来，我们了解一下点染的具体过程。

## 点染过程

### 1. 从运动中恢复出结构

第一步是使用运动恢复结构 (Structure from Motion，SfM) 方法从一组图像中估计出点云。SfM 方法可以让我们从一组 2D 图像中估计出 3D 点云。我们可以直接调用 [COLMAP](https://colmap.github.io/) 库来完成这一步。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/points.png)

### 2. 用高斯图像对云中的每个点进行建模

接下来，把每个点建模成一个 3D 高斯图像。从 SfM 数据中，我们能推断出每个高斯图像的位置和颜色。这对于一般的栅格化已经够用了，但如果要产生更高质量的表征的话，我们还需要对每个高斯图像进行训练，以推断出更精细的位置和颜色，并推断出协方差和透明度。

### 3. 训练

与神经网络类似，我们使用随机梯度下降法进行训练，但这里没有神经网络的层的概念 (都是 3D 高斯函数)。训练步骤如下:

1. 用当前所有可微高斯图像渲染出图像 (稍后详细介绍)
2. 根据渲染图像和真实图像之间的差异计算损失
3. 根据损失调整每个高斯图像的参数
4. 根据情况对当前相关高斯图像进行自动致密化及修剪

步骤 1-3 比较简单，下面我们稍微解释一下第 4 步的工作:

- 如果某高斯图像的梯度很大 (即它错得比较离谱)，则对其进行分割或克隆
  - 如果该高斯图像很小，则克隆它
  - 如果该高斯图像很大，则将其分割

- 如果该高斯图像的 alpha 太低，则将其删除

这么做能帮助高斯图像更好地拟合精细的细节，同时修剪掉不必要的高斯图像。

### 4. 可微高斯栅格化

如前所述，3D 高斯点染是一种 _栅格化_ 方法，即我们可以用其将数据渲染到屏幕上。作为众多栅格化方法的 _其中之一_ ，它有两个特点:

1. 快
2. 可微

你可在 [此处](https://github.com/graphdeco-inria/diff-gaussian-rasterization) 找到可微高斯渲染器的原始实现。其主要步骤为:

1. 针对给定相机视角，把每个 3D 高斯图像投影到 2D。
2. 按深度对高斯图像进行排序。
3. 对每个像素，从前到后计算每个高斯函数在该像素点的值，并将所有值混合以得到最终像素值。

更多细节及优化可参阅 [论文](https://huggingface.co/papers/2308.04079)。

渲染器可微这一点很重要，因为这样我们就可以用随机梯度下降法来训练它。但这一点仅对训练阶段很重要，训后的高斯函数是可以用不可微的方式来表示的。

## 有啥用？

为什么 3D 高斯点染受到如此多的关注？最直接的原因是其非凡的实力。有了它，对高画质场景的实时渲染成为了可能。有了这个能力，我们可以解锁更多可能的应用。

比如说，可以用它来渲染动画吗？即将发表的论文 [Dynamic 3D Gaussians: tracking by Persistent Dynamic View Synthesis](https://arxiv.org/pdf/2308.09713) 似乎表明这有戏。还有更多其他问题有待研究。它能对反射进行建模吗？可以不经参考图像的训练就直接建模吗……

最后，当前人们对 [具身智能 (Embodied AI)](https://ieeexplore.ieee.org/iel7/7433297/9741092/09687596.pdf) 兴趣日隆。但作为人工智能的一个研究领域，当前最先进的具身智能的性能仍然比人类低好几个数量级，其中大部分的挑战在 3D 空间的表示上。鉴于 3D 高斯分布可以产生致密的 3D 空间表示，这对具身智能研究有何影响？

所有这些问题都引发了人们对 3D 高斯点染的广泛关注。时间会告诉我们答案！

## 图形学的未来

3D 高斯点染会左右图形学的未来吗？我们先来看下该方法的优缺点:

**优点**

1. 高品质、逼真的场景
2. 快速、实时的渲染
3. 更快的训练速度

**缺点**

1. 更高的显存使用率 (4GB 用于显示，12GB 用于训练)
2. 更大的磁盘占用 (每场景 1GB+)
3. 与现有渲染管线不兼容
4. 仅能绘制静态图像 (当前)

到目前为止，3D 高斯点染的 CUDA 原始实现尚未与 Vulkan、DirectX、WebGPU 等产品级渲染管道进行适配，因此尚不清楚其会对现有渲染管线带来什么影响。

已有的适配如下:

1. [远程显示器](https://huggingface.co/spaces/dylanebert/gaussian-viewer)
2. [WebGPU 显示器](https://github.com/cvlab-epfl/gaussian-splatting-web)
3. [WebGL 显示器](https://huggingface.co/spaces/cakewalk/splat)
4. [Unity 显示器](https://github.com/aras-p/UnityGaussianSplatting)
5. [优化过的 WebGL 显示器](https://gsplat.tech/)

这些显示器要么依赖于远程流式传输 (1)，要么依赖于传统的基于 2x2 像素块的栅格化方法 (2-5)。虽然基于 2x2 像素块的方法与数十年来的图形技术兼容，但它可能会导致质量/性能的降低。然而，[第 5 个显示器](https://gsplat.tech/) 的工作又表明，尽管采用基于 2x2 像素块的方法，通过巧妙的优化我们仍可以达到高的质量及性能。

那么有没有可能需要针对生产环境重实现 3D 高斯点染代码呢？答案是 _有可能_ 。当前主要的性能瓶颈在对数百万个高斯图像进行排序上，在论文的原始实现中，这一过程是通过 [CUB 库的基数排序](https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html) 原语来高效实现的，但该高性能原语仅在 CUDA 中可用。我们相信，经过努力，其他渲染管线也可以达到相仿的性能水平。

如果你有任何问题或有兴趣加入我们的工作，请加入 [Hugging Face Discord](https://hf.co/join/discord)！