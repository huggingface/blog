---
title: "Introduction to 3D Gaussian Splatting"
thumbnail: /blog/assets/124_ml-for-games/thumbnail-gaussian-splatting.png
authors:
- user: dylanebert
---

# Introduction to 3D Gaussian Splatting



3D Gaussian Splatting is a rasterization technique described in [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://huggingface.co/papers/2308.04079) that allows real-time rendering of photorealistic scenes learned from small samples of images. This article will break down how it works and what it means for the future of graphics.

## What is 3D Gaussian Splatting?

3D Gaussian Splatting is, at its core, a rasterization technique. That means:

1. Have data describing the scene.
2. Draw the data on the screen.

This is analogous to triangle rasterization in computer graphics, which is used to draw many triangles on the screen.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/triangle.png)

However, instead of triangles, it's gaussians. Here's a single rasterized gaussian, with a border drawn for clarity.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/single-gaussian.png)

It's described by the following parameters:

- **Position**: where it's located (XYZ)
- **Covariance**: how it's stretched/scaled (3x3 matrix)
- **Color**: what color it is (RGB)
- **Alpha**: how transparent it is (α)

In practice, multiple gaussians are drawn at once.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/three-gaussians.png)

That's three gaussians. Now what about 7 million gaussians?

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/bicycle.png)

Here's what it looks like with each gaussian rasterized fully opaque:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/ellipsoids.png)

That's a very brief overview of what 3D Gaussian Splatting is. Next, let's walk through the full procedure described in the paper.

## How it works

### 1. Structure from Motion

The first step is to use the Structure from Motion (SfM) method to estimate a point cloud from a set of images. This is a method for estimating a 3D point cloud from a set of 2D images. This can be done with the [COLMAP](https://colmap.github.io/) library.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gaussian/points.png)

### 2. Convert to Gaussians

Next, each point is converted to a gaussian. This is already sufficient for rasterization. However, only position and color can be inferred from the SfM data. To learn a representation that yields high quality results, we need to train it.

### 3. Training

The training procedure uses Stochastic Gradient Descent, similar to a neural network, but without the layers. The training steps are:

1. Rasterize the gaussians to an image using differentiable gaussian rasterization (more on that later)
2. Calculate the loss based on the difference between the rasterized image and ground truth image
3. Adjust the gaussian parameters according to the loss
4. Apply automated densification and pruning

Steps 1-3 are conceptually pretty straightforward. Step 4 involves the following:

- If the gradient is large for a given gaussian (i.e. it's too wrong), split/clone it
  - If the gaussian is small, clone it
  - If the gaussian is large, split it
- If the alpha of a gaussian gets too low, remove it

This procedure helps the gaussians better fit fine-grained details, while pruning unnecessary gaussians.

### 4. Differentiable Gaussian Rasterization

As mentioned earlier, 3D Gaussian Splatting is a *rasterization* approach, which draws the data to the screen. However, some important elements are also that it's:

1. Fast
2. Differentiable

The original implementation of the rasterizer can be found [here](https://github.com/graphdeco-inria/diff-gaussian-rasterization). The rasterization involves:

1. Project each gaussian into 2D from the camera perspective.
2. Sort the gaussians by depth.
3. For each pixel, iterate over each gaussian front-to-back, blending them together.

Additional optimizations are described in [the paper](https://huggingface.co/papers/2308.04079).

It's also essential that the rasterizer is differentiable, so that it can be trained with stochastic gradient descent. However, this is only relevant for training - the trained gaussians can also be rendered with a non-differentiable approach.

## Who cares?

Why has there been so much attention on 3D Gaussian Splatting? The obvious answer is that the results speak for themselves - it's high-quality scenes in real-time. However, there may be more to story.

There are many unknowns as to what else can be done with Gaussian Splatting. Can they be animated? The upcoming paper [Dynamic 3D Gaussians: tracking by Persistent Dynamic View Synthesis](https://arxiv.org/pdf/2308.09713) suggests that they can. There are many other unknowns as well. Can they do reflections? Can they be modeled without training on reference images?

Finally, there is growing research interest in [Embodied AI](https://ieeexplore.ieee.org/iel7/7433297/9741092/09687596.pdf). This is an area of AI research where state-of-the-art performance is still orders of magnitude below human performance, with much of the challenge being in representing 3D space. Given that 3D Gaussian Splatting yields a very dense representation of 3D space, what might the implications be for Embodied AI research?

These questions call attention the the method. It remains to be seen what the actual impact will be.

## The future of graphics

So what does this mean for the future of graphics? Well, let's break it up into pros/cons:

**Pros**
1. High-quality, photorealistic scenes
2. Fast, real-time rasterization
3. Relatively fast to train

**Cons**
1. High VRAM usage (4GB to view, 12GB to train)
2. Large disk size (1GB+ for a scene)
3. Incompatible with existing rendering pipelines
3. Static (for now)

So far, the original CUDA implementation has not been adapted to production rendering pipelines, like Vulkan, DirectX, WebGPU, etc, so it's yet to be seen what the impact will be.

There have already been the following adaptations:
1. [Remote viewer](https://huggingface.co/spaces/dylanebert/gaussian-viewer)
2. [WebGPU viewer](https://github.com/cvlab-epfl/gaussian-splatting-web)
3. [WebGL viewer](https://huggingface.co/spaces/cakewalk/splat)
4. [Unity viewer](https://github.com/aras-p/UnityGaussianSplatting)
5. [Optimized WebGL viewer](https://gsplat.tech/)

These rely either on remote streaming (1) or a traditional quad-based rasterization approach (2-5). While a quad-based approach is compatible with decades of graphics technologies, it may result in lower quality/performance. However, [viewer #5](https://gsplat.tech/) demonstrates that optimization tricks can result in high quality/performance, despite a quad-based approach.

So will we see 3D Gaussian Splatting fully reimplemented in a production environment? The answer is *probably yes*. The primary bottleneck is sorting millions of gaussians, which is done efficiently in the original implementation using [CUB device radix sort](https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html), a highly optimized sort only available in CUDA. However, with enough effort, it's certainly possible to achieve this level of performance in other rendering pipelines.

If you have any questions or would like to get involved, join the [Hugging Face Discord](https://hf.co/join/discord)!