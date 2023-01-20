---
title: "3D Asset Generation: AI for Game Development #3"
thumbnail: /blog/assets/124_ml-for-games/thumbnail3.png
---

<h1>3D Asset Generation: AI for Game Development #3</h1>

<div class="author-card">
    <a href="/dylanebert">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1672164046414-624b4a964056e2a6914a05c5.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>dylanebert</code>
            <span class="fullname">Dylan Ebert</span>
        </div>
  </a>
</div>
 
</head>

<body>

**Welcome to AI for Game Development!** In this series, we'll be using AI tools to create a fully functional farming game in just 5 days. By the end of this series, you will have learned how you can incorporate a variety of AI tools into your game development workflow. I will show you how you can use AI tools for:

1. Art Style
2. Game Design
3. 3D Assets
4. 2D Assets
5. Story

Want the quick video version? You can watch it [here](https://www.tiktok.com/@individualkex/video/7184106492180630827). Otherwise, if you want the technical details, keep reading!

<!-- TODO: Update above link to video -->

**Note:** This tutorial is intended for readers who are familiar with Unity development and C#. If you're new to these technologies, check out the [Unity for Beginners](https://www.tiktok.com/@individualkex/video/7086863567412038954) series before continuing.

## Day 3: 3D Assets

In [Part 2](https://huggingface.co/blog/ml-for-games-2) of this tutorial series, we used **AI for Game Design**. More specifically, we used ChatGPT to brainstorm the design for our game.

In this part, we'll talk about how you can use AI to generate 3D Assets. The short answer is: you can't. That's because text-to-3D isn't at the point it can be practically applied to game development, *yet*. However, that's changing very quickly. Keep reading to learn about [The Current State of Text-to-3D](#the-current-state-of-text-to-3d), [Why It Isn't Useful (yet)](#why-it-isnt-useful-yet), and [The Future of Text-to-3D](#the-future-of-text-to-3d).

### The Current State of Text-to-3D

As discussed in [Part 1](https://huggingface.co/blog/ml-for-games-1), text-to-image tools such as Stable Diffusion are incredibly useful in the game development workflow. However, what about text-to-3D, or generating 3D models from text descriptions? There have been many very recent developments in this area:

- [DreamFusion](https://dreamfusion3d.github.io/) uses 2D diffusion to generate 3D assets.
- [CLIPMatrix](https://arxiv.org/abs/2109.12922) and [CLIP-Mesh-SMPLX](https://github.com/NasirKhalid24/CLIP-Mesh-SMPLX) generate textured meshes directly.
- [CLIP-Forge](https://github.com/autodeskailab/clip-forge) uses language to generate voxel-based models.
- [CLIP-NeRF](https://github.com/cassiePython/CLIPNeRF) drives NeRFs with text and images.
- [Point-E](https://huggingface.co/spaces/openai/point-e) and [Pulsar+CLIP](https://colab.research.google.com/drive/1IvV3HGoNjRoyAKIX-aqSWa-t70PW3nPs) use language to generate 3D point clouds.

Many of these approaches, excluding CLIPMatrix and CLIP-Mesh-SMPLX, are based on [view synthesis](https://en.wikipedia.org/wiki/View_synthesis), or generating novel views of a subject, as opposed to conventional 3D rendering. This is the idea behind [NeRFs](https://developer.nvidia.com/blog/getting-started-with-nvidia-instant-nerfs/) or Neural Radiance Fields, which use neural networks for view synthesis.

<figure class="image text-center">
  <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2022/05/Excavator_NeRF.gif" alt="NeRF">
  <figcaption>View synthesis using NeRFs.</figcaption>
</figure>

What does all of this mean if you're a game developer? Currently, nothing. This technology hasn't reached the point that it's useful in game development *yet*. Let's talk about why.

### Why It Isn't Useful (yet)

**Note:** This section is intended for readers who are familiar with conventional 3D rendering techniques, such as [meshes](https://en.wikipedia.org/wiki/Polygon_mesh), [UV mapping](https://en.wikipedia.org/wiki/UV_mapping) and [photogrammetry](https://en.wikipedia.org/wiki/Photogrammetry).

While view synthesis is impressive, the world of 3D runs on meshes, which are not the same as NeRFs. There is, however, [ongoing work on converting NeRFs to meshes](https://github.com/NVlabs/instant-ngp). In practice, this is reminiscient of [photogrammetry](https://en.wikipedia.org/wiki/Photogrammetry), where multiple photos of real-world objects are combined to author 3D assets.

<figure class="image text-center">
  <img src="https://github.com/NVlabs/instant-ngp/raw/master/docs/assets_readme/testbed.png" alt="NeRF-to-mesh">
  <figcaption>NVlabs instant-ngp, which supports NeRF-to-mesh conversion.</figcaption>
</figure>

The practical use of assets generated using the text-to-NeRF-to-mesh pipeline is limited in a similar way to assets produced using photogrammetry. That is, the resulting mesh is not immediately game-ready, and requires significant work and expertise to become a game-ready asset. In this sense, NeRF-to-mesh may be a useful tool as-is, but doesn't yet reach the transformative potential of text-to-3D.

Since NeRF-to-mesh, like photogrammetry, is currently most suited to creating ultra-high-fidelity assets with significant manual post-processing, it doesn't really make sense for creating a farming game in 5 days. In which case, I decided to just use cubes of different colors to represent the crops in the game.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/cubes.png" alt="Stable Diffusion Demo Space">
</figure>

Things are changing rapidly in this area, though, and there may be a viable solution in the near future. Next, I'll talk about some of the directions text-to-3D may be going.

### The Future of Text-to-3D

While text-to-3D has come a long way recently, there is still a significant gap between where we are now and what could have an impact along the lines of text-to-image. I can only speculate on how this gap will be closed. There are two possible directions that are most apparent:

1. Improvements in NeRF-to-mesh and mesh generation. As we've seen, current generation models are similar to photogrammetry in that they require a lot of work to produce game-ready assets. While this is useful in some scenarios, like creating realistic high-fidelity assets, it's still more time-consuming than making low-poly assets from scratch, especially if you're like me and use an ultra-low-poly art style.
2. New rendering techniques that allow NeRFs to be rendered directly in-engine. While there have been no official announcements, one could speculate that [NVIDIA](https://www.nvidia.com/en-us/omniverse/) and [Google](https://dreamfusion3d.github.io/), among others, may be working on this.

Of course, only time will tell. If you want to keep up with advancements as they come, feel free to follow me on [Twitter](https://twitter.com/dylan_ebert_). If there are new developments I've missed, feel free to reach out!

In the next part, we'll be using **AI to Generate 2D Assets**.

<!-- TODO: Add link to next part -->

#### Attribution

Thanks to Poli [@multimodalart](https://huggingface.co/multimodalart) for providing info on the latest open source text-to-3D.
