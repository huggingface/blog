---
title: "用 Claude 和 Hugging Face 生成图像"
thumbnail: /blog/assets/claude-and-mcp/thumbnail.png
authors:
- user: evalstate
translators:
- user: HCS9527
---

# 用 Claude 和 Hugging Face 生成图像

> [!TIP]
> **TL;DR:** 通过将 Claude 连接到 Hugging Face Spaces，现在可以比以往更轻松地使用最先进的 AI 模型生成细节丰富的图像。本文介绍具体方法和原因，并介绍近期发布的几款模型：它们擅长生成自然图像，或生成包含文本的图像。


> [!WARNING]
> 2025 年 10 月更新：Anthropic 更新 [Connector Directory Policy](https://support.claude.com/en/articles/11697096-anthropic-mcp-directory-policy) 后，需要使用「Add custom connector」选项，并将 `Remote MCP server URL` 设置为 <https://huggingface.co/mcp?login>，才能在 Claude 中使用图像生成和其他 Gradio Spaces。

## 简介

图像生成模型近期的进展提升了生成真实图像和融入高质量文本的能力。通过将这些模型直接连接到 Claude，使用它们也比以往更容易。

这种图像生成方式有几个优势：
 - AI 可以帮助构建更详细的提示词，从而提升生成图像的质量。
 - AI 可以「看到」生成的图像，然后帮助迭代设计和技术细节，直到得到理想结果。
 - 可以轻松切换到最新模型，或选择最适合当前需求的模型。

首先，创建一个免费的 [Hugging Face 账户](https://huggingface.co/join)，然后在聊天输入框的「Search and tools」菜单中连接 Claude。下方视频展示了所需的完整步骤：

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt="claude-auth-flow.mp4"
       autoplay loop autobuffer muted playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/claude-images-mcp/claude-auth-flow.mp4" type="video/mp4">
   </video>
  <figcaption>将 Claude 连接到 Hugging Face</figcaption>
 </figure>

在后台，Claude 现在已配置为可以使用 [Hugging Face MCP Server](https://huggingface.co/mcp?login) 提供的工具，从而直接扩展自身能力。其中包括运行在 Spaces 上、由 [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) 提供算力的最新 AI 应用。Hugging Face 账户会提供免费额度，可用于调用这些大型且能力强的模型。

Claude 连接到 Hugging Face 后，下面将展示如何配置 Claude 可调用的图像生成工具。

## 用 Flux.1 Krea Dev 生成自然图像

[FLUX.1 Krea [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) 旨在消除生成图像中常见的「AI 感」，例如塑料感皮肤、过度饱和的色彩，或过于平滑的纹理。如果需要看起来像专业摄影师拍摄、而不是由计算机生成的图像，Krea 可以提供真实的纹理、自然光照和可信的审美效果，而这些正是其他 AI 模型较难稳定做到的。关于它们的实现方式，可以阅读 [Krea 的博客](https://www.krea.ai/blog/flux-krea-open-source-release)。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/claude-images-mcp/bfl_krea_image_wide.avif" alt="Krea 示例">
  <figcaption>Krea 图像生成器示例</figcaption>
</figure>

若要在 Claude 中使用 **Krea**，前往 [`huggingface.co/mcp/settings`](https://huggingface.co/settings/mcp)，并将 `mcp-tools/FLUX.1-Krea-dev` 添加到「Spaces Tools」。这样就可以让 Claude 生成美观、逼真的图像。

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt="adding-mcp-space.mp4"
       autoplay loop autobuffer muted playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/claude-images-mcp/adding-mcp-space.mp4" type="video/mp4">
   </video>
  <figcaption>在 <a href='https://huggingface.co/settings/mcp'>MCP Settings</a> 中添加 Space</figcaption>
 </figure>


然后，可以在 Claude 中尝试这样的提示词：

> "Use Krea to create an image of a vibrant garden with victorian house".

即可开始生成图像。

## Qwen Image

[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) 是一个能力强的 AI 图像生成器，擅长遵循提示词并准确渲染文本。因此，它非常适合设计海报、标识、信息图和营销材料等对文字质量要求较高的内容。关于 Qwen-Image 模型的更多信息，可以阅读它们的[博客文章](https://qwenlm.github.io/blog/qwen-image/)。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/claude-images-mcp/qwen_sample.jpg" alt="Qwen 示例">
  <figcaption>Qwen-Image 图像生成器示例</figcaption>
</figure>

若要使用 **Qwen-Image**，在 [MCP Servers 设置](https://huggingface.co/settings/mcp) 页面添加 `mcp-tools/qwen-image`，然后确认它已在 Claude 中启用，即可开始使用。

Qwen-Image 自带提示词增强器（Prompt Enhancer），可帮助编写适合该模型的详细提示词。若要试用，可以在提示词菜单中选择「Qwen Prompt Enhancer」，输入想法后提交给 Claude。

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt="qwen_image_prompt.mp4"
       autoplay loop autobuffer muted playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/claude-images-mcp/qwen_image_prompt.mp4" type="video/mp4">
   </video>
  <figcaption>在 Claude 中使用 Qwen Prompt Enhancer</a></figcaption>
 </figure>

也可以同时启用 Krea 和 Qwen-Image，并让 Claude 同时使用它们，以便比较结果。例如："Use Krea and Qwen to generate a street scene with 'Hugging Face' graffiti sprayed on the wall"。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/claude-images-mcp/krea_hf_example.webp" alt="Hugging Face 涂鸦街景">
  <figcaption>Hugging Face 涂鸦街景</figcaption>
</figure>


## 结语

将 Claude 连接到 Hugging Face Spaces 后，使用最先进的模型就像点击按钮一样简单，即使这些模型刚刚发布也一样。可以访问 [https://huggingface.co/spaces](https://huggingface.co/spaces) 上的 AI App Directory，并使用 [Video Generation](https://huggingface.co/spaces/Lightricks/ltx-video-distilled)、[Web Search](https://huggingface.co/spaces/victor/websearch)、[Image Editing](https://huggingface.co/spaces/mcp-tools/FLUX.1-Kontext-Dev) 以及数以千计的其他应用构建出色项目。使用 [Pro 账户](https://huggingface.co/pro) 可以获得更高的使用上限等权益。欢迎在下方评论区分享发现和创作成果。
