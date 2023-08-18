---
title: "手把手教你使用人工智能生成 3D 素材"
thumbnail: /blog/assets/124_ml-for-games/thumbnail-3d.jpg
authors:
- user: dylanebert
translators:
- user: chenglu
---

# 手把手教你使用人工智能生成 3D 素材

## 引言

生成式 AI 已成为游戏开发中艺术工作流的重要组成部分。然而，正如我在 [之前的文章](https://huggingface.co/blog/zh/ml-for-games-3) 中描述的，从文本到 3D 的实用性仍落后于 2D。不过，这种情况正在改变。本文我们将重新审视 3D 素材生成的实用工作流程，并逐步了解如何将生成型 AI 集成到 PS1 风格的 3D 工作流中。

![最终结果](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/3d/result.png)

为什么选择 PS1 风格？因为它对当前文本到 3D 模型的低保真度更为宽容，使我们能够以尽可能少的努力从文本转换为可用的 3D 素材。

### 预备知识

本教程假设你具备一些 Blender 和 3D 概念的基本知识，例如材质和 UV 映射。

## 第一步：生成 3D 模型

首先访问 Shap-E Hugging Face Space [这里](https://huggingface.co/spaces/hysts/Shap-E)或下方。此空间使用 OpenAI 最新的扩散模型 [Shap-E model](https://github.com/openai/shap-e) 从文本生成 3D 模型。

<gradio-app theme_mode="light" space="hysts/Shap-E"></gradio-app>

输入 "Dilapidated Shack" 作为你的提示并点击 'Generate'。当你对模型满意时，下载它以进行下一步。

![shap-e space](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/3d/shape.png)

## 第二步：导入并精简模型

接下来，打开 [Blender](https://www.blender.org/download/)（版本 3.1 或更高）。转到 File -> Import -> GLTF 2.0，并导入你下载的文件。你可能会注意到，该模型的多边形数量远远超过了许多实际应用（如游戏）的推荐数量。

![导入 blender 中的模型](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/3d/import.png)

要减少多边形数量，请选择你的模型，导航到 Modifiers，并选择 "Decimate" 修饰符。将比率调整为较低的数字（例如 0.02）。这可能看起来*不*太好。然而，在本教程中，我们将接受低保真度。

## 第三步：安装 Dream Textures

为了给我们的模型添加纹理，我们将使用 [Dream Textures](https://github.com/carson-katri/dream-textures)，这是一个用于 Blender 的稳定扩散纹理生成器。按照 [官方仓库](https://github.com/carson-katri/dream-textures) 上的说明下载并安装插件。

![安装 dream textures](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/3d/dreamtextures.png)

安装并启用后，打开插件首选项。搜索并下载 [texture-diffusion](https://huggingface.co/dream-textures/texture-diffusion) 模型。

## 第四步：生成纹理

让我们生成一个自定义纹理。在 Blender 中打开 UV 编辑器，按 'N' 打开属性菜单。点击 'Dream' 标签并选择 texture-diffusion 模型。将 Prompt 设置为 'texture'、Seamless 设置为 'both'。这将确保生成的图像是无缝纹理。

在 'subject' 下，输入你想要的纹理，例如 'Wood Wall'，然后点击 'Generate'。当你对结果满意时，为其命名并保存。

![生成纹理](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/3d/generate.png)

要应用纹理，请选择你的模型并导航到 'Material'。添加新材料，在 'base color' 下点击点并选择 'Image Texture'。最后，选择你新生成的纹理。

## 第五步：UV 映射

接下来是 UV 映射，它将我们的 2D 纹理包裹在 3D 模型周围。选择你的模型，按 'Tab' 进入编辑模式。然后，按 'U' 展开模型并选择 'Smart UV Project'。

要预览你的纹理模型，请切换到渲染视图（按住 'Z' 并选择 'Rendered'）。你可以放大 UV 映射，使其在模型上无缝平铺。请记住，我们的目标是复古的 PS1 风格，所以不要做得太好。

![uv 映射](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/3d/uv.png)

## 第六步：导出模型

当您对模型感到满意时，就可以导出它了。使用 File -> Export -> FBX，这个 3D 素材就生成了。

## 第七步：在 Unity 中导入

最后，让我们看看我们的模型在实际中的效果。将其导入 [Unity](https://unity.cn/download) 或你选择的游戏引擎中。为了重现怀旧的 PS1 美学，我用自定义顶点照明、无阴影、大量雾气和故障后处理进行了定制。你可以在 [这里](https://www.david-colson.com/2021/11/30/ps1-style-renderer.html) 了解更多关于重现 PS1 美学的信息。

现在我们就拥有了一个在虚拟环境中的低保真、纹理 3D 模型！

![最终结果](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/3d/result.png)

## 总结

关于如何使用生成型 AI 工作流程创建实用 3D 素材的教程就此结束。虽然结果保真度不高，但潜力巨大：通过足够的努力，这种方法可以用来生成一个低保真风格的无限世界。随着这些模型的改进，将这些技术转移到高保真或逼真的风格将会成为可能！
