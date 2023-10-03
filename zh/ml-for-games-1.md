---
title: "基于AI进行游戏开发：5天！创建一个农场游戏！第1部分"
thumbnail: /blog/assets/124_ml-for-games/thumbnail.png
authors:
- user: dylanebert
translators:
- user: SuSung-boy
- user: inferjay
  proofreader: true
---

# 基于AI进行游戏开发：5天！创建一个农场游戏！第1部分


**欢迎使用 AI 进行游戏开发！** 在本系列中，我们将使用各种 AI 工具，在 5 天内创建一个功能完备的农场游戏。到本系列结束时，你将了解到如何将多种 AI 工具整合到游戏开发流程中。本系列文章将向你展示如何将 AI 工具用于：

1.  美术风格
2.  游戏设计
3.  3D 素材
4.  2D 素材
5.  剧情

想要观看视频快速了解？请 [点击这里](https://www.tiktok.com/@individualkex/video/7184106492180630827) 快速了解本文内容，如果你需要掌握更多技术细节，请继续阅读！

> 注意：本教程面向熟悉 Unity 开发和 C# 语言的读者。如果你不熟悉这些技术，我们建议你先了解一下 Unity 然后再继续阅读，本文的作者制作了一系列 [Unity 初学者视频](https://www.tiktok.com/@individualkex/video/7086863567412038954?is_from_webapp=1&sender_device=pc&web_id=7043883634428052997)，如果你希望我们发布到 Hugging Face 账号，请在评论区留言告诉我们。

第 1 天：确立美术风格
============

游戏开发流程第一步是 **确立美术风格**。对于要创建的农场游戏，本文将使用 Stable Diffusion 工具来帮助其美术风格的确立。Stable Diffusion 是一种基于文本描述生成图像的开源模型。接下来会介绍如何使用该工具为农场游戏创建视觉美术风格。

Stable Diffusion 基本设置
---------------------

运行 Stable Diffusion 有两种方案可选：**本地或在线**。如果你拥有一台配备良好 GPU 的台式机并想使用全功能工具库，那么更建议 [本地方案](#locally)。除此之外，你还可以尝试 [在线方案](#online)，请继续阅读本文详细查看本地方案和线上方案：

本地方案 <a name="locally"></a>
----

本文将使用 [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 在本地运行 Stable Diffusion。这是比较流行的本地运行 Stable Diffusion 的方案，不过要成功将其设置运行起来，还需要一些技术知识。如果你使用 Windows 且具有 8GB 以上内存的 Nvidia GPU，请按以下指示执行。否则，请在 [代码仓库](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 的 README 文件中查看其他平台的运行说明，更或者可以选择 [在线方案](#online)。 

### 在 Windows 上安装

**要求：** 具有 8 GB 以上内存的 Nvidia GPU。

1.  安装 [Python 3.10.6](https://www.python.org/downloads/windows/)，安装时勾选 "Add Python to PATH"
2.  安装 [git](https://git-scm.com/download/win)
3.  在命令提示符中输入以下内容来克隆所需仓库
    
	```
	git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
	```

4.  下载 [Stable Diffusion v1.5 checkpoint](https://huggingface.co/runwayml/stable-diffusion-v1-5)，并将其移动到仓库的 `models` 目录下  
5.  运行 `webui-user.bat` 来启动 WebUI  
6.  浏览器中访问 `localhost://7860`。如果一切正常，你将看到如下内容:
    
<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/webui.png" alt="Stable Diffusion WebUI">
</figure> 

### 在线方案 <a name="online"></a>

如果你不具备本地运行 Stable Diffusion 的条件，或者偏好简易的解决方案，同样有多种在线运行方案供你选择。

🤗 Hugging Face 提供的 [Space](https://huggingface.co/spaces) 应用中包含众多免费在线方案，例如 [Stable Diffusion 2.1 Demo](https://huggingface.co/spaces/stabilityai/stable-diffusion) 或 [camemduru webui](https://huggingface.co/spaces/camenduru/webui)。你可以 [查看更多在线服务](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)，甚至可以使用 🤗 [Diffusers](https://huggingface.co/docs/diffusers/index) 编写你专属的免费运行方案！你也可以查看简单的 [代码示例](https://colab.research.google.com/drive/1HebngGyjKj7nLdXfj6Qi0N1nh7WvD74z) 以快速上手。 

**注意：** 本系列的部分内容将使用 image2image 等高级功能，有些在线服务未提供这些功能。

### 生成概念艺术图片 <a name="generating"></a>

首先让我们生成一些概念图。只需几步，非常简单：

1.  输入提示语  
2.  点击生成
    
<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/sd-demo.png" alt="Stable Diffusion Demo Space">
</figure>

但问题是，生成的图片是你真正想要的结果吗？如果不是，如何才能获得呢？这里要提醒你一下，输入提示语，本身就需要一些技巧。所以如果你生成的第一张图片非你所想也没关系，网络上有非常多神级资源可以帮助改善提示语。你可以查看 Reddit 上的帖子了解 [书写指南](https://www.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/)，也可以 [点击这里](https://youtube.com/shorts/8PGucf999nI?feature=share) 查看我做的视频，带你 20 秒了解图片生成提示语的技巧！

上述书写技巧的共通之处是使用诸如 [lexica.art](https://lexica.art/) 网站之类的图片库来查看其他创作者使用提示语在 Stable Diffusion 生成的内容范式，从中寻找与你期望风格相似的图片，从而获得书写提示语的灵感。实际上没有所谓的标准答案，不过在你使用 Stable Diffusion 1.5 生成概念艺术图片时，建议遵循以下温馨提示：

*   使用描述词: 描述词会限制生成图片的形式，如 isometric, simple, solid shapes 等。这样生成图片的美术风格在游戏中会更容易重现。
*   使用同义关键词: 一些关键词 (如 low poly) 虽然契合主题，但生成的图片质量通常较低。尝试找到它们的同义词，替换以保证生成质量。
*   使用指定艺术家的名字: 这种方式可以有效地引导模型采用指定艺术家的绘画风格，从而生成更高质量的图片。

我输入这样的提示语: `isometric render of a farm by a river, simple, solid shapes, james gilleard, atey ghailan` 生成图片如下：

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/concept.png" alt="Stable Diffusion Concept Art">
</figure>

### 使用 Unity 重现概念艺术

接下来，如何使用生成的概念艺术图片来制作游戏？本文将使用流行游戏引擎 [Unity](https://unity.com/) 来使游戏鲜活起来。

1.  使用带有 [通用渲染管道](https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@15.0/manual/index.html) 的 [Unity 2021.9.3f1](https://unity.com/releases/editor/whats-new/2021.3.9) 创建一个 Unity 项目。
    
2.  使用基本形状绘制场景草图。例如，要添加一个立方体形状，**右键单击 -> 3D对象 (3D Object) -> 立方体 (Cube)**
    
	<figure class="image text-center">
	  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gray.png" alt="Gray Scene">
	</figure>
    
3.  设置 [材质](https://docs.unity3d.com/Manual/Materials.html): 可以参考前面生成的概念艺术图片对各部分进行设置。这里选用 Unity 内置的基本材质
    
	<figure class="image text-center">
		  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/color.png" alt="Scene with Materials">
	</figure>
    
4.  设置 [光照](https://docs.unity3d.com/Manual/Lighting.html): 这里使用暖调自然光 (`#FFE08C`，强度 `1.25`) 和柔和环境光 (`#B3AF91`)
    
	<figure class="image text-center">
	  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/lighting.png" alt="Scene with Lighting">
	</figure>
    
5.  设置 [摄像机](https://docs.unity3d.com/ScriptReference/Camera.html): 这里使用 **正交投影** 来匹配概念艺术图片的投影形式
    
	<figure class="image text-center">
	  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/camera.png" alt="Scene with Camera">
	</figure>
    
6.  设置 [水着色器](https://assetstore.unity.com/packages/vfx/shaders/stylized-water-shader-71207): 可以给游戏场景增加一些水流，这里使用 Unity 资源商店中的程式化水着色器
    
	<figure class="image text-center">
	  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/water.png" alt="Scene with Water">
	</figure>
    
7.  最后，设置后 [处理效果](https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@7.1/manual/integration-with-post-processing.html): 这里使用 ACES 色调映射和 +0.2 曝光
    
	<figure class="image text-center">
	  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/post-processing.png" alt="Final Result">
	</figure> 

至此，一个简单上手而引人入胜的游戏场景，不到一天就创建完成了！如果你有任何问题，或者想跃跃欲试参与后续内容？现在来 [加入我们的 Discord 频道](https://hf.co/join/discord) 与我们交流吧！

在 [下一篇文章](https://huggingface.co/blog/zh/ml-for-games-2) 中，我们将 **使用 AI 进行游戏设计**，敬请关注我们获得后续更新！
