---
title: "Gradio 5 现已发布"
thumbnail: /blog/assets/gradio-5/thumbnail.png
authors:
- user: abidlabs
translators:
- user: chenglu
---

# Gradio 5 现已发布

在过去的几个月里，我们一直在努力工作，现在我们非常激动地宣布 **Gradio 5 的稳定版发布**。

有了 Gradio 5，开发者可以构建 **生产级的机器学习 Web 应用程序**，这些应用不仅性能优越、可扩展、设计精美、易于访问，而且还遵循了最佳的 Web 安全实践。更重要的是，只需几行 Python 代码即可实现。

想要体验 Gradio 5，只需在终端中输入以下命令：

```
pip install --upgrade gradio
```

然后开始构建你的 [第一个 Gradio 应用](https://www.gradio.app/guides/quickstart)。

## Gradio 5：面向生产环境的机器学习应用构建工具

如果你之前使用过 Gradio，可能会想知道 Gradio 5 有什么不同。

Gradio 5 的目标是倾听和解决 Gradio 开发者在构建生产级应用时遇到的常见问题。例如，我们听到一些开发者提到：

* “Gradio 应用加载太慢” → Gradio 5 带来了重大的性能改进，包括通过服务端渲染 (SSR) 提供 Gradio 应用，几乎可以在浏览器中瞬间加载应用。_告别加载时的转圈圈_！ 🏎️💨

<video width="600" controls playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-5/gradio-4-vs-5-load.mp4">
</video>

* “这个 Gradio 应用看起来有点过时” → Gradio 5 对许多核心组件进行了现代化设计改进，包括按钮、标签页、滑块以及高级聊天界面。我们还发布了一组全新的内置主题，让你可以轻松创建外观时尚的 Gradio 应用 🎨。

* “我无法在 Gradio 中构建实时应用” → Gradio 5 实现了低延迟的流式处理！我们使用 base64 编码和 websockets 自动加速，还通过自定义组件支持 WebRTC。此外，还增加了大量文档和示例演示，涵盖了常见的流式处理用例，如基于摄像头的物体检测、视频流处理、实时语音转录与生成，以及对话型聊天机器人。 🎤

* “LLM 不了解 Gradio” → Gradio 5 推出了一个实验性的 AI Playground，你可以在其中使用 AI 来生成或修改 Gradio 应用，并立即在浏览器中预览：[https://www.gradio.app/playground](https://www.gradio.app/playground)

<video width="600" controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-5/simple-playground.mp4">
</video>

Gradio 5 在保留简单直观的开发者 API 的同时，提供了所有这些新功能。作为面向各种机器学习应用的生产级 Web 框架，Gradio 5 还在 Web 安全性方面做了重大改进（包括第三方审计）——更多详情将在即将发布的博客中介绍！

## 不兼容的更改

在 Gradio 4.x 中没有出现弃用警告的 Gradio 应用应该可以继续在 Gradio 5 中正常运行，[少数例外请参考 Gradio 5 中的不兼容更改列表](https://github.com/gradio-app/gradio/issues/9463)。

## Gradio 的下一步计划

我们在 Gradio 5 中的许多更改是为了支持即将发布的新功能。敬请期待以下内容：

* 支持多页面的 Gradio 应用，以及原生的导航栏和侧边栏
* 支持通过 PWA 在移动设备上运行 Gradio 应用，甚至可能支持原生应用
* 更多媒体组件，以支持新兴的图像和视频处理模式
* 更丰富的 DataFrame 组件，支持常见的电子表格操作
* 与机器学习模型和 API 提供商的一键集成
* 进一步减少 Gradio 应用的内存消耗

以及更多功能！有了 Gradio 5 提供的稳固基础，我们非常期待让开发者使用 Gradio 构建各种机器学习应用。

## 立即试用 Gradio 5

以下是一些运行 Gradio 5 的 Hugging Face Spaces：

* https://huggingface.co/spaces/akhaliq/depth-pro
* https://huggingface.co/spaces/hf-audio/whisper-large-v3-turbo
* https://huggingface.co/spaces/gradio/chatbot_streaming_main
* https://huggingface.co/spaces/gradio/scatter_plot_demo_main
