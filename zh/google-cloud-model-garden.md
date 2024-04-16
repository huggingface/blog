---
title: "在 Google Cloud 上轻松部署开放大语言模型"
thumbnail: /blog/assets/173_gcp-partnership/thumbnail.jpg
authors:
- user: philschmid
- user: jeffboudier
translators:
- user: chenglu
---

# 在 Google Cloud 上轻松部署开放大语言模型

今天，我们想向大家宣布：“**在 Google Cloud 上部署**”功能正式上线！

这是 Hugging Face Hub 上的一个新功能，让开发者可以轻松地将数千个基础模型使用 Vertex AI 或 Google Kubernetes Engine (GKE) 部署到 Google Cloud。

Model Garden (模型库) 是 Google Cloud Vertex AI 平台的一个工具，用户能够发现、定制和部署来自 Google 及其合作伙伴的各种模型。不论是在 Hugging Face 模型页面还是在 Vertex AI 模型库页面，开发者们都可以轻松简单地将开放模型作为 API 端点部署在自己的 Google Cloud 账户内。我们也将启用 Hugging Face 上最受欢迎的开放模型进行推理，这一切都得益于我们的生产级解决方案 [文本生成推理](https://github.com/huggingface/text-generation-inference/)。

借助“在 Google Cloud 上部署”，开发者可以在自己的安全 Google Cloud 环境中直接构建准备就绪的生成式 AI 应用，无需自行管理基础设施和服务器。

## 为 AI 开发者构建

这一全新的体验是基于我们今年早些时候宣布的 [战略合作关系](https://huggingface.co/blog/gcp-partnership) 进一步扩展的，目的是简化 Google 客户访问和部署开放生成式 AI 模型的过程。开发者和机构面临的一个主要挑战是，部署模型需要投入大量时间和资源，且必须确保部署的安全性和可靠性。

“在 Google Cloud 上部署”提供了一个简单且管理化的解决方案，专为 Hugging Face 模型提供了专门的配置和资源。只需简单点击几下，就可以在 Google Cloud 的 Vertex AI 上创建一个准备就绪的端点。

Google 产品经理 Wenming Ye 表示：“Vertex AI 的 Model Garden 与 Hugging Face Hub 的集成，让在 Vertex AI 和 GKE 上发现和部署开放模型变得无缝衔接，无论您是从 Hub 开始，还是直接在 Google Cloud 控制台中。我们迫不及待想看到 Google 开发者们将会用 Hugging Face 模型创建出什么样的创新。”

## 从 HF Hub 开启模型部署

在 Google Cloud 上部署 Hugging Face 模型变得非常简单。以下是如何部署 [Zephyr Gemma](https://console.cloud.google.com/vertex-ai/publishers/HuggingFaceH4/model-garden/zephyr-7b-gemma-v0.1;hfSource=true;action=deploy?authuser=1) 的步骤指导。从今天开始，所有带有 [text-generation-inference](https://huggingface.co/models?pipeline_tag=text-generation-inference&sort=trending) 标签的模型都将受到支持。

![model-card](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/model-card.png)

只需打开“部署”菜单，选择“Google Cloud”即可。这将直接带您进入 Google Cloud 控制台，您可以在 Vertex AI 或 GKE 上轻松一键部署 Zephyr Gemma。

![vertex-ai-model-garden](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/vertex-ai-model-garden.png)

进入 Vertex AI 模型库之后，您可以选择 Vertex AI 或 GKE 作为部署环境。如果选择 Vertex AI，您可以通过点击“部署”一键完成部署过程。如果选择 GKE，您可以根据提供的指南和模板，在新建或现有的 Kubernetes 集群上部署模型。

## 从 Vertex AI 模型库开启模型部署

Vertex AI 模型库是 Google 开发者寻找可用于生成式 AI 项目的现成模型的理想场所。从今天开始，Vertex Model Garden 将提供一种全新的体验，使开发者能够轻松部署 Hugging Face 上可用的最流行的开放大语言模型！

在 Google Vertex AI 模型库中，您会发现一个新的“从 Hugging Face 部署”选项，允许您直接在 Google Cloud 控制台内搜索并部署 Hugging Face 模型。

![deploy-from-huggingface.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/deploy-from-huggingface.png)

点击“从 Hugging Face 部署”后，将显示一个表单，您可以在其中快速查找模型 ID。Hugging Face 上数以百计最受欢迎的开放大语言模型已经准备就绪，提供了经过测试的硬件配置。

![model-selection.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/model-selection.png)

找到想要部署的模型后，选择该模型，Vertex AI 会自动填充所有必要的配置，以便您将模型部署到 Vertex AI 或 GKE 上。通过“在 Hugging Face 上查看”功能，您甚至可以确认选择的模型是否正确。如果您使用的是受限模型，请确保提供您的 Hugging Face 访问令牌，以授权下载模型。

![from-deploy.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/from-deploy.png)

就是这样！从 Vertex AI 模型库直接将模型如 Zephyr Gemma 部署到您的 Google Cloud 账户，只需简单几步。

## 这只是开始

我们很高兴能够与 Google Cloud 合作，让 AI 更加开放和易于访问。无论是从 Hugging Face Hub 开始，还是在 Google Cloud 控制台内，部署开放模型到 Google Cloud 上都变得前所未有的简单。但我们不会止步于此——敬请期待，我们将开启更多在 Google Cloud 上利用开放模型构建 AI 的新体验！