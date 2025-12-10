---
title: "构建开放的未来——我们与 Google Cloud 的全新合作伙伴关系"
thumbnail: /blog/assets/google-cloud/google-cloud-thumbnail.png
authors:
- user: jeffboudier
- user: pagezyhf
translators:
- user: chenglu
---

# 构建开放的未来——我们与 Google Cloud 的全新合作伙伴关系

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud/google%20cloud%20blogpost%20title.png">

今天，我们很高兴宣布与 Google Cloud 建立更深入的合作伙伴关系，让更多企业能够使用开源模型构建自己的人工智能。

Hugging Face 的 Jeff Boudier 表示：
*“Google 一直是开放式 AI 领域的重要推动者——从最初的 Transformer 到如今的 Gemma 模型。 我相信未来每一家企业都将能构建并定制属于自己的 AI。通过这次战略合作，我们让这一切在 Google Cloud 上变得更加简单。”*

Google Cloud 产品管理高级总监 Ryan J. Salva 表示：
*“Hugging Face 是推动全球大中小企业访问、使用并定制超过 200 万个开源模型的核心力量，我们也自豪地为社区贡献了超过 1000 个模型。”*
*“携手合作，我们将让 Google Cloud 成为使用开源模型构建 AI 的最佳平台。”*


## 面向 Google Cloud 客户的合作

Google Cloud 客户已经在其领先的 AI 服务中使用来自 Hugging Face 的开源模型。在 Vertex AI 中，最受欢迎的开源模型只需几次点击即可在 Model Garden 中部署。
对于希望更灵活掌控 AI 基础设施的客户，也可在 GKE AI/ML 中找到类似的模型库，或使用由 Hugging Face 维护的预配置环境。客户还可以通过 Cloud Run GPU 运行推理任务，实现无服务器的开源模型部署。

我们与 Google Cloud 的共同目标是：充分利用各项服务的独特能力，为客户提供无缝、灵活的使用体验。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud/google-cloud-partnership.png">


## 通往开源模型的高速通道——为 Google Cloud 用户打造的快捷体验

过去三年中，Google Cloud 用户对 Hugging Face 的使用量增长了 10 倍，如今每月的模型下载量已达数十 PB，处理请求高达数十亿次。

为了让 Google Cloud 客户在使用 Hugging Face 的模型和数据集时获得最佳体验，我们正携手打造一个全新的 CDN Gateway。该网关基于 Hugging Face 的 Xet 优化存储与数据传输技术，以及 Google Cloud 的先进存储与网络能力构建。

通过这个 CDN Gateway，Hugging Face 的模型和数据集将直接缓存在 Google Cloud 上，大幅减少下载时间，并提升 Google Cloud 用户在模型供应链方面的稳定性和可靠性。
无论你是在使用 Vertex、GKE、Cloud Run，还是在 Compute Engine 中自建 AI 系统，都能享受到更快的响应速度与更简化的模型管理。


## 面向 Hugging Face 用户的合作

Hugging Face 的 [Inference Endpoints](https://endpoints.huggingface.co/) 是最快捷的模型部署方式，只需几次点击即可完成。从这次合作开始，我们将把 Google Cloud 的高性能与高性价比带给 Hugging Face 用户，首先体现在 Inference Endpoints 上——未来将推出更多实例类型，并带来价格下调！

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud/inference-endpoints-google-cloud.png">

我们将确保这一系列产品与技术合作成果能惠及 Hugging Face 上超过 1000 万名 AI 构建者。
从模型页面直接部署到 Vertex Model Garden 或 GKE 只需几步操作；在 Hugging Face 企业账户中私有托管的模型，也能像使用公共模型一样方便、安全。

Google 的 TPU（定制 AI 加速芯片）如今已发展到第七代，性能和软件支持持续提升。我们希望 Hugging Face 用户能够充分利用当前和未来的 TPU，在构建开源模型时享受到与 GPU 同样便捷的体验。
得益于库的原生支持，TPU 的使用将比以往更简单。

此外，这次合作还将让 Hugging Face 借助 Google 在安全技术方面的领先优势，使平台上数百万个开源模型更安全。
依托 [VirusTotal](https://www.virustotal.com/gui/home/upload)、[Google Threat Intelligence](https://cloud.google.com/security/products/threat-intelligence) 和 [Mandiant](https://www.mandiant.com/) 的技术支持，我们将共同保障 Hugging Face Hub 上的模型、数据集和 Spaces 的安全。


## 共同构建开放的 AI 未来

我们期待一个未来：每家企业都能用开源模型构建属于自己的 AI，并在安全可控的基础设施上进行部署。
我们非常期待通过与 Google Cloud 的深入合作，让这一愿景加速实现——无论你使用的是 Vertex AI Model Garden、Google Kubernetes Engine、Cloud Run，还是 Hugging Face Inference Endpoints。

有什么想让我们在这次合作中改进或新增的内容？欢迎在评论区告诉我们！

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud/mcface-billion-model-served-compressed.png">
