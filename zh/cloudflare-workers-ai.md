---
title: "为 Hugging Face 用户带来无服务器 GPU 推理服务"
thumbnail: /blog/assets/cloudflare-workers-ai/thumbnail.jpg
authors:
- user: philschmid
- user: jeffboudier
- user: rita3ko
  guest: true
- user: nkothariCF
  guest: true
translators:
- user: chenglu
---

# 为 Hugging Face 用户带来无服务器 GPU 推理服务

今天，我们非常兴奋地宣布 **部署到 Cloudflare Workers AI** 功能正式上线，这是 Hugging Face Hub 平台上的一项新服务，它使得通过 Cloudflare 边缘数据中心部署的先进 GPU、轻松使用开放模型作为无服务器 API 成为可能。

从今天开始，我们将把 Hugging Face 上一些最受欢迎的开放模型整合到 Cloudflare Workers AI 中，这一切都得益于我们的生产环境部署的解决方案，例如 [文本生成推理 (TGI)](https://github.com/huggingface/text-generation-inference/)。

通过 **部署到 Cloudflare Workers AI** 服务，开发者可以在无需管理 GPU 基础架构和服务器的情况下，以极低的运营成本构建强大的生成式 AI（Generative AI）应用，你只需 **为实际计算消耗付费，无需为闲置资源支付费用**。

## 开发者的生成式 AI 工具

这项新服务基于我们去年与 Cloudflare 共同宣布的 [战略合作伙伴关系](https://blog.cloudflare.com/zh-cn/partnering-with-hugging-face-deploying-ai-easier-affordable-zh-cn/)——简化开放生成式 AI 模型的访问与部署过程。开发者和机构们共同面临着一个主要的问题——GPU 资源稀缺及部署服务器的固定成本。

Cloudflare Workers AI 上的部署提供了一个简便、低成本的解决方案，通过 [按请求计费模式](https://developers.cloudflare.com/workers-ai/platform/pricing)，为这些挑战提出了一个无服务器访问、运行的 Hugging Face 模型的解决方案。

举个具体例子，假设你开发了一个 RAG 应用，每天大约处理 1000 个请求，每个请求包含 1000 个 Token 输入和 100 个 Token 输出，使用的是 Meta Llama 2 7B 模型。这样的 LLM 推理生产成本约为每天 1 美元。

![Cloudflare 价格页面](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/cloudflare-workers-ai/pricing.png)

> 我们很高兴能够这么快地实现这一集成。将 Cloudflare 全球网络中的无服务器 GPU 能力，与 Hugging Face 上最流行的开源模型结合起来，将为我们全球社区带来大量激动人心的创新。
>
> John Graham-Cumming，Cloudflare 首席技术官

## 使用方法

在 Cloudflare Workers AI 上使用 Hugging Face 模型非常简单。下面是一个如何在 Nous Research 最新模型 Mistral 7B 上使用 Hermes 2 Pro 的逐步指南。

你可以在 [Cloudflare Collection](https://huggingface.co/collections/Cloudflare/hf-curated-models-available-on-workers-ai-66036e7ad5064318b3e45db6) 中找到所有可用的模型。

> 注意：你需要拥有 [Cloudflare 账户](https://developers.cloudflare.com/fundamentals/setup/find-account-and-zone-ids/) 和 [API 令牌](https://dash.cloudflare.com/profile/api-tokens)。

你可以在所有支持的模型页面上找到“部署到 Cloudflare”的选项，包括如 Llama、Gemma 或 Mistral 等模型。

![model card](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/cloudflare-workers-ai/model-card.jpg)

打开“部署”菜单，选择“Cloudflare Workers AI”，这将打开一个包含如何使用此模型和发送请求指南的界面。

> 注意：如果你希望使用的模型没有“Cloudflare Workers AI”选项，意味着它目前不支持。我们正与 Cloudflare 合作扩展模型的可用性。你可以通过 [api-enterprise@huggingface.co](mailto:api-enterprise@huggingface.co) 联系我们，提交你的请求。

![推理代码](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/cloudflare-workers-ai/modal.jpg)

当前有两种方式可以使用此集成：通过 [Workers AI REST API](https://developers.cloudflare.com/workers-ai/get-started/rest-api/) 或直接在 Workers 中使用 [Cloudflare AI SDK](https://developers.cloudflare.com/workers-ai/get-started/workers-wrangler/#1-create-a-worker-project)。选择你偏好的方式并将代码复制到你的环境中。当使用 REST API 时，需要确保已定义 <code>[ACCOUNT_ID](https://developers.cloudflare.com/fundamentals/setup/find-account-and-zone-ids/)</code> 和 <code>[API_TOKEN](https://dash.cloudflare.com/profile/api-tokens)</code> 变量。

就这样！现在你可以开始向托管在 Cloudflare Workers AI 上的 Hugging Face 模型发送请求。请确保使用模型所期望的正确提示与模板。

## 我们的旅程刚刚开始

我们很高兴能与 Cloudflare 合作，让 AI 技术更加易于开发者访问。我们将与 Cloudflare 团队合作，为你带来更多模型和体验！