---
title: "Ryght 在 Hugging Face 专家助力下赋能医疗保健和生命科学之旅" 
thumbnail: /blog/assets/ryght-case-study/thumbnail.jpg
authors:
- user: andrewrreed
- user: johnnybio
  guest: true
  org: RyghtAI
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# Ryght 在 Hugging Face 专家助力下赋能医疗保健和生命科学之旅

> [!NOTE] 本文是 Ryght 团队的客座博文。

## Ryght 是何方神圣？

Ryght 的使命是构建一个专为医疗保健和生命科学领域量身定制的企业级生成式人工智能平台。最近，公司正式公开了 [Ryght 预览版](https://www.ryght.ai/signup?utm_campaign=Preview%20Launch%20April%2016%2C%2024&utm_source=Huggging%20Face%20Blog%20-%20Preview%20Launch%20Sign%20Up) 平台。

当前，生命科学公司不断地从各种不同来源 (实验室数据、电子病历、基因组学、保险索赔、药学、临床等) 收集大量数据，并期望从中获取洞见。但他们分析这些数据的方法已经跟不上数据本身，目前典型的工作模式往往需要一个大型团队来完成从简单查询到开发有用的机器学习模型的所有工作。这一模式已无法满足药物开发、临床试验以及商业活动对可操作知识的巨大需求，更别谈精准医学的兴起所带来的更大的需求了。

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ryght-case-study/click-through.gif" alt="Ryght Laptop" style="width: 90%; height: auto;"><br>
</p>

[Ryght](https://hubs.li/Q02sLGKL0) 的目标是让生命科学专业人士能够快速、安全地从数据中挖掘出他们所需的洞见。为此，其正在构建一个 SaaS 平台，为本专业的人员和组织提供定制的 AI copilot 解决方案，以助力他们对各种复杂数据源进行记录、分析及研究。

Ryght 认识到 AI 领域节奏快速且多变的特点，因此一开始就加入 [Hugging Face 专家支持计划](https://huggingface.co/support)，将 Hugging Face 作为技术咨询合作伙伴。

## 共同克服挑战

> ##### _我们与 Hugging Face 专家支持计划的合作对加快我们生成式人工智能平台的开发起到了至关重要的作用。快速发展的人工智能领域有可能彻底改变我们的行业，而 Hugging Face 的高性能、企业级的文本生成推理 (TGI) 和文本嵌入推理 (TEI) 服务本身就是游戏规则的改写者。 - [Johnny Crupi](https://www.linkedin.com/in/johncrupi/)，[Ryght 首席技术官](http://www.ryght.ai/?utm_campaign=hf&utm_source=hf_blog)_

在着手构建生成式人工智能平台的过程中，Ryght 面临着多重挑战。

### 1. 快速提升团队技能并在多变的环境中随时了解最新情况

随着人工智能和机器学习技术的快速发展，确保团队及时了解最新的技术、工具以及最佳实践至关重要。这一领域的学习曲线呈现出持续陡峭的特点，因此需要齐心协力才能及时跟上。

与 Hugging Face 的人工智能生态系统核心专家团队的合作，有助于 Ryght 跟上本垂直领域的最新发展以及最新模型。通过开放异步的沟通渠道、定期的咨询会以及专题技术研讨会等多种形式，充分地保证了目的的实现。

### 2. 在众多方案中找到最 [经济] 的机器学习方案

人工智能领域充满了创新，催生了大量的工具、库、模型及方法。对于像 Ryght 这样的初创公司来说，必须消除这种噪声并确定哪些机器学习策略最适合生命科学这一独特场景。这不仅需要了解当前的技术水平，还需要对技术在未来的相关性和可扩展性有深刻的洞见。

Hugging Face 作为 Ryght 技术团队的合作伙伴，在解决方案设计、概念验证开发和生产工作负载优化全过程中提供了有力的协助，包括: 针对应用场景推荐最适合 Ryght 需求的库、框架和模型，并提供了如何使用这些软件和模型的示例。这些指导最终简化了决策过程并缩短了开发时间。

### 3. 开发专注于安全性、隐私性及灵活性的高性能解决方案

鉴于其目标是企业级的解决方案，因此 Ryght 把安全、隐私和可治理性放在最重要的位置。因此在设计方案架构时，需要提供支持各种大语言模型 (LLM) 的灵活性，这是生命科学领域内容生成和查询处理系统的关键诉求。

基于对开源社区的快速创新，特别是医学 LLM 创新的理解，其最终采用了“即插即用”的 LLM 架构。这种设计使其能够在新 LLM 出现时能无缝地评估并集成它们。

在 Ryght 的平台中，每个 LLM 均可注册并链接至一个或多个特定于客户的推理端点。这种设计不仅可以保护各客户的连接，还提供了在不同 LLM 之间切换的能力，提供了很好的灵活性。Ryght 通过采用 Hugging Face 的 [文本生成推理 (TGI)](https://huggingface.co/docs/text-generation-inference/index) 和 [推理端点](https://huggingface.co/inference-endpoints/dedicate) 实现了该设计。

除了 TGI 之外，Ryght 还将 [文本嵌入推理 (TEI)](https://huggingface.co/docs/text-embeddings-inference/en/index) 集成到其 ML 平台中。使用 TEI 和开源嵌入模型提供服务，与仅依赖私有嵌入服务相比，可以使 Ryght 能够享受更快的推理速度、免去对速率限制的担忧，并得到可以为自己的微调模型提供服务的灵活性，而微调模型可以更好地满足生命科学领域的独特要求。

为了同时满足多个客户的需求，系统需要能处理大量并发请求，同时保持低延迟。因此，Ryght 的嵌入和推理服务不仅仅是简单的模型调用，还需要支持包括组批、排队和跨 GPU 分布式模型处理等高级特性。这些特性对于避免性能瓶颈并确保用户不会遇到延迟，从而保持最佳的系统响应时间至关重要。

## 总结

Ryght 与 Hugging Face 在 ML 服务上的战略合作伙伴关系以及深度集成凸显了其致力于在医疗保健和生命科学领域提供尖端解决方案的承诺。通过采用灵活、安全和可扩展的架构，其确保自己的平台始终处于创新前沿，为客户提供无与伦比的服务和专业知识，以应对现代医疗领域的复杂性。

[Ryght 预览版](https://hubs.li/Q02sLFl_0) 现已作为一个可轻松上手的、免费、安全的平台向生命科学知识工作者公开，欢迎大家使用。Ryght 的 copilot 库包含各种工具，可加速信息检索、复杂非结构化数据的综合及结构化，以及文档构建等任务，把之前需要数周才能完成的工作缩短至数天或数小时。如你对定制方案及合作方案有兴趣，请联系其 [AI 专家团队](https://hubs.li/Q02sLG9V0)，以讨论企业级 Ryght 服务。

如果你有兴趣了解有关 Hugging Face 专家支持计划的更多信息，请 [通过此处](https://huggingface.co/contact/sales?from=support) 联系我们，我们将联系你讨论你的需求！