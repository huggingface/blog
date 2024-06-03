---
title: "Hugging Face 与 Wiz Research 合作提高人工智能安全性"
thumbnail: /blog/assets/wiz_security/security.png
authors:
- user: JJoe206
- user: GuillaumeSalouHF
- user: michellehbn
- user: XciD
- user: mcpotato
- user: Narsil
- user: julien-c
translators:
- user: xiaodouzi
- user: zhongdongy
  proofreader: true
---

# Hugging Face 与 Wiz Research 合作提高人工智能安全性

我们很高兴地宣布，我们正在与 Wiz 合作，目标是提高我们平台和整个 AI/ML 生态系统的安全性。

Wiz 研究人员 [与 Hugging Face 就我们平台的安全性进行合作并分享了他们的发现](https://www.wiz.io/blog/wiz-and-hugging-face-address-risks-to-ai-infrastruct)。 Wiz 是一家云安全公司，帮助客户以安全的方式构建和维护软件。 随着这项研究的发布，我们将借此机会重点介绍一些相关的 Hugging Face 安全改进。

Hugging Face 最近集成了 Wiz 进行漏洞管理，这是一个持续主动的流程，可确保我们的平台免受安全漏洞的影响。 此外，我们还使用 Wiz 进行云安全态势管理 (CSPM)，它使我们能够安全地配置云环境并进行监控以确保其安全。

我们最喜欢的 Wiz 功能之一是从存储到计算再到网络的漏洞的整体视图。 我们运行多个 Kubernetes (k8s) 集群，并拥有跨多个区域和云提供商的资源，因此在单个位置拥有包含每个漏洞的完整上下文图的中央报告非常有帮助。 我们还构建了他们的工具以自动修复我们产品中检测到的问题，特别是在 Spaces 中。

在联合工作的过程中，Wiz 的安全研究团队通过使用 pickle 在系统内运行任意代码，识别出了我们沙箱计算环境的不足之处。在阅读此博客和 Wiz 的安全研究报告时，请记住，我们已经解决了与该漏洞相关的所有问题，并将继续在威胁检测和事件响应过程中保持警惕。

## Hugging Face 安全

在 Hugging Face，我们非常重视安全性。随着人工智能的快速发展，新的威胁向量似乎每天都会出现。即使 Hugging Face 宣布了与技术领域一些最大名字的多项合作伙伴关系和业务关系，我们仍然致力于让我们的用户和 AI 社区能够负责任地实验和操作 AI/ML 系统和技术。我们致力于保障我们的平台安全，并推动 AI/ML 的民主化，使社区能够贡献力量并成为这一将影响我们所有人的范式转变的一部分。我们撰写这篇博客，重申我们保护用户和客户免受安全威胁的承诺。下面我们还将讨论 Hugging Face 关于支持有争议的 pickle 文件的理念，并探讨远离 pickle 格式的共同责任。

在不久的将来，还会有许多令人兴奋的安全改进和公告。这些出版物不仅会讨论 Hugging Face 平台社区面临的安全风险，还会涵盖 AI 的系统性安全风险以及最佳缓解实践。我们始终致力于保障我们的产品、基础设施和 AI 社区的安全，请关注后续的安全博客文章和白皮书。

## 面向社区的开源安全协作和工具

我们高度重视与社区的透明度和合作，这包括参与漏洞的识别和披露、共同解决安全问题以及开发安全工具。以下是通过合作实现的安全成果示例，这些成果有助于整个 AI 社区降低安全风险:

- Picklescan 是与微软合作开发的; 该项目由 Matthieu Maitre 发起，由于我们内部也有一个相同工具的版本，因此我们联手并为 Picklescan 做出了贡献。如果您想了解更多关于其工作原理的信息，请参考以下文档页面: https://huggingface.co/docs/hub/en/security-pickle

- Safetensors 是由 Nicolas Patry 开发的一种比 pickle 文件更安全的替代方案。Safetensors 在与 EuletherAI 和 Stability AI 的合作项目中，由 Trail of Bits 进行了审核。

   https://huggingface.co/docs/safetensors/en/index

- 我们有一个强大的漏洞赏金计划，吸引了来自世界各地的众多出色研究人员。识别出安全漏洞的研究人员可以通过 security@huggingface.co 咨询加入我们的计划。

- 恶意软件扫描: https://huggingface.co/docs/hub/en/security-malware

- 隐私扫描: 请访问以下链接了解更多信息: https://huggingface.co/docs/hub/security-secrets

- 如前所述，我们还与 Wiz 合作降低平台安全风险。

- 我们正在启动一系列安全出版物，以解决 AI/ML 社区面临的安全问题。

## 开源 AI/ML 用户的安全最佳实践

- AI/ML 引入了新的攻击向量，但对于许多这些攻击，其缓解措施早已存在并为人所知。安全专业人员应确保对 AI 资源和模型应用相关的安全控制。此外，以下是一些在使用开源软件和模型时的资源和最佳实践:
- 了解贡献者: 仅使用来自可信来源的模型并注意提交签名。 https://huggingface.co/docs/hub/en/security-gpg
- 不要在生产环境中使用 pickle 文件
- 使用 Safetensors: https://huggingface.co/docs/safetensors/en/index
- 回顾 OWASP 前 10 名: https://owasp.org/www-project-top-ten/
- 在您的 Hugging Face 帐户上启用 MFA
- 建立一个安全开发生命周期，包括由具有适当安全培训的安全专业人员或工程师进行代码审查。
- 在非生产和虚拟化的测试/开发环境中测试模型。

## Pickle 文件——不容忽视的安全隐患

Pickle 文件一直是 Wiz 的研究核心以及近期安全研究人员关于 Hugging Face 的其他出版物的关注点。Pickle 文件长期以来被认为存在安全风险，欲了解更多信息，请参阅我们的文档文件: https://huggingface.co/docs/hub/en/security-pickle

尽管这些已知的安全缺陷存在，AI/ML 社区仍然经常使用 pickle 文件 (或类似容易被利用的格式)。其中许多使用案例风险较低或仅用于测试目的，使得 pickle 文件的熟悉性和易用性比安全的替代方案更具吸引力。

作为开源人工智能平台，我们有以下选择:

- 完全禁止 pickle 文件
- 对 pickle 文件不执行任何操作
- 找到一个中间立场，既允许使用 pickle，又可以合理、切实地减轻与 pickle 文件相关的风险

我们目前选择了第三个选项，即折中的方案。这一选择对我们的工程和安全团队来说是一种负担，但我们已投入大量努力来降低风险，同时允许 AI 社区使用他们选择的工具。我们针对 pickle 相关风险实施的一些关键缓解措施包括:

- 创建概述风险的清晰文档
- 开发自动扫描工具
- 使用扫描工具和标记具有安全漏洞的模型并发出明确的警告
- 我们甚至提供了一个安全的解决方案来代替 pickle (Safetensors)
- 我们还将 Safetensors 设为我们平台上的一等公民，以保护可能不了解风险的社区成员
- 除了上述内容之外，我们还必须显着细分和增强使用模型的区域的安全性，以解决其中潜在的漏洞

我们打算继续在保护和保障 AI 社区方面保持领先地位。我们的一部分工作将是监控和应对与 pickle 文件相关的风险。虽然逐步停止对 pickle 的支持也不排除在外，但我们会尽力平衡此类决定对社区的影响。

需要注意的是，上游的开源社区以及大型科技和安全公司在贡献解决方案方面基本上保持沉默，留下 Hugging Face 独自定义理念，并大量投资于开发和实施缓解措施，以确保解决方案既可接受又可行。

## 结束语

我在撰写这篇博客文章时，与 Safetensors 的创建者 Nicolas Patry 进行了广泛交流，他要求我向 AI 开源社区和 AI 爱好者发出行动号召:

- 主动开始用 Safetensors 替换您的 pickle 文件。如前所述，pickle 包含固有的安全缺陷，并且可能在不久的将来不再受支持。
- 继续向您喜欢的库的上游提交关于安全性的议题/PR，以尽可能推动上游的安全默认设置。

AI 行业正在迅速变化，不断有新的攻击向量和漏洞被发现。Hugging Face 拥有独一无二的社区，我们与大家紧密合作，以帮助我们维护一个安全的平台。

请记住，通过适当的渠道负责任地披露安全漏洞/错误，以避免潜在的法律责任和违法行为。

想加入讨论吗？请通过 security@huggingface.co 联系我们，或者在 LinkedIn/Twitter 上关注我们。

---

> 英文原文: <url>https://hf.co/blog/hugging-face-wiz-security-blog</url>
>
> 原文作者: Josef Fukano, Guillaume Salou, Michelle Habonneau, Adrien, Luc Georges, Nicolas Patry, Julien Chaumond
>
> 译者: xiaodouzi