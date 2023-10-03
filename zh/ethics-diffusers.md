---
title: "开发 Diffusers 库的道德行为指南" 
thumbnail: /blog/assets/ethics-diffusers/thumbnail.png
authors:
- user: giadap
translators:
- user: innovation64
- user: zhongdongy
  proofreader: true
---

# 开发 Diffusers 库的道德行为指南


我们正在努力让我们每次发布的库更加负责！

我们很荣幸宣布我们发布了 [道德守则](https://huggingface.co/docs/diffusers/main/en/conceptual/ethical_guidelines)，并将作为一部分其放入 [ Diffusers 库的说明文档](https://huggingface.co/docs/diffusers/main/en/index)。

由于扩散模型在现实世界上的实际应用例子会对社会造成潜在的负面影响，该守则旨在引导对于社区做出贡献的 Diffusers 库维护者进行技术决策。我们希望对于我们的决策进行更加透明，尤其是，我们想确认一些价值观来指导决策。

我们将道德准则作为一个引导价值，做出具体行动，然后持续适应新的条件的循环过程。基于此，我们致力于随着时间去不断更正我们的价值准则，不断跟进 Diffusers 项目的发展，并从社区持续收集反馈，使得准则始终保持有效。

# 道德守则

- **透明**: 我们致力于在管理 PR、向用户解释我们的选择以及做出技术决策方面保持透明。
- **一致性**: 我们致力于保证我们的用户在项目管理中得到同等程度的关注，保持技术上的稳定和一致。
- **简单性**: 为了让 Diffusers 库易于使用和利用，我们致力于保持项目目标的精简和连贯性。
- **可访问性**: Diffusers 项目帮助更多贡献者降低进入门槛即便没有专业技术也可以运行项目。这样做使得社区更容易获得研究成果。
- **可再现性**: 我们的目标是在使用 Diffusers 库时，使上游代码、模型和数据集的可再现性保持透明。
- **责任**: 作为一个社区，通过团队合作，我们通过预测和减轻该技术的潜在风险和危险来对我们的用户承担集体责任。

# 安全特性和机制

此外，我们提供了一个暂不全面的并希望不断扩展的列表，该列表是关于 Hugging Face 团队和更广泛的社区的实施的安全功能和机制。

- **[社区选项](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)**: 它使社区能够讨论并更好地协作项目。
- **标签功能**: 仓库的作者可以将他们的内容标记为“不适合所有人”
- **偏差探索和评估**: Hugging Face 团队提供了一个 [Space](https://huggingface.co/spaces/society-ethics/DiffusionBiasExplorer) 以交互方式演示 Stable Diffusion 和 DALL-E 中的偏差。从这个意义上说，我们支持和鼓励有偏差的探索和评估。
- **鼓励安全部署**

  - **[Safe Stable Diffusion](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion_safe)**: 它缓解了众所周知的问题，像 Stable Diffusion，在未经过滤的，网络抓取的数据集上训练的模型往往会遭受不当的退化。相关论文: [Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models](https://arxiv.org/abs/2211.05105).
  - **在 Hub 上分阶段发布**: 特别在敏感的情况下，应限制对某些仓库的访问。这是发布阶段的一个中间步骤，允许仓库的作者对其使用有更多的控制权限。

- **许可**: [OpenRAILs](https://huggingface.co/blog/open_rail), 是一种新型许可，可让我们确保自由访问，同时拥有一组限制，以确保更多负责任的用途。