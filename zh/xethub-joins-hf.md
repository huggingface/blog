---
title: "XetHub 加入 Hugging Face!"
thumbnail: /blog/assets/xethub-joins-hf/thumbnail.png
authors:
- user: yuchenglow
  org: xet-team
- user: julien-c
translators:
- user: AdinaY
---

# XetHub 加入 Hugging Face！

我们非常激动地正式宣布，Hugging Face 已收购 XetHub 🔥

XetHub 是一家位于西雅图的公司，由 Yucheng Low、Ajit Banerjee 和 Rajat Arya 创立，他们之前在 Apple 工作，构建和扩展了 Apple 的内部机器学习基础设施。XetHub 的使命是为 AI 开发提供软件工程的最佳实践。XetHub 开发了技术，能够使 Git 扩展到 TB 级别的存储库，并使团队能够探索、理解和共同处理大型不断变化的数据集和模型。不久之后，他们加入了一支由 12 名才华横溢的团队成员组成的团队。你可以在他们的新组织页面关注他们：[hf.co/xet-team](https://huggingface.co/xet-team)。

## 我们在 Hugging Face 的共同目标

> XetHub 团队将帮助我们通过切换到我们自己的、更好的 LFS 版本作为 Hub 存储库的存储后端，解锁 Hugging Face 数据集和模型的未来五年增长。
>
> —— Julien Chaumond, Hugging Face CTO

早在 2020 年，当我们构建第一个 Hugging Face Hub 版本时，我们决定将其构建在 Git LFS 之上，因为它相当知名，并且是启动 Hub 使用的合理选择。

然而，我们当时就知道，某个时候我们会希望切换到我们自己的、更优化的存储和版本控制后端。Git LFS——即使它代表的是大文件存储——也从未适合我们在 AI 中处理的那种类型的大文件，这些文件不仅大，而且非常大 😃。

## 未来的示例用例 🔥 – 这将如何在 Hub 上实现

假设你有一个 10GB 的 Parquet 文件。你添加了一行。今天你需要重新上传 10GB。使用 XetHub 的分块文件和重复数据删除技术，你只需要重新上传包含新行的几个块。

另一个例子是 GGUF 模型文件：假设 [@bartowski](https://huggingface.co/bartowski) 想要更新 Llama 3.1 405B 存储库的 GGUF 头部中的一个元数据值。将来，bartowski 只需重新上传几千字节的单个块，使这个过程更加高效 🔥。

随着该领域在未来几个月内转向万亿参数模型（感谢 Maxime Labonne 提供新的 [BigLlama-3.1-1T](https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct) 🤯），我们希望这种新技术将解锁社区和企业内部的新规模。

最后，随着大数据集和大模型的

出现，协作也面临挑战。团队如何共同处理大型数据、模型和代码？用户如何理解他们的数据和模型是如何演变的？我们将努力找到更好的解决方案来回答这些问题。

## Hub 存储库的有趣当前统计数据 🤯🤯

- 存储库数量：130 万个模型，45 万个数据集，68 万个空间
- 累计总大小：LFS 中存储了 12PB（2.8 亿个文件）/ git（非 LFS）中存储了 7.3TB
- Hub 每日请求次数：10 亿次
- Cloudfront 每日带宽：6PB 🤯

## 来自 [@ylow](https://huggingface.co/yuchenglow) 的个人话语

我在 AI/ML 领域工作了 15 年以上，见证了深度学习如何慢慢接管视觉、语音、文本，甚至越来越多的每个数据领域。

我严重低估了数据的力量。几年前看起来不可能的任务（如图像生成），实际上通过数量级更多的数据和能够吸收这些数据的模型变得可能。从历史上看，这是一再重复的机器学习历史教训。

自从我的博士学位以来，我一直在数据领域工作。首先在初创公司（GraphLab/Dato/Turi）中，我使结构化数据和机器学习算法在单机上扩展。之后被 Apple 收购，我致力于将 AI 数据管理扩展到超过 100PB，支持数十个内部团队每年发布数百个功能。2021 年，与我的联合创始人们一起，在 Madrona 和其他天使投资者的支持下，创立了 XetHub，将我们在实现大规模协作方面的经验带给全世界。

XetHub 的目标是使 ML 团队像软件团队一样运作，通过将 Git 文件存储扩展到 TB 级别，无缝实现实验和可重复性，并提供可视化功能来理解数据集和模型的演变。

我和整个 XetHub 团队都非常高兴能够加入 Hugging Face，并继续我们的使命，通过将 XetHub 技术整合到 Hub 中，使 AI 协作和开发更加容易，并向全球最大的 ML 社区发布这些功能！

## 最后，我们的基础设施团队正在招聘 👯

如果你喜欢这些主题，并希望为开源 AI 运动构建和扩展协作平台，请联系我们！
