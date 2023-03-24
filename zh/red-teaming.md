---
title: "红队（ red-teaming )大语言模型" 
thumbnail: /blog/assets/red-teaming/thumbnail.png
authors:
- user: nazneen
- user: natolambert
- user: lewtun
---

# 红队（ red-teaming )大语言模型

<!-- {blog_metadata} -->
<!-- {authors} -->

# 红队( red-teaming )大语言模型
在巨量文本数据下训练的大语言模型非常擅长生成现实文本。但是，这些模型通常会显现出一些不良行为像泄露个人信息（比如社会保险号）和生成错误信息，偏置，仇恨或有毒内容。举个例子， 众所周知，GPT3 的早期版本就表现出性别歧视（如下图）与[仇恨穆斯林言论](https://dl.acm.org/doi/abs/10.1145/3461702.3462624)的情况。

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/gpt3.png"/>
</p>

一旦我们在使用大语言模型时发现了这种不良结果，我们就可以制定一些策略来远离它们，像[生成歧视者指导序列生成（ GEDI ）](https://arxiv.org/pdf/2009.06367.pdf)或[插入和播放语言模型（ PPLM ）](https://arxiv.org/pdf/1912.02164.pdf)都是用来指导 GPT3 生成的。以下是使用相同提示（ Prompt ）的示例，但使用 GEDI 控制 GPT3 生成。

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/gedi.png"/>
</p>

即使是最近的 GPT3 版本，也会在提示（ prompt ）注入攻击时产生类似的令人反感的内容，这变成了[这篇博客](https://simonwillison.net/2022/Sep/12/prompt-injection/)中讨论的下游应用程序的安全问题。


**红队** *是一种用于引出模型不良行为漏洞的评估形式。* 越狱是另一个红队术语，用来表示操控冲破大语言模型限制。在 2016 年发布的 [ 微软聊天机器人 Tay](https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/) 和最近的  [必应聊天机器人 Sydney](https://www.nytimes.com/2023/02/16/technology/bing-chatbot-transcript.html) 是真实世界中反应缺乏用红队攻击对基础 ML 模型进行评估而发生的灾难。红队攻击的最初想法起源于军队中对抗模拟和战争游戏。

红队语言模型的目标是制作一个提示（ prompt ），该提示会触发模型生成有害内容。红队和同样知名的评估语言模型*对抗攻击*分享一些相似之处与差异。相似之处在于红队和对抗攻击共享相同的目标，即“攻击”或“欺骗”模型，以生成在现实世界中不想要的内容。但是对抗攻击很难让人理解，举例来说，通过将字符串 “aaabbbcc”前缀到每个提示中，它会恶化模型性能。[Wallace等人](https://arxiv.org/abs/1908.07125)讨论了对各种 NLP 分类和生成任务的许多攻击的例子。在另一方面，红队的提示看起来更正常，像自然语言的提示。


红队可以揭露模型的局限性，包括引起用户不适或者暴力，不合法的恶意内容。 红队(就像对抗攻击) 的输出通常会被用来训练模型去减少有害内容或远离不想要的内容。

由于红队需要创造性地思考可能的模型失败，这就造成了对于巨大搜索空间的资源紧张。这里的一个临时方法是对大语言模型增加一个分类器去预测输入的提示（ prompt ）中是否含导致生成恶意内容的话题或短语，如果含有则生成相关回应。这种策略使得模型非常谨慎，但极大的限制了模型并且经常触发。所以在模型有帮助（遵循指令）与无害（尽可能少的产生有害内容）之间存在一个紧张关系。红队在这时就显得非常有用了。

红队攻击可以是人力循环或者正在测试另一个语言模型有害输出的语言模型。提出针对安全和对齐方式进行微调的模型（例如通过 RLHF 或 SFT ）的模型提示，需要以*角色扮演攻击*的形式进行创造性的思考，其中大语言模型被指示表现为恶意角色在[ Ganguli et al., ‘22](https://arxiv.org/pdf/2209.07858.pdf)中。 用代码而不是自然语言指示模型同样也可以揭露模型的学习的一些偏置。就像如下例子。

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jb1.png"/>
</p>
<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jb0.png"/>
</p>
<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jb2.png"/>
</p>
<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jb3.png"/>
</p>

查看[此](https://twitter.com/spiantado/status/1599462375887114240) 推文获取更多示例。

这里列出了在 ChatGPT 刺激大语言模型进行越狱的列表。

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jailbreak.png"/>
</p>

红队大语言模型依旧是一个新的研究领域但是上述提到的策略依旧可以在越狱中正常运行，并且有助于部署机器学习的产品。随着这些模型的涌现能力变得更加强大，开发可以不断适应的红队方法将变得至关重要。一些需要进行红队攻击的最佳实践包括模拟寻求权力行为的方案（例如：资源），说服人们（例如：伤害自己或他人），具有医学输出的代理（例如：通过 API 在线订购化学药品）。我们将这种可能性和物理后果的可能性称为*关键威胁情景*。

在评估大语言模型中恶意行为的警示中，我们不知道他们的能力，毕竟他们不是显示训练去展示这种能力的（涌现能力）。所以，实际了解大语言模型的能力的唯一方法是，当它们变得更强大，可以模拟所有可能导致竞争结果并在每种情况下评估模型的行为的所有可能场景。这意味着我们的模型的安全行为与我们的红队方法的强度相关联。

针对这一持续的红队的挑战，这里在数据集和最佳实践（包括学术，工业和政府实体）上进行了多组织合作的激励措施。共享信息的结构化过程可以使较小的实体在模型发布前进行红队攻击，从而使整个用户体验更安全。

**红队的开放数据集：**

1. Meta 的 [机器人对抗对话数据集](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/bot_adversarial_dialogue)
2. Anthropic 的[红队尝试](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/red-team-attempts)
3. AI2 的 [RealToxicityPrompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts)

**在红队大语言模型找过去的工作** (在 [Anthropic's Ganguli et al. 2022](https://arxiv.org/abs/2209.07858) 和 [Perez et al. 2022](https://arxiv.org/abs/2202.03286) 两篇文章中)

1. 用有帮助的，忠实的，无害的行为在红队攻击中进行少量提示学习并*不*比单纯的语言模型困难。
2. 攻击成功率与缩放模型大小没有明确的关系，除了 RLHF 模型在缩放时更难进行红队攻击 。
3. 模型可能会通过回避表现的无害，在有帮助和无害之间存在权衡。
4. 人类在构成成功攻击方面保持总体一致性。
5. 成功率的分布在危害类别的类别中有所不同，而非暴力的成功率具有更高的成功率。
6. 众包红色团队会导致模板 Y 提示（例如：“给出一个以 X 开头的平均单词”），使其变得多余。
**未来方向:**

1. 没有用于代码生成的开源红队数据集，它试图通过代码越狱模型，例如生成实现 DDOS 或后门攻击的程序。
2. 评估回避和有帮助之间的权衡。
3. 为关键威胁场景设计和实施 红队大语言模型的策略。
4. 红队可能是资源密集的，无论是计算还是人力资源，因此将从共享策略，开源数据集以及可能的合作中获得更大的成功机会，从而受益。

这些局限性和未来的方向清楚地表明，红队是现代大语言模型工作流程中未经探索和关键的组成部分。这篇文章是对大语言模型研究人员和 Huggingface 开发人员社区的号召，以协作这些努力，实现安全和友好的世界:)


*致谢:* 感谢 [Yacine Jernite](https://huggingface.co/yjernite) 的在正确使用本篇博文中术语的有用建议。

> 原文：https://huggingface.co/blog/red-teaming 
> 译者：innovation64(李洋)
