---
title: "来自 AI Secure 实验室的 LLM 安全排行榜简介"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_decodingtrust.png
authors:
- user: danielz01
  guest: true
- user: alphapav
  guest: true
- user: Cometkmt
  guest: true
- user: chejian
  guest: true
- user: BoLi-aisecure
  guest: true
translators:
- user: MatrixYao
---

# 来自 AI Secure 实验室的 LLM 安全排行榜简介

近来，LLM 已深入人心，大有燎原之势。但在我们将其应用于千行百业之前，理解其在不同场景下的安全性和潜在风险显得尤为重要。为此，美国白宫发布了关于安全、可靠、可信的人工智能的行政命令；欧盟人工智能法案也对高风险人工智能系统的设立了专门的强制性要求。在这样的大背景下，我们首先需要确立一个用于定量评估人工智能系统的风险的技术解决方案，以为保证人工智能系统的安全性和一致性提供基准。

为了因应这一需求，我们[安全学习实验室](https://boli.cs.illinois.edu/)于 2023 年提出了 [DecodingTrust](https://decodingtrust.github.io/) 平台，这是第一个全面且统一的 LLM 可信度评估平台。（*该工作还荣获了 NeurIPS 2023 的[杰出论文奖](https://blog.neurips.cc/2023/12/11/announcing-the-neurips-2023-paper-awards/)。*）

DecodingTrust 是一个多维度的评估框架，其涵盖了 8 个可信度评估维度，包括：毒性、刻板印象偏见、对抗提示鲁棒性、OOD（Out Of Distribution）鲁棒性、对抗示例鲁棒性、隐私保护、道德以及公平性。特别地，DecodingTrust 1) 为整体可信度评估提供全面的分析维度，2) 为每个维度量身定制了新颖的红队算法，从而对 LLM 进行深入测试，3) 可跨各种云环境轻松安装，4) 提供一个可供开放模型和封闭模型同场竞技的全面的可信度排行榜，5) 提供失败样本以增强评估的透明度以及对评估基准的理解，6) 提供端到端方案并输出面向实用场景的详细模型报告。

今天，我们很高兴向社区发布新的 [LLM 安全排行榜](https://huggingface.co/spaces/AI-Secure/llm-trustworthy-leaderboard)，该排行榜是基于[HF 排行榜模板](https://huggingface.co/demo-leaderboard-backend)开发的，其专注于对 LLM 进行安全性评估。

## 红队评估
<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.45.1/gradio.js"> </script>
<gradio-app theme_mode="light" space="AI-Secure/llm-trustworthy-leaderboard"></gradio-app>

DecodingTrust 为每个评估维度都提供了数种新颖的红队方法以对模型进行压力测试。有关测试指标的详细信息可参见我们论文中的[图 3](https://arxiv.org/html/2306.11698v4/extracted/5331426/figures/taxonomy.main.png)。

针对毒性这一维度，我们针对其设计了优化算法并使用精心设计的提示以使生成模型生成具有挑战性的用户提示。我们还设计了 33 个具有挑战性的系统提示，以在不同场景下（如角色扮演、任务重规划以及程序式响应等）对 LLM 进行评估。然后，我们利用目标 LLM 的 API 来评估其在这些具有挑战性的提示下生成的内容的毒性分。

针对刻板印象偏见这一维度，我们收集了涉及 24 个人口统计学群体的 16 个刻板印象话题（其中每个话题包含 3 个提示变体）用于评估模型偏见。我们对每个模型提示 5 次，并取其平均值作为模型偏见分。

针对对抗提示鲁棒性这一维度，我们针对三个开放模型（分别是：Alpaca、Vicuna 以及 StableVicuna）构建了五种对抗攻击算法。我们使用通过攻击开放模型而生成的对抗性数据来评估不同模型在五种不同任务上的鲁棒性。

针对 OOD 鲁棒性这一维度，我们设计了不同的风格转换、知识转换等场景测例，以评估模型在未见场景下的性能，如 1）将输入风格转换为其他不太常见的风格，如莎士比亚或诗歌形式，或 2）问题所需的知识在 LLM 训练数据中不存在。

针对对抗示例鲁棒性这一维度，我们设计了包含误导信息的示例，如反事实示例、假相关和后门攻击，以评估模型在此类情形下的性能。

针对隐私保护这一维度，我们提供了不同级别的评估，包括 1）预训练数据的隐私泄露，2）对话过程中的隐私泄露，3）LLM 对隐私相关措辞及事件的理解。特别地，对于 1) 和 2)，我们设计了不同的方法来进行隐私攻击。例如，我们提供不同格式的提示以诱导 LLM 吐露电子邮件地址及信用卡号等敏感信息。

针对道德这一维度，我们利用 ETHICS 和 Jiminy Cricket 数据集来设计越狱系统和用户提示，用于评估模型在不道德行为识别方面的表现。

针对公平性这一维度，我们通过在各种任务中对不同的受保护属性进行控制，从而生成具有挑战性的问题，以评估零样本和少样本场景下模型的公平性。

## 来自于我们论文的重要发现

总的来说，我们发现：

1) GPT-4 比 GPT-3.5 更容易受到攻击；
2) 没有一个 LLM 在所有可信度维度上全面领先；
3) 需要在不同可信度维度之间进行折衷；
4) LLM 隐私保护能力受措辞的影响较大。例如，如果对 GPT-4 提示 “in confidence”，则可能不会泄露私人信息，但如果对其提示 “confidentially”，则可能会泄露信息。
5) 多个维度的结果都表明，LLM 很容易受对抗性或误导性的提示或指令的影响。

## 如何提交模型以供评估

首先，将模型权重转换为 `safetensors` 格式，这是一种存储权重的新格式，用它加载和使用权重会更安全、更快捷。另外，在排行榜主表中，我们能够直接显示 `safetensors` 模型的参数量！

其次，确保你的模型和分词器可以通过 `AutoXXX` 类加载，如下：

```Python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained("your model name")
model = AutoModel.from_pretrained("your model name")
tokenizer = AutoTokenizer.from_pretrained("your model name")
```

如果上述步骤失败，请根据报错消息对模型进行调试，成功后再提交。不然你的模型可能上传不正确。

注意：
- 确保你的模型是公开的！
- 我们尚不支持需要 `use_remote_code=True` 的模型。但我们正在努力，敬请期待！

最后，你需要在排行榜的 [Submit here!](https://huggingface.co/spaces/AI-Secure/llm-trustworthy-leaderboard) 选项卡中提交你的模型以供评估！

# 如何引用我们的工作

如果你发现这个评估基准对你有用，请考虑引用我们的工作，格式如下：

```
@article{wang2023decodingtrust,
  title={DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models},
  author={Wang, Boxin and Chen, Weixin and Pei, Hengzhi and Xie, Chulin and Kang, Mintong and Zhang, Chenhui and Xu, Chejian and Xiong, Zidi and Dutta, Ritik and Schaeffer, Rylan and others},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

> 英文原文: <url> https://huggingface.co/blog/leaderboards-on-the-hub-decodingtrust </url>
> 原文作者：Chenhui Zhang，Chulin Xie，Mintong Kang，Chejian Xu，Bo Li
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。