---
title: "ChatGPT 背后的“功臣”——RLHF 技术详解" 
thumbnail: /blog/assets/120_rlhf/thumbnail.png
authors:
- user: natolambert
- user: LouisCastricato
  guest: true
- user: lvwerra
- user: Dahoas
  guest: true
translators:
- user: hell0w0r1d
- user: inferjay
  proofreader: true
---

# ChatGPT 背后的“功臣”——RLHF 技术详解


OpenAI 推出的 ChatGPT 对话模型掀起了新的 AI 热潮，它面对多种多样的问题对答如流，似乎已经打破了机器和人的边界。这一工作的背后是大型语言模型 (Large Language Model，LLM) 生成领域的新训练范式：RLHF (Reinforcement Learning from Human Feedback) ，即以强化学习方式依据人类反馈优化语言模型。

过去几年里各种 LLM 根据人类输入提示 (prompt) 生成多样化文本的能力令人印象深刻。然而，对生成结果的评估是主观和依赖上下文的，例如，我们希望模型生成一个有创意的故事、一段真实的信息性文本，或者是可执行的代码片段，这些结果难以用现有的基于规则的文本生成指标 (如 [BLEU](https://en.wikipedia.org/wiki/BLEU) 和 [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))) 来衡量。除了评估指标，现有的模型通常以预测下一个单词的方式和简单的损失函数 (如交叉熵) 来建模，没有显式地引入人的偏好和主观意见。

如果我们 **用生成文本的人工反馈作为性能衡量标准，或者更进一步用该反馈作为损失来优化模型**，那不是更好吗？这就是 RLHF 的思想：使用强化学习的方式直接优化带有人类反馈的语言模型。RLHF 使得在一般文本数据语料库上训练的语言模型能和复杂的人类价值观对齐。

看看 [ChatGPT](https://openai.com/blog/chatgpt/) 是如何解释 RLHF 的：

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/chatgpt-explains.png" width="500" />
</p>

ChatGPT 解释的很好，但还没有完全讲透；让我们更具体一点吧！

# RLHF 技术分解

RLHF 是一项涉及多个模型和不同训练阶段的复杂概念，这里我们按三个步骤分解：

1. 预训练一个语言模型 (LM) ；
2. 聚合问答数据并训练一个奖励模型 (Reward Model，RM) ；
3. 用强化学习 (RL) 方式微调 LM。

### Step 1. 预训练语言模型

首先，我们使用经典的预训练目标训练一个语言模型。对这一步的模型，OpenAI 在其第一个流行的 RLHF 模型 [InstructGPT](https://openai.com/blog/instruction-following/) 中使用了较小版本的 GPT-3; Anthropic 使用了 1000 万 ～ 520 亿参数的 Transformer 模型进行训练；DeepMind 使用了自家的 2800 亿参数模型 [Gopher](https://arxiv.org/abs/2112.11446)。

这里可以用额外的文本或者条件对这个 LM 进行微调，例如 OpenAI 对 “更可取” (preferable) 的人工生成文本进行了微调，而 Anthropic 按 “有用、诚实和无害” 的标准在上下文线索上蒸馏了原始的 LM。这里或许使用了昂贵的增强数据，但并不是 RLHF 必须的一步。由于 RLHF 还是一个尚待探索的领域，对于” 哪种模型” 适合作为 RLHF 的起点并没有明确的答案。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/pretraining.png" width="500" />
</p>

接下来，我们会基于 LM 来生成训练 **奖励模型** (RM，也叫偏好模型) 的数据，并在这一步引入人类的偏好信息。

### Step 2. 训练奖励模型
    
RM 的训练是 RLHF 区别于旧范式的开端。这一模型接收一系列文本并返回一个标量奖励，数值上对应人的偏好。我们可以用端到端的方式用 LM 建模，或者用模块化的系统建模 (比如对输出进行排名，再将排名转换为奖励) 。这一奖励数值将对后续无缝接入现有的 RL 算法至关重要。

关于模型选择方面，RM 可以是另一个经过微调的 LM，也可以是根据偏好数据从头开始训练的 LM。例如 Anthropic 提出了一种特殊的预训练方式，即用偏好模型预训练 (Preference Model Pretraining，PMP) 来替换一般预训练后的微调过程。因为前者被认为对样本数据的利用率更高。但对于哪种 RM 更好尚无定论。

关于训练文本方面，RM 的提示 - 生成对文本是从预定义数据集中采样生成的，并用初始的 LM 给这些提示生成文本。Anthropic 的数据主要是通过 Amazon Mechanical Turk 上的聊天工具生成的，并在 Hub 上 [可用](https://huggingface.co/datasets/Anthropic/hh-rlhf)，而 OpenAI 使用了用户提交给 GPT API 的 prompt。

关于训练奖励数值方面，这里需要人工对 LM 生成的回答进行排名。起初我们可能会认为应该直接对文本标注分数来训练 RM，但是由于标注者的价值观不同导致这些分数未经过校准并且充满噪音。通过排名可以比较多个模型的输出并构建更好的规范数据集。

对具体的排名方式，一种成功的方式是对不同 LM 在相同提示下的输出进行比较，然后使用 [Elo](https://en.wikipedia.org/wiki/Elo_rating_system) 系统建立一个完整的排名。这些不同的排名结果将被归一化为用于训练的标量奖励值。

这个过程中一个有趣的产物是目前成功的 RLHF 系统使用了和生成模型具有 不同 大小的 LM (例如 OpenAI 使用了 175B 的 LM 和 6B 的 RM，Anthropic 使用的 LM 和 RM 从 10B 到 52B 大小不等，DeepMind 使用了 70B 的 Chinchilla 模型分别作为 LM 和 RM) 。一种直觉是，偏好模型和生成模型需要具有类似的能力来理解提供给它们的文本。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/reward-model.png" width="600" />
</p>

接下来是最后一步：利用 RM 输出的奖励，用强化学习方式微调优化 LM。

### Step 3. 用强化学习微调

长期以来出于工程和算法原因，人们认为用强化学习训练 LM 是不可能的。而目前多个组织找到的可行方案是使用策略梯度强化学习 (Policy Gradient RL) 算法、近端策略优化 (Proximal Policy Optimization，PPO) 微调初始 LM 的部分或全部参数。因为微调整个 10B～100B+ 参数的成本过高 (相关工作参考低秩适应 [LoRA](https://arxiv.org/abs/2106.09685) 和 DeepMind 的 [Sparrow](https://arxiv.org/abs/2209.14375) LM) 。PPO 算法已经存在了相对较长的时间，有大量关于其原理的指南，因而成为 RLHF 中的有利选择。

事实证明，RLHF 的许多核心 RL 进步一直在弄清楚如何将熟悉的 RL 算法应用到更新如此大的模型。

让我们首先将微调任务表述为 RL 问题。首先，该 **策略** (policy) 是一个接受提示并返回一系列文本 (或文本的概率分布) 的 LM。这个策略的 **行动空间** (action space) 是 LM 的词表对应的所有词元 (一般在 50k 数量级) ，**观察空间** (observation space) 是可能的输入词元序列，也比较大 (词汇量 ^ 输入标记的数量) 。**奖励函数** 是偏好模型和策略转变约束 (Policy shift constraint) 的结合。

PPO 算法确定的奖励函数具体计算如下：将提示 *x* 输入初始 LM 和当前微调的 LM，分别得到了输出文本 *y1*, *y2*，将来自当前策略的文本传递给 RM 得到一个标量的奖励 \\( r_\theta \\)。将两个模型的生成文本进行比较计算差异的惩罚项，在来自 OpenAI、Anthropic 和 DeepMind 的多篇论文中设计为输出词分布序列之间的 Kullback–Leibler [(KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) 散度的缩放，即 \\( r = r_\theta - \lambda r_\text{KL} \\) 。这一项被用于惩罚 RL 策略在每个训练批次中生成大幅偏离初始模型，以确保模型输出合理连贯的文本。如果去掉这一惩罚项可能导致模型在优化中生成乱码文本来愚弄奖励模型提供高奖励值。此外，OpenAI 在 InstructGPT 上实验了在 PPO 添加新的预训练梯度，可以预见到奖励函数的公式会随着 RLHF 研究的进展而继续进化。

最后根据 PPO 算法，我们按当前批次数据的奖励指标进行优化 (来自 PPO 算法 on-policy 的特性) 。PPO 算法是一种信赖域优化 (Trust Region Optimization，TRO) 算法，它使用梯度约束确保更新步骤不会破坏学习过程的稳定性。DeepMind 对 Gopher 使用了类似的奖励设置，但是使用 A2C ([synchronous advantage actor-critic](http://proceedings.mlr.press/v48/mniha16.html?ref=https://githubhelp.com)) 算法来优化梯度。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/rlhf.png" width="650" />
</p>

作为一个可选项，RLHF 可以通过迭代 RM 和策略共同优化。随着策略模型更新，用户可以继续将输出和早期的输出进行合并排名。Anthropic 在他们的论文中讨论了 [迭代在线 RLHF](https://arxiv.org/abs/2204.05862)，其中策略的迭代包含在跨模型的 Elo 排名系统中。这样引入策略和 RM 演变的复杂动态，代表了一个复杂和开放的研究问题。

# Open-source tools for RLHF

Today, there are already a few active repositories for RLHF in PyTorch that grew out of this. The primary repositories are Transformers Reinforcement Learning ([TRL](https://github.com/lvwerra/trl)), [TRLX](https://github.com/CarperAI/trlx) which originated as a fork of TRL, and Reinforcement Learning for Language models ([RL4LMs](https://github.com/allenai/RL4LMs)).

TRL is designed to fine-tune pretrained LMs in the Hugging Face ecosystem with PPO. TRLX is an expanded fork of TRL built by [CarperAI](https://carper.ai/) to handle larger models for online and offline training. At the moment, TRLX has an API capable of production-ready RLHF with PPO and Implicit Language Q-Learning [ILQL](https://sea-snell.github.io/ILQL_site/) at the scales required for LLM deployment (e.g. 33 billion parameters). Future versions of TRLX will allow for language models up to 200B parameters. As such, interfacing with TRLX is optimized for machine learning engineers with experience at this scale.

[RL4LMs](https://github.com/allenai/RL4LMs) offers building blocks for fine-tuning and evaluating LLMs with a wide variety of RL algorithms (PPO, NLPO, A2C and TRPO), reward functions and metrics. Moreover, the library is easily customizable, which allows training of any encoder-decoder or encoder transformer-based LM on any arbitrary user-specified reward function. Notably, it is well-tested and benchmarked on a broad range of tasks in [recent work](https://arxiv.org/abs/2210.01241) amounting up to 2000 experiments highlighting several practical insights on data budget comparison (expert demonstrations vs. reward modeling), handling reward hacking and training instabilities, etc.
RL4LMs current plans include distributed training of larger models and new RL algorithms. 

Both TRLX and RL4LMs are under heavy further development, so expect more features beyond these soon. 

There is a large [dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) created by Anthropic available on the Hub.

# RLHF 的未来

尽管 RLHF 取得了一定的成果和关注，但依然存在局限。这些模型依然会毫无不确定性地输出有害或者不真实的文本。这种不完美也是 RLHF 的长期挑战和动力 —— 在人类的固有领域中运行意味着永远不会到达一个完美的标准。

收集人类偏好数据的质量和数量决定了 RLHF 系统性能的上限。RLHF 系统需要两种人类偏好数据：人工生成的文本和对模型输出的偏好标签。生成高质量回答需要雇佣兼职人员 (而不能依赖产品用户和众包) 。另一方面，训练 RM 需要的奖励标签规模大概是 50k 左右，所以并不那么昂贵 (当然远超了学术实验室的预算) 。目前相关的数据集只有一个基于通用 LM 的 RLHF 数据集 (来自 [Anthropic](https://huggingface.co/datasets/Anthropic/hh-rlhf) 和几个较小的子任务数据集 (如来自 [OpenAI](https://github.com/openai/summarize-from-feedback) 的摘要数据集) 。另一个挑战来自标注者的偏见。几个人类标注者可能有不同意见，导致了训练数据存在一些潜在差异。

除开数据方面的限制，一些有待开发的设计选项可以让 RLHF 取得长足进步。例如对 RL 优化器的改进方面，PPO 是一种较旧的算法，但目前没有什么结构性原因让其他算法可以在现有 RLHF 工作中更具有优势。另外，微调 LM 策略的一大成本是策略生成的文本都需要在 RM 上进行评估，通过离线 RL 优化策略可以节约这些大模型 RM 的预测成本。最近，出现了新的 RL 算法如隐式语言 Q 学习 (Implicit Language Q-Learning，[ILQL](https://sea-snell.github.io/ILQL_site/)) 也适用于当前 RL 的优化。在 RL 训练过程的其他核心权衡，例如探索和开发 (exploration-exploitation) 的平衡也有待尝试和记录。探索这些方向至少能加深我们对 RLHF 的理解，更进一步提升系统的表现。

### 参考资料

首先介绍一些相关的开源工作：

关于 [RLHF 的第一个项目](https://github.com/openai/lm-human-preferences)，来自 OpenAI， 一些 PyTorch 的 repo：

* [trl](https://github.com/lvwerra/trl)
* [trlx](https://github.com/CarperAI/trlx)
* [RL4LMs](https://github.com/allenai/RL4LMs)

此外，Huggingface Hub 上有一个由 Anthropic 创建的大型 [数据集](https://hf.co/datasets/Anthropic/hh-rlhf)。

相关论文包括在现有 LM 前的 RLHF 进展和基于当前 LM 的 RLHF 工作：

- [TAMER: Training an Agent Manually via Evaluative Reinforcement](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ICDL08-knox.pdf) (Knox and Stone 2008)
- [Interactive Learning from Policy-Dependent Human Feedback](http://proceedings.mlr.press/v70/macglashan17a/macglashan17a.pdf) (MacGlashan et al. 2017)
- [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) (Christiano et al. 2017)
- [Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces](https://ojs.aaai.org/index.php/AAAI/article/view/11485)
- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (Zieglar et al. 2019)
- [Learning to summarize with human feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) (Stiennon et al., 2020)
- [Recursively Summarizing Books with Human Feedback](https://arxiv.org/abs/2109.10862) (OpenAI Alignment Team 2021)
- [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) (OpenAI, 2021)
- InstructGPT: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (OpenAI Alignment Team 2022)
- [InstructGPT: Training language models to follow instructions with human feedback (OpenAI Alignment Team 2022)](https://openai.com/blog/instruction-following/)
- GopherCite: [Teaching language models to support answers with verified quotes](https://www.deepmind.com/publications/gophercite-teaching-language-models-to-support-answers-with-verified-quotes) (Menick et al. 2022)
- Sparrow: [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/abs/2209.14375) (Glaese et al. 2022)
- [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/) (OpenAI 2022)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) (Gao et al. 2022)
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) (Anthropic, 2022)
- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) (Ganguli et al. 2022)
- [Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning](https://arxiv.org/abs/2208.02294) (Cohen at al. 2022)
- [Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://arxiv.org/abs/2210.01241) (Ramamurthy and Ammanabrolu et al. 2022)
- [Kojima et al. 2021](https://arxiv.org/abs/2108.04812)
- [Suhr and Artzi 2022](https://arxiv.org/abs/2212.09710)
- [Sokolov et al. 2016](https://arxiv.org/abs/1601.04468), [Gao et al. 2022](https://arxiv.org/abs/2203.10079)
* [Ranzato et al. 2015](https://arxiv.org/abs/1511.06732)
* [Bahdanau et al. 2016](https://arxiv.org/abs/1607.07086)
* [Nguyen et al. 2017](https://arxiv.org/abs/1707.07402)

## Citation

If you found this useful for your academic work, please consider citing our work, in text:

```
Lambert, et al., "Illustrating Reinforcement Learning from Human Feedback (RLHF)", Hugging Face Blog, 2022.
```

BibTeX citation:

```
@article{lambert2022illustrating,
  author = {Lambert, Nathan and Castricato, Louis and von Werra, Leandro and Havrilla, Alex},
  title = {Illustrating Reinforcement Learning from Human Feedback (RLHF)},
  journal = {Hugging Face Blog},
  year = {2022},
  note = {https://huggingface.co/blog/rlhf},
}
```

*Thanks to [Robert Kirk](https://robertkirk.github.io/) for fixing some factual errors regarding specific implementations of RLHF. Thanks to [Peter Stone](https://www.cs.utexas.edu/~pstone/), [Khanh X. Nguyen](https://machineslearner.com/) and [Yoav Artzi](https://yoavartzi.com/) for helping expand the related works further into history. *
*Thanks to Stas Bekman for fixing some typos or confusing phrases.* 
