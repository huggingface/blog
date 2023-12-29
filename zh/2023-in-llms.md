---
title: "2023, 开源大模型之年"
thumbnail: /blog/assets/cv_state/thumbnail.png
authors:
- user: clefourrier
translators:
- user: xinyu66
---

# 2023, 开源大模型之年

在 2023 年，大型语言模型（Large Language Models，简称 LLMs）受到了公众的广泛关注，许多人对这些模型的本质及其功能有了基本的了解。是否开源的议题同样引起了广泛的讨论。在 Hugging Face，我们对开源模型抱有极大热情。开源模型的优势在于，它们不仅促进了研究的可复制性，还鼓励社区参与到人工智能模型的开发中来，这样做有助于我们更容易地审视模型中可能存在的偏差和局限性。此外，通过重复利用已有的检查点，我们还能够减少整个领域的碳足迹（这只是 [众多优点](https://huggingface.co/papers/2302.04844) 中的一部分）。

让我们一起回顾开源 LLMs 在过去一年的发展历程吧！

*为了确保本文篇幅适中，我们将不涉及代码模型的相关内容。*

## 🍜 预训练大型语言模型的配方
首先，如何获得一个大型语言模型呢？（如果你对此已有所了解，可以跳过这部分内容。）

模型的 **架构**（即其代码表示）定义了它的具体实现和数学结构：这包括所有的相关参数，以及这些参数如何与输入数据进行交互。目前，大多数高性能的大型语言模型（LLMs）都是基于 “仅解码器”（decoder-only）的 Transformer 架构的衍生版本，有关原始 Transformer 的详细信息可以参考其 [发表的论文](https://huggingface.co/papers/1706.03762)。

**训练数据集** 是模型训练过程中（即参数被学习时）所依赖的全部样本和信息的集合，它使模型能够学习到特定的数据模式。这些数据通常包括多样的文本材料，既可以是各种自然语言文本，如法语、英语、汉语等，也可以是各类编程语言代码，比如 Python、C 语言等，或者是任何能够以文本形式表现的结构化信息，例如 Markdown 或 LaTeX 中的表格、公式等。

**分词器** 是定义如何将训练数据集中的文本转化为数字的工具（因为模型是一个数学函数，因此需要数字作为输入）。分词是通过将文本转换为称为 “词元” 的子单元（可以是单词、子词或字符，具体取决于分词方法）来完成的。分词器的词汇量大小决定了其能够将文本分割成的不同词元的种类数目，这个数字通常介于 32,000 到 200,000 之间。数据集的规模常常用它包含的 **词元数量** 来衡量。经过分词后，如今的数据集范围从几千亿词元到几万亿词元不等，这些词元是构成数据集的基本单元。

**训练超参数** 定义了模型训练的方法。这些参数决定了模型应如何调整自身以适应新的数据样本，以及模型参数更新的速度应该是多快。

一旦确定了这些超参数，接下来需要的就是 1）充足的计算资源来进行模型训练；2）具备专业技能的人员来执行和监督训练过程。训练过程本身包括在训练所用的硬件上初始化模型架构，以及依据前述超参数在训练数据集上应用训练算法。训练的成果是一系列模型权重 —— 这些就是经过学习的 **模型参数**，也正是人们通常所说的开放获取的预训练模型。这些权重可以用于后续的 **推理过程**，即对新的输入数据进行预测，例如生成文本。

预训练的大型语言模型（LLM）在完成初始训练后，还可以根据具体任务进行定制化或进一步调整。特别是当这些模型的参数被开放共享时，它们可以作为不同用例和应用的基础，经过一种称为 “微调” 的过程进行优化。微调包括在与原始预训练数据集不同的、通常更小且更专业化的数据集上，对模型执行额外的训练步骤，目的是为了针对特定应用场景优化模型性能。尽管微调步骤在计算资源消耗上有一定成本，但这一成本通常远低于从零开始训练一个全新模型所需的财务投入和环境代价。这也是高品质开源预训练模型极具吸引力的一个原因，它们使得即便是计算预算有限的从业者也能够自由地使用和改进这些模型。

## 🗝️ 2022 年，从规模竞赛转向数据竞赛

在 2023 年之前，社区有哪些开源模型可用？

直至 2022 年初，机器学习界普遍认为，模型的规模越大（即拥有的参数越多），其性能也越出色。特别是，模型一旦超过某个特定的规模阈值，其能力似乎会实现质的飞跃，这两种现象分别被称为 `突现能力` 和 `规模定律`。2022 年推出的多个预训练开源模型家族大多遵循这种范例。

1. [BLOOM](https://huggingface.co/papers/2211.05100) (BigScience Large Open-science Open-access Multilingual Language Model)  
BLOOM 是由 BigScience 研究团队推出的 [一系列模型](https://huggingface.co/bigscience/bloom)。BigScience 是一个由 Hugging Face 协调，联合法国的 GENCI 和 IDRIS 组织共同参与的国际合作项目，涵盖了来自 60 个国家、250 个研究机构的 1000 名科研人员。这些模型采用了仅包含解码器的 transformer 架构，并进行了细微调整，比如引入了嵌入后归一化[^1] 和 ALiBi 位置嵌入[^2] 技术。在这一系列模型中，最大的一个拥有 1760 亿个参数，它接受了 46 种人类语言和 13 种编程语言的 3500 亿个多语言数据词元的训练。大量的训练数据已经向公众开放，包括数据的来源、策划和处理过程的详细信息。它是目前为止发布的最大的开源多语言模型。
2. [OPT](https://huggingface.co/papers/2205.01068) (Open Pre-trained Transformer)  
Meta 发布的 [OPT 模型](https://huggingface.co/facebook/opt-66b) 系列采用了仅包含解码器的 Transformer 架构。这些模型借鉴了 GPT-3 论文中的技术，如特定的权重初始化和预归一化策略，并对注意力机制进行了改进，比如引入了交替的密集型与局部带状注意力层。系列中最大的模型拥有 1750 亿个参数，其训练数据涵盖了来自公共领域的 1800 亿个数据词元，包括书籍、Reddit 社交平台数据、新闻、维基百科以及其他多种互联网来源。这一系列模型在性能上与 GPT-3 不相上下，并且通过编码优化减少了计算资源的消耗。
3. [GLM-130B](https://huggingface.co/papers/2210.02414) (General Language Model)  
清华大学联合智谱 AI 共同发布了 [GLM-130B 模型](https://huggingface.co/THUDM/glm-roberta-large)。该模型基于完整的 Transformer 架构，并引入了一些创新（如采用 DeepNorm 进行层后归一化、使用旋转式位置嵌入）。GLM-130B 拥有 1300 亿参数，是在包含英文和中文的互联网数据集上训练的，这些数据集包括 The Pile、WuDao 语料库以及其他中文语料库，共计 4000 亿个词元。在性能上，GLM-130B 与 GPT-3 模型不相上下。
4. 较小或更专业的开源大语言模型  
近期，一些较小型的开源模型也相继发布，这些模型主要服务于科研领域：Meta 推出了 [Galactica](https://huggingface.co/papers/2211.09085) 系列的大型语言模型（LLM），其中规模最大的模型拥有高达 [120B](https://huggingface.co/facebook/galactica-120b) 参数，这些模型是在科学文献中的 1060 亿个词元基础上进行预训练的。EleutherAI 则发布了 [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b) 模型，这是一个完全开源的仅解码器式 Transformer 模型（包括模型架构、权重和数据），在 5000 亿词元上经过训练，并采用了 RoPE 以及对注意力机制和初始化过程的若干改进，为科学研究提供了一个完整的工具集。

这些巨大的模型令人振奋，然而，它们的运行成本也高得惊人！在进行推理计算（即从模型中得出预测结果）时，模型必须被加载到内存中，而一个具有一千亿参数的模型往往需要占用高达 220GB 的内存空间（这个过程我们将在后文中详细阐述），这样的内存需求对于大多数机构和专业人士来说都是难以承担的！

然而，2022 年 3 月，DeepMind 发表了一篇 [论文](https://huggingface.co/papers/2203.15556)，探讨了在固定计算预算条件下，模型参数与数据量的最优配比。简而言之，如果你的模型训练预算有限，应该如何平衡模型大小和数据规模？研究者们发现，在平均计算预算下，对于大型语言模型（LLMs），更高效的策略是维持一个相对较小的模型，并在更广泛的数据集上进行训练。他们开发的模型 Chinchilla（未公开）拥有 700 亿个参数，仅为某些大型模型参数总数的三分之一，却在高达 1.4 万亿个词元的数据集上进行了训练，是其他模型所使用数据量的三到四倍。结果显示，Chinchilla 在性能上不仅媲美甚至超越了其他更大的同类型模型，无论是开源还是非开源的。

这种范式的变化，尽管可能已在封闭的实验室环境中为人所知，但它却让整个开放的科学界感到措手不及。

## 🌊 2023, 开放发布之年

### *小型* 大语言模型的崛起

2023 年，仅解码器（decoder-only）式的 Transformer 模型迎来了爆发式增长。几乎每月都有新的预训练模型问世，发展速度之快以至于渐渐演变为每周甚至每日都有新模型的推出。Meta 在 2 月推出了 LLaMA 模型；Eleuther AI 在 4 月带来了 Pythia 模型；MosaicML 在 5 月推出了 MPT 模型；Salesforce 和 TIIUAE 则在 6 月分别发布了 X-GEN 和 Falcon 模型。Meta 紧随其后，在 7 月发布了 LLaMA 的升级版本 LLaMA 2。进入下半年，9 月阿里巴巴发布了 Qwen 模型；Mistral.AI 推出了同名 Mistral 模型；01-ai 在 11 月发布了 Yi 模型；Deci 推出了 DeciLM 模型；而 Upstage 则在 12 月带来了 Phi-2 和 SOLAR 模型。这一系列的模型发布，不仅展示了人工智能领域的快速进步，也预示着技术的不断迭代与革新。

这些发布包括了：a) 模型权重（在不同程度的开源许可下）；b) 对于较小规模的模型（介于 30 亿至 700 亿参数之间），它们的性能都相当出色，因此立刻被社区采用。这些模型几乎都采用仅解码器的 Transformer 架构，并且进行了各种调整（比如 ALiBi 或 RoPE、RMS 预归一化、SwiGLU），以及对注意力函数的一些改变（如 Flash-Attention、GQA、滑动窗口注意力），并且在不同的代码库实现中进行了优化，以提高训练或推理速度。这些调整很可能在一定程度上影响模型的性能和训练速度；然而，由于所有架构都已经连同权重一起公开发布，剩下的核心差异主要在于训练数据和模型的许可方式。

Meta AI 发布的 [LLaMA](https://huggingface.co/papers/2302.13971) 系列是该系列中的首款模型。研究团队的目标是在既定的计算预算内训练不同规模的模型，以求达到最优性能。他们首次明确将训练预算与推理成本（即在满足特定性能目标时，模型推理所需的成本）并重考虑。基于这样的考量，他们选择在更大量的数据和更多的训练步骤上，训练规模较小的模型，以期在较小的模型尺度上获得更高的性能（这是对训练计算效率的一种权衡）。在 LLaMA 系列中，最大的模型拥有 650 亿参数，经过了 1.4 万亿的词元训练，而规模较小的模型 —— 分别具有 60 亿和 130 亿参数 —— 则在 1 万亿词元训练后完成。在大多数基准测试中，130 亿参数的 LLaMA 小型模型的表现超过了 GPT-3，而 650 亿参数的 LLaMA 大模型在发布时则代表了最先进的技术水平。然而，这些模型的权重是以非商业许可的形式发布的，这限制了它们在社区中的应用范围。

Eleuther AI 是一个开源的非营利实验室，它发布了一系列名为 [Pythia](https://huggingface.co/papers/2304.01373) 的大型语言模型（LLMs）。这些模型有不同的规模，全部采用公开数据进行训练，目的是为了帮助研究人员理解大型语言模型训练的不同阶段。有关 Pythia 模型的更多信息，可以通过它们在 Hugging Face 上的 [系列合集](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1) 查看。

MosaicML 公司在两个月后推出了 [MPT 模型](https://www.mosaicml.com/blog/mpt-7b)，该模型的性能优越，并且支持商业用途，同时公司还公开了其训练的具体细节。MPT 的首个版本是一个 [7B](https://huggingface.co/mosaicml/mpt-7b) 的模型，紧接着在 6 月份，公司发布了一个更大的 30B 版本。这两个模型都是基于 1 万亿个英语和编程语言的词元训练而成，训练数据包括了 C4、CommonCrawl、The Stack、S2ORC 等数据集。

MPT 模型推出后不久，TIIUAE 团队便发布了 [Falcon 系列模型](https://huggingface.co/collections/tiiuae/falcon-64fb432660017eeec9837b5a) 中的 [7B](https://huggingface.co/tiiuae/falcon-7b) 和 30B 版本。这些模型在 1 至 1.5 万亿个英文和代码词元上进行了训练，训练数据包括来自 RefinedWeb、Project Gutenberg、Reddit、StackOverflow、GitHub、arXiv、Wikipedia 等多个来源。同年晚些时候，TIIUAE 还发布了一款更为庞大的 180B 模型。Falcon 模型的细节、所用数据以及训练过程均在一份技术报告及随后发表的 [研究论文](https://huggingface.co/papers/2311.16867) 中有详尽的描述。

先前的模型在公开时通常会公开其数据集，但随后推出的模型很少公布其训练过程中使用的具体信息，这使得重现它们的成果变得困难。尽管如此，这些模型通过发布它们的权重参数，为研究社区提供了一个研究和进一步开发的起点。

Salesforce 在夏初推出了 [X-Gen](https://huggingface.co/papers/2309.03450) [模型](https://huggingface.co/Salesforce/xgen-7b-4k-base)，这是一款拥有 70 亿参数的模型，训练数据包括了 15 万亿个 “自然语言和代码” 词元，训练过程分为多个步骤，并采用了数据调度系统（并非所有数据同时输入模型）。

X-Gen 在 Meta 推出的更为引人注目的新的 [LLaMA-2](https://huggingface.co/papers/2307.09288) 家族的阴影下显得有些黯然失色。LLaMA-2 是 Meta 推出的一个新的模型系列，规模从 [7B](https://huggingface.co/meta-llama/Llama-2-7b) 到 70B 不等，这些模型是在 2 万亿个 “来自公开来源的词元” 上训练而成的，采用了宽松的社区许可证，并经过了人类偏好的精细调整（RLHF），即所谓的对齐过程。

随后，新兴初创企业 Mistral 推出了其首款模型 ——[Mistral-7B](https://huggingface.co/papers/2310.06825)，[该模型](https://huggingface.co/mistralai/Mistral-7B-v0.1) 是基于互联网公开数据集的大量数据训练而成，具体数据量尚未公布。随着 2023 年末的临近，模型发布活动日益频繁。Mistral 紧接着发布了更为庞大的第二款模型 Mixtral 8x7B。与此同时，Deci.AI 公司也带来了其令人瞩目的首款模型 [DeciLM](https://huggingface.co/Deci/DeciLM-7B)，upstage 公司也不甘落后，推出了规模更大的 [SOLAR](https://huggingface.co/upstage/SOLAR-10.7B-v1.0) 模型。这些模型均采用了来源和数量未公开的数据进行训练。在各大排行榜和公开基准测试中，这些模型均展现出稳步的进步。

在 2023 年年底，值得关注的一大事件是中国训练并公开发布了多个性能显著提升的模型。其中，阿里巴巴推出了其双语（英汉）模型 [Qwen](https://huggingface.co/papers/2309.16609) 系列，其参数规模从 [70 亿](https://huggingface.co/Qwen/Qwen-72B) 至 700 亿不等，经过了 240 亿词元数据的训练。与此同时，01-AI 公司也发布了 [Yi](https://huggingface.co/01-ai/Yi-34B) 系列模型，其参数规模介于 60 亿至 340 亿之间，训练数据量达到了 300 亿词元。这些模型在公开排行榜（如 [Open LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)）以及一些极具挑战性的基准测试（例如 [Skill-Mix](https://huggingface.co/papers/2310.17567)）中的表现，均超过了之前的模型。2023 年底的另一强有力的新竞争者是 [DeepSeek AI](https://huggingface.co/deepseek-ai)，他们推出了 “DeepSeek-Coder”，该模型从零开始训练了 200 亿词元数据，其中包含 87% 的代码和 13% 的英汉混合自然语言。


### 随处可见的对话模型
2023 年，与前一年相比，几乎所有新发布的预训练模型都配备了预训练版本和对话微调版本，这些版本采纳了多种现有的调整方法。尽管适用于聊天环境的模型调整技术在 2022 年及以前已有所开发，但这些技术在 2023 年得到了广泛应用并迅速兴起，这突显了聊天模型在普罗大众中使用的快速增长，以及通过与模型的互动对其进行的人工评估（即 “氛围检查” 评估）。本文将详细介绍几种著名的训练调整预训练模型以进行聊天的方法，实际上，相关的变体还有很多！

**基于对话的微调** 是一种特殊形式的监督式微调。在这种方法中，我们使用的标注数据是对话形式的，类似于社交媒体上的多轮对话记录。通过这种方式，可以对模型进行特定的微调。在这个过程中，我们可以采用与模型训练阶段相同的技术。例如，在处理仅解码器 Transformer 模型时，可以训练模型通过自回归方法，即逐一预测接下来的词元。

**指令微调**（Instruction-based Fine-Tuning，IFT）采用相同的方法，但使用指令数据集，该数据集包含一系列类似查询的提示以及答案（如果需要，还可以包含可选的附加输入）。这些数据集教导模型如何遵循指示，并且可以是人类生成的，也可以是大型语言模型生成的。

利用大规模模型输出的合成数据集（由模型生成的数据集，例如来自 GPT-4 的生成，可以是来自指示或用户与模型之间的交互）是实现指导微调和聊天微调的一种方式。这通常被称为 “蒸馏”，因为它涉及从性能较高的模型中获取知识，以训练或微调较小的模型。

这两种方法都相对容易执行：你只需找到或创建相应的数据集，然后采用与训练时相同的技术对模型进行调整即可。去年，发布了众多指导性数据集，它们有效提升了模型在对话场景中的表现。想要了解更多关于此主题的信息，可以参阅这篇介绍性博文的 [链接](https://huggingface.co/blog/dialog-agents)。然而，尽管模型的性能有了显著提升，但它们仍未能完全达到人类的预期水平。
 
**从人类反馈中强化学习**（Reinforcement Learning from Human Feedback，RLHF）是一项旨在使模型输出与人类偏好（基于特定标准）相一致的特定方法。具体操作流程如下：模型根据给定的提示生成多个潜在答案；人类评估者对这些答案进行排序；然后，这些排序结果用于训练一个偏好模型（该模型学习如何给出反映人类对答案偏好程度的评分）；最后，利用偏好模型通过强化学习对语言模型进行进一步的微调。更详细的信息，请参阅这篇 [博客文章](https://huggingface.co/blog/rlhf)，原始 [RLHF 论文](https://huggingface.co/papers/1909.08593)，或者 Anthropic 关于 [RLHF 的论文](https://huggingface.co/papers/2204.05862)。需要注意的是，这是一种成本较高的方法（注释 / 排名 + 训练新模型 + 微调的整个过程成本很高），主要用于确保模型的输出与安全目标相符。为了降低成本，人们开发了一种低成本的变体方法，即利用高质量的语言模型来对模型输出进行评分，而不是完全依赖人类评价，这种方法称为从 **人工智能反馈中学习的强化学习**（Reinforcement Learning from AI Feedback, RLAIF）。

**直接偏好优化**（Direct Preference Optimization, DPO）是 RLHF 的另一种变体，其核心优势在于无需训练和运用独立的偏好模型。这一方法同样需要人类或人工智能生成的排序数据集，但它通过直接利用这些数据来更新模型，即通过对比模型现有的策略（即预测行为）与理想的策略（即能够预测出最优排序答案的行为）。换言之，模型本身即扮演了对齐和偏好模型的双重角色，这不仅简化了优化流程，而且根据报告，还能够实现与其他方法相媲美的性能水平。

回到来自（大多数）私企的小型开放权重模型的浪潮，其中很多模型都发布了经过精细调整的对应版本：MPT-7B 还配备了一个指令微调和一个对话版本，Falcon 和 XGen 模型的指令微调版本在年底发布，Llama-2、Qwen 和 Yi 发布了对话版本，DeciLM 则发布了一个指令微调版本。Llama-2 的发布尤其引人注目，因为它在预训练和指令微调模型中都特别注重安全性。

### 社区的进展如何？
虽然随着新模型的发布，聊天模型和指令微调模型通常会立即推出，但社区成员和研究人员并没有把这看作是理所应当的。在这些基础模型提供的沃土上，涌现出了一个庞大而活跃的微调爱好者社区。这些微调专家经常会构建新的数据集，并对模型进行细致的微调，以此来展现新发布模型的出色性能。

在 2023 年伊始，一些专为指令交互和对话微调设计的数据集已经被发布。例如，代表人类偏好的数据集包括 OpenAI 的 [WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons) 数据集、Anthropic 的 [HH-RLHF](https://github.com/anthropics/hh-rlhf) 数据集以及 OpenAI 的 [摘要](https://huggingface.co/datasets/openai/summarize_from_feedback) 数据集，它们在这一领域是开拓者。指令数据集的例子包括 BigScience 的 [公共提示池](https://huggingface.co/datasets/bigscience/P3)、Google 的 FLAN 1 和 2（[FLAN](https://github.com/google-research/FLAN) 数据集）、AllenAI 的 [自然指令](https://github.com/allenai/natural-instructions) 数据集、由不同机构的研究人员开发的自动生成指令框架 [自我指令](https://github.com/yizhongw/self-instruct)、由专家创建的指令基准 [超自然指令](https://aclanthology.org/2022.emnlp-main.340/)（有时用作微调数据），以及由特拉维夫大学和 Meta 合作生成的自动指令数据集 [非自然指令](https://aclanthology.org/2023.acl-long.806.pdf) 等。

❄️ 冬 2022/2023: 一月，来自中国多个研究机构的研究人员共同发布了 [人类 ChatGPT 指令语料库](https://huggingface.co/datasets/Hello-SimpleAI/HC3)（HC3），其中包含了人类与模型对各种问题的回答。3 月份，发布活动接连不断：斯坦福大学推出了 [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 模型，这是首个遵循指令的 LLaMA 模型（7B），以及相关的数据集，包括用大型语言模型生成的 52K 条指令。非营利开源实验室 LAION 发布了 [开放指令通用数据集](https://laion.ai/blog/oig-dataset/)（OIG），包含 4300 万条指令，这些指令既有通过数据增强创建的，也有编译自其他现有数据源的。同月，位于加州大学伯克利分校的 LMSYS 组织发布了 [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)，这也是一个基于 ChatGPT 聊天数据的 LLaMA 精调模型（13B），这些聊天数据是用户与 ChatGPT 之间的对话，由用户自己公开分享在 [ShareGPT](https://share-gpt.com/) 上。还发布了 [Guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) 数据集，它是 Alpaca 数据集的扩展版（增加了 50 万条多语言条目），以及相关的 LLaMA-7B 精调模型。

🌱 春：四月，伯克利人工智能研究实验室（Berkeley AI Research lab，BAIR）发布了 [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)，这是一个经过聊天调优的 LLaMA 模型，它使用了多个先前的数据集（包括 Alpaca、HH-RLHF、WebGPT、ShareGPT），而 DataBricks 则发布了 [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) 数据集，这是一个由 15K 条人工生成的指令组成的数据集，以及相关的 Pythia 微调模型。五月，清华大学发布了 [UltraChat](https://arxiv.org/abs/2305.14233)，这是一个包含 1.5M 对话指令的数据集，以及在该数据集上进行微调的 UltraLLaMA 模型。随后，微软发布了 [GPT4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) 数据集 / 框架，用于生成 GPT4 的指令。六月，微软研究院分享了一种新方法 [Orca](https://arxiv.org/pdf/2306.02707.pdf)，通过使用大型模型的推理轨迹（逐步解释其推理过程）来构建指令数据集，该方法很快被社区（尤其是 Alignementlab.ai）复现，他们创建了 [Open Orca](https://huggingface.co/Open-Orca) 数据集，包含数百万条条目，随后用于微调多个模型（如 Llama、Mistral 等）。五月和六月期间，[Camel-AI](https://huggingface.co/camel-ai) 发布了多个关于不同话题（物理、生物、化学等）的指令或聊天数据集，每个领域都有超过 20K 的示例。同样在六月，发布了 [Airoboros](https://github.com/jondurbin/airoboros) 框架，用于使用模型生成的数据微调模型（遵循自我指导方法），以及一系列的 [指令数据集](https://huggingface.co/jondurbin)。 

🌻 夏：八月，由中国的非营利组织 OpenBMB 发布了 [UltraLM](https://github.com/thunlp/UltraChat)（一种基于 LLaMA 的高性能聊天模型微调版本），随后在九月，他们又发布了相关的偏好数据集 [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)，这是一个包含与 GPT4 对比的输入反馈数据集，并附有注释。在整个夏天，一个名为 [NousResearch](https://huggingface.co/NousResearch) 的集体发布了多个基于私有和公开指导数据集的微调版本（特别是 Hermes 和 Capybara 系列）。九月，清华大学的一个学生团队发布了 [OpenChat](https://huggingface.co/openchat/openchat_3.5)，这是一个应用了新的强化学习微调策略的 LLaMA 微调版本。 

🍂 秋：十月，Hugging Face 发布了 [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) 模型，这是一个在 UltraChat 和 UltraFeedback 上使用 DPO 和 AIF 技术对 Mistral 模型进行微调的产物。同时，社区成员发布了 [OpenHermes 2](https://huggingface.co/teknium/OpenHermes-2-Mistral-7B)，这是一个在来自网络或使用 Axolotl 生成的 900K 条目上对 Mistral-7B 模型进行微调的版本。Lmsys 发布了 LMSYS-Chat-1M，包含了与 25 个大型语言模型（LLMs）的真实用户对话。十一月，OpenBuddy 发布了 OpenBuddy-Zephyr，这是一个对 Zephyr 模型进行微调的多轮对话模型。同月，NVIDIA 发布了 [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) 数据集，这是一个对齐微调数据集，提供了提示、相关模型回应以及基于几个标准对这些回答的评分，而微软研究院则发布了 [Orca-2](https://huggingface.co/microsoft/Orca-2-13b) 模型，这是一个在新的合成推理数据集上对 Llama 2 模型进行微调的版本。十二月，伯克利大学发布了 [Starling](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha) 模型，这是一个对 Open-Chat 模型进行 RLAIF 微调的版本，以及相关的数据集 [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar)，包含了 20 万条比较数据。 

正如我们看到的，今年整个领域的发展既依赖于通过使用高质量的预训练大型语言模型（LLMs）创建新数据集，也依赖于社区发布的各种开源模型，这使得该领域进步飞速！如果你现在在模型名称中看到这些名字中的任何一个，你就能够大概了解它的来源了🤗。

* 还有一些更专业的数据集，例如用于数学问题微调的 [MetaMath](https://meta-math.github.io/) 和 [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)，以及涉及数学和代码指令的 [Evol-Instruct](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k)，还有 [CodeAlpaca](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) 与 [CodeCapybara](https://github.com/FSoft-AI4Code/CodeCapybara) 等代码指令相关的数据集也已发布。虽然这些数据集同样被用于提升模型在特定任务上的表现，但我们在此不会详细介绍它们。你还可以访问 [令人心动的指令数据集](https://github.com/jianzhnie/awesome-instruction-datasets) 来查看其他相关数据集的集合。*

## 开启定制模型的大门
### 模型融合：极致的定制化
在开源社区的典范实践中，一个重要的里程碑是模型与数据的融合。随着每一次代码合并或提交，追溯所使用数据的来源变得愈发复杂 —— 许多公开的数据集本身就是其他数据集的汇编。同样，由于卓越性能的模型往往是在相似模型的基础上经过层层微调得来的（可参考 Mistral 的 [衍生模型树](https://huggingface.co/spaces/davanstrien/mistral-graph)），模型的发展历史也变得难以梳理。在这篇摘要中，我们尚未有足够的篇幅深入探讨这一引人入胜的技术领域，但在最后，我们将简要介绍一下它的概念。

然而，“模型融合” 究竟是什么意思呢？

**模型融合** 是一种将不同模型的权重融合到一个单一模型中的方法，其理想目标是将每个模型的各自优势结合在一个统一的模型中。目前已有一些技术实现了这一目标，这些技术大多在社区论坛中得到扩展和发布，这是一个全球范围内的去中心化研究的典型案例，涵盖了从业者、研究人员到业余爱好者的广泛社区。其中一种最简单的公开方法是平均一组具有共同架构的模型的参数（[示例 1](https://huggingface.co/papers/2204.03044)，[示例 2](https://huggingface.co/papers/2109.01903)），但还存在更复杂的参数组合方法，例如确定每个模型中对特定任务最有影响力的参数（加权平均），或者在合并前考虑模型间参数的相互干扰，从而选择保留哪些参数（关联融合）。

这些技术使任何人都能轻松地生成模型的组合，而且由于大多数现代模型都是基于同一架构的变体，这一过程变得尤为简便。这也是 [Open LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 上一些模型名称如 `llama2-zephyr-orca-ultra` 的原因。这个特定的例子很可能是将 `llama2` 和 `zephyr` 模型合并后，再在 orca 和 ultra 数据集上进行微调的结果。通常，更多的细节可以在 Hugging Face 中心的相应模型卡片上找到。

### 参数高效微调：触手可及的个性化体验
有时候，你可能需要进行更为细致的个性化调整，但受限于硬件显存大小，无法加载完整模型进行微调。其实，你知道吗？微调时并不必须要用到模型的全部。

你或许想尝试一种叫做 **参数高效微调**（Parameter-Efficient Fine-Tuning，PEFT）的方法。
这项技术首先会冻结你所关注的预训练模型中的参数，然后在其基础上附加一些新的参数层，也就是我们所说的 “适配器”。接下来，你只需对这些专为你的任务设计的轻量级适配器权重进行微调，这些权重远小于原始模型的规模。这样，你仅需分享你的小型适配器权重（以及底层模型）即可！你可以在 [这里](https://github.com/huggingface/peft) 探索一系列引人入胜的 PEFT 技术。

### 量化：模型普及于各处
我们已经看到，性能出色的模型现在形态各异…… 但即便如此，并不意味着它们对所有人都是触手可及的！一个拥有 300 亿参数的模型仅仅加载到内存中（还未开始使用）就可能需要超过 66GB 的 RAM，而并非社区中的每个人都有能力配备这样的硬件。

这就是量化技术的用武之地！量化是一种特殊的技术，它通过改变模型参数的精度来减少模型的大小。

量化是什么意思呢？

在计算机中，数字是以一定的精度存储的，例如 `float32`、`float16`、`int8` 等。精度不仅指明了数字类型（是浮点数还是整数），同时也指出了数字存储所占用的内存大小：例如 `float32` 是在计算机上以 32 位存储的浮点数。要了解更深入的解释，请参见这个 [链接](https://huggingface.co/docs/optimum/concept_guides/quantization#going-further-how-do-machines-represent-numbers)。因此，数据的精度越高，它所占用的物理内存就越多，这是因为需要更多的位来存储这些数据。

因此，如果你降低精度，就会减少模型参数在存储上占用的内存，进而减小模型的大小！这也意味着你降低了计算的实际精度，可能会降低模型的性能。然而，我们发现，在较大的模型上，这种性能下降实际上是 [非常有限](https://huggingface.co/blog/overview-quantization-transformers) 的。

回到我们之前的例子中，一个含有 300 亿参数的模型，在使用 `float16` 格式时需要不到 66GB 的内存。如果采用 `8bit`，内存需求将减半至 33GB；若使用 `4bit` 编码，则只需大约 16GB，进一步降低了内存的要求，使得模型更易于部署和使用。

精度转换有多种方法，涉及不同的 “转换” 策略，每种策略都有其独特的优势和局限。目前流行的转换方法包括 [bitsandbytes](https://huggingface.co/papers/2208.07339)、[GPTQ](https://huggingface.co/papers/2210.17323), 和 [AWQ](https://huggingface.co/papers/2306.00978) 等。有些开发者，例如 [TheBloke](https://huggingface.co/TheBloke)，甚至正在将所有流行的模型进行转换，以便更容易地被社区使用。所有这些方法都是相对较新并且仍在不断发展之中，我们期待随着时间的推移，这些技术能够取得更多的进步。

## 接下来呢？
年尾尚未到来！在这最后时刻，已经迎来了一些惊喜：新的架构是否终将超越简单高效的 Transformer 模型呢？

最新发布包括：
- 混合专家模型：
    - [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)，该模型由 8 个子模型（仅解码器的 Transformer 模型）组成，对于每个输入，一个路由器会选择两个最佳子模型并将它们的输出求和。
- 几种状态空间模型（通过潜在空间将输入映射到输出的模型，可以根据任务需求表达为 RNN 或 CNN）：
    - [Mamba](https://huggingface.co/papers/2312.00752)，增加了选择机制的状态空间模型
    - [Striped Hyena](https://huggingface.co/togethercomputer/StripedHyena-Nous-7B)，具有快速卷积核的状态空间模型

目前来说，这些新方法是否会取代 Transformer 模型还为时尚早，但状态空间模型确实非常有前景！

## 要点回顾
- 今年，从大型企业到初创公司，再到研究实验室，各种主体纷纷开放发布模型，这极大地赋能了社区，使其以前所未有的速度开始进行实验和探索。
- 模型公告的开放性呈现出起伏变化，从年初的公开发布（数据集组合、权重、架构）到年末对训练数据守口如瓶，导致无法复现。
- 开源模型出现在包括中国在内许多新的地方，有几个新的参与者将自己定位为语言模型竞争中的强劲竞争者。
- 个性化定制的可能性达到了前所未有的高度，新策略的出现（如强化学习优化的微调、适配器、合并技术），虽然这仅仅是个开始。
- 更小的模型尺寸和量化升级使得大型语言模型对更多人来说变得真正唾手可得！
- 新的架构也随之出现 —— 它们是否最终会取代 Transformer 架构，仍是一个值得关注的问题。

各位朋友，就是这样了！   

希望你喜欢我们今年的回顾，从中学到了一些知识，并且和我一样，对于人工智能进步现在如此依赖开源和社区努力感到无比热情！🤗

[^1]: 嵌入后归一化是使模型训练更加稳定的一个技巧。
[^2]: ALiBi 位置编码通过在模型中对序列内距离较远的词元进行交互时施加惩罚，优化了序列处理能力（相比之下，传统的位置编码仅仅记录了序列中各词元的顺序及其相对位置信息）。  
