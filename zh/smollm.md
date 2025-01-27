---
title: SmolLM：一个超快速、超高性能的小模型集合
thumbnail: /blog/assets/smollm/banner.png
authors:
- user: loubnabnl
- user: anton-l
- user: eliebak
translators:
- user: hugging-hoi2022
- user: zhongdongy
  proofreader: true
---

# SmolLM: 一个超快速、超高性能的小模型集合

## 简介

本文将介绍 [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-models-6695016cad7167254ce15966)。它集合了一系列最尖端的 135M、360M、1.7B 参数量的小模型，这些模型均在一个全新的高质量数据集上训练。本文将介绍数据整理、模型评测、使用方法等相关过程。

## 引言

近期，人们对能在本地设备上运行的小语言模型的兴趣日渐增长。这一趋势不仅激发了相关业者对蒸馏或量化等大模型压缩技术的探索，同时也有很多工作开始尝试在大数据集上从头训练小模型。

微软的 Phi 系列、阿里巴巴的 Qwen2 (小于 2B 参数量) 以及 Meta 的 MobileLLM 均展示了这样的结论: 如果设计得当、训练充分，小模型也可以获得很好的性能。然而，这其中关于数据整理、训练细节的相关信息大多都未被披露。

在本文中，我们将介绍 [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-models-6695016cad7167254ce15966)。这是一个包含一系列最顶尖的小语言模型的集合，这些模型的参数量包括 135M、360M 和 1.7B。这些模型基于 [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) 这一仔细整理的高质量数据集而构建，该数据集包含以下三个子集:

- **Cosmopedia v2**: 通过 Mixtral 模型合成的、包含课文和故事等内容的数据集 (token 数量为 28B)
- **Python-Edu**: 数据样本取自 [The Stack](https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids) 数据集、[根据教育价值打分](https://huggingface.co/HuggingFaceTB/python-edu-scorer) 筛选出来的数据集 (token 数量为 4B)
- **FineWeb-Edu**: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 数据集经过去重且 [根据教育价值打分](https://huggingface.co/HuggingFaceTB/python-edu-scorer) 筛选出来的数据集 (token 数量为 220B)

我们的评测结果显示，在对应的参数量区间内，SmolLM 的模型在一系列常识性推理和世界知识评测标准上均超越了现有的模型。在本文中，我们将介绍训练语料中三个子集的整理方法，并讨论 SmolLM 的训练和评测过程。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled.png" alt="" style="width: 90%; height: auto;"><br>
<em>SmolLM 的模型在不同推理和常识评测标准上的测试结果</em>
</p>

## 数据整理

### Cosmopedia 数据集: 从 v1 到 v2

Cosmopedia v2 是 Cosmopedia 数据集的增强版。Cosmopedia 是当前最大的合成数据集，常被用来进行与训练。它包含超过三百万的课文、博客、故事等数据，这些数据均由 Mixtral-8x7B-Instruct-v0.1 模型生成。绝大部分数据是通过这种方式生成的: 搜集网页内容 (称为“种子样本”)，提供内容所属的主题类别，然后让模型扩写来生成。如图 1 就展示了其中的一个样本示例。 这里我们使用大量网络样本来提高数据的多样性，并扩展提示词的话题范围。[这篇文章](https://huggingface.co/blog/cosmopedia) 详细介绍了 Cosmopedia 数据集。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%201.png" alt="" style="width: 90%; height: auto;"><br>
<em>图 1: Cosmopedia 提示词示例.</em>
</p>

为了在 v2 版的数据集中进一步优化数据质量，我们曾尝试过以下两种策略:

- 针对同一提示词，使用多个高性能模型去生成数据
- 优化提示词本身

针对第一种策略，我们曾尝试了 llama3-70B-Instruct、Mixtral-8x22B-Instruct-v0.1 以及 Qwen1.5-72B-Chat，但当我们在这些生成数据上训练后，我们发现效果提升很有限。因此，下文我们将聚焦于第二种策略: 我们是怎样改进提示词的。

#### 寻找更好的主题和种子样本

每个提示词都包含三个主要部分: 主题、种子样本和生成风格，这三部分确定了意向受众和我们希望模型生成的内容的类型。

为确保生成的一致性，我们需要将相关性强的种子样本归类到对应的主题里面。在 Cosmopedia v1 里，我们通过对 FineWeb 里的样本进行聚类，来确保主题和对应的样本是一致的 (如图 2)。但这种方法有两点局限性:

1. 这些主题虽然很全面地反映了 web/FineWeb 数据的聚类结果，但可能并没有全面反映真实世界的科目主题分布。
2. 每个聚类内部的样本并没有被进一步过滤，所以可能包含很多低质量样本。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%202.png" alt="" style="width: 90%; height: auto;"><br>
<em>图 2: FineWeb 的聚类结果</em>
</p>

因此，在 v2 版数据集中，我们使用 [BISAC 书籍分类](https://www.bisg.org/complete-bisac-subject-headings-list) 定义的 3.4 万个主题来代替无监督的聚类。 BISAC 已被作为一个通用标准，来给书籍进行科目分类。所以使用这种方法不仅能全面涵盖各类主题，也可以使得我们使用的主题在教育价值层面更有专业性。具体而言，我们先使用 BISAC 里 51 个大类中的 5000 个主题，让 Mixtral 模型针对每个主题生成它的多种二级子类。下图就展示了最终各个大类别下的子类主题数量分布。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%203.png" alt="" style="width: 90%; height: auto;"><br>
<em>图 3: 不同大类下面的主题数量的统计直方图</em>
</p>

在定义好了主题后，我们还需要找到和主题相关的数据条目。和使用搜索引擎类似，我们制作了一个搜索工具，用来检索和每个主题有强相关性的数据。我们使用 BISAC 的大类和子类主题作为搜索的关键词，在 [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 数据集的 [CC-MAIN-2024-10](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/data/CC-MAIN-2024-10) 和 [CC-MAIN-2023-50](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/data/CC-MAIN-2023-50) 文件夹中进行搜索，两个文件夹包含有超过 5.2 亿的样本。对于每个搜索关键词，我们检索出 1000 条最接近的数据条目。相关代码可以见 [这里](https://github.com/huggingface/cosmopedia/tree/main/fulltext_search)。

最终，我们集成了涵盖 3.4 万个主题的 3400 万条数据。接下来需要确定的是，哪种生成风格效果最好。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%204.png" alt="" style="width: 90%; height: auto;"><br>
<em>图 4: “Medical” 大类下的子类主题和对应的网页数据样本.</em>
</p>

#### 生成风格

为了确定最有效的生成风格，我们通过训练 1.8B 模型进行了对比实验，其中我们使用不同的 Cosmopedia v1 子集数据，共计有 80 亿 token 的数据量。在生成训练数据时，我们只生成 20 亿 token 的数据量，训练 4 轮，以此来节省时间 (使用 Mixtral 生成 20 亿 token 需要大约 1000 个 GPU 小时)。训练和评测的各项配置和 [FineWeb ablation models](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) 一致。每个训练我们都跑两遍，每次用不同的随机种子，最终评测分数取两次的平均。

至于训练结果对比，我们对比了 Cosmopedia v1 的这些子集:

- 两个 web 样本集: [web_samples_v1](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/main/data/web_samples_v1) 和 [web_samples_v2](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/main/data/web_samples_v2)
- [stories](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/main/data/stories) 子集
- [stanford](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/main/data/stanford) 和 [openstax](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/main/data/openstax) 两个子集

我们发现，当训练文本是基于 stanford 和 openstax 的主题和种子样本时，总体的性能最好，其 MMLU 和 ARC 指标均高于两个 web 样本集。而 stories 仅仅有助于常识性的相关指标。在实现了 v2 版数据集检索新主题和种子样本的代码后，我们也可以对比这次实验的指标数据，来判断我们新生成的提示词的质量好坏。

接下来，我们还要探索哪种受众风格最好。我们使用相同的课文类提示词生成课文内容，但针对两种目标受众: 中学生和大学生。我们发现，在针对中学生受众的生成数据上训练，模型在除了 MMLU 的各项指标上取得了最好的分数。一个合理的解释是，这些指标一般都是对初级或中级的科学知识进行考察，而 MMLU 则包含了针对高级甚至专家级知识的问题。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%205.png" alt="" style="width: 90%; height: auto;"><br>
<em>不同受众的课文数据上的评测结果</em>
</p>
<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%206.png" alt="" style="width: 90%; height: auto;"><br>
<em>不同受众的课文数据上的评测结果</em>
</p>

对于 v2 版本数据，我们生成的数据中，40% 面向中学生受众，30% 面向大学生受众，剩下 30% 混合了各种不同受众群体，且融合了 v1 中 stories、stanford 等风格的课文风格。除此之外，我们还生成了 10 亿代码相关的课文，这部分数据基于 [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText) 数据集的 [Python](https://huggingface.co/datasets/math-ai/AutoMathText/tree/main/data/code/python) 代码部分。

最终，我们生成了 3900 万合成数据，按 token 数量算，规模达到了 20 亿，涵盖课文、故事、文章、代码，假想受众的多样性也很高，涵盖主题超过 3.4 万。

### FineWeb-Edu 数据集

FineWeb-Edu 数据集由我们在几个月前随着 [FineWeb 数据集的技术报告](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) 公开，它包含 **1.3 万亿** 的 token。其内容来自教育相关的网页，这些网页信息从 🍷 FineWeb 数据集中过滤而来。

在过滤数据的过程中，我们开发了一个 [关于教育价值质量的分类器](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier)，它的训练使用了 Llama3-70B-Instruct 生产的标注信息。我们使用这一分类器，在 FineWeb 里找出教育价值最高的一批网页内容。下图实验表明，在过滤出来的 FineWeb-Edu 训练的模型，在常用指标上明显由于 FineWeb。这也说明我们的分类器是有用的。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%207.png" alt="" style="width: 90%; height: auto;"><br>
<em>FineWeb-Edu 和其它公开网页数据集的训练效果对比</em>
</p>

在 Smollm-Corpus 数据集中，我们加入了 2200 亿去重过的、来自 FineWeb 的 token。

### Stack-Edu-Python 数据集

这里，我们也用了和 FineWeb-Edu 一样的方法。我们用 Llmama3 对 [The Stack](https://huggingface.co/datasets/bigcode/the-stack) 数据集中 50 万的 python 代码段根据教育价值进行打分，然后使用这些打过分的数据训来年了一个 [分类器](https://huggingface.co/HuggingFaceTB/python-edu-scorer)。然后我们在 Starcoder 模型的训练语料库的 python 子集中使用这个分类器。我们只保留 4 分及以上的样本，最终我们从 400 亿的 token 中得到了一个包含 40 亿 token 的新数据集。

下图展示了模型在不同数据集上 (使用 4 或 3 作为阈值过滤的、未进行过滤的) 训练的效果。我们可以看到，模型在 Python-Edu 上收敛速度比在未过滤数据上训练快 3 倍还多。而且在只使用了 120 亿 token 的训练数据后，就达到了 top-1 16% 的通过率。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%208.png" alt="" style="width: 90%; height: auto;"><br>
<em>Python-Edu 和未过滤数据的训练效果对比</em>
</p>

## 模型训练

SmolLM 包含三种不同参数量大小的模型，它们均在下图所示的混合数据上训练:

- 参数量为 135M 和 360M 的模型，均使用 [Smollm-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) 的 6000 亿 token 数据量进行训练
- 参数量为 1.7B 的模型，则使用 [Smollm-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) 1 万亿 token 的数据量进行了训练

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%209.png" alt="" style="width: 60%; height: auto;"><br>
<em>Training mixture of SmolLM models.</em>
</p>

### 超参数的选择

我们使用一种梯形的学习率变化策略，总训练时长的最后 20% 作为冷却时间。需要注意的是，梯形学习率变化的原始验证实验只使用了小规模训练，而我们的工作将其扩展到了大模型领域。

对于模型结构，我们的 135M 和 360M 模型均使用了和 [MobileLLM](https://arxiv.org/abs/2402.14905) 类似的设计，加入了 Grouped-Query Attention 结构，且优先深度扩展而不是宽度; 而 1.7T 的模型则使用了相对传统的设计。此外，三种模型均使用了 embedding tying，上下文长度均为 2048 个 token。使用长上下文微调还可以进一步扩展我们模型的上下文长度。

具体模型结构细节信息可见下表:

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%2010.png" alt="" style="width: 90%; height: auto;"><br>
<em>SmolLM 模型结构细节</em>
</p>

我们使用的分词器 (tokenizer) 是在 Smollm-Corpus 上训练的，其词汇量为 49152。

### 实验

使用梯形学习率的一个好处是，我们可以更快速地验证模型在 scaling law 下的扩展实验 (参考 [Hägele et al.](https://arxiv.org/pdf/2405.18392) 这篇论文)。这里我们使用 SmolLM-125M 做一个关于 scaling law 的小实验，来验证这一点。我们在不同的正常训练节点上进行学习率冷却，来结束训练。我们观察到，随着模型训练时间越来越长，性能是持续上升的，这一现象即使在 Chinchilla 最优点 (参数量和训练数据的最优配比) 之后也存在。根据这些实验现象，我们决定用 1T 量级 token 的数据去训练 1.7B 的模型，而 135M 和 360M 的模型则在 600B 量级的 token 上训练。因为在训练了 400B 量级的 token 后，两个较小的模型在一些指标上就已经进步缓慢了。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%2011.png" alt="" style="width: 90%; height: auto;"><br>
<em>SmolLM 的 125M 参数量模型在不同量级数据上训练的评测结果</em>
</p>

我们还尝试添加指令数据集以及在学习率冷却阶段对 Cosmopedia 子集进行上采样，但这些收效甚微。可能的原因是，我们的混合数据集质量已经足够高了，所以这些改进效果很有限。

在训练两个较小模型的过程中，我们记录了各项评测指标的变化情况。见下图:

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%2012.png" alt="" style="width: 90%; height: auto;"><br>
<em> 训练过程中 SmolLM-135M 和 SmolLM-360M 在不同指标上的变化</em>
</p>

## 模型评测

我们对不同参数量的 SmolLM 模型进行了评测，并和当前最好的一些模型进行了对比。我们使用了多种指标，评测内容包括常识推理和世界知识。我们使用 `lighteval` 和 [这些配置](https://github.com/huggingface/cosmopedia/tree/main/evaluation) 进行评测。对于人类主观评测，我们使用了 bigcode-evaluation-harness，其中 temperature 设为 0.2，top-p 为 0.95，样本量为 20。针对未开源的 MobileLLM，其测试结果均取自论文中的数据。

我们发现:

- 在 200M 参数量以下的模型中，SmolLM-135M 在各项指标上都超越了当前最好的模型 MobileLLM-125M。相比于 MobileLLM-125M 使用 1T token 的数据量去训练，SmolLM-135M 只使用了 600B 的数据量。
- 在 500M 参数量以下的模型中，SmolLM-360M 也超越了其它模型。相比于 MobileLLM-350M 和 Qwen2-500M，SmolLM-360M 参数量和训练数据均更少。
- 在 2B 参数量以下的模型中，SmolLM-1.7B 也超越了包括 Phi1.5 和 MobileLLM-1.5B 等模型。
- SmolLM-1.7B 还在 Python 编程能力上表现抢眼 (我们测评的 Qwen2-1.5B 分数和 Qwen 团队给出的不同，我们的实验配置是: temperature 设为 0.2，top-p 设为 0.95，样本量为 20)。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%2014.png" alt="" style="width: 90%; height: auto;"><br>
<em>SmolLM 和其它小语言模型的对比，除 MobileLLM 外，所有实验的配置均相同，因为 MobileLLM 未开源</em>
</p>
<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/image.png" alt="" style="width: 50%; height: auto;"><br>
<em>SmolLM 模型的人工评测</em>
</p>

我们也使用公开数据集对模型进行了指令精调。三个模型均在 [WebInstructSub dataset](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub) 和 StarCoder2-Self-OSS-Instruct 进行了一轮训练。随后，我们也进行了 DPO 训练，其中，我们使用 [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) 训练 135M 和 1.7B 的模型，使用 [argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k) 训练 360M 的模型。相关训练配置和 Zephyr-Gemma 的 [说明文档](https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-gemma/README.md) 相同，除了 SFT 的学习率被我们改为了 3e-4。

下表展示了经指令精调的 SmolLM 模型 (SmolLM-Instruct) 和其它模型在 IFEval 上的对比。Qwen2-1.5B-Instruct 取得了最高分，SmolLM-Instruct 模型则在模型大小和性能上取得了很好的权衡，而且仅使用了公开可用的数据集。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%2016.png" alt="" style="width: 60%; height: auto;"><br>
<em>SmolLM-Instruct 模型在 IFEval 的评测结果</em>
</p>

## 如何本地运行 SmolLM 模型？

我们的小模型可以在各种本地的硬件上运行。举例来说，iPhone 15 有 6GB 的内存，iPhone 15 Pro 有 8GB 内存，从手机到笔记本电脑，诸多设备都足以运行我们的模型。下表中，我们记录了模型运行时实际的内存占用情况:

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/Untitled%2013.png" alt="" style="width: 60%; height: auto;"><br>
<em>SmolLM 模型内存占用情况</em>
</p>

除了 `transformers` 库可以直接使用的模型权重外，我们也开放了 ONNX 模型，并计划为 `llama.cpp` 提供 GGUF 版模型。此外，[SmolLM-135M](https://huggingface.co/spaces/HuggingFaceTB/SmolLM-135M-Instruct-WebGPU) 和 [SmolLM-360M](https://huggingface.co/spaces/HuggingFaceTB/SmolLM-360M-Instruct-WebGPU) 的 WebGPU 演示页面也可以使用。

## 总结

本文介绍了 SmolLM 系列模型，通过实验证明了，只要训练充分、数据质量足够好，小模型也可以取得很好的性能。本文在此用 SmolLM 提供了一个示例，强有力地证明了模型大小和模型性能可以做到完美权衡。

## 其它资源

- SmolLM 模型集合: [https://huggingface.co/collections/HuggingFaceTB/smollm-models-6695016cad7167254ce15966](https://huggingface.co/collections/HuggingFaceTB/smollm-models-6695016cad7167254ce15966)
- SmolLM-Corpus 数据集: [https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
- WebGPU 演示页面: [https://huggingface.co/spaces/HuggingFaceTB/SmolLM-135M-Instruct-WebGPU](https://huggingface.co/spaces/HuggingFaceTB/SmolLM-135M-Instruct-WebGPU) and [https://huggingface.co/spaces/HuggingFaceTB/SmolLM-360M-Instruct-WebGPU](https://huggingface.co/spaces/HuggingFaceTB/SmolLM-360M-Instruct-WebGPU)