---
title: "大语言模型：新的摩尔定律？"
thumbnail: /blog/assets/33_large_language_models/01_model_size.jpg
authors:
- user: juliensimon
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 大语言模型：新的摩尔定律？


不久前，微软和 Nvidia [推出了](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) Megatron-Turing NLG 530B，一种基于 Transformer 的模型，被誉为是 “*世界上最大且最强的生成语言模型*”。

毫无疑问，此项成果对于机器学习工程来讲是一场令人印象深刻的能力展示，表明我们的工程能力已经能够训练如此巨大的模型。然而，我们应该为这种超级模型的趋势感到兴奋吗？我个人倾向于否定的回答。我将在通过本文阐述我的理由。

<kbd>
  <img src="../assets/33_large_language_models/01_model_size.jpg">
</kbd>

### 这是你的深度学习大脑

研究人员估计，人脑平均包含 [860](https://pubmed.ncbi.nlm.nih.gov/19226510/) 亿个神经元和 100 万亿个突触。可以肯定的是，这里面并非所有的神经元和突触都用于语言。有趣的是，GPT-4 [预计](https://www.wired.com/story/cerebras-chip-cluster-neural-networks-ai/) 有大约 100 万亿个参数...... 虽然这个类比很粗略，但难道我们不应该怀疑一下构建与人脑大小相当的语言模型长期来讲是否是最佳方案？

当然，我们的大脑是一个了不起的器官，它经过数百万年的进化而产生，而深度学习模型仅有几十年的历史。不过，我们的直觉告诉我们: 有些东西无法计算 (这是个双关语，:)) 。

### 深度学习，深度销金窟？

如你所料，在庞大的文本数据集上训练一个 5300 亿参数的模型需要相当多的基础设施。事实上，Microsoft 和 Nvidia 使用了数百台 DGX A100 GPU 服务器，每台 19 万 9 千美元。如果再把网络设备、托管成本等因素考虑进去的话，任何想要重现该实验的组织或个人都必须花费近 1 亿美元。来根薯条压压惊？

说真的，有哪些组织有那种值得花费 1 亿美元来构建深度学习基础设施的业务？再少点，又有哪些组织有那种可以值得花费 1000 万美元基础设施的业务？很少。既然很少，那么请问，这些模型为谁而生呢？

### GPU 集群的热

尽管训练大模型需要杰出的工程能力，但在 GPU 上训练深度学习模型本身却是一种蛮力技术。根据规格表，每台 DGX 服务器可消耗高达 6.5 千瓦的功率。同时，数据中心 (或服务器机柜) 至少需要同样多的冷却能力。除非你是史塔克家族的人 (Starks) ，需要在冬天让临冬城 (Winterfell) 保持温暖，否则你必须处理散热问题。

此外，随着公众对气候和社会责任问题意识的增强，还需要考虑碳足迹问题。根据马萨诸塞大学 2019 年的一项 [研究](https://arxiv.org/pdf/1906.02243.pdf)，“*在 GPU 上训练一次 BERT 产生的碳足迹大致与一次跨美飞行相当*”。

BERT-Large 有 3.4 亿个参数。我们可以通过此推断 Megatron-Turing 的碳足迹大致如何……认识我的人都知道，我并不是一个热血环保主义者。尽管如此，这些数字也不容忽视。

### 所以呢？

我对 Megatron-Turing NLG 530B 和接下来可能会出现的模型巨兽感到兴奋吗？不。我认为值得增加成本、复杂性以及碳足迹去换取 (相对较小的) 测试基准上的改进吗？不。我认为构建和推广这些庞大的模型能帮助组织理解和应用机器学习吗？不。

我想知道这一切有什么意义。为了科学而科学？好的老营销策略？技术至上？可能每个都有一点。如果是这些意义的话，我就不奉陪了。

相反，我更专注于实用且可操作的技术，大家都可以使用这些技术来构建高质量的机器学习解决方案。

### 使用预训练模型

在绝大多数情况下，你不需要自定义模型架构。也许你会 想要 自己定制一个模型架构 (这是另一回事)，但请注意此处猛兽出没，仅限资深玩家！

一个好的起点是寻找已经针对你要解决的任务预训练过的 [模型](https://huggingface.co/models) (例如，[英文文本摘要](https://huggingface.co/models?language=en&pipeline_tag=summarization&sort=downloads)) 。

然后，你应该快速尝试一些模型，用它们来预测你自己的数据。如果指标效果不错，那么打完收工！如果还需要更高一点的准确率，你应该考虑对模型进行微调 (稍后会详细介绍) 。

### 使用较小的模型

在评估模型时，你应该从那些精度满足要求的模型中选择尺寸最小的那个。它预测得更快，并且需要更少的硬件资源来进行训练和推理。节俭需要从一开始就做起。

这其实也不算什么新招。计算机视觉从业者会记得 [SqueezeNet](https://arxiv.org/abs/1602.07360) 2017 年问世时，与 [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) 相比，模型尺寸减少了 50 倍，而准确率却与 AlexNet 相当甚至更高。多聪明！

自然语言处理社区也在致力于使用迁移学习技术缩减模型尺寸，如使用 [知识蒸馏技术](https://en.wikipedia.org/wiki/Knowledge_distillation)。[DistilBERT](https://arxiv.org/abs/1910.01108)  也许是其中最广为人知的工作。与原始 BERT 模型相比，它保留了 97% 的语言理解能力，同时尺寸缩小了 40%，速度提高了 60%。你可以 [Hugging Face](https://huggingface.co/distilbert-base-uncased) 尝试一下 DistilBERT。同样的方法也已经应用于其他模型，例如 Facebook 的 [BART](https://arxiv.org/abs/1910.13461)，你可以 [在 Hugging Face 尝试 DistilBART](https://huggingface.co/models?search=distilbart)。

[Big Science](https://bigscience.huggingface.co/) 项目的最新模型也令人印象深刻。下面这张来自于 [论文](https://arxiv.org/abs/2110.08207) 的图表明，他们的 T0 模型在许多任务上都优于 GPT-3，同时尺寸小 16 倍。

<kbd>
  <img src="../assets/33_large_language_models/02_t0.png">
</kbd>

你可以 [Hugging Face](https://huggingface.co/bigscience/T0pp) 尝试 T0。我们需要更多的此类研究！

### 微调模型

如果你需要特化一个模型，你不应该从头开始训练它。相反，你应该对其进行微调，也就是说，仅针对你自己的数据训练几个回合。如果你缺少数据，也许 [这些数据集](https://huggingface.co/datasets) 中的某个可以帮助你入门。

猜对了，这是进行迁移学习的另一种方式，它会帮助你节省一切！

- 收集、存储、清理和标注的数据更少，
- 更快的实验和迭代，
- 生产过程所需的资源更少。

换句话说: 节省时间，节省金钱，节省硬件资源，拯救世界！

如果你需要教程，Hugging Face [课程](https://huggingface.co/course)可以帮助你立即入门。

### 使用云基础设施

不管你是否喜欢它们，事实是云公司懂得如何构建高效的基础设施。可持续性研究表明，基于云的基础设施比其他替代方案更节能减排: 请参阅 [AWS](https://sustainability.aboutamazon.com/environment/the-cloud)、[Azure](https://azure.microsoft.com/en-us/global-infrastructure/sustainability) 和 [Google](https://cloud.google.com/sustainability)。Earth.org [宣称](https://earth.org/environmental-impact-of-cloud-computing/)虽然云基础设施并不完美，“*[它] 比替代方案更节能，并促进了环境友好的服务及经济增长。*"

在易用性、灵活性和随用随付方面，云肯定有很多优势。它也比你想象的更环保。如果你的 GPU 不够用，为什么不尝试在 AWS 的机器学习托管服务 [Amazon SageMaker](https://aws.amazon.com/sagemaker/) 上微调你的 Hugging Face 模型？我们为你准备了 [大量示例](https://huggingface.co/docs/sagemaker/train)。

### 优化你的模型

从编译器到虚拟机，软件工程师长期以来一直在使用能够针对任何运行硬件自动优化代码的工具。

然而，机器学习社区仍在这个课题上苦苦挣扎，这是有充分理由的。优化模型的尺寸和速度是一项极其复杂的任务，其中涉及以下技术:

- 专用硬件加速: 如训练加速硬件 ([Graphcore](https://www.graphcore.ai/)、[Habana](https://habana.ai/)) 、推理加速硬件 ([Google TPU](https://cloud.google.com/tpu)，[AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/))。
- 剪枝: 删除对预测结果影响很小或没有影响的模型参数。
- 融合: 合并模型层 (例如，卷积和激活) 。
- 量化: 以较小的位深存储模型参数 (例如，使用 8 位而不是 32 位)

幸运的是，自动化工具开始出现，例如 [Optimum](https://huggingface.co/hardware) 开源库和 [Infinity](https://huggingface.co/infinity)，Infinity 是一个最低能以 1 毫秒的延迟提供 Transformers 推理能力的容器化解决方案。

### 结论 

在过去的几年里，大语言模型的尺寸平均每年增长 10 倍。这开始看起来像另一个 [摩尔定律](https://en.wikipedia.org/wiki/Moore%27s_law)。

这条路似曾相识，我们应该知道这条路迟早会遇到收益递减、成本增加、复杂性等问题以及新的风险。指数的结局往往不是会很好。还记得 [Meltdown and Spectre](https://meltdownattack.com/) 吗？我们想知道人工智能的 Meltdown and Spectre 会是什么吗？

与其赌上你的时间精力和金钱去追求万亿参数的模型，我们构建一些更实际也更有效的，能造福所有开发者解决现实问题的解决方案不是更好吗？

*对 Hugging Face 可以在哪些方面帮助您的组织构建和部署生产级别的机器学习解决方案感兴趣？欢迎联系 [julsimon@huggingface.co](mailto:julsimon@huggingface.co) (猎头和销售勿扰哦))。*

