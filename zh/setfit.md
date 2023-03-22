---
title: "SetFit: 高效的无提示少样本学习"
thumbnail: /blog/assets/103_setfit/intel_hf_logo.png
authors:
- user: Unso
- user: lewtun
- user: luketheduke
- user: dkorat
- user: orenpereg
- user: moshew
---

<h1>SetFit: 高效的无提示少样本学习 </h1>

<!-- {blog_metadata} -->
<!-- {authors} -->

<p align="center">
    <em > 与标准微调相比，SetFit 能更高效地利用训练样本，同时对噪声也更鲁棒。</em>
</p>

如何处理少标签或无标签的训练数据是每个数据科学家的梦魇😱。最近几年来，基于预训练语言模型的少样本（few-shot）学习出现并成为解决这类问题的颇有前途的方案。

HuggingFace 非常高兴向大家介绍 SetFit：一个基于 [Sentence Transformers](https://sbert.net/) 的高效的少样本微调（fine-tune）框架，该工作由 HuggingFace 和我们的研究伙伴 [Intel Labs](https://www.intel.com/content/www/us/en/research/overview.html) 以及 [UKP Lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp) 合作完成。SetFit 仅需使用很少的标注数据就能达到较高的准确率。举个例子，在客户评论情感数据集（Customer Reviews (CR) sentiment dataset）上，每类仅使用 8 个标注样本，SetFit 的准确率就和在 3 千个标注样本的训练全集上微调 RoBERTa Large 相当了🤯！

与其他少样本学习方案相比，SetFit 有很多独有的特点：

<p>🗣 <strong > 无需提示词或语言器（verbalisers）</strong>：目前的少样本微调技术都需要手工设计的提示或语言器，用于将训练样本转换成适合目标语言模型的格式。SetFit 通过直接从少量标注训练样本中生成丰富的嵌入，完全省去了提示。</p>

<p>🏎 <strong > 快速训练 </strong>: SetFit 不需要使用像 T0 或 GPT-3 这样的大规模语言模型就能达到高准确率。因此，典型情况下，它的训练和推理会快一个数量级或以上。</p>

<p>🌎 <strong > 支持多语言 </strong>: SetFit 可与 Hub 上的任一 Sentence Tranformer 一起使用，这意味着如果你想让它支持多语言文本分类，你只要简单地微调一个多语言的 checkpoint 就好了。</p>

如果你想知道更多细节，可以获取我们的 [论文](https://arxiv.org/abs/2209.11055), [数据](https://huggingface.co/SetFit), 及 [代码](https://github.com/huggingface/setfit)。在本博文中，我们主要解释 SetFit 是如何工作的以及如何使用SetFit训练一个你自己的模型。我们开始吧！

## SetFit 如何工作？

在设计 SetFit 时，我们始终牢记高效、简单两个原则。SetFit 主要包含两个阶段：首先在少量标注样例（典型值是每类 8 个或 16 个样例）上微调一个 Sentence Transformer 模型。然后，用微调得到的 Sentence Tranformer 的模型生成文本的嵌入（embedding），并用这些嵌入训练一个分类头（classification head）。

<div style="width: 700px; margin:0 auto;">

![](assets/103_setfit/setfit_diagram_process.png)
</div>
<div style="margin:0 auto;">
    <em>SetFit 的两阶段训练过程 </em>
</div>

SetFit 利用 Sentence Transformer 的能力去生成基于句对（paired sentences）的稠密嵌入。在第一步微调阶段，它使用对比训练（contrastive training）来最大化利用有限的标注数据。首先，通过选择类内（in-class）和类外（out-class）句子来构造正句对和负句对，然后在这些句对（或三元组（triplets））上训练 Sentence Transformer 模型并生成每个样本的稠密向量。第二步，根据每个样本的嵌入向量和各自的类标签，训练分类头。推理时，未见过的样本通过微调后的 Sentence Transformer 并生成嵌入，生成的嵌入随后被送入分类头并输出类标签的预测。

只需要把基础 Sentence Transformer 模型换成多语言版的，SetFit 就可以无缝地在多语言环境下运行。在我们的 [实验](https://arxiv.org/abs/2209.11055) 中，SetFit 在德语、日语、中文、法语以及西班牙语中，在单语言和跨语言的条件下，都取得了不错的分类性能。

## 测试 SetFit
尽管与现存的少样本模型相比，SetFit 的模型要小得多，但在各种各样的测试基准上，SetFit 还是表现出了与当前最先进的方法相当或更好的性能。在 [RAFT](https://huggingface.co/spaces/ought/raft-leaderboard) 这个少样本分类测试基准上，参数量为 335M 的 SetFit Roberta（使用 [`all-roberta-large-v1`](https://huggingface.co/sentence-transformers/all-roberta-large-v1) 模型）性能超过了 PET 和 GPT-3。它的排名仅在人类平均性能以及 11B 参数的 T-few 之后，而 T-few 模型的参数量是 SetFit Roberta 的 30 倍。SetFit 还在 11 个 RAFT 任务中的 7 个任务上表现好于人类基线。

| Rank | Method | Accuracy | Model Size | 
| :------: | ------ | :------: | :------: | 
| 2 | T-Few | 75.8 | 11B | 
| 4 | Human Baseline | 73.5 | N/A | 
| 6 | SetFit (Roberta Large) | 71.3 | 355M |
| 9 | PET | 69.6 | 235M |
| 11 | SetFit (MP-Net) | 66.9 | 110M |
| 12 | GPT-3 | 62.7 | 175 B |

<p align="center">
    <em>RAFT 排行榜上表现突出的方法 (截至 2022 年 9 月)</em>
</p>

在其他的数据集上，SeiFit 在各种各样的任务中也展示出了鲁棒的性能。如下图所示，每类仅需 8 个样本，其典型性能就超越了 PERFECT、ADAPET 以及微调后的原始 transformer 模型。SetFit 还取得了与 T-Few 3B 相当的结果，尽管它无需提示且模型小了 27 倍。

<div style="margin:0 auto; width:700px">

![](assets/103_setfit/three-tasks.png)

</div>
<div style="margin:0 auto;">
    <em > 在 3 个分类数据集上比较 SetFit 与其他方法的性能 </em>
</div>

## 快速训练与推理

<div style="margin:0 auto; width:400px">

![](assets/103_setfit/bars.png)

</div>

<p align="center">
    在每类 8 个标注样本的条件下，比较 T-Few 3B 和 SetFit（MPNet）的训练成本和平均性能。
</p>

因为 SetFit 可以用相对较小的模型取得高准确率，所以它训练起来可以非常快，而且成本也低不少。举个例子，在每类 8 个标注样本的数据集上使用 NVIDIA V100 训练 SetFit 只需要 30 秒，共花费 0.025 美金；相比较而言，相同的实验下，训练 T-Few 3B 需要一张 NVIDIA A100，时间上要 11 分钟，需花费 0.7 美金，成本高 28 倍以上。事实上，SetFit 不仅可以运行在那种你在 Google Colab 找到的 GPU 单卡上，甚至在 CPU 上你也仅需几分钟即可以训练一个模型。如上图所示，SetFit 的加速与模型大小相当，因此 [推理](https://arxiv.org/abs/2209.11055) 时，我们也可以获得相似的性能提升，进一步地，对 SetFit 模型进行蒸馏可以获得 123 倍的加速🤯。


## 训练你自己的模型

为了利于社区用户使用 SetFit，我们创建了一个小型 [`setfit` 库](https://github.com/huggingface/setfit) ，这样你仅需几行代码就可以训练自己的模型了。

第一件事就是运行如下命令安装库：
```sh
pip install setfit
```
接着，我们导入 `SetFitModel` 和 `SetFitTrainer`，它们是流水线化 SetFit 训练过程的两个核心类：

```python
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer
```

现在，我们开始从 HuggingFace Hub 上下载一个文本分类数据集。我们使用 [SentEval-CR](https://huggingface.co/datasets/SetFit/SentEval-CR) 数据集，它是一个客户评论数据集。

```python
dataset = load_dataset ("SetFit/SentEval-CR")
```

为了模拟仅有几个标注样例的真实场景，我们从数据集中每类采样 8 个样本：

```python
# Select N examples per class (8 in this case)
train_ds = dataset ["train"].shuffle (seed=42).select (range (8 * 2))
test_ds = dataset ["test"]
```
既然我们有数据集了，下一步是从 Hub 里加载一个预训练 Sentence Transformer 模型，并用它去实例化 `SetFitTrainer`。这里我们使用 [paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) 模型，我们发现该模型在多个数据集下都能得出很好的结果。

```python
# Load SetFit model from Hub
model = SetFitModel.from_pretrained ("sentence-transformers/paraphrase-mpnet-base-v2")

# Create trainer
trainer = SetFitTrainer (
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20, # Number of text pairs to generate for contrastive learning
    num_epochs=1 # Number of epochs to use for contrastive learning
)
```

最后一步是训练和评估模型：

```python
# Train and evaluate!
trainer.train ()
metrics = trainer.evaluate ()
```

就这样，你已经训练了你的第一个 SetFit 模型！记得把你训练后的模型上传到 Hub 里：）。

```python
# Push model to the Hub
# Make sure you're logged in with huggingface-cli login first
trainer.push_to_hub ("my-awesome-setfit-model")
```

虽然在上面的例子中我们只展示了如何用一个特定类型的模型走完全程，但其实我们可以针对不同的性能和任务，切换使用任意的 [Sentence Transformer](https://huggingface.co/models?library=sentence-transformers&sort=downloads) 模型。举个例子，使用多语言 Sentence Transformer 可以将少样本分类扩展至多语言的场景。

## 下一步

我们已经向大家展示了 SetFit 是用于少样本分类任务的有效方法。在接下来的几个月里，我们会继续探索将该方法扩展至自然语言推理和词分类任务并观察其效果。同时，我们也会很高兴看到业界从业者如何应用 SetFit 到他们自己的应用场景。如果你有任何问题或者反馈，请在我们的 [GitHub 仓库](https://github.com/huggingface/setfit) 上提问题🤗。

少样本学习快乐！

>>>> 英文原文: <url> https://huggingface.co/blog/setfit </url>
>>>>
>>>> 译者: Matrix Yao (姚伟峰)