---
title: "SetFitABSA：基于 SetFit 的少样本、方面级情感分析"
thumbnail: /blog/assets/setfit-absa/intel_hf_logo_2.png
authors:
- user: ronenlap
  guest: true
- user: tomaarsen
- user: lewtun
- user: danielkorat
  guest: true
- user: orenpereg
  guest: true
- user: moshew
  guest: true
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# SetFitABSA: 基于 SetFit 的少样本、方面级情感分析

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/setfit-absa/method.png" width=500>
</p>
<p align="center">
    <em>SetFitABSA 是一种可以有效从文本中检测方面级情感的技术。</em>
</p>

方面级情感分析 (Aspect-Based Sentiment Analysis，ABSA) 是一种检测文本中特定方面的情感的任务。例如，在“这款手机的屏幕很棒，但电池太小”一句中，分别有“屏幕”和“电池”两个方面，它们的情感极性分别是正面和负面。

ABSA 应用颇为广泛，有了它我们可以通过分析顾客对产品或服务的多方面反馈，并从中提取出有价值的见解。然而，ABSA 要求在样本标注时对训练样本中涉及的各个方面进行词元级的识别，因此为 ABSA 标注训练数据成为了一件繁琐的任务。

为了缓解这一问题，英特尔实验室和 Hugging Face 联袂推出了 SetFitABSA，以用于少样本场景下的特定领域 ABSA 模型训练。实验表明，SetFitABSA 性能相当不错，其在少样本场景下表现甚至优于 Llama2 和 T5 等大型生成模型。

与基于 LLM 的方法相比，SetFitABSA 有两个独特优势:

<p>🗣 <strong> 无需提示: </strong> 在使用基于 LLM 的少样本上下文学习时，提示的作用非常关键，因此一般需要精心设计，这一要求使得最终效果对用词十分敏感且非常依赖用户的专业知识，因此整个方案会比较脆弱。SetFitABSA 直接从少量带标签的文本示例中生成丰富的嵌入，因此可完全无需提示。</p>

<p>🏎 <strong> 快速训练: </strong> SetFitABSA 仅需少量的已标注训练样本。此外，其训练数据格式非常简单，无需专门的标注工具，因此数据标注过程简单而快速。</p>

本文，我们将解释 SetFitABSA 的工作原理以及如何使用 [SetFit 库](https://github.com/huggingface/setfit) 训练你自己的模型。我们开始吧！

## 工作原理与流程

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/setfit-absa/method.png" width=700>
</p>
<p align="center">
    <em>SetFitABSA 的三段式训练流程 </em>
</p>

SetFitABSA 主要分三步。第一步从文本中提取候选方面; 第二步从候选方面中筛选出真正的方面，这一步主要由一个候选方面分类器来完成; 最后一步对每个方面判断其情感极性。其中，第二步和第三步均基于 SetFit 模型。

### 训练

**1. 候选方面提取**

这里我们假设: 方面一般指的是产品和服务的特性，因此其主要是名词或复合名词 (即几个相邻名词组成的字符串)。我们使用 [spaCy](https://spacy.io/) 从少样本训练集的句子中提取并标注出名词/复合名词。由于并非所有提取的名词/复合名词都是方面，因此我们叫它们 **候选方面**。

**2. 方面/非方面分类**

有了候选方面后，我们还需要训练一个模型，以便能够区分这些候选方面中哪些是真正的方面。为此，我们需要一些带有方面/无方面标签的训练样本。如果候选方面存在于训练集中我们即视其为 `True` ，否则视其为 `False` :

- **训练样本:** “Waiters aren’t friendly but the cream pasta is out of this world.”
- **分词:** [Waiters, are, n’t, friendly, but, the, cream, pasta, is, out, of, this, world, .]
- **提取候选方面:** [<strong style="color:orange">Waiters</strong>, are, n’t, friendly, but, the, <strong style="color:orange">cream</strong>, <strong style="color:orange">pasta</strong>, is, out, of, this, <strong style="color:orange">world</strong>, .]
- **训练集标签，其格式为 [BIO](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)):** [B-ASP, O, O, O, O、O、B-ASP、I-ASP、O、O、O、O、O、.]
- **根据训练集标签，生成方面/非方面标签:** [<strong style="color:green">Waiters</strong>, are, n’t, friendly, but, the, <strong style="color:green">cream</strong>, <strong style="color:green">pasta</strong>, is, out, of, this, <strong style="color:red">world</strong>, .]

至此，我们对所有候选方面进行了标注，下一步就是如何训练方面分类模型？也就是说，我们如何使用 SetFit 这一句子分类框架来对词元进行分类？我们使用的方法是: 将每个候选方面与其所在的句子串接起来，我们使用以下模板创建训练样本:

```
候选方面:所在句子
```

将该模板应用于上面的例子，我们会生成 3 个训练样本 - 其中 2 个标签为 `True` ，1 个标签为 `False` :

| 文本                                                        | 标签 |
|:------------------------------------------------------------------------------|:------|
| Waiters:Waiters aren't friendly but the cream pasta is out of this world.     | 1     |
| cream pasta:Waiters aren't friendly but the cream pasta is out of this world. | 1     |
| world:Waiters aren't friendly but the cream pasta is out of this world.       | 0     |
| ...                                                                           | ...   |

生成训练样本后，我们就可以借助 SetFit 的强大功能仅用少许样本训练一个特定领域的二元分类器，以从输入文本评论中提取出方面。这是我们第一个微调 SetFit 模型。

**3. 情感极性分类**

一旦系统从文本中提取到方面，它需要判断每个方面的情感极性 (如积极、消极或中性)。为此，我们需要第二个 SetFit 模型，其训练方式与上面相似，如下例所示:

- **训练样本:** “Waiters aren’t friendly but the cream pasta is out of this world.”
- **分词:** [Waiters, are, n’t, friendly, but, the, cream, pasta, is, out, of, this, world, .]
- **标签:** [NEG, O, O, O, O, O, POS, POS, O, O, O, O, O, .]

| 文本                                                                          | 标签 |
|:------------------------------------------------------------------------------|:------|
| Waiters:Waiters aren't friendly but the cream pasta is out of this world.     | NEG   |
| cream pasta:Waiters aren't friendly but the cream pasta is out of this world. | POS   |
| ...                                                                           | ...   |

注意，与方面提取模型不同，这里训练集中就不用包含非方面样本了，因为任务是对真正的方面进行情感极性分类。

## 推理

推理时，我们首先使用 spaCy 对输入句子进行候选方面提取，并用模板 `aspect_candidate:test_sentence` 生成推理样本。接下来，用方面/非方面分类器过滤掉非方面。最后，过滤出的方面会被送到情感极性分类器，该分类器预测每个方面的情感极性。

因此，我们的系统可以接收普通文本作为输入，并输出文本中的方面及其对应的情感:

**模型输入:**

```
"their dinner specials are fantastic."
```

**模型输出:**

```
[{'span': 'dinner specials', 'polarity': 'positive'}]
```

## 基准测试

我们将 SetFitABSA 与 [AWS AI 实验室](https://arxiv.org/pdf/2210.06629.pdf) 和 [Salesforce AI 研究院](https://arxiv.org/pdf/2204.05356.pdf) 的最新成果进行比较，这两项工作主要采用了对 T5 和 GPT2 进行提示微调的方法以实现方面级情感分析。为了对我们的工作进行全面测评，我们还将我们的模型与基于上下文学习的 Llama-2-chat 进行了比较。

我们采用的测评数据集是 2014 年语义评估挑战赛 ([SemEval14](https://aclanthology.org/S14-2004.pdf)) 中的 Laptop14 和 Restaurant14 ABSA [数据集](https://huggingface.co/datasets/alexcadillon/SemEval2014Task4)。测评任务选择的是术语提取中间任务 (SB1) 及完整 ABSA 任务 (包括方面提取及情感极性预测，即 SB1+SB2)。

### 模型尺寸比较

|        模型        |  尺寸 (参数量)  |
|:------------------:|:-------------:|
|    Llama-2-chat    |      7B       |
|      T5-base       |     220M      |
|     GPT2-base      |     124M      |
|    GPT2-medium     |     355M      |
| **SetFit (MPNet)** |    2x 110M    |

请注意，SB1 任务使用的 SetFitABSA 的参数量为 110M; SB2 任务再加上一个 110M 的模型。因此 SB1+SB2 时， SetFitABSA 的总参数量为 220M。

### 性能比较

我们看到，当训练样本较少时，SetFitABSA 有明显优势，尽管它比 T5 小 2 倍，比 GPT2-medium 小 3 倍。即便是与 64 倍参数量的 Llama 2 相比，SetFitABSA 也能获得相当或更好的效果。

**SetFitABSA vs GPT2**

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/setfit-absa/SetFitABSA_vs_GPT2.png" width=700>
</p>

**SetFitABSA vs T5**

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/setfit-absa/SetFitABSA_vs_T5.png" width=700>
</p>

请注意，为公平起见，在比较 SetFitABSA 与各基线 (GPT2、T5 等) 时，我们使用了相同的数据集划分。

**SetFitABSA vs Llama2**

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/setfit-absa/SetFitABSA_vs_Llama2.png" width=700>
</p>

我们注意到，增加 Llama2 提示中的上下文样本的数目并不能改善效果。这种现象 [之前也在 ChatGPT 中发现过](https://www.analyticsvidhya.com/blog/2023/09/power-of-llms-zero-shot-and-few-shot-prompting/)，我们认为后续值得深入调查一下。

## 训练你自己的模型

SetFitABSA 是 SetFit 框架的一个功能。要训​​练 ABSA 模型，首先需要安装包含 `absa` 功能的 `setfit` :

```shell
python -m pip install -U "setfit[absa]"
```

此外，我们必须安装 `en_core_web_lg` 版的 spaCy 模型:

```shell
python -m spacy download en_core_web_lg
```

接着开始准备训练数据集。训练集是一个 `Dataset` 对象，其包含 `text` 、 `span` 、 `label` 、 `ordinal` 四列:

- **text**: 含有方面的完整句子或文本。
- **span**: 句子中包含的方面。可以是一个单词或多个单词，如 “food”。
- **label**: 每个 span (即方面) 的情感极性标签，如 “positive”。这些标签的名称是在标注时确定的。
- **ordinal**: 如果某一方面在文本中出现了多次，则该列表示其在文本中出现的次序。这个值通常是 0，因为每个方面通常在对应文本中只出现一次。

举个例子，训练文本 “Restaurant with wonderful food but worst service I ever seen” 中包含两个方面，因此其在训练集表中占据两行，如下:

| text                                                         | span    | label    | ordinal |
|:-------------------------------------------------------------|:--------|:---------|:--------|
| Restaurant with wonderful food but worst service I ever seen | food    | positive | 0       |
| Restaurant with wonderful food but worst service I ever seen | service | negative | 0       |
| ...                                                          | ...     | ...      | ...     |

一旦准备好训练数据集，我们就可以创建一个 ABSA 训练器并运行训练。SetFit 模型的训练效率相当高，但由于 SetFitABSA 涉及两个依次训练的模型，因此建议使用 GPU 进行训练，以缩短训练时间。例如，以下训练脚本在免费的 Google Colab T4 GPU 上仅需约 10 分钟就可以完成 SetFitABSA 模型的训练。

```python
from datasets import load_dataset
from setfit import AbsaTrainer, AbsaModel

# Create a training dataset as above
# For convenience we will use an already prepared dataset here
train_dataset = load_dataset("tomaarsen/setfit-absa-semeval-restaurants", split="train[:128]")

# Create a model with a chosen sentence transformer from the Hub
model = AbsaModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Create a trainer:
trainer = AbsaTrainer(model, train_dataset=train_dataset)
# Execute training:
trainer.train()
```

好了！自此，我们成功训得一个特定领域的 ABSA 模型。我们可以将训得的模型保存到硬盘或将其上传到 Hugging Face Hub。请记住，该模型包含两个子模型，因此每个子模型都需有自己的路径:

```python
model.save_pretrained(
    "models/setfit-absa-model-aspect",
    "models/setfit-absa-model-polarity"
)
# 或
model.push_to_hub(
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-aspect",
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity"
)
```

现在我们使用训得的模型进行推理。首先加载模型:

```python
from setfit import AbsaModel

model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-aspect",
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity"
)
```

然后，使用模型的预测 API 进行推理。输入一个字符串列表，其中每个字符串代表一个评论文本:

```python
preds = model.predict([
    "Best pizza outside of Italy and really tasty.",
    "The food variations are great and the prices are absolutely fair.",
    "Unfortunately, you have to expect some waiting time and get a note with a waiting number if it should be very full."
])

print(preds)
# [
# [{'span': 'pizza', 'polarity': 'positive'}],
# [{'span': 'food variations', 'polarity': 'positive'}, {'span': 'prices', 'polarity': 'positive'}],
# [{'span': 'waiting time', 'polarity': 'neutral'}, {'span': 'waiting number', 'polarity': 'neutral'}]
# ]
```

有关训练选项、如何保存和加载模型以及如何推理等更多详细信息，请参阅 SetFit [文档](https://huggingface.co/docs/setfit/how_to/absa)。

## 参考文献

- Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Harris Papageorgiou, Ion Androutsopoulos, and Suresh Manandhar. 2014. SemEval-2014 task 4: Aspect based sentiment analysis. In Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014), pages 27–35.
- Siddharth Varia, Shuai Wang, Kishaloy Halder, Robert Vacareanu, Miguel Ballesteros, Yassine Benajiba, Neha Anna John, Rishita Anubhai, Smaranda Muresan, Dan Roth, 2023 “Instruction Tuning for Few-Shot Aspect-Based Sentiment Analysis”. https://arxiv.org/abs/2210.06629
- Ehsan Hosseini-Asl, Wenhao Liu, Caiming Xiong, 2022. “A Generative Language Model for Few-shot Aspect-Based Sentiment Analysis”. https://arxiv.org/abs/2204.05356
- Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, Oren Pereg, 2022. “Efficient Few-Shot Learning Without Prompts”. https://arxiv.org/abs/2209.11055
