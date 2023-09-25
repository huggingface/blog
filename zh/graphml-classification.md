---
title: "使用 Transformers 进行图分类" 
thumbnail: /blog/assets/125_intro-to-graphml/thumbnail_classification.png
authors:
- user: clefourrier
translators:
- user: MatrixYao
---

# 使用 Transformers 进行图分类

<div class="blog-metadata">
    <small>Published April 14, 2023.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/graphml-classification.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/clefourrier"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1644340617257-noauth.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>clefourrier</code>
            <span class="fullname">Clémentine Fourrier</span>
        </div>
    </a>
</div>

在之前的[博文](https://huggingface.co/blog/intro-graphml)中，我们探讨了图机器学习的一些理论知识。这一篇我们将探索如何使用 Transformers 库进行图分类。（你也可以从[此处](https://github.com/huggingface/blog/blob/main/notebooks/graphml-classification.ipynb)下载演示 notebook，跟着一起做！）

目前，Transformers 中唯一可用的图 transformer 模型是微软的 [Graphormer](https://arxiv.org/abs/2106.05234)，因此本文的例子将会基于该模型。我们期待看到大家会使用并集成哪些其他模型进 🤗。

## 软件
要学习本教程，需要安装 `datasets` 和 `transformers`（版本号 >= 4.27.2），你可以使用 `pip install -U datasets transformers` 来安装。

## 数据
你可以使用自己的图数据集，也可以使用 [Hub 上已有的数据集](https://huggingface.co/datasets?task_categories=task_categories:graph-ml&sort=downloads)。本文我们主要使用已有的数据集，你也可以随时[添加你的数据集](https://huggingface.co/docs/datasets/upload_dataset)到 Hugging Face！

### 数据加载
从 Hub 加载图数据集非常简单。这里，我们加载 OGB 库中的 `ogbg-mohiv` 数据集（该数据集是斯坦福 [开放图基准（Open Graph Benchmark，OGB）](https://ogb.stanford.edu/) 的一部分）：

```python
from datasets import load_dataset

# There is only one split on the hub
dataset = load_dataset("OGB/ogbg-molhiv")

dataset = dataset.shuffle(seed=0)
```

这个数据集含三个拆分，`train`、`validation` 和 `test`，所有这些拆分每一行都表示一个图，每个图包含 5 个数据列（ `edge_index`、`edge_attr`、`y`、`num_nodes`、`node_feat` )，你可以通过执行 `print(dataset)` 来查看。

如果你还安装了其他图处理库，你还可以用这些库把图可视化出来，并进一步检查数据集。例如，使用 PyGeometric 和 matplotlib：

```python
import networkx as nx
import matplotlib.pyplot as plt

# We want to plot the first train graph
graph = dataset["train"][0]

edges = graph["edge_index"]
num_edges = len(edges[0])
num_nodes = graph["num_nodes"]

# Conversion to networkx format
G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from([(edges[0][i], edges[1][i]) for i in range(num_edges)])

# Plot
nx.draw(G)
```

### 格式
在 Hub 上，图数据集主要存储为图列表形式（使用 `jsonl` 格式）。

单个图表示为一个字典，以下是我们图分类数据集的理想格式：

- `edge_index` 包含图上每条边对应的节点 ID，存储为包含两个`节点列表`的列表（即由一个源节点列表和一个目的节点列表组成的列表）。
    - **类型**：2个整数列表的列表。
    - **示例**：包含四个节点（0、1、2 和 3）且连接为 1->2、1->3 和 3->1 的图将具有 `edge_index = [[1, 1, 3]、[2、3、1]]`。你可能会注意到此处不存在节点 0，因为在本数据中它与其他节点无边连接。这就是下一个属性很重要的原因。
- `num_nodes` 表示图中可用节点的数目（默认情况下，假定节点按顺序编号）。
    - **类型**：整数
    - **示例**：在上例中，`num_nodes = 4`。
- `y` 每个图的预测标签（可以是类、属性值或是不同任务的多个二分类标签）。
    - **Type**：整数列表（用于多分类）、浮点数（用于回归）或 0/1 列表（用于二元多任务分类）
    - **示例**：我们可以预测图规模（小 = 0，中 = 1，大 = 2）。本例中，`y = [0]`。
- `node_feat` 包含图中每个节点的可用特征（如果存在），按节点 ID 排序。
    - **类型**：整数列表的列表（可选）
    - **例子**：如上例中的节点可以有一些类型特征（就像分子图中的节点是不同的原子，不同的原子有不同的类型一样）。打比方，本例中 `node_feat = [[1], [0], [1], [1]]`。
- `edge_attr` 包含图中每条边的可用属性（如果存在），按 `edge_index` 排序。
    - **类型**：整数列表的列表（可选）
    - **例子**：仍使用上例，边也可以有类型（如分子中的键），如 edge_attr = [[0], [1], [1]]`。

### 预处理
图 transformer 框架通常需要根据数据集进行特定的预处理，以生成有助于目标学习任务（在我们的案例中为分类）的特征和属性。
在这里，我们使用 `Graphormer` 的默认预处理，它生成进度/出度信息、节点间的最短路径以及模型感兴趣的其他属性。
 
```python
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

dataset_processed = dataset.map(preprocess_item, batched=False)
```

我们也可以在 `DataCollat​​or` 的参数中动态进行预处理（通过将 `on_the_fly_processing` 设置为 True）。但并非所有数据集都像 `ogbg-molhiv` 那样小，对于大图，动态预处理成本太高，因此需要预先进行预处理，并存储预处理后的数据供后续训练实验使用。

## 模型

### 模型加载
这里，我们加载一个已有的预训练模型及其 checkpoint 并在我们的下游任务上对其进行微调，该任务是一个二分类任务（因此 `num_classes = 2` ）。我们还可以在回归任务 (`num_classes = 1`) 或多任务分类上微调我们的模型。

```python
from transformers import GraphormerForGraphClassification

model = GraphormerForGraphClassification.from_pretrained(
    "clefourrier/pcqm4mv2_graphormer_base",
    num_classes=2, # num_classes for the downstream task 
    ignore_mismatched_sizes=True,
)
```

我们来看下细节。

在代码中调用 `from_pretrained` 方法来下载并缓存模型权重。由于类的数量（用于预测）取决于数据集，我们将新的 `num_classes` 和`ignore_mismatched_sizes` 与 `model_checkpoint` 一起传给该函数。这会触发函数创建一个自定义的、特定于该下游任务的分类头，这个头与原模型中的解码器头很可能是不同的。

我们也可以创建一个新的随机初始化的模型来从头开始训练，此时，我们既可以复用给定检查点的超参配置，也可以自己手动选择超参配置。

### 训练或微调
为了简化模型训练，我们使用 `Trainer`。我们需要定义训练相关的配置以及评估指标来实例化 `Trainer`。我们主要使用 `TrainingArguments`类，这是一个包含所有配置项的类，用于定制训练配置。我们要给它一个文件夹名称，用于保存模型的 checkpoint。

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    "graph-classification",
    logging_dir="graph-classification",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    auto_find_batch_size=True, # batch size can be changed automatically to prevent OOMs
    gradient_accumulation_steps=10,
    dataloader_num_workers=4, #1, 
    num_train_epochs=20,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    push_to_hub=False,
)
```

对于图数据集，调整 batch size 和梯度累积步数来保证有效 batch size 够大同时又要避免内存不足，这件事尤为重要。

最后一个参数 `push_to_hub` 允许 `Trainer` 在训练期间定期将模型推送到 Hub，这个通常由保存步长来决定。

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed["train"],
    eval_dataset=dataset_processed["validation"],
    data_collator=GraphormerDataCollator(),
)

```

在用于图分类的 `Trainer` 中，对给定的图数据集使用正确的数据整理器（data collator）很重要，这个数据整理器会将图转换为用于训练的 batch 数据。

```python
train_results = trainer.train()
trainer.push_to_hub()
```

训练完后，可以使用 `push_to_hub` 将模型与所有其他训练相关信息一起保存到 hub。

由于此模型比较大，因此在 CPU (Intel Core i7) 上训练/微调 20 个 epoch 大约需要一天时间。想要更快点的话，你可以使用强大的 GPU 和并行化方法，你只需在 Colab notebook 中或直接在你选择的其他集群上启动代码即可。

## 结束语
现在你已经知道如何使用 `transformers` 来训练图分类模型，我们希望你尝试在 Hub 上分享你最喜欢的图 transformer 模型的 checkpoints、模型以及数据集，以供社区的其他人使用！