---
title: "Graph Classification with Transformers" 
thumbnail: /blog/assets/126_graphml-classification/thumbnail.png
---

# Graph classification with Transformers

<div class="blog-metadata">
    <small>Published January 30, 2023.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/intro-graphml.md">
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

In the previous [blog](https://huggingface.co/blog/intro-graphml), we explored some of the theoretical aspects of machine learning on graphs. In this one, we will explore how you can do graph classification using the Transformers library.

At the moment, the only graph transformer model available in Transformers is Microsoft's Graphormer, so this is the one we will use here. We are looking forward to see what other models people will use and integrate :hugging_face:

## Requirements
To follow this tutorial, you need to have installed `datasets` and `transformers` (version >= **TODO**).

## Data
To use graph data, you can either start from your own datasets, or use [those available on the hub][https://huggingface.co/graphs-datasets). We'll focus on using already available ones, but feel free to add your own datasets!

### Format
On the hub, graph datasets are mostly stored as lists of graphs (using the `jsonl` format). 

A single graph is a dictionnary, and here is the expected format for our graph classification datasets:
- `edge_index` 
	- Type: list of 2 lists of integers.
	- It contains the indices of nodes in edges, stored as a list containing two parallel lists of edge indices. 
	- Example: a graph containing four nodes (0, 1, 2 and 3) and where connections are 1->2, 1->3 and 3->1* will have `edge_index`=[[1, 1, 3], [2, 3, 1]]. You might notice here that node 0 is not present here, as it is not part of an edge per se. This is why the next attribute is important.
- `num_nodes` 
	- Type: integer 
	- It indicates the total number of nodes available in the graph (by default, it is assumed that nodes are numbered sequentially). 
	- Example: In our above example, `num_nodes` = 4.
- `y`
	- Type: list of either integers (for multi-class classification), floats (for regression), or lists of ones and zeroes (for binary multi-task classification)
	- It maps each graph to what we want to predict from it (be it a class, a property value, or several binary label for different tasks).
	- Example: We could predict if the graph is small sized (0), medium sized (1) or big. Here, `y` = [0].
- `node_feat` 
	- Type: list of lists of integer (Optional) 
	- This contains the available features (if present) for each node of the graph, ordered by node index.
- `edge_attr`
	- Type: list of lists of integers (Optional)
	- This contains the available attributes (if present) for each edge of the graph, following the `edge_index` ordering.

### Loading
Loading a graph dataset from the hub is very easy. Let's load the `ogbg-mohiv` dataset (a baseline from the Open Graph Benchmark by Stanford), stored in the `graphs-datasets` repository: 

```python
from datasets import load_dataset

# There is only one split on the hub
dataset = load_dataset("graphs-datasets/ogbg-molhiv")

dataset = dataset.shuffle(seed=0)
```

This dataset already has three splits, `train`, `validation`, and `test`, and all these splits contain our 5 columns of interest (`edge_index`, `edge_attr`, `y`, `num_nodes`, `node_feat`), which you can see by doing `print(dataset)`. 

If you have other graph libraries, you can use them to plot your graphs and further inspect the dataset. For example, using PyGeometric and matplotlib:
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

### Preprocessing
Graph transformer frameworks usually apply specific preprocessing to their datasets, to generate added features and properties which help learning.
Here, we use the Graphormer's default preprocessing, which generates in/out degree information, shortest path between nodes matrices, and other properties of interest for the model. 
 
```python
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

dataset_processed = dataset.map(preprocess_item, batched=False)
```

It is also possible to apply this preprocessing on the fly, in the DataCollator's parameters (by setting `on_the_fly_processing` to True): not all datasets are as small as `ogbg-molhiv`, and for large graphs, it might be too costly to store all the preprocessed data beforehand. 

## Training or fine-tuning

### Creating a model
Here, we load an existing pretrained model/checkpoint and to fine-tune it on our downstream task, a binary classification task (hence `num_classes = 2`). 
```python
from transformers import GraphormerForGraphClassification

# To train from scratch
model = GraphormerForGraphClassification.from_pretrained(
    "clefourrier/pcqm4mv2_graphormer_base",
    num_classes=2, # num_classes for the downstream task 
    ignore_mismatched_sizes=True,
)
```

It is also possible to create a new randomly initialized model to train from scratch, either following the known parameters of a given checkpoint, or by manually choosing them.

### Training or fine-tuning
For graph datasets, it is particularly important to play around with batch sizes and gradient accumulation steps, to train on enough samples, while avoiding out of memory errors.
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="tmp",
    logging_dir="tmp",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    auto_find_batch_size=True, # batch size can be changed automatically to prevent OOMs
    gradient_accumulation_steps=10,
    dataloader_num_workers=4, #1, 
    num_train_epochs=20,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed["train"],
    eval_dataset=dataset_processed["validation"],
    data_collator=GraphormerDataCollator(),
)

trainer.train() 
```

As this model is quite big, it takes about a day to train/fine-tune for 20 epochs on CPU (IntelCore i7). It can be worth using powerful GPUs and parallelization instead, leveraging `accelerate`.


## Ending note
Now that you know how to use `transformers` to train a graph classification model, we hope you will try to share your favorite graph transformer checkpoints, models and datasets on the hub for the rest of the community to use!