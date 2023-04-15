---
title: "Graph Classification with Transformers" 
thumbnail: /blog/assets/125_intro-to-graphml/thumbnail_classification.png
---

# Graph classification with Transformers

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

In the previous [blog](https://huggingface.co/blog/intro-graphml), we explored some of the theoretical aspects of machine learning on graphs. This one will explore how you can do graph classification using the Transformers library. (You can also follow along by downloading the demo notebook [here](https://github.com/huggingface/blog/blob/main/notebooks/graphml-classification.ipynb)!)

At the moment, the only graph transformer model available in Transformers is Microsoft's [Graphormer](https://arxiv.org/abs/2106.05234), so this is the one we will use here. We are looking forward to seeing what other models people will use and integrate 🤗

## Requirements
To follow this tutorial, you need to have installed `datasets` and `transformers` (version >= 4.27.2), which you can do with `pip install -U datasets transformers`.

## Data
To use graph data, you can either start from your own datasets, or use [those available on the Hub](https://huggingface.co/datasets?task_categories=task_categories:graph-ml&sort=downloads). We'll focus on using already available ones, but feel free to [add your datasets](https://huggingface.co/docs/datasets/upload_dataset)!

### Loading
Loading a graph dataset from the Hub is very easy. Let's load the `ogbg-mohiv` dataset (a baseline from the [Open Graph Benchmark](https://ogb.stanford.edu/) by Stanford), stored in the `OGB` repository: 

```python
from datasets import load_dataset

# There is only one split on the hub
dataset = load_dataset("OGB/ogbg-molhiv")

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

### Format
On the Hub, graph datasets are mostly stored as lists of graphs (using the `jsonl` format). 

A single graph is a dictionary, and here is the expected format for our graph classification datasets:
- `edge_index` contains the indices of nodes in edges, stored as a list containing two parallel lists of edge indices. 
    - **Type**: list of 2 lists of integers.
    - **Example**: a graph containing four nodes (0, 1, 2 and 3) and where connections are 1->2, 1->3 and 3->1 will have `edge_index = [[1, 1, 3], [2, 3, 1]]`. You might notice here that node 0 is not present here, as it is not part of an edge per se. This is why the next attribute is important.
- `num_nodes` indicates the total number of nodes available in the graph (by default, it is assumed that nodes are numbered sequentially). 
    - **Type**: integer 
    - **Example**: In our above example, `num_nodes = 4`.
- `y` maps each graph to what we want to predict from it (be it a class, a property value, or several binary label for different tasks).
    - **Type**: list of either integers (for multi-class classification), floats (for regression), or lists of ones and zeroes (for binary multi-task classification)
    - **Example**: We could predict the graph size (small = 0, medium = 1, big = 2). Here, `y = [0]`.
- `node_feat` contains the available features (if present) for each node of the graph, ordered by node index.
    - **Type**: list of lists of integer (Optional) 
    - **Example**: Our above nodes could have, for example, types (like different atoms in a molecule). This could give `node_feat = [[1], [0], [1], [1]]`. 
- `edge_attr` contains the available attributes (if present) for each edge of the graph, following the `edge_index` ordering.
    - **Type**: list of lists of integers (Optional)
    - **Example**: Our above edges could have, for example, types (like molecular bonds). This could give `edge_attr = [[0], [1], [1]]`.

### Preprocessing
Graph transformer frameworks usually apply specific preprocessing to their datasets to generate added features and properties which help the underlying learning task (classification in our case).
Here, we use Graphormer's default preprocessing, which generates in/out degree information, the shortest path between node matrices, and other properties of interest for the model. 
 
```python
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

dataset_processed = dataset.map(preprocess_item, batched=False)
```

It is also possible to apply this preprocessing on the fly, in the DataCollator's parameters (by setting `on_the_fly_processing` to True): not all datasets are as small as `ogbg-molhiv`, and for large graphs, it might be too costly to store all the preprocessed data beforehand. 

## Model

### Loading
Here, we load an existing pretrained model/checkpoint and fine-tune it on our downstream task, which is a binary classification task (hence `num_classes = 2`). We could also fine-tune our model on regression tasks (`num_classes = 1`) or on multi-task classification.
```python
from transformers import GraphormerForGraphClassification

model = GraphormerForGraphClassification.from_pretrained(
    "clefourrier/pcqm4mv2_graphormer_base",
    num_classes=2, # num_classes for the downstream task 
    ignore_mismatched_sizes=True,
)
```
Let's look at this in more detail. 

Calling the `from_pretrained` method on our model downloads and caches the weights for us. As the number of classes (for prediction) is dataset dependent, we pass the new `num_classes` as well as `ignore_mismatched_sizes` alongside the `model_checkpoint`. This makes sure a custom classification head is created, specific to our task, hence likely different from the original decoder head.

It is also possible to create a new randomly initialized model to train from scratch, either following the known parameters of a given checkpoint or by manually choosing them.

### Training or fine-tuning
To train our model simply, we will use a `Trainer`. To instantiate it, we will need to define the training configuration and the evaluation metric. The most important is the `TrainingArguments`, which is a class that contains all the attributes to customize the training. It requires a folder name, which will be used to save the checkpoints of the model.

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
For graph datasets, it is particularly important to play around with batch sizes and gradient accumulation steps to train on enough samples while avoiding out-of-memory errors. 

The last argument `push_to_hub` allows the Trainer to push the model to the Hub regularly during training, as each saving step.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed["train"],
    eval_dataset=dataset_processed["validation"],
    data_collator=GraphormerDataCollator(),
)

```
In the `Trainer` for graph classification, it is important to pass the specific data collator for the given graph dataset, which will convert individual graphs to batches for training. 

```python
train_results = trainer.train()
trainer.push_to_hub()
```
When the model is trained, it can be saved to the hub with all the associated training artefacts using `push_to_hub`.

As this model is quite big, it takes about a day to train/fine-tune for 20 epochs on CPU (IntelCore i7). To go faster, you could use powerful GPUs and parallelization instead, by launching the code either in a Colab notebook or directly on the cluster of your choice.


## Ending note
Now that you know how to use `transformers` to train a graph classification model, we hope you will try to share your favorite graph transformer checkpoints, models, and datasets on the Hub for the rest of the community to use!
