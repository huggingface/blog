---
title: "Introduction to Graph Machine Learning" 
thumbnail: /blog/assets/125_intro-to-graphml/thumbnail.png
authors:
- user: clefourrier
---

# Introduction to Graph Machine Learning

{blog_metadata}

{authors}

In this blog post, we cover the basics of graph machine learning. 

We first study what graphs are, why they are used, and how best to represent them. We then cover briefly how people learn on graphs, from pre-neural methods (exploring graph features at the same time) to what are commonly called Graph Neural Networks. Lastly, we peek into the world of Transformers for graphs.

## Graphs

### What is a graph?

In its essence, a graph is a description of items linked by relations.

Examples of graphs include social networks (Twitter, Mastodon, any citation networks linking papers and authors), molecules, knowledge graphs (such as UML diagrams, encyclopedias, and any website with hyperlinks between its pages), sentences expressed as their syntactic trees, any 3D mesh, and more! It is, therefore, not hyperbolic to say that graphs are everywhere.

The items of a graph (or network) are called its *nodes* (or vertices), and their connections its *edges* (or links). For example, in a social network, nodes are users and edges their connections; in a molecule, nodes are atoms and edges their molecular bond.

* A graph with either typed nodes or typed edges is called **heterogeneous** (example: citation networks with items that can be either papers or authors have typed nodes, and XML diagram where relations are typed have typed edges). It cannot be represented solely through its topology, it needs additional information. This post focuses on homogeneous graphs. 
* A graph can also be **directed** (like a follower network, where A follows B does not imply B follows A) or **undirected** (like a molecule, where the relation between atoms goes both ways). Edges can connect different nodes or one node to itself (self-edges), but not all nodes need to be connected.

If you want to use your data, you must first consider its best characterisation (homogeneous/heterogeneous, directed/undirected, and so on).

### What are graphs used for?

Let's look at a panel of possible tasks we can do on graphs.

At the **graph level**, the main tasks are:

- graph generation, used in drug discovery to generate new plausible molecules,
- graph evolution (given a graph, predict how it will evolve over time), used in physics to predict the evolution of systems
- graph level prediction (categorisation or regression tasks from graphs), such as predicting the toxicity of molecules.

At the **node level**, it's usually a node property prediction. For example, [Alphafold](https://www.deepmind.com/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology) uses node property prediction to predict the 3D coordinates of atoms given the overall graph of the molecule, and therefore predict how molecules get folded in 3D space, a hard bio-chemistry problem.

At the **edge level**, it's either edge property prediction or missing edge prediction. Edge property prediction helps drug side effect prediction predict adverse side effects given a pair of drugs. Missing edge prediction is used in recommendation systems to predict whether two nodes in a graph are related.

It is also possible to work at the **sub-graph level** on community detection or subgraph property prediction. Social networks use community detection to determine how people are connected. Subgraph property prediction can be found in itinerary systems (such as [Google Maps](https://www.deepmind.com/blog/traffic-prediction-with-advanced-graph-neural-networks)) to predict estimated times of arrival.

Working on these tasks can be done in two ways. 

When you want to predict the evolution of a specific graph, you work in a **transductive** setting, where everything (training, validation, and testing) is done on the same single graph. *If this is your setup, be careful! Creating train/eval/test datasets from a single graph is not trivial.* However, a lot of the work is done using different graphs (separate train/eval/test splits), which is called an **inductive** setting.

### How do we represent graphs?

The common ways to represent a graph to process and operate it are either:

* as the set of all its edges (possibly complemented with the set of all its nodes)
* or as the adjacency matrix between all its nodes. An adjacency matrix is a square matrix (of node size * node size) that indicates which nodes are directly connected to which others (where \(A_{ij} = 1\) if \(n_i\) and \(n_j\) are connected, else 0). *Note: most graphs are not densely connected and therefore have sparse adjacency matrices, which can make computations harder.*

However, though these representations seem familiar, do not be fooled!

Graphs are very different from typical objects used in ML because their topology is more complex than just "a sequence" (such as text and audio) or "an ordered grid" (images and videos, for example)): even if they can be represented as lists or matrices, their representation should not be considered an ordered object!

But what does this mean? If you have a sentence and shuffle its words, you create a new sentence. If you have an image and rearrange its columns, you create a new image. 

<div align="center">
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/125_intro-to-graphml/assembled_hf.png" width="500" />
  <figcaption>On the left, the Hugging Face logo - on the right, a shuffled Hugging Face logo, which is quite a different new image.</figcaption>
</figure>
</div>

This is not the case for a graph: if you shuffle its edge list or the columns of its adjacency matrix, it is still the same graph. (We explain this more formally a bit lower, look for permutation invariance).

<div align="center">
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/125_intro-to-graphml/assembled_graphs.png" width="1000" />
  <figcaption>On the left, a small graph (nodes in yellow, edges in orange). In the centre, its adjacency matrix, with columns and rows ordered in the alphabetical node order: on the row for node A (first row), we can read that it is connected to E and C. On the right, a shuffled adjacency matrix (the columns are no longer sorted alphabetically), which is also a valid representation of the graph: A is still connected to E and C.</figcaption>
</figure>
</div>

## Graph representations through ML

The usual process to work on graphs with machine learning is first to generate a meaningful representation for your items of interest (nodes, edges, or full graphs depending on your task), then to use these to train a predictor for your target task. We want (as in other modalities) to constrain the mathematical representations of your objects so that similar objects are mathematically close. However, this similarity is hard to define strictly in graph ML: for example, are two nodes more similar when they have the same labels or the same neighbours?

Note: *In the following sections, we will focus on generating node representations. 
Once you have node-level representations, it is possible to obtain edge or graph-level information. For edge-level information, you can concatenate node pair representations or do a dot product. For graph-level information, it is possible to do a global pooling (average, sum, etc.) on the concatenated tensor of all the node-level representations. Still, it will smooth and lose information over the graph -- a recursive hierarchical pooling can make more sense, or add a virtual node, connected to all other nodes in the graph, and use its representation as the overall graph representation.*

### Pre-neural approaches

#### Simply using engineered features

Before neural networks, graphs and their items of interest could be represented as combinations of features, in a task-specific fashion. Now, these features are still used for data augmentation and [semi-supervised learning](https://arxiv.org/abs/2202.08871), though [more complex feature generation methods](https://arxiv.org/abs/2208.11973) exist; it can be essential to find how best to provide them to your network depending on your task. 

**Node-level** features can give information about importance (how important is this node for the graph?) and/or structure based (what is the shape of the graph around the node?), and can be combined.

The node **centrality** measures the node importance in the graph. It can be computed recursively by summing the centrality of each node’s neighbours until convergence, or through shortest distance measures between nodes, for example. The node **degree** is the quantity of direct neighbours it has. The **clustering coefficient** measures how connected the node neighbours are. **Graphlets degree vectors** count how many different graphlets are rooted at a given node, where graphlets are all the mini graphs you can create with a given number of connected nodes (with three connected nodes, you can have a line with two edges, or a triangle with three edges).

<div align="center">
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/125_intro-to-graphml/graphlets.png" width="700" />
  <figcaption>The 2-to 5-node graphlets (Pržulj, 2007)</figcaption>
</figure>
</div>

**Edge-level** features complement the representation with more detailed information about the connectedness of the nodes, and include the **shortest distance** between two nodes, their **common neighbours**, and their **Katz index** (which is the number of possible walks of up to a certain length between two nodes - it can be computed directly from the adjacency matrix).

**Graph level features** contain high-level information about graph similarity and specificities. Total **graphlet counts**, though computationally expensive, provide information about the shape of sub-graphs. **Kernel methods** measure similarity between graphs through different "bag of nodes" methods (similar to bag of words).

### Walk-based approaches

[**Walk-based approaches**](https://en.wikipedia.org/wiki/Random_walk) use the probability of visiting a node j from a node i on a random walk to define similarity metrics; these approaches combine both local and global information. [**Node2Vec**](https://snap.stanford.edu/node2vec/), for example, simulates random walks between nodes of a graph, then processes these walks with a skip-gram, [much like we would do with words in sentences](https://arxiv.org/abs/1301.3781), to compute embeddings. These approaches can also be used to [accelerate computations](https://arxiv.org/abs/1208.3071) of the [**Page Rank method**](http://infolab.stanford.edu/pub/papers/google.pdf), which assigns an importance score to each node (based on its connectivity to other nodes, evaluated as its frequency of visit by random walk, for example).

However, these methods have limits: they cannot obtain embeddings for new nodes, do not capture structural similarity between nodes finely, and cannot use added features.

## Graph Neural Networks

Neural networks can generalise to unseen data. Given the representation constraints we evoked earlier, what should a good neural network be to work on graphs?

It should:

- be permutation invariant:
    - Equation: \\(f(P(G))=f(G)\\) with f the network, P the permutation function, G the graph
    - Explanation: the representation of a graph and its permutations should be the same after going through the network
- be permutation equivariant
    - Equation: \\(P(f(G))=f(P(G))\\) with f the network, P the permutation function, G the graph
    - Explanation: permuting the nodes before passing them to the network should be equivalent to permuting their representations

Typical neural networks, such as RNNs or CNNs are not permutation invariant. A new architecture, the [Graph Neural Network](https://ieeexplore.ieee.org/abstract/document/1517930), was therefore introduced (initially as a state-based machine).

A GNN is made of successive layers. A GNN layer represents a node as the combination (**aggregation**) of the representations of its neighbours and itself from the previous layer (**message passing**), plus usually an activation to add some nonlinearity.

**Comparison to other models**: A CNN can be seen as a GNN with fixed neighbour sizes (through the sliding window) and ordering (it is not permutation equivariant). A [Transformer](https://arxiv.org/abs/1706.03762v3) without positional embeddings can be seen as a GNN on a fully-connected input graph.

### Aggregation and message passing

There are many ways to aggregate messages from neighbour nodes, summing, averaging, for example. Some notable works following this idea include:

- [Graph Convolutional Networks](https://tkipf.github.io/graph-convolutional-networks/) averages the normalised representation of the neighbours for a node (most GNNs are actually GCNs);
- [Graph Attention Networks](https://petar-v.com/GAT/) learn to weigh the different neighbours based on their importance (like transformers);
- [GraphSAGE](https://snap.stanford.edu/graphsage/) samples neighbours at different hops before aggregating their information in several steps with max pooling.
- [Graph Isomorphism Networks](https://arxiv.org/pdf/1810.00826v3.pdf) aggregates representation by applying an MLP to the sum of the neighbours' node representations.

**Choosing an aggregation**: Some aggregation techniques (notably mean/max pooling) can encounter failure cases when creating representations which finely differentiate nodes with different neighbourhoods of similar nodes (ex: through mean pooling, a neighbourhood with 4 nodes, represented as 1,1,-1,-1, averaged as 0, is not going to be different from one with only 3 nodes represented as -1, 0, 1).

### GNN shape and the over-smoothing problem

At each new layer, the node representation includes more and more nodes. 

A node, through the first layer, is the aggregation of its direct neighbours. Through the second layer, it is still the aggregation of its direct neighbours, but this time, their representations include their own neighbours (from the first layer). After n layers, the representation of all nodes becomes an aggregation of all their neighbours at distance n, therefore, of the full graph if its diameter is smaller than n! 

If your network has too many layers, there is a risk that each node becomes an aggregation of the full graph (and that node representations converge to the same one for all nodes). This is called **the oversmoothing problem**

This can be solved by :

- scaling the GNN to have a layer number small enough to not approximate each node as the whole network (by first analysing the graph diameter and shape)
- increasing the complexity of the layers
- adding non message passing layers to process the messages (such as simple MLPs)
- adding skip-connections.

The oversmoothing problem is an important area of study in graph ML, as it prevents GNNs to scale up, like Transformers have been shown to in other modalities.

## Graph Transformers

A Transformer without its positional encoding layer is permutation invariant, and Transformers are known to scale well, so recently, people have started looking at adapting Transformers to graphs ([Survey)](https://github.com/ChandlerBang/awesome-graph-transformer). Most methods focus on the best ways to represent graphs by looking for the best features and best ways to represent positional information and changing the attention to fit this new data.

Here are some interesting methods which got state-of-the-art results or close on one of the hardest available benchmarks as of writing, [Stanford's Open Graph Benchmark](https://ogb.stanford.edu/):

- [*Graph Transformer for Graph-to-Sequence Learning*](https://arxiv.org/abs/1911.07470) (Cai and Lam, 2020) introduced a Graph Encoder, which represents nodes as a concatenation of their embeddings and positional embeddings, node relations as the shortest paths between them, and combine both in a relation-augmented self attention.
- [*Rethinking Graph Transformers with Spectral Attention*](https://arxiv.org/abs/2106.03893) (Kreuzer et al, 2021) introduced Spectral Attention Networks (SANs). These combine node features with learned positional encoding (computed from Laplacian eigenvectors/values), to use as keys and queries in the attention, with attention values being the edge features.
- [*GRPE: Relative Positional Encoding for Graph Transformer*](https://arxiv.org/abs/2201.12787) (Park et al, 2021) introduced the Graph Relative Positional Encoding Transformer. It represents a graph by combining a graph-level positional encoding with node information, edge level positional encoding with node information, and combining both in the attention.
- [*Global Self-Attention as a Replacement for Graph Convolution*](https://arxiv.org/abs/2108.03348) (Hussain et al, 2021) introduced the Edge Augmented Transformer. This architecture embeds nodes and edges separately, and aggregates them in a modified attention.
- [*Do Transformers Really Perform Badly for Graph Representation*](https://arxiv.org/abs/2106.05234) (Ying et al, 2021) introduces Microsoft's [**Graphormer**](https://www.microsoft.com/en-us/research/project/graphormer/), which won first place on the OGB when it came out. This architecture uses node features as query/key/values in the attention, and sums their representation with a combination of centrality, spatial, and edge encodings in the attention mechanism.

The most recent approach is [*Pure Transformers are Powerful Graph Learners*](https://arxiv.org/abs/2207.02505) (Kim et al, 2022), which introduced **TokenGT**. This method represents input graphs as a sequence of node and edge embeddings (augmented with orthonormal node identifiers and trainable type identifiers), with no positional embedding, and provides this sequence to Transformers as input. It is extremely simple, yet smart!

A bit different, [*Recipe for a General, Powerful, Scalable Graph Transformer*](https://arxiv.org/abs/2205.12454) (Rampášek et al, 2022) introduces, not a model, but a framework, called **GraphGPS**. It allows to combine message passing networks with linear (long range) transformers to create hybrid networks easily. This framework also contains several tools to compute positional and structural encodings (node, graph, edge level), feature augmentation, random walks, etc.

Using transformers for graphs is still very much a field in its infancy, but it looks promising, as it could alleviate several limitations of GNNs, such as scaling to larger/denser graphs, or increasing model size without oversmoothing.

# Further resources

If you want to delve deeper, you can look at some of these courses:

- Academic format
    - [Stanford's Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/)
    - [McGill's Graph Representation Learning](https://cs.mcgill.ca/~wlh/comp766/)
- Video format
    - [Geometric Deep Learning course](https://www.youtube.com/playlist?list=PLn2-dEmQeTfSLXW8yXP4q_Ii58wFdxb3C)
- Books
    - [Graph Representation Learning*, Hamilton](https://www.cs.mcgill.ca/~wlh/grl_book/)
- Surveys
    - [Graph Neural Networks Study Guide](https://github.com/dair-ai/GNNs-Recipe)
- Research directions
    - [GraphML in 2023](https://towardsdatascience.com/graph-ml-in-2023-the-state-of-affairs-1ba920cb9232) summarizes plausible interesting directions for GraphML in 2023.

Nice libraries to work on graphs are [PyGeometric](https://pytorch-geometric.readthedocs.io/en/latest/) or the [Deep Graph Library](https://www.dgl.ai/) (for graph ML) and [NetworkX](https://networkx.org/) (to manipulate graphs more generally).

If you need quality benchmarks you can check out:

- [OGB, the Open Graph Benchmark](https://ogb.stanford.edu/): the reference graph benchmark datasets, for different tasks and data scales.
- [Benchmarking GNNs](https://github.com/graphdeeplearning/benchmarking-gnns): Library and datasets to benchmark graph ML networks and their expressivity. The associated paper notably studies which datasets are relevant from a statistical standpoint, what graph properties they allow to evaluate, and which datasets should no longer be used as benchmarks.
- [Long Range Graph Benchmark](https://github.com/vijaydwivedi75/lrgb): recent (Nov2022) benchmark looking at long range graph information
- [Taxonomy of Benchmarks in Graph Representation Learning](https://openreview.net/pdf?id=EM-Z3QFj8n): paper published at the 2022 Learning on Graphs conference, which analyses and sort existing benchmarks datasets

For more datasets, see:

- [Paper with code Graph tasks Leaderboards](https://paperswithcode.com/area/graphs): Leaderboard for public datasets and benchmarks - careful, not all the benchmarks on this leaderboard are still relevant
- [TU datasets](https://chrsmrrs.github.io/datasets/docs/datasets/): Compilation of publicly available datasets, now ordered by categories and features. Most of these datasets can also be loaded with PyG, and a number of them have been ported to Datasets
- [SNAP datasets: Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/):
- [MoleculeNet datasets](https://moleculenet.org/datasets-1)
- [Relational datasets repository](https://relational.fit.cvut.cz/)


### External images attribution
Emojis in the thumbnail come from Openmoji (CC-BY-SA 4.0), the Graphlets figure comes from *Biological network comparison using graphlet degree distribution* (Pržulj, 2007). 
