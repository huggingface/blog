---
title: "The Reformer - Pushing the limits of language modeling"
thumbnail: https://huggingface.co/blog/assets/03_reformer/thumbnail.png
---

<h1 class="no-top-margin">The Reformer - Pushing the limits of language modeling</h1>

<div class="blog-metadata">
    <small>Published July 3, 2020.</small>
    <a target="_blank" class="btn-readme" href="https://github.com/huggingface/blog/blob/master/reformer.md">
        <img src="/front/assets/icon-github.svg">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/patrickvonplaten">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1584435275418-5dfcb1aada6d0311fd3d5448.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>patrickvonplaten</code>
            <span class="fullname">Patrick von Platen</span>
        </div>
    </a>
</div>

<a href="https://colab.research.google.com/github/patrickvonplaten/blog/blob/add_reformer_notebook/Reformer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## ***How the Reformer uses less than 8GB of RAM to train on sequences of half a million tokens***

The Reformer model as introduced by [Kitaev, Kaiser et al. (2020)](https://arxiv.org/pdf/2001.04451.pdf) is one of the most memory-efficient transformer models for long sequence modeling as of today.

Recently, long sequence modeling has experienced a surge of interest as can be seen by the many submissions from this year alone - [Beltagy et al. (2020)](https://arxiv.org/abs/2004.05150), [Roy et al. (2020)](https://arxiv.org/abs/2003.05997), [Tay et al.](https://arxiv.org/abs/2002.11296), [Wang et al.](https://arxiv.org/abs/2006.04768) to name  a few. 
The motivation behind long sequence modeling is that many tasks in NLP, *e.g.* summarization, question answering, require the model to process longer input sequences than models, such as BERT, are able to handle. In tasks that require the model to process a large input sequence, long sequence models do not have to cut the input sequence to avoid memory overflow and thus have been shown to outperform standard "BERT"-like models *cf.* [Beltagy et al. (2020)](https://arxiv.org/abs/2004.05150). 

The Reformer pushes the limit of longe sequence modeling by its ability to process up to half a million tokens at once as shown in this [demo](https://github.com/patrickvonplaten/notebooks/blob/master/PyTorch_Reformer.ipynb). As a comparison, a conventional `bert-base-uncased` model limits the input length to only 512 tokens. In Reformer, each part of the standard transformer architecture is re-engineered to optimize for minimal memory requirement without a significant drop in performance.

The memory improvements can be attributed to **4** features which the Reformer authors introduced to the transformer world:

1.   **Reformer Self-Attention Layer** - *How to efficiently implement self-attention without being restricted to a local context?*
2.  **Chunked Feed Forward Layers** - *How to get a better time-memory trade-off for large feed forward layers?*
3.   **Reversible Residual Layers**  - *How to drastically reduce memory consumption in training by a smart residual architecture?*
4.   **Axial Positional Encodings** - *How to make positional encodings usable for extremely large input sequences?*

The goal of this blog post is to give the reader an **in-depth** understanding of each of the four Reformer features mentioned above. While the explanations are focussed on the Reformer, the reader should get a better intuition under which circumstances each of the four features can be effective for other transformer models as well. 
The four sections are only loosely connected, so they can very well be read individually.

Reformer is part of the 🤗Transformers library. For all users of the Reformer, it is advised to go through this very detailed blog post to better understand how the model works and how to correctly set its configuration. All equations are accompanied by their equivalent name for the Reformer config, *e.g.* `config.<param_name>`, so that the reader can quickly relate to the official docs and configuration file.

**Note**: *Axial Positional Encodings* are not explained in the official Reformer paper, but are extensively used in the official codebase. This blog post gives the first in-depth explanation of Axial Positional Encodings.

## 1. Reformer Self-Attention Layer

Reformer uses two kinds of special self-attention layers: *local* self-attention layers and Locality Sensitive Hashing (*LSH*) self-attention layers.

To better introduce these new self-attention layers, we will briefly recap 
conventional self-attention as introduced in [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762).

This blog post uses the same notation and coloring as the popular blog post [The illustrated transformer](http://jalammar.github.io/illustrated-transformer/), so the reader is strongly advised to read this blog first. 

**Important**: While Reformer was originally introduced for causal self-attention, it can very well be used for bi-directional self-attention as well. In this post, Reformer's self-attention is presented for *bidirectional* self-attention.

### Recap Global Self-Attention

The core of every Transformer model is the **self-attention** layer. To recap the conventional self-attention layer, which we refer to here as the **global self-attention** layer, let us assume we apply a transformer layer on the embedding vector sequence \\(\mathbf{X} = \mathbf{x}_1, \ldots, \mathbf{x}_n\\) where each vector \\(\mathbf{x}_{i}\\) is of size `config.hidden_size`, *i.e.* \\(d_h\\). 

In short, a global self-attention layer projects \\(\mathbf{X}\\) to the query, key and value matrices \\(\mathbf{Q}, \mathbf{K}, \mathbf{V}\\) and computes the output \\(\mathbf{Z}\\) using the *softmax* operation as follows:
 \\(\mathbf{Z} = \text{SelfAttn}(\mathbf{X}) = \text{softmax}(\mathbf{Q}\mathbf{K}^T) \mathbf{V}\\) with \\(\mathbf{Z}\\) being of dimension \\(d_h \times n\\) (leaving out the key normalization factor and self-attention weights \\(\mathbf{W}^{O}\\) for simplicity). For more detail on the complete transformer operation, see [the illustrated transformer](http://jalammar.github.io/illustrated-transformer/).

Visually, we can illustrate this operation as follows for \\(n=16, d_h=3\\):

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/conventional_attention.png)

Note that for all visualizations `batch_size` and `config.num_attention_{h}eads` is assumed to be 1. Some vectors, *e.g.* \\(\mathbf{x_3}\\) and its corresponding output vector \\(\mathbf{z_3}\\) are marked so that *LSH self-attention* can later be better explained. The presented logic can effortlessly be extended for multi-head self-attention (`config.num_attention_{h}eads` > 1). The reader is advised to read [the illustrated transformer](http://jalammar.github.io/illustrated-transformer/) as a reference for multi-head self-attention.

Important to remember is that for each output vector \\(\mathbf{z}_{i}\\), the whole input sequence \\(\mathbf{X}\\) is processed. The tensor of the inner dot-product \\(\mathbf{Q}\mathbf{K}^T\\) has an asymptotic memory complexity of \\(\mathcal{O}(n^2)\\) which usually represents the memory bottleneck in a transformer model. 

This is also the reason why `bert-base-cased` has a `config.max_position_embedding_size` of only 512.

### Local Self-Attention

 **Local self-attention** is the obvious solution to reducing the \\(\mathcal{O}(n^2)\\) memory bottleneck, allowing us to model longer sequences with a reduced computational cost. 
In local self-attention the input \\( \mathbf{X} = \mathbf{X}_{1:n} = \mathbf{x}_{1}, \ldots, \mathbf{x}_{n} \\) 
is cut into \\(n_{c}\\) chunks: \\( \mathbf{X} = \left[\mathbf{X}_{1:l_{c}}, \ldots, \mathbf{X}_{(n_{c} - 1) * l_{c} : n_{c} * l_{c}}\right] \\) each 
of length `config.local_chunk_length`, *i.e.* \\(l_{c}\\), and subsequently global self-attention is applied on each chunk separately.

Let's take our input sequence for \\(n=16, d_h=3\\) again for visualization:

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/input.png)

Assuming \\(l_{c} = 4, n_{c} = 4\\), chunked attention can be illustrated as follows:

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/chunked_attention_1.png)

As can be seen, the attention operation is applied for each chunk \\(\mathbf{X}_{1:4}, \mathbf{X}_{5:8}, \mathbf{X}_{9:12}, \mathbf{X}_{13:16}\\) individually.
The first drawback of this architecture becomes obvious: Some input vectors have no access to their immediate context, *e.g.* \\(\mathbf{x}_9\\) has no access to \\(\mathbf{x}_{8}\\) and vice-versa in our example. This is problematic because these tokens are not able to learn word representations that take their immediate context into account.

A simple remedy is to augment each chunk with `config.local_num_chunks_before`, *i.e.* \\(n_{p}\\), chunks and `config.local_num_chunks_after`, *i.e.* \\(n_{a}\\), so that every input vector has at least access to \\(n_{p}\\) previous input vectors and \\(n_{a}\\) following input vectors. This can also be understood as chunking with overlap whereas \\(n_{p}\\) and \\(n_{a}\\) define the amount of overlap each chunk has with all previous chunks and following chunks. We denote this extended local self-attention as follows: 

$$\mathbf{Z}^{\text{loc}} = \left[\mathbf{Z}_{1:l_{c}}^{\text{loc}}, \ldots, \mathbf{Z}_{(n_{c} - 1) * l_{c} : n_{c} * l_{c}}^{\text{loc}}\right] \text{, with } \mathbf{Z}_{l_{c} * (i - 1) + 1 : l_{c} * i}^{\text{loc}} = \text{SelfAttn}(\mathbf{X}_{l_{c} * (i - 1 - n_{p}) + 1: l_{c} * (i + n_{a})})\left[n_{p} * l_{c}: -n_{a} * l_{c}\right], \forall i \in \{1, \ldots, n_{c} \}$$

Okay, this formula looks quite complicated. Let's make it easier.
In Reformer's self-attention layers \\(n_{a}\\) is usually set to 0 and \\(n_{p}\\) is set to 1, so let's write down the formula again for \\(i = 1\\):

$$\mathbf{Z}_{1:l_{c}}^{\text{loc}} = \text{SelfAttn}(\mathbf{X}_{-l_{c} + 1: l_{c}})\left[l_{c}:\right]$$

We notice that we have a circular relationship so that the first segment can attend the last segment as well. Let's illustrate this slightly enhanced local attention again. First, we apply self-attention within each windowed segment and keep only the central output segment.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/local_attention_2.png)

Finally, the relevant output is concatenated to \\(\mathbf{Z}^{\text{loc}}\\) and looks as follows.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/local_attention_3.png)

Note that local self-attention is implemented efficiently way so that no output is computed and subsequently "thrown-out" as shown here for illustration purposes by the red cross.

It's important to note here that extending the input vectors for each chunked self-attention function allows *each* single output vector \\( \mathbf{z}_{i} \\) of this self-attention function to learn better vector representations. E.g. each of the output vectors \\( \mathbf{z}_{5}^{\text{loc}}, \mathbf{z}_{6}^{\text{loc}}, \mathbf{z}_{7}^{\text{loc}}, \mathbf{z}_{8}^{\text{loc}} \\) can take into account all of the input vectors \\( \mathbf{X}_{1:8} \\) to learn better representations.

The gain in memory consumption is quite obvious: The \\( \mathcal{O}(n^2) \\) memory complexity is broken down for each segment individually so that the total asymptotic memory consumption is reduced to \\( \mathcal{O}(n_{c} * l_{c}^2) = \mathcal{O}(n * l_{c}) \\).

This enhanced local self-attention is better than the vanilla local self-attention architecture but still has a major drawback in that every input vector can only attend to a local context of predefined size. For NLP tasks that do not require the transformer model to learn long-range dependencies between the input vectors, which include arguably *e.g.* speech recognition, named entity recognition and causal language modeling of short sentences, this might not be a big issue. Many NLP tasks do require the model to learn long-range dependencies, so that local self-attention could lead to significant performance degradation, *e.g.* 
* *Question-answering*: the model has to learn the relationship between the question tokens and relevant answer tokens which will most likely not be in the same local range
* *Multiple-Choice*: the model has to compare multiple answer token segments to each other which are usually separated by a significant length
* *Summarization*: the model has to learn the relationship between a long sequence of context tokens and a shorter sequence of summary tokens, whereas the relevant relationships between context and summary can most likely not be captured by local self-attention
* etc...

Local self-attention on its own is most likely not sufficient for the transformer model to learn the relevant relationships of input vectors (tokens) to each other.

Therefore, Reformer additionally employs an efficient self-attention layer that approximates global self-attention, called *LSH self-attention*.

### LSH Self-Attention

Alright, now that we have understood how local self-attention works, we can take a stab at the probably most innovative piece of Reformer: **Locality sensitive hashing (LSH) Self-Attention**. 

The premise of LSH self-attention is to be more or less as efficient as local self-attention while approximating global self-attention.

LSH self-attention relies on the LSH algorithm as presented in [Andoni et al (2015)](https://arxiv.org/abs/1509.02897), hence its name.

The idea behind LSH self-attention is based on the insight that if \\(n\\) is large, the softmax applied on the \\(\mathbf{Q}\mathbf{K}^T\\) attention dot-product weights only very few value vectors  with values significantly larger than 0 for each query vector. 

Let's explain this in more detail.
Let \\(\mathbf{k}_{i} \in \mathbf{K} = \left[\mathbf{k}_1, \ldots, \mathbf{k}_n \right]^T\\) and \\(\mathbf{q}_{i} \in \mathbf{Q} = \left[\mathbf{q}_1, \ldots, \mathbf{q}_n\right]^T\\) be the key and query vectors. For each \\(\mathbf{q}_{i}\\), the computation \\(\text{softmax}(\mathbf{q}_{i}^T \mathbf{K}^T)\\) can be approximated by using only those key vectors of \\(\mathbf{k}_{j}\\) that have a high cosine similarity with \\(\mathbf{q}_{i}\\). This owes to the fact that the softmax function puts exponentially more weight on larger input values.
So far so good, the next problem is to efficiently find the vectors that have a
high cosine similarity with \\(\mathbf{q}_{i}\\) for all \\(i\\).

First, the authors of Reformer notice that sharing the query and key projections: \\(\mathbf{Q} = \mathbf{K}\\) does not impact the performance of a transformer model \\({}^1\\). Now, instead of having to find the key vectors of high cosine similarity for each query vector \\(q_i\\), only the cosine similarity of query vectors to each other has to be found. 
This is important because there is a transitive property to the query-query vector dot product approximation: If \\(\mathbf{q}_{i}\\) has a high cosine similarity to the query vectors \\(\mathbf{q}_{j}\\) and \\(\mathbf{q}_{k}\\), then \\(\mathbf{q}_{j}\\) also has a high cosine similarity to \\(\mathbf{q}_{k}\\). Therefore, the query vectors can be clustered into buckets, such that all query vectors that belong to the same bucket have a high cosine similarity to each other. Let's define \\(C_{m}\\) as the *mth* set of position indices, such that their corresponding query vectors are in the same bucket: \\(C_{m} = \{ i | \text{ s.t. } \mathbf{q}_{i} \in \text{mth cluster}\}\\) and `config.num_buckets`, *i.e.* \\(n_{b}\\), as the number of buckets.

For each set of indices \\(C_{m}\\), the softmax function on the corresponding bucket of query vectors \\(\text{softmax}(\mathbf{Q}_{i \in C_{m}} \mathbf{Q}^T_{i \in C_{m}})\\)  approximates the softmax function of global self-attention with shared query and key projections \\(\text{softmax}(\mathbf{q}_{i}^T \mathbf{Q}^T)\\) for all position indices \\(i\\) in \\(C_{m}\\).

Second, the authors make use of the **LSH** algorithm to cluster the query vectors into a predefined number of buckets \\(n_{b}\\). The LSH algorithm is an ideal choice here because it is very efficient and is an approximation of the nearest neighbor algorithm for cosine similarity. Explaining the LSH scheme is out-of-scope for this notebook, so let's just keep in mind that for each vector \\(\mathbf{q}_{i}\\) the LSH algorithm attributes its position index \\(i\\) to one of \\(n_{b}\\) predefined buckets, *i.e.* \\(\text{LSH}(\mathbf{q}_{i}) = m\\) with \\(i \in \{1, \ldots, n\}\\) and \\(m \in \{1, \ldots, n_{b}\}\\).

Visually, we can illustrate this as follows for our original example:

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/lsh_hashing.png)

Third, it can be noted that having clustered all query vectors in \\(n_{b}\\) buckets, the corresponding set of indices \\(C_{m}\\) can be used to permute the input vectors \\(\mathbf{x}_1, \ldots, \mathbf{x}_n\\) accordingly \\({}^2\\) so that shared query-key self-attention can be applied piecewise similar to local attention. 

Let's clarify with our example input vectors \\(\mathbf{X} = \mathbf{x}_1, ..., \mathbf{x}_{16}\\) and assume `config.num_buckets=4` and `config.lsh_chunk_length = 4`. Looking at the graphic above we can see that we have assigned each query vector \\( \mathbf{q}_1, \ldots, \mathbf{q}_{16} \\) to one of the clusters \\( \mathcal{C}_{1}, \mathcal{C}_{2}, \mathcal{C}_{3}, \mathcal{C}_{4} \\) . 
If we now sort the corresponding input vectors \\( \mathbf{x}_1, \ldots, \mathbf{x}_{16} \\) accordingly, we get the following permuted input \\( \mathbf{X'} \\):

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/lsh_perm.png)

The self-attention mechanism should be applied for each cluster individually so that for each cluster \\( \mathcal{C}_m \\) the corresponding output is calculated as follows: \\( \mathbf{Z}^{\text{LSH}}_{i \in \mathcal{C}_m} = \text{SelfAttn}_{\mathbf{Q}=\mathbf{K}}(\mathbf{X}_{i \in \mathcal{C}_m}) \\).

Let's illustrate this again for our example.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/lsh_cluster_attn.png)

As can be seen, the self-attention function operates on different sizes of matrices, which is suboptimal for efficient batching in GPU and TPU. 

To overcome this problem, the permuted input can be chunked the same way it is done for local attention so that each chunk is of size `config.lsh_chunk_length`. By chunking the permuted input, a bucket might be split into two different chunks. To remedy this problem, in LSH self-attention each chunk attends to its previous chunk `config.lsh_num_chunks_before=1` in addition to itself, the same way local self-attention does (`config.lsh_num_chunks_after` is usually set to 0). This way, we can be assured that all vectors in a bucket attend to each other with a high probability \\({}^3\\).

All in all for all chunks \\( k \in \{1, \ldots, n_{c}\} \\), LSH self-attention can be noted down as follows:

$$ \mathbf{Z'}_{l_{c} * k + 1:l_{c} * (k + 1)}^{\text{LSH}} = \text{SelfAttn}_{\mathbf{Q} = \mathbf{K}}(\mathbf{X'}_{l_{c} * k + 1): l_{c} * (k + 1)})\left[l_{c}:\right] $$

with \\(\mathbf{X'}\\) and \\( \mathbf{Z'} \\) being the input and output vectors permuted according to the LSH algorithm.
Enough complicated formulas, let's illustrate LSH self-attention.

The permuted vectors \\(\mathbf{X'}\\) as shown above are chunked and shared query key self-attention is applied to each chunk.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/lsh_attention_2.png)

Finally, the output \\(\mathbf{Z'}^{\text{LSH}}\\) is reordered to its original permutation.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/lsh_attention_3.png)

One important feature to mention here as well is that the accuracy of LSH self-attention can be improved by running LSH self-attention `config.num_hashes`, e.g. \\(n_{h} \\) times in parallel, each with a different random LSH hash. 
By setting `config.num_hashes > 1`, for each output position \\( i \\), multiple output vectors \\( \mathbf{z}^{\text{LSH}, 1}_{i}, \ldots, \mathbf{z}^{\text{LSH}, n_{h}}_{i} \\) are computed 
and subsequently merged: \\( \mathbf{z}^{\text{LSH}}_{i} = \sum_k^{n_{h}} \mathbf{Z}^{\text{LSH}, k}_{i} * \text{weight}^k_i \\). The \\( \text{weight}^k_i \\) represents the importance of the output vectors \\( \mathbf{z}^{\text{LSH}, k}_{i} \\) of hashing round \\( k \\) in comparison to the other hashing rounds, and is exponentially proportional to the normalization term of their softmax computation. The intuition behind this is that if the corresponding query vector \\( \mathbf{q}_{i}^{k} \\) have a high cosine similarity with all other query vectors in its respective chunk, then the softmax normalization term of this chunk tends to be high, so that the corresponding output vectors \\( \mathbf{q}_{i}^{k} \\) should be a better approximation to global attention and thus receive more weight than output vectors of hashing rounds with a lower softmax normalization term. For more detail see Appendix A of the [paper](https://arxiv.org/pdf/2001.04451.pdf). For our example, multi-round LSH self-attention can be illustrated as follows.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/lsh_attention_4.png)

Great. That's it. Now we know how LSH self-attention works in Reformer. 

Regarding the memory complexity, we now have two terms that compete which each other to be the memory bottleneck: the dot-product: \\( \mathcal{O}(n_{h} * n_{c} * l_{c}^2) = \mathcal{O}(n * n_{h} * l_{c}) \\) and the required memory for LSH bucketing: \\( \mathcal{O}(n * n_{h} * \frac{n_{b}}{2}) \\) with \\( l_{c} \\) being the chunk length. Because for large \\( n \\), the number of buckets \\( \frac{n_{b}}{2} \\) grows much faster than the chunk length \\( l_{c} \\), the user can again factorize the number of buckets `config.num_buckets` as explained [here](https://huggingface.co/transformers/model_doc/reformer.html#lsh-self-attention).

Let's recap quickly what we have gone through above:

1. We want to approximate global attention using the knowledge that the softmax operation only puts significant weights on very few key vectors.
2. If key vectors are equal to query vectors this means that *for each* query vector \\( \mathbf{q}_{i} \\), the softmax only puts significant weight on other query vectors that are similar in terms of cosine similarity.
3. This relationship works in both ways, meaning if \\( \mathbf{q}_{j} \\) is similar to \\( \mathbf{q}_{i} \\) than \\(\mathbf{q}_{j} \\) is also similar to \\( \mathbf{q}_{i} \\), so that we can do a global clustering before applying self-attention on a permuted input.
4. We apply local self-attention on the permuted input and re-order the output to its original permutation.

---
 \\( {}^{1} \\) The authors run some preliminary experiments confirming that shared query key self-attention performs more or less as well as standard self-attention.

 \\( {}^{2} \\) To be more exact the query vectors within a bucket are sorted according to their original order. This means if, *e.g.* the vectors \\( \mathbf{q}_1, \mathbf{q}_3, \mathbf{q}_7 \\) are all hashed to bucket 2, the order of the vectors in bucket 2 would still be \\( \mathbf{q}_1 \\), followed by \\( \mathbf{q}_3 \\) and \\( \mathbf{q}_7 \\).

 \\( {}^3 \\) On a side note, it is to mention the authors put a mask on the query vector \\( \mathbf{q}_{i} \\) to prevent the vector from attending to itself. Because the cosine similarity of a vector to itself will always be as high or higher than the cosine similarity to other vectors, the query vectors in shared query key self-attention are strongly discouraged to attend to themselves.



### Benchmark

Benchmark tools were recently added to Transformers - see [here](https://github.com/huggingface/transformers/blob/master/notebooks/05-benchmark.ipynb) for a more detailed explanation.

To show how much memory can be saved using "local" + "LSH" self-attention, the Reformer model `google/reformer-enwik8` is benchmarked for different `local_attn_chunk_length` and `lsh_attn_chunk_length`. The default configuration and usage of the `google/reformer-enwik8` model can be checked in more detail [here](https://huggingface.co/google/reformer-enwik8).

Let's first do some necessary imports and installs.


```
#@title Installs and Imports
# pip installs
!pip -qq install git+https://github.com/huggingface/transformers.git
!pip install -qq py3nvml

from transformers import ReformerConfig, PyTorchBenchmark, PyTorchBenchmarkArguments
```

First, let's benchmark the memory usage of the Reformer model using *global* self-attention. This can be achieved by setting `lsh_attn_chunk_length` = `local_attn_chunk_length` = 8192 so that for all input sequences smaller or equal to 8192, the model automatically switches to global self-attention.


```
config = ReformerConfig.from_pretrained("google/reformer-enwik8", lsh_attn_chunk_length=16386, local_attn_chunk_length=16386, lsh_num_chunks_before=0, local_num_chunks_before=0)
benchmark_args = PyTorchBenchmarkArguments(sequence_lengths=[2048, 4096, 8192, 16386], batch_sizes=[1], models=["Reformer"], no_speed=True, no_env_print=True)
benchmark = PyTorchBenchmark(configs=[config], args=benchmark_args)
result = benchmark.run()
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1279.0, style=ProgressStyle(description…


    
    1 / 1
    Doesn't fit on GPU. CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 11.17 GiB total capacity; 8.87 GiB already allocated; 1.92 GiB free; 8.88 GiB reserved in total by PyTorch)
    
    ====================      INFERENCE - MEMORY - RESULT       ====================
    --------------------------------------------------------------------------------
              Model Name             Batch Size     Seq Length    Memory in MB 
    --------------------------------------------------------------------------------
               Reformer                  1              2048            1465     
               Reformer                  1              4096            2757     
               Reformer                  1              8192            7893     
               Reformer                  1             16386            N/A      
    --------------------------------------------------------------------------------


The longer the input sequence, the more visible is the quadratic relationship \\( \mathcal{O}(n^2) \\) between input sequence and peak memory usage. As can be seen, in practice it would require a much longer input sequence to clearly observe that doubling the input sequence quadruples the peak memory usage.

For this a `google/reformer-enwik8` model using global attention, a sequence length of over 16K results in a memory overflow.

Now, let's activate *local* and *LSH* self-attention by using the model's default parameters.


```
  config = ReformerConfig.from_pretrained("google/reformer-enwik8")
  benchmark_args = PyTorchBenchmarkArguments(sequence_lengths=[2048, 4096, 8192, 16384, 32768, 65436], batch_sizes=[1], models=["Reformer"], no_speed=True, no_env_print=True)
  benchmark = PyTorchBenchmark(configs=[config], args=benchmark_args)
  result = benchmark.run()
```

    1 / 1
    Doesn't fit on GPU. CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 11.17 GiB total capacity; 7.85 GiB already allocated; 1.74 GiB free; 9.06 GiB reserved in total by PyTorch)
    Doesn't fit on GPU. CUDA out of memory. Tried to allocate 4.00 GiB (GPU 0; 11.17 GiB total capacity; 6.56 GiB already allocated; 3.99 GiB free; 6.81 GiB reserved in total by PyTorch)
    
    ====================      INFERENCE - MEMORY - RESULT       ====================
    --------------------------------------------------------------------------------
              Model Name             Batch Size     Seq Length    Memory in MB 
    --------------------------------------------------------------------------------
               Reformer                  1              2048            1785     
               Reformer                  1              4096            2621     
               Reformer                  1              8192            4281     
               Reformer                  1             16384            7607     
               Reformer                  1             32768            N/A      
               Reformer                  1             65436            N/A      
    --------------------------------------------------------------------------------


As expected using local and LSH self-attention is much more memory efficient for longer input sequences, so that the model runs out of memory only at 16K tokens for a 11GB RAM GPU in this notebook.

## 2. Chunked Feed Forward Layers

Transformer-based models often employ very large feed forward layers after the self-attention layer in parallel. Thereby, this layer can take up a significant amount of the overall memory and sometimes even represent the memory bottleneck of a model.
First introduced in the Reformer paper, feed forward chunking is a technique that allows to effectively trade better memory consumption for increased time consumption.


### Chunked Feed Forward Layer in Reformer

In Reformer, the *LSH*- or *local* self-attention layer is usually followed by a residual connection, which then defines the first part in a *transformer block*. For more detail on this please refer to this [blog](http://jalammar.github.io/illustrated-transformer/). 

The output of the first part of the *transformer block*, called *normed self-attention* output can be written as \\( \mathbf{\overline{Z}} = \mathbf{Z} + \mathbf{X} \\), with \\( \mathbf{Z} \\) being either \\( \mathbf{Z}^{\text{LSH}} \\) or \\( \mathbf{Z}^\text{loc} \\) in Reformer.

For our example input \\( \mathbf{x}_1, \ldots, \mathbf{x}_{16} \\), we illustrate the normed self-attention output as follows.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/layer_normed_output.png)

Now, the second part of a *transformer block* usually consists of two feed forward layers \\( ^{1} \\), defined as \\( \text{Linear}_{\text{int}}(\ldots) \\) that processes \\( \mathbf{\overline{Z}} \\), to an intermediate output \\( \mathbf{Y}_{\text{int}} \\) and \\( \text{Linear}_{\text{out}}(\ldots) \\) that processes the intermediate output to the output \\( \mathbf{Y}_{\text{out}} \\). The two feed forward layers can be defined by 

$$\mathbf{Y}_{\text{out}} = \text{Linear}_{\text{out}}(\mathbf{Y}_\text{int}) = 
\text{Linear}_{\text{out}}(\text{Linear}_{\text{int}}(\mathbf{\overline{Z}})).$$

It is important to remember at this point that mathematically the output of a feed forward layer at position \\( \mathbf{y}_{\text{out}, i} \\) only depends on the input at this position \\( \mathbf{\overline{y}}_{i} \\). In contrast to the self-attention layer, every output \\( \mathbf{y}_{\text{out}, i} \\) is therefore completely independent of all inputs \\( \mathbf{\overline{y}}_{j \ne i} \\) of different positions.

Let's illustrate the feed forward layers for \\( \mathbf{\overline{z}}_1, \ldots, \mathbf{\overline{z}}_{16} \\).

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/feed_forward.png)

As can be depicted from the illustration, all input vectors \\( \mathbf{\overline{z}}_{i} \\) are processed by the same feed forward layer in parallel.

It becomes interesting when one takes a look at the output dimensions of the feed forward layers. In Reformer, the output dimension of \\( \text{Linear}_{\text{int}} \\) is defined as `config.feed_forward_size`, *e.g.* \\( d_{f} \\), and the output dimension of \\( \text{Linear}_{\text{int}} \\) is defined as `config.hidden_size`, *i.e.* \\( d_{h} \\). 

The Reformer authors observed that in a transformer model the intermediate dimension \\( d_{f} \\) usually tends to be much larger than the output dimension\\( ^{2} \\) \\( d_{h} \\). This means that the tensor \\( \mathbf{\mathbf{Y}}_\text{int} \\) of dimension \\( d_{f} \times n \\) allocates a significant amount of the total memory and can even become the memory bottleneck.

To get a better feeling for the differences in dimensions let's picture the matrices \\( \mathbf{Y}_\text{int} \\) and \\( \mathbf{Y}_\text{out} \\) for our example.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/feed_forward_matrix.png)

It is becoming quite obvious that the tensor \\( \mathbf{Y}_\text{int} \\) holds much more memory (\\( \frac{d_{f}}{d_{h}} \times n \\) as much to be exact) than \\( \mathbf{Y}_{\text{out}} \\). But, is it even necessary to compute the full intermediate matrix \\( \mathbf{Y}_\text{int} \\) ? Not really, because relevant is only the output matrix \\( \mathbf{Y}_\text{out} \\). 
To trade memory for speed, one can thus chunk the linear layers computation to only process one chunk at the time. Defining `config.chunk_size_feed_forward` as \\( c_{f} \\), chunked linear layers are defined as \\( \mathbf{Y}_{\text{out}} = \left[\mathbf{Y}_{\text{out}, 1: c_{f}}, \ldots, \mathbf{Y}_{\text{out}, (n - c_{f}): n}\right] \\) with \\( \mathbf{Y}_{\text{out}, (c_{f} * i): (i * c_{f} + i)} = \text{Linear}_{\text{out}}(\text{Linear}_{\text{int}}(\mathbf{\overline{Z}}_{(c_{f} * i): (i * c_{f} + i)})) \\). 
In practice, it just means that the output is incrementally computed and concatenated to avoid having to store the whole intermediate tensor \\( \mathbf{Y}_{\text{int}} \\) in memory.

Assuming \\( c_{f}=1 \\) for our example we can illustrate the incremental computation of the output for position \\( i=9 \\) as follows. 

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/chunked_feed_forward.png)

By processing the inputs in chunks of size 1, the only tensors that have to be stored in memory at the same time are \\( \mathbf{Y}_\text{out} \\) of a maximum size of \\( 16 \times d_{h} \\), \\( \mathbf{y}_{\text{int}, i} \\) of size \\( d_{f} \\) and the input \\( \mathbf{\overline{Z}} \\) of size \\( 16 \times d_{h} \\), with \\( d_{h} \\) being `config.hidden_size`\\( ^{3} \\).

Finally, it is important to remember that *chunked linear layers* yield a mathematically equivalent output to conventional linear layers and can therefore be applied to all transformer linear layers. Making use of `config.chunk_size_feed_forward` therefore allows a better trade-off between memory and speed in certain use cases.

---
 \\( {}^1 \\) For a simpler explanation, the layer norm layer which is normally applied to \\( \mathbf{\overline{Z}} \\) before being processed by the feed forward layers is omitted for now.

 \\( {}^2 \\) In `bert-base-uncased`, *e.g.* the intermediate dimension \\( d_{f} \\) is with 3072 four times larger than the output dimension \\( d_{h} \\).

 \\( {}^3 \\) As a reminder, the output `config.num_attention_{h}eads` is assumed to be 1 for the sake of clarity and illustration in this notebook, so that the output of the self-attention layers can be assumed to be of size `config.hidden_size`.

More information on chunked linear / feed forward layers can also be found [here](https://huggingface.co/transformers/glossary.html#feed-forward-chunking) on the 🤗Transformers docs.


### Benchmark

Let's test how much memory can be saved by using chunked feed forward layers.


```
#@title Installs and Imports
# pip installs
!pip -qq install git+https://github.com/huggingface/transformers.git
!pip install -qq py3nvml

from transformers import ReformerConfig, PyTorchBenchmark, PyTorchBenchmarkArguments
```

      Building wheel for transformers (setup.py) ... [?25l[?25hdone


First, let's compare the default `google/reformer-enwik8` model without chunked feed forward layers to the one with chunked feed forward layers.


```
config_no_chunk = ReformerConfig.from_pretrained("google/reformer-enwik8")  # no chunk
config_chunk = ReformerConfig.from_pretrained("google/reformer-enwik8", chunk_size_feed_forward=1)  # feed forward chunk
benchmark_args = PyTorchBenchmarkArguments(sequence_lengths=[1024, 2048, 4096], batch_sizes=[8], models=["Reformer-No-Chunk", "Reformer-Chunk"], no_speed=True, no_env_print=True)
benchmark = PyTorchBenchmark(configs=[config_no_chunk, config_chunk], args=benchmark_args)
result = benchmark.run()
```

    1 / 2
    Doesn't fit on GPU. CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 11.17 GiB total capacity; 7.85 GiB already allocated; 1.74 GiB free; 9.06 GiB reserved in total by PyTorch)
    2 / 2
    Doesn't fit on GPU. CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 11.17 GiB total capacity; 7.85 GiB already allocated; 1.24 GiB free; 9.56 GiB reserved in total by PyTorch)
    
    ====================      INFERENCE - MEMORY - RESULT       ====================
    --------------------------------------------------------------------------------
              Model Name             Batch Size     Seq Length    Memory in MB 
    --------------------------------------------------------------------------------
          Reformer-No-Chunk              8              1024            4281     
          Reformer-No-Chunk              8              2048            7607     
          Reformer-No-Chunk              8              4096            N/A      
            Reformer-Chunk               8              1024            4309     
            Reformer-Chunk               8              2048            7669     
            Reformer-Chunk               8              4096            N/A      
    --------------------------------------------------------------------------------


Interesting, chunked feed forward layers do not seem to help here at all. The reason is that `config.feed_forward_size` is not sufficiently large to make a real difference. Only at longer sequence lengths of 4096, a slight decrease in memory usage can be seen. 

Let's see what happens to the memory peak usage if we increase the size of the feed forward layer by a factor of 4 and reduce the number of attention heads also by a factor of 4 so that the feed forward layer becomes the memory bottleneck.


```
config_no_chunk = ReformerConfig.from_pretrained("google/reformer-enwik8", chunk_size_feed_forward=0, num_attention_{h}eads=2, feed_forward_size=16384)  # no chuck
config_chunk = ReformerConfig.from_pretrained("google/reformer-enwik8", chunk_size_feed_forward=1, num_attention_{h}eads=2, feed_forward_size=16384)  # feed forward chunk
benchmark_args = PyTorchBenchmarkArguments(sequence_lengths=[1024, 2048, 4096], batch_sizes=[8], models=["Reformer-No-Chunk", "Reformer-Chunk"], no_speed=True, no_env_print=True)
benchmark = PyTorchBenchmark(configs=[config_no_chunk, config_chunk], args=benchmark_args)
result = benchmark.run()
```

    1 / 2
    2 / 2
    
    ====================      INFERENCE - MEMORY - RESULT       ====================
    --------------------------------------------------------------------------------
              Model Name             Batch Size     Seq Length    Memory in MB 
    --------------------------------------------------------------------------------
          Reformer-No-Chunk              8              1024            3743     
          Reformer-No-Chunk              8              2048            5539     
          Reformer-No-Chunk              8              4096            9087     
            Reformer-Chunk               8              1024            2973     
            Reformer-Chunk               8              2048            3999     
            Reformer-Chunk               8              4096            6011     
    --------------------------------------------------------------------------------


Now a clear decrease in peak memory usage can be seen for longer input sequences. 
As a conclusion, it should be noted chunked feed forward layers only makes sense for models having few attention heads and large feed forward layers.

## 3. Reversible Residual Layers

Reversible residual layers were first introduced in [N. Gomez et al](https://arxiv.org/abs/1707.04585) and used to reduce memory consumption when training the popular *ResNet* model. Mathematically, reversible residual layers are slightly different 
to "real" residual layers but do not require the activations to be saved during the forward pass, which can drastically reduce memory consumption for training.

### Reversible Residual Layers in Reformer

Let's start by investigating why training a model requires 
much more memory than the inference of the model.

When running a model in inference, the required memory equals more or less the memory it takes to compute the **single** largest tensor in the model.
On the other hand, when training a model, the required memory equals more or less the **sum** of all differentiable tensors.

This is not surprising when considering how auto differentiation works in deep learning frameworks. These lecture [slides](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf) by Roger Grosse of the University of Toronto are great to better understand auto differentiation.

In a nutshell, in order to calculate the gradient of a differentiable function (*e.g.* a layer), auto differentiation requires the gradient of the function's output and the function's input and output tensor. While the gradients are dynamically computed and subsequently discarded, the input and output tensors (*a.k.a* activations) of a function are stored during the forward pass.

Alright, let's apply this to a transformer model. A transformer model includes a stack of multiple so-called transformer layers. Each additional transformer layer forces the model to store more activations during the forward pass and thus increases the required memory for training. 
Let's take a more detailed look. A transformer layer essentially consists of two residual layers. The first residual layer represents the *self-attention* mechanism as explained in section 1) and the second residual layer represents the *linear* or feed-forward layers as explained in section 2).

Using the same notation as before, the input of a transformer layer *i.e.* \\( \mathbf{X} \\) is first normalized \\( ^{1} \\) and subsequently processed by the self-attention layer to get the output \\( \mathbf{Z} = \text{SelfAttn}(\text{LayerNorm}(\mathbf{X})) \\). We will abbreviate these two layers with \\( G \\) so that \\( \mathbf{Z} = G(\mathbf{X}) \\). 
Next, the residual \\( \mathbf{Z} \\) is added to the input \\( \mathbf{\overline{Z}} = \mathbf{Z} + \mathbf{X} \\) and the sum is fed into the second residual layer - the two linear layers. \\( \mathbf{\overline{Z}} \\) is processed by a second normalization layer, followed by the two linear layers to get \\( \mathbf{Y} = \text{Linear}(\text{LayerNorm}(\mathbf{Z} + \mathbf{X})) \\). We will abbreviate the second normalization layer and the two linear layers with \\( F \\) yielding \\( \mathbf{Y} = F(\mathbf{\overline{Z}}) \\). 
Finally, the residual \\( \mathbf{Y} \\) is added to \\( \mathbf{\overline{Z}} \\) to give the output of the transformer layer \\( \mathbf{\overline{Y}} = \mathbf{Y} + \mathbf{\overline{Z}} \\).

Let's illustrate a complete transformer layer using the example of \\( \mathbf{x}_1, \ldots, \mathbf{x}_{16} \\).

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/normal_trans_resnet.png)

To calculate the gradient of *e.g.* the self-attention block \\( G \\), three tensors have to be known beforehand: the gradient \\( \partial \mathbf{Z} \\), the output \\( \mathbf{Z} \\), and the input \\( \mathbf{X} \\). While \\( \partial \mathbf{Z} \\) can be calculated on-the-fly and discarded afterward, the values for \\( \mathbf{Z} \\) and \\( \mathbf{X} \\) have to be calculated and stored during the forward pass since it is not possible to recalculate them easily on-the-fly during backpropagation. Therefore, during the forward pass, large tensor outputs, such as the query-key dot product matrix \\( \mathbf{Q}\mathbf{K}^T \\) or the intermediate output of the linear layers \\( \mathbf{Y}^{\text{int}} \\), have to be stored in memory \\( ^{2} \\).

Here, reversible residual layers come to our help. The idea is relatively straight-forward. The residual block is designed in a way so that instead of having to store the input and output tensor of a function, both can easily be recalculated during the backward pass so that no tensor has to be stored in memory during the forward pass. 
This is achieved by using two input streams \\( \mathbf{X}^{(1)}, \mathbf{X}^{(2)} \\), and two output streams \\( \mathbf{\overline{Y}}^{(1)}, \mathbf{\overline{Y}}^{(2)} \\). The first residual \\( \mathbf{Z} \\) is computed by the first output stream \\( \mathbf{Z} = G(\mathbf{X}^{(1)}) \\) and subsequently added to the input of the second input stream, so that \\( \mathbf{\overline{Z}} = \mathbf{Z} + \mathbf{X}^{(2)} \\). 
Similarly, the residual \\( \mathbf{Y} = F(\mathbf{\overline{Z}}) \\) is added to the first input stream again, so that the two output streams are defined by \\( \mathbf{Y}^{(1)} = \mathbf{Y} + \mathbf{X}^{(1)} \\) and \\( \mathbf{Y}^{(2)} = \mathbf{X}^{(2)} + \mathbf{Z} = \mathbf{\overline{Z}} \\).

The reversible transformer layer can be visualized for \\( \mathbf{x}_1, \ldots, \mathbf{x}_{16} \\) as follows.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/rev_trans_resnet.png)

As can be seen, the outputs \\( \mathbf{\overline{Y}}^{(1)}, \mathbf{\overline{Y}}^{(2)} \\) are calculated in a very similar way than \\( \mathbf{\overline{Y}} \\) of the non-reversible layer, but they are mathematically different. The authors of Reformer observe in some initial experiments that the performance of a reversible transformer model matches the performance of a standard transformer model. 
The first visible difference to the standard transformer layer is that there are two input streams and output streams \\( ^{3} \\), which at first slightly increases the required memory for both the forward pass.
The two-stream architecture is crucial though for not having to save any activations during the forward pass. Let's explain. For backpropagation, the reversible transformer layer has to calculate the gradients \\( \partial G \\) and \\( \partial F \\). In addition to the gradients \\( \partial \mathbf{Y} \\) and \\( \partial \mathbf{Z} \\) which can be calculated on-the-fly, the tensor values \\( \mathbf{Y} \\), \\( \mathbf{\overline{Z}} \\) have to be known for \\( \partial F \\) and the tensor values \\( \mathbf{Z} \\) and \\( \mathbf{X}^{(1)} \\) for \\( \partial G \\) to make auto-differentiation work.

If we assume to know \\( \mathbf{\overline{Y}}^{(1)}, \mathbf{\overline{Y}}^{(2)} \\), it can easily be depicted from the graph that one can calculate \\( \mathbf{X}^{(1)}, \mathbf{X}^{(2)} \\) as follows. \\( \mathbf{X}^{(1)} = F(\mathbf{\overline{Y}}^{(1)}) - \mathbf{\overline{Y}}^{(1)} \\). Great, now that \\( \mathbf{X}^{(1)} \\) is known, \\( \mathbf{X}^{(2)} \\) can be computed by \\( \mathbf{X}^{(2)} = \mathbf{\overline{Y}}^{(1)} - G(\mathbf{X}^{(1)}) \\). Alright now, \\( \mathbf{Z} \\) and \\( \mathbf{Y} \\) are trivial to compute via \\( \mathbf{Y} = \mathbf{\overline{Y}}^{(1)} - \mathbf{X}^{(1)} \\) and \\( \mathbf{Z} = \mathbf{\overline{Y}}^{(2)} - \mathbf{X}^{(2)} \\). So as a conclusion, if only the outputs \\( \mathbf{\overline{Y}}^{(1)}, \mathbf{\overline{Y}}^{(2)} \\) of the **last** reversible transformer layer are stored during the forward pass, all other relevant activations can be derived by making use of \\( G \\) and \\( F \\) during the backward pass and passing \\( \mathbf{X}^{(1)} \\) and \\( \mathbf{X}^{(2)} \\). The overhead of two forward passes of \\( G \\) and \\( F \\) per reversible transformer layer during the backpropagation is traded against not having to store any activations during the forward pass. Not a bad deal!

**Note**: Since recently, major deep learning frameworks have released code that allows to store only certain activations and recompute larger ones during the backward propagation (Tensoflow [here](https://www.tensorflow.org/api_docs/python/tf/recompute_grad) and PyTorch [here](https://pytorch.org/docs/stable/checkpoint.html)). For standard reversible layers, this still means that at least one activation has to be stored for each transformer layer, but by defining which activations can dynamically be recomputed a lot of memory can be saved.

---
 \\( ^{1} \\) In the previous two sections, we have omitted the layer norm layers preceding both the self-attention layer and the linear layers. The reader should know that both \\( \mathbf{X} \\) and \\( \mathbf{\overline{Z}} \\) are both processed by layer normalization before being fed into self-attention and the linear layers respectively.
 \\( ^{2} \\) While in the design the dimension of \\( \mathbf{Q}\mathbf{K} \\) is written as \\( n \times n \\), in a *LSH self-attention* or *local self-attention* layer the dimension would only be \\( n \times l_{c} \times n_{h} \\) or \\( n \times l_{c} \\) respectively with \\( l_{c} \\) being the chunk length and \\( n_{h} \\) the number of hashes
 \\( ^{3} \\) In the first reversible transformer layer \\( \mathbf{X}^{(2)} \\) is set to be equal to \\( \mathbf{X}^{(1)} \\).


### Benchmark

In order to measure the effect of reversible residual layers, we will compare the memory consumption of BERT with Reformer in training for an increasing number of layers.


```
#@title Installs and Imports
# pip installs
!pip -qq install git+https://github.com/huggingface/transformers.git
!pip install -qq py3nvml

from transformers import ReformerConfig, BertConfig, PyTorchBenchmark, PyTorchBenchmarkArguments
```

Let's measure the required memory for the standard `bert-base-uncased` BERT model by increasing the number of layers from 4 to 12.


```
config_4_layers_bert = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers=4)
config_8_layers_bert = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers=8)
config_12_layers_bert = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers=12)
benchmark_args = PyTorchBenchmarkArguments(sequence_lengths=[512], batch_sizes=[8], models=["Bert-4-Layers", "Bert-8-Layers", "Bert-12-Layers"], training=True, no_inference=True, no_speed=True, no_env_print=True)
benchmark = PyTorchBenchmark(configs=[config_4_layers_bert, config_8_layers_bert, config_12_layers_bert], args=benchmark_args)
result = benchmark.run()
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=433.0, style=ProgressStyle(description_…


    
    1 / 3
    2 / 3
    3 / 3
    
    ====================        TRAIN - MEMORY - RESULTS        ====================
    --------------------------------------------------------------------------------
              Model Name             Batch Size     Seq Length    Memory in MB 
    --------------------------------------------------------------------------------
            Bert-4-Layers                8              512             4103     
            Bert-8-Layers                8              512             5759     
            Bert-12-Layers               8              512             7415     
    --------------------------------------------------------------------------------


It can be seen that adding a single layer of BERT linearly increases the required memory by more than 400MB.


```
config_4_layers_reformer = ReformerConfig.from_pretrained("google/reformer-enwik8", num_hidden_layers=4, num_hashes=1)
config_8_layers_reformer = ReformerConfig.from_pretrained("google/reformer-enwik8", num_hidden_layers=8, num_hashes=1)
config_12_layers_reformer = ReformerConfig.from_pretrained("google/reformer-enwik8", num_hidden_layers=12, num_hashes=1)
benchmark_args = PyTorchBenchmarkArguments(sequence_lengths=[512], batch_sizes=[8], models=["Reformer-4-Layers", "Reformer-8-Layers", "Reformer-12-Layers"], training=True, no_inference=True, no_speed=True, no_env_print=True)
benchmark = PyTorchBenchmark(configs=[config_4_layers_reformer, config_8_layers_reformer, config_12_layers_reformer], args=benchmark_args)
result = benchmark.run()
```

    1 / 3
    2 / 3
    3 / 3
    
    ====================        TRAIN - MEMORY - RESULTS        ====================
    --------------------------------------------------------------------------------
              Model Name             Batch Size     Seq Length    Memory in MB 
    --------------------------------------------------------------------------------
          Reformer-4-Layers              8              512             4607     
          Reformer-8-Layers              8              512             4987     
          Reformer-12-Layers             8              512             5367     
    --------------------------------------------------------------------------------


For Reformer, on the other hand, adding a layer adds significantly less memory in practice. Adding a single layer increases the required memory on average by less than 100MB so that a much larger 12-Layer `reformer-enwik8` model requires less memory than a 12-Layer `bert-base-uncased` model.

## 4. Axial Positional Encodings

Reformer makes it possible to process huge input sequences. However, for such long input sequences standard positional encoding weight matrices alone would use more than 1GB to store its weights.
To prevent such large positional encoding matrices, the official Reformer code introduced *Axial Position Encodings*. 

**Important:** *Axial Position Encodings were not explained in the official paper, but can be well understood from looking into the code and talking to the authors*


### Axial Positional Encodings in Reformer

Transformers need positional encodings to account for the order of words in the input because self-attention layers have *no notion of order*. 
Positional encodings are usually defined by a simple look-up matrix \\( \mathbf{E} = \left[\mathbf{e}_1, \ldots, \mathbf{e}_{n_\text{max}}\right] \\) The positional encoding vector \\( \mathbf{e}_{i} \\) is then simply added to the *ith* input vector \\( \mathbf{x}_{i} + \mathbf{e}_{i} \\) so that the model can distinguish if an input vector (*a.k.a* token) is at position \\( i \\) or \\( j \\). 
For every input position, the model needs to be able to look up the corresponding positional encoding vector so that the dimension of \\( \mathbf{E} \\) is defined by the maximum length of input vectors the model can process `config.max_position_embeddings`, *i.e.* \\( n_\text{max} \\), and the `config.hidden_size`, *i.e.* \\( d_{h} \\) of the input vectors. 

Assuming \\( d_{h}=4 \\) and \\( n_\text{max}=49 \\), such a positional encoding matrix can be visualized as follows:

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/positional_encodings_default.png)

Here, we showcase only the positional encodings \\( \mathbf{e}_{1} \\), \\( \mathbf{e}_{2} \\), and \\( \mathbf{e}_{49} \\) each of dimension, *a.k.a* height 4.

Let's imagine, we want to train a Reformer model on sequences of a length of up to 0.5M tokens and an input vector `config.hidden_size` of 1024 (see notebook [here](https://github.com/patrickvonplaten/notebooks/blob/master/PyTorch_Reformer.ipynb)). The corresponding positional embeddings have a size of \\( 0.5M \times 1024 \sim 512M \\) parameters, which corresponds to a size of 2GB.

Such positional encodings would use an unnecessarily large amount of memory both when loading the model in memory and when saving the model on a hard drive.

The Reformer authors managed to drastically shrink the positional encodings in size by cutting the `config.hidden_size` dimension in two and smartly factorizing 
the \\( n_\text{max} \\) dimension. 
In Transformer, the user can decide into which shape \\( n_\text{max} \\) can be factorized into by setting `config.axial_pos_shape` to an appropriate 
list of two values \\( n_\text{max}^1 \\) and \\( n_\text{max}^2 \\) so that \\( n_\text{max}^1 \times n_\text{max}^2 = n_\text{max} \\). By setting `config.axial_pos_embds_dim` to an 
appropriate list of two values \\( d_{h}^{1} \\) and \\( d_{h}^2 \\) so that \\( d_{h}^1 + d_{h}^2 = d_{h} \\), the user can decide how the hidden size dimension should be cut. 
Now, let's visualize and explain more intuitively.

One can think of factorizing \\( n_{\text{max}} \\) as folding the dimension into a third axis, which is shown in the following for the factorization `config.axial_pos_shape = [7, 7]`:

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/3d_positional_encoding.png)

Each of the three standing rectangular prisms corresponds to one of the encoding vectors \\( \mathbf{e}_{1}, \mathbf{e}_{2}, \mathbf{e}_{49} \\), but we can see that the 49 encoding vectors are divided into 7 rows of 7 vectors each.
Now the idea is to use only one row of 7 encoding vectors and expand those vectors to the other 6 rows, essentially reusing their values. 
Because it is discouraged to have the same values for different encoding vectors, each vector of dimension (*a.k.a* height) `config.hidden_size=4` is cut into the lower encoding vector \\( \mathbf{e}_\text{down} \\) of size \\( 1 \\) and \\( \mathbf{e}_\text{up} \\) of size \\( 3 \\), so that the lower part can be expanded along the row dimension and the upper part can be expanded along the column dimension.
Let's visualize for more clarity.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/3d_positional_encoding_cut.png)

We can see that we have cut the embedding vectors into \\( \mathbf{e}_\text{down} \\) (*in blue*) and \\( \mathbf{e}_\text{up} \\) (*in yellow*).
Now for the "sub"-vectors \\( \mathbf{E}_\text{down} = \left[\mathbf{e}_{\text{down},1}, \ldots, \mathbf{e}_{\text{down},49}\right] \\) only the first row, *a.k.a.* the width in the graphic, of \\( 7 \\) is kept and expanded along the column dimension, *a.k.a.* the depth of the graphic. Inversely, for the "sub"-vectors \\( \mathbf{E}_\text{up} = \left[\mathbf{e}_{\text{up},1}, \ldots, \mathbf{e}_{\text{up},49}\right] \\) only the first column of \\( 7 \\) is kept and expanded along the row dimension.
The resulting embedding vectors \\( \mathbf{e'}_{i} \\) then correspond to

$$
  \begin{align}
    \mathbf{e'}_{i} &= \begin{bmatrix}
           \mathbf{e}_{\text{down, } i \% n_\text{max}^1} \\
           \mathbf{e}_{\text{up, } \left \lfloor{\frac{i}{n_{\text{max}^2}}}\right \rceil}
         \end{bmatrix}
  \end{align}
$$
whereas \\( n_\text{max}^1 = 7 \\) and \\( n_\text{max}^2 = 7 \\) in our example.
These new encodings \\( \mathbf{E'} = \left[\mathbf{e'}_{1}, \ldots, \mathbf{e'}_{n_\text{max}}\right] \\) are called **Axial Position Encodings**. 

In the following, these axial position encodings are illustrated in more detail for our example.

![alt text](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/axial_pos_encoding.png)

Now it should be more understandable how the final positional encoding vectors \\( \mathbf{E'} \\) are calculated only from \\( \mathbf{E}_{\text{down}} \\) of dimension \\( d_{h}^1 \times n_{\text{max}^1} \\) and \\( \mathbf{E}_{\text{up}} \\) of dimension \\( d_{h}^2 \times n_{\text{max}}^2 \\).

The crucial aspect to see here is that Axial Positional Encodings make sure that none of the vectors \\( \left[\mathbf{e'}_1, \ldots, \mathbf{e'}_{n_{\text{max}}}\right] \\) are equal to each other by design and that the overall size of the encoding matrix is reduced from \\( n_{\text{max}} \times d_{h} \\) to \\( n_{\text{max}}^1 \times d_{h}^1 + n_\text{max}^2 \times d_{h}^2 \\).
By allowing each axial positional encoding vector to be different by design the model is given much more flexibility to learn efficient positional representations if axial positional encodings are learned by the model.

To demonstrate the drastic reduction in size, 
let's assume we would have set `config.axial_pos_shape = [1024, 512]` and `config.axial_pos_embds_dim = [512, 512]` for a Reformer model that can process inputs up to a length of 0.5M tokens. The resulting axial positional encoding matrix would have had a size of only \\( 1024 \times 512 + 512 \times 512 \sim 800K \\) parameters which corresponds to roughly 3MB. This is a drastic reduction from the 2GB a standard positional encoding matrix would require in this case.

For a more condensed and math-heavy explanation please refer to the 🤗Transformers docs [here](https://huggingface.co/transformers/model_doc/reformer.html#axial-positional-encodings).

### Benchmark

Lastly, let's also compare the peak memory consumption of conventional positional embeddings to *axial positional embeddings*.


```
#@title Installs and Imports
# pip installs
!pip -qq install git+https://github.com/huggingface/transformers.git
!pip install -qq py3nvml

from transformers import ReformerConfig, PyTorchBenchmark, PyTorchBenchmarkArguments, ReformerModel
```

Positional embeddings depend only on two configuration parameters: The maximum allowed length of input sequences `config.max_position_embeddings` and `config.hidden_size`. Let's use a model that pushes the maximum allowed length of input sequences to half a million tokens, called `google/reformer-crime-and-punishment`, to see the effect of using axial positional embeddings.

To begin with, we will compare the shape of axial position encodings with standard positional encodings and the number of parameters in the model.


```
config_no_pos_axial_embeds = ReformerConfig.from_pretrained("google/reformer-crime-and-punishment", axial_pos_embds=False)  # disable axial positional embeddings
config_pos_axial_embeds = ReformerConfig.from_pretrained("google/reformer-crime-and-punishment", axial_pos_embds=True, axial_pos_embds_dim=(64, 192), axial_pos_shape=(512, 1024))  # enable axial positional embeddings

print("Default Positional Encodings")
print(20 * '-')
model = ReformerModel(config_no_pos_axial_embeds)
print(f"Positional embeddings shape: {model.embeddings.position_embeddings}")
print(f"Num parameters of model: {model.num_parameters()}")
print(20 * '-' + '\n\n')

print("Axial Positional Encodings")
print(20 * '-')
model = ReformerModel(config_pos_axial_embeds)
print(f"Positional embeddings shape: {model.embeddings.position_embeddings}")
print(f"Num parameters of model: {model.num_parameters()}")
print(20 * '-' + '\n\n')
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1151.0, style=ProgressStyle(description…


    
    Default Positional Encodings
    --------------------
    Positional embeddings shape: PositionEmbeddings(
      (embedding): Embedding(524288, 256)
    )
    Num parameters of model: 136572416
    --------------------
    
    
    Axial Positional Encodings
    --------------------
    Positional embeddings shape: AxialPositionEmbeddings(
      (weights): ParameterList(
          (0): Parameter containing: [torch.FloatTensor of size 512x1x64]
          (1): Parameter containing: [torch.FloatTensor of size 1x1024x192]
      )
    )
    Num parameters of model: 2584064
    --------------------
    
    


Having read the theory, the shape of the axial positional encoding weights should not be a surprise to the reader.

Regarding the results, it can be seen that for models being capable of processing such long input sequences, it is not practical to use default positional encodings. 
In the case of `google/reformer-crime-and-punishment`, standard positional encodings alone contain more than 100M parameters. 
Axial positional encodings reduce this number to just over 200K.

Lastly, let's also compare the required memory at inference time.


```
benchmark_args = PyTorchBenchmarkArguments(sequence_lengths=[512], batch_sizes=[8], models=["Reformer-No-Axial-Pos-Embeddings", "Reformer-Axial-Pos-Embeddings"], no_speed=True, no_env_print=True)
benchmark = PyTorchBenchmark(configs=[config_no_pos_axial_embeds, config_pos_axial_embeds], args=benchmark_args)
result = benchmark.run()
```

    1 / 2
    2 / 2
    
    ====================      INFERENCE - MEMORY - RESULT       ====================
    --------------------------------------------------------------------------------
              Model Name             Batch Size     Seq Length    Memory in MB 
    --------------------------------------------------------------------------------
    Reformer-No-Axial-Pos-Embeddin       8              512             959      
    Reformer-Axial-Pos-Embeddings        8              512             447      
    --------------------------------------------------------------------------------


It can be seen that using axial positional embeddings reduces the memory requirement to approximately half in the case of `google/reformer-crime-and-punishment`.
