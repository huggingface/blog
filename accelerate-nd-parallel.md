---
title: "Accelerate ND-Parallel: A guide to Efficient Multi-GPU Training"
thumbnail: /blog/assets/accelerate-nd-parallel/thumbnail.png
authors:
- user: smohammadi
  guest: true
  org: axolotl-ai-co
- user: siro1
- user: winglian
  guest: true
  org: axolotl-ai-co
- user: marcsun13
- user: djsaunde
  guest: true
  org: axolotl-ai-co
orgs:
- axolotl-ai-co
---
# Accelerate ND-Parallel: A guide to Efficient Multi-GPU Training

Training large models across multiple GPUs can be challenging due to the complexities of different parallelism strategies. In Accelerate, together with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl/), we have integrated a quick and easy way to use any combination of parallelism strategies in your training script!

Here is how to add it to your training script:

```python
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin

# configure your desired parallelisms here - this particular configuration requires at least 2 nodes with 8 GPUs each. 
# setting any parallelism degree to 1 disables it i.e. dp_replicate_size=1 disables DP.
pc = ParallelismConfig(
    dp_shard_size=2, # Fully Sharded Data Parallel degree
    dp_replicate_size=2, # Data Parallel degree
    cp_size=2, # Context Parallel degree
    tp_size=2, # Tensor Parallel degree
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2,
    auto_wrap_policy="transformer_based_wrap",
    transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
    state_dict_type="SHARDED_STATE_DICT",
)

accelerator = Accelerator(
    parallelism_config=pc,
    fsdp_plugin=fsdp_plugin
)

model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Hermes-3-Llama-3.1-8B", 
    device_mesh=accelerator.torch_device_mesh
)

model = accelerator.prepare(model)
```

We've also included a more comprehensive end-to-end [training script](https://github.com/huggingface/accelerate/blob/main/examples/fsdp2/nd_parallel.py) in the Accelerate repo which demonstrates how to setup your dataloader, optimizer, and training loop, and how to save your model after training.

To further streamline fine-tuning models at scale and compose parallelism strategies with a variety of fine-tuning techniques, we've also integrated this technique into Axolotl. To help you get started right away we've tested some [example configs](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/distributed-parallel) which you can modify to suit your needs - try one out with:

```bash
# note: this requires a minimum world size of 16 
axolotl train examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml
```

You can also check out the [Axolotl ND-Parallelism docs](https://docs.axolotl.ai/docs/nd_parallelism.html) for more details - adding ND parallel techniques to your existing configs is as simple as adding one or more of the following fields to your Axolotl config file:

```yaml
# Fully Sharded Data Parallel degree (note: also requires the fsdp_config field) 
# see https://docs.axolotl.ai/docs/multi-gpu.html#sec-fsdp for more details
dp_shard_size: 2
# Data Parallel degree
dp_replicate_size: 2
# Context Parallel Degree
context_parallel_size: 2
# Tensor Parallel Degree
tensor_parallel_size: 2
```

We've made it easy to configure the degrees of different parallelism strategies and how they are combined through the [`ParallelismConfig`](https://github.com/huggingface/accelerate/blob/v1.10.0/src/accelerate/parallelism_config.py) class in Accelerate, or through config fields in Axolotl, but how do we know which configuration will work best for our use case? As we scale to training models with tens or even hundreds of billions of parameters, the primary challenge comes from understanding the different parallelism strategies and how they interact to minimise communication overhead across devices. In this post, we'll walk through how the different parallelism strategies work, and when and how you might want to compose them. 

## Contents

- [Data Parallelism](#data-parallelism)
- [Fully Sharded Data Parallelism](#fully-sharded-data-parallelism)
- [Tensor Parallelism](#tensor-parallelism)
- [Context Parallelism](#context-parallelism)
- [ND Parallelisms](#nd-parallelisms)
  - [Hybrid Sharded Data Parallelism](#hybrid-sharded-data-parallelism)
  - [Fully Sharded Data Parallelism + Tensor Parallelism](#fully-sharded-data-parallelism--tensor-parallelism)
  - [Fully Sharded Data Parallelism + Context Parallelism](#fully-sharded-data-parallelism--context-parallelism)
  - [Hybrid Sharded Data Parallelism + Tensor Parallelism](#hybrid-sharded-data-parallelism--tensor-parallelism)
- [Usage Notes](#usage-notes)

## Data Parallelism 


<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/accelerate-nd-parallel/dp.png" alt="Diagram for Data Parallel">
  <figcaption> Distributed Data Parallel replicates the entire model across each device, and evenly divides the data into sub-batches for each device. (<b><i>Source: <a href="https://martynassubonis.substack.com/p/tensor-and-fully-sharded-data-parallelism">Martynas Šubonis</i></b></a>).
 </figcaption>
</figure>

Data parallelism (DP) is the most common technique for training models across multiple GPUs, and involves replicating the model, gradients and optimizer states across each device, whilst evenly distributing data batches between GPUs, and synchronising gradients across devices before updating parameters. This can significantly increase throughput compared to single-device training, but requires that your model is able to fit on a single device. 

We can control the number of replicas of the model with the `dp_replicate_size` parameter in Accelerate's `ParallelismConfig` or config field in Axolotl. It's worth noting that DP is a *top-most-level* parallelism strategy, meaning that if we use `dp_replicate_size=2` and we compose it with other parallelism strategies, there would be 2 replicas of the model, each also influenced by the other parallelism strategies. For example, if we use `dp_replicate_size=2` and `tp_size=2`, we would have 2 replicas of the model, each with 2 tensor parallel shards.

> [!NOTE]
> We use the term *shard* to describe data on a single device which is a partition of a larger piece of data.

## Fully Sharded Data Parallelism

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/accelerate-nd-parallel/fsdp.png" alt="Diagram for Fully Sharded Data Parallel">
  <figcaption> Fully Sharded Data Parallel evenly divides each of the model's parameters across each device, and, like DDP, evenly divides the data into sub-batches for each device. To complete a forward and backwards pass, FSDP must <i>gather</i> the weights of each parameter before the forwards/backwards pass so that each device obtains a full copy of the parameter. (<b><i>Source: <a href="https://martynassubonis.substack.com/p/tensor-and-fully-sharded-data-parallelism">Martynas Šubonis</i></b></a>).
 </figcaption>
</figure>

What if our model is too large to fit on a single device? Fully sharded data parallel (FSDP) addresses this issue by sharding (distributing evenly) the model’s weights, gradients, and optimizer states across GPUs (this is inspired by DeepSpeed’s ZeRO-3), whilst each device still receives its portion of the full batch of data. As you may notice from the diagram above, rather than requiring a full copy of the entire model on each device, we only gather the weights for a single layer at a time before the forward pass, after which the weights may be sharded again.

In this way, we trade memory usage for the communication overhead of gathering sharded parameters before each forward and backward pass, and reduce-scatter-ing local gradients. We can control this trade-off in FSDP by tuning the granularity at which parameters are gathered. On one extreme, we can gather and re-shard every layer of our model, which would result in the lowest peak memory usage, but incur the highest communication costs. In practice, a common approach is to gather the weights for an entire transformer decoder block at a time. 

Whilst we can make further memory-compute trade-offs and offload model parameters and gradients to the CPU to train larger models, this can be prohibitively slow. Instead, let’s consider how we can effectively utilise even more devices to train larger models whilst maintaining high data throughput.

We use the term *node* to refer to a single machine which hosts multiple GPUs (up to a maximum of 8), with fast intra-node communication channels using e.g. NVLink between GPUs. When using multiple nodes for training, we rely on relatively slower inter-node communication channels between machines using e.g. Infiniband. We also refer to the total number of devices in the process pool as the world size - e.g. a single node with 8 GPUs represents a world size of 8, and 4 nodes would represent a world size of 32.

When using FSDP across multiple nodes, we treat the entire set of devices across nodes as if we were training on a single node. For example, with 4 nodes containing 8 GPUs each, we perform our sharding across 32 devices, and perform our collective all-reduce and reduce-scatter operations using both inter-and-intra-node communication backends. In this manner, FSDP alone can scale to a substantial number of GPUs with a large global batch size to increase data throughput. However, there comes a point where several challenges arise that may require composing FSDP with other parallelism techniques. We usually try to avoid doing FSDP across more than a full node, as the communication overhead can become too high, we'll talk about how to address this in the section on [Hybrid Sharded Data Parallelism](#hybrid-sharded-data-parallelism).

> [!TIP]
> You can use the `dp_shard_size` parameter in Accelerate's `ParallelismConfig` together with a prepared [`FullyShardedDataParallelPlugin`](https://huggingface.co/docs/accelerate/v1.10.0/en/package_reference/utilities#accelerate.FullyShardedDataParallelPlugin), or set the `dp_shard_size` config field in Axolotl to set the degree of FSDP applied to your model. 


## Tensor Parallelism

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/accelerate-nd-parallel/tp.png" alt="Diagram for Tensor Parallel">
  <figcaption> Tensor Parallelism splits large linear layers across devices, typically using column-wise sharding for the first layer and row-wise sharding for the subsequent layer. This approach requires only a single AllReduce communication operation to combine the sharded outputs, minimizing communication overhead while distributing both memory and compute across devices within a node.
 </figcaption>
</figure>

Tensor Parallel (TP) is a kind of model parallelism technique, where shards of the model permanently live on separate devices, and in contrast to data parallel techniques, each device receives an identical batch of data. TP works by distributing the computation of linear layers across devices, so each device only computes a portion of the matrix multiplication. This technique works best when there are large linear layers, such as the feed-forward layers in transformer models, which can be split across devices. We can also use TP on each of the query, key, value, and output projections in the attention layers with almost no extra communication cost.

To achieve the best performance, parameters of consecutive layers can be distributed in a specific fashion, minimizing the required communication. When working with pairs of linear layers, we can split the first layer column-wise, and the subsequent layer row-wise, allowing us to compute the output with only a single all-reduce operation to combine the sharded outputs. 

Unlike the dynamic sharding behaviour of FSDP, TP creates static memory partitions which result in a constant memory usage reduction scaling with the TP group size. This becomes crucial for massive models where even a single decoder layer is too large to fit into memory during the FSDP all-gather (recall that common practice in FSDP is to gather the weights of an entire decoder layer at a time). However, unlike FSDP which scales relatively linearly across nodes (up to a point - ~512 GPUs on a homogenous cluster, significantly less across lower-bandwidth connections), TP is only effective within the boundaries of a single node. TP requires frequent activation synchronization between devices during computation, as each device computes only a portion of the output, requiring the outputs from other devices to be communicated before continuing the forward pass. Thus, if we wish to utilise TP in a multi-node setup, we must consider composing TP with other parallelism techniques, while keeping TP only within a single node. Due to its large communications overhead, TP is not recommended for PCIe linked GPUs.

> [!TIP]
> In Accelerate, the TP size is configured through `tp_size` in `ParallelismConfig`, whilst in Axolotl you can use the `tensor_parallel_size` config field.

## Context Parallelism 

Recently, reasoning capabilities in LLMs resulted in sequence lengths skyrocketing as models use more and more tokens to solve complex tasks. To achieve this behaviour through fine-tuning, we need a way to train models on very large sequence lengths - which can sometimes reach up to a million tokens!

Since the attention operation in transformers scales quadratically with context length, this becomes impossible on a single GPU. For example, when fine-tuning a relatively small model such as Mistral-7B (which uses 32 attention heads), if we use a sequence length of 128k a single attention matrix will utilise 128k * 128k * 2 bytes * `num_heads=32` = ~32GB * 32 = ~1TB of activations memory! Whilst this example is not realistic when using optimised attention implementations such as FlashAttention, it helps illustrate the growth in memory requirements from increasing the context length.

With context parallelism (CP), we can shard the inputs across the sequence dimension, resulting in each device only processing a chunk of the full context and computing a smaller portion of the full, prohibitively large, attention matrix. To see how this works, recall that the attention computation is described by the equation:

\\( \text{Attention}(Q, K, V) = \text{softmax}(QK^T)V \\)

Where \\( Q \\), \\( K \\), and \\( V \\) are the query, key, and value matrices respectively. Each query vector (row, or input embedding) of \\( Q \\) must compute the attention scores against *every* key vector of \\( K \\) in the entire sequence to correctly apply the softmax normalisation. These attention scores are then weighted with *all* value vectors in \\( V \\).

The crucial detail here lies in the fact that each row in \\( Q \\) can compute its attention score independently of one another, but each query vector still requires the full \\( K \\) and \\( V \\) matrices. In other words, given an input with sequence length $n$, we can expand our above attention equation as:

$$
\begin{align}
\text{Attention}(Q, K, V)_1 &= \text{softmax}(Q_1 K^T) V \\
\text{Attention}(Q, K, V)_2 &= \text{softmax}(Q_2 K^T) V \\
&\vdots \\
\text{Attention}(Q, K, V)_n &= \text{softmax}(Q_n K^T) V
\end{align}
$$

where we denote each row of the query matrix as \\( Q_1, Q_2, ..., Q_n \\). This can be generalized as:

\\( \text{Attention}(Q, K, V)_i = \text{softmax}(Q_i K^T) V \quad \forall i \in \{1, 2, ..., n\} \\)

When we shard the inputs across devices, the resulting \\( Q \\), \\( K \\), and \\( V \\) matrices (computed from these input shards) are also automatically sharded along the sequence dimension - each GPU computes queries, keys, and values only for its portion of the sequence. For example, with a world size of \\( W \\) GPUs and sequence length \\( n \\):

- GPU 0 computes \\( Q_{1:n/W} \\), \\( K_{1:n/W} \\), \\( V_{1:n/W} \\)
- GPU 1 computes \\( Q_{n/W+1:2n/W} \\), \\( K_{n/W+1:2n/W} \\), \\( V_{n/W+1:2n/W} \\)
- ...
- GPU \\( (W-1) \\) computes \\( Q_{(W-1)n/W+1:n} \\), \\( K_{(W-1)n/W+1:n} \\), \\( V_{(W-1)n/W+1:n} \\)

How do we ensure the attention is computed correctly? As established above, each device only needs its own shard of \\( Q \\), but requires the full \\( K \\) and \\( V \\) matrices to compute the attention correctly. We can achieve this by using a technique called [RingAttention](https://openreview.net/forum?id=WsRHpHH4s0), which works as follows:
1. Initially, each GPU holds its shard of \\( Q \\), \\( K \\), \\( V \\) (e.g., GPU 0 holds \\( Q_{1:n/W} \\), \\( K_{1:n/W} \\),
\\( V_{1:n/W} \\)).
2. Each GPU then computes a partial attention matrix \\( A_{i,j} \\) for its shard of \\( Q_i \\) and its local
shard of \\( K_j \\), \\( V_j \\).
3. Each GPU sends its shard of \\( K \\), \\( V \\) to the next GPU in the ring.
4. Each GPU receives a different shard of \\( K \\), \\( V \\) from the previous GPU in the ring.
5. Each GPU computes additional partial attention matrices \\( A_{i,j+1} \\), \\( A_{i,j+2} \\), etc. using
the received \\( K \\), \\( V \\) shards.
6. Each GPU repeats this process until all shards of \\( K \\), \\( V \\) have been received and all partial
  attention matrices \\( A_{i,*} \\) have been computed.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/accelerate-nd-parallel/cp.png" alt="Diagram for Context Parallel">
  <figcaption>  Context Parallelism shards the input sequence across GPUs, with each device holding queries and key-value pairs for its assigned segment. Ring-attention circulates K,V shards between GPUs (shown by the arrows), allowing each query to compute attention scores against keys and values from the entire sequence. The final attention output combines information from all sequence positions while distributing memory and compute across devices. </figcaption>
</figure>


Accelerate enables this with the [`accelerator.maybe_context_parallel`](https://huggingface.co/docs/accelerate/v1.10.0/en/package_reference/accelerator#accelerate.Accelerator.maybe_context_parallel) decorator, which is also showcased in the Accelerate [example script](https://github.com/huggingface/accelerate/blob/main/examples/fsdp2/nd_parallel.py). You can also learn more about how it works and its limitations in our [CP concept guide](https://huggingface.co/docs/accelerate/main/en/concept_guides/context_parallelism).


> [!TIP]
> Similar to TP, in Accelerate the CP size is configured through `cp_size` in `ParallelismConfig`, whilst in Axolotl you can use the `context_parallel_size` config field.

## ND Parallelisms

In the multi-node setting, data parallel techniques such as FSDP treat the entire network topology as if it existed along a single dimension. You may find this approach limiting for a variety of reasons:
- When scaling to more nodes, FSDP's collective operations become bottlenecked by inter-node latency, making training prohibitively slow.
- As we mentioned above, massive models may have decoder layers which cannot fit into GPU memory, or which may be too large to perform a forward pass with, even in a sharded state.
- It could be impossible to achieve your ideal batch size - either the batch becomes too large for pure data parallelism to handle efficiently, or too small due to memory constraints from model size.

To try and address some of these problems, we can think of multi-node clusters as having a two-dimensional topology: fast intra-node communication between devices along one axis, and relatively slower inter-node communication along another axis. Let’s consider how we can compose the parallelism techniques we’ve introduced so far to take advantage of this.


### Hybrid Sharded Data Parallelism


<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/accelerate-nd-parallel/hsdp.png" alt="Diagram for Hybrid Sharded Data Parallel">
  <figcaption>  Hybrid Sharded Data Parallelism performs FSDP within each replica group and synchronizes gradients across replica groups via AllReduce, combining the memory efficiency of FSDP with the communication efficiency of DP across nodes.
 </figcaption>
</figure>

Hybrid Sharded Data Parallelism (HSDP) is a kind of 2D parallelism which performs FSDP within a node, and DP across nodes - that is to say the model is replicated across each node, and sharded using FSDP within each node. This allows the greater communication overhead of FSDP to utilize the faster intra-node links, whilst DP minimises the slower inter-node communication overhead to a single gradient synchronisation step. You might consider this approach if you were facing problem 1 and wished to speed up training at the cost of increased memory usage.

It’s important to note that we can freely configure the shape of our 2D network topology, as we aren’t constrained to the dimensions being aligned with physical node boundaries - you might apply FSDP across 2 nodes whilst replicating across groups of 2 nodes, which would result in lower memory usage but slower throughput, but still reduce the intra-node FSDP communication overhead by a factor of two. This is a knob we encourage you to tune to your specific hardware setup and fine-tuning needs.

> [!TIP]
> You can enable HSDP by defining both `dp_shard_size` and `dp_replicate_size` in Accelerate's `ParallelismConfig` or through Axolotl's config fields.

### Fully Sharded Data Parallelism + Tensor Parallelism 

As we mentioned earlier, TP should be applied within a node to utilize the high-bandwidth intra-node communications, thus, combining TP and FSDP involves sharding the model across nodes using FSDP, and within a node using TP. To a certain degree, this potentially offers a neat solution to all three of the issues above: the latency costs from FSDP could be reduced by a factor of 8, layers that are too large to fit on a single device are now evenly distributed across devices, and since each TP group receives an identical batch of data, we can also reduce our global batch size by a factor of 8. However, if this remains insufficient, we are unable to increase the TP size across nodes and must consider an alternative approach.

> [!TIP]
> In Accelerate you can combine TP and FSDP by defining both `dp_shard_size` and `tp_size` in `ParallelismConfig`, whilst in Axolotl you can add both of the `dp_shard_size` and `tensor_parallel_size` config fields.

### Fully Sharded Data Parallelism + Context Parallelism 

This is a 2D parallelism strategy that combines FSDP and CP, and while this is not very commonly used as CP already combines with FSDP (more on why in the [accelerate concept guide](https://huggingface.co/docs/accelerate/main/en/concept_guides/context_parallelism)), it can be useful in some cases i.e. when requiring a large sequence length, consequently requiring a large `cp_size`. If this still doesn't fit into your memory budget, you can apply FSDP on top of this, further reducing the memory usage.

> [!TIP]
> In Accelerate you can combine CP and FSDP by defining both `dp_shard_size` and `cp_size` in `ParallelismConfig`, whilst in Axolotl you can add both of the `dp_shard_size` and `context_parallel_size` config fields.

### Hybrid Sharded Data Parallelism + Tensor Parallelism 

With a sufficiently large world size (note: while the minimum world size for 3D parallelism is 8, it is most effective at much larger scales), we can consider combining HSDP with TP which creates a hierarchy where DP first replicates the model across groups of nodes, FSDP then shards the model within each group, and TP splits individual layers within each node. You might consider this approach when facing all of the scaling constraints we mentioned above, as it provides the most flexibility to adapt to your specific training setup by making trade-offs between memory usage and throughput.

> [!TIP]
> In Accelerate you can combine HSDP and TP by defining all of `dp_shard_size`, `dp_replicate_size`, and `tp_size` in `ParallelismConfig`. Similarly in Axolotl you can add all of the `dp_shard_size`, `dp_replicate_size`, and `tensor_parallel_size` config fields.

## Usage notes

There are additional ways to combine multiple parallelisms which we haven't covered, such as 4D parallel using HSDP + TP + CP, but they operate very similarly to the techniques we've already covered. Most of all, we encourage you to play with different techniques and configurations - this is the best way to gain an intuition for the different ways in which you can make memory/throughput trade-offs.

Below are some additional tips you may find useful when working in distributed settings:
- When using FSDP and working with models that are too large to fit in a single device, enabling both CPU RAM efficient loading and sharded state dict checkpointing technique is crucial. You can enable this through the `cpu_ram_efficient_loading` and `state_dict_type` parameters in Accelerate's [`FullyShardedDataParallelPlugin`](https://huggingface.co/docs/accelerate/v1.10.0/en/package_reference/utilities#accelerate.FullyShardedDataParallelPlugin), 
  ```python
  fsdp2_plugin = FullyShardedDataParallelPlugin(
      fsdp_version=2,
      auto_wrap_policy="transformer_based_wrap",
      transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
      state_dict_type="SHARDED_STATE_DICT", 
      cpu_ram_efficient_loading=True
  )
  ``` 
  or through the `cpu_ram_efficient_loading` and `state_dict_type` config fields inside the `fsdp_config` in Axolotl:
    ```yaml
    fsdp_version: 2
    fsdp_config:
      auto_wrap_policy: TRANSFORMER_BASED_WRAP
      transformer_layer_cls_to_wrap: LlamaDecoderLayer
      state_dict_type: SHARDED_STATE_DICT
      cpu_ram_efficient_loading: True
  ```
- The total batch size used during training plays an important factor in training stability, memory usage, and data throughput. When using DP and/or FSDP the effective batch size is calculated as:

  `effective_batch_size = micro_batch_size * gradient_accumulation_steps * dp_world_size`. 

  where `dp_world_size = (dp_shard_size * dp_replicate_size) / tp_size`. You can increase your batch size by increasing your total micro batch size or gradient accumulation steps in your training loop, or setting the `micro_batch_size` and `gradient_accumulation_steps` config fields in Axolotl, or increasing the total `dp_world_size` by adding more GPUs. As we mentioned above, this imposes a *minimum* total batch size of `dp_world_size` - when using pure DP/FSDP, this will be your total world size, and if this is too high the only way to decrease the total batch size is by introducing tensor parallelism. Finally, with a fixed number of GPUs and in memory-constrained scenarios, we recommend increasing `gradient_accumulation_steps` instead of `micro_batch_size` to achieve larger effective batch sizes, and vice-versa.
- Correspondingly, when your effective batch size increases due to introducing data parallelism, you should scale your learning rate to maintain training stability. Common approaches include linear scaling `scaled_lr = base_lr * (effective_batch_size / base_batch_size)` or square root scaling `scaled_lr = base_lr * sqrt(effective_batch_size / base_batch_size)`. 
- When memory constraints persist even with parallelism strategies, gradient checkpointing can provide additional memory savings by trading compute for memory. During the forward pass, only a subset of activations are kept in memory (typically at transformer block boundaries), and intermediate activations are recomputed during the backward pass. This technique works seamlessly with all parallelism strategies covered above. In Accelerate, you can enable it by setting `activation_checkpointing=true` in `FullyShardedDataParallelPlugin`:
  ```python
  fsdp2_plugin = FullyShardedDataParallelPlugin(
      fsdp_version=2,
      auto_wrap_policy="transformer_based_wrap",
      transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
      state_dict_type="SHARDED_STATE_DICT", 
      cpu_ram_efficient_loading=True,
      activation_checkpointing=True
  )
  ``` 
  and similarly in Axolotl:
    ```yaml
    fsdp_version: 2
    fsdp_config:
      auto_wrap_policy: TRANSFORMER_BASED_WRAP
      transformer_layer_cls_to_wrap: LlamaDecoderLayer
      state_dict_type: SHARDED_STATE_DICT
      cpu_ram_efficient_loading: True
      activation_checkpointing: True
    ```
  Note that gradient checkpointing typically increases training time by ~20-30% due to activation recomputation, but can reduce activation memory by 60-80%, making it particularly valuable when training very large models or using long sequence lengths.
