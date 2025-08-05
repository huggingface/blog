---
title: "Accelerate ND-Parallel: A guide to Efficient Multi-GPU Training"
thumbnail: /blog/assets/accelerate-nd-parallel/thumbnail.png
authors:
- user: siro1
- user: winglian
- user: smohammadi
- user: djsaunde
---

# Accelerate ND-Parallel: A guide to Efficient Multi-GPU Training

Training large models on multiple GPUs can be challenging due to the complexities of different parallelism strategies.
In Accelerate, together with [Axolotl](https://huggingface.co/axolotl-ai-co), we have integrated a quick and easy way
to use any combination of parallelism strategies in your training script!

Here is how to add it to your training script:

```python
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig

pc = ParallelismConfig(
    dp_shard_size=2,
    dp_replicate_size=2,
    cp_size=2,
    tp_size=2,
)

accelerator = Accelerator(
    parallelism_config=pc,
)
model = AutoModelForCausalLM.from_pretrained("your-model-name", tp_size=pc.tp_size, device_mesh=accelerator.torch_device_mesh)
model = accelerator.prepare(model)
```

This feature is also integrated into [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), allowing you to compose
a variety of fine-tuning optimizations and get started with fine-tuning models at scale in just a few minutes.

To get up and running quickly, you can check the examples in the [accelerate repository](https://github.com/huggingface/accelerate/blob/main/examples/fsdp2/nd_parallel.py) or their counterpart in [Axolotl](TODO)

You can see we are using the `ParallelismConfig` class to define the parallelism combination and its shape, but how
do we figure out what shape will work the best for our case? Understanding the different parallelism strategies
and how they interact is the primary challenge when training models at the scale of 10s or 100s of billions of parameters. In this post, we'll walk through the different parallelism strategies and how to compose them to enable
training at such scale.

## Data Parallelism (dp_replicate_size)

Data parallelism (DP) is the most common technique for training models across multiple GPUs, and involves replicating the model, gradients and optimizer states across each device, whilst evenly distributing data batches between GPUs, and synchronising gradients across devices before updating parameters. This can significantly increase throughput compared to single-device training, but requires that your model is able to fit on a single GPU. We can control
the number of replicas of the model with the `dp_replicate_size` parameter.

DP is a `top-most-level` parallelism strategy, meaning that if we use `dp_replicate_size=2` and we compose it with other parallelism strategies, there would be 2 replicas of the model, each also influenced by the other parallelism strategies. For example, if we use `dp_replicate_size=2` and `tp_size=2`, we would have 2 replicas of the model, each with 2 tensor parallel shards, but more on that later.

## Fully Sharded Data Parallelism (dp_shard_size)

What if our model is too large to fit on a single GPU? Fully sharded data parallel (FSDP) addresses this issue by sharding (distributing evenly) the model’s weights, gradients, and optimizer states across GPUs (this is inspired by DeepSpeed’s ZeRO-3), whilst each device still receives its portion of the full batch of data. As you may notice from the diagram above, rather than requiring a full copy of the entire model on each device, we only gather the weights for a single layer at a time before the forward pass, after which the weights may be sharded again.

In this way, we trade memory usage for the communication overhead of gathering sharded parameters before each forward and backward pass, and reduce-scatter-ing local gradients. We can control this trade-off in FSDP by tuning the granularity at which parameters are gathered. One one extreme, we can gather and re-reshard every layer of our model, which would result in the lowest peak memory usage, but incur the highest communication costs. In practice, a common approach is to gather the weights for an entire transformer decoder block at a time.

Whilst we can make further memory-compute tradeoffs and offload model parameters and gradients to the CPU to train larger models, this can be prohibitively slow. Instead, let’s consider how we can effectively utilise even more devices to train larger models whilst maintaining high data throughput.

We use the term node to refer to a single machine which hosts multiple GPUs (often 8), with fast intra-node communication channels using e.g. NVLink between devices. When utilising multiple nodes for training, we rely on relatively slower inter-node communication channels between machines using e.g. Infiniband. We also refer to the total number of devices in the process pool as the world size - e.g. a single node with 8 GPUs represents a world size of 8, and 4 nodes would represent a world size of 32.

When using FSDP across multiple nodes, we treat the entire set of devices across nodes as if we were training on a single node. For example, with 4 nodes containing 8 GPUs each, we perform our sharding across 32 devices, and perform our collective all-reduce and reduce-scatter operations using both inter-and-intra-node communication backends. In this manner, FSDP alone can scale to a substantial number of GPUs with a large global batch size to increase data throughput. However, there comes a point where several challenges arise that may require composing FSDP with other parallelism techniques. We usually try to avoid doing FSDP across more than a full node, as the communication overhead can become too high, we'll talk about how to address this in the section on `Hybrid Sharded Data Parallelism`.

## Tensor Parallelism (tp_size)

Tensor Parallel (TP) is a kind of model parallelism technique, where shards of the model permanently live on separate devices, and in contrast to data parallel techniques, each device receives an identical batch of data. TP works by distributing the computation of linear layers across devices, so each device only computes a portion of the matrix multiplication. This technique works best when there are large linear layers, such as the feed-forward layers in transformer models, which can be split across devices. We can also use TP on the `qkvo` projections in the attention layers, with almost no extra cost.

To achieve best performance, parameters of consecutive layers can be distributed in a specific fashion, minimizing the required communication. When working with pairs of linear layers, we can split the first layer column-wise, and the subsequent layer row-wise, allowing us to compute the output with only a single all-reduce operation to combine the sharded outputs. 

Unlike the dynamic sharding behaviour of FSDP, TP creates static memory partitions which result in a constant memory usage reduction scaling with the total world size. This becomes crucial for massive models where even a single decoder layer is too large to fit into memory during the FSDP all-gather (recall that common practice in FSDP is to gather the weights of an entire decoder layer at a time). However, unlike FSDP which scales relatively linearly across nodes (up to a point - ~512 GPUs on a homogenous cluster, way less across the internet), TP is only effective within the boundaries of a single node. TP requires frequent activation synchronization between devices during computation, as 
each device computes only a portion of the output, requiring the outputs from other devices to be communicated before
continuing the forward pass. Thus if we wish to utilise TP in a multi-node setup, we must consider composing parallelism techniques, while keeping TP only within a single node.

## Context Parallelism (cp_size)

Recently, reasoning made sequence lengths skyrocket - we need a way to train models on large sequence lengths, sometimes reaching even millions
With quadratic scaling of attention, it is almost impossible on a single gpu - with sequence length of 128k, 1 attention matrix will take 128k * 128k * 2 bytes * num_heads = ~32 GB * num_heads bytes.
With CP we can shard the inputs across sequence dimension, resulting in each GPU having smaller `qkv` matrices, thus
computing only a portion of the attention matrix.

How do we ensure the attention is computed correctly? Remember that we only need our shard of `q`, but we need the
full `k` and `v` matrices to compute the attention. We can achieve this by using a technique called `ring-attention`,
which works as follows:
1. Each GPU holds its shard of `q, k, v`.
2. Each GPU computes the partial attention matrix for its shard of `q` and its shards of `k, v`.
3. Each GPU sends its shard of `k, v` to the next GPU in the ring.
4. Each GPU receives the shard of `k, v` from the previous GPU in the ring.
5. Each GPU computes another part of the local attention matrix using the received `k, v` shards.
6. Each GPU repeats this process until all shards of `k, v` have been received and processed.

Accelerate enables this with the `accelerator.maybe_context_parallel` decorator, which is also showcased in 
the [accelerate example](https://github.com/huggingface/accelerate/blob/main/examples/fsdp2/nd_parallel.py).
You can also learn more about how it works and its limitations in our [concept guide](https://huggingface.co/docs/accelerate/main/en/concept_guides/context_parallelism).

## N-D Paralllelism

Now that you know how the different parallelism strategies work, let's see how we can compose them to train large models efficiently. Let's start with the simple ones: 2D Parallelism

## Hybrid Sharded Data Parallelism (dp_replicate_size, dp_shard_size)

Hybrid Sharded Data Parallelism (HSDP) is a kind of 2D parallelism which performs FSDP within a node, and DP across nodes - that is to say the model is replicated across each node, and sharded using FSDP within each node. This allows the greater communication overhead of FSDP to utilize the faster intra-node links, whilst DP minimises the slower inter-node communication overhead to a single gradient synchronisation step.

It’s important to note that we can freely configure the shape of our 2D network topology, as we aren’t constrained to the dimensions being aligned with physical node boundaries - you might apply FSDP across 2 nodes whilst replicating across groups of 2 nodes, which would result in lower memory usage but slower throughput, but still reduce the intra-node FSDP communication overhead by a factor of two. This is a knob we encourage you to tune to your specific hardware setup and fine-tuning needs.

## FSDP + Tensor Parallelism (dp_shard_size, tp_size)

As we mentioned earlier, TP should be applied within a node to utilize the high-bandwidth intra-node communications, thus, combining TP and FSDP involves sharding the model across nodes using FSDP, and within a node using TP. To a certain degree, this potentially offers a neat solution to all three of the issues above: the latency costs from FSDP could be reduced by a factor of 8, layers that are too large to fit on a single device are now evenly distributed across devices, and since each TP group receives an identical batch of data, we can also reduce our global batch size by a factor of 8. However, if this remains insufficient, we are unable to increase the TP size across nodes and must consider an alternative approach.

## FSDP + Context Parallelism (dp_shard_size, cp_size)

This is a 2D parallelism strategy that combines FSDP and CP, it's not very commonly used, as CP already combines with
FSDP (more on why in the [concept guide](https://huggingface.co/docs/accelerate/main/en/concept_guides/context_parallelism)), but it can be useful in some cases. I.e. when requiring a large sequence length, consequently
requiring a large `cp_size`. If this still doesn't fit into your memory budget, you can apply FSDP on top of this, further reducing the memory usage.

## Hybrid Sharded Data Parallelism + Tensor Parallelism (dp_replicate_size, dp_shard_size, tp_size)

With a sufficiently large world size (note: while the minimum world size for 3D parallelism is 8, it is most effective at scale), we can consider combining HSDP with TP which creates a hierarchy where DP first replicates the model across groups of nodes, FSDP then shards the model within each group, and TP splits individual layers within each node. This is a very common strategy, providing good trade-offs between memory usage and throughput.

## Some notes:

While we may talk about the remaining parallelism combinations, we feel like it's going to have very diminishing returns. You can combine any of the above parallelism strategies, in any way. We encourage you to experiment with 
this, gain some intuition, because the future is distributed!

