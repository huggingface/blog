---
title: "Optimizing diffusion inference for production-ready speeds - II" 
thumbnail: /blog/assets/productionizing-diffusion/productionizing-diffusion-thumbnail-2.png
authors:
- user: a-r-r-o-w
---

# Optimizing diffusion inference for production-ready speeds - II

Diffusion models have rapidly advanced generative modeling across a wide range of modalities - from images and video to music, 3D objects, and even text generation and world simulations recently. They are now central to state-of-the-art image and video generation, offering high-quality, controllable, and diverse outputs. However, their computational cost remains a bottleneck for real-world deployment. In this series, we explore techniques to optimize diffusion inference for text-to-image and text-to-video generation.

This post is second in a four-part series. We will cover the following topics:
1. How text-to-image diffusion models work and their computational challenges?
2. Standard optimizations for transformer-based diffusion models
3. Going deep: using faster kernels, non-trivial fusions, precomputations
4. Context parallelism
5. Quantization
6. Caching
7. LoRA
8. Training
9. Practice: Wan text-to-video
10. Optimizing inference for uncommon deployment environments using Triton

| Post | Topics covered |
|------|----------------|
| Optimizing diffusion inference for production-ready speeds - I   | 1, 2 |
| Optimizing diffusion inference for production-ready speeds - II  | 3, 4 |
| Optimizing diffusion inference for production-ready speeds - III | 5, 6 |
| Optimizing diffusion inference for production-ready speeds - IV  | 7, 8, 9, 10 |

The code for the entire series is available at [huggingface/productionizing-diffusion](https://github.com/huggingface/productionizing-diffusion). For this post, refer to the `post_2` directory. The guides are written to work on A100/H100 or better GPUs, but the ideas can be adapted to other hardware as well.

## Table of contents

- [Optimizations](#optimizations)
  - [Modeling rewrites](#modeling-rewrites)
  - [Precomputations](#precomputations)
  - [Fused Adaptive LayerNorm Linears](#fused-adaptive-layernorm-linears)
  - [Flash Attention 2 & 3 & 4](#flash-attention-2--3--4)
  - [Context Parallelism](#context-parallelism)
  - [CUDA Streams](#cuda-streams)
- [Benchmarks](#benchmarks)
  - [Cost Analysis](#cost-analysis)
- [Additional reading](#additional-reading)

## Optimizations

In this section, we'll cover some of the more advanced optimizations to improve inference speed. They require a slightly deeper understanding of the model architectures. The important takeaway is that implementations written for maintainability, research/educatational purposes, readability, or training purposes are not necessarily optimized for inference speed. In my personal opinoin, model implementations should be written differently for inference and training purposes.

Some definitions before we proceed:
- **Throughput**: This is the number of requests that can be processed in a given time period. It is usually measured in requests per second (RPS). When referring to throughput in this series of posts, we will be referring to the total number of tokens that can be processed per second. Maximizing memory usage by processing more context in parallel and increasing GPU utilization are common ways to increase throughput.
- **Latency**: This is the time taken to process a single request. It is usually measured in seconds (s). Latency is of utmost importance for inference providers, as it directly affects user experience. Lower latency means faster response times and better user satisfaction. Our focus in this series of articles is on reducing latency for a single request - to make a single image/video generation request as fast as possible.

### Modeling rewrites

Pytorch provides a clean interface to a suite of expert-optimized kernel implementations that run extremely fast on any modern GPU. Many common operations in model implementations have high-performance kernels available. For example, in Flux, a frequently used pattern combines layer normalization with pointwise scaling and shifting:

```python
x = self.norm(x) * (1 + scale) + shift
```

This is known as "AdaLN-Zero" (adaptive layer normalization; see section 3.2 of [Scalable Diffusion Models with Transformers](https://huggingface.co/papers/2212.09748) for more details). It involves a reduction (the mean/standard deviation computation), normalization (elementwise subtraction by mean, and division by standard deviation), followed by pointwise multiplication (with `1 + scale`) and addition (with `shift`).

When naively implemented, this results in multiple kernel launches. These operations are memory-bound. Ideally, we want to operate in the compute-bound regime, where execution is limited by arithmetic throughput rather than memory latency. One key optimization is to fuse multiple memory-bound operations, that is combining multiple operations into a single kernel to reduce memory accesses. For a deeper explanation, see Horace He’s blog post: [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html).

![Fused example](https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/fused.png)

These execution patterns can easily be detected and fused using the torch compiler. However, further gains can be achieved by using explicitly fused primitives like `torch.addcmul` and `torch.mv`, which offer performance benefits in both eager and compiled modes. Usage of these primitives is inspired from the Adam optimizer [implementation](https://github.com/pytorch/pytorch/blob/39b71d11fc2dd9b4da6d23a34eb29aefbb1df672/torch/optim/adam.py#L416).

```python
norm_x = self.norm(x)
x = torch.addcmul(shift, norm_x, 1 + scale)
```

The reason for why this is slightly faster can be found by inspecting the underlying C++ code, which does a single kernel launch, loop unrolling, vectorized operations, and other optimizations: [addcmul_cuda_kernel](https://github.com/pytorch/pytorch/blob/39b71d11fc2dd9b4da6d23a34eb29aefbb1df672/aten/src/ATen/native/cuda/PointwiseOpsKernel.cu#L70), [gpu_kernel](https://github.com/pytorch/pytorch/blob/39b71d11fc2dd9b4da6d23a34eb29aefbb1df672/aten/src/ATen/native/cuda/Loops.cuh#L103) and [gpu_kernel_impl](https://github.com/pytorch/pytorch/blob/39b71d11fc2dd9b4da6d23a34eb29aefbb1df672/aten/src/ATen/native/cuda/CUDALoops.cuh#L568)

<details>
<summary> addcmul benchmark </summary>

```python
import torch


def adaln_zero(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    norm_x = torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:], eps=1e-6)
    return norm_x * (1 + scale) + shift


def adaln_zero_addcmul(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    norm_x = torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:], eps=1e-6)
    return torch.addcmul(shift, norm_x, 1 + scale)


def timeit(func, x, *args, num_warmups=10, num_repeats=100):
    for _ in range(num_warmups):
        func(x, *args)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_repeats):
        func(x, *args)
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / num_repeats
    return elapsed_time


torch.manual_seed(0)
device = "cuda"
dtype = torch.bfloat16
num_warmups = 10
num_repeats = 100
batch_size, seq_len, hidden_size = 1, 4608, 3072

x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
scale = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)
shift = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)

time_eager = timeit(adaln_zero, x, scale, shift, num_warmups=num_warmups, num_repeats=num_repeats)
time_addcmul = timeit(adaln_zero_addcmul, x, scale, shift, num_warmups=num_warmups, num_repeats=num_repeats)
time_compile = timeit(torch.compile(adaln_zero, mode="default", fullgraph=True), x, scale, shift, num_warmups=num_warmups, num_repeats=num_repeats)

print(f"{time_eager=:.5f} ms")
print(f"{time_addcmul=:.5f} ms")
print(f"{time_compile=:.5f} ms")
```

```
time_eager=0.17715 ms
time_addcmul=0.11731 ms
time_compile=0.07861 ms
```

</details>

### Precomputations

Precomputing repeated operations in the forward pass that do not depend on user-provided input can help speedup inference in some cases. It really depends on the model architecture and the compute cost of the given operation.

In Flux, we can notice that the computations for [guidance embeddings](https://github.com/huggingface/diffusers/blob/3b079ec3fadfc95240bc1c48ae86de28b72cc9f2/src/diffusers/models/embeddings.py#L1620), [context projections](https://github.com/huggingface/diffusers/blob/3b079ec3fadfc95240bc1c48ae86de28b72cc9f2/src/diffusers/models/transformers/transformer_flux.py#L260), [pooled projections](https://github.com/huggingface/diffusers/blob/3b079ec3fadfc95240bc1c48ae86de28b72cc9f2/src/diffusers/models/embeddings.py#L1621) and [RoPE embeddings](https://github.com/huggingface/diffusers/blob/3b079ec3fadfc95240bc1c48ae86de28b72cc9f2/src/diffusers/models/transformers/transformer_flux.py#L251) do not vary over inference steps. So, we can precompute them once and reuse them in the forward pass.

Another possibility to consider is the [timestep embeddings](https://github.com/huggingface/diffusers/blob/3b079ec3fadfc95240bc1c48ae86de28b72cc9f2/src/diffusers/models/embeddings.py#L1644). In inference provider UIs and APIs, the number of inference steps is often fixed, or has a maximum limit, or is not user-configurable. In such cases, we can precompute the timestep embeddings for all possible inference steps and cache them. This only has minimal memory overhead while saving a few milliseconds. We will assume that the max inference steps that a user can set is `50` and write our implementation accordingly.

### Fused Adaptive Layernorm Linear

Remember the Fused QKV idea from the previous post? It is a great example of how to fuse multiple linear layers (multiple matmuls), into a single layer (one matmul). Without fusing, they would incur overheads from 3 kernel launches at each block, whereas the fused variant would result in a single matmul kernel launch. Recall: "Any set of linear layers that operate on the same input can be fused into a single linear layer". This optimization is relatively underused in practice, but can yield significant speedups, especially in large models with many linear layers.

In Flux, we have `19` dual-stream transformer blocks and `38` single-stream transformer blocks. Each dual stream block has two AdaLN layers, and each single stream block has one AdaLN layer. Each AdaLN layer has a linear layer. Overall, we can fuse `19 * 2 + 38 = 76` linear layers into a single layer. In practice, this results in some speedup for inference, but comes at a large memory cost. To keep the implementation clean, we keep two separate fused linear layers for each type of block (as there is not much difference in having one fused linear layer for all AdaLN layers, compared to two separate fused linear layers for dual and single stream blocks, because they already contribute to very large matmul operations).

```
Total dual stream layers: 19
Total single stream layers: 38

Total AdaLN linears in dual stream blocks: 19 (image_stream) + 19 (text_stream) = 38
Total AdaLN linears in single stream blocks: 38 (joint image+text stream)

(in_features, out_features) of dual stream AdaLN linears: (3072, 6 * 3072)
(in_features, out_features) of single stream AdaLN linears: (3072, 3 * 3072)

(in_features, out_features) of fused dual stream AdaLN linear: (3072, 38 * 6 * 3072) = (3072, 700416)
(in_features, out_features) of fused single stream AdaLN linear: (3072, 38 * 3 * 3072) = (3072, 350208)

batch_size = 1
context_length = 4096 + 512 = 4608  # 4096 for image, 512 for text

memory_required for fused dual stream AdaLN linear outputs: (1 * 4608 * 700416 * 2) / (1024 ** 3) = 6.01 GB
memory_required for fused single stream AdaLN linear outputs: (1 * 4608 * 350208 * 2) / (1024 ** 3) = 3.00 GB
```

We require an extra ~9 GB memory upfront. Flux is a relatively small model and leaves plenty of memory available on an 80 GB GPU (A100/H100), and so we can afford this optimization.

Note: With unusual matmul shapes like this, where the output dimension is much larger than the input dimension, there may be an opportunity to optimize matmul algorithms to outperform the underlying cuBLAS implementations that pytorch calls into. I'm actively looking into this and may post updates in the future. However, this operation takes an extremely low percentage of the total inference time, so it may not be worth the extra effort.

### Flash Attention 2 & 3 & 4

[Flash Attention](https://github.com/Dao-AILab/flash-attention) represents a significant breakthrough in accelerating transformers. The main idea is to reduce the number of global memory (gmem) accesses and instead make better use of high-bandwidth memory and on-chip SRAM. Since gmem accesses are very slow compared to caches/shared memory, minimizing these accesses is key to high performance - akin to the difference between intercontinental travel and intra-city transit.

FA2 is well-suited for older GPUs and has been integrated natively into Pytorch, providing substantial improvements over previous methods. FA3/FA4 targets newer GPUs (Hopper and Blackwell architectures) and is optimized to leverage newer hardware features like the [TMA](https://pytorch.org/blog/hopper-tma-unit/) unit, warp specialization for overlapping computation with memory access, FP8 support, and more. It is the fastest attention implementation available today - an essential ingredient adopted by many inference providers.

Additionally, the cuDNN attention backend in PyTorch offers performance that closely approaches FA3’s performance. It is readily accessible without the often onerous process of compiling FA3 from source, which can be time-consuming and requires substantial RAM. Despite this convenience, the cuDNN backend still lags behind FA3 on my [personal benchmarks](https://gist.github.com/a-r-r-o-w/58425fd303633e3c3702283b4687599d), and so we'll make use of FA3 in our implementation.

<details>
<summary> Flash Attention setup </summary>

**Installation**

```shell
# Refer to https://github.com/Dao-AILab/flash-attention/tree/main/README.md
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper
# We install v2.7.4.post1 because the latest release (2.8.x) might cause
# some installation issues which are hard to debug
git checkout v2.7.4.post1
python setup.py install
```

**Usage**

```python
try:
    from flash_attn import flash_attn_func
except:
    print("Flash Attention 2 not found.")

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except:
    print("Flash Attention 3 not found.")


# For fullgraph=True tracing to be compatible
@torch.library.custom_op("flash_attn_3::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _wrapped_flash_attn_3(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    out, lse = flash_attn_3_func(query, key, value)
    return out


@torch.library.register_fake("flash_attn_3::_flash_attn_forward")
def _(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(query)


def _attention_flash_attn_2(query, key, value):
    return flash_attn_func(query, key, value)


def _attention_flash_attn_3(query, key, value):
    out = _wrapped_flash_attn_3(query, key, value)
    return out
```

If you do not want to go through the hassle of setting up and compiling the wheels of Flash Attention yourself, please check out the [HF kernels](https://github.com/huggingface/kernels) project and the accompanying [FA2](https://huggingface.co/kernels-community/flash-attn) and [FA3](https://huggingface.co/kernels-community/flash-attn3) kernels.

</details>

### Context Parallelism

Parallelizing computation across multiple GPUs is a common technique to enable faster training/inference speeds. There are multiple approaches for parallelism. Some parallelism strategies increase throughput, others reduce latency, while some optimize for other metrics like memory usage and model size.

A quick overview of the different parallelism strategies used in inference services:
- **Data parallelism**: This is parallelization across the batch dimension. This approach is useful when scaling the number of requests that can be processed simultaneously. It is not useful for speeding up inference for a single request. Batching multiple requests together on a single GPU may increase the overall throughput, but have an adverse effect on latency due to increased compute requirements.
- **Tensor parallelism**: This is parallelization across the embedding dimension. Most operations in transformer models are a series of matrix multiplications on the embedding dimension. Tensor parallelism splits these matrix multiplications across multiple GPUs, allowing for larger models to be fit into memory. This is useful for training and inference of really large models, but does not help much for diffusion models like Flux, which is relatively small in size.
- **Context parallelism**: This is parallelization across the sequence length dimension. This is, by far, the most useful parallelism strategy for diffusion inference, as it allows us to process longer sequences in parallel, reducing the time taken to process a single request. It is the key ingredient behind various inference provider services being able to generate images and videos in just a few seconds. More about CP is discussed below.

Note: We don't cover an explanation for pipeline parallelism here, as it is mostly beneficial for training large models and increasing throughput. To the best of my knowledge, this technique is not used for inference acceleration for diffusion image/video models. We also don't cover expert parallelism, as it is not applicable to Flux (which does not use MoE layers). EP might be useful for [HiDream](https://github.com/HiDream-ai/HiDream-I1) - a model with a similar architecture to Flux, which we'll cover in a future post.

Parallelism across the sequence dimension involves sharding the input sequence into multiple segments and processing each segment on a separate GPU. For example, take a sequence of length `4096` to apply CP across `4` GPUs. Each GPU will process an input segment of length `1024`. Outputs will be gathered from all GPUs and concatenated to obtain the final result. Each GPU encounters a much smaller part of the problem shape, which allows for faster processing.

Applying CP across the sequence dimension is not as straightforward as it may seem though. For layers that operate on the embedding dimension, it can be applied trivially (e.g. linear layers and normalization layers). However, for layers that operate on the sequence dimension, such as attention layers, the implementation is more complex in practice. Let's understand why.

$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V; Q, K, V \in \mathbb{R}^{seq\_len \times head\_dim} $

The commonly used [attention](https://huggingface.co/papers/1706.03762) operation is simply two matrix multiplications in succession, but with a softmax in between. The first matmul computes the attention scores `P` (`batch_size, num_heads, seq_len, seq_len`) using the `Q` and `K` tensors. The softmax computation involves a reduction and elementwise operations across the last dim to compute scores `S`. The second matmul computes the output `O` (`batch_size, num_heads, seq_len, head_dim`) using the `S` and `V` tensors. For the sake of simplicity, we can ignore the `batch_size` and `num_heads` dimensions, as the computation is independent across them.

Applying parallelism across the two matmuls would be very straightforward if we actually performed two separate matmuls. In practice, we have highly-optimized attention implementations that perform the entire output computation in a single fused kernel. Relevant reading includes the Online Softmax and Flash Attention papers.

There are three well-known approaches for implementing context parallelism:
- **Ring Attention**: [[Paper](https://arxiv.org/abs/2310.01889)] - Ring Attention splits the input sequence across GPUs, so each GPU (rank) holds a shard of tokens. For layers that act on the embedding dimension (like linear or normalization layers), no communication is needed - each token is independent. Attention layers are trickier: to compute the attention output for a query, you need the full set of keys and values. Ring Attention solves this by having each rank progressively gather the key/value shards from all other ranks in a "ring" fashion. Partial attention outputs are updated online as new shards arrive, similar to online softmax trick.
- **Ulysses Attention**: [[Paper](https://arxiv.org/abs/2309.14509)] - Ulysses Attention takes a different approach. Like Ring, it leaves embedding-dimension layers untouched. But for attention, instead of incrementally updating partial outputs, it gathers the full sequence across all ranks, but splits the attention heads among ranks (with an all-to-all communication). Each rank computes full attention outputs for its assigned heads. Finally, attention heads are gathered and sequence length is split across all ranks to match the original sequence layout. This method makes the computation more structured and often easier to implement with existing attention kernels. For Ring attention, the online update is only possible if the underlying kernel returns the [Log-Sum-Exp](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/).
- **Unified Attention**: [[Paper](https://arxiv.org/abs/2405.07719)] - This approach is essentially a hybrid that combines the strengths of Ring and Ulysses. It allows blockwise computation of partial attention outputs like Ring, while also leveraging attention-head parallelism like Ulysses. This makes it flexible: you can tune it to favor lower memory usage, higher throughput, or a balance of both, depending on your hardware and model size.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/ulysses_ring.png" alt="Ring & Ulysses visualized" width="768px">

<sup> [Image source](https://arxiv.org/abs/2405.07719) </sup>

A full deep-dive of each of these approaches is beyond the scope of this post, but we can quickly build an intuition of what happens on each rank by taking a look at their sequential single-GPU implementations*.

<details>
<summary> Sequential ring attention </summary>

```python
import torch

torch.manual_seed(42)


def torch_sdpa(query, key, value):
    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_cudnn_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=None,
            compute_log_sumexp=True,
        )
    )
    return out, lse


# partial_queries, partial_keys, partial_values are lists of tensors. Each tensor in the list can be thought
# of as the sharded chunk held by each rank in the distributed setting.
def ring_sdpa_sequential(partial_queries, partial_keys, partial_values, *, world_size: int = 1, convert_to_fp32: bool = True):
    outputs, lses = [], []
    
    # This loop does not actually exist in the distributed setting. We are simulating what happens on each
    # rank sequentially, so think of this loop being parallelized across the GPUs.
    for rank in range(world_size):
        # Each rank has its own chunk of the full context QKV projections.
        query, key, value = partial_queries[rank], partial_keys[rank], partial_values[rank]
        
        # The next rank is the one that will communicate its KVs to the current rank. We will iteratively
        # update this so that we can have access to all the KVs from all ranks.
        next_rank = (rank + 1) % world_size
        
        prev_out = prev_lse = None

        # This loop simulates the communication between ranks in a ring fashion.
        for i in range(world_size):
            if i > 0:
                # Communicate with the next rank to get its KVs.
                key, value = partial_keys[next_rank], partial_values[next_rank]
                # Update next_rank to the next in the ring.
                next_rank = (next_rank + 1) % world_size
            
            # Compute local attention with the KVs available on current rank and the partial Q.
            out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
                torch.ops.aten._scaled_dot_product_cudnn_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_bias=None,
                    compute_log_sumexp=True,
                )
            )

            if convert_to_fp32:
                out = out.to(torch.float32)
                lse = lse.to(torch.float32)
            
            # Refer to https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795 to understand
            # the online update trick. It is similar to what we do in online-softmax and flash-attention.
            lse = lse.unsqueeze(-1)
            if prev_out is not None:
                out = prev_out - torch.nn.functional.sigmoid(lse - prev_lse) * (prev_out - out)
                lse = prev_lse - torch.nn.functional.logsigmoid(prev_lse - lse)
            prev_out = out
            prev_lse = lse
        
        out = out.to(query.dtype)
        lse = lse.squeeze(-1)

        # In the distributed setting, we would gather the outputs from all ranks at the end of the transformer blocks.
        # Here, we simply append the outputs/lses into a list that will have size `world_size`.
        outputs.append(out)
        lses.append(lse)
    
    return outputs, lses


device = "cuda"
dtype = torch.bfloat16
world_size = 4

batch_size = 1
image_sequence_length = 4096
text_sequence_length = 512
sequence_length = image_sequence_length + text_sequence_length
num_attention_heads = 24
attention_head_dim = 128

query = torch.randn(batch_size, num_attention_heads, sequence_length, attention_head_dim, device=device, dtype=dtype)
key = torch.randn(batch_size, num_attention_heads, sequence_length, attention_head_dim, device=device, dtype=dtype)
value = torch.randn(batch_size, num_attention_heads, sequence_length, attention_head_dim, device=device, dtype=dtype)
partial_queries = query.chunk(world_size, dim=2)
partial_keys = key.chunk(world_size, dim=2)
partial_values = value.chunk(world_size, dim=2)

torch_sdpa_out, torch_sdpa_lse = torch_sdpa(query, key, value)
ring_sdpa_out, ring_sdpa_lse = ring_sdpa_sequential(partial_queries, partial_keys, partial_values, world_size=world_size)

all_ring_sdpa_out = torch.cat(ring_sdpa_out, dim=2)
all_ring_sdpa_lse = torch.cat(ring_sdpa_lse, dim=2)

assert torch_sdpa_out.shape == all_ring_sdpa_out.shape, "Output shapes do not match!"
assert torch_sdpa_lse.shape == all_ring_sdpa_lse.shape, "LSE shapes do not match!"
assert torch.allclose(all_ring_sdpa_out, torch_sdpa_out, atol=1e-3, rtol=1e-3), "Outputs do not match!"
assert torch.allclose(all_ring_sdpa_lse, torch_sdpa_lse, atol=1e-3, rtol=1e-3), "LSE values do not match!"
```

</details>

<details>
<summary> Sequential ulysses attention </summary>

```python
import torch

torch.manual_seed(42)


def torch_sdpa(query, key, value):
    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_cudnn_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=None,
            compute_log_sumexp=True,
        )
    )
    return out, lse


# partial_queries, partial_keys, partial_values are lists of tensors. Each tensor in the list can be thought
# of as the sharded chunk held by each rank in the distributed setting.
def ulysses_sdpa_sequential(partial_queries, partial_keys, partial_values, *, world_size: int = 1):
    B, H, S_LOCAL, D = partial_queries[0].shape
    H_LOCAL = H // world_size
    
    outputs, lses = [], []
    
    # This loop does not actually exist in the distributed setting. We are simulating what happens on each
    # rank sequentially. The equivalent of this loop is the 3 all-to-all communication steps in the distributed setting.
    # We enter ulysses attention with sharded QKV projections on each rank.
    # The shape of each partial QKV chunk is (B, H, S // world_size, D).
    # We reshape them to (world_size, S // world_size, B, H // world_size, D) to gather the entire sequence but shard the attention heads.
    # This is done via the all-to-all communication step.
    for partials in [partial_queries, partial_keys, partial_values]:
        for rank in range(world_size):
            x_local = partials[rank]
            # (B, H, S // world_size, D) -> (world_size, S // world_size, B, H // world_size, D)
            partials[rank] = x_local.reshape(B, world_size, H_LOCAL, S_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()    
        x = all_to_all_single_sequential(partials, world_size)
        for rank in range(world_size):
            x_local = x[rank]
            # (S, B, H // world_size, D) -> (B, H // world_size, S, D)
            partials[rank] = x_local.permute(1, 2, 0, 3).contiguous()

    # This loop does not actually exist in the distributed setting. We are simulating what happens on each rank.
    for rank in range(world_size):
      # Compute full attention across sequence but sharded across heads.
        query_local, key_local, value_local = partial_queries[rank], partial_keys[rank], partial_values[rank]
        out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
            torch.ops.aten._scaled_dot_product_cudnn_attention(
                query=query_local,
                key=key_local,
                value=value_local,
                attn_bias=None,
                compute_log_sumexp=True,
            )
        )
        outputs.append(out)
        lses.append(lse)
    
    # The following loops don't actually exist in the distributed setting. We are simulating what happens on each rank
    # for the all-to-all communication steps, and do the reverse of what was done in the first loop.
    # That is, we take outputs of shape (B, H // world_size, S, D),
    # reshape to (world_size, H // world_size, B, S // world_size, D),
    # perform all-to-all communication,
    # and then reshape back to (B, H, S // world_size, D).
    for rank in range(world_size):
        out_local = outputs[rank]
        lse_local = lses[rank]
        # (B, H // world_size, S, D) -> (B, H // world_size, world_size, S // world_size, D) -> (world_size, H // world_size, B, S // world_size, D)
        outputs[rank] = out_local.reshape(B, H_LOCAL, world_size, S_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        lses[rank] = lse_local.reshape(B, H_LOCAL, world_size, S_LOCAL).permute(2, 1, 0, 3).contiguous()
    outputs = all_to_all_single_sequential(outputs, world_size)
    lses = all_to_all_single_sequential(lses, world_size)
    for rank in range(world_size):
        out_local = outputs[rank]
        lse_local = lses[rank]
        # (H, B, S // world_size, D) -> (B, H, S // world_size, D)
        outputs[rank] = out_local.permute(1, 0, 2, 3).contiguous()
        lses[rank] = lse_local.permute(1, 0, 2).contiguous()

    return outputs, lses


def all_to_all_single_sequential(partials, world_size):
    output_partials = []
    for i in range(world_size):
        received_chunks = [p[i] for p in partials]
        output_partials.append(torch.cat(received_chunks, dim=0))
    return output_partials


device = "cuda"
dtype = torch.bfloat16
world_size = 4

batch_size = 1
image_sequence_length = 4096
text_sequence_length = 512
sequence_length = image_sequence_length + text_sequence_length
num_attention_heads = 24
attention_head_dim = 128

query = torch.randn(batch_size, num_attention_heads, sequence_length, attention_head_dim, device=device, dtype=dtype)
key = torch.randn(batch_size, num_attention_heads, sequence_length, attention_head_dim, device=device, dtype=dtype)
value = torch.randn(batch_size, num_attention_heads, sequence_length, attention_head_dim, device=device, dtype=dtype)
partial_queries = list(query.chunk(world_size, dim=2))
partial_keys = list(key.chunk(world_size, dim=2))
partial_values = list(value.chunk(world_size, dim=2))

torch_sdpa_out, torch_sdpa_lse = torch_sdpa(query, key, value)
ulysses_sdpa_out, ulysses_sdpa_lse = ulysses_sdpa_sequential(partial_queries, partial_keys, partial_values, world_size=world_size)

all_ulysses_sdpa_out = torch.cat(ulysses_sdpa_out, dim=2)
all_ulysses_sdpa_lse = torch.cat(ulysses_sdpa_lse, dim=2)

assert torch_sdpa_out.shape == all_ulysses_sdpa_out.shape, "Output shapes do not match!"
assert torch_sdpa_lse.shape == all_ulysses_sdpa_lse.shape, "LSE shapes do not match!"
assert torch.allclose(all_ulysses_sdpa_out, torch_sdpa_out, atol=1e-3, rtol=1e-3), "Outputs do not match!"
assert torch.allclose(all_ulysses_sdpa_lse, torch_sdpa_lse, atol=1e-3, rtol=1e-3), "LSEs do not match!"
```

</details>

<details>
<summary> Sequential unified attention </summary>

```python
import torch

torch.manual_seed(42)


def torch_sdpa(query, key, value):
    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_cudnn_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=None,
            compute_log_sumexp=True,
        )
    )
    return out, lse


def ring_sdpa_sequential(partial_queries, partial_keys, partial_values, *, ring_size: int = 1, convert_to_fp32: bool = True):
    outputs, lses = [], []
    
    for rank in range(ring_size):
        query, key, value = partial_queries[rank], partial_keys[rank], partial_values[rank]
        next_rank = (rank + 1) % ring_size
        prev_out = prev_lse = None

        for i in range(ring_size):
            if i > 0:
                key, value = partial_keys[next_rank], partial_values[next_rank]
                next_rank = (next_rank + 1) % ring_size
            
            out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
                torch.ops.aten._scaled_dot_product_cudnn_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_bias=None,
                    compute_log_sumexp=True,
                )
            )

            if convert_to_fp32:
                out = out.to(torch.float32)
                lse = lse.to(torch.float32)
            
            # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
            lse = lse.unsqueeze(-1)
            if prev_out is not None:
                out = prev_out - torch.nn.functional.sigmoid(lse - prev_lse) * (prev_out - out)
                lse = prev_lse - torch.nn.functional.logsigmoid(prev_lse - lse)
            prev_out = out
            prev_lse = lse
        
        out = out.to(query.dtype)
        lse = lse.squeeze(-1)
        outputs.append(out)
        lses.append(lse)
    
    return outputs, lses


def unified_ulysses_ring_sdpa_sequential(partial_queries, partial_keys, partial_values, *, ulysses_size: int = 1, ring_size: int = 1):
    B, H, S_LOCAL, D = partial_queries[0][0].shape
    H_LOCAL = H // ulysses_size
    
    outputs, lses = [], []
    
    for partials in [partial_queries, partial_keys, partial_values]:
        for ring_rank in range(ring_size):
            for rank in range(ulysses_size):
                x_local = partials[ring_rank][rank]
                partials[ring_rank][rank] = x_local.reshape(B, ulysses_size, H_LOCAL, S_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()    
            x = all_to_all_single_sequential(partials[ring_rank], ulysses_size)
            for rank in range(ulysses_size):
                x_local = x[rank]
                partials[ring_rank][rank] = x_local.permute(1, 2, 0, 3).contiguous()

    partial_queries = [list(x) for x in zip(*partial_queries)]
    partial_keys = [list(x) for x in zip(*partial_keys)]
    partial_values = [list(x) for x in zip(*partial_values)]
    
    for rank in range(ulysses_size):
        ring_outputs, ring_lses = ring_sdpa_sequential(partial_queries[rank], partial_keys[rank], partial_values[rank], ring_size=ring_size)
        outputs.append(ring_outputs)
        lses.append(ring_lses)
    
    outputs = [list(x) for x in zip(*outputs)]
    lses = [list(x) for x in zip(*lses)]
    
    for ring_rank in range(ring_size):
        for rank in range(ulysses_size):
            outputs[ring_rank][rank] = outputs[ring_rank][rank].reshape(B, H_LOCAL, ulysses_size, S_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
            lses[ring_rank][rank] = lses[ring_rank][rank].reshape(B, H_LOCAL, ulysses_size, S_LOCAL).permute(2, 1, 0, 3).contiguous()
        outputs[ring_rank] = all_to_all_single_sequential(outputs[ring_rank], ulysses_size)
        lses[ring_rank] = all_to_all_single_sequential(lses[ring_rank], ulysses_size)
        for rank in range(ulysses_size):
            outputs[ring_rank][rank] = outputs[ring_rank][rank].permute(1, 0, 2, 3).contiguous()
            lses[ring_rank][rank] = lses[ring_rank][rank].permute(1, 0, 2).contiguous()

    return outputs, lses


def all_to_all_single_sequential(partials, world_size):
    output_partials = []
    for i in range(world_size):
        received_chunks = [p[i] for p in partials]
        output_partials.append(torch.cat(received_chunks, dim=0))
    return output_partials


device = "cuda"
dtype = torch.bfloat16
WORLD_SIZE = 8
ulysses_size = 4
ring_size = 2
assert ulysses_size * ring_size == WORLD_SIZE, "ulysses_size * ring_size must equal WORLD_SIZE"

batch_size = 1
image_sequence_length = 4096
text_sequence_length = 512
sequence_length = image_sequence_length + text_sequence_length
num_attention_heads = 24
attention_head_dim = 128

query = torch.randn(batch_size, num_attention_heads, sequence_length, attention_head_dim, device=device, dtype=dtype)
key = torch.randn(batch_size, num_attention_heads, sequence_length, attention_head_dim, device=device, dtype=dtype)
value = torch.randn(batch_size, num_attention_heads, sequence_length, attention_head_dim, device=device, dtype=dtype)

partial_queries = list(query.chunk(WORLD_SIZE, dim=2))
partial_keys = list(key.chunk(WORLD_SIZE, dim=2))
partial_values = list(value.chunk(WORLD_SIZE, dim=2))

# R=1, U=4 => [[tensor1, tensor2, tensor3, tensor4]]
# R=2, U=2 => [[tensor1, tensor2], [tensor3, tensor4]]
# R=4, U=1 => [[tensor1], [tensor2], [tensor3], [tensor4]]
partial_queries = [partial_queries[i:i + ulysses_size] for i in range(0, WORLD_SIZE, ulysses_size)]
partial_keys = [partial_keys[i:i + ulysses_size] for i in range(0, WORLD_SIZE, ulysses_size)]
partial_values = [partial_values[i:i + ulysses_size] for i in range(0, WORLD_SIZE, ulysses_size)]

torch_sdpa_out, torch_sdpa_lse = torch_sdpa(query, key, value)
unified_sdpa_out, unified_sdpa_lse = unified_ulysses_ring_sdpa_sequential(partial_queries, partial_keys, partial_values, ulysses_size=ulysses_size, ring_size=ring_size)

all_unified_sdpa_out = torch.cat([torch.cat(out, dim=2) for out in unified_sdpa_out], dim=2)
all_unified_sdpa_lse = torch.cat([torch.cat(lse, dim=2) for lse in unified_sdpa_lse], dim=2)

assert torch_sdpa_out.shape == all_unified_sdpa_out.shape, "Output shapes do not match!"
assert torch_sdpa_lse.shape == all_unified_sdpa_lse.shape, "LSE shapes do not match!"
assert torch.allclose(all_unified_sdpa_out, torch_sdpa_out, atol=1e-3, rtol=1e-3), "Outputs do not match!"
assert torch.allclose(all_unified_sdpa_lse, torch_sdpa_lse, atol=1e-3, rtol=1e-3), "LSEs do not match!"
```

</details>

The following code snippets also demonstrate a templated distributed version of the above ideas. These can be used with any underlying attention provider like Torch, FA3, xformers, etc. The templated implementation is inspired from [pytorch experimental](https://github.com/pytorch/pytorch/blob/c78fce9e79b79686b87f4007cbaec819bdd0223f/torch/distributed/tensor/experimental/_attention.py#L283).

<details>
<summary> Templated implementations for Ring, Ulysses and Unified Attention </summary>

```python
import argparse
from dataclasses import dataclass
from typing import Callable, Literal, List

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed import DeviceMesh


@dataclass
class ContextParallelOptions:
    mode: Literal["ring", "ulysses", "unified"] = "ring"
    ring_mesh: DeviceMesh | None = None
    ulysses_mesh: DeviceMesh | None = None
    convert_to_fp32: bool = True
    op: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None


cp_options = ContextParallelOptions()


def _templated_ring_attention(query, key, value):
    rank = cp_options.ring_mesh.get_rank()
    world_size = cp_options.ring_mesh.size()

    if world_size == 1:
        return cp_options.op(query, key, value)
    
    next_rank = (rank + 1) % world_size
    prev_out = prev_lse = None
    
    kv_buffer = torch.cat([key.flatten(), value.flatten()]).contiguous()
    kv_buffer = funcol.all_gather_tensor(kv_buffer, gather_dim=0, group=cp_options.ring_mesh.get_group())
    kv_buffer = kv_buffer.chunk(world_size)

    for i in range(world_size):
        if i > 0:
            kv = kv_buffer[next_rank]
            key = kv[:key.numel()].reshape_as(key)
            value = kv[key.numel():].reshape_as(value)
            next_rank = (next_rank + 1) % world_size
        
        out, lse = cp_options.op(query, key, value)
        
        if cp_options.convert_to_fp32:
            out = out.to(torch.float32)
            lse = lse.to(torch.float32)
        
        lse = lse.unsqueeze(-1)
        if prev_out is not None:
            out = prev_out - torch.nn.functional.sigmoid(lse - prev_lse) * (prev_out - out)
            lse = prev_lse - torch.nn.functional.logsigmoid(prev_lse - lse)
        prev_out = out
        prev_lse = lse
    
    out = out.to(query.dtype)
    lse = lse.squeeze(-1)
    return out, lse


def _templated_ulysses_attention(query, key, value):
    world_size = cp_options.ulysses_mesh.size()
    group = cp_options.ulysses_mesh.get_group()
    
    if world_size == 1:
        return cp_options.op(query, key, value)

    B, H, S_LOCAL, D = query.shape
    H_LOCAL = H // world_size
    query, key, value = (
        x.reshape(B, world_size, H_LOCAL, S_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
        for x in (query, key, value)
    )
    query, key, value = (
        funcol.all_to_all_single(x, None, None, group=group).wait()
        for x in (query, key, value)
    )
    query, key, value = (
        x.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
        for x in (query, key, value)
    )
    out, lse = cp_options.op(query, key, value)
    out = out.reshape(B, H_LOCAL, world_size, S_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    lse = lse.reshape(B, H_LOCAL, world_size, S_LOCAL).permute(2, 1, 0, 3).contiguous()
    out = funcol.all_to_all_single(out, None, None, group=group).wait()
    lse = funcol.all_to_all_single(lse, None, None, group=group).wait()
    out = out.flatten(0, 1).permute(1, 0, 2, 3).contiguous()
    lse = lse.flatten(0, 1).permute(1, 0, 2).contiguous()
    return out, lse


def _templated_unified_attention(query, key, value):
    ring_size = cp_options.ring_mesh.size()
    ulysses_size = cp_options.ulysses_mesh.size()
    ulysses_group = cp_options.ulysses_mesh.get_group()
    world_size = ring_size * ulysses_size
    
    if world_size == 1:
        return cp_options.op(query, key, value)

    B, H, S_LOCAL, D = query.shape
    H_LOCAL = H // ulysses_size
    query, key, value = (
        x.reshape(B, ulysses_size, H_LOCAL, S_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
        for x in (query, key, value)
    )
    query, key, value = (
        funcol.all_to_all_single(x, None, None, group=ulysses_group).wait()
        for x in (query, key, value)
    )
    query, key, value = (
        x.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
        for x in (query, key, value)
    )
    out, lse = _templated_ring_attention(query, key, value)
    out = out.reshape(B, H_LOCAL, ulysses_size, S_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    lse = lse.reshape(B, H_LOCAL, ulysses_size, S_LOCAL).permute(2, 1, 0, 3).contiguous()
    out = funcol.all_to_all_single(out, None, None, group=ulysses_group).wait()
    lse = funcol.all_to_all_single(lse, None, None, group=ulysses_group).wait()
    out = out.flatten(0, 1).permute(1, 0, 2, 3).contiguous()
    lse = lse.flatten(0, 1).permute(1, 0, 2).contiguous()
    return out, lse


def torch_cudnn_attention(query, key, value):
    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_cudnn_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=None,
            compute_log_sumexp=True,
        )
    )
    return out, lse


def torch_flash_attention(query, key, value):
    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_flash_attention(
            query=query,
            key=key,
            value=value,
        )
    )
    return out, lse


OPS = {
    "cudnn": torch_cudnn_attention,
    "flash": torch_flash_attention,
}
WORLD_SIZE = -1
RANK = -1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=24)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 4224, 4352, 4480, 4608, 8192])
    parser.add_argument(
        "--ops",
        type=str,
        nargs="+",
        choices=list(OPS.keys()),
        default=list(OPS.keys()),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main(
    ring_degree: int,
    ulysses_degree: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    seq_lens: List[int],
    ops: List[str],
    seed: int,
):
    global cp_options, WORLD_SIZE, RANK

    mesh_names = ["ring", "ulysses"]
    mesh_dims = [ring_degree, ulysses_degree]
    mesh = dist.device_mesh.init_device_mesh("cuda", mesh_dims, mesh_dim_names=mesh_names)
    cp_options.ring_mesh = mesh["ring"]
    cp_options.ulysses_mesh = mesh["ulysses"]
    cp_options.convert_to_fp32 = True
    cp_attention = None
    num_warmups = 5
    num_repeats = 10
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    if ring_degree > 1 and ulysses_degree > 1:
        cp_options.mode = "unified"
        cp_attention = _templated_unified_attention
    elif ulysses_degree > 1:
        cp_options.mode = "ulysses"
        cp_attention = _templated_ulysses_attention
    else:
        cp_options.mode = "ring"
        cp_attention = _templated_ring_attention

    results = {}
    for op_name in ops:
        op = OPS[op_name]
        cp_options.op = op
        results[op_name] = {}

        for seq_len in seq_lens:
            shape = (batch_size, num_heads, seq_len, head_dim)
            query = torch.randn(shape, device=device, dtype=dtype)
            key = torch.randn(shape, device=device, dtype=dtype)
            value = torch.randn(shape, device=device, dtype=dtype)

            dist.broadcast(query, src=0)
            dist.broadcast(key, src=0)
            dist.broadcast(value, src=0)
            dist.barrier()
            torch.cuda.synchronize()

            reference_out, reference_lse = torch_cudnn_attention(query, key, value)
            query, key, value = (x.chunk(WORLD_SIZE, dim=2)[RANK].contiguous() for x in (query, key, value))

            for _ in range(num_warmups):
                if WORLD_SIZE == 1:
                    out, lse = op(query, key, value)
                else:
                    out, lse = cp_attention(query, key, value)
                out = funcol.all_gather_tensor(out, gather_dim=2, group=mesh._flatten().get_group())
                lse = funcol.all_gather_tensor(lse, gather_dim=2, group=mesh._flatten().get_group())
            torch.cuda.synchronize()

            diff = out - reference_out
            absdiff = torch.abs(diff)
            absmax = torch.max(absdiff)
            mae = torch.mean(absdiff)
            mse = torch.mean(diff * diff)
            if RANK == 0:
                print(f"op: {op_name}, seq_len: {seq_len}, absmax: {absmax:.5f}, mae: {mae:.5f}, mse: {mse:.5f}")

            # if not torch.allclose(out, reference_out, atol=1e-2, rtol=1e-2):
            #     raise ValueError(f"Output mismatch for op: {op_name}, seq_len: {seq_len}")
            # if not torch.allclose(lse, reference_lse, atol=1e-2, rtol=1e-2):
            #     raise ValueError(f"LSE mismatch for op: {op_name}, seq_len: {seq_len}")
            dist.barrier()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(num_repeats):
                if WORLD_SIZE == 1:
                    out, lse = op(query, key, value)
                else:
                    out, lse = cp_attention(query, key, value)
            end_event.record()
            torch.cuda.synchronize()
            dist.barrier()
            elapsed_time = start_event.elapsed_time(end_event) / num_repeats
            results[op_name][seq_len] = elapsed_time
    
    if RANK == 0:
        print("Benchmark results:")
        for op_name, seq_times in results.items():
            print(f"\n\n===== op: {op_name} =====")
            for seq_len, time in seq_times.items():
                print(f"  {seq_len=}, {time:.5f} ms")


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)

    try:
        dist.init_process_group(backend="nccl")
        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
        torch.cuda.set_device(RANK)

        if args.ring_degree * args.ulysses_degree != WORLD_SIZE:
            raise ValueError(
                f"ring_degree * ulysses_degree must equal world size, got {args.ring_degree} * {args.ulysses_degree} != {WORLD_SIZE}"
            )

        main(
            ring_degree=args.ring_degree,
            ulysses_degree=args.ulysses_degree,
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            seq_lens=args.seq_lens,
            ops=args.ops,
            seed=args.seed,
        )
    finally:
        dist.destroy_process_group()
```

</details>

<sub><sup><b>*</b></sup> Breaking down parallelism ideas by first implementing them sequentially is a very helpful way of building intuition about the various algorithms applied to massively parallel systems like GPUs. </sub>

### CUDA Streams

Due to this post being very long already, this section has been moved. Check out the first section of the next post to learn more about how to leverage CUDA streams!

## Benchmarks

In the previous post, we benchmarked against [xDiT](https://github.com/xdit-project/xDiT) and [ParaAttention](https://github.com/chengzeyi/ParaAttention). The same context mentioned there applies here too (so make sure to read that first).

For Ours and ParaAttention, we report only the time taken by the transformer, excluding the overhead from running text encoder and VAE (despite them being negligible, the added wall time is still significant from an overall deployment perspective). For xDiT, we use the [reported H100 timings](https://github.com/xdit-project/xDiT/blob/21dcdcf1fbf427d2b8a39f37110efadc38ef9ed1/docs/performance/flux.md) from their benchmarks directly instead of running our own tests because my personal benchmarks were consistently slower (i.e. more investigation is needed on my end to match environments).

<table>
<tr>
  <th> A100 </th>
  <th> H100 </th>
</tr>
<tr>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_post_2-a100.png" width="512px" /></td>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_post_2-h100.png" width="512px" /></td>
</tr>
</table>

<sup> It's important to note again that we're only benchmarking the time taken by the transformer, and not including the overhead from running text encoder and VAE (even if it's negligible, the added wall time is still significant from an overall deployment perspective). </sup>

The benchmarks show that our final implementation is significantly faster than the other implementations. This comes from the extreme hard work of other major libraries and frameworks. We merely connected a few dots standing on the shoulders of giants. A very heartfelt thank you to those shoulders! But, as always, there is more that can be done, and most importantly we must first talk about the tradeoffs...

In trying to achieve maximal performance, engineers and researchers spend a lot of time rewriting models and optimizing algorithms for better hardware utilization. The pareto principle applies here too: 80% of the performance can be achieved with 20% of the effort. A simple `torch.compile` and Flash Attention 3 bring us down to ~3.8 seconds. Everything else requires a lot of effort and time to implement, debug, and maintain. This highlights the importance of frameworks like xDiT and ParaAttention, which provide a good balance between performance and ease of use for most users.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/power-law-pytorch.png" width="512px" alt="Pareto Principle in action" />

<sup> Ah, the road to maximal speedup. Paved with developer tears. Image credits: ChatGPT  </sup>

We can now wrap our implementation into a neatly packaged [Gradio](https://www.gradio.app/) app and deploy on a 4xH100 machine! On my personal benchmarks, running requests end-to-end with all involved models (effectively calculating wall time per request), our response time is **~1.5** to **1.6** seconds (and lower if using prompt length bucketing)! This is faster than even the fastest production deployments for Flux, as per [Artificial Analysis](https://artificialanalysis.ai/text-to-image/model-family/flux), which take an average of **1.7-1.8** seconds per request.

This is a significant achievement! If you've read so far, you now have a good understanding of many performance optimizations that can be applied for scaling inference. Congratulations! 🎉 In the next post, we shall take the speedup even further!

### Cost Analysis

Assuming you run the optimized model (cudagraph + compile + ulysses=4) with a fully set up environment on H100 SXM and 2.75 TiB SSD, the table below shows:
- time taken to generate 1000 images
- how much it would cost to generate 1000 images on different providers
- images per hour
- images per $1000

Note:
- The timing reports below are for running the entire pipeline end-to-end and includes the time taken by the text encoders, denoiser and VAE decoder (i.e. not the same as benchmarks above which only report the transformer inference time).
- The cost analysis is based on the pricing of different cloud providers as of July 19th, 2025.
- The prices are for running the optimized inference on 4xH100 GPU with 2.75 TiB SSD storage.
- The reported numbers in "Time for 1000 images" for Runpod/Lambda/Modal is calculated as `100 * avg_of_5(time taken to generate 10 images in independent requests)`.
- For Replicate and Fal, we compare the cost of running their inference service and calculate the time based on reported numbers at [Artificial Analysis](https://artificialanalysis.ai/text-to-image/model-family/flux).

<table>
<tr>
  <th> Provider </th>
  <th> Pricing per hour </th>
  <th> Time for 1000 images (hours) </th>
  <th> Cost for 1000 images ($) </th>
  <th> Images per hour </th>
  <th> Images per $1000 </th>
</tr>
<tr>
  <td> Runpod </td>
  <td> $2.69 x4 (compute) + $0.19 (storage) </td>
  <td> 1.51 * 1000 / (60 * 60) = 0.419 </td>
  <td> $4.59 </td>
  <td> 2384 </td>
  <td> 217864 </td>
</tr>
<tr>
  <td> Lambda </td>
  <td> $3.09 x4 (compute + storage) </td>
  <td> 1.59 * 1000 / (60 * 60) = 0.442 </td>
  <td> $5.46 </td>
  <td> 2264 </td>
  <td> 183150 </td>
</tr>
<tr>
  <td> Fal </td>
  <td> - </td>
  <td> 1.778 * 1000 / (60 * 60) = 0.494 </td>
  <td> $0.025 (per 1024px image) * 1000 = $25  </td>
  <td> 2024 </td>
  <td> 40000 </td>
</tr>
<tr>
  <td> Replicate </td>
  <td> - </td>
  <td> 2.934 * 1000 / (60 * 60) = 0.815 </td>
  <td> $0.025 (per 1024px image) * 1000 = $25 </td>
  <td> 1227 </td>
  <td> 40000 </td>
</tr>
<tr>
  <td> Modal </td>
  <td> N/A </td>
  <td> N/A </td>
  <td> N/A </td>
  <td> N/A </td>
  <td> N/A </td>
</tr>
</table>

The overall number of images from our optimized implementation is lower than what we saw in post 1, but this is because of reduced GPU utilization. We should note that instead of processing `4608` tokens on a single GPU, we now process only `4608 / 4 = 1152` tokens per GPU using context parallelism. This makes the matrix multiplication sizes much lower and we are more memory and overhead bound. If we maximize for throughput instead of latency, the amount of generated images per dollar becomes much higher compared to our previous implementation in post 1, and ginormously higher compared to inference services!

Many acknowledgements and thanks to [Zeyi](https://github.com/chengzeyi) as he's the reason for my exploration in performance optimization of diffusion models. Many ideas in speeding them up were first popularized in his open-source work and they are great resources to learn from. Using this post, I would like to shoutout his company, [WaveSpeed](https://wavespeed.ai/), which provides faster inference and cheaper overall costs for running diffusion models compared to other inference services. If you're looking for a production-ready solution, I highly recommend checking them out!

## Additional reading

- [Online Softmax](https://arxiv.org/abs/1805.02867)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [Flash Attention 3](https://pytorch.org/blog/flashattention-3/)
- [The Log-Sum-Exp trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- [Large scale transformers with tensor parallel](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [Tensor parallelism in three levels of difficulty](https://www.determined.ai/blog/tp)
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
- [GPU MODE - Ring Attention](https://www.youtube.com/watch?v=ws7angQYIxI)
