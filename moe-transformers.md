---
title: "Mixture of Experts (MoEs) in Transformers"
thumbnail: /blog/assets/moe-transformers/thumbnail.png
authors:
- user: ariG23498
- user: pcuenq
- user: merve
- user: IlyasMoutawwakil
- user: ArthurZ
- user: sergiopaniego
- user: Molbap
---

# Mixture of Experts (MoEs) in Transformers

## Introduction

Over the past few years, scaling dense language models has driven most progress in LLMs. From early models like the original [ULMFiT](https://nlp.fast.ai/classification/2018/05/15/introducing-ulmfit.html) (~30M parameters) or GPT-2 (1.5B parameters, which at the time was considered "too dangerous to release" 🧌), and eventually to today’s hundred-billion–parameter systems, the recipe was simple:

> More data + more parameters gives better performance.

[Scaling laws](https://huggingface.co/papers/2001.08361) reinforced this trend, but dense scaling has practical limits:

- Training becomes increasingly expensive.
- Inference latency grows.
- Deployment requires significant memory and hardware.

This is where Mixture of Experts (MoEs) enter the picture.

> [!TIP]
> If you're already familiar with MoEs and want to jump straight into the engineering work done in transformers, you can head directly to [Transformers and MoEs](#transformers-and-moes).

## From Dense to Sparse: What Are MoEs?

A Mixture of Experts model keeps the Transformer backbone, but replaces certain dense feed-forward layers with a set of **experts**. An “expert” is not a topic-specialized module (e.g., "math expert", "code expert"). It is simply a learnable sub-network. For each token, a **router** selects a small subset of experts to process it.

| ![MoE routing diagram](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/moe_routing.png) |
| :--: |
| Figure 1: Expert 1 among 4 experts is activated (Source: [Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)) |

Different tokens activate different experts, based on their hidden representations.

> Model capacity depends on total parameters, but inference speed depends on active parameters.

This is the key idea.

For example, take [`gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b). It has 21B total parameters, but uses 4 active experts per token, out of a total of 32 experts. Considering the shared components plus the active experts, this model uses ~3.6B active parameters per token. Running this model on an M3 Ultra Mac, which has a memory bandwidth of about 800 GB, we could estimate generation speed as ~ `800 / (3.6 * 2)` in `bfloat16`, where each parameter takes 2 bytes. This yields about **111 tokens per second**. The actual performance number we get is ~115 tok/s, which is very close to the back-of-the-envelope calculation.

<video controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/gpt-oss-20-inference.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

This super fast speed confirms the model works approximately as a 3.6B parameter one, but it has the same capacity (or quality) as a 21B parameter model.

*(Note: speed would be even faster if we used kernels for the native mxfp4 quantization the model uses).*

MoEs are attractive for these reasons:

1. Better Compute Efficiency

    Given a fixed training FLOP budget, MoEs often outperform dense counterparts.

    | ![MoE vs Dense training graphs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/faster_training.png) |
    | :--: |
    | Figure 2: Dense vs. MoE training curves (Source: [OLMoE: Open Mixture-of-Experts Language Models](https://huggingface.co/papers/2409.02060)) |

    This means faster iteration and better scaling efficiency.

2. A Natural Parallelization Axis

    Experts provide a structural boundary in the computation graph. Since different tokens engage different experts, we can parallelize across experts (we discuss this later in [Expert Parallelism](#expert-parallelism)).

3. Industry Adoption

    Recent major MoE releases of open models that happened in the past few weeks include [Qwen 3.5](https://huggingface.co/collections/Qwen/qwen35), [MiniMax M2](https://huggingface.co/collections/MiniMaxAI/minimax-m2), [GLM-5](https://huggingface.co/collections/zai-org/glm-5), or [Kimi K2.5](https://huggingface.co/collections/moonshotai/kimi-k25).

    The trend accelerated after the success of [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) in January 2025, building on earlier systems like [DeepSeek V2](https://huggingface.co/deepseek-ai/DeepSeek-V2). Another early MoE was [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), released in December 2023.

    | ![2-year timeline of MoE model addition in the transformers package](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/moe_2y_timeline.png) |
    | :--: |
    | Figure 3: 2-year timeline of MoE model addition to the `transformers` library. DeepSeek R1 marks a clear inflection point. |

    Closed labs use MoEs too. ChatGPT has long been [*rumored*](https://x.com/soumithchintala/status/1671267150101721090) to use a sparse architecture, and the open [gpt-oss models](https://huggingface.co/collections/openai/gpt-oss) certainly do.

> [!TIP]
> If you want to learn more about MoEs in general, we strongly suggest reading [this blog](https://huggingface.co/blog/moe) and watching our recent [YouTube video on routing](https://youtu.be/CDnkFbW-uEQ).

## Transformers and MoEs

Most tooling in the ecosystem, including model loading, device placement, quantization, and backend execution was originally designed for **dense** models. MoEs challenge these assumptions.

Making MoEs **first-class citizens** in `transformers` means redesigning parts of the loading pipeline, execution model, and distributed abstractions, not just adding new model classes. We’ll focus on how the `transformers` library has evolved to support sparse architectures across:

* [Weight Loading Refactor](#weight-loading-refactor)
* [Expert Backend](#expert-backend)
* [Expert Parallelism](#expert-parallelism)
* [Training MoEs with transformers](#training-moes-with-transformers)

## Weight Loading Refactor

[`AutoModelForCausalLM.from_pretrained("model_id")`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) downloads and loads model weights into a PyTorch model. For dense models, loading is relatively straightforward where each tensor in the checkpoint maps one-to-one to a parameter in the runtime module.

For MoEs, it’s more complicated. In most MoE checkpoints, each expert is serialized independently. If you peek inside the [DeepSeek-V3 checkpoint index](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/model.safetensors.index.json), you’ll see keys like:

```bash
model.layers.3.mlp.experts.0.gate_proj.weight
...
model.layers.3.mlp.experts.255.gate_proj.weight
```

Each expert has its own set of weight matrices, essentially 256 (0 to 255 total, taking DeepSeek-V3 as an example) small feed-forward networks saved side by side. At runtime, however, GPUs execute optimized kernels. Modern MoE kernels such as [grouped GEMMs and fused MoE implementations](https://huggingface.co/kernels-community/megablocks) are designed to process *all experts in a single operation*, not by looping over them one at a time.

To do that efficiently, they require expert weights to be packed into a single **contiguous tensor**.

So we have a mismatch:

- **Checkpoint:** 256 separate tensors
- **Runtime:** 1 packed tensor

Bridging this gap systematically is what the [weight loading refactor](https://github.com/huggingface/transformers/pull/41580) enables.

With the introduction of a [generic WeightConverter](https://huggingface.co/docs/transformers/main/en/weightconverter), the mental model shifted from:

> A checkpoint already matches my runtime layout; loading is mostly a key-by-key copy.

to:

> A checkpoint is just a serialized source of tensors. Loading is a **conversion pipeline** that transforms them into the runtime layout we want.

### Dynamic Weight Loading with `WeightConverter`

The central abstraction introduced by this refactor is **dynamic weight loading** via a [`WeightConverter`](https://huggingface.co/docs/transformers/main/en/internal/weight_converter).

`WeightConverter` lets us define:

```
source key patterns → target key(s) + operations
```

Primitive operations (chunk, concatenate, etc.) are composable. Two that are particularly useful for MoEs:


- [`MergeModulelist`](https://github.com/huggingface/transformers/blob/main/src/transformers/core_model_loading.py) merges a list of tensors into a single tensor. For example, you can compose `MergeModulelist` with `Concatenate` to stack the experts in a MoE and pack them into one tensor.

    ```python
    WeightConverter(
        ["block_sparse_moe.experts.*.w1.weight", "block_sparse_moe.experts.*.w3.weight",],
        "mlp.experts.gate_up_proj",
        operations=[
            MergeModulelist(dim=0),
            Concatenate(dim=1),
        ],
    )
    ```


- [`SplitModulelist`](https://github.com/huggingface/transformers/blob/b71de73468429eb02da18caa50e9b5200400a4ed/src/transformers/core_model_loading.py#L208) splits a tensor back into a list of tensors. For example, you can split a stack of experts back into individual experts.

    ```python
    WeightConverter(
        "mlp.experts.down_proj",
        "block_sparse_moe.experts.*.w2.weight",
        operations=[SplitModulelist(dim=0)],
    )
    ```

### Lazy Materialization of Tensors

The refactor improves not just *what* conversions exist, but *how* they’re scheduled.

The loader scans checkpoint keys once, matches them against converter patterns, and groups tensors per converter. Once a key is identified as needed, it’s registered as a *future* and materialized via a thread pool. Conversion operations run only once their dependencies are ready. For example, `MergeModulelist` waits until all experts for a layer are loaded.

This avoids repeated scans and reduces memory peaks.

### Benchmark: Weight-Loading Pipeline Improvements

To evaluate the improvements introduced by the new weight-loading pipeline, we benchmarked the v4 vs v5 versions of `transformers`. The focus is on loading speed of large MoE models, which is often a bottleneck in training and inference.

We benchmarked v4 vs v5 using:

- v4 branch: https://github.com/ariG23498/transformers/tree/bench-v4
- v5 branch: https://github.com/ariG23498/transformers/tree/bench-v5

Example:

```python
from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen1.5-110B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_id)
```

Two relevant environment variables:

- `HF_ENABLE_PARALLEL_LOADING`: Enables parallel shard loading via threads.

- `HF_DEACTIVATE_ASYNC_LOAD`:Disables the new async pipeline (v5 escape hatch).

### Results

**Model:** `Qwen/Qwen1.5-110B-Chat`
**GPU:** 1× A100 (80GB)

| Version | Strategy | Loading Mode | Time |
|----------|----------|--------------|------|
| v4.57.6 | `device_map="auto"` | Threadpool | 66.24s |
| v4.57.6 | `device_map="auto"` | Sequential | 67.29s |
| v4.57.6 | TP | — | OOM |
| v5 | `device_map="auto"` | Async (default) | 20.71s |
| v5 | `device_map="auto"` | Sync | 45.3s |
| v5 | TP | Async | 10.1s |
| v5 | TP | Sync | 19.28s |

| ![Loading benchmarks](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/loading_benchmark.png) |
| :--: |
| Figure 4: Loading benchmarks (v4 vs v5) |

The speedup is not just “more threads.”

It’s the combination of **Single-pass routing**, **Async materialization**, and **Conversion-aware scheduling** which together avoid unnecessary materialization and memory peaks while enabling expert packing and projection fusion at load time.

### Where Quantization Fits In

With this refactor we can now create the runtime module structure first and then convert the weights into the structure. We can now optionally attach quantization within the conversion pipeline, making quantization part of the weight loading pipeline itself. This is crucial because quantizing “per expert” only makes sense once experts exist in a predictable packed layout.

This end to end pipeline was not possible earlier and now it comes to the users as an exposed API.

## Expert Backend

Once experts are packed into a single runtime tensor, another question arises:

> How do you actually route through them efficiently?

In a Mixture of Experts model, each token is routed to different experts. This means the runtime must dispatch tokens to their selected expert weights, execute the projections efficiently, apply the routing weights and then collect and reorder the results.

This is what the [Experts Backend system](https://huggingface.co/docs/transformers/experts_interface) (introduced in [PR #42697](https://github.com/huggingface/transformers/pull/42697)) addresses. The Experts Backend introduces a **pluggable execution architecture** that decouples expert computation from the model implementation. Instead of hardcoding one dispatch strategy inside each MoE model, the system allows expert layers to dynamically select a backend at runtime.

This is implemented via a decorator pattern:

```python
@use_experts_implementation
```

The decorator wraps expert classes and dispatches computation to the selected backend automatically.

Three backends are currently provided:

1. `eager` which loops over the selected experts and applies projections per expert. This is used for correctness reference and debugging.

2. `batched_mm` uses the [`torch.bmm`](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html) API. This duplicate selected expert weights per token and performs a single batched GEMM. This backend is very well suited for small batch, GPU-heavy workloads where memory is available.

3. `grouped_mm` uses [`torch._grouped_mm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html) API. Here we sort tokens by expert ID, group them, and then perform a single grouped GEMM. This backend shines with large batches or memory-constrained setups.

| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/expert_backend.png) |
| :--: |
| Figure: Expert backend illustration |


## Expert Parallelism

Mixture of Experts (MoE) models can have hundreds of billions of parameters (far more than what fits on a single GPU). Expert parallelism (EP) addresses this by distributing experts across multiple devices. Each device loads only its assigned subset of experts, computes for those experts and then participates in result aggregation. This approach scales models to far larger parameter counts without increasing computation cost because each token activates only a few experts.

Expert parallelism is enabled via `enable_expert_parallel`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed.configuration_utils import DistributedConfig

distributed_config = DistributedConfig(enable_expert_parallel=True)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-120b",
    dtype="auto",
    distributed_config=distributed_config,
)
```

Launch with:

```bash
torchrun --nproc-per-node N script.py
```

Where `N` evenly divides the total number of experts, and possibly matches the number of GPUs in your node.

When `enable_expert_parallel=True`, the model switches from the standard tensor-parallel (TP) plan to an expert-parallel (EP) plan with specialized sharding strategies.

Core components of EP lie in:

1. [`GroupedGemmParallel`](https://github.com/huggingface/transformers/blob/b71de73468429eb02da18caa50e9b5200400a4ed/src/transformers/integrations/tensor_parallel.py#L934): This splits the expert weights along the expert dimension (`dim=0`). Here each device loads only `num_experts / num_devices`. 
    
2. [`RouterParallel`](https://github.com/huggingface/transformers/blob/b71de73468429eb02da18caa50e9b5200400a4ed/src/transformers/integrations/tensor_parallel.py#L977): This remaps global expert indices to local indices, masks out experts not assigned to the current rank, ensures each device computes only with its local experts and uses an all-reduce to combine partial outputs across devices.

## Training MoEs with Transformers

MoEs are excellent for scaling inference, but training them is significantly more complex.

MoEs have a Massive parameter count, the distributed expert communication is complicated, there are routing in-stabilities that need to be handled. To address this, we collaborated with **Unsloth** to enable significantly faster Mixture-of-Experts training:

- ~12× faster MoE training
- \>35% VRAM reduction
- ~6× longer context
- 12–30× overall speedup compared to v4

We leverage the Expert Backend abstraction, standardize around PyTorch’s `torch._grouped_mm` API and use custom Triton grouped-GEMM + LoRA kernels. Unsloth builds on top of the Transformers (and TRL) optimizations to push performance further.

> [!TIP]
> For full details, we recommend reading: [Unsloth’s official guide](https://unsloth.ai/docs/new/faster-moe)

## Conclusion

As sparse architectures continue to evolve, we want the transformers library to evolve with them. If you’re building with MoEs or experimenting with new sparse ideas, we’d love to hear from you. Let us know what abstractions, kernels, or workflows you’d like to see next in `transformers`.
