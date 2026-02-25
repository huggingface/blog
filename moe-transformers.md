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

Over the past few years, scaling dense language models has driven most progress in LLMs. From early models like [ULMFiT](https://nlp.fast.ai/classification/2018/05/15/introducing-ulmfit.html) (~30M parameters), GPT-2 (1.5B parameters, which at the time was considered "too dangerous to release" 🧌), and eventually to today’s hundred-billion–parameter systems, the recipe was simple:

> More data + more parameters → better performance.

[Scaling laws](https://huggingface.co/papers/2001.08361) reinforced this trend. But dense scaling has practical limits:

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

> **Model capacity depends on total parameters, but inference speed depends on active parameters.**

This is the key idea.

For example, take [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b). It has 21B total parameters, but uses 4 active experts per token, out of a total of 32 experts. Considering the shared components plus the active experts, this model uses ~3.6B active parameters per token. Running this model on an M3 Ultra Mac, which has a memory bandwidth of about 800 GB, we could estimate generation speed as ~ `800 / (3.6 * 2)` in `bfloat16`, where each parameter takes 2 bytes. This yields about 111 tokens per second. The actual performance number we get is ~115 tok/s, which is very close to the back-of-the-envelope calculation.

This super fast speed confirms the model works approximately as a 3.6B parameter one, but it has the same capacity (or quality) as a 21B parameter model.

(Note: speed would be even faster if we used kernels for the native mxfp4 quantization the model uses).

MoEs are attractive for these reasons:

1. Better Compute Efficiency

    Given a fixed training FLOP budget, MoEs often outperform dense counterparts.

    | ![MoE vs Dense training graphs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/faster_training.png) |
    | :--: |
    | Figure 2: Dense vs. MoE training curves (Source: [OLMoE: Open Mixture-of-Experts Language Models](https://huggingface.co/papers/2409.02060)) |

    This means faster iteration and better scaling efficiency.

2. A Natural Parallelization Axis

    Experts provide a structural boundary in the computation graph. Since different tokens engage different experts, systems can parallelize across experts (we discuss this later in [Expert Parallelism](#expert-parallelism)).

3. Industry Adoption

    Sparse architectures are no longer experimental.

    Recent major MoE releases include:

    - [Qwen](https://huggingface.co/collections/Qwen/qwen35)
    - [MiniMax](https://huggingface.co/collections/MiniMaxAI/minimax-m2)
    - [Z.ai](https://huggingface.co/collections/zai-org/glm-5)
    - [Moonshot AI](https://huggingface.co/collections/moonshotai/kimi-k25)

    The trend accelerated after the success of [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) in January 2025, building on earlier systems like:

    - [DeepSeek V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)
    - [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

    | ![2-year timeline of MoE model addition in the transformers package](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/moe_2y_timeline.png) |
    | :--: |
    | Figure 3: 2-year timeline of MoE model addition to the `transformers` library. DeepSeek R1 marks a clear inflection point. |

    Closed labs use MoEs too — ChatGPT has long been [*rumored*](https://x.com/soumithchintala/status/1671267150101721090) to use a sparse architecture, and the open [gpt-oss models](https://huggingface.co/collections/openai/gpt-oss) certainly do.

MoEs are conceptually elegant, but systemically demanding.

> [!TIP]
> If you want to learn more about MoEs in general, we strongly suggest reading [this blog](https://huggingface.co/blog/moe) and watching our recent [YouTube video on routing](https://youtu.be/CDnkFbW-uEQ).

## Transformers and MoEs

Most tooling in the ecosystem, including model loading, device placement, quantization, and backend execution was originally designed for **dense** models. MoEs challenge these assumptions.

Making MoEs **first-class citizens** in `transformers` means redesigning parts of the loading pipeline, execution model, and distributed abstractions not just adding new model classes. In the rest of this post, we’ll focus on how the `transformers` library has evolved to support sparse architectures across:

* [Weight Loading Refactor](#weight-loading-refactor)
* [The loading pipeline under the hood](#the-loading-pipeline-under-the-hood)
* [Why this is a MoE feature, not just a loader feature](#why-this-is-a-moe-feature,-not-just-a-loader-feature)
* [Where quantization fits in](#where-quantization-fits-in)
* [Benchmark](#benchmark)
* [Expert Backend](#expert-backend)
* [Expert Parallelism](#expert-parallelism)
* [Training MoEs with transformers](#training-moes-with-transformers)

## Weight Loading Refactor

[`AutoModelForCausalLM.from_pretrained("model_id")`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) downloads and loads model weights into a PyTorch model. For dense models, loading is relatively straightforward where each tensor in the checkpoint maps one-to-one to a parameter in the runtime module.

For MoEs, it’s more complicated. In most MoE checkpoints, each expert is serialized independently. If you peek inside the [DeepSeek-V3 checkpoint index](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/model.safetensors.index.json), you’ll see keys like:

```text
model.layers.3.mlp.experts.0.gate_proj.weight
...
model.layers.3.mlp.experts.255.gate_proj.weight
```

Each expert has its own set of weight matrices, essentially 256 (0 to 255 total experts) small feed-forward networks saved side by side.

At runtime, however, GPUs execute optimized kernels. Modern MoE kernels such as [grouped GEMMs and fused MoE implementations](https://huggingface.co/kernels-community/megablocks) are designed to process *all experts in a single operation*, not by looping over them one at a time.

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

- [`MergeModulelist`](https://github.com/huggingface/transformers/blob/main/src/transformers/core_model_loading.py):

    Stacks per-expert tensors into a single packed tensor:

    - **Checkpoint:**  
    `experts.0.W`, `experts.1.W`, …  

    - **Runtime:**  
    `experts.W_packed` with shape `[n_experts, …]`

    This makes expert packing a **first-class operation** at load time.

- [`SplitModulelist`](https://github.com/huggingface/transformers/blob/b71de73468429eb02da18caa50e9b5200400a4ed/src/transformers/core_model_loading.py#L208):
    
    Performs the reverse operation when needed.

    These utilities live in `transformers.core_model_loading` and are the workhorses behind efficient MoE loading.

### The Loading Pipeline Under the Hood

The refactor improves not just *what* conversions exist, but *how* they’re scheduled.

The new loading strategy:

1. **Single-pass key discovery + routing**  
   The loader scans checkpoint keys once, matches them against converter patterns, and groups tensors per converter.

2. **Asynchronous materialization**  
   Once a key is identified as needed, it’s registered as a future and materialized via a thread pool.

3. **Conversion-aware scheduling**  
   Conversion ops run only once their dependencies are ready.  
   For example, `MergeModulelist` waits until all experts for a layer are loaded.

This avoids repeated scans and reduces memory peaks.

### Expert Packing

Load-time transformation:

- Checkpoint: `experts.0.W`, `experts.1.W`, …  
- Runtime: `experts.W_packed` (shape `[n_experts, …]`)

### Fusing `gate` + `up` into `gate_up_proj`

Some runtimes require fused projections.

Dynamic loading supports this by composing:

- `MergeModulelist(dim=0)`  
- `Concatenate(dim=…)`

This ensures:

- Tensor-parallel sharding sees canonical shapes  
- Quantization operates on final runtime tensors  
- Kernel dispatch uses the correct layout

## Where Quantization Fits In

MoEs amplify the importance of quantization.

The refactor establishes a clean contract:

1. Create the runtime module structure first (including quantized modules).
2. Convert weights into that structure.
3. Optionally attach quantization within the conversion pipeline.

This is crucial because quantizing “per expert” only makes sense once experts exist in a predictable packed layout.

## Benchmark

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

- **`HF_ENABLE_PARALLEL_LOADING`**  
  Enables parallel shard loading via threads.

- **`HF_DEACTIVATE_ASYNC_LOAD`**  
  Disables the new async pipeline (v5 escape hatch).

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

It’s the combination of:

1. **Single-pass routing**
2. **Async materialization**
3. **Conversion-aware scheduling**

which together avoid unnecessary materialization and memory peaks while enabling expert packing and projection fusion at load time.

## Expert Backend

Once experts are packed into a single runtime tensor, another question arises:

> How do you actually compute through them efficiently?

In a dense model, every token flows through the same weights.  
In a Mixture of Experts model, each token is routed to different experts. This means the runtime must:

1. Dispatch tokens to their selected expert weights  
2. Execute the projections efficiently  
3. Apply routing weights  
4. Collect and reorder the results  

The optimal strategy depends heavily on:

- Batch size  
- Hardware
- Compilation mode  
- Memory constraints  

This is what the [Experts Backend system](https://huggingface.co/docs/transformers/experts_interface) (introduced in [PR #42697](https://github.com/huggingface/transformers/pull/42697)) addresses.

The Experts Backend introduces a ***pluggable execution architecture*** that decouples expert computation from the model implementation. Instead of hardcoding one dispatch strategy inside each MoE model, the system allows expert layers to dynamically select a backend at runtime.

This is implemented via a decorator pattern:

```python
@use_experts_implementation
```

The decorator wraps expert classes and dispatches computation to the selected backend automatically.

This design enables:

- Swapping execution strategies without changing model code  
- Adapting to hardware and batch size dynamically  
- Integrating new kernel optimizations transparently  

In short: **expert computation becomes a backend concern, not a model concern.**

Three backends are currently provided:

1. `eager`

    Reference implementation.

    - Loops over selected experts
    - Applies projections per expert
    - No compilation required
    - Reasonable GPU baseline
    - On CPU: slower than `grouped_mm`, faster than `batched_mm`

    Best for: correctness reference and debugging.

2. `batched_mm`

    Uses [`torch.bmm`](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html).

    Strategy:

    - Duplicate selected expert weights per token
    - Perform a single batched GEMM

    Characteristics:

    - Fastest for small inputs on GPU
    - Especially effective with compilation
    - Higher memory usage (due to parameter duplication)
    - Not recommended on CPU

    Best for: small batch, GPU-heavy workloads where memory is available.

3. `grouped_mm`

    Uses [`torch._grouped_mm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html).

    Strategy:

    - Sort tokens by expert ID
    - Group them
    - Perform a single grouped GEMM

    Characteristics:

    - Best for larger inputs on GPU
    - Most memory-efficient (no weight duplication)
    - Most efficient backend on CPU across input sizes

    Best for: large batches or memory-constrained setups.

| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/expert_backend.png) |
| :--: |
| Figure: Expert backend comparison |


## Expert Parallelism

Mixture of Experts (MoE) models can have hundreds of billions of parameters (far more than what fits on a single GPU). Expert parallelism (EP) addresses this by distributing experts across multiple devices.

Each device:

- Loads only its assigned subset of experts  
- Computes only for those experts  
- Participates in result aggregation  

Since each token activates only a few experts, this enables scaling to massive model sizes without increasing per-device memory or computation.

Expert parallelism is enabled via `enable_expert_parallel`:

```python
import torch
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig

distributed_config = DistributedConfig(enable_expert_parallel=True)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    dtype="auto",
    distributed_config=distributed_config,
)
```

When `enable_expert_parallel=True`, the model switches from the standard tensor-parallel (TP) plan to an expert-parallel (EP) plan with specialized sharding strategies.

### Core Components

1. [`GroupedGemmParallel`](https://github.com/huggingface/transformers/blob/b71de73468429eb02da18caa50e9b5200400a4ed/src/transformers/integrations/tensor_parallel.py#L934)

    - Splits expert weights along the expert dimension (`dim=0`)
    - Each device loads only `num_experts / num_devices`
    - Requires the number of devices to evenly divide the total number of experts

2. [`RouterParallel`](https://github.com/huggingface/transformers/blob/b71de73468429eb02da18caa50e9b5200400a4ed/src/transformers/integrations/tensor_parallel.py#L977)

    - Remaps global expert indices to local indices
    - Masks out experts not assigned to the current rank
    - Ensures each device computes only with its local experts
    - Uses an all-reduce to combine partial outputs across devices

#### EP Plan

Each model defines a model-specific `base_model_ep_plan` in its configuration.

This maps MoE components to parallel strategies, for example:

- Expert weights: `grouped_gemm`
- Router: `ep_router`

When EP is enabled, the system automatically applies the EP plan instead of the standard TP plan.

Each expert layer follows the pattern:

- Router uses `ep_router` for token routing  
- Expert weights use `grouped_gemm` for sharded computation  

Launch with:

```bash
torchrun --nproc-per-node N
```

Where `N` evenly divides the total number of experts.

## Training MoEs with Transformers

MoEs are excellent for scaling inference — but training them is significantly more complex.

The challenges include:

- Massive parameter counts  
- Distributed expert communication  
- Routing stability  
- Memory pressure  

To address this, we collaborated with **Unsloth** to enable significantly faster Mixture-of-Experts training:

- ~12× faster MoE training  
- \>35% VRAM reduction  
- ~6× longer context  
- 12–30× overall speedup compared to v4  

We leverage:

- The Expert Backend abstraction  
- Standardization around PyTorch’s `torch._grouped_mm`  
- Custom Triton grouped-GEMM + LoRA kernels  

Unsloth builds on top of the Transformers (and TRL) optimizations to push performance further.

> [!TIP]
> For full details, we recommend reading: [Unsloth’s official guide](https://unsloth.ai/docs/new/faster-moe)

## Conclusion

As sparse architectures continue to evolve, we want the transformers library to evolve with them. If you’re building with MoEs or experimenting with new sparse ideas, we’d love to hear from you. Let us know what abstractions, kernels, or workflows you’d like to see next in `transformers`.