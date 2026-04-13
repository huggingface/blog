---
title: "Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries"
thumbnail: /blog/assets/async-rl-landscape/thumbnail.png
authors:
  - user: aminediroHF
  - user: qgallouedec
  - user: kashif
  - user: lewtun
  - user: edbeeching
  - user: albertvillanova
  - user: nouamanetazi
  - user: lvwerra
  - user: sergiopaniego
---

# Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries

> [!NOTE] 
> **TL;DR** -- For those of you who don't have time to read 5,000 words about async RL plumbing (we get it, you have models to train):
>
> - **The problem:** In synchronous RL (reinforcement learning) training, data generation (model inference to create data samples) dominates wall-clock time -- a single batch of 32K-token rollouts on a 32B (32-billion parameter) model can take _hours,_ while the GPUs used for training remain idle.
> - **The solution everyone converged on:** Disaggregate (separate) inference and training onto different GPU pools, connect them with a rollout buffer (temporary storage for model outputs), and transfer weights asynchronously (without waiting), so neither side waits for the other.
> - **We surveyed 16 open-source libraries** that implement this pattern and compared them across 7 axes: orchestration primitives, buffer design, weight sync protocols, staleness management, partial rollout handling, LoRA support, and distributed training backends.
> - **Key findings:** Ray dominates orchestration (8/16 surveyed distributed computing libraries). The NCCL (NVIDIA Collective Communications Library) broadcast is the default method for transferring model weights. Staleness management refers to how outdated data samples are handled, ranging from simply dropping old samples to using advanced importance-sampling correction. LoRA (Low-Rank Adaptation) training is sparsely supported. Distributed MoE (Mixture of Experts) support is the emerging differentiator.
>
> If you'd rather skip straight to the good part, [here's the full comparison table](#4-global-overview-sixteen-libraries-at-a-glance) (no reading required, we won't judge).
>
> But seriously, if you stick around, you might learn a thing or two about why your GPUs are idle 60% of the time.

---

<details>
<summary><b>Click to expand Table of Contents</b></summary>

- [1. Motivation: From synchronous RL training to async architectures](#1-motivation-from-synchronous-rl-training-to-async-architectures)
  - [1.1 How TRL Does RL Training Today](#11-how-trl-does-rl-training-today)
  - [1.2 Colocated vs. Disaggregated Training](#12-colocated-vs-disaggregated-training)
  - [1.3 The Generation Bottleneck](#13-the-generation-bottleneck)
  - [1.4 The Core Insight](#14-the-core-insight)
- [2. Libraries Surveyed](#2-libraries-surveyed)
- [3. The Comparison Framework: Seven Axes](#3-the-comparison-framework-seven-axes)
  - [Axis 1: Orchestration & Concurrency Primitive](#axis-1-orchestration--concurrency-primitive)
  - [Axis 2: Rollout Buffer Design](#axis-2-rollout-buffer-design)
  - [Axis 3: Weight Synchronisation Protocol](#axis-3-weight-synchronisation-protocol)
  - [Axis 4: Staleness Management](#axis-4-staleness-management)
  - [Axis 5: Partial Rollout Handling](#axis-5-partial-rollout-handling)
  - [Axis 6: LoRA Training Support](#axis-6-lora-training-support)
  - [Axis 7: Distributed Training Backend & Parallelism](#axis-7-distributed-training-backend--parallelism)
- [4. Global Overview: Sixteen Libraries at a Glance](#4-global-overview-sixteen-libraries-at-a-glance)
- [5. The Next Wave: Design Implications](#5-the-next-wave-design-implications)
  - [5.1 Critic-Free Algorithms: Memory Freed, But Weight Sync Pressure Increases](#51-critic-free-algorithms-memory-freed-but-weight-sync-pressure-increases)
  - [5.2 Process Rewards: A New Synchronisation Barrier](#52-process-rewards-a-new-synchronisation-barrier)
  - [5.3 Multi-Agent Co-Evolution: The Straggler Problem Compounds](#53-multi-agent-co-evolution-the-straggler-problem-compounds)
  - [5.4 Training-Inference Mismatch: The Deepseek v3.2 MoE Case Study](#54-training-inference-mismatch-the-deepseek-v32-moe-case-study)
  - [5.5 Distillation: The Same Async Problem Under a Different Name](#55-distillation-the-same-async-problem-under-a-different-name)
- [6. Design Choices for TRL's Async Trainer](#6-design-choices-for-trls-async-trainer)
  - [Design Principle: Keep Orchestration Lightweight](#design-principle-keep-orchestration-lightweight)
  - [1. Bounded Queue with Per-Token `model_version` (No Double-Buffering)](#1-bounded-queue-with-per-token-model_version-no-double-buffering)
  - [2. NCCL Weight Sync with Packed Transfers](#2-nccl-weight-sync-with-packed-transfers)
  - [3. Partial Rollout Support for Agentic Workloads](#3-partial-rollout-support-for-agentic-workloads)
  </details>

## 1. Motivation: From synchronous RL training to async architectures

Async RL training has emerged as the dominant paradigm for post-training at scale. Several trends in modern post-training have made synchronous training loops nearly impossible to scale:

- **Long rollouts from reasoning models.** Chain-of-thought training produces very long rollouts, and a single synchronous generation batch can take hours to complete on a single GPU. During all of that time, training GPUs sit completely idle.
- **Value-function-free trainers like GRPO** use group-relative advantages. This means generating up to G times more rollouts per prompt, and the entire batch is gated by the slowest completion in the group.
- **The rise of agentic RL training.** When models interact with tools, sandboxes, and external environments across multi-turn trajectories, rollout lengths and latencies become highly variable. A simple API call might return in seconds, while a complex reasoning chain with tool use can run for minutes or hours. MiniMax's [Forge](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm) framework, used to train MiniMax-M2.5, illustrates the scale this reaches in practice: context lengths up to 200K tokens, over a hundred thousand distinct agent scaffolds and environments, and daily throughput on the order of millions of samples. At this scale, any synchronous barrier between generation and training becomes a severe bottleneck. The straggler problem alone (where a handful of slow rollouts block an entire batch) can idle hundreds of GPUs.

The open-source ecosystem has converged on a common architectural response: disaggregate inference from training onto separate GPU pools, connect them with a rollout buffer, and let both sides run concurrently.

We are developing a new async trainer for [TRL](https://github.com/huggingface/trl), one of the most widely used libraries for model post-training. To guide our design, we surveyed **sixteen open-source libraries** that were built from the ground up around asynchronous training and compared them across **seven axes**: orchestration primitives, buffer design, weight sync protocols, staleness management, partial rollout handling, LoRA support, and distributed training backends. This article distills the design principles we extracted from that survey.

Beyond RL, the need for async infrastructure is increasingly evident. For example, **on-policy distillation**, where a student generates sequences and a teacher scores them, mirrors GRPO but swaps the reward function for a teacher forward pass. Recognizing this structural similarity, everything in this survey applies equally to async distillation. We'll return to this broader point in Section 5.

### 1.1 How TRL Does RL Training Today

TRL's current `GRPOTrainer` implements the full GRPO loop (prompt sampling, generation, reward scoring, advantage computation, gradient update, and weight sync) in a single synchronous `training_step()` call. This design is simple and correct, but it cannot overlap generation with training, leaving significant GPU utilisation on the table.

Looking at the `GRPOTrainer`, we have the following phases sequentially within each training step:

1. **Prompt sampling:** draw a batch of prompts from the dataset. Nothing crazy here, let's continue.
2. **Generation**, calls `model.generate()` (or forward requests to a vLLM server) to produce G completions per prompt. This is autoregressive and dominates wall-clock time.
3. **Reward scoring:** evaluate each completion against one or many reward functions.
4. **Advantage computation**
5. **Forward and backward passes:** compute the clipped policy gradient loss and backpropagate.
6. **Optimizer step**, update model weights.
7. **Weight sync**, push updated weights to the inference engine (vLLM) so the next generation uses the new policy.

Each phase **blocks** until completion before the next begins. The timeline looks like this:

<figure class="image text-center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/69a5684704dd904ad8fdf1c6/TxWJ3tSZOQ0uRVwG_lldN.png" alt="Synchronous TRL training timeline">
</figure>

TRL offers the `steps_per_generation` config option to reuse a single set of rollouts across multiple gradient steps (temporal reuse), amortizing the generation cost. But the generation call itself remains fully synchronous and blocking; the trainer cannot begin gradient computation until every completion in the batch has finished.

The library also supports running vLLM in `server` mode as a separate process. It frees the training GPU during generation, but two hard synchronisation barriers remain: the **HTTP calls until all completions return**, and the weight sync blocks both the trainer and vLLM during the transfer.

### 1.2 Colocated vs. Disaggregated Training

Before discussing async training, it is essential to understand the two deployment topologies for RL training with a separate inference engine:

- **Colocated mode** places inference and training on the **same set of GPUs**. A single GPU (or TP group) holds both the training model (under FSDP or ZeRO) and the inference engine (vLLM or SGLang). Only one role is active at a time: during generation, the training model's parameters may be offloaded or resharded into an inference-friendly layout (e.g., from FSDP shards to vLLM's tensor-parallel layout); during training, the inference engine is paused or put to sleep. Weight "sync" is essentially free; it is at most an in-place resharding on the same GPU, not a network transfer. The advantage of the colocated mode is simplicity and cost; you need fewer total GPUs. The fundamental limitation is that **inference and training cannot overlap**. For example, here is the Trl with vllm in `colocate_mode`:

<figure class="image text-center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/69a5684704dd904ad8fdf1c6/sbOsV__Ji122HdJs0JxbV.png" alt="TRL with vLLM in colocate mode">
</figure>

- **Disaggregated mode** places inference and training on **separate GPU pools**. The inference pool runs vLLM or SGLang continuously; the training pool runs the optimizer continuously. The two pools communicate via a weight synchronisation protocol (NCCL broadcast, filesystem checkpoint, HTTP, etc.) and a data transfer mechanism (Ray object store, Redis streams, shared memory, etc.)

The biggest advantage of the disaggregated mode is that **inference and training can run concurrently**. While the trainer computes gradients on batch N, the inference pool is already generating rollouts for batch N+K, enabling async training. However, this benefit comes at a cost: additional GPUs are required.

Concurrency, asynchronicity, and parallelism are distinct concepts that often get conflated. In this article, when we say "**async training,**" we mean something specific: **generation and training running in parallel, with effective overlap**; the inference pool is producing the next batch of rollouts while the training pool is computing gradients on the current batch. This is fundamentally a disaggregated-mode capability. Colocated mode can benefit from optimisations like sleep/wake memory management or fast in-place resharding to speed up inference, but it cannot achieve true simultaneous overlap; inference and training still take turns on the same GPUs. Every library in this survey that implements meaningful async overlap uses disaggregated mode as the foundation.

### 1.3 The Generation Bottleneck

In RL training for reasoning models, **autoregressive generation dominates wall-clock time**. A single rollout for a math or coding task can produce 8K–64K tokens of chain-of-thought reasoning (see [QED-Nano rollout lengths](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost#outcome-reward-rl-with-long-response-lengths)).

To ground this concretely, consider [vLLM benchmarks on a single H100 80GB GPU](https://www.databasemart.com/blog/vllm-gpu-benchmark-h100) (bf16, no quantisation, offline throughput mode). A **7B model** (DeepSeek-R1-Distill-Qwen-7B) achieves ~6,300 output tokens/s aggregate throughput; a **32B model** (DeepSeek-R1-Distill-Qwen-32B) drops to ~1,200 output tokens/s. These are _total_ throughput across all concurrent requests, the number the inference engine can push through per second, regardless of how many sequences share the GPU.

Now consider a typical GRPO training step: **G=8 completions per prompt × 64 prompts/batch = 512 rollouts**. How long does generation take?

| Output length per rollout | Total output tokens (512 rollouts) | Time on 1×H100 (7B @ ~6K tok/s) | Time on 1×H100 (32B @ ~1.2K tok/s) |
| :------------------------ | :--------------------------------- | :------------------------------ | :--------------------------------- |
| 2K tokens (short CoT)     | ~1M tokens                         | **~3 min**                      | **~14 min**                        |
| 8K tokens (medium CoT)    | ~4M tokens                         | **~11 min**                     | **~56 min**                        |
| 32K tokens (long CoT)     | ~16M tokens                        | **~45 min**                     | **~3.7 hours**                     |

Even at the short end (2K tokens generated with a 7B model), generation alone consumes several minutes per training step. At the long end, where frontier reasoning models increasingly operate, a single generation phase can take _hours_ on one GPU. Scaling to 8 inference GPUs divides these times by roughly 8× (assuming linear throughput scaling), but even then, 32K-token rollouts on a 32B model still take ~28 minutes per step.

The **straggler problem** compounds this further. In group-based algorithms like GRPO, you sample G completions per prompt. The batch cannot proceed until the _slowest_ completion finishes. Chain-of-thought output lengths are highly variable; a single prompt might produce completions ranging from 1K to 32K tokens. The batch is gated by the longest completion, and continuous batching only partially mitigates this: shorter sequences free up slots for new work, but the _last_ sequence in a GRPO group still blocks the group's reward computation and training step.

### 1.4 The Core Insight

Every library in this survey has independently converged on the same architectural principle: **physically separate inference GPUs from training GPUs, and push weights asynchronously**, so generation never stops and training never waits.

The inference pool runs continuously, feeding completed rollouts into a buffer. The training pool pulls from the buffer, computes gradient updates, and periodically pushes new weights back to the inference pool to keep it in sync. The two loops run at their own pace, decoupled by the buffer.

This setup is highly scalable, but it introduces a new class of problems: staleness (rollouts generated under an old policy), weight synchronisation overhead, partial rollout handling, etc. The rest of this article dissects in detail how current open-source libraries address these issues.

---

## 2. Libraries Surveyed

| Library           | Organisation          | Repo                                                                                     | GitHub ⭐ (Mar. '26) |
| ----------------- | --------------------- | ---------------------------------------------------------------------------------------- | -------------------: |
| **AReaL**         | inclusionAI/Ant Group | [github.com/inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)                     |                4,338 |
| **ART**           | CoreWeave             | [github.com/OpenPipe/ART](https://github.com/OpenPipe/ART)                               |                8,952 |
| **Atropos**       | NousResearch          | [github.com/NousResearch/atropos](https://github.com/NousResearch/atropos)               |                  878 |
| **MILES**         | radixark              | [github.com/radixark/miles](https://github.com/radixark/miles)                           |                  950 |
| **NeMo-RL**       | NVIDIA                | [github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)                           |                1,383 |
| **OAT**           | SAIL-SG               | [github.com/sail-sg/oat](https://github.com/sail-sg/oat)                                 |                  637 |
| **open-instruct** | AI2 (AllenAI)         | [github.com/allenai/open-instruct](https://github.com/allenai/open-instruct)             |                3,611 |
| **PipelineRL**    | ServiceNow            | [github.com/ServiceNow/PipelineRL](https://github.com/ServiceNow/PipelineRL)             |                  374 |
| **PRIME-RL**      | PrimeIntellect        | [github.com/PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)   |                1,114 |
| **ROLL**          | Alibaba               | [github.com/alibaba/ROLL](https://github.com/alibaba/ROLL)                               |                2,921 |
| **SkyRL**         | NovaSky-AI            | [github.com/NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)                       |                1,664 |
| **SLIME**         | THUDM                 | [github.com/THUDM/slime](https://github.com/THUDM/slime)                                 |                4,595 |
| **TorchForge**    | Meta                  | [github.com/meta-pytorch/torchforge](https://github.com/meta-pytorch/torchforge)         |                  632 |
| **Tunix**         | Google                | [github.com/google/tunix](https://github.com/google/tunix)                               |                2,175 |
| **verl**          | ByteDance             | [github.com/verl-project/verl](https://github.com/verl-project/verl)                     |               19,673 |
| **verifiers-rl**  | PrimeIntellect        | [github.com/PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) |                3,876 |

---

## 3. The Comparison Framework: Seven Axes

To make sense of the rapidly expanding ecosystem of async RL libraries, we propose seven orthogonal axes of comparison. Each axis captures a fundamental design decision that shapes the library's performance, complexity, and trade-offs.

- **Axis 1 – Orchestration & Concurrency Primitive:** how distributed components are coordinated (Ray actors, asyncio, pub/sub, HTTP).
- **Axis 2 – Rollout Buffer Design:** how rollouts flow from inference to training.
- **Axis 3 – Weight Synchronisation Protocol:** how updated weights reach inference servers, and whether the system must pause to accept them or continue generating.
- **Axis 4 – Staleness Management:** how off-policy rollouts are handled: version rejection, depth bounding, or importance-sampling correction.
- **Axis 5 – Partial Rollout Handling:** what happens to in-flight generations when a weight update arrives mid-sequence.
- **Axis 6 – LoRA Training Support:** General LoRA support and whether adapter-only parameters can be trained and synced, enabling sub-millisecond weight transfers.
- **Axis 7 – Distributed Training Backend & Parallelism:** what parallelism strategy is used for training, constraining max model size.

### Axis 1: Orchestration & Concurrency Primitive

_How does the system coordinate its distributed components?_

The choice of orchestration framework determines the programming model, failure semantics, and scalability ceiling. Rather than listing per-library implementation details, the landscape decomposes cleanly into four **orchestration types**, fundamental coordination paradigms that differ in abstraction level, failure model, and deployment requirements:

| Orchestration Type            | What It Is                                                                                                                                                                                                                                   | Libraries                                                                                                          | Trade-offs                                                                                                                                                             |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Distributed Actor Model**   | Components are _actors_, isolated stateful processes with mailboxes, managed by a runtime that handles scheduling, resource placement, fault tolerance, and object transfer. Communication is via asynchronous RPC / futures / object store. | **Ray:** verl, SkyRL, NeMo-RL, SLIME, MILES, ROLL, OAT, open-instruct. **Monarch:** TorchForge.                    | Richest abstraction; solves scheduling and fault tolerance out-of-the-box. Adds a non-trivial runtime dependency and framework-specific debugging overhead.            |
| **Native Python Concurrency** | Components are threads, coroutines (`asyncio`), `threading` primitives, `multiprocessing` child processes, and queues. No external orchestration runtime.                                                                                    | verifiers-rl, PipelineRL (intra-pool), ART (`asyncio` + child-process proxies), AReaL (`asyncio`-based event loop) | Minimal dependencies, easy to debug, full control. Limited to single-node unless paired with additional IPC (Redis, HTTP, NCCL) for multi-node communication.          |
| **Pub/Sub Message Bus**       | Components are decoupled producers and consumers communicating through append-only streams or message queues. Not orchestration per se, a _data transport layer_ between independently running pools.                                        | PipelineRL (inter-pool: Redis `XADD`/`XREAD` streams for multi-node, append-only JSONL files for single-node)      | Clean decoupling across pool boundaries without RPC. Does not manage process lifecycle, scheduling, or fault recovery; must be paired with another orchestration type. |
| **HTTP Microservices**        | Components are independent services communicating via REST APIs. Language-agnostic, maximum decoupling.                                                                                                                                      | Atropos                                                                                                            | Any inference server, any language, zero shared state. Highest latency (if NCCL); no shared object store; fault tolerance is the user's responsibility.                |

> [!NOTE]
> **Note on Tunix:** Tunix (Google) uses a JAX-native mesh model with `ThreadPoolExecutor` for async overlap and `jax.device_put` for cross-mesh weight transfer. It is architecturally distinct enough from the PyTorch ecosystem that direct comparison on orchestration is not meaningful; it lives in the XLA/TPU world with its own coordination primitives.

The table above reveals a striking pattern: **eight of the sixteen libraries surveyed use Ray as their orchestration backbone**. This is not a coincidence; it reflects a deep architectural fit between the actor model and the structure of RL training. A [survey by Anyscale](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms) (the company behind Ray) of open-source RL libraries for LLMs confirms this convergence. RL training at large scales involves fundamentally heterogeneous components (inference engines, training engines, environments, reward models) that must be orchestrated across a cluster, often on different hardware types, with different scaling requirements and failure modes. Ray's actor model maps directly onto this:

1. **Actor isolation and heterogeneous resources.** Each RL component (vLLM inference server, FSDP trainer, reward model, environment pool) becomes a Ray actor with its own resource requirements (`num_gpus`, `num_cpus`, `memory`). Placement groups give fine-grained control over GPU affinity without manual SSH/torchrun orchestration.

2. **Scheduling and autoscaling.** Ray's scheduler handles the combinatorial problem of placing heterogeneous actors across a cluster. When generation requires 8× more GPU-hours than training, you can just tell Ray to scale your inference actors independently.

3. **Fault tolerance.** Long RL training runs (days to weeks) are vulnerable to GPU failures, OOM kills, and network partitions. Ray's actor restart policies and object store replication provide resilience that would require significant custom infrastructure with raw `asyncio` and `multiprocessing`. Concrete example of the fault tolerance: `open-instruct`, for example, relies on Ray's actor supervision to recover from vLLM engine crashes mid-rollout.

4. **Object store for zero-copy data transfer.** Rollout data can be large, tens of GB per batch for very long-context reasoning. Ray's shared-memory object store enables zero-copy transfer between actors on the same node, avoiding serialization overhead that usually comes with `multiprocessing.Queue` approaches.

5. **Ecosystem maturity.** Ray has been battle-tested at scale since 2017, with production deployments on thousands of GPUs. The debugging overhead is real (Ray Dashboard, distributed stack traces, placement group failures), but the alternative, building equivalent coordination from scratch, is worse at the multi-node scale. That said, Ray is a heavy dependency: it pulls in its own scheduler, object store, and dashboard, adding operational complexity that not every team needs. This is exactly why libraries like PRIME-RL, PipelineRL, and AReaL opted for lightweight native-Python coordination (asyncio, threading, Redis streams) instead --- when you control the full stack and your deployment topology is fixed, the simplicity and debuggability of vanilla Python often outweigh the conveniences Ray provides.

The cost is a hard dependency on a non-trivial runtime. This trade-off can be worthwhile, especially for production-scale training (64+ GPUs, multi-day runs, complicated reward computation).

While Ray's actor model is the main player on the field, [Monarch](https://github.com/pytorch/monarch) emerged as a new PyTorch-native distributed actor framework from Meta, purpose-built for GPU workloads. Like Ray, Monarch is based on the actor model; components are independent actors with mailboxes communicating via messages, but it is designed from the ground up for the PyTorch/CUDA ecosystem rather than being a general-purpose distributed runtime.

Monarch offers several capabilities particularly relevant to async RL. An [example implementation of async RL with Monarch](https://allenwang28.github.io/monarch-gpu-mode/05_rl_intro.html) (from the GPU Mode lecture series) demonstrates the architecture: generators, a replay buffer, and a trainer are modelled as Monarch actors, with the replay buffer absorbing latency variance from straggler rollouts and RDMA weight sync pushing updated parameters to generators without blocking training. The pattern is structurally identical to Ray-based designs (verl, SkyRL, open-instruct) but implemented with pure PyTorch-native primitives.

### Axis 2: Rollout Buffer Design

_How do generated rollouts flow from inference to training, and how deep is the pipeline?_

The buffer is the data structure sitting between generation and training. Its depth controls the maximum degree of asynchrony, and therefore the maximum staleness.

| Pattern                            | Depth     | Libraries                                                                                                                                      | Characteristic                                                                   |
| ---------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **No buffer** (synchronous)        | 0         | TRL (current), **ART** (gather-all-then-train)                                                                                                 | Generation and training alternate strictly; zero staleness, maximum idle time    |
| **Double-buffer** (one-step-ahead) | 1         | verifiers-rl, SLIME (async mode), MILES, OAT                                                                                                   | Submit generation N+1 at the start of training step N; overlap exactly one batch |
| **Bounded async queue**            | 2–K       | SkyRL, verl (fully async), NeMo-RL, ROLL, PRIME-RL, TorchForge, Tunix, **open-instruct** (`async_steps`), **AReaL** (`max_head_offpolicyness`) | Multiple batches in flight; staleness bounded by queue capacity                  |
| **Unbounded / stream**             | Unlimited | PipelineRL (Redis streams), SLIME (fully async mode), Atropos                                                                                  | Continuous generation; staleness bounded only by explicit version control        |

The [double-buffer pattern](https://en.wikipedia.org/wiki/Multiple_buffering) is the simplest upgrade from synchronous to asynchronous training: it overlaps exactly one generation with one training step and introduces at most one step of policy lag !

Deeper queues, on the other hand, improve throughput but require staleness management.

The buffer controls how much data is in flight. But data is only half the equation. The other half is getting updated weights _back_ to the inference servers before those rollouts go stale. That's where weight sync comes in!

### Axis 3: Weight Synchronisation Protocol

_How do new model weights reach the inference servers after a gradient update?_

> **Scope note:** This axis focuses on **disaggregated mode**, where inference and training run on separate GPU pools, since that is the deployment topology where async overlap (and therefore weight sync design) actually matters. Colocated setups (same GPUs for both roles) are inherently synchronous and do not face the transport/interrupt trade-offs discussed below.

This is the most architecturally consequential axis. The protocol determines sync latency, interrupt granularity, and whether partial rollouts are possible.

There is a critical distinction to make here: the **transport mechanism** and the **interrupt model**. Most libraries pause generation at a coarse boundary, an HTTP request, a full batch, or even a full training step, before initiating weight transfer. PipelineRL is the outlier: it never stops generating at all.

**Transport mechanism:**

| Mechanism                | Latency    | Libraries                                                                           |
| :----------------------- | :--------- | :---------------------------------------------------------------------------------- |
| **NCCL Broadcast**       | ~100–500ms | PipelineRL, SkyRL, SLIME, MILES, ROLL, OAT, NeMo-RL, PRIME-RL, open-instruct, AReaL |
| **NCCL + Bucketing**     | ~20ms      | verl                                                                                |
| **KV + Shared Memory**   | Low        | TorchForge                                                                          |
| **Filesystem + HTTP**    | Medium     | PRIME-RL, AReaL, ART                                                                |
| **CUDA IPC (Zero-copy)** | Very Low   | NeMo-RL, MILES                                                                      |
| **JAX Cross-mesh**       | Low        | Tunix                                                                               |
| **HTTP PUT**             | High       | verifiers-rl                                                                        |
| **Filesystem + Restart** | Very High  | Atropos                                                                             |

**In the interrupt model, when does the generation pause to accept new weights?**

This is where PipelineRL fundamentally diverges from every other library. Rather than listing each library individually, the landscape collapses into five conceptual tiers, ordered from finest to coarsest interrupt granularity:

| Interrupt Granularity                    | What Happens                                                                                                                                    | Libraries                                                    |
| :--------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| **Never** (In-flight per-forward-pass)   | Sequences never stop. The weight swap happens between token decode steps (~1-10ms gap). Running sequences seamlessly continue with new weights. | PipelineRL, open-instruct (opt-in)                           |
| **Per HTTP Request** (Abort + Resync)    | In-flight HTTP requests are aborted. Partial tokens are resubmitted with a prefix-resume mechanism or recycled for retry.                       | SkyRL, SLIME, MILES                                          |
| **Soft Pause** (Drain in-flight)         | No new generation requests are accepted while in-progress ones finish naturally. Once drained, weights are synced and generation resumes.       | PRIME-RL, AReaL, open-instruct (default), verl (async)       |
| **Per Training Step / Batch** (Blocking) | Generation must fully complete. The trainer and inference engine take turns blocking each other.                                                | NeMo-RL, ROLL, OAT, TorchForge, Tunix, verifiers-rl, Atropos |

The "never-stop" tier is qualitatively different from all others: PipelineRL, for example, hooks into the inference engine so that the lock is acquired and released _per transformer forward pass_ (one token step for one sequence). A weight update waits at most one forward pass (\~few ms), swaps all parameters, and generation resumes immediately. Every other library stops generation at a coarser boundary, from one HTTP request (\~hundreds of ms) up to a full batch boundary (\~seconds).

Weight sync controls _when_ new weights arrive. But async training means rollouts are always being generated under _some_ policy version, and that _generating_ policy might be several gradient steps behind the trainer. How libraries handle this policy lag is staleness management.

### Axis 4: Staleness Management

_How does the system handle the fact that generated rollouts may come from an older policy than the one being trained?_

Once generation and training overlap, samples become off-policy. Three **orthogonal** strategies have emerged for managing this staleness, and most production systems combine more than one:

**Strategy 1: Per-sample version rejection.** Every sample is tagged with the integer policy version that generated it. At training time, samples whose version falls behind the current policy by more than a threshold are hard-dropped before entering the loss computation. Simple and correct, but wastes the precious compute spent generating discarded samples.

**Strategy 2, Depth Bounding.** The queue or buffer between generation and training has a bounded capacity (or an explicit staleness gate), which architecturally limits how far behind any sample can be. This ranges from depth=1 (one-step-ahead double buffering, where staleness is impossible by construction) to explicit capacity formulas tied to version gaps. No per-sample version tracking is required; the bound is enforced by the system's pipeline depth.

**Strategy 3, IS-weighted loss correction.** Stale samples that reach the trainer are reweighted by the importance sampling ratio \\(\frac{\pi*{\theta}(a \mid s)}{\pi*{\text{old}}(a \mid s)}\\), typically clipped (Truncated IS). Some libraries also apply OPSM (zero-out loss for off-policy samples with negative advantage). This preserves throughput; no samples are discarded, but there is a cost in gradient variance from the IS ratios.

These strategies are orthogonal: a system can use version rejection alone, depth bounding alone, IS correction alone, or any combination of them. Synchronous systems avoid the problem entirely by never overlapping generation and training.

| Library           | Version Rejection | Depth Bounding | IS Correction | Key Config / Notes                                                                                                   |
| ----------------- | :---------------: | :------------: | :-----------: | -------------------------------------------------------------------------------------------------------------------- |
| **AReaL**         |        ❌         |       ✅       |      ⚠️       | `max_head_offpolicyness` capacity formula; optional `use_decoupled_loss` adds IS weight capped at 5.0                |
| **ART**           |         —         |       —        |       —       | Synchronous; all rollouts collected before training; no staleness by design                                          |
| **Atropos**       |        ❌         |       ✅       |      ❌       | `max_batches_offpolicy`, ceiling on buffered batches                                                                 |
| **MILES**         |        ❌         |       ❌       |      ✅       | TIS + OPSM                                                                                                           |
| **NeMo-RL**       |        ✅         |       ❌       |      ❌       | `max_trajectory_age_steps`, per-sample version drop                                                                  |
| **OAT**           |        ❌         |       ❌       |      ✅       | Clipped TIS ratio                                                                                                    |
| **open-instruct** |        ❌         |       ✅       |      ⚠️       | `async_steps` cap (default 1, production 8); optional `--truncated_importance_sampling_ratio_cap ρ` adds clipped TIS |
| **PipelineRL**    |        ✅         |       ❌       |      ❌       | `max_lag`, integer version tag per sample; drop if age exceeds threshold                                             |
| **PRIME-RL**      |        ✅         |       ✅       |      ✅       | Full hybrid: `max_async_level` version gap + `max_off_policy_steps` cancellation + IPO trust-region IS               |
| **ROLL**          |        ❌         |       ❌       |      ✅       | Richest IS suite: TIS, TOPR, CISPO, Kimi15, six off-policy loss variants                                             |
| **SkyRL**         |        ❌         |       ✅       |      ❌       | `max_staleness_steps`, capacity gate blocks new rollouts when exceeded                                               |
| **SLIME**         |        ❌         |       ❌       |      ✅       | TIS + OPSM (off-policy masking for partial rollouts)                                                                 |
| **TorchForge**    |        ✅         |       ❌       |      ❌       | `max_policy_age`, per-sample version tag; hard drop                                                                  |
| **Tunix**         |        ❌         |       ✅       |      ❌       | Bounded queue + sync per step; staleness structurally limited                                                        |
| **verl**          |        ❌         |       ❌       |      ✅       | Clipped TIS ratio; optional OPSM                                                                                     |
| **verifiers-rl**  |        ❌         |       ✅       |      ❌       | Depth=1 FIFO + sync every step; staleness impossible by construction                                                 |

> ✅ = yes, ❌ = no, ⚠️ = optional / configurable, — = not applicable (synchronous)

- **Version rejection** is simple and correct, but wastes compute when many samples are discarded.
- **IS correction** preserves throughput at the cost of gradient variance.
- **Depth bounding** is the coarsest mechanism, but it avoids per-sample bookkeeping entirely.

The trend in production systems (PRIME-RL, AReaL, open-instruct) is toward **hybrid approaches** that combine depth bounding with optional IS correction, getting the architectural simplicity of bounded queues with the loss-level safety net of importance weighting for stable training.

Staleness management handles data that was generated under an old policy. But what about data that's _still being generated_ when a weight update lands?

### Axis 5: Partial Rollout Handling

_What happens to a generation in progress when a weight update arrives?_

This is critical for long-context tasks where a single rollout can take minutes. Four strategies:

| Strategy                                      | Libraries                         | Description                                                                                                                                                                                                                                                                  |
| --------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Implicit continuation**                     | PipelineRL                        | Sequences are never interrupted. Weights swap between forward passes; the sequence simply continues with new weights. Stored logprobs remain valid because training uses the _recorded_ \\(\pi\_{\text{old}}\\), not recomputed.                                             |
| **Abort + retry with prefix**                 | SkyRL, SLIME                      | Active sequences are aborted. Partial tokens are accumulated, then resubmitted with a prefix-resume mechanism using the new weights.                                                                                                                                         |
| **Explicit save/resume**                      | verl (fully async)                | The rollout worker saves partial token IDs and logprobs to a buffer, waits for sync, then resumes from the saved prefix.                                                                                                                                                     |
| **Group cancellation (generation continues)** | PRIME-RL                          | Stale rollout groups have their async tasks cancelled; the inference server continues serving in-flight HTTP requests whose results are discarded. Weight sync triggers between HTTP requests without interrupting mid-request generation.                                   |
| **No partial rollout support**                | verifiers-rl, OAT, Atropos, Tunix | Weight sync only happens at batch boundaries. In-flight generations must complete before sync begins.                                                                                                                                                                        |
| **Soft pause, in-flight sequences complete**  | **AReaL**                         | A pause signal blocks new KV-cache allocations but does not abort in-progress sequences. The task dispatcher stops submitting new tasks; running tasks run to completion. After weight sync, generation dispatch resumes.                                                    |
| **Full sleep, no in-flight at sync time**     | **ART**                           | By design, training only begins after all rollouts are collected. There are never in-progress sequences when sleep is triggered. Level-1 sleep (in-progress requests exist) offloads KV cache to CPU; level-2 sleep discards it entirely.                                    |
| **Drain-or-inflight (configurable)**          | **open-instruct**                 | Default: a stop flag gates new prefetching; weight update waits for active tasks to drain. With in-flight updates enabled, drain is bypassed and weights broadcast while tokens are still being generated; sequences in progress continue with a mix of old and new weights. |

So far, every axis has assumed full-parameter training. But in LoRA training, you're only training a few million adapter parameters instead of billions, the weight sync problem nearly disappears. Let's look at how these libraries support LoRA training.

### Axis 6: LoRA Training Support

_Does the library support parameter-efficient training via LoRA adapters, in what modes, and does it exploit adapter-only weight sync?_

LoRA is arguably the most practically consequential axis for teams with limited GPU budgets. It reduces the trainable parameter count by 99%+, halves peak activation memory, and, when the inference server is LoRA-aware, enables _adapter-only weight sync_: instead of broadcasting every parameter of a 7B+ model (~100–500ms NCCL), only the adapter deltas are pushed to vLLM, which at rank 32 amounts to ~50 MB, a sub-millisecond transfer.

| Library           | LoRA Supported             | Mode Restriction                 | LoRA Backend                                   | Adapter-Only Sync                                                                                 |
| ----------------- | -------------------------- | -------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **AReaL**         | ✅ Yes                     | FSDP2 only (not Megatron/Archon) | HF `peft`                                      | ✅ Yes (disk-based sync; only trainable params transferred; vLLM adapter hot-swap)                |
| **ART**           | ✅ Yes (primary design)    | Both (shared + dedicated GPU)    | Unsloth/`peft` (default); custom Megatron LoRA | ✅ Yes (only adapter saved/loaded; in-process or HTTP adapter hot-swap; base weights never moved) |
| **Atropos**       | ✅ Yes                     | Disaggregated                    | HF `peft`                                      | ✅ Yes (`lora_only` / `lora_restart` modes)                                                       |
| **MILES**         | ✅ Yes                     | Both (colocated + disaggregated) | Megatron-Bridge                                | ✅ Yes (adapter sync config for SGLang)                                                           |
| **NeMo-RL**       | ✅ Partial\*               | Both                             | Custom (not `peft`)                            | ❌ No evidence                                                                                    |
| **OAT**           | ✅ Yes                     | Both                             | HF `peft`                                      | ✅ Yes (LoRA-only sync mode)                                                                      |
| **open-instruct** | ⚠️ Code exists, not wired‡ | —                                | HF `peft` (SFT/DPO only)                       | ❌ No (LoRA not applied in the RL trainer)                                                        |
| **PipelineRL**    | ✅ Yes                     | Non-colocated                    | HF `peft`                                      | ❌ No (full NCCL broadcast)                                                                       |
| **PRIME-RL**      | ✅ Yes                     | Disaggregated                    | Custom MultiLoRA (not `peft`)                  | ✅ Yes (adapter-only state dict extraction)                                                       |
| **ROLL**          | ✅ Partial†                | DeepSpeed backend only           | HF `peft` / TRL                                | ❌ No evidence                                                                                    |
| **SkyRL**         | ✅ Yes                     | Both                             | `peft` (FSDP) / Megatron-Bridge (Megatron)     | ✅ Yes (filesystem-based adapter sync)                                                            |
| **SLIME**         | ❌ No                      | —                                | —                                              | ❌ No                                                                                             |
| **TorchForge**    | ❌ No                      | —                                | —                                              | ❌ No                                                                                             |
| **Tunix**         | ✅ Yes                     | Both                             | qwix (JAX-native)                              | ✅ Yes (auto-detected)                                                                            |
| **verl**          | ✅ Yes (most complete)     | Both                             | `peft` (FSDP) / Megatron-Bridge (Megatron)     | ✅ Yes (unmerged adapter sync)                                                                    |
| **verifiers-rl**  | ✅ Yes (via prime-rl)      | Disaggregated                    | HF `peft` + FSDP2 + vLLM                       | ✅ Yes (vLLM LoRA serving)                                                                        |

\* NeMo-RL: LoRA for GRPO and DPO is supported only on the DTensor backend; the Megatron Core backend is SFT-only (RL LoRA listed as "coming soon"). Uses a custom DTensor-compatible LoRA module (not `peft`), optionally with Triton kernels.

† ROLL: LoRA is officially supported with the DeepSpeed training backend only. Megatron-backend LoRA appeared in the Feb 2026 changelog but remains experimental.

‡ open-instruct: The model config exposes LoRA-related fields (`use_peft`, `lora_r`, `lora_alpha`), and adapter saving is handled in the checkpoint logic. However, the `peft` model is never initialised in the RL training path; LoRA remains an SFT/DPO-only feature for the RL trainer as of March 2026.

**Three LoRA implementation families:**

1. **HuggingFace `peft`** (PipelineRL, SkyRL/FSDP, verifiers-rl, ROLL, OAT, Atropos): The most common choice. Standard checkpoint format (`adapter_model.safetensors`), compatible with any HF Transformers training loop. ZeRO-3 interactions require care: OAT, for example, needs to disable the fused LM head; ROLL must disable gradient checkpointing entirely.

2. **Megatron-Bridge** (verl/Megatron, SkyRL/Megatron, MILES): Required for 3D-parallel training (TP × PP × DP). Supports multiple LoRA types: `lora`, `canonical_lora` (splits merged QKV → separate Q/K/V adapters), `vlm_lora`, and `dora`. The `canonical_lora` variant avoids the QKV merge, thereby improving training stability. MILES saves checkpoints in both HF `peft` format and Megatron-native per-rank format.

3. **Custom implementations** (NeMo-RL, PRIME-RL, Tunix/qwix): Library-specific LoRA modules not interoperable with `peft` checkpoints. PRIME-RL uniquely supports multiple simultaneous adapters in a single run for multi-experiment parallelism. Tunix uses Google's `qwix` JAX library, which adds built-in QLoRA (NF4 quantization) and TPU-native gradient routing. NeMo-RL uses a custom DTensor-compatible module with an optional Triton fused kernel.

**The adapter-only weight sync opportunity (interaction with Axis 3):**

Eight of the thirteen libraries support pushing **only the LoRA adapter deltas** to the inference server. This changes the character of the weight sync problem (Axis 3) entirely. When using full-parameter training, the interrupt model (per-forward-pass lock vs. per-request abort vs. per-batch pause) determines how much generation is wasted during an NCCL broadcast. When using LoRA with adapter-only sync, the transfer is so small that nearly any interrupt model delivers equivalent throughput! Even Atropos's brute-force HTTP hot-swap becomes viable.

---

### Axis 7: Distributed Training Backend & Parallelism

_What parallelism strategy does the library use for training, and how does this constrain or enable the async architecture?_

This axis cuts across every other axis. The choice of training backend determines how large a model can fit per GPU, how many collective operations are needed to gather weights before broadcasting to inference servers, and which model architectures can be trained at all. It is the single most consequential decision for teams scaling beyond 30B parameters or moving from dense to Mixture-of-Experts models.

| Library           | Training Backend           | Parallelism            | HF Model Loading    | MoE / EP Support |
| :---------------- | :------------------------- | :--------------------- | :------------------ | :--------------- |
| **AReaL**         | FSDP2, Megatron, Archon    | DP, SP, TP, PP, CP, EP | ✅ Direct / Convert | ✅               |
| **ART**           | Unsloth, Megatron          | DP, TP, EP             | ✅ Direct / Convert | ✅               |
| **Atropos**       | PyTorch Native, TRL        | DP                     | ✅ Direct           | ❌               |
| **MILES**         | Megatron, FSDP2            | DP, TP, PP             | 🔄 Convert          | ✅               |
| **NeMo-RL**       | FSDP2, Megatron            | DP, SP, TP, PP, CP, EP | ✅ Direct / Convert | ✅               |
| **OAT**           | DeepSpeed                  | DP, TP                 | ✅ Direct           | ❌               |
| **open-instruct** | DeepSpeed                  | DP, SP                 | ✅ Direct           | ❌               |
| **PipelineRL**    | DeepSpeed                  | DP, SP                 | ✅ Direct           | ❌               |
| **PRIME-RL**      | FSDP2                      | DP, TP, CP, EP         | ✅ Direct           | ✅               |
| **ROLL**          | DeepSpeed, Megatron, FSDP2 | DP, SP, TP, PP, CP, EP | ✅ Direct / Convert | ✅               |
| **SkyRL**         | FSDP, Megatron             | DP, SP, TP, PP, EP     | ✅ Direct / Convert | ✅               |
| **SLIME**         | Megatron                   | DP, TP, PP, SP         | 🔄 Convert          | ✅               |
| **TorchForge**    | FSDP2                      | DP, TP, CP             | ✅ via TorchTitan   | ❌               |
| **Tunix**         | JAX/XLA                    | DP, TP                 | ❌ Custom Flax      | ❌               |
| **verl**          | FSDP, Megatron             | DP, SP, TP, PP, CP, EP | ✅ Direct / Convert | ✅               |
| **verifiers-rl**  | DeepSpeed                  | DP                     | ✅ Direct           | ❌               |

The training backend creates direct implications for async RL library design:

**Weight sync speed is a direct function of the training backend, and faster sync means less staleness.**

In a disaggregated async setup, weight sync does _not_ necessarily stall inference. The key design decision is **how the weight update interacts with in-flight generation**; four strategies exist, ordered from least to most disruptive:

- **Atomic swap, no interruption.** The full weight update is dispatched as a single blocking RPC to the inference engine. Each forward pass sees either all old or all new weights, never a mix. Generation pauses for at most one forward-pass gap (~few ms). (PipelineRL)
- **Per-parameter streaming, no interruption.** Each parameter is sent as a separate RPC + NCCL broadcast. Forward passes interleave between individual parameter updates, so in-flight sequences genuinely see a mix of old and new weights across layers. Maximum overlap, but weakest consistency. (open-instruct, inflight mode)
- **Dispatch gate, drain in-flight, then sync.** New requests are held back while in-progress sequences complete naturally; weights are broadcast only after the pipeline drains. No wasted tokens, but a sync bubble proportional to the longest in-flight sequence. (PRIME-RL, AReaL, open-instruct default, verl fully-async)
- **Hard pause or abort.** Inference is paused, or in-flight requests are aborted before weight transfer begins. Cleanest consistency, highest wasted compute. (verl, SkyRL)

But even in libraries where inference continues, **slower sync means longer periods where inference runs on stale weights**. The policy version gap between the trainer and the inference pool grows with sync duration. Something to take into account.

**MoE support is an increasingly important differentiator as the field moves toward sparse models.**\
The trend is clear: frontier models are sparse (DeepSeek-V3, Qwen3-MoE, Mixtral, DBRX), and open-weight MoEs are becoming the default starting point for post-training. Training these models requires Expert Parallelism (EP), distributing different experts to different ranks, which most async RL libraries do not support. Only Megatron-backed libraries (verl, SLIME, MILES, ROLL, NeMo-RL) and PRIME-RL's FSDP2+EP path handle EP correctly. ZeRO-based libraries (PipelineRL, verifiers-rl, OAT, open-instruct) can _load_ MoE HuggingFace model classes, but without EP each expert is sharded across all ZeRO-3 ranks rather than being placed on a dedicated rank; every forward pass AllGathers every expert, negating the sparsity advantage entirely. EP also complicates weight sync: before broadcasting to vLLM/SGLang (which typically serves all experts from a single TP group), the trainer must AllGather expert parameters from every EP rank, an \\( O(N*{\text{experts}} \times E*{\text{size}}) \\) communication (where \\( E\_{\text{size}} \\) is the parameter count per expert) that does not exist for dense models. For a 235B MoE with 256 experts, this is a substantial sync cost. Libraries that want to remain relevant for post-training on the next generation of open MoE models need EP-aware training \*and\* EP-aware weight sync.

**MoE LoRA is an emerging requirement, and a tricky one.**\
LoRA on dense models is well-understood (Axis 6): attach adapters to attention projections, train them, sync only the adapter deltas. MoE LoRA is harder because the natural target is the _expert FFN layers_, meaning each expert gets its own adapter. For a model with 64 experts and rank-32 LoRA on each expert's gate/up/down projections, the adapter count jumps from ~20 (dense) to ~200+ (MoE), and the adapters are distributed across EP ranks. Weight sync must gather adapters from every EP rank before pushing them to the inference server, a coordination problem that does not exist for dense LoRA. On the training side, **ART** explicitly implements MoE expert LoRA layers (Megatron EP path with per-expert LoRA and manual allreduce), **MILES** supports LoRA via Megatron-Bridge which can target expert layers, and **PRIME-RL** also supports [MoE LoRA via a custom `MultiMoELoRA` module](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/models/layers/lora/multi_moe.py). verl's Megatron-Bridge path supports LoRA types including `vlm_lora`, but MoE-specific expert LoRA is not documented. On the inference side, **vLLM** has supported [per-expert LoRA loading and serving for MoEs](https://github.com/vllm-project/vllm/blob/2488a82f89b15ad2ebed12160dcc423d44210db2/vllm/lora/ops/triton_ops/fused_moe_lora_op.py#L158) via fused Triton kernels for several releases, while **SGLang** has an [open PR to add MoE LoRA support](https://github.com/sgl-project/sglang/pull/14105). Support for expert parallel inference with MoE LoRAs is currently in the works for both vLLM and SGLang. As MoE models become the default for post-training, MoE LoRA with efficient adapter-only sync will be a key capability gap to close.

That covers the seven axes, each captures a different facet of the same underlying problem. Together, they give us a complete lens for comparing libraries. Time to put it all on one page.

---

## 4. Global Overview: Sixteen Libraries at a Glance

> [!NOTE]
> **Note:** This overview reflects the state of these libraries as of March 2026. The ecosystem is evolving rapidly; specific features, backends, and integrations may change in the near future.

| Library           | Org            | Orchestration Type                                       | Inference Server       | Weight Sync                                 | Staleness Management     | Partial Rollout                     | Training Backend                      | Dist. Parallelism                                                        | LoRA Support                               |
| ----------------- | -------------- | -------------------------------------------------------- | ---------------------- | ------------------------------------------- | ------------------------ | ----------------------------------- | ------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------ |
| **AReaL**         | inclusionAI    | Native Python (asyncio + HTTP RPC); pluggable Ray/Slurm  | vLLM, SGLang           | NCCL chunked OR filesystem safetensors      | Depth + IS (optional)    | 🟧 Soft pause (in-flight complete)  | FSDP2 or Megatron-LM or Archon        | FSDP2: DP+SP+TP; Megatron: TP+SP+PP+CP+EP; Archon: FSDP2+TP+SP+PP+EP     | ✅ `peft` (Adapter-only)                   |
| **ART**           | OpenPipe       | Native Python (asyncio + mp child processes)             | vLLM                   | LoRA adapter swap (no full weight transfer) | Synchronous (none)       | ❌ No                               | Unsloth (single-GPU); Megatron-LM     | None (Unsloth); TP×EP×DP (Megatron)                                      | ✅ `peft` / Megatron LoRA (Adapter-only)   |
| **Atropos**       | NousResearch   | HTTP Microservices (FastAPI)                             | vLLM, SGLang, OpenAI   | FS checkpoint + vLLM restart                | Depth bounding           | ❌ No                               | Single-GPU PyTorch; TRL/Accelerate    | None (native); FSDP/ZeRO via TRL adapter                                 | ✅ `peft` (Adapter-only)                   |
| **MILES**         | radixark       | Distributed Actor (Ray)                                  | SGLang                 | NCCL OR CUDA IPC                            | IS correction            | 🟧 Abort + recycle to buffer        | Megatron-LM (primary); FSDP2          | Megatron: TP×PP×DP; FSDP2 available; colocated CUDA IPC                  | ✅ Megatron-Bridge (Adapter-only)          |
| **NeMo-RL**       | NVIDIA         | Distributed Actor (Ray)                                  | vLLM, SGLang, Megatron | NCCL OR CUDA IPC-ZMQ OR HTTP                | Version rejection        | ✅ In-flight continuation           | DTensor (FSDP2+TP) or Megatron-Bridge | DTensor: TP+SP+CP+FSDP2; Megatron: TP×PP×CP×EP×ETP + FSDP2               | 🟧 Custom (No adapter-only sync)           |
| **OAT**           | SAIL-SG        | Distributed Actor (Ray)                                  | vLLM                   | NCCL per-param + ZeRO-3 gather              | IS correction            | ❌ No                               | DeepSpeed ZeRO-2/3                    | ZeRO-2 / ZeRO-3 DP; AutoTP                                               | ✅ `peft` (Adapter-only)                   |
| **open-instruct** | AI2 (AllenAI)  | Distributed Actor (Ray)                                  | vLLM                   | NCCL broadcast; optional in-flight updates  | Depth + IS (optional)    | 🟧 Drain-or-inflight (configurable) | DeepSpeed ZeRO-0/2/3                  | ZeRO-3 DP + Ulysses SP; vLLM TP (inference only)                         | ❌ No                                      |
| **PipelineRL**    | ServiceNow     | Native Python + Pub/Sub (asyncio + Redis/JSONL)          | vLLM                   | NCCL pg + HTTP notify                       | Version rejection        | ✅ Implicit continuation            | DeepSpeed ZeRO-3                      | ZeRO-3 DP + Ring SP; ZeRO++ available                                    | ✅ `peft` (Full sync)                      |
| **PRIME-RL**      | PrimeIntellect | Native Python (asyncio + FS/ZMQ)                         | vLLM                   | Filesystem safetensors + HTTP OR NCCL       | Version + depth + IS     | 🟧 Group cancellation               | FSDP2 (exclusively)                   | FSDP2 per-block + TP + CP + EP; pp=1                                     | ✅ Custom MultiLoRA (Adapter-only)         |
| **ROLL**          | Alibaba        | Distributed Actor (Ray)                                  | vLLM, SGLang           | NCCL via dedicated update group             | IS correction            | ❌ No                               | DeepSpeed ZeRO or Megatron or FSDP2   | DS: ZeRO+Ulysses SP; Megatron: TP×PP×CP×EP; FSDP2: HSDP+Ulysses          | 🟧 `peft` (DeepSpeed only)                 |
| **SkyRL**         | NovaSky-AI     | Distributed Actor (Ray) + Native Python                  | vLLM, SGLang           | NCCL pg                                     | Depth bounding           | 🟧 Abort + retry with prefix        | FSDP/FSDP2 or Megatron-Bridge         | FSDP: ZeRO shard + Ulysses SP; Megatron: full 5D via bridge; JAX backend | ✅ `peft` / Megatron-Bridge (Adapter-only) |
| **SLIME**         | THUDM          | Distributed Actor (Ray)                                  | SGLang                 | NCCL pg, bucketed                           | IS correction            | 🟧 Abort + recycle to buffer        | Megatron-LM                           | TP×PP×DP; Megatron→HF conversion; MoE EP all-gather                      | ❌ No                                      |
| **TorchForge**    | Meta           | Distributed Actor (Monarch)                              | vLLM                   | torchstore + shared memory prefetch         | Version rejection        | ❌ No                               | FSDP2 via TorchTitan                  | FSDP2 + TP; CP partial; PP not yet implemented                           | ❌ No                                      |
| **Tunix**         | Google         | Native Python (ThreadPoolExecutor + asyncio); JAX-native | vLLM, SGLang, JAX      | Cross-mesh reshard                          | Depth bounding           | ❌ No                               | JAX/XLA 2D mesh                       | 2D JAX mesh: FSDP + TP; no PP; TPU-primary                               | ✅ qwix / QLoRA (Adapter-only)             |
| **verl**          | ByteDance      | Distributed Actor (Ray)                                  | vLLM, SGLang           | NCCL + checkpoint-engine buckets            | IS correction            | ✅ Explicit save/resume             | FSDP1/FSDP2 or Megatron-Core          | FSDP: ZeRO-2/3/HSDP + Ulysses SP; Megatron: TP×PP×VPP×CP×EP×ETP          | ✅ `peft` / Megatron-Bridge (Adapter-only) |
| **verifiers-rl**  | PrimeIntellect | Native Python (threading + asyncio)                      | vLLM                   | PyNCCL broadcast                            | Depth bounding (depth=1) | ❌ No                               | DeepSpeed ZeRO-3 (Accelerate)         | ZeRO-3 DP only; no TP/PP                                                 | ✅ `peft` (Adapter-only)                   |

That's the current state of play. But the field is moving fast, and several emerging trends are about to stress-test these architectures in ways their designers may not have anticipated.

---

## 5. The Next Wave: Design Implications

The trends below are not a catalogue of new techniques; each one creates concrete pressure on the infrastructure and algorithmic choices made today. The question is not "what is the frontier?" but "if this trend wins, what breaks in my current stack?"

### 5.1 Critic-Free Algorithms: Memory Freed, But Weight Sync Pressure Increases

PPO's value network doubles the memory footprint of any training node. The field is converging on critic-free variants (GRPO, REINFORCE++, Online DPO) precisely because long CoT reasoning makes this overhead prohibitive at 8K–64K context lengths.

**What this unlocks:** Eliminating the critic frees ~50% of training GPU memory. This slack can be reallocated to: (a) larger rollout batches, directly reducing the straggler variance problem, or (b) co-locating inference and training on the same GPUs, which eliminates the need for a separate NCCL weight sync process group entirely.

**What it does not solve:** Critic-free methods still require frequent weight pushes to inference servers. In fact, they can _increase_ sync pressure: without a value network to provide a stable baseline, GRPO-style algorithms require larger group sizes (G=8–32) to get low-variance advantage estimates, which means more rollouts per step and faster policy drift. Libraries that sync only at coarse boundaries (per training step or per K steps) will see staleness grow faster under critic-free training.

**Asymmetric trajectory filtering** (GRPO-RoC: oversample rollouts, strictly filter positives, uniformly downsample negatives; CISPO/DAPO-style asymmetric clipping in DeepSeek-V3.2 and MiniMax-M1) has a subtler impact on staleness. The issue is not the batch shrinking per se; it is the _composition_ of the surviving batch. Positive trajectories (correct solutions to easy prompts) converge faster and are retained preferentially; harder prompts yield mostly negative trajectories that are discarded. The result: the samples that survive filtering are systematically _older_ than the average rollout in the buffer, because the easy prompts they solve were issued earlier in training. A buffer full of nominally "fresh" rollouts can contain surviving positives spanning a wide range of policy versions. Admission control that tracks staleness at the batch level (e.g., SkyRL's `max_staleness_steps` capacity gate, Atropos's `max_batches_offpolicy`) cannot detect this intra-batch version spread. Per-sample version tagging (Axis 4) is not optional in this regime; the trainer must be able to reject or IS-correct individual samples whose policy version diverges too far, even if the batch they belong to was admitted recently.

Critic-free methods simplify the training side. But the _scoring_ side is about to get more expensive: process reward models score intermediate reasoning steps, not just final answers, and that introduces a whole new synchronisation bottleneck.

### 5.2 Process Rewards: A New Synchronisation Barrier

Outcome reward is scalar and cheap, one call to a verifier at the end of a rollout. Process reward models (PRMs) score intermediate steps, which require either (a) a separate PRM forward pass over the full reasoning trace, or (b) an online utility function computed token-by-token during generation.

**PRPO** (entropy-spike segmentation with PRM scoring per segment) and **DEEP-GRPO** (pivot identification via online utility) both incur computational overhead _between generation and training_. In the current library landscape, this phase maps awkwardly onto the preprocessor pool (PipelineRL) or requires an additional Ray actor (verl, NeMo-RL). Neither is designed for it.

**The key implication:** PRM-based credit assignment breaks the assumption that rewards are cheap to compute. A PRM forward pass over a 32K-token reasoning trace from a 7B model can be very costly. At G=8 completions per prompt, the reward computation could consume non-negligible wall time relative to the generation itself. Two consequences:

1. **Async reward pipelines become necessary.** PRIME-RL runs reward scoring concurrently with training as part of its fully async Orchestrator-Trainer pipeline; the Orchestrator handles scoring while the Trainer performs backward and optimizer steps independently. For PRM-based methods, this pipelined reward computation is not optional; synchronous reward scoring will dominate training wall time.
2. **The separate preprocessor pool becomes necessary**. Running reference logprobs computation and PRM scoring on a dedicated GPU tier, for example, pipelined between generation and training, is the correct architecture for dense credit assignment.

**DEEP-GRPO's pivot resampling** introduces a third-generation pattern alongside standard rollouts and partial rollout resumes: _local resampling from a mid-sequence state_. This requires saving KV cache state at pivot points, which **no current async library supports out of the box**. Weight sync at pivot boundaries could be a new correctness requirement: if weights change between the pivot generation and the local resample, the advantage estimate is corrupted. We can, of course, recompute the KV-cache in a single prefill, but it could waste precious compute in our training.

### 5.3 Multi-Agent Co-Evolution: The Straggler Problem Compounds

Single-agent GRPO trains one policy generating G completions per prompt. Emerging, multi-agent self-play means the effective "group" spans multiple model invocations sequentially chained. The reward is only available after all models in the chain are complete.

**Straggler dynamics change qualitatively.** In single-agent GRPO, the straggler is the longest completion in a group, a tail event in a unimodal length distribution. In multi-agent pipelines, the straggler is the _product_ of two or more length distributions. In a Proposer/Solver multi-agent architecture, if each has a 90th percentile completion time (5× the median), the joint 90th percentile is roughly 25× the median.

**RL on swarms of agents implies a new unit of work.** Today, the atomic unit in every library is a single (prompt, completion, reward) triple. In multi-agent training, the atomic unit becomes an _episode_, a directed graph of turns, tool calls, and inter-agent messages. Buffer design, staleness tracking, and advantage computation all need to operate over episodes. Replaying or forking episodes could also be necessary.

Straggler problems across agents are bad enough when the model is at least internally consistent. With MoE architectures, even a single model can disagree with itself across inference and training frameworks and this raises a new set of emerging problems in RL training.

### 5.4 Training-Inference Mismatch: The Deepseek v3.2 MoE Case Study

The training-inference mismatch problem is endemic in async RL; anytime rollout data is generated under policy \\(\pi*{\text{old}}\\) and gradient updates are computed under \\(\pi*{\theta}\\), the two policies diverge. Most libraries address this with IS correction or hard version rejection. But DeepSeek-V3.2's production experience reveals two **structural** sources of mismatch that IS correction cannot fix.

**Source 1: MoE expert routing inconsistency.** Mixture-of-Experts models activate a sparse subset of experts per token. Inference frameworks (vLLM, SGLang) and training frameworks (Megatron, FSDP) implement the router independently, and differences in floating-point rounding in the gating function can lead to _different expert selections for identical inputs_. When expert routing diverges, the active parameter subspace shifts discontinuously; a gradient step computed assuming Expert A was active is applied to weights that are active under Expert B. DeepSeek-V3.2 found this "induces abrupt shifts in the active parameter subspace, which destabilizes optimization and exacerbates off-policy issues."

Their solution, **Keep Routing**, preserves the exact expert routing paths used during sampling (inference) and enforces those paths during the training forward pass. This requires the inference framework to record and return routing decisions alongside token logprobs, and the training framework to accept and enforce them. No current open-source async RL library implements this. For any team training MoE models (DeepSeek-V3 class, Mixtral, future open MoEs), this is a correctness issue, not a performance issue.

**Source 2: Sampling truncation mask mismatch.** Top-p and top-k sampling truncate the vocabulary at generation time, excluding low-probability tokens from the sampling distribution. During training, the full vocabulary is visible to \\(\pi*{\theta}\\). This violates the importance sampling identity: the action spaces of \\(\pi*{\text{old}}\\) (truncated) and \\(\pi*{\theta}\\) (full) differ, so the IS ratio \\(\pi*{\theta}(o*t) / \pi*{\text{old}}(o_t)\\) is undefined for tokens that were masked during sampling.

DeepSeek-V3.2's **Keep Sampling Mask** solution: record the truncation mask during sampling and apply it to \\(\pi\_{\theta}\\) during the training forward pass, so both policies operate over the same vocabulary subset. This requires passing the mask back from the inference server to the trainer, which is again something no current library infrastructure supports.

**Implications for library design:** Both Keep Routing and Keep Sampling Mask require the inference server to return _additional metadata_ alongside token logprobs, routing decisions, and sampling masks. The current API contract between inference servers (vLLM, SGLang) and trainers is `(token_ids, logprobs, finish_reason)`. Extending this to `(token_ids, logprobs, finish_reason, expert_routing, sampling_mask)` is a breaking change to every library's data flow.

### 5.5 Distillation: The Same Async Problem Under a Different Name

On-policy distillation, where a student model generates sequences, and a teacher model scores them with token-level logprobs, is structurally the same as the async coordination problem in GRPO.

Every design axis in this survey, rollout buffers, weight sync protocols, staleness management, and partial rollout handling, applies identically to distillation. The generation pool produces student rollouts, the teacher scores them (replacing the verifier), and the trainer computes a backward pass with either an advantage-modified GRPO loss or a standalone KL objective. Self-distillation adds one more coordination requirement: the teacher is a frozen snapshot of the student from step _N−k_, so the system must periodically checkpoint the policy and hot-swap the teacher server without disrupting the pipeline, a primitive that no library has fully automated.

**The practical implication for library design is that async RL infrastructure should not be built as a GRPO-specific system**. The generation--scoring--training pipeline is a general pattern that covers RL with outcome rewards, RL with process rewards, on-policy distillation, and self-distillation. Libraries like **SLIME, MILES, PRIME-RL, AReaL, and NeMo-RL** already support both GRPO and on-policy distillation precisely because their async scaffolding treats the reward/scoring phase as a pluggable component rather than a hardcoded verifier call. Any async trainer that aspires to generality should do the same: define the scoring phase as an interface (an HTTP endpoint, a Ray actor, or a co-located forward pass), and let the buffer, staleness, and weight-sync machinery operate identically regardless of what fills it.

---

## 6. Design Choices for TRL's Async Trainer

Having surveyed the full landscape, orchestration models, buffer designs, weight sync protocols, staleness strategies, and partial rollout handling, we can now lay out concrete design choices for an async trainer in TRL, along with the future-proof directions we intend to explore.

### Design Principle: Keep Orchestration Lightweight

One of the strengths of the current TRL implementation is that it does not depend on a heavy orchestrator system to manage the training lifecycle. Data inside the library remains native Python objects without external-library coloring. We want to preserve this: orchestration should stay as simple as possible, with no dependency on heavyweight external frameworks.

### 1. Bounded Queue with Per-Token `model_version` (No Double-Buffering)

Rather than starting with double-buffering and graduating to something more granular, we go straight to a **bounded queue where every token is tagged with the `model_version` that produced it**. This is the lowest possible granularity from the start; it enables importance-sampling correction at the token level, supports simple admission gating (drop or down-weight tokens beyond a staleness threshold), and avoids the architectural debt of retrofitting token-level provenance onto a batch-level buffer later.

### 2. NCCL Weight Sync with Packed Transfers

NCCL process groups are a necessity, and we already use them. Adding bucketing should be the next step as vLLM's [`NCCLWeightTransferEngine`](https://github.com/vllm-project/vllm/blob/f3c6c9c9d794fac5e74b59bc75da6e9d1921eeac/vllm/distributed/weight_transfer/nccl_engine.py) with `packed=True` directly supports bucketed broadcast: it packs parameters into configurable-size `uint8` buffers (default 1 GB, double-buffered across CUDA streams) and broadcasts them via a dedicated NCCL communicator separate from the training process group. This eliminates the per-parameter call overhead that dominates naive broadcast, yielding a massive sync speedup.

Beyond vLLM's built-in engine, we will explore high-performance weight packing libraries for more demanding scenarios:

- **[Awex](https://github.com/inclusionAI/asystem-awex)** (inclusionAI), a dedicated weight synchronization framework for RL training that handles the hard problem of cross-engine transfer: training engines (Megatron, DeepSpeed) and inference engines (SGLang, vLLM) use completely different parallelism strategies and tensor layouts. Awex abstracts this behind a unified conversion layer and deterministic P2P transfer plans. It supports both separated-GPU and co-located (CUDA IPC) modes.

- **[Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake)**, SGLang has moved toward integrating the Mooncake Transfer Engine as its high-performance transport layer, with integrations spanning PD disaggregation, hierarchical KV caching, and elastic expert parallelism. For weight sync specifically, the companion **[checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)** project uses Mooncake's RDMA-backed P2P transfers to update trillion-parameter models (Kimi-K2, 256×H20 GPUs) in ~16–17 seconds. Mooncake is now part of the PyTorch Ecosystem and also serves as a backend plugin for [NVIDIA's NIXL transfer library](https://github.com/ai-dynamo/nixl).

### 3. Partial Rollout Support for Agentic Workloads

Multi-turn tool-use tasks in complex environments can take minutes per rollout. Without a mechanism to handle in-flight rollouts during weight updates, sync windows become pipeline stalls. We will probably explore two strategies experimentally:

- **Prefix-resume**: when weights update mid-rollout, save the KV cache prefix and resume generation from the checkpoint under the new policy. This preserves partial work but requires support from the inference engine for mid-sequence weight swaps.
- **Abort-and-retry**: discard in-flight rollouts that exceed a staleness threshold and re-queue the prompt. Simpler to implement, but wastes compute proportional to the average rollout length at the time of abort.

That's the map, stay tuned, we are working on a concrete async GRPO trainer in TRL, and we'll announce it shortly 🧑‍🍳!
