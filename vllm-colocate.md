---
title: "No GPU left behind: Unlocking Efficiency with Co-located vLLM in TRL" 
thumbnail: /blog/assets/liger-grpo/thumbnail.png
authors:
- user: toslali-ibm
  guest: true
  org: ibm-ai-platform
- user: mirinflim
  guest: true
  org: ibm-ai-platform
- user: qgallouedec
- user: esnible
  guest: true
  org: ibm-ai-platform
- user: rganti
  guest: true
  org: ibm-ai-platform
- user: mudhakar
  guest: true
  org: ibm-ai-platform
---

# No GPU left behind: Unlocking Efficiency with Co-located vLLM in TRL

## 🚀 Introduction

TRL supports training LLMs using GRPO, an online learning algorithm recently introduced in the [*DeepSeekMath* paper](https://huggingface.co/papers/2402.03300). In GRPO, the model learns from its own outputs: it generates responses during training, receives feedback, and uses that feedback to improve itself over time.

This makes generation a critical step in the training loop — and also a major bottleneck. To speed up generation, TRL integrates with vLLM. This combination lets you train powerful models more efficiently in GRPO setup. However, there’s a catch.

## 🧨 The Problem

Before TRL v0.18.0, vLLM was only supported in **server mode**, running as a separate process on different GPUs from the training job. It communicated with the training script over HTTP, which made the setup modular and easy to use — but also introduced GPU inefficiencies.

Here’s what happens:

- During training, the model needs to generate completions frequently.
- The trainer sends a request to the vLLM server, which runs on its own GPUs.
- While vLLM generates, the **training GPUs sit idle** and wait.
- Once generation is done, **vLLM GPUs become idle**, and training resumes.

This “ping-pong” between training and generation causes:

- Wasted GPU time on both sides
- Increased demand for **extra GPUs** just to run inference
- Reduced overall **throughput and higher cost**

In online learning methods like GRPO — where generation happens constantly — this inefficiency becomes even more painful. You spend more on hardware, but don't get the performance you'd expect.

**So, the key question becomes:**  *Can we share the same GPUs for both training and generation, instead of separating them?*

## 💡 The Opportunity

The main issue was that training and inference ran on separate GPUs, leading to idle time and underutilization. The natural solution? Run both on the same GPUs. Instead of having vLLM operate as a standalone server in its own process and devices, what if vLLM could run alongside the training code, within the same distributed process group? This would let us launch a single distributed job where training and inference share the same devices, switching between tasks efficiently without wasting resources.

This approach is what we refer to as **colocation**. Training and inference are co-located on the same GPUs and coordinated via the same process group, allowing them to take turns smoothly — no extra hardware needed.

Previously, this wasn’t possible in TRL, which relied on vLLM as an external HTTP server. That changed with our [PR #3394](https://github.com/huggingface/trl/pull/3394), which added support for vLLM’s external launcher and true integration into the training process.

### What It Enables

- **Unified Execution**: By embedding vLLM in the same process group, both training and inference tasks can share the same GPUs, taking turns instead of waiting on each other. This reduces idle time and boosts overall efficiency.

- **Skip HTTP Communication**: No need for REST API calls or networking — vLLM runs inline with the training loop, avoiding overhead and latency.

- **Torchrun Compatibility**: Works seamlessly with `torchrun`, so it's easy to scale across nodes with minimal config changes.

- **TP and DP Support**: Compatible with Tensor Parallelism and Data Parallelism, making it suitable for large-scale training runs.

- **SPMD Execution Pattern**: Uses a Single Program, Multiple Data (SPMD) model, where each GPU runs its own instance of the engine in sync. Ideal for distributed multi-GPU, multi-node setups.

- **Simplified Deployment**: You no longer need to maintain a separate server script — vLLM is launched and controlled directly inside your training job.

- **Enhanced Throughput**: By avoiding idle GPUs and eliminating inter-process communication, the system delivers faster training and generation, especially important in online learning setups like GRPO.

- **Robust Inter-process Communication**: This is more robust because it avoids the complexity of setting up distributed process groups between independent processes, as required in server mode.

Thanks to this feature, co-located training and inference is no longer a hack — it’s now **first-class, scalable, and production-ready**.

## 🧩 Design: From Separate Servers to Shared GPUs

The shift from server TRL to co-located TRL is all about smarter GPU usage. The diagram below shows the difference:

![gpus-design](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-colocate/gpus-design.png)

### Server TRL Setup (Top Row)

In the server TRL setup, training and inference run on separate GPUs. For example:

- GPUs 0 through 2 are used for training.
- GPU 3 is fully dedicated to running vLLM as a separate server.

During training steps, **GPU 3 sits idle**.
During generation steps (inference), **GPUs 0–2 are idle** while GPU 3 generates outputs.

This leads to:

- Inefficient GPU usage, with devices frequently waiting on each other
- Extra GPUs provisioned solely for inference
- Increased cost and complexity

### Co-located TRL Setup (Bottom Row)

In contrast, the co-located TRL setup runs both training and vLLM on the **same GPUs**. Each GPU:

- Runs the training loop
- Launches a vLLM engine within the **same process**

Training and inference **take turns** using the GPU’s resources — no need for dedicated devices or separate processes.

This design:

- Reduces idle time
- Minimizes inter-process and HTTP communication
- Fully utilizes available GPU memory and compute
- Delivers **faster throughput** without increasing hardware requirements

## 🛠️ Implementation Notes

Instead of launching vLLM as a server, [the trainer now launches vLLM **in-process**](https://github.com/huggingface/trl/blob/fef915e36f12f759b384e4ab6f650208130aa232/trl/trainer/grpo_trainer.py#L647-L658) using the external launcher, as shown below:

```python
self.llm = LLM(
    model=model.name_or_path,
    tensor_parallel_size=args.vllm_tensor_parallel_size,
    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
    max_num_seqs=self.args.per_device_train_batch_size
        * self.vllm_tensor_parallel_size
        * self.args.gradient_accumulation_steps,
    max_model_len=self.max_prompt_length + self.max_completion_length,
    distributed_executor_backend="external_launcher",
    # Feed identical seed for tp groups to ensure sampling results are the same across workers
    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
)
```

Co-located vLLM respects the torch.distributed process group and rank structure. This allows vLLM to be initialized alongside training without conflict and makes TP/DP setups work seamlessly:

```python
if self.vllm_tensor_parallel_size > 1:
    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
        [
            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
        ]
    )
```

Co-located vLLM no longer relies on REST APIs — it runs directly in memory and communicates via native Python calls:

```python
if self.vllm_tensor_parallel_size > 1:
    orig_size = len(prompts_text)
    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
else:
    all_prompts_text = prompts_text

with profiling_context(self, "vLLM.generate"):
    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

if self.vllm_tensor_parallel_size > 1:
    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
    completion_ids = completion_ids[tp_slice]
```

To use this setup, simply set vllm_mode="colocate" in your GRPO configuration:

```python
training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

> Note: Depending on the model size and the overall GPU memory requirements for training, you may need to adjust the vllm_gpu_memory_utilization parameter in `GRPOConfig` to avoid underutilization or out-of-memory errors.

## 📊 Showcase: Co-located vs. Plain TRL Performance

To measure the impact of colocation, we ran a series of experiments comparing the traditional **server mode** (where vLLM runs on a separate GPU as a standalone server) with the new **co-locate mode** (where training and inference share the same GPUs).

In **server mode**, only 7 GPUs are used for training because 1 GPU is fully dedicated to the vLLM inference server.

In **co-locate mode**, all 8 GPUs are used for training — increasing the effective batch size by default.

To ensure a fair comparison, we **normalized throughput in server mode by a factor of 8/7**. This adjustment accounts for the greater training capacity in co-locate mode and allows us to compare the two setups under equal training conditions.

### Experiment 1: 1.5B Model — Varying Batch Sizes

- As the batch size increases, throughput improves in both setups.
- **Co-located setup reaches up to 1.43× speedup** at the largest batch size.
- Larger batches make better use of shared GPU memory in co-located mode.
![small-b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-colocate/small-b.png)

### Experiment 2: 1.5B Model — Varying Tensor Parallelism (TP)

- In the co-located setup, increasing TP **reduces performance**.
- More sharding introduces more communication overhead — which is **not ideal for smaller models**.
- **Takeaway**: For small models, avoid over-sharding in co-located mode.
![small-tp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-colocate/small-tp.png)

### Experiment 3: 7B Model — Varying Batch Sizes

- Again, co-located mode **scales better with batch size**.
- Gains reach **1.35× speedup** at the largest batch tested.
![med-b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-colocate/med-b.png)

### Experiment 4: 7B Model — Varying Tensor Parallelism (TP)

- Opposite trend from the 1.5B model.
- With 7B, **more TP improves throughput**, reaching up to **1.73× speedup**.
- **Larger models benefit from sharding** in co-located setups.
![med-tp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-colocate/med-tp.png)

## 📊 Scaling to 72B Model

When training large models like **Qwen2.5-Math-72B**, it's important to use the right strategies to make training efficient, scalable, and stable across many GPUs and nodes. In our setup, we combined **co-located vLLM** with several key optimizations to make this work efficiently.

### Sleep Mode in vLLM

When using co-located training, managing GPU memory is crucial so that both training and inference can run smoothly on the same devices. To support this, we added vLLM’s `sleep()` API into the GRPO training loop.

The `sleep()` function temporarily pauses the vLLM engine and frees up GPU memory. It supports two levels:

- **Level 1**: Unloads model weights from GPU (keeps them in CPU memory) and clears the KV cache.
  Useful when the same model will be reused soon.

- **Level 2**: Unloads both model weights and KV cache entirely.
  Best for scenarios where the model will change or won’t be reused right away.

In GRPO, the model is updated after every step — so we use **Level 2 sleep**.

Benefits of Level 2 sleep:

- **Maximizes free GPU memory** for training
- **Avoids memory contention** between training and generation
- Keeps colocation efficient, even for large models like Qwen2.5-72B

This small addition makes a **big difference** in enabling smooth and scalable co-located training.

### DeepSpeed Optimizations

To train large models like Qwen2.5-72B, we rely on **DeepSpeed ZeRO Stage 3**, the same setup used in plain TRL.

ZeRO helps scale large models by distributing memory across GPUs. Stage 3 goes further by partitioning:

- Model weights
- Gradients
- Optimizer states

This is essential for models that can’t fit on a single GPU. With ZeRO Stage 3, each GPU handles only a portion of the model.

Additional options we enable:

- `"offload_optimizer": {"device": "cpu"}`
  Moves optimizer states to CPU to free GPU memory — critical in co-located setups.

- `"overlap_comm": true`
  Enables communication overlap with computation, speeding up training.

- `"contiguous_gradients": true`
  Allocates gradients in a single memory block, improving memory access and reducing fragmentation.

These optimizations help **train 72B models efficiently**, and ensure colocation remains stable under tight memory constraints.

### Accelerate Integration

As recommended in TRL, we use **Accelerate**, a lightweight library that simplifies distributed training. It handles:

- Multi-GPU and multi-node job launching
- Data parallelism
- Gradient accumulation
- Distributed data loading

This makes the setup clean, scalable, and easy to maintain.

### Experiment 5: Qwen2.5-Math-72B — Throughput, Accuracy, and Benchmark Results

#### Throughput

Even with **4 fewer GPUs**, the **co-locate setup is ~1.26× faster** than plain TRL.
This highlights the effectiveness of smarter GPU sharing and memory cleanup using `sleep()`.
![72b-tput](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-colocate/72b-tput.png)

#### Reward Curve

Training reward plots for co-locate and plain setups are **nearly identical**, demonstrating that:

- **Co-located training preserves accuracy**
- There’s **no regression in model learning performance**
![blogpost_72b_rewards](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-colocate/blogpost_72b_rewards.png)

#### Math500 Benchmark

We evaluated three models: **Base model**, **Co-locate-trained model**, **Plain-trained model** on the Math500 benchmark. Both trained models **outperform the base**, and the **co-locate model performs on par** with the plain-trained model — confirming that colocation does not compromise downstream performance.
![blogpost_72b_math500](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-colocate/blogpost_72b_math500.png)

## 🎓 Challenges & Lessons Learned & next steps

Through our work on scaling GRPO training with co-located vLLM, we've faced several critical challenges and learned important lessons about efficiency, flexibility, and system design when training large models.

### Challenges

- **Tensor Parallelism Bug in vLLM ≥ 0.8.0.** Tensor Parallelism (TP) with external_launcher stopped working in vLLM version 0.8.0 and above. This was tracked under Issue [#15895](https://github.com/vllm-project/vllm/issues/15895). To identify the breaking point, we followed the approach described in this vLLM developer [blog post](https://blog.vllm.ai/2025/01/10/dev-experience.html), which provides wheels for every commit. After bisecting, we identified the breaking commit as [cc10281](https://github.com/vllm-project/vllm/commit/cc10281498fc2a6eb804274dcf22e6cb766f7aa7). The root cause was determinism — the newer versions required explicitly setting the random seed. Once the seed was set, the issue went away.

- **Level 2 Sleep Buffer Bug.** Initially, level 2 sleep didn’t work correctly when we tried to reload weights using load_weights. This issue was tracked in [Issue #16564](https://github.com/vllm-project/vllm/issues/16564). The problem was that model buffers (like running mean/var in BatchNorm) weren’t restored after waking up from sleep. The fix came with PR [#16889](https://github.com/vllm-project/vllm/pull/16889), which added logic to explicitly restore buffers when waking up from level 2 sleep. We now keep a copy of the original buffers and manually reapply them after loading new weights.

- **Segmentation Fault on Exit.** There’s still an open issue with vLLM sleep causing a segmentation fault at the end of training when closing processes. This was reported in Issue [#16993](https://github.com/vllm-project/vllm/issues/16993). This crash happens during shutdown but does not break training itself, so we were able to complete all demos and experiments shared in this blog. However, we’re waiting for an official fix before integrating sleep() fully into TRL upstream.

These challenges were not blockers, but they required careful debugging, version control, and a deeper understanding of how vLLM manages memory and parallelism under the hood.

### Lessons Learned

- Co-located inference dramatically improves GPU utilization. By allowing training and generation to share the same GPUs, we eliminate idle time and reduce hardware requirements — achieving higher throughput even with fewer GPUs.

- vLLM's sleep() feature is essential for large-scale colocation. It enables fine-grained control over memory usage, allowing training to fully reclaim GPU memory between generation steps — a key enabler for models like Qwen2.5-72B.

- DeepSpeed ZeRO Stage 3 is essential for training large models. It allows extremely large networks to fit into memory by distributing model weights, gradients, and optimizer states across multiple GPUs. In our experience, enabling contiguous_gradients helped reduce memory fragmentation, while offloading the optimizer to the CPU freed up critical GPU memory — both of which were especially helpful in colocated setups.

- Colocation is powerful but comes with trade-offs. It works best when GPU memory is carefully managed, often requiring manual tuning of memory usage parameters like vllm_gpu_memory_utilization. While it offers clear throughput benefits and reduces idle GPU time, colocation may not be ideal for models with tight memory budgets or when memory fragmentation is not well controlled. When done right, though, it unlocks significant efficiency gains.

- TP/DP compatibility, Accelerate, and torchrun support make deployment seamless. Despite the complexity of the underlying architecture, the entire system can be launched and scaled with standard distributed tools.

- Co-located training maintains model quality. Across multiple benchmarks (Math500, AIME24), co-located and plain setups produced comparable results, validating that performance isn’t sacrificed for efficiency.

## ✅ Conclusion

This blog post explored how co-locating vLLM with GRPO training unlocks significant efficiency gains when training large language models — including models as large as Qwen2.5-72B.

Traditionally, TRL only supported vLLM in server mode, which required separate processes and GPUs for inference, leading to wasted compute and idle time. With the introduction of vLLM’s external launcher and the colocation PR in TRL [PR #3394](https://github.com/huggingface/trl/pull/3394), we can now run training and inference within the same distributed process group, on the same GPUs, with full support for TP, DP, and Accelerate.

While challenges remain — such as version-specific vLLM bugs and edge cases such as with sleep() — the overall results show that co-located GRPO is a practical, scalable solution for training large models efficiently. We’re excited to continue refining this setup, integrating features like FSDP, and pushing the limits of large model training — making it faster, cheaper, and more accessible for everyone building the next generation of LLMs.

## ✅ Give It a Try!

Below is an example to try out GRPO training with co-located vLLM.

### 📄 `train_grpo_colocate.py`

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# Load dataset
dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

# Define training arguments
training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO",
    logging_steps=1,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_tensor_parallel_size=1,
    vllm_gpu_memory_utilization=0.3,
    max_prompt_length=512,
    max_completion_length=1024,
    max_steps=2,
    num_generations=4,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    push_to_hub=False,
    report_to=None
)

# Create and run the trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```
