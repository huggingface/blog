---
title: "Ulysses Sequence Parallelism: Training with Million-Token Contexts"
thumbnail: /blog/assets/ulysses/thumbnail.png
authors:
- user: kashif
- user: stas
---

# Ulysses Sequence Parallelism: Training with Million-Token Contexts

Training large language models on long sequences has become essential for building capable AI systems. As models are increasingly used for tasks like document analysis, code understanding, complex reasoning, and RAG workloads, the need to process sequences of hundreds of thousands—or even millions—of tokens has grown dramatically. To put this in perspective, an average book is roughly 250k tokens, so training on multi-document contexts or book-length inputs requires handling sequences well beyond what fits on a single GPU. However, training with such long contexts presents significant memory challenges: the attention computation scales quadratically with sequence length, quickly exceeding GPU memory for contexts beyond tens of thousands of tokens.

Ulysses Sequence Parallelism (part of the [Arctic Long Sequence Training (ALST) protocol](https://huggingface.co/papers/2506.13996)) provides an elegant solution by distributing the attention computation across multiple GPUs through attention head parallelism. In this post, we'll explore how Ulysses works and how it's been integrated across the Hugging Face ecosystem—from Accelerate to the Transformers Trainer and TRL's SFTTrainer.

## Contents

- [The Challenge of Long Sequence Training](#the-challenge-of-long-sequence-training)
- [How Ulysses Works](#how-ulysses-works)
- [Integration with Accelerate](#integration-with-accelerate)
- [Integration with Transformers Trainer](#integration-with-transformers-trainer)
- [Integration with TRL's SFTTrainer](#integration-with-trl-sfttrainer)
- [Comparing Ulysses and Ring Attention](#comparing-ulysses-and-ring-attention)
- [Best Practices](#best-practices)
- [Benchmarks](#benchmarks)
- [Resources](#resources)

## The Challenge of Long Sequence Training

The attention mechanism in transformers scales quadratically with sequence length. For a sequence of length  \\( n \\), standard attention requires  \\( O(n^2) \\) FLOPs and  \\( O(n^2) \\) memory to compute and store the attention score matrix. Optimized implementations like [FlashAttention](https://huggingface.co/papers/2205.14135) reduce the **memory** to  \\( O(n) \\) by tiling the computation and never materializing the full attention matrix—but the  \\( O(n^2) \\) compute remains. For very long sequences (32k+ tokens), even with FlashAttention, training still pushes the limits of single-GPU memory.

Consider these scenarios where long-context training is essential:
- **Document understanding**: Processing entire books, legal documents, or research papers
- **Code analysis**: Understanding large codebases with multiple interconnected files
- **Reasoning tasks**: Models that "think" step-by-step may generate thousands of tokens during inference
- **Retrieval-augmented generation**: Incorporating many retrieved passages into the context

Traditional data parallelism doesn't help here—each GPU still needs to process the full sequence inside the attention block. We need a way to split the sequence itself across multiple devices.

## How Ulysses Works

Ulysses Sequence Parallelism (SP), introduced in the [DeepSpeed Ulysses paper](https://huggingface.co/papers/2309.14509), takes a clever approach: in addition to splitting on the sequence dimension, it also partitions the attention heads across GPUs.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ulysses/ulysses_overview.png" alt="Ulysses Sequence Parallelism Overview">
  <figcaption>Ulysses splits input sequences along the sequence dimension and uses all-to-all communication to exchange key-value pairs, enabling each GPU to compute a subset of attention heads. (<b><i>Source: <a href="https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/">Snowflake Engineering Blog</a></i></b>)</figcaption>
</figure>

Here's how it works:

1. **Sequence Sharding**: The input sequence is split along the sequence dimension across  \\( P \\) GPUs. Each GPU  \\( i \\) holds tokens  \\( [i \cdot n/P, (i+1) \cdot n/P) \\).

2. **QKV Projection**: Each GPU computes the query, key, and value projections for its local sequence chunk.

3. **All-to-All Communication**: An all-to-all collective operation redistributes the data so that each GPU holds *all* sequence positions after the projections, but only for a subset of attention heads.

4. **Local Attention**: Each GPU computes attention for its assigned heads using standard attention mechanisms (FlashAttention or SDPA).

5. **All-to-All Communication**: Another all-to-all operation reverses the redistribution, returning to sequence-sharded format.

6. **Output Projection**: Each GPU computes the output projection for its local sequence chunk.

The key insight is that attention heads are independent—each head can be computed separately. By trading sequence locality for head locality, Ulysses enables efficient parallelization with relatively low communication overhead.

### Communication Complexity

Ulysses requires two all-to-all operations per attention layer, with total communication volume of \\( O(n \cdot d / P) \\) per GPU, where:
- \\( n \\) is the sequence length
- \\( d \\) is the hidden dimension
- \\( P \\) is the parallelism degree

Ring Attention has the same asymptotic communication volume — \\( O(n \cdot d) \\) total across \\( P-1 \\) ring steps — but pays it in sequential point-to-point transfers rather than a single collective. In practice, Ulysses benefits from lower latency on high-bandwidth interconnects (NVLink, InfiniBand) because all-to-all can exploit full bisectional bandwidth in a single step, whereas Ring Attention serializes over \\( P-1 \\) hops.

## Integration with Accelerate

Accelerate provides the foundation for Ulysses sequence parallelism through its [`ParallelismConfig`](https://huggingface.co/docs/accelerate/concept_guides/sequence_parallelism#alstulysses-sp-backend-configuration) class and DeepSpeed integration.

### Configuration

```python
from accelerate import Accelerator
from accelerate.utils import ParallelismConfig, DeepSpeedSequenceParallelConfig

parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=4,  # Split across 4 GPUs
    dp_shard_size=1,  # Must satisfy: dp_replicate × dp_shard × sp_size = num_processes
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length=None,  # None for variable-length sequences
        sp_seq_length_is_variable=True,
        sp_attn_implementation="flash_attention_2",  # or "sdpa"
    ),
)

accelerator = Accelerator(parallelism_config=parallelism_config)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `sp_size` | Number of GPUs for sequence parallelism |
| `sp_backend` | Must be `"deepspeed"` for Ulysses |
| `sp_seq_length_is_variable` | Set to `True` for varying sequence lengths across batches |
| `sp_attn_implementation` | `"flash_attention_2"`, `"flash_attention_3"`, or `"sdpa"` |

### Using the Accelerator

When you call `accelerator.prepare()`, Ulysses is automatically set up:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# This registers the model with Ulysses and wraps the dataloader
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

The `prepare()` call:
1. Registers the model with DeepSpeed's `UlyssesSPAttentionHF`
2. Wraps the dataloader with `UlyssesSPDataLoaderAdapter` to handle sequence sharding
3. Automatically injects `shift_labels` for correct loss computation

### Loss Aggregation

With Ulysses, each GPU computes loss on different parts of the sequence. The losses must be aggregated properly, weighted by the number of valid tokens per rank. If you're using the Transformers `Trainer` or TRL's `SFTTrainer`, this is handled automatically—the code below is only needed when writing a custom Accelerate training loop:

```python
sp_size = parallelism_config.sp_size
if sp_size > 1:
    from deepspeed.utils import groups

    sp_group = groups._get_sequence_parallel_group()

    # Gather losses and token counts from all SP ranks
    losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
    good_tokens = (batch["shift_labels"] != -100).view(-1).sum()
    good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)

    # Weighted aggregation
    total_loss = sum(
        losses_per_rank[i] * good_tokens_per_rank[i]
        for i in range(sp_size)
        if good_tokens_per_rank[i] > 0
    )
    loss = total_loss / max(sum(good_tokens_per_rank), 1)

accelerator.backward(loss)
```

> [!NOTE]
> The loss aggregation ensures correct gradients when tokens are unevenly distributed across ranks (e.g., when some ranks contain only padding or masked out prompt tokens).

> [!TIP]
> Both Ulysses and Ring Attention use `position_ids` instead of `attention_mask` for causal masking during training. A 4D attention mask at these sequence lengths would be just as prohibitive as the attention scores themselves—at 128k tokens, that's another ~1TB tensor. Position IDs achieve the same causal behavior with  \\( O(n) \\) memory instead of  \\( O(n^2) \\). During evaluation/inference, DeepSpeed's SP attention layer can bypass the SP operations entirely (via `disable_in_eval`) and fall back to the model's default attention implementation.

## Integration with Transformers Trainer

The Transformers [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) provides seamless Ulysses integration through [`TrainingArguments.parallelism_config`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.parallelism_config). It handles all the SP-specific details automatically—dataloader wrapping, sequence sharding, and loss aggregation—so you don't need to write any of the custom loss code shown above.

### Configuration

Just pass the same `parallelism_config` from above into `TrainingArguments`:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    parallelism_config=parallelism_config,  # same ParallelismConfig as above
    per_device_train_batch_size=1,
)
```

### What the Trainer Handles Automatically

1. **Dataloader Wrapping**: After model preparation, the Trainer wraps the dataloader with `UlyssesSPDataLoaderAdapter`

2. **Loss Computation**: The `compute_loss` method detects SP mode and routes to specialized `_deepspeed_sp_compute_loss` which handles:
   - Gathering losses across SP ranks
   - Computing valid token counts per rank
   - Weighted loss aggregation

3. **Batch Size Calculation**: The effective data parallel world size accounts for SP:
   ```python
   dp_world_size = world_size // tp_size // cp_size // sp_size
   ```

4. **Dataloader Length Adjustment**: Training step calculations are adjusted for SP's effect on iteration count

### Launch Command

Use an accelerate config file or command-line arguments:

```bash
accelerate launch \
    --config_file deepspeed_ulysses.yaml \
    train.py \
    --per_device_train_batch_size 1
```

## Integration with TRL SFTTrainer

TRL's [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) builds on the Transformers Trainer and adds specific optimizations for supervised fine-tuning with long sequences.

### Configuration

```python
from trl import SFTConfig, SFTTrainer
from accelerate.utils import ParallelismConfig, DeepSpeedSequenceParallelConfig

parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=2,
    dp_shard_size=2,  # 2D parallelism: SP × DP = 4 GPUs
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length_is_variable=True,
        sp_attn_implementation="flash_attention_2",
    ),
)

training_args = SFTConfig(
    ...,
    parallelism_config=parallelism_config,
    max_length=32768,
    pad_to_multiple_of=2,  # Must equal sp_size
    per_device_train_batch_size=1,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### Key SFTConfig Parameters for Ulysses

| Parameter | Description |
|-----------|-------------|
| `pad_to_multiple_of` | Must equal `sp_size` to ensure sequence divisibility |
| `max_length` | Global sequence length (before splitting across GPUs) |
| `packing` | Works well with SP — packing reduces padding waste, especially for variable-length sequences |

### Accelerate Config File

Create `alst_ulysses_4gpu.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: bf16
num_processes: 4
deepspeed_config:
  zero_stage: 3
  seq_parallel_communication_data_type: bf16
parallelism_config:
  parallelism_config_sp_size: 2
  parallelism_config_sp_backend: deepspeed
  parallelism_config_dp_shard_size: 2
  parallelism_config_sp_seq_length_is_variable: true
  parallelism_config_sp_attn_implementation: flash_attention_2
```

### Complete Training Command

```bash
accelerate launch --config_file alst_ulysses_4gpu.yaml \
    trl/scripts/sft.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --dataset_name trl-lib/Capybara \
    --max_length 32768 \
    --packing \
    --pad_to_multiple_of 2 \
    --per_device_train_batch_size 1
```

### Shift Labels Handling

The SFTTrainer automatically handles pre-shifted labels when Ulysses is enabled:

```python
# When using SP, labels are pre-shifted by the dataloader adapter
# The trainer detects this and uses shift_labels directly
labels = inputs["labels"] if "shift_labels" not in inputs else None

# Loss computation uses the pre-shifted labels
if "shift_labels" in inputs:
    shift_logits = outputs.logits.contiguous()
    shift_labels = inputs["shift_labels"]
else:
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
```

## Comparing Ulysses and Ring Attention

Both Ulysses and Ring Attention enable long-context training, but they have different characteristics:

| Aspect | Ulysses (DeepSpeed) | Ring Attention (FSDP2) |
|--------|---------------------|------------------------|
| **Parallelism Method** | Attention head partitioning | Ring-based KV exchange |
| **Backend** | DeepSpeed ZeRO | PyTorch FSDP2 |
| **Attention Support** | FlashAttention 2/3, SDPA | SDPA only |
| **Communication** | Two `all-to-all`s per layer | P2P ring communication |
| **Comm volume per GPU** | O(total_seq x hidden / sp_size) | O(total_seq x hidden) |
| **Sequence Divisibility** | `sp_size` | `cp_size * 2` |
| **Num Head Constraint** | `num_heads >= sp_size` | None |

### When to Choose Ulysses vs Ring Attention

Since switching between the two only requires changing the accelerate config, we recommend trying both and comparing performance and memory usage on your specific setup. The main constraint is that Ulysses requires `num_heads >= sp_size`, while Ring Attention has no such limitation.

## Best Practices

### 1. Sequence Length Divisibility

Always ensure your sequence length is divisible by `sp_size`:

```python
training_args = SFTConfig(
    pad_to_multiple_of=4,  # For sp_size=4
    max_length=32768,  # Must be divisible by 4
)
```

### 2. Use Flash Attention

Flash Attention 2 provides cleaner output and better performance than SDPA:

```python
parallelism_config = ParallelismConfig(
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_attn_implementation="flash_attention_2",
    ),
)
```

Use Flash Attention 3 for Hopper and look out for Flash Attention 4 release for Blackwell (FA2 on Blackwell is quite slow).

### 3. Combine with DeepSpeed ZeRO

For very large models, combine Ulysses with ZeRO Stage 3:

```yaml
deepspeed_config:
  zero_stage: 3
  offload_optimizer:
    device: cpu
```

If the model is huge, you can offload the params as well by adding to the above:

```yaml
  offload_param:
    device: cpu
```

### 5. Use memory fragmentation-friendly PyTorch allocator

This environment variable will allow for a longer sequence length:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 6. 2D Parallelism Configuration

Balance SP and DP for your GPU count:

| GPUs | `sp_size` | `dp_shard_size` | Use Case |
|------|-----------|-----------------|----------|
| 4 | 2 | 2 | Balanced throughput and sequence length |
| 4 | 4 | 1 | Maximum sequence length |
| 8 | 2 | 4 | Large-scale with longer sequences |
| 8 | 4 | 2 | Emphasis on sequence length |

Remember: `dp_replicate_size × dp_shard_size × sp_size = num_processes`

### 7. Liger-Kernel

If your desired model architecture is supported by [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), it is fully compatible with Ulysses SP and can be enabled with a single flag:

```python
training_args = SFTConfig(
    use_liger_kernel=True,
)
```

The main memory saving comes from `FusedLinearCrossEntropy` which avoids materializing the full logits tensor during loss calculation. The savings grow with longer sequences where the logits tensor is larger.

Additionally, you can enable [`TiledMLP`](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/#tiled-mlp-computation) to further extend sequence length — like `FusedLinearCrossEntropy`, it saves working memory by tiling large matrix operations.

### 8. Token Distribution Across Ranks

You don't need to worry about manually balancing tokens across SP ranks—the loss aggregation code handles uneven distributions gracefully (including ranks with zero valid tokens). With random batching over a reasonably sized dataset, the distribution evens out statistically over training.

## Benchmarks

To quantify the benefits of Ulysses SP, we trained [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on the [Gutenberg English](https://huggingface.co/datasets/sedthh/gutenberg_english) streaming dataset using TRL's SFTTrainer. All experiments ran on H100 80GB GPUs with DeepSpeed ZeRO-3, CPU optimizer offloading, gradient checkpointing, and flash-attn2 as the attention backend.

### Setup

| Config | GPUs | SP | DP | Seq Length | Grad Acc | Global Batch |
|--------|------|----|----|-----------|----------|-------------|
| Baseline | 1 | 1 | 1 | 8,192 | 8 | 8 |
| SP=4 | 4 | 4 | 1 | 8,192 | 8 | 8 |
| SP=4 (32K) | 4 | 4 | 1 | 32,768 | 8 | 8 |
| SP=4 (64K) | 4 | 4 | 1 | 65,536 | 8 | 8 |
| SP=4 (96K) | 4 | 4 | 1 | 98,304 | 8 | 8 |

The benchmark runs in the table above use the same global batch size (8 micro-batches), cosine learning-rate schedule, and seed, so those benchmark loss curves are directly comparable.

### Loss Curve Matching Diagnostics (4 GPU)

To verify SP-vs-DP loss equivalence, we ran controlled 4-GPU A/B experiments with identical seed, model, optimizer, learning-rate schedule, and data order.

#### Methodology for Fair DP vs SP Comparison

Compared setups:

- `DP=4, SP=1, GAS=1` (baseline)
- `DP=1, SP=4, GAS=4` (Ulysses SP)

For fair comparison, `GAS` must scale with `SP`:

- Ulysses SP splits the sequence across `SP` ranks, so each SP rank sees roughly `1/SP` of the sequence tokens per micro-step.
- If `GAS` is unchanged, each optimizer step in SP aggregates fewer total tokens than the DP baseline.
- Setting `GAS=SP` keeps effective tokens per optimizer step matched:
  - DP tokens/step: `dp_world_size * micro_batch * seq_len * GAS = 4 * B * L * 1`
  - SP tokens/step: `dp_world_size * micro_batch * (L/SP) * GAS * SP_ranks = 1 * B * (L/4) * 4 * 4 = 4 * B * L`

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ulysses/gutenberg_canonical_loss_dp4_vs_sp4.png" alt="Canonical loss on Gutenberg for DP=4 vs SP=4">
  <figcaption>On Gutenberg text (20 steps), canonical loss matches within logging precision between <code>DP=4,SP=1,GAS=1</code> and <code>DP=1,SP=4,GAS=4</code>.</figcaption>
</figure>

Measured over 20 steps on 4 GPUs in controlled equivalence harnesses:

| Harness | Metric | DP vs SP setting | Mean abs diff | Max abs diff |
|--------|--------|-------------------|---------------|--------------|
| `Trainer` | `loss` | DP=4, SP=1 vs DP=1, SP=4 | 0.0054 | 0.0131 |
| `SFTTrainer` | logged `loss` | DP=4, SP=1 vs DP=1, SP=4 | 0.0811 | 0.0812 |
| `SFTTrainer` | canonical NLL | DP=4, SP=1 vs DP=1, SP=4 | 0.000004 | 0.000005 |

Takeaway: under matched token budget, SP and non-SP match on canonical token-normalized loss. The remaining difference is in trainer-reported logging (`loss`), not in the underlying cross-entropy objective.

### Memory Reduction

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ulysses/memory_usage.png" alt="Peak GPU Memory per Rank">
  <figcaption>SP=4 reduces per-GPU memory by 3.3x at the same sequence length, enabling training at up to 96K tokens on 4× H100 80GB. At 128K, the model OOMs.</figcaption>
</figure>

| Config | Seq Length | Peak Memory | Notes |
|--------|-----------|-------------|-------|
| Baseline (1 GPU) | 8K | 76.4 GB | Near H100 limit |
| SP=4 (4 GPU) | 8K | 23.4 GB | **3.3x reduction** |
| SP=4 (4 GPU) | 32K | 35.0 GB | 4x longer, 1.5x more memory |
| SP=4 (4 GPU) | 64K | 50.5 GB | 8x longer, fits comfortably |
| SP=4 (4 GPU) | 96K | 66.0 GB | **12x longer**, still fits |
| SP=4 (4 GPU) | 128K | OOM | Exceeds 80 GB limit |

Without SP, the baseline already uses 76.4 GB at just 8K tokens — barely fitting on an H100. With SP=4, the same 8K sequence uses only 23.4 GB per GPU, freeing up memory to scale sequence length. The memory savings compound: at 96K tokens (12x longer than the baseline), peak memory is 66 GB — still within the H100's capacity. At 128K, the model requires ~81 GB and OOMs, establishing the practical limit for this configuration.

### Throughput

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ulysses/throughput.png" alt="Training Throughput">
  <figcaption>Longer sequences with SP process dramatically more tokens per second. SP=4 at 64K achieves 3.7x the throughput of the baseline.</figcaption>
</figure>

| Config | Seq Length | Tokens/s | vs Baseline |
|--------|-----------|---------|-------------|
| Baseline (1 GPU) | 8K | 3,633 | — |
| SP=4 (4 GPU) | 8K | 3,933 | ~1x |
| SP=4 (4 GPU) | 32K | 7,733 | **2.1x** |
| SP=4 (4 GPU) | 64K | 13,396 | **3.7x** |

At the same sequence length (8K), SP=4 has comparable throughput to the single-GPU baseline — the all-to-all communication overhead is minimal on NVLink-connected GPUs. The real benefit comes from longer sequences: each training step processes proportionally more tokens, so throughput scales with sequence length. At 64K, SP=4 processes 13,396 tokens/second — 3.7x the baseline.

> [!NOTE]
> These results use only 4 GPUs with SP=4. With 8 GPUs (SP=8), you can push to even longer sequences — up to 256K+ tokens — or use 2D parallelism (SP=4, DP=2) to combine long-context training with data-parallel throughput.

## Requirements

- HF Accelerate: `deepspeed>=0.18.1 accelerate>=1.12`
- HF Trainer: `deepspeed>=0.18.1 accelerate>=1.12 transformers>=5.0`
- HF TRL: `deepspeed>=0.18.1 accelerate>=1.12 transformers>=5.0 trl>=0.18.0`

In all of the above ideally use `flash_attention_2` for Ampere GPUs, and `flash_attention_3` for Hopper GPUs.

## Resources

### Documentation
- [Accelerate: Context Parallelism Guide](https://huggingface.co/docs/accelerate/concept_guides/context_parallelism)
- [TRL: Distributing Training](https://huggingface.co/docs/trl/distributing_training)
- [DeepSpeed Sequence Parallelism](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism)

### Examples
- [Accelerate ALST Example](https://github.com/huggingface/accelerate/tree/main/examples/alst_ulysses_sequence_parallelism)
- [TRL Accelerate Configs](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs)

### Papers
- [Arctic Long Sequence Training: Scalable And Efficient Training For Multi-Million Token Sequences](https://huggingface.co/papers/2506.13996)
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://huggingface.co/papers/2309.14509)


### Related Blog Posts
- [Accelerate ND-Parallel: A Guide to Efficient Multi-GPU Training](https://huggingface.co/blog/accelerate-nd-parallel)
- [Understanding Ulysses and Ring Attention](https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention)
- [Enabling Long-Context Training with Sequence Parallelism in Axolotl](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl)
