---
title: "kernrl: Teaching LLMs to Write Fast GPU Kernels"
thumbnail: /blog/assets/kernrl/thumbnail.png
authors:
- user: Infatoshi
date: 2026-01-20
tags:
- openenv
- reinforcement-learning
- cuda
- triton
- gpu
- grpo
---

# kernrl: Teaching LLMs to Write Fast GPU Kernels

What if we could train language models to write optimized GPU code? Not just syntactically correct code, but kernels that actually run faster than PyTorch's defaults?

That's the goal of **kernrl** - an RL environment where agents learn to optimize GPU kernels through trial and error, receiving real performance feedback from actual hardware.

## The Problem: GPU Programming is Hard

Writing efficient GPU code requires understanding memory hierarchies, thread synchronization, and hardware-specific optimizations. Even experienced engineers spend significant time tuning kernels for different architectures.

Consider a simple softmax operation. PyTorch's implementation works, but a hand-tuned Triton kernel can be 2-5x faster by:
- Fusing the max, subtract, exp, sum, and divide operations
- Using efficient memory access patterns
- Avoiding unnecessary global memory round-trips

```python
# PyTorch baseline - multiple kernel launches
def softmax(x):
    max_val = x.max(dim=-1, keepdim=True)
    x = x - max_val
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# Triton kernel - single fused operation
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    softmax_output = numerator / tl.sum(numerator, axis=0)

    tl.store(output_ptr + row_idx * n_cols + col_offsets, softmax_output, mask=mask)
```

## Enter kernrl

kernrl is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that frames GPU kernel optimization as an RL problem:

- **State**: A PyTorch reference implementation + GPU info
- **Action**: CUDA/Triton kernel code
- **Reward**: Based on compilation, correctness, and speedup

The environment evaluates submitted kernels on real GPU hardware, providing concrete feedback:

```python
from kernrl import kernrl_env, KernelAction

env = kernrl_env(base_url="http://localhost:8000")
obs = env.reset(problem_id="L1_23_Softmax")

# Submit a kernel
result = env.step(KernelAction(code=triton_kernel_code))

print(f"Compiled: {result.observation.compilation_success}")
print(f"Correct: {result.observation.correctness_pass}")
print(f"Speedup: {result.observation.speedup}x")
```

## 89 Problems Across 10 Difficulty Levels

kernrl includes a diverse problem set spanning from basic operations to cutting-edge architectures:

| Level | Category | Examples |
|-------|----------|----------|
| 1 | Simple Ops | matmul, softmax, conv2d, layernorm |
| 2 | Fused Ops | matmul+GELU+softmax, conv+batchnorm |
| 3 | Attention | Vision attention, causal attention, transformer blocks |
| 4 | Novel Layers | DeepSeek MLA, MoE, GQA, FP8 matmul, INT4 GEMM |
| 5 | Scientific | N-body simulation, stencils, sparse matrix ops |
| 6 | Graphics | Ray tracing, histogram, bilateral filter |
| 7 | Signal | FFT, convolution, median filter |
| 8 | Video | Motion estimation, optical flow, deblocking |
| 9 | Primitives | Prefix scan, radix sort, stream compaction |
| 10 | Cryptography | SHA-256, AES, ChaCha20 |

Level 4 is particularly interesting - it includes architectures like DeepSeek's Multi-head Latent Attention and Mixture of Experts that weren't in most training data, testing whether models can truly reason about kernel optimization rather than memorize solutions.

## Training with GRPO

We use TRL's GRPOTrainer with a custom rollout function that interacts with the kernrl environment:

```python
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

def rollout_func(prompts, trainer):
    # Generate kernel code
    outputs = generate_rollout_completions(trainer, prompts)

    # Evaluate in environment
    env_rewards = []
    for completion in outputs:
        code = extract_code(completion)
        result = env.step(KernelAction(code=code))
        env_rewards.append(compute_reward(result))

    return {
        "prompt_ids": [...],
        "completion_ids": [...],
        "logprobs": [...],
        "env_reward": env_rewards,
    }

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    reward_funcs=[reward_from_env],
    rollout_func=rollout_func,
    args=GRPOConfig(use_vllm=True),
)
trainer.train()
```

The reward structure encourages incremental progress:
- **+0.1** for successful compilation
- **+0.3** for correctness (output matches reference within tolerance)
- **+0.3** for beating the baseline
- **+0.0 to +0.6** bonus scaled by log2(speedup)

This means a model can learn even before achieving speedups - first learn to write valid code, then correct code, then fast code.

## Why This Matters

### For AI Research
GPU kernel optimization is a well-defined domain with clear metrics (correctness + speed), making it ideal for studying:
- How LLMs reason about low-level code optimization
- Whether RL can teach models skills not present in training data
- Transfer learning between GPU architectures (H100 vs B200)

### For Practical Applications
As AI models grow, inference costs dominate. A model that can automatically optimize kernels could:
- Reduce serving costs by finding faster implementations
- Adapt to new hardware without manual tuning
- Enable efficient deployment on edge devices

### For the Community
kernrl provides:
- A standardized benchmark for kernel optimization capabilities
- Real hardware evaluation (not just syntax checking)
- Integration with the broader OpenEnv ecosystem

## Try It Yourself

**HuggingFace Space**: [huggingface.co/spaces/Infatoshi/kernrl](https://huggingface.co/spaces/Infatoshi/kernrl)

**Training Notebook**: [huggingface.co/Infatoshi/kernrl-training](https://huggingface.co/Infatoshi/kernrl-training)

**OpenEnv PR**: [github.com/meta-pytorch/OpenEnv/pull/308](https://github.com/meta-pytorch/OpenEnv/pull/308)

To run locally with GPU:

```bash
# Clone OpenEnv
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv/envs/kernrl

# Install
pip install -e .

# Start server
uvicorn kernrl.server.app:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t kernrl -f server/Dockerfile .
docker run --gpus all -p 8000:8000 kernrl
```

## What's Next

We're excited to see what the community builds with kernrl:

- **Curriculum learning**: Start with L1, progressively add harder problems
- **Multi-turn optimization**: Let models iterate based on profiling feedback
- **Architecture-specific training**: Specialize models for H100 vs B200
- **Novel reward shaping**: Incorporate memory bandwidth, occupancy metrics

The code is open source and contributions are welcome. Whether you're interested in RL, GPU programming, or both - we'd love to see what optimizations your models can discover.

---

*kernrl was built for the OpenEnv Challenge. Special thanks to the Meta PyTorch team for the OpenEnv framework and Hugging Face for TRL.*
