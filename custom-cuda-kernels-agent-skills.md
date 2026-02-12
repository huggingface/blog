---
title: "Custom Kernels for All from Codex and Claude" 
thumbnail: /blog/assets/custom-cuda-kernels/meme.png
authors:
- user: burtenshaw
- user: sayakpaul
- user: ariG23498
- user: evalstate  
---

<!-- TODO: PLEASE ADD YOURSELF TO THE AUTHORS LIST -->

# Custom Kernels for All from Codex and Claude

![oprah custom cuda kernels](assets/custom-cuda-kernels/meme.png)

tl;dr: We built an agent skill that teaches coding agents how to write production CUDA kernels. Then we pointed Claude and Codex at two real targets: a **diffusers** pipeline and a **transformers** model. The agents produced working kernels for both, with correct PyTorch bindings and benchmarks, end to end.

Writing CUDA kernels is hard. Writing CUDA kernels that correctly integrate with `transformers` and `diffusers` is harder. There are architecture-specific memory access patterns, vectorization strategies, warp shuffle reductions, and a dozen integration pitfalls that trip up even experienced developers. It is exactly the kind of specialized, high-stakes problem where agent skills shine.

We gave coding agents the domain knowledge they need, like which GPU architecture to target, how to structure a kernel-builder project, when to use shared memory versus registers, and how to write PyTorch bindings. The agents did the rest. If you have used the [LLM training skill](https://huggingface.co/blog/hf-skills-training) or read [We Got Claude to Teach Open Models](https://huggingface.co/blog/upskill), the pattern will feel familiar: package domain expertise into a skill, point the agent at a problem, and let it work.

## Why a skill for kernels?

The [Kernel Hub](https://huggingface.co/blog/hello-hf-kernels) solved the distribution of custom hardware kernels. You can load pre-compiled kernels from the Hub with a single `get_kernel` call. No builds, no flags. However, someone still needs to **write the kernels**. That is the gap this skill fills.

CUDA kernel development has a brutal surface area:

- Hardware-specific optimization guides for each generation of GPU. H100, A100, and T4 each have different compute capabilities, shared memory sizes, and bandwidth profiles  
- In Libraries, `diffusers` and `transformers` have different module hierarchies, normalization conventions, and integration patterns. Custom kernels need to be registered in PyTorch for `torch.compile` to recognize.  
- For distribution, kernels can depend on CUDA, Pytorch, and Python versions creating massive environment matrices.

This is domain knowledge that gets lost in documentation tabs and Stack Overflow answers. An agent skill packages it into context that loads on demand.

First, let's show how to use the skill right away, then we'll dive into the details of how we benchmarked the kernels.

## Installing the skill

The skill ships with the `kernels` library. Install it into your coding agent with a single command:

```shell
pip install kernels
kernels skills add cuda-kernels --claude
```

This drops the skill into `.claude/skills/cuda-kernels/` where Claude Code and Cursor pick it up automatically. For other agents:

```shell
# Codex
kernels skills add cuda-kernels --codex

# OpenCode
kernels skills add cuda-kernels --opencode

# Custom destination
kernels skills add cuda-kernels --dest ./my-agent/skills/

# Install globally (available across all projects)
kernels skills add cuda-kernels --global

# Overwrite an existing installation
kernels skills add cuda-kernels --claude --force
```

Once installed, prompt your agent:

```
Build a vectorized RMSNorm kernel for H100 targeting the Qwen3-8B model in transformers.
```

Or, you can go for something more open-ended:

```
Build an optimized attention kernel for H100 targeting the Qwen3-8B model in transformers. Benchmark it against the PyTorch baseline and validate improvements in end-to-end performance.
```

The agent can read the skill, select the right architecture parameters, generate the CUDA source, write the PyTorch bindings, set up `build.toml`, and create a benchmark script.

If you're working on more complex kernels, or architecture-specific optimizations, that aren't covered in the skill, then the skill supplies the fundamental building blocks and patterns to get you started. We are also open to contributions on the [skill itself](https://github.com/huggingface/kernels/tree/main/.docs/skills).

## What is in the skill

The skill is roughly **550 tokens** of structured guidance plus reference scripts, GPU optimization guides, troubleshooting docs, and complete working examples. Agentic coding tools like Codex and Claude can read this and produce a working kernel project.

It covers:

- NVIDIA GPU Architecture-aware optimization for H100, A100, and T4 (compute capabilities, memory bandwidth, shared memory sizes, block sizing)  
- Integration patterns for both `diffusers` and `transformers`, including the pitfalls specific to each library  
- Kernel templates with vectorized memory access patterns for BF16, FP16, and FP32  
- Benchmarking workflows for both isolated kernel micro-benchmarks and end-to-end pipeline comparisons  
- HuggingFace Kernel Hub integration via `get_kernel` for loading community kernels

```
.claude/skills/cuda-kernels/
├── SKILL.md                              # Main instructions (~550 tokens)
├── scripts/
│   ├── benchmark_example.py              # End-to-end benchmark template
│   ├── benchmark_rmsnorm.py              # Isolated kernel micro-benchmark
│   ├── ltx_kernel_injection_example.py   # Diffusers integration pattern
│   ├── transformers_injection_example.py # Transformers integration pattern
│   └── huggingface_kernels_example.py    # Kernel Hub integration
└── references/
    ├── diffusers-integration.md          # Diffusers guide with pitfalls
    ├── transformers-integration.md       # Transformers guide
    ├── huggingface-kernels-integration.md
    ├── h100-optimization-guide.md
    ├── a100-optimization-guide.md
    ├── t4-optimization-guide.md
    ├── kernel-templates.md
    └── troubleshooting.md
```

When an agent loads this, it gets everything it needs to go from "write me an RMSNorm kernel" to a buildable, benchmarkable project. It will grep and glob the skill to find the relevant files and directories. So it's important to structure the skill in a way that is easy to find.

The agent is instructed to generate kernels that conform to the templates in `references/kernel-templates.md` and produce a complete kernel project:

```
examples/your_model/
├── kernel_src/
│   └── rmsnorm.cu              # Vectorized CUDA kernel
├── torch-ext/
│   ├── your_kernels/__init__.py
│   └── torch_binding.cpp       # PyTorch C++ bindings
├── benchmark_rmsnorm.py        # Micro-benchmark script
├── build.toml                  # kernel-builder config
├── setup.py                    # pip install -e .
└── pyproject.toml
```

We tested this on two real targets.

## Benchmarking the kernels: Diffusers (LTX-Video on H100)

The agent built RMSNorm, RoPE 3D, GEGLU, and AdaLN kernels for [LTX-Video](https://huggingface.co/Lightricks/LTX-Video), a video generation pipeline from `diffusers`. The full example is at `examples/ltx_video/`. We optimized the RMSNorm kernel for H100. Both benchmarks were run on H100 80GB HBM3 at precision BFloat16.

If you want to check out the generated kernel, got to [this example](https://github.com/burtenshaw/kernel-skill/tree/main/examples/ltx_video)

### Isolated RMSNorm benchmark

First, we compare the isolated RMSNorm kernel performance against the PyTorch baseline. This is the main speedup in the optimized pipeline.

| Shape | Custom (ms) | PyTorch (ms) | Speedup |
| :---- | :---: | :---: | :---: |
| [1x1024x2048] | 0.039 | 0.064 | **1.64x** |
| [2x1024x2048] | 0.040 | 0.073 | **1.82x** |
| [4x1024x2048] | 0.052 | 0.093 | **1.78x** |
| [1x4096x2048] | 0.052 | 0.093 | **1.79x** |
| [2x4096x3072] | 0.102 | 0.209 | **2.04x** |
| [1x8192x2048] | 0.083 | 0.150 | **1.81x** |
| [4x4096x3072] | 0.173 | 0.393 | **2.26x** |

**Average speedup: 1.88x** and a bandwidth efficiency: 34.7% of H100 theoretical (3,350 GB/s)

### End-to-end video generation (49 frames, 30 steps, H100 80GB)

Next, we compare the end-to-end video generation performance of the optimized kernels against the baseline (no compile) and the `torch.compile` baseline.

| Configuration | Time (s) | it/s | Speedup |
| :---- | :---: | :---: | :---: |
| Baseline (no compile) | 2.87 | 12.58 | 1.00x |
| **Generated Optimized Kernels** | 2.70 | 13.52 | **1.06x** |
| Baseline + torch.compile | 2.14 | 19.05 | 1.34x |

RMSNorm accounts for ~5% of total compute in LTX-Video. The remaining time is spent in attention, linear projections, and VAE decode. The 6% end-to-end speedup from a single kernel type is consistent with that profile.

## Benchmarking the kernels: Transformers (Qwen3-8B on H100)

The agent built an RMSNorm kernel for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), a large language model from `transformers` with 65 RMSNorm modules across 32 layers. The full example is at `examples/qwen3_8b/`. We optimized the RMSNorm kernel for H100. Both benchmarks were run on H100 80GB HBM3 at precision BFloat16.

If you want to explore the kernel, check it out [here.](https://github.com/burtenshaw/kernel-skill/tree/main/examples/qwen3_8b)

### Isolated RMSNorm benchmark

Once again, we compare the isolated RMSNorm kernel performance against the PyTorch baseline. 

| Shape | Custom (ms) | PyTorch (ms) | Speedup |
| :---- | :---: | :---: | :---: |
| [1x128x4096] | 0.040 | 0.062 | **1.58x** |
| [1x512x4096] | 0.038 | 0.064 | **1.69x** |
| [1x1024x4096] | 0.037 | 0.071 | **1.90x** |
| [1x2048x4096] | 0.045 | 0.091 | **2.03x** |
| [1x4096x4096] | 0.071 | 0.150 | **2.12x** |
| [4x512x4096] | 0.056 | 0.093 | **1.67x** |
| [8x256x4096] | 0.045 | 0.092 | **2.06x** |
| [1x8192x4096] | 0.109 | 0.269 | **2.47x** |

**Average speedup: 1.94x** and a bandwidth efficiency: 22.3% of H100 theoretical (3,350 GB/s)

Speedup scales with sequence length: 1.58x at 128 tokens, 2.47x at 8192 tokens. For long-context inference, the custom kernel roughly halves RMSNorm latency.

## Publishing your kernel to the Hub

The agent gives you a working kernel. The [Kernel Hub](https://huggingface.co/kernels-community) lets you share it so anyone can load it without compilation. Here is the full path from agent output to published kernel.

### 1. Verify the project structure

The agent produces a project that already follows the [kernel-builder](https://huggingface.co/docs/kernels/en/builder/writing-kernels) layout:

```
your_kernel/
├── build.toml               # Build configuration
├── kernel_src/
│   └── rmsnorm.cu           # CUDA kernel source
└── torch-ext/
    ├── torch_binding.cpp    # Registers Torch ops
    └── your_kernels/
        └── __init__.py      # Python API wrapping _ops
```

The `build.toml` tells `kernel-builder` what to build. The agent generates this for you, including the correct `cuda-capabilities` for your target GPU:

```
[general]
name = "your_kernels"
backends = ["cuda"]

[torch]
src = ["torch-ext/torch_binding.cpp"]

[kernel.rmsnorm]
backend = "cuda"
src = ["kernel_src/rmsnorm.cu"]
depends = ["torch"]
cuda-capabilities = ["9.0"]  # H100
```

### 2. Build all variants with Nix

Kernel Hub kernels must support all recent PyTorch and CUDA configurations. The kernel-builder Nix flake handles this automatically. Copy the [example `flake.nix`](https://github.com/huggingface/kernels/blob/main/builder/examples/relu/flake.nix) into your project and run:

```shell
nix flake update
nix run .#build-and-copy -L
```

This builds the kernel for every required PyTorch/CUDA variant and places the results in `build/`. For faster builds, enable the HuggingFace Nix cache:

```shell
nix run nixpkgs#cachix -- use huggingface
```

### 3. Create a Hub repo and push

Create a model repo on the Hub and upload the built kernel:

```shell
huggingface-cli repo create your-org/your-kernel --type model
huggingface-cli upload your-org/your-kernel ./build
```

### 4. Others load it in one line

Once published, anyone can use your kernel with zero compilation:

```py
from kernels import get_kernel

rmsnorm = get_kernel("your-org/your-kernel")
```

`get_kernel` detects the user's Python, PyTorch, and CUDA versions and downloads the matching pre-compiled binary. No builds, no flags, typically ready in seconds.

The skill and the Hub are complementary. The skill handles development. The Hub handles distribution. Build a kernel with the skill, validate it with the benchmark scripts, publish it to the Hub, and it becomes a one-liner for everyone else.


## Resources

- [CUDA Kernels Skill (SKILL.md)](http://.claude/skills/cuda-kernels/SKILL.md)  
- [LTX-Video Example](http://examples/ltx_video/)  
- [Qwen3-8B Example](http://examples/qwen3_8b/)  
- [HuggingFace Kernel Hub Blog](https://huggingface.co/blog/hello-hf-kernels)  
- [Kernels Hub in TRL](https://huggingface.co/docs/trl/en/kernels_hub)  
- [We Got Claude to Fine-Tune an Open Source LLM](https://huggingface.co/blog/hf-skills-training)  
- [We Got Claude to Teach Open Models](https://huggingface.co/blog/upskill)  
- [HuggingFace Kernels Community](https://huggingface.co/kernels-community)