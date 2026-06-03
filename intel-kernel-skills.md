---
title: "Intel XPU Kernel Skill: LLM-driven Triton kernel optimization for the Hugging Face Kernel Hub"
thumbnail: /blog/assets/intel-xpu-kernels-skill/banner.png
authors:
- user: danf
  guest: true
  org: Intel
- user: burtenshaw
---

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-xpu-kernels-skill/xpu-kernels-top.png" width=700 alt="An LLM-driven workflow generates and benchmarks Triton kernels for an Intel XPU GPU." />
</p>

# Intel XPU Kernel Skill: LLM-driven Triton kernel optimization for the Hugging Face Kernel Hub

*By Intel DCG AI Software Group*

[Xe-Forge](https://github.com/IntelLabs/Xe-Forge) ([Spoczynski et al., 2026](https://arxiv.org/abs/2605.26118)) is an Intel project that uses an LLM to optimize Triton kernels for Intel Arc Pro GPUs (Xe2). It applies a sequence of optimization stages — fusion, dtype fixes, memory access, block pointers, XPU-specific tuning, autotuning — and validates each one on the GPU before moving on. The agent loop, called CoVeR (Chain-of-Verification-and-Refinement), proposes a candidate, runs it, and iterates if it fails or regresses. A small knowledge base of Xe2-specific patterns (tensor descriptors, GRF mode 256, tile swizzling) is read at the start of each session because these aren't well-represented in LLM training data.

On Arc Pro B70, Xe-Forge delivers a 1.26× geomean speedup over PyTorch eager across the full 100 KernelBench Level-2 kernels — with 69% of problems seeing a net speedup — and up to 13.3× on Flash Attention forward.

The [xpu-kernels skill](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/xpu-kernels) packages Xe-Forge's Claude Code engine — the same workflow, tools, and knowledge base — as an Agent Skill, so a coding agent can run the loop without cloning the full project, and the finalized kernel can be published to the [Hugging Face Kernel Hub](https://huggingface.co/kernels) and loaded with `get_kernel(...)`. This post shows that:

- An LLM agent equipped with the right tools and knowledge base can autonomously turn a PyTorch reference — *or an existing, already hand-tuned Triton kernel* — into a faster Triton kernel for Intel XPU.

- A branching trial-loop (analyze → validate → benchmark → profile → finalize) consistently finds speedups over PyTorch eager, naive Triton baselines, *and production Triton kernels such as vLLM's attention and MoE ops* on Arc Pro B70 and Battlemage / Arc Pro B50.

- The resulting kernels are packaged so they can be uploaded to the [Hugging Face Kernel Hub](https://huggingface.co/kernels) and consumed via the [`kernels`](https://github.com/huggingface/kernels) Python package.

👉 Skill: <https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/xpu-kernels> \
👉 Xe-Forge: <https://github.com/IntelLabs/Xe-Forge> \
👉 Kernels Project: <https://huggingface.co/kernels>

## Why a kernel skill?

Coding agents and compilers already produce correct Triton kernels reliably. The gap is the measure-decide-rewrite loop that turns *correct* into *fast* — and, just as importantly, *fast* into *faster*. The same loop that takes a PyTorch reference to an optimized Triton kernel also takes an **already-optimized Triton kernel and makes it faster still**: it treats the existing kernel as the baseline and searches for XPU-aware wins (tensor descriptors, GRF mode, swizzling, dtype contracts) the original author didn't apply. That's how we get speedups over production kernels like vLLM's attention and MoE ops, not just over eager. On Xe2 that gap is wider for two reasons. First, the relevant API choices on Intel XPU (tensor descriptors over block pointers, `grf_mode='256'` for compute-heavy kernels, GROUP_SIZE_M swizzling, the rule against autotuning `BLOCK_D`) are underrepresented in LLM training data, so prompts about "fast Triton on Intel Arc" tend to produce CUDA-flavored code that compiles but doesn't run well. Second, kernel optimization isn't a single decision — tile sizes, dtype contracts, fusion boundaries, and accumulator precision interact, and a one-shot LLM rewrite usually regresses somewhere.

Xe-Forge addresses both: a knowledge base supplies the missing facts, and the CoVeR loop runs each candidate on the GPU and iterates on the measurement. Xe-Forge ships two engines that share the same stages and knowledge base — a fully automated DSPy pipeline, and a Claude Code engine that hands the loop to an interactive coding agent. The `xpu-kernels` skill is the Claude Code engine, repackaged so it works inside any compatible coding-agent client without cloning Xe-Forge.

## How It Works

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-xpu-kernels-skill/system-diagram.png" width=900 alt="System diagram: coding agent with the xpu-kernels skill produces a Triton .py file, which the kernel-builder Rust CLI compiles and uploads as a Hub package, then a downstream user loads it via get_kernel()." /><br>
<em>Figure 1: How the pieces fit together. The <strong>xpu-kernels skill</strong> is the agent's authoring loop and outputs a single Triton <code>.py</code> file. The <strong>kernel-builder CLI</strong> is a Rust tool that compiles and uploads that file as a Hub-compatible package. Downstream code consumes it with <code>get_kernel(...)</code>.</em>
</p>

- **Hardware:** Intel Arc Pro B70 (primary — Xe2, 32 Xe-cores / 256 XVEs, 22.94 FP32 TFLOPS, 367 peak INT8 TOPS, 32 GB GDDR6, 608 GB/s, 230 W TBP) and Battlemage G21 / Arc Pro B50 (also verified — Xe2, 16 Xe-cores / 128 XVEs, 10.65 FP32 TFLOPS, 170 peak INT8 TOPS, 16 GB GDDR6, 224 GB/s, 70 W TBP). Both are Xe2; the skill's optimization patterns apply to both.
- **Compiler:** [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton).
- **Harness:** [ai-bench](https://github.com/libxsmm/AI-bench) — a unified benchmark harness for AI kernels across PyTorch, Triton, Helion, MLIR, Gluon, and SYCL backends — measures correctness and performance against a PyTorch (or naive Triton) baseline.
- **Profiler:** Intel VTune 2025+ is optionally invoked for hardware counters; it can be disabled in `scripts/config.yaml`.
- **Hub integration:** generated kernels follow the [Kernel Hub layout](https://github.com/huggingface/kernels) so they can be published and then loaded with `kernels.get_kernel("<org>/<name>")`.

- **Agent Interface:** the skill's `SKILL.md` instructs the agent to read `scripts/config.yaml` and the relevant reference docs, then enter a well-defined trial loop. The agent only writes Triton kernel files (`*_triton.py` or trial files `t<id>.py`) — it never touches benchmark or test scripts.

- **Trial tree:** `trial_manager.py` keeps a branching tree of attempts. The agent can branch back to the best trial and try a different optimization strategy when it regresses or plateaus.

- **Design Goals:**

  - **Reproducibility:** the same skill, knowledge base, and tool versions produce comparable trial trees across runs.

  - **Safety:** GPU jobs are explicitly serialized (one XPU, one job at a time). Validation runs CPU-only before any GPU time is spent.

  - **Hub-readiness:** the finalized kernel is a self-contained Triton file that fits the Kernel Hub packaging used by [`kernels-community`](https://huggingface.co/kernels-community).

## The xpu-kernels workflow

The skill enforces a five-step loop. The agent must run **all** `max_trials` from `config.yaml`, preventing premature termination on a plateau:

- **Analyze:** `python scripts/analyze_kernel.py <pytorch_file>` extracts shapes, dtypes, fused ops, and fusion opportunities. The agent reads `references/correctness.yaml` and `references/xpu_optimizations.yaml` before writing a single line.

- **Initialize:** `python scripts/trial_manager.py init <name> <baseline>` opens a trial tree.

- **Trial loop (per trial):**

  - **Validate** — `validate_triton.py` checks Triton syntax and XPU-specific constraints (no autotuned `BLOCK_D`, no Python `min`/`max`, correct `boundary_check` indices, no mixing of block-pointer and tensor-descriptor APIs, etc.). CPU only.

  - **Save** — `trial_manager.py save ... --parent <id> --strategy "..."` records the attempt and a one-line strategy.

  - **Benchmark** — `benchmark.py` runs the baseline and the new kernel through ai-bench. After the first trial, baseline time is cached (`--baseline-us`) so subsequent trials only measure the candidate.

  - **Profile** — when `vtune_enabled: true`, `xpu_profiler.py` collects GPU hardware counters and emits recommendations (low occupancy → bigger tiles, overhead-bound → pre-pack to bf16, etc.).

  - **Decide** — speedup improved → continue on this branch; regressed or correctness failed → branch back to best trial; plateau → try a fundamentally different algorithm (different tiling, fusion, persistent kernel, Stream-K, ...).

- **Finalize:** `trial_manager.py finalize <name> <out>_triton.py` copies the best trial out, ready to be packaged for the Hub.

## Knowledge base

The skill is only as good as its references. We ship the curated subset of the Xe-Forge knowledge base that matters most for XPU:

- **`references/correctness.yaml`** — hard constraints to avoid silent miscompares (BLOCK_D, int64 batch indices, atomic accumulation, ai-bench Model class shape).

- **`references/xpu_optimizations.yaml`** — Intel-specific patterns: tensor descriptors, GRF mode 256, tile swizzling, bf16 inputs with fp32 accumulation.

- **`references/optimization_levels.yaml`** — progressive L1–L5 levels with a "try harder" decision tree the agent consults on a plateau.

- **`references/fusion_patterns.yaml` / `memory_patterns.yaml` / `dtype_optimizations.yaml` / `persistent_kernel_patterns.yaml`** — when to fuse vs. split, how to coalesce, mixed precision choices, Stream-K and persistent kernels.

- **`references/implementation_reference.md`** — code templates and the ai-bench `Model` class pattern.

- **`references/optimization_strategies.md` / `workflow_details.md`** — strategy reference, the "try harder" decision tree, and the detailed workflow / benchmarking notes the agent reads at session start.

- **`references/huggingface-kernels-integration.md` / `kernelbench-classification.md`** — recipes for packaging a finalized kernel for the Hugging Face Kernel Hub and the operator taxonomy used to label trials.

## Hugging Face Kernel Hub integration

A finalized kernel is a self-contained Triton file. Once packaged with `kernel-builder`, it can be published to the Hub and consumed exactly like any other [kernels-community](https://huggingface.co/kernels-community) kernel:

```python
import torch
from kernels import get_kernel

# Load an XPU-optimized kernel produced by the skill
fused_gemm = get_kernel("<your-org>/xpu-fused-gemm-sigmoid")

x = torch.randn((1024, 1024), dtype=torch.bfloat16, device="xpu")
w = torch.randn((1024, 1024), dtype=torch.bfloat16, device="xpu")
y = fused_gemm.run(x, w)
```

This closes the loop with the broader Hugging Face stack: the same `get_kernel(...)` API used in `kernels`'s [README quick-start](https://github.com/huggingface/kernels) now reaches Intel XPU optimized kernels generated by the skill.

## Evaluation

We evaluated the skill on Intel Arc Pro B70 (primary) and additionally on Battlemage G21 / Arc Pro B50 to confirm portability. Speedup is the median over AI-Bench trials versus the indicated baseline; bf16 unless noted.

> **Measurement scope.** All numbers below are measured inside the skill's authoring loop with the [ai-bench](https://github.com/libxsmm/AI-bench) harness — it times the candidate Triton kernel against the baseline Triton kernel directly. This is *not* an end-to-end measurement of a kernel that has been compiled and published through the `kernel-builder` CLI, nor of vLLM running with the optimized kernel swapped in. We report the kernel-level speedup that the loop optimizes against; integration overhead, dispatch, and any framework-level effects are out of scope here.

### Arc Pro B70 — primary results


#### Flash Attention forward (fp16)

The most demanding test isn't beating PyTorch eager — eager is unfused and unoptimized, so a win there is expected. The harder, more meaningful test is whether the skill can take a kernel that has *already* been hand-written and tuned by experts and make it faster. The clearest case is Flash Attention forward, benchmarked against the Flash Attention kernel shipped in the Intel XPU Triton backend across attention configurations spanning head counts (A), head dim (D), and sequence lengths (S) from 2K to 16K. The pattern is striking: the **original** kernels are scattered low — and fall *further* as the sequence grows, down to ~5 TFLOPS at S=16384 — while the **optimized** kernels snap into a tight ~60–80 TFLOPS band regardless of configuration. In other words, the skill removes the sequence-length cliff: the longest, most arithmetic-intensive configs see the largest gains (up to **13.3×**), because that's exactly where the stock kernel was leaving the most on the table.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-xpu-kernels-skill/flash-attention-roofline.png" width=900 alt="Roofline plot for FlashAttention forward on Arc Pro B70: arithmetic intensity (FLOP/Byte, log x-axis from ~800 to ~10000) vs. performance (TFLOPS, log y-axis), with the compute-bound roof drawn near the top. Original kernels (blue circles) are scattered from ~5 to ~40 TFLOPS and trend downward as arithmetic intensity rises, while optimized kernels (red stars) cluster tightly at ~60-80 TFLOPS across all configurations. Each pair is joined by a vertical arrow; points are labeled by head count, head dim, and sequence length (e.g. A=32 S=8192, A=40 S=16384), and the largest jumps occur at the longest sequence lengths."/><br>
<em>Figure 2: Roofline for FlashAttention forward on Arc Pro B70, against the Flash Attention kernel shipped in the Intel XPU Triton backend. <strong>Original</strong> (blue circles) vs. <strong>Optimized</strong> (red stars), joined by an arrow; labels give the attention configuration (A = heads, D = head dim, S = sequence length). The original kernel degrades as the sequence grows — down to ~5 TFLOPS at S=16384 — whereas the optimized kernels hold a steady ~60–80 TFLOPS band, so the biggest speedups land on the longest, most compute-intensive sequences.</em>
</p>

#### Production Triton kernels: vLLM attention & MoE

Beyond Flash Attention, we pointed the skill at the Triton kernels behind vLLM's attention and MoE paths — `BatchedMoE`, `FusedMoE`, and `UnifiedAttention` — again using the **vLLM Triton kernel itself as the baseline**, so the skill's job was purely *Triton → faster Triton* with no PyTorch reference involved. Each numbered point is a distinct kernel *configuration* — a real model's shapes, dtypes, and batch settings — **not** an end-to-end run of that model.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-xpu-kernels-skill/vllm-attention-roofline.png" width=900 alt="Roofline plot (Intel B70) for vLLM attention and MoE Triton kernels: arithmetic intensity (FLOP/Byte, log x-axis) vs. performance (TFLOPS, log y-axis), with the 608 GB/s memory-bandwidth ridge and 160 TFLOPS compute-bound roof drawn. Kernels are colored by family — BatchedMoE, FusedMoE, UnifiedAttention — with the original kernel as a circle and the skill-optimized kernel as a star, connected by an arrow. Most stars sit well above their circles, several reaching the roof; 24 numbered configurations map to model-driven shapes such as Gemma2-27B prefill and llama3.3-70B decode."/><br>
<em>Figure 3: Roofline for vLLM's attention/MoE Triton kernels on Arc Pro B70 (608 GB/s, 160 TFLOPS peak). Each configuration appears as an <strong>Original</strong> kernel (circle) and the <strong>Optimized</strong> kernel the skill produced (star), joined by an arrow; color denotes the kernel family (BatchedMoE / FusedMoE / UnifiedAttention), and the numbered legend keys each point to its configuration. Optimization moves points up toward the hardware roof, so the speedups come from better use of available compute and bandwidth.</em>
</p>

These 24 configurations are drawn from production models — Gemma2/3-27B, gpt-oss 20B, Llama3.1-8B, Llama3.3-70B, Llama4 Scout, and Qwen2.5/3 — and cover prefill, decode, chunked prefill, sliding-window, and fp8/int8 KV-cache and weight-quantized setups. Lifting each configuration's original kernel to its Xe-Forge–optimized counterpart gives a **geometric-mean 2.8× speedup** across the suite. Relative gains are largest on memory-bound configs lifted off near-zero baselines (up to **35×** for Qwen3-30B-A3B-Instruct), while the highest absolute throughput comes from the compute-bound Gemma3-27B prefill kernel, which reaches the **160 TFLOPS** peak on Arc Pro B70.

These are gains *on top of* an already-optimized kernel, across a spread of attention and MoE configurations — the clearest evidence that the skill contributes Intel-specific optimization knowledge rather than just fusing what eager left on the table.

#### Breadth: KernelBench Level 2

For breadth across a wider operator mix, the underlying **Xe-Forge framework** was run across the full **100 KernelBench Level-2** fused patterns (GEMM+Sigmoid+Scaling+ResidualAdd, GEMM+GELU+Softmax, Conv+BatchNorm+ReLU, …) vs. PyTorch eager on Arc Pro B70, reaching a **1.26× geomean speedup** with a **69% win rate** (the fraction of problems that see a net speedup); the full per-problem analysis is in the [Xe-Forge paper](https://arxiv.org/abs/2605.26118).

**Key insight:** the bottleneck for LLM-driven kernel optimization on a less-represented architecture is *knowledge access*, not raw model capability. A small, curated reference set plus a strict tool-driven loop is enough to make a general coding agent productive on Intel XPU — productive enough to improve on kernels that experts have already optimized.

## Try It Yourself

End-to-end in three steps: **install the skill → use the agent to *write* an optimized Triton kernel file → use the `kernel-builder` CLI to *build and publish* that kernel as a Hub package.**

> Two distinct things share the `kernel-builder` name and it's worth being explicit:
> - The **xpu-kernels skill** is the agent-driven authoring loop. It produces a single self-contained Triton `.py` file. It does **not** build, package, or upload anything.
> - The **`kernel-builder` CLI** is a separate Rust tool (also living in `huggingface/kernels`) that takes a Triton/CUDA/ROCm source file, runs a Nix-based build for the requested backends, and uploads the resulting Hub-compatible kernel package. The skill's only output is the source file that the CLI then consumes.

### 1. Install the skill into your coding agent

The `kernel-builder` CLI has a `skills add` subcommand that copies the skill files into the right place for your assistant. Run this in your project directory:

```bash
# Install kernel-builder (provides the `kernel-builder` CLI)
pip install kernels

# Drop the xpu-kernels skill into this project for Claude Code
# (also: --codex, --opencode)
kernel-builder skills add --skill xpu-kernels --claude
```

This drops `SKILL.md`, `scripts/`, and `references/` into your project — that's all the skill is. For the full Xe-Forge experience (ai-bench harness, test kernels, VTune), also clone the upstream:

```bash
git clone https://github.com/IntelLabs/Xe-Forge && cd Xe-Forge
uv sync --extra intel
```

### 2. Have the agent write an optimized Triton kernel

Open your assistant in the project, point it at a PyTorch reference, and let the trial loop run. With Claude Code:

```bash
claude "Use the xpu-kernels skill to optimize my_op_pytorch.py. \
  Run all max_trials and finalize as my_op_triton.py."
```

The agent analyzes the baseline, generates Triton variants, validates and benchmarks each on the XPU, and writes the best trial to `my_op_triton.py`. The output is a plain Python source file — no compilation, no Hub upload.

### 3. Build and publish the kernel with the `kernel-builder` CLI

Take the Triton file the skill produced and feed it to the builder:

```bash
# Scaffold a Hub kernel project (build.toml, project layout) for XPU
kernel-builder init --name my-op --backends xpu

# Copy my_op_triton.py into the scaffold, then build a Hub-compatible
# package and upload it
kernel-builder build-and-upload my-op --name <your-org>/my-op
```

`build-and-upload` is what actually compiles per build variant and pushes the artifact to the Hub.

### 4. Load it like any Hub kernel

```python
import torch
from kernels import get_kernel

my_op = get_kernel("<your-org>/my-op")

x = torch.randn((1024, 1024), dtype=torch.bfloat16, device="xpu")
y = my_op.run(x)
```

That's it — the kernel works inside any `transformers` / `diffusers` / custom XPU code path that consumes Hub kernels.

**Links:**

- Skill: <https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/xpu-kernels>
- `kernels` package: <https://github.com/huggingface/kernels>
- Xe-Forge: <https://github.com/IntelLabs/Xe-Forge> · [paper](https://arxiv.org/abs/2605.26118)
- Intel XPU Backend for Triton: <https://github.com/intel/intel-xpu-backend-for-triton>
- KernelBench: <https://github.com/ScalingIntelligence/KernelBench>
- Companion blog — *Custom Kernels for All from Codex and Claude*: <https://huggingface.co/blog/custom-cuda-kernels-agent-skills>
- Kernel Hub community: <https://huggingface.co/kernels-community>

Contributions, issues, and new reference patterns are welcome. 🚀

## Citation

If you use this work in your research, please cite the Xe-Forge paper:

```bibtex
@article{spoczynski2026xeforge,
  title   = {Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization for Intel GPU},
  author  = {Spoczynski, Marcin and Fleischer, Daniel and Berchansky, Moshe and
             Stan, Gabriela Ben-Melech and Guskin, Shira and Xu, Weilin and
             Siemieniuk, Adam and Heinecke, Alexander},
  journal = {arXiv preprint arXiv:2605.26118},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.26118}
}
```

## Limitations & Future Work

- **Hardware scope:** verified on Arc Pro B70 and Battlemage G21 / Arc Pro B50 (both Xe2). Other Intel XPUs may require updated patterns in `references/xpu_optimizations.yaml`.

- **Workload scope:** the knowledge base focuses on GEMM, fused KernelBench Level 2 patterns, reductions, attention forward, and MoE kernels (including the production vLLM Triton attention and MoE kernels). Backward passes, sparse, and quantized kernels are future work.

- **Single-XPU serialization:** the loop assumes one XPU per machine. Multi-GPU benchmarking requires changes to `benchmark.py` and `xpu_profiler.py`.

- Generated kernels are LLM-produced and must be validated. The mandatory `validate_triton.py` + `benchmark.py` correctness check catches most issues, but any production deployment should add its own regression tests.

