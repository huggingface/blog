---
title: "Intel XPU Kernel Skill: LLM-driven Triton kernel optimization for the Hugging Face Kernel Hub"
thumbnail: /blog/assets/intel-xpu-kernels-skill/banner.png
authors:
- user: danf
  guest: true
  org: Intel
- user: moshew
  guest: true
  org: Intel
- user: burtenshaw
---

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-xpu-kernels-skill/xpu-kernels-top.png" width=700 alt="An LLM-driven workflow generates and benchmarks Triton kernels for an Intel XPU GPU." />
</p>

# Intel XPU Kernel Skill: LLM-driven Triton kernel optimization for the Hugging Face Kernel Hub

*By Intel DCG AI Software and OCTO Parallel Computing Lab*

[Xe-Forge](https://github.com/IntelLabs/Xe-Forge) ([Spoczynski et al., 2026](https://arxiv.org/abs/2605.26118)) is an Intel project that uses an LLM to optimize Triton kernels for Intel Arc Pro GPUs (Xe2). It applies a sequence of optimization stages — fusion, dtype fixes, memory access, block pointers, XPU-specific tuning, autotuning — and validates each one on the GPU before moving on. The agent loop, called CoVeR (Chain-of-Verification-and-Refinement), proposes a candidate, runs it, and iterates if it fails or regresses. A small knowledge base of Xe2-specific patterns (tensor descriptors, GRF mode 256, tile swizzling) is read at the start of each session because these aren't well-represented in LLM training data.

On Arc Pro B70, Xe-Forge delivers a **1.26× geomean** speedup over PyTorch eager across the full 100 KernelBench Level-2 kernels (69% win rate), a **2.8× geomean** over vLLM's production attention and MoE Triton kernels, and up to **13.3×** on Flash Attention forward.

The [xpu-kernels skill](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/xpu-kernels) packages Xe-Forge's Claude Code engine — the same workflow, tools, and knowledge base — as an Agent Skill, so a coding agent can run the loop without cloning the full project. Point it at a PyTorch reference *or an already hand-tuned Triton kernel*, and the branching trial loop (analyze → validate → benchmark → profile → decide) searches for XPU-aware speedups — over PyTorch eager, naive Triton baselines, and production kernels like vLLM's attention and MoE ops alike. The finalized kernel is a self-contained Triton file you publish to the [Hugging Face Kernel Hub](https://huggingface.co/kernels) and load with `get_kernel(...)`.

👉 Skill: <https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/xpu-kernels> \
👉 Xe-Forge: <https://github.com/IntelLabs/Xe-Forge> \
👉 Kernels Project: <https://huggingface.co/kernels>

## Why a kernel skill

Coding agents and compilers already produce *correct* Triton kernels reliably. The gap is the measure-decide-rewrite loop that turns correct into *fast* — and, just as importantly, fast into *faster* by treating an already-tuned kernel as the baseline. On Xe2 that gap is wide for two reasons. First, the API choices that matter on Intel XPU (tensor descriptors over block pointers, `grf_mode='256'` for compute-heavy kernels, GROUP_SIZE_M swizzling, the rule against autotuning `BLOCK_D`) are underrepresented in LLM training data, so prompts about "fast Triton on Intel Arc" tend to produce CUDA-flavored code that compiles but runs poorly. Second, kernel optimization isn't one decision — tile sizes, dtype contracts, fusion boundaries, and accumulator precision interact, so a one-shot rewrite usually regresses somewhere. Xe-Forge addresses both: a knowledge base supplies the missing XPU facts, and the CoVeR loop runs each candidate on the GPU and iterates on the measurement.

## How it works

The skill is three things in one bundle: an instruction file (`SKILL.md`) that tells the agent how to run, the `scripts/` tools it drives, and the `references/` knowledge base it consults. Pointed at a PyTorch reference (or an existing Triton kernel), the agent reads that knowledge base, then runs a branching trial loop — analyze, validate, benchmark, profile, decide — on the XPU, emitting a single self-contained Triton `.py` file. From there a separate Rust tool, the `kernel-builder` CLI, compiles that file per build variant and uploads it as a Hub-compatible package, which downstream code loads with `get_kernel(...)`. Figure 1 traces that path; the rest of this section walks the loop and the environment it runs in.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-xpu-kernels-skill/system-diagram.png" width=900 alt="System diagram: coding agent with the xpu-kernels skill produces a Triton .py file, which the kernel-builder Rust CLI compiles and uploads as a Hub package, then a downstream user loads it via get_kernel()." /><br>
<em>Figure 1: How the pieces fit together. The <strong>xpu-kernels skill</strong> (instruction file, scripts, and references) drives the agent's trial loop and outputs a single Triton <code>.py</code> file. The <strong>kernel-builder CLI</strong> is a Rust tool that compiles and uploads that file as a Hub-compatible package. Downstream code consumes it with <code>get_kernel(...)</code>.</em>
</p>

### The trial loop

The skill's `SKILL.md` tells the agent to read `scripts/config.yaml` and the relevant references, then run a strict, tool-driven loop. The agent only ever writes Triton kernel files (`*_triton.py` or trial files `t<id>.py`) — it never touches the benchmark or test scripts — and it must run **all** `max_trials` from `config.yaml`, so it can't stop early on a plateau:

- **Analyze** —  extracts shapes, dtypes, fused ops, and fusion opportunities; the agent reads `correctness.yaml` and `xpu_optimizations.yaml` before writing a line.
- **Initialize** — the tool creates a branching trial tree.
- **Per trial:**
  - **Validate** (CPU only) — checks Triton syntax and XPU constraints: no autotuned `BLOCK_D`, no Python `min`/`max`, correct `boundary_check` indices, no mixing of block-pointer and tensor-descriptor APIs.
  - **Save** — records the attempt and a one-line strategy under its parent trial.
  - **Benchmark** — runs baseline and candidate through ai-bench; baseline time is cached after the first trial, so later trials only measure the candidate.
  - **Profile** (optional, `vtune_enabled`) — VTune collects GPU hardware counters and emits recommendations (low occupancy → bigger tiles, overhead-bound → pre-pack to bf16).
  - **Decide** — improved → continue on this branch; regressed or wrong → branch back to the best trial; plateau → try a fundamentally different algorithm (different tiling, fusion, persistent kernel, Stream-K).
- **Finalize** — the best trial is copied out as a self-contained `<name>_triton.py`, ready for `kernel-builder`.

### The environment

- **Hardware:** Intel Arc Pro B70 (primary — Xe2, 32 Xe-cores, 32 GB GDDR6, 608 GB/s) and Battlemage G21 / Arc Pro B50 (also verified — Xe2, 16 Xe-cores, 16 GB GDDR6, 224 GB/s, 70 W TBP). Both are Xe2, so the skill's optimization patterns apply to both.
- **Compiler:** [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton).
- **Harness:** [AI-Bench](https://github.com/libxsmm/AI-bench) — a unified benchmark harness across PyTorch, Triton, Helion, MLIR, Gluon, and SYCL backends — measures correctness and performance against the baseline.
- **Profiler:** Intel VTune 2025+, optionally invoked for hardware counters; disable it in `scripts/config.yaml`.

## Knowledge base

The skill is only as good as its references, so it ships the curated subset of the Xe-Forge knowledge base that matters most for XPU. It covers three things the agent needs but the base model lacks: **correctness constraints** that prevent silent miscompares (`BLOCK_D`, int64 batch indices, atomic accumulation, the ai-bench `Model` shape), **Intel-specific optimization patterns** ([tensor descriptors](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html), [GRF mode 256](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/small-register-mode-vs-large-register-mode.html), [tile swizzling](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), bf16 inputs with fp32 accumulation, fusion/memory/dtype/persistent-kernel recipes), and a **tiered "try harder" decision tree** (L1–L5) the agent consults when a trial plateaus. Code templates and Hub-packaging recipes round out the set.

The linked pages above are the official Intel and Triton references for those patterns — worth a read if you want the background — but the point of the knowledge base is that you don't have to: the facts that actually matter for fast XPU kernels are distilled into the [skill's `references/`](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/xpu-kernels), in the form the agent consults at every trial.

## Evaluation

We evaluated the skill on Intel Arc Pro B70 (primary) and additionally on Battlemage G21 / Arc Pro B50 to confirm portability. Speedup is the median over AI-Bench trials versus the indicated baseline; bf16 unless noted. These are **kernel-level speedups** — AI-Bench times the candidate Triton kernel directly against the baseline kernel — not end-to-end measurements of a fully compiled model or of a framework like vLLM running with the optimized kernel swapped in.

### Flash Attention forward (fp16)

The most demanding test isn't beating PyTorch eager — eager is unfused and unoptimized, so a win there is expected. The harder, more meaningful test is whether the skill can take a kernel that has *already* been hand-written and tuned by experts and make it faster. The clearest case is Flash Attention forward, benchmarked against the Flash Attention kernel shipped in the Intel XPU Triton backend across attention configurations spanning head counts (A), head dim (D), and sequence lengths (S) from 2K to 16K. The pattern is striking: the **original** kernels are scattered low — and fall *further* as the sequence grows, down to ~5 TFLOPS at S=16384 — while the **optimized** kernels snap into a tight ~60–80 TFLOPS band regardless of configuration. In other words, the skill removes the sequence-length cliff: the longest, most arithmetic-intensive configs see the largest gains (up to **13.3×**), because that's exactly where the stock kernel was leaving the most on the table.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-xpu-kernels-skill/flash-attention-roofline.png" width=900 alt="Roofline plot for FlashAttention forward on Arc Pro B70: arithmetic intensity (FLOP/Byte, log x-axis from ~800 to ~10000) vs. performance (TFLOPS, log y-axis), with the compute-bound roof drawn near the top. Original kernels (blue circles) are scattered from ~5 to ~40 TFLOPS and trend downward as arithmetic intensity rises, while optimized kernels (red stars) cluster tightly at ~60-80 TFLOPS across all configurations. Each pair is joined by a vertical arrow; points are labeled by head count, head dim, and sequence length (e.g. A=32 S=8192, A=40 S=16384), and the largest jumps occur at the longest sequence lengths."/><br>
<em>Figure 2: Roofline for FlashAttention forward on Arc Pro B70, against the Flash Attention kernel shipped in the Intel XPU Triton backend. <strong>Original</strong> (blue circles) vs. <strong>Optimized</strong> (red stars), joined by an arrow; labels give the attention configuration (A = heads, D = head dim, S = sequence length).</em>
</p>

### Production Triton kernels: vLLM attention & MoE

Beyond Flash Attention, we pointed the skill at the Triton kernels behind vLLM's attention and MoE paths — `BatchedMoE`, `FusedMoE`, and `UnifiedAttention` — again using the **vLLM Triton kernel itself as the baseline**, so the skill's job was purely *Triton → faster Triton* with no PyTorch reference involved. Each numbered point is a distinct kernel *configuration* — a real model's shapes, dtypes, and batch settings — **not** an end-to-end run of that model.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-xpu-kernels-skill/vllm-attention-roofline.png" width=900 alt="Roofline plot (Intel B70) for vLLM attention and MoE Triton kernels: arithmetic intensity (FLOP/Byte, log x-axis) vs. performance (TFLOPS, log y-axis), with the 608 GB/s memory-bandwidth ridge and 160 TFLOPS compute-bound roof drawn. Kernels are colored by family — BatchedMoE, FusedMoE, UnifiedAttention — with the original kernel as a circle and the skill-optimized kernel as a star, connected by an arrow. Most stars sit well above their circles, several reaching the roof; 24 numbered configurations map to model-driven shapes such as Gemma2-27B prefill and llama3.3-70B decode."/><br>
<em>Figure 3: Roofline for vLLM's attention/MoE Triton kernels on Arc Pro B70 (608 GB/s, 160 TFLOPS peak). Each configuration appears as an <strong>Original</strong> kernel (circle) and the <strong>Optimized</strong> kernel the skill produced (star), joined by an arrow; color denotes the kernel family (BatchedMoE / FusedMoE / UnifiedAttention), and the numbered legend keys each point to its configuration. Optimization moves points up toward the hardware roof, so the speedups come from better use of available compute and bandwidth.</em>
</p>

These 24 configurations are drawn from production models — Gemma2/3-27B, gpt-oss 20B, Llama3.1-8B, Llama3.3-70B, Llama4 Scout, and Qwen2.5/3 — and cover prefill, decode, chunked prefill, sliding-window, and fp8/int8 KV-cache and weight-quantized setups. Lifting each configuration's original kernel to its Xe-Forge–optimized counterpart gives a **geometric-mean 2.8× speedup** across the suite. Relative gains are largest on memory-bound configs lifted off near-zero baselines (up to **35×** for Qwen3-30B-A3B-Instruct), while the highest absolute throughput comes from the compute-bound Gemma3-27B prefill kernel, which reaches the **160 TFLOPS** peak on Arc Pro B70. These are gains *on top of* an already-optimized kernel — the clearest evidence that the skill contributes Intel-specific optimization knowledge, not just the fusion that PyTorch eager left on the table.

### Breadth: KernelBench Level 2

For breadth across a wider operator mix, the underlying **Xe-Forge framework** was run across the full **100 KernelBench Level-2** fused patterns (GEMM+Sigmoid+Scaling+ResidualAdd, GEMM+GELU+Softmax, Conv+BatchNorm+ReLU, …) vs. PyTorch eager on Arc Pro B70, reaching a **1.26× geomean speedup** with a **69% win rate** (the fraction of problems that see a net speedup); the full per-problem analysis is in the [Xe-Forge paper](https://arxiv.org/abs/2605.26118).

**Key insight:** the bottleneck for LLM-driven kernel optimization on a less-represented architecture is *knowledge access*, not raw model capability. A small, curated reference set plus a strict tool-driven loop is enough to make a general coding agent productive on Intel XPU — productive enough to improve on kernels that experts have already optimized.

## Try it yourself

Three steps: **install the skill → let the agent write the kernel → build, publish, and load it from the Hub.**

### 1. Install the skill

```bash
pip install kernels

# Drop the xpu-kernels skill into this project (also: --codex, --opencode)
kernel-builder skills add --skill xpu-kernels --claude
```

This adds `SKILL.md`, `scripts/`, and `references/` to your project — that's the whole skill.

### 2. Let the agent write the kernel

Point your assistant at a PyTorch reference (or an existing Triton kernel) and let the trial loop run:

```bash
claude "Use the xpu-kernels skill to optimize my_op_pytorch.py. \
  Run all max_trials and finalize as my_op_triton.py."
```

The agent analyzes the baseline, generates and benchmarks Triton variants on the XPU, and writes the best one to `my_op_triton.py` — a plain Python source file.

### 3. Build, publish, and load it

That source file still needs to be compiled into a Hub package before it can be loaded. The [`kernel-builder` CLI](https://huggingface.co/docs/kernels/builder/writing-kernels) handles that step — it builds the kernel per variant and uploads it to the Hub — see its docs for the commands. Once published, it loads like any other Hub kernel:

```python
import torch
from kernels import get_kernel

my_op = get_kernel("<your-org>/my-op")

x = torch.randn((1024, 1024), dtype=torch.bfloat16, device="xpu")
y = my_op.run(x)
```

That's it — the kernel works inside any `transformers` / `diffusers` / custom code path that consumes Hub kernels.

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

## Limitations & future work

- **Hardware scope:** verified on Arc Pro B70 and Battlemage G21 / Arc Pro B50 (both Xe2). Other Intel XPUs may require updated patterns in `references/xpu_optimizations.yaml`.

- **Workload scope:** the knowledge base focuses on GEMM, fused KernelBench Level 2 patterns, reductions, attention forward, and MoE kernels (including the production vLLM Triton attention and MoE kernels). Backward passes, sparse, and quantized kernels are future work.

- **Single-XPU serialization:** the loop assumes one XPU per machine. Multi-GPU benchmarking requires changes to `benchmark.py` and `xpu_profiler.py`.

- **Validation:** generated kernels are LLM-produced and must be validated. The mandatory `validate_triton.py` + `benchmark.py` correctness check catches most issues, but any production deployment should add its own regression tests.

