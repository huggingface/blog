---
title: "Intel XPU Kernel Skills: LLM-driven Triton kernel optimization for the Hugging Face Kernel Hub"
thumbnail: /blog/assets/intel-kernel-skills/banner.png
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
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-kernel-skills/xpu-kernels-figure.jpg" width=700 alt="An LLM-driven workflow generates and benchmarks Triton kernels for an Intel XPU GPU." />
</p>

# Intel XPU Kernel Skills: LLM-driven Triton kernel optimization for the Hugging Face Kernel Hub

*By Intel AI Software Group*

[Xe-Forge](https://github.com/IntelLabs/Xe-Forge) ([Spoczynski et al., 2026](https://arxiv.org/abs/2605.26118)) is an Intel project that uses an LLM to optimize Triton kernels for Intel Arc Pro GPUs (Xe2). It applies a sequence of optimization stages — fusion, dtype fixes, memory access, block pointers, XPU-specific tuning, autotuning — and validates each one on the GPU before moving on. The agent loop, called CoVeR (Chain-of-Verification-and-Refinement), proposes a candidate, runs it, and iterates if it fails or regresses. A small knowledge base of Xe2-specific patterns (tensor descriptors, GRF mode 256, tile swizzling) is read at the start of each session because these aren't well-represented in LLM training data.

On Arc Pro B70, Xe-Forge reports a 1.17× geomean speedup over PyTorch eager across 97 KernelBench Level-2 kernels and 2–13.3× on Flash Attention forward.

The [xpu-kernels skill](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/xpu-kernels) packages Xe-Forge's Claude Code engine — the same workflow, tools, and knowledge base — as an Agent Skill, so a coding agent can run the loop without cloning the full project, and the finalized kernel can be published to the [Hugging Face Kernel Hub](https://huggingface.co/kernels) and loaded with `get_kernel(...)`.

We release three skills together — **`cuda-kernels`**, **`rocm-kernels`**, and **`xpu-kernels`** — each tuned to its hardware vendor. This post focuses on the Intel XPU one and shows that:

- 🤖 An LLM agent equipped with the right tools and knowledge base can autonomously turn a PyTorch reference into an optimized Triton kernel for Intel XPU.

- ⚡ A branching trial-loop (analyze → validate → benchmark → profile → finalize) consistently finds speedups over both PyTorch eager and the naive Triton baselines on Arc Pro B70 and Battlemage / Arc Pro B50.

- 📦 The resulting kernels are packaged so they can be uploaded to the [Hugging Face Kernel Hub](https://huggingface.co/kernels) and consumed via the [`kernels`](https://github.com/huggingface/kernels) Python package.

👉 Skill: <https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/xpu-kernels> \
👉 Original framework: <https://github.com/IntelLabs/Xe-Forge> \
👉 Kernel Hub: <https://huggingface.co/kernels>

## Why a kernel skill?

Two things make Xe2 a hard target for an off-the-shelf coding agent. First, the relevant API choices on Intel XPU (tensor descriptors over block pointers, `grf_mode='256'` for compute-heavy kernels, GROUP_SIZE_M swizzling, the rule against autotuning `BLOCK_D`) are underrepresented in LLM training data, so prompts about "fast Triton on Intel Arc" tend to produce CUDA-flavored code that compiles but doesn't run well. Second, kernel optimization isn't a single decision — tile sizes, dtype contracts, fusion boundaries, and accumulator precision interact, and a one-shot LLM rewrite usually regresses somewhere.

Xe-Forge addresses both: a knowledge base supplies the missing facts, and the CoVeR loop runs each candidate on the GPU and iterates on the measurement. Xe-Forge ships two engines that share the same stages and knowledge base — a fully automated DSPy pipeline, and a Claude Code engine that hands the loop to an interactive coding agent. The `xpu-kernels` skill is the Claude Code engine, repackaged so it works inside any compatible coding-agent client without cloning Xe-Forge.

## How It Works

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-kernel-skills/system-diagram.png" width=900 alt="System diagram: coding agent with the xpu-kernels skill produces a Triton .py file, which the kernel-builder Rust CLI compiles and uploads as a Hub package, then a downstream user loads it via get_kernel()." /><br>
<em>Figure 1: How the pieces fit together. The <strong>xpu-kernels skill</strong> is the agent's authoring loop and outputs a single Triton <code>.py</code> file. The <strong>kernel-builder CLI</strong> is a separate Rust tool that compiles and uploads that file as a Hub-compatible package. Downstream code consumes it with <code>get_kernel(...)</code>.</em>
</p>

- **Hardware:** Intel Arc Pro B70 (primary — Xe2, 32 Xe-cores / 256 XVEs, 22.94 FP32 TFLOPS, 367 peak INT8 TOPS, 32 GB GDDR6, 608 GB/s, 230 W TBP) and Battlemage G21 / Arc Pro B50 (also verified — Xe2, 16 Xe-cores / 128 XVEs, 10.65 FP32 TFLOPS, 170 peak INT8 TOPS, 16 GB GDDR6, 224 GB/s, 70 W TBP). Both are Xe2; the skill's optimization patterns apply to both.
- **Compiler:** [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton).
- **Harness:** [ai-bench](https://github.com/libxsmm/AI-bench) — a unified benchmark harness for AI kernels across PyTorch, Triton, Helion, MLIR, Gluon, and SYCL backends — measures correctness and performance against a PyTorch (or naive Triton) baseline.
- **Profiler:** Intel VTune 2025+ is optionally invoked for hardware counters; it can be disabled in `scripts/config.yaml`.
- **Hub integration:** generated kernels follow the [Kernel Hub layout](https://github.com/huggingface/kernels) so they can be published and then loaded with `kernels.get_kernel("<org>/<name>")`.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-kernel-skills/trial-tree.png" width=800 alt="A branching trial tree from a real run: nodes are kernel variants labelled with their measured speedup vs. baseline, edges are optimization strategies, regressions branch back to the best ancestor, and the chosen finalized branch is highlighted." /><br>
<em>Figure 2: A real branching trial tree produced by <code>trial_manager.py</code>. Each node is a kernel variant with its measured speedup; edges are the strategy applied (fusion, GRF mode, swizzling, ...). When a trial regresses or plateaus the agent branches back to the best ancestor and tries a different strategy. The highlighted path is the trajectory of the finalized kernel. The 5-step trial loop (analyze → validate → benchmark → profile → decide) drives each node; the tree is what <em>navigates</em> them.</em>
</p>

- **Agent Interface:** the skill's `SKILL.md` instructs the agent to read `scripts/config.yaml` and the relevant reference docs, then enter a fixed trial loop. The agent only writes Triton kernel files (`*_triton.py` or trial files `t<id>.py`) — never benchmark or test scripts.

- **Trial tree:** `trial_manager.py` keeps a branching tree of attempts. The agent can branch back to the best trial and try a different optimization strategy when it regresses or plateaus.

- **Design Goals:**

  - **Reproducibility:** the same skill, knowledge base, and tool versions produce comparable trial trees across runs.

  - **Safety:** GPU jobs are explicitly serialized (one XPU, one job at a time). Validation runs CPU-only before any GPU time is spent.

  - **Hub-readiness:** the finalized kernel is a self-contained Triton file that fits the Kernel Hub packaging used by [`kernels-community`](https://huggingface.co/kernels-community).

## The xpu-kernels workflow

The skill enforces a five-step loop. The agent must run **all** `max_trials` from `config.yaml` (early stop only on >5× speedup), preventing premature termination on a plateau:

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

We evaluated the skill on Intel Arc Pro B70 (primary) and additionally on Battlemage G21 / Arc Pro B50 to confirm portability. Speedup is the median over ai-bench trials versus the indicated baseline; bf16 unless noted.

### Arc Pro B70 — primary results

Aligned with the Xe-Forge paper's evaluation hardware. **(UPDATE NUMBERS — fill in geomean, % improved, peak, FA range from your B70 runs.)**

- **KernelBench Level 2 — fused kernels (bf16):** GEMM+Sigmoid+Scaling+ResidualAdd, GEMM+GELU+Softmax, Conv+BatchNorm+ReLU, and other fused patterns vs. PyTorch eager. **(UPDATE NUMBERS: geomean speedup, fraction improved, peak speedup.)**

- **Flash Attention Forward (fp16):** vs. the Flash Attention kernel shipped in the Intel XPU Triton backend, across multiple sequence lengths. **(UPDATE NUMBERS: speedup range across configs, peak TFLOPS.)**

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-kernel-skills/kernelbench-l2-bars.png" width=1000 alt="Per-kernel speedup over PyTorch eager across the 97 KernelBench Level-2 fused kernels on Arc Pro B70, sorted ascending. A horizontal line marks 1.0× (parity with eager) and a second line marks the geometric mean. The fraction of kernels above 1.0× is annotated."/><br>
<em>Figure 3: Per-kernel speedup over PyTorch eager across the 97 KernelBench Level-2 fused kernels on Arc Pro B70, sorted ascending. The 1.0× line marks parity; the geomean line marks the headline number. This view shows the full distribution — including the kernels that regress — instead of a single summary statistic.</em>
</p>

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-kernel-skills/flash-attention-vs-seqlen.png" width=900 alt="Speedup of the agent-optimized Flash Attention forward kernel vs. the Flash Attention kernel shipped in the Intel XPU Triton backend, plotted against sequence length, with separate lines for Arc Pro B70 and Arc Pro B50."/><br>
<em>Figure 4: Flash Attention forward speedup vs. sequence length on Arc Pro B70 and B50, against the Flash Attention kernel shipped in the Intel XPU Triton backend.</em>
</p>

### Arc Pro B50 — portability check

The same skill, with no source changes, runs on the smaller Xe2 card and produces correct, faster kernels. **(UPDATE NUMBERS — fill in B50 geomean, fraction improved, peak; mention any kernels that did not improve.)**

### Cross-cutting observations

- We compare three configurations: **PyTorch eager**, the **naive Triton baseline** shipped in `test_kernels/`, and the **agent-optimized Triton** produced by the skill. The naive baselines exist to demonstrate that the speedup is not "Triton vs. eager" — it is the *optimized* Triton, with tensor descriptors, GRF 256, and swizzling, that wins.

- The agent reliably discovers the Intel-specific patterns when the references are in scope; without them, the same model produces CUDA-flavored Triton that often validates but rarely speeds up.

**Key insight:** the bottleneck for LLM-driven kernel optimization on a less-represented architecture is *knowledge access*, not raw model capability. A small, curated reference set plus a strict tool-driven loop is enough to make a general coding agent productive on Intel XPU.

## Why It Matters

- **Vendor coverage:** the same skill format now covers CUDA, ROCm, and Intel XPU, so kernels for all three vendors can be authored and published through the same Hub pipeline.

- **Reproducibility:** the strict trial loop, `config.yaml`, and cached baseline times make runs comparable across users and machines.

- **Hub-native output:** the finalized kernel is loaded with `get_kernel(...)` like any other Hub kernel.

## Conclusion

The interesting part is Xe-Forge: a CoVeR loop, a multi-stage pipeline, and an Xe2 knowledge base, evaluated in the [paper](https://arxiv.org/abs/2605.26118). The skill is the integration that lets you run that loop from a coding agent and ship the result through the Hugging Face Kernel Hub.

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

The agent analyzes the baseline, generates Triton variants, validates and benchmarks each on the XPU, and writes the best trial to `my_op_triton.py`. **The skill stops here.** The output is a plain Python source file — no compilation, no Hub upload.

### 3. Build and publish the kernel with the `kernel-builder` CLI

This is a separate step using the Rust CLI; it has nothing to do with the agent. Take the Triton file the skill produced and feed it to the builder:

```bash
# Scaffold a Hub kernel project (build.toml, project layout) for XPU
kernel-builder init my-op --backends xpu

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
- `kernels` Python package: <https://github.com/huggingface/kernels>
- Original framework: <https://github.com/IntelLabs/Xe-Forge>

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

- **Workload scope:** the knowledge base focuses on GEMM, fused KernelBench Level 2 patterns, reductions, and Flash Attention forward. Backward passes, sparse, and quantized kernels are future work.

- **Single-XPU serialization:** the loop assumes one XPU per machine. Multi-GPU benchmarking requires changes to `benchmark.py` and `xpu_profiler.py`.

- Generated kernels are LLM-produced and must be validated. The mandatory `validate_triton.py` + `benchmark.py` correctness check catches most issues, but any production deployment should add its own regression tests.

## References

[1] Spoczynski, M., Fleischer, D., Berchansky, M., Stan, G. B., Guskin, S., Xu, W., Siemieniuk, A., Heinecke, A. 2026. "Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization for Intel GPU." arXiv:2605.26118. <https://arxiv.org/abs/2605.26118> — code: <https://github.com/IntelLabs/Xe-Forge>

[2] Hugging Face. 2025. "Kernels: Build compute kernels and load them from the Hub." <https://github.com/huggingface/kernels>

[3] Intel. 2025. "Intel XPU Backend for Triton." <https://github.com/intel/intel-xpu-backend-for-triton>

[4] Ouyang, Simon, et al. 2024. "KernelBench: Can LLMs Write Efficient GPU Kernels?" <https://github.com/ScalingIntelligence/KernelBench>
