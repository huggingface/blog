---
title: "Intel XPU Kernel Skills: LLM-driven Triton kernel optimization for the Hugging Face Kernel Hub"
thumbnail: /blog/assets/intel-kernel-skills/banner.png
authors:
- user: danf
  guest: true
  org: Intel
- user: mber
  guest: true
  org: Intel
- user: moshew
  guest: true
  org: Intel
---

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-kernel-skills/xpu-kernels-figure.jpg" width=700 alt="An LLM-driven workflow generates and benchmarks Triton kernels for an Intel XPU GPU." />
</p>

# Intel XPU Kernel Skills: LLM-driven Triton kernel optimization for the Hugging Face Kernel Hub

*By Intel AI Software Group*

The **[xpu-kernels skill](https://github.com/huggingface/kernel-builder/tree/main/skills/xpu-kernels)** is an Agent Skill that ships inside [`kernel-builder`](https://github.com/huggingface/kernels) and turns a coding agent (Claude Code, OpenCode, Cursor, etc.) into an autonomous Triton kernel writer for **Intel XPU GPUs** (Battlemage G21 / Arc Pro B50). It is adapted from **[Xe-Forge](https://github.com/IntelLabs/Xe-Forge)** — an LLM-driven optimization framework that transforms PyTorch code into fast Triton kernels — and is designed to plug directly into the **[Hugging Face Kernel Hub](https://huggingface.co/kernels)** so that the kernels you generate can be loaded with `get_kernel(...)` like any other Hub kernel.

We release three skills together — **`cuda-kernels`**, **`rocm-kernels`**, and **`xpu-kernels`** — each tuned to its hardware vendor. This post focuses on the Intel XPU one and shows that:

- 🤖 An LLM agent equipped with the right tools and knowledge base can autonomously turn a PyTorch reference into an optimized Triton kernel for Intel XPU.

- ⚡ A branching trial-loop (analyze → validate → benchmark → profile → finalize) consistently finds speedups over both PyTorch eager and the naive Triton baselines on Battlemage / Arc Pro B50.

- 📦 The resulting kernels are packaged so they can be uploaded to the [Hugging Face Kernel Hub](https://huggingface.co/kernels) and consumed via the [`kernels`](https://github.com/huggingface/kernels) Python package.

👉 Skill: <https://github.com/huggingface/kernel-builder/tree/main/skills/xpu-kernels> \
👉 Original framework: <https://github.com/IntelLabs/Xe-Forge> \
👉 Kernel Hub: <https://huggingface.co/kernels>

## Why a kernel skill?

Writing fast GPU kernels is hard, and writing them for a *new* architecture is harder. Intel's XPU stack — Xe2 with tensor descriptors, GRF mode 256, tile swizzling — is well documented but not yet baked into most LLM training corpora. As a result, an off-the-shelf coding agent asked for "a fast Triton GEMM on Intel Arc" will happily produce CUDA-flavored code that either does not compile, runs slowly, or silently returns wrong results.

We focused on two goals:

1. **Give the agent the right tools** — analysis, validation, benchmarking, and profiling — so it can close the loop on Intel XPU hardware without writing custom scripts.

2. **Give the agent the right knowledge** — XPU-specific patterns, hard correctness constraints, and integration recipes — packaged as references that are read at the start of every session.

The **xpu-kernels** skill bundles both: a small CLI toolbox under `scripts/` and a curated knowledge base under `references/`, all wrapped in a `SKILL.md` that defines the workflow.

## How It Works

- **Hardware:** Intel Battlemage G21 / Arc Pro B50 (Xe2, 128 XVEs, ~500 GB/s).
- **Compiler:** [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton).
- **Harness:** [ai-bench](https://github.com/libxsmm/AI-bench) measures correctness and performance against a PyTorch (or naive Triton) baseline.
- **Profiler:** Intel VTune 2025+ is optionally invoked for hardware counters; it can be disabled in `scripts/config.yaml`.
- **Hub integration:** generated kernels follow the [Kernel Hub layout](https://github.com/huggingface/kernels) so they can be published and then loaded with `kernels.get_kernel("<org>/<name>")`.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-kernel-skills/xe-forge-loop.png" width=600 alt="The xpu-kernels trial loop: analyze, validate, benchmark, profile, finalize." /><br>
<em>Figure 1: The xpu-kernels trial loop. CPU-only steps (analyze, validate, trial bookkeeping) can run in parallel; GPU steps (benchmark, profile) are serialized on the single XPU.</em>
</p>

- **Agent Interface:** the skill's `SKILL.md` instructs the agent to read `scripts/config.yaml` and the relevant reference docs, then enter a fixed trial loop. The agent only writes Triton kernel files (`*_triton.py` or trial files `t<id>.py`) — never benchmark or test scripts.

- **Trial tree:** `trial_manager.py` keeps a branching tree of attempts. The agent can branch back to the best trial and try a different optimization strategy when it regresses or plateaus.

- **Design Goals:**

  - **Reproducibility:** the same skill, knowledge base, and tool versions produce comparable trial trees across runs.

  - **Safety:** GPU jobs are explicitly serialized (one XPU, one job at a time). Validation runs CPU-only before any GPU time is spent.

  - **Hub-readiness:** the finalized kernel is a self-contained Triton file that fits the Kernel Hub packaging used by [`kernels-community`](https://huggingface.co/kernels-community).

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-kernel-skills/output-example.png" width=800 alt="A short Triton kernel for Intel XPU using tensor descriptors, GRF mode 256, and tile swizzling, produced by the agent."/><br>
<em>Figure 2: A trial produced by the agent — a fused GEMM+Sigmoid kernel using tensor descriptors, GRF mode 256, and tile swizzling. The skill's references nudge the model toward these XPU-specific patterns.</em>
</p>

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

We evaluated the skill on Intel Battlemage G21 / Arc Pro B50 (128 XVEs, bf16 unless noted). Speedup is the median over ai-bench trials versus the indicated baseline.

- **KernelBench Level 2 — fused kernels (bf16):** GEMM+Sigmoid+Scaling+ResidualAdd, GEMM+GELU+Softmax, Conv+BatchNorm+ReLU, and other fused patterns vs. PyTorch eager.

- **Flash Attention Forward (fp16):** vs. the Flash Attention kernel shipped in the Intel XPU Triton backend, across multiple sequence lengths.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-kernel-skills/main-results.png" width=1000 alt="Speedup table across KernelBench Level 2 and Flash Attention forward."/>
</p>

- We compare three configurations: **PyTorch eager**, the **naive Triton baseline** shipped in `test_kernels/`, and the **agent-optimized Triton** produced by the skill. The naive baselines exist to demonstrate that the speedup is not "Triton vs. eager" — it is the *optimized* Triton, with tensor descriptors, GRF 256, and swizzling, that wins.

- The agent reliably discovers the Intel-specific patterns when the references are in scope; without them, the same model produces CUDA-flavored Triton that often validates but rarely speeds up.

**Key insight:** the bottleneck for LLM-driven kernel optimization on a less-represented architecture is *knowledge access*, not raw model capability. A small, curated reference set plus a strict tool-driven loop is enough to make a general coding agent productive on Intel XPU.

## Why It Matters

- **Vendor coverage:** the same `kernel-builder` skill format covers CUDA, ROCm, and now Intel XPU, so kernels for all three vendors can be authored and published consistently.

- **Reproducibility:** the strict trial loop + `config.yaml` + cached baseline times make runs comparable across users and machines.

- **Hub-native:** the output is ready to be loaded with `get_kernel(...)`, so optimized XPU kernels become a one-line dependency for downstream Hugging Face users.

## Conclusion

The xpu-kernels skill demonstrates a practical, reproducible way to bring LLM-driven kernel optimization to a less-represented GPU architecture. By packaging Xe-Forge's tools, knowledge base, and trial loop as a `kernel-builder` skill, we give any compatible coding agent the missing context to produce correct and fast Triton kernels for Intel XPU — and to publish them on the Hugging Face Kernel Hub alongside CUDA and ROCm kernels.

## Try It Yourself

- Skill: <https://github.com/huggingface/kernel-builder/tree/main/skills/xpu-kernels>
- `kernels` Python package: <https://github.com/huggingface/kernels>
- Original framework: <https://github.com/IntelLabs/Xe-Forge>

Contributions, issues, and new reference patterns are welcome. 🚀

## Citation

If you use the xpu-kernels skill in your research, please cite:

```bibtex
@software{xpu_kernel_skill_2025,
  author    = {Fleischer, Daniel and Berchansky, Moshe and Wasserblat, Moshe},
  title     = {xpu-kernels: An LLM-driven Triton Kernel Skill for Intel XPU},
  year      = {2025},
  publisher = {Intel AI Labs},
  url       = {https://github.com/huggingface/kernel-builder/tree/main/skills/xpu-kernels}
}
```

## Limitations & Future Work

- **Hardware scope:** verified on Battlemage G21 / Arc Pro B50 (Xe2). Other Intel XPUs may require updated patterns in `references/xpu_optimizations.yaml`.

- **Workload scope:** the knowledge base focuses on GEMM, fused KernelBench Level 2 patterns, reductions, and Flash Attention forward. Backward passes, sparse, and quantized kernels are future work.

- **Single-XPU serialization:** the loop assumes one XPU per machine. Multi-GPU benchmarking requires changes to `benchmark.py` and `xpu_profiler.py`.

- Generated kernels are LLM-produced and must be validated. The mandatory `validate_triton.py` + `benchmark.py` correctness check catches most issues, but any production deployment should add its own regression tests.

## References

[1] Intel Labs. 2025. "Xe-Forge: LLM-driven Triton kernel optimization for Intel XPU." <https://github.com/IntelLabs/Xe-Forge>

[2] Hugging Face. 2025. "Kernels: load compute kernels from the Hub." <https://github.com/huggingface/kernels>

[3] Hugging Face. 2025. "kernel-builder: build kernels for the Hugging Face Kernel Hub." <https://github.com/huggingface/kernel-builder>

[4] Intel. 2025. "Intel XPU Backend for Triton." <https://github.com/intel/intel-xpu-backend-for-triton>

[5] Ouyang, Simon, et al. 2024. "KernelBench: Can LLMs Write Efficient GPU Kernels?" <https://github.com/ScalingIntelligence/KernelBench>
