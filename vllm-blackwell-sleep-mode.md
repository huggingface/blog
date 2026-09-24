---
title: "Frontier MoE sleep/wake at TP=4 on consumer Blackwell — 2s wake, 4.5s swap"
thumbnail: /blog/assets/vllm-blackwell-sleep-mode/thumbnail.png
authors:
- user: Doradus-AI
  guest: true
---

# Frontier MoE sleep/wake at TP=4 on consumer Blackwell — 2s wake, 4.5s swap

> **Full debugging story is at the canonical post on [doradusresearch.ai/blog/vllm-blackwell-sleep-mode](https://doradusresearch.ai/blog/vllm-blackwell-sleep-mode/) and the prebuilt image is at `ghcr.io/doradusresearch/vllm-blackwell-sm12x-bundle:v4`.** What follows is the condensed version — the result, the recipe, and the gotchas worth landing on the HF index.

## The result

vLLM `--enable-sleep-mode` model rotation, in production on 4× RTX PRO 6000 Blackwell:

- **`/wake_up`: ~2 seconds**
- **Cross-peer swap (sleep model A, wake model B on the same 4 GPUs): ~4.5 seconds** — down from ~50
- Live load: **DeepSeek-V4-Flash + MiMo-V2.5-Flash** sharing the same TP=4 pool, hot-rotating on demand
- Numbers above were measured across three live cycles right before publication (4.40s / 4.48s / 4.50s)

Sleep/wake for small dense models on a single GPU is trivial — sub-second, llama-swap solves it. The interesting case is frontier MoE at TP=4: ~17 GiB of sharded weights per GPU, sparse routing tables, KV cache across four ranks, plus a co-resident peer model competing for the same physical cards. For months that combination on consumer Blackwell either OOM'd, hung in `cuMemMap`, or produced silent garbage outputs after a few rotation cycles.

## The recipe (TL;DR)

1. **Base on PR [#41834](https://github.com/vllm-project/vllm/pull/41834)** for SM12x DSv4 enablement (Triton sparse-MLA fallback, DeepGEMM-free paths, MLA prefix-cache fix).
2. **Cherry-pick PR [#35489](https://github.com/vllm-project/vllm/pull/35489)** — one-line `error_code = no_error;` reset that fixes `cuMemMap` EINVAL on consumer Blackwell. Open since March, not in any released image.
3. **Cherry-pick PR [#34600](https://github.com/vllm-project/vllm/pull/34600)** — proper rollback in `wake_up` on partial alloc failure.
4. **Drop `--gpu-memory-utilization` from 0.85 to 0.60 for DSv4 specifically (initial draft said 0.70 — corrected after observing MiMo's real sleep residue is ~9 GiB/GPU, not the ~4 GiB vLLM's docs assume).** MiMo (no sparse-MLA workspace) can stay at 0.85.
5. *(Optional, recommended for tighter margins)* Apply the SM12x workspace-shrink patches from **[vllm#42856](https://github.com/vllm-project/vllm/pull/42856)** — three arch-gated hunks that reduce PR #41834's 22 GiB workspace footprint to ~7 GiB on consumer Blackwell. Already baked into the GHCR image.

PR #41834 ships ~22 GiB of non-cumem GPU state per GPU (sparse-MLA workspace, marlin scratch, cuda-graph private pools) that lives **outside** vLLM's cumem allocator's budget. On H200 (140 GiB) this is invisible headroom. On consumer Blackwell (95 GiB) at 0.85 utilization it overflows. PyTorch's OOM message labels this `22.7 GiB allocated in private pools (e.g., CUDA Graphs)`, which sent us chasing graph-capture configuration for days — `torch.cuda.MemPool` catches *any* tensor allocated via a MemPool, including non-graph workspace tensors.

## Live config (both models)

| Setting | DSv4-Flash | MiMo-V2.5-Flash |
|---|---|---|
| `--tensor-parallel-size` | 4 | 4 |
| `--max-model-len` | 131072 (128K) | 65536 (64K) |
| `--max-num-seqs` | 12 | 6 |
| `--gpu-memory-utilization` | **0.60** | 0.85 |
| `--enable-sleep-mode` | ✓ | ✓ |

## First-time cold load

Budget **~12 minutes per model** from `docker run` to first request — that's safetensors load (DSv4 is 46 shards, ~14 s/shard average from local NVMe) plus CUDA-graph capture. Image pull adds more on a fresh host (the bundle is ~29 GB). Subsequent restarts on the same host are dominated by the shard load — page-cache hits help but don't eliminate it. Plan accordingly.

> **Honest status note (2026-05-17 — bundle:v5 validation, DeepGEMM gap, multi-cycle proof, v6 attempt + architectural fork):** Both bundle images are now on GHCR (anonymous pulls): `ghcr.io/doradusresearch/vllm-blackwell-sm12x-bundle:v4` (DSv4) + `:v5` (MiMo + non-sparse-MLA). **MiMo on bundle:v5 is validated cycle-stable** — three back-to-back wake/sleep cycles measured 2.66s/2.99s, 2.57s/2.92s, 2.56s/2.94s (steady ~2.6s wake / ~2.95s sleep). First `/sleep` of a fresh cold-load is ~49s (CUDA-graph teardown + cumem private-pool init, paid once per process, not per cycle). **DSv4 on bundle:v5 failed at engine init** with `RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM to be installed` from the upstream layer-level `sparse_attn_indexer.py:442`. DeepGEMM doesn't ship SM_120 kernels, so install-it doesn't fix it. **We tried to ship a unified bundle:v6** — ported v4's SM_120 Triton kernels into v5 (sm12x_mqa.py + sm12x_deep_gemm_fallbacks.py + sm12x_fp8_einsum.py), wired SM_120 dispatch into v5's `vllm/utils/deep_gemm.py` for `fp8_fp4_mqa_logits` + `fp8_fp4_paged_mqa_logits` + `tf32_hc_prenorm_gemm`, relaxed the sparse_attn_indexer raise, and forced `_einsum_recipe=(1,128,128)` + `_tma_aligned_scales=False` on SM_120. **DSv4 still failed**: `DeepSeek V4 fp8 einsum weight rows must be divisible by out_rank=1024, got 256`. The wo_a weight tensor from v5's loader is `(256, 4096)` per-rank, but v4's Triton dispatcher requires `b.shape[0] ≥ out_rank` — v5 shards wo_a across TP ranks (each rank gets 1/N slice), v4 stored multi-group-concat and computed group offsets inside the kernel. **Fundamental weight-layout fork**, not a recipe gap. Punted bundle:v6 to a follow-up — the right place is probably upstream in vLLM PR #41834's wake. **Production rotation pool stays hybrid (live, validated):** DSv4 on bundle:v4, MiMo on bundle:v5. **10 back-to-back swap operations (5 full cycles in both directions), all OK, cross-peer swap held a tight 4.55–4.81s band across all 10 ops (no drift, no cycle leak). MiMo decode post-rotation: 137–143 tok/s essentially flat from 1.6K to 39K input tokens.** Two images, one rotation pool. **Architectural observation:** bundle:v5 is model-agnostic for everything except sparse-MLA / Lightning Indexer family — dense, dense-MoE, and non-sparse-MLA models all rotate cleanly on it today. Only DSv4-class models drag in the DeepGEMM dependency.

## What it doesn't fix

Sleep-mode is **attention-type sensitive** on consumer Blackwell:

| Model | Attention | Status |
|---|---|---|
| DeepSeek-V4-Flash | sparse-MLA (Triton fallback via PR #41834) | Full sleep/wake at TP=4. Live. |
| MiMo-V2.5-Flash | dense MLA | Full sleep/wake at TP=4. Live. |
| Q3-Coder-Next-80B-A3B | hybrid DeltaNet + SWA | `/sleep level=2` does not release cleanly on cu129 — always-awake. Tracking [vllm#41602](https://github.com/vllm-project/vllm/pull/41602). |

And **decode TPS is PCIe-sync-bound**: profiling shows `nccl_allreduce` at 55.1 s and `sparse_accumulate_indexed_attention` at 117.9 s cumulative. Compute isn't the cap; inter-rank PCIe round-trip latency at TP=4 is. Batched serving amortizes it; single-stream latency-sensitive decode of DSv4 is not the right workload for this stack.

## Decode TPS across context (DSv4-Flash, TP=4, single-stream)

Live sweep against the production rotation pool. Each request requested 256 output tokens forced by an instruction-structured prompt.

| Input tokens | TTFT (s) | Decode TPS | E2E (s) |
|---:|---:|---:|---:|
| 477 | 0.54 | **69.7** | 4.2 |
| 1,724 | 0.68 | 68.9 | 4.4 |
| 6,669 | 2.09 | 66.0 | 6.0 |
| 26,535 | 9.37 | 58.6 | 13.7 |
| 52,980 | 17.34 | 51.8 | 22.3 |
| 79,468 | 23.06 | 45.8 | 28.6 |
| 99,291 | 21.12 | **41.9** | 27.2 |

Decode degrades from ~70 → ~42 tok/s across the full advertised context — that's the PCIe-sync curve in numbers. Useful for chat-length and document-review workloads; long-context retrieval at 100K is workable but latency is prefill-bound, not decode-bound.

## Try it

```bash
docker pull ghcr.io/doradusresearch/vllm-blackwell-sm12x-bundle:v4
```

Bundles PR #41834 + #35489 + #34600 + SM12x-gated workspace shrinks on top of `vllm/vllm-openai:v0.20.2-cu129-ubuntu2404`. Apache-2.0.

→ **Full debugging story** (including the upstream-fixing PR #42856 we contributed back from this work): **[doradusresearch.ai/blog/vllm-blackwell-sleep-mode](https://doradusresearch.ai/blog/vllm-blackwell-sleep-mode/)**.
