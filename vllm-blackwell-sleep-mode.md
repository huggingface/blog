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
4. **Drop `--gpu-memory-utilization` from 0.85 to 0.70 for DSv4 specifically.** MiMo (no sparse-MLA workspace) can stay at 0.85.
5. *(Optional, recommended for tighter margins)* Apply the SM12x workspace-shrink patches from **[vllm#42856](https://github.com/vllm-project/vllm/pull/42856)** — three arch-gated hunks that reduce PR #41834's 22 GiB workspace footprint to ~7 GiB on consumer Blackwell. Already baked into the GHCR image.

PR #41834 ships ~22 GiB of non-cumem GPU state per GPU (sparse-MLA workspace, marlin scratch, cuda-graph private pools) that lives **outside** vLLM's cumem allocator's budget. On H200 (140 GiB) this is invisible headroom. On consumer Blackwell (95 GiB) at 0.85 utilization it overflows. PyTorch's OOM message labels this `22.7 GiB allocated in private pools (e.g., CUDA Graphs)`, which sent us chasing graph-capture configuration for days — `torch.cuda.MemPool` catches *any* tensor allocated via a MemPool, including non-graph workspace tensors.

## Live config (both models)

| Setting | DSv4-Flash | MiMo-V2.5-Flash |
|---|---|---|
| `--tensor-parallel-size` | 4 | 4 |
| `--max-model-len` | 131072 (128K) | 65536 (64K) |
| `--max-num-seqs` | 12 | 6 |
| `--gpu-memory-utilization` | **0.70** | 0.85 |
| `--enable-sleep-mode` | ✓ | ✓ |

## First-time cold load

Budget **~12 minutes per model** from `docker run` to first request — that's safetensors load (DSv4 is 46 shards, ~14 s/shard average from local NVMe) plus CUDA-graph capture. Image pull adds more on a fresh host (the bundle is ~29 GB). Subsequent restarts on the same host are dominated by the shard load — page-cache hits help but don't eliminate it. Plan accordingly.

## What it doesn't fix

Sleep-mode is **attention-type sensitive** on consumer Blackwell:

| Model | Attention | Status |
|---|---|---|
| DeepSeek-V4-Flash | sparse-MLA (Triton fallback via PR #41834) | Full sleep/wake at TP=4. Live. |
| MiMo-V2.5-Flash | dense MLA | Full sleep/wake at TP=4. Live. |
| Q3-Coder-Next-80B-A3B | hybrid DeltaNet + SWA | `/sleep level=2` does not release cleanly on cu129 — always-awake. Tracking [vllm#41602](https://github.com/vllm-project/vllm/pull/41602). |

And **decode TPS is PCIe-sync-bound**: profiling shows `nccl_allreduce` at 55.1 s and `sparse_accumulate_indexed_attention` at 117.9 s cumulative. Compute isn't the cap; inter-rank PCIe round-trip latency at TP=4 is. Batched serving amortizes it; single-stream latency-sensitive decode of DSv4 is not the right workload for this stack.

## Try it

```bash
docker pull ghcr.io/doradusresearch/vllm-blackwell-sm12x-bundle:v4
```

Bundles PR #41834 + #35489 + #34600 + SM12x-gated workspace shrinks on top of `vllm/vllm-openai:v0.20.2-cu129-ubuntu2404`. Apache-2.0.

→ **Full debugging story** (including the upstream-fixing PR #42856 we contributed back from this work): **[doradusresearch.ai/blog/vllm-blackwell-sleep-mode](https://doradusresearch.ai/blog/vllm-blackwell-sleep-mode/)**.
