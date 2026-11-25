---
title: "Hill-climbing Inkling 2 Mini inference: 40% faster on one RTX PRO 6000 with an agent, HF Jobs and Inference Endpoints"
thumbnail: /blog/assets/inkling-2-mini-inference-hillclimb/thumbnail.png
authors:
- user: nielsr
---

# Hill-climbing Inkling 2 Mini inference: 40% faster on one RTX PRO 6000 with an agent, HF Jobs and Inference Endpoints

Serving a new model fast is mostly a search problem: which engine, which flags, which speculative decoding setup, and which of them actually help on *your* GPU.

[Inkling 2 Mini](https://huggingface.co/blog/inkling2mini) came out last week, and it is not a simple model to serve. It is a looped MoE: 24 layers of weights, but the middle 12 layers run three times, so every token visits 48 layers. The weights ship in mixed NVFP4 and MXFP8, and the checkpoint bundles a DSpark drafter for speculative decoding. Both vLLM and SGLang support it, but only from nightly builds, and the recipes are written for B200- and GB300-class machines.

We wanted to know how fast it can go on a single RTX PRO 6000, the GPU you get with an `nvidia-rtx-pro-6000 x1` [Inference Endpoint](https://huggingface.co/docs/inference-endpoints). So we gave a coding agent the [`inference-optimisation`](https://github.com/sybil-solutions/inference-optimisation-skill) skill, the `hf` CLI and access to [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-overview), and let it hill-climb. One afternoon and 46 valid benchmark runs later, the deployed endpoint generates **190 tokens/s for a single user, up from 136**. It also passes long-context, concurrency and greedy-equivalence checks.

## TL;DR

- On 1× RTX PRO 6000, Inkling 2 Mini went from **136 → 190.6 tok/s** single-user decode (+40%) and **~570 → 716 tok/s** aggregate decode at 8 concurrent users (+26%).
- The winning config is vLLM nightly with DSpark speculative decoding (7 drafts, probabilistic draft sampling). It needs `--max-model-len 262144` and `--gpu-memory-utilization 0.85` to fit on a 96 GB card.
- The speed-up is lossless: temperature-0 outputs are byte-identical with and without speculation, and needles at 127K and 250K tokens are retrieved 3/3.
- Every experiment ran as an HF Job on the same GPU type the endpoint uses. The final config is deployed as an Inference Endpoint from a pinned image digest.
- Everything is public: the experiment log, ledgers, Job driver and deploy script are in [nielsrogge/inkling-2-mini-hillclimb](https://github.com/NielsRogge/inkling-2-mini-hillclimb).

## Table of Contents

- [The setup: a skill, Jobs and Endpoints](#the-setup-a-skill-jobs-and-endpoints)
- [Ceilings first](#ceilings-first)
- [The loop](#the-loop)
- [Results](#results)
- [What did not work](#what-did-not-work)
- [Measurement traps](#measurement-traps)
- [Qualifying and deploying](#qualifying-and-deploying)
- [Try it yourself](#try-it-yourself)
- [Closing words](#closing-words)

## The setup: a skill, Jobs and Endpoints

The `inference-optimisation` skill is a procedure for hill-climbing a model server, not a list of flags. Its rules are simple:

- **Compute the theoretical ceiling first**, so you know whether a number is good.
- **Change one thing per experiment.**
- **Run every candidate between two controls** (control, candidate, control). Accept it only if it beats both neighbours and regresses nothing.
- **Log everything**, including failures, in a `PLAN.md`.
- **Gate the winner** on long-context retrieval, a concurrent soak and a greedy equivalence check before calling it "best".

It ships a small, engine-agnostic benchmark harness (`bench.py`). The harness measures single-user decode (median of N), aggregate decode at concurrency N, cold prefill at several lengths, and TTFT. Token counts come from the server's usage stats, never from streamed chunks.

The skill was written for a local machine. We ran it on Hugging Face infrastructure instead:

- **Hugging Face Jobs** for compute. The `rtx-pro-6000` Job flavor is the same GPU as the `nvidia-rtx-pro-6000` Inference Endpoint ($2.75/h for both), so benchmark numbers carry over directly. Each Job runs one engine image and a list of server configs one after another. A ~150-line driver starts each server, runs an unmeasured warmup, calls `bench.py` unchanged, drains and stops the server, and prints one ledger line per run.
- **Inference Endpoints** for the final deployment, created with `hf endpoints deploy` from the exact image digest and flags that passed the gates.
- **The `hf` CLI** for both: `hf jobs run`, `hf jobs logs`, `hf endpoints deploy`, `hf endpoints describe`.

Since Jobs run on separate machines, several sweeps can run at once without disturbing each other's measurements. We typically had an SGLang sweep and a vLLM sweep in flight at the same time.

## Ceilings first

Before any tuning, the agent measured the card inside a Job: **1,469 GB/s** copy bandwidth, **416 TFLOPS** BF16 and **792 TFLOPS** FP8. It also read the tensor sizes from the safetensors header. Decode is memory-bandwidth bound, so the question is how many bytes each token has to read.

The looped architecture matters here. Every one of the 48 layer visits re-reads its attention weights and its 8-of-128 routed experts. Per token that is about 2.7 GB of attention/dense weights, about 2 GB of experts and 0.9 GB of LM head. Dividing measured bandwidth by bytes per token gives a single-user ceiling of **~256 tok/s without speculation**.

Speculative decoding changes the arithmetic in a way that is easy to miss on an MoE. Verifying 8 positions at once touches the *union* of their experts, about 40% of each layer's experts instead of 6%, so a verify step reads far more bytes than a plain decode step. At the acceptance we actually measured (about 2.3 tokens per step at temperature 1.0), the ceiling with DSpark is **~201 tok/s**. That number turned out to be the most useful one in the whole project.

## The loop

The agent started from what was already running in our org: SGLang with the FA4 attention backend and no speculative decoding. That is the config of the existing Inkling 2 Mini endpoints. It then explored two axes in parallel Jobs.

1. **Engine.** SGLang (`lmsysorg/sglang:inkling-2-mini`) vs vLLM nightly (`vllm/vllm-openai:nightly`), with no speculation.
2. **DSpark speculative decoding** in each engine, followed by every lossless knob each engine exposes for it.

When a knob was not documented, the agent read the engine source from the image itself, using a cheap `cpu-upgrade` Job that greps the installed package. That is how it found SGLang's `SGLANG_OPT_*` environment flags and vLLM's `speculative-config` fields such as `rejection_sample_method` and `enable_adaptive_verification`.

All runs used a fixed panel of six prompts: code, an explanation, a story, a comparison table, a bash script and a history summary. They used temperature 1.0 (the model's default), thinking on, 512 output tokens, and 12 repetitions for anything involving speculation. Speculation makes per-request speed depend on content, so this matters. On SGLang, DSpark single-user speed varied by ±4.5% between identical runs. That set the bar for accepting a change.

## Results

![Single-user decode speed of Inkling 2 Mini on one RTX PRO 6000 for four configurations](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inkling-2-mini-inference-hillclimb/results.png)

_Single-user decode speed, median of 12 requests. The deployed vLLM + DSpark config reaches about 95% of the bandwidth ceiling at its measured acceptance length._

| Config (1× RTX PRO 6000) | 1-user decode (tok/s) | 8-user aggregate (tok/s) | Prefill 4K / 32K (tok/s) | TTFT |
|---|---:|---:|---:|---:|
| SGLang, no speculation (existing endpoint config) | 134–139 | 543–598 | 18.7–19.1K / 16.9–19.1K | 41–114 ms |
| vLLM, no speculation | 157.0 | 638 | 20.7K / 20.7K | 33 ms |
| SGLang + DSpark (8 drafts) | 166.6 | — ¹ | 18.0–19.5K / 18.9–19.1K | 38–51 ms |
| **vLLM + DSpark (7 drafts)** | **190.6** | **716** | 20.7K / 20.6K | 40 ms |
| Deployed endpoint, measured over the internet | 194.9 | 673 | 11.7K / 19.2K ² | 198 ms ² |

¹ SGLang's DSpark recipe for RTX PRO 6000 caps `--max-running-requests` at 4. With 8 it reached 660 tok/s, still below vLLM.
² Includes about 160 ms of network round trip from our client to the endpoint.

Three changes account for all of the gain:

- **Engine choice: +17%.** Without speculation, vLLM nightly decodes at 157 tok/s against SGLang's 136 on this card. Prefill is on par once both are warm.
- **DSpark speculative decoding: +21% more on vLLM** (157 → 190.6). The four vLLM DSpark runs landed at 189.3, 190.4, 191.1 and 190.7. vLLM's per-request numbers repeat almost exactly from run to run, which made it much easier to hill-climb than SGLang.
- **Making DSpark fit on 96 GB.** Out of the box, vLLM's DSpark recipe fails to start on this card. It needs both flags below; see [Measurement traps](#measurement-traps) for why.

```bash
vllm serve thinkingmachines/Inkling-2-Mini \
  --reasoning-parser inkling \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

Why does it stop at 190? Measured acceptance is about 2.3 tokens per verify step. The per-position acceptance rate drops from 0.6 for the first draft to 0.33 and then 0.19, and each verify step reads ~40% of every layer's experts. Put those into the ceiling model and you get ~201 tok/s. 190.6 is about **95% of that ceiling**. Without speculation, vLLM reaches 61% of its 256 tok/s ceiling. So speculation did not just add speed: it moved the bottleneck from kernel latency back to memory bandwidth, where there is little left to take.

## What did not work

The skill insists on logging negative results, and on this model there were more of them than wins:

| Lever | Engine | Result |
|---|---|---|
| Shorter draft length (3–6) | both | Not possible. Both engines require 7 drafts (the checkpoint's block size − 1). |
| `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` | SGLang | +4% in one round, did not reproduce. Rejected. |
| `SGLANG_OPT_USE_FUSED_DRAFT_SAMPLE=1` | SGLang | +2%, inside noise. Rejected. |
| `SGLANG_OPT_MOE_QUANT_ONCE=1` | SGLang | +1%, inside noise. Rejected. |
| Greedy draft sampling | both | about −9% (SGLang) and −11% (vLLM) at temperature 1.0. Rejected. |
| `rejection_sample_method: "block"` | vLLM | 190.1 vs 190.3–190.5 for the controls, no effect. Rejected. |
| `enable_adaptive_verification` | vLLM | Not supported for the Inkling 2 drafter (assertion at startup). |
| `draft_lm_head_quantization: "fp8"` | vLLM | Only supported for DeepSeek DSpark drafters. |

Two lossy options were excluded by design: `rejection_sample_method: "fly"` and `"synthetic"`. They change the output distribution, and the goal was a speed-up you do not have to evaluate.

## Measurement traps

Most of the time did not go into tuning. It went into making sure the numbers were real. Four traps are worth sharing:

- **Triton JIT during the first measured request.** vLLM's first 4K-prefill measurement came out at 202 tok/s instead of 20K. The server log showed `Triton kernel JIT compilation during inference` exactly during those requests. The fix was an unmeasured warmup that covers every prompt length and the concurrent burst before `bench.py` starts. Even then, the first run in each Job under-reported 32K prefill, so those numbers are flagged in the ledger.
- **A server that never stopped.** The first driver killed the server's process group through a PID written by `setsid … &`, which is not guaranteed to be the group id. The "stopped" server kept the GPU and port 8000, so the next launch failed, and the dead process was a zombie that still looked alive. The driver now owns the server via `subprocess.Popen(start_new_session=True)`. Before each launch it waits until GPU memory is released *and* the port is free.
- **"Engine core initialization failed" is not the root cause.** vLLM DSpark failed twice with only that line at the end of the log. The real errors were 200 lines higher. First, the KV cache for the default 1M-token context does not fit next to the drafter (52.6 GiB needed, 48.1 GiB free). Second, once the context was capped, the KV cache took all remaining memory and the drafter's 5 GiB allocation went OOM. The driver now prints every error line in the server log, not just the tail.
- **Noise depends on the engine.** vLLM's speculative runs repeated to within 0.4%. SGLang's varied by 4.5%, enough to turn a +4% flag into a false win once. Repeating every "win" between fresh controls caught it.

## Qualifying and deploying

Before deploying, the winning config was re-run on a **pinned image digest** (`vllm/vllm-openai@sha256:eaa63fd6…`). This is the same vLLM build used during the hill-climb (`0.34.0rc1.dev1235+gf2f9edcac`), confirmed from inside the Job. It then had to pass the skill's acceptance gates:

| Gate | Result |
|---|---|
| Needle retrieval at ~50% of context (126,537 prompt tokens) | 3/3, TTFT 12.5 s |
| Needle retrieval at ~95% of context (249,970 prompt tokens) | 3/3, TTFT 28.7 s |
| Concurrent soak: 4 × ~58K-token requests at once | 4/4 succeeded, 3/3 retrieval each |
| Greedy equivalence, DSpark vs no speculation (6 prompts, temperature 0) | 6/6 byte-identical |
| Speed on the pinned image | 189.9 / 190.2 tok/s single-user |

Then a single `hf endpoints deploy` call created the endpoint. It uses the custom image by digest and pins the model revision. It runs `vllm serve /repository` with the flags above, uses `authenticated` access, and scales to zero after 60 idle minutes:

```bash
hf endpoints deploy inkling2mini-tuned-dspark --namespace <your-org> \
  --repo thinkingmachines/Inkling-2-Mini --revision 7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6 \
  --framework custom --custom-image "vllm/vllm-openai@sha256:eaa63fd6ee3aac77f3158e80ed79ec425db3a53f7cd666bf1954e96abfaf19a7" \
  --health-route /health --port 8000 \
  --vendor aws --region us-east-2 --accelerator gpu --instance-type nvidia-rtx-pro-6000 --instance-size x1 \
  --min-replica 0 --max-replica 1 --scale-to-zero-timeout 60 --type authenticated \
  --container-command "vllm serve" \
  --container-args "/repository --served-model-name thinkingmachines/Inkling-2-Mini --host 0.0.0.0 --port 8000 --reasoning-parser inkling --max-model-len 262144 --gpu-memory-utilization 0.85 --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}'"
```

Running the same `bench.py` against the live endpoint gave 194.9 tok/s single-user, the same as the Jobs measurement.

## Try it yourself

Everything needed to reproduce this, or to run the same loop on another model, is in [nielsrogge/inkling-2-mini-hillclimb](https://github.com/NielsRogge/inkling-2-mini-hillclimb):

- `PLAN.md`: the full experiment log, with every run, verdict and reason, including the failures above.
- `THEORY.md`: the measured hardware and ceiling arithmetic.
- `jobs/driver.py`: runs a list of server configs inside one Job, with warmup, `bench.py` and optional gates (needles, soak, greedy dump).
- `submit.sh`: `./submit.sh rtx-pro-6000 <image> <job-name> configs/plan.json`
- `deploy.sh`: creates the endpoint from the qualified config.
- `ledger-*.jsonl`: every raw measurement.

To point it at another model, write a `configs/plan-*.json` with one server command per run, interleaving controls and candidates, and submit it to the Job flavor that matches your target endpoint.

## Closing words

The interesting part of this experiment is not the final flag set. It is how little had to be invented. The skill provided the discipline: ceilings first, one change at a time, controls on both sides, and gates before "best". Jobs provided a GPU identical to the deployment target, by the minute, with as many parallel sweeps as we wanted. Inference Endpoints turned the winning Job config into a production endpoint with one command.

Two things would move the number further, and both are beyond config tuning. The first is a drafter with higher acceptance at temperature 1.0. The second is a verify path that does not have to read 40% of the experts per layer. If you build either, the harness above will tell you within one Job whether it helped.
