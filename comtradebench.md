---
title: "ComtradeBench: An OpenEnv Benchmark for Reliable LLM Tool-Use Under Adversarial Conditions"
thumbnail: /blog/assets/comtradebench/thumbnail.png
authors:
- user: yonghongzhang
---

<p align="center">
  <img src="/blog/assets/comtradebench/banner.png" width="100%" alt="ComtradeBench — An OpenEnv Benchmark for Reliable LLM Tool-Use"/>
</p>

<p align="center">
  <a href="https://github.com/yonghongzhang-io/comtrade-openenv">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub"/>
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/yonghongzhang/comtrade-env">
    <img src="https://img.shields.io/badge/HF%20Space-Live%20Demo-FFD21E?logo=huggingface&logoColor=black" alt="HF Space"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/OpenEnv-Native-4B8BBE" alt="OpenEnv"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Tasks-10-brightgreen" alt="10 Tasks"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Training-GRPO-orange" alt="GRPO"/>
</p>

<p align="center"><em>AgentBeats Phase 2 — OpenEnv Challenge Submission &nbsp;|&nbsp; Author: MateFin</em></p>

---

## Agents should be judged by whether they finish the job

Large language models are often evaluated on what they can **say**.  
Real agents, however, are judged by whether they can **finish the job** when tools fail.

In practical API workflows, failure rarely comes from language alone. Pages drift. Duplicate rows appear across requests. Rate limits interrupt execution. Transient server errors force retries. Summary rows contaminate aggregates. Budgets make brute-force strategies impossible.

These are not unusual edge cases. **They are normal operating conditions for production systems.**

ComtradeBench is an OpenEnv benchmark designed to measure exactly this problem: can an LLM agent execute a multi-step API workflow reliably under realistic failure modes?

---

## Why this benchmark matters

Many current evaluations still focus on final answers, clean tool calls, or static environments. But deployed agents fail for more operational reasons:

| Failure | What goes wrong |
|---------|----------------|
| Miss pages | Incomplete data submitted as complete |
| Retry incorrectly | Page skipped after error — silent data gap |
| Double-count duplicates | Overcounted rows, inflated aggregates |
| Leak summary rows | Contaminated totals corrupt downstream analysis |
| Waste budget | Redundant fetches exhaust request limit |
| Recover silently | No auditable trace — failure invisible in production |

These are **execution failures**, not just reasoning failures.

If we want useful agents, we need benchmarks that measure reliable task completion under imperfect conditions — not only answer quality in idealized settings.

---

## What ComtradeBench is

> ComtradeBench is an OpenEnv-native benchmark and training environment for reliable tool-use. The domain is trade-data retrieval; the problem is broader: robust multi-step API execution under shifting, imperfect, and partially adversarial conditions.

The environment asks an agent to retrieve, clean, and submit records from a paginated API while handling:

- **Pagination drift** — page ordering randomized between calls
- **Duplicate records** — within-page (8%) and cross-page (3%) overlap
- **Transient errors** — HTTP 429 rate-limits and HTTP 500 server faults
- **Totals trap** — synthetic summary rows mixed into real data
- **Mixed faults** — rate-limit retry + dedup simultaneously
- **Constrained budget** — halved request limit, no room for waste

The goal is not to test whether the agent can *describe* the workflow.  
The goal is to test whether it can *execute* it — correctly, completely, efficiently, and robustly.

---

## Environment design

Each episode gives the agent a parameterized retrieval task and a limited request budget. The agent interacts through **three MCP tools only**:

```
get_task_info()         →  task parameters + request budget
fetch_page(page, size)  →  {rows, has_more}  or  {status: 429|500, retry: true}
submit_results(...)     →  {reward, score, breakdown}
```

The benchmark is structured as a **curriculum of ten tasks**:

| # | Task | Core challenge |
|---|------|----------------|
| T1 | Single page | Baseline correctness |
| T2 | Multi-page pagination | Merge 2,345+ rows across pages |
| T3 | Duplicates | Primary-key deduplication |
| T4 | HTTP 429 | Backoff + retry without data loss |
| T5 | HTTP 500 | Transient error recovery |
| T6 | Page drift | Canonicalize under non-deterministic ordering |
| T7 | Totals trap | Filter `is_total=true` rows |
| T8 | Mixed faults | Retry AND dedup simultaneously |
| **T9** | **Adaptive adversary** | **Fault intensity escalates mid-episode** |
| **T10** | **Constrained budget** | **50 requests instead of 100** |

T9 is, to our knowledge, among the earliest OpenEnv-style tasks to model **within-episode fault escalation** — where the environment becomes harder as the agent makes progress.

---

## Why OpenEnv

We built ComtradeBench on OpenEnv because this benchmark is meant to be more than a one-off simulator.

OpenEnv gives us a standard environment interface, reproducible execution, and clean integration with evaluation and post-training workflows. The same environment code runs both in-process during GRPO training and as a deployed Docker service during evaluation — with no divergence.

Our goal is not only to score agents, but to provide a **reusable environment where robustness can be studied and trained systematically**.

---

## Scoring what actually matters

ComtradeBench uses structured evaluation across **six dimensions** — not a binary pass/fail:

| Dimension | Weight | What it measures |
|-----------|:------:|-----------------|
| Correctness | **30%** | All expected rows present with correct field values |
| Completeness | 15% | Zero missing records |
| Robustness | 15% | Correct fault handling with logged evidence |
| Efficiency | 15% | Request count vs. task-optimal minimum |
| Data Quality | 15% | No duplicates or leaked totals rows |
| Observability | 10% | Structured execution trace in the run log |

**Why multi-dimensional scoring matters:**  
An agent that retrieves correct data but skips retry logging loses 15 points on Robustness. An agent that skips pages to save budget loses Completeness and all Efficiency credit. These behaviors are not equivalent — the benchmark does not treat them as equivalent.

The **Observability** dimension deserves special note: requiring structured log entries incentivizes the agent to maintain explicit execution state. This is not artificial — structured logs are how production ETL pipelines are monitored and debugged.

---

## Baselines and results

### Rule-based baseline (no LLM)

A deterministic rule-based agent achieves **96.8 / 100** average across all ten tasks, confirming the environment is well-calibrated and solvable.

| Task | Score | Reward |
|------|------:|-------:|
| T1 Single page | 98.0 | 0.980 |
| T2 Multi-page | 98.0 | 0.980 |
| T3 Duplicates | 98.0 | 0.980 |
| T4 Rate limit (429) | 95.0 | 0.950 |
| T5 Server error (500) | 95.7 | 0.957 |
| T6 Page drift | 94.0 | 0.940 |
| T7 Totals trap | 98.0 | 0.980 |
| T8 Mixed faults | 96.4 | 0.964 |
| T9 Adaptive adversary | 96.9 | 0.969 |
| T10 Constrained budget | 98.0 | 0.980 |
| **Average** | **96.8** | **0.968** |

### LLM agent — Kimi / Moonshot V1-128k (apples-to-apples across all 10 tasks)

All 10 tasks run under the same `moonshot-v1-128k` variant at `temperature=0.0`, `seed=42`.

| Task | Score | Reward | Delta vs baseline |
|------|------:|-------:|------------------:|
| T1 Single page | 98.7 | 0.987 | +0.7 |
| T2 Multi-page | 98.7 | 0.987 | +0.7 |
| T3 Duplicates | 98.7 | 0.987 | +0.7 |
| T4 Rate limit (429) | 95.7 | 0.957 | +0.7 |
| T5 Server error (500) | 96.3 | 0.963 | +0.6 |
| T6 Page drift | 94.7 | 0.947 | +0.7 |
| T7 Totals trap | 98.7 | 0.987 | +0.7 |
| T8 Mixed faults | 97.3 | 0.973 | +0.9 |
| T9 Adaptive adversary | 97.5 | 0.975 | +0.6 |
| T10 Constrained budget | 98.7 | 0.987 | +0.7 |
| **Average (T1-T10)** | **97.5** | **0.975** | **+0.7** |

Kimi-128k matches or slightly exceeds the rule-based baseline on **all 10 tasks**. But the
interesting findings are not in this table — they are in the cross-model and ablation data below.

### Cross-model comparison — five LLMs, four independent findings

Five LLMs (four closed/open frontier-class models plus one open-source mid-size baseline), same agent loop, same default prompt, seed 42 baseline plus 5-seed multi-run on T9:

| Model | T1-T8 avg | T9 score | T10 score | T1-T10 avg |
|-------|----------:|---------:|----------:|-----------:|
| Rule-based baseline | 96.5 | 96.9 | 98.0 | 96.8 |
| **Kimi Moonshot V1-128k** | **97.4** | **97.5 (std 0.0 across 5 seeds)** | **98.7** | **97.5** |
| **Claude Sonnet 4.6** | **97.4** | **97.5** | **98.7** | **97.5** |
| **Qwen2.5-7B-Instruct** ⭐ (open, zero-shot) | **97.2** | **97.5** | **98.7** | **97.2** |
| **GPT-5** | 95.0 | **75.7** | 95.7 | 93.2 |
| Llama 3.3 70B (Groq) | 97.4 | 18.7 – 97.5 (bimodal)† | 95.7 | 89.3 |

† Llama T9 is bimodal: the seed-42 run we originally published hit 18.7, but re-running on
{42, 137, 2024, 7, 31} produced {97.5, 94.5, 0, 0, 0} — and the zeros turn out to be **Groq
daily-token-limit 429s**, not model failures. On the two seeds that actually ran to completion
Llama matches frontier. The correct statement is *Llama on T9 is high-variance*, not *Llama
collapses uniformly*.

**Four independent findings pop out.**

**1. T9 separates execution-oriented from reasoning-oriented frontier.**
Kimi / Claude execute T9 in ~8 s with 7 tool calls and score 97.5. GPT-5 "thinks" for ~223 s with
*2* tool calls and scores **75.7** — a 21.8-point gap between frontier models that a pass/fail
benchmark would completely miss. The breakdown tells the story: GPT-5's Efficiency drops to 6/15
(using almost the whole budget in reasoning-time) and Observability to ~4/10 (2 steps leave no
audit trail). The benchmark measures *execution behaviour* under adversity, not raw reasoning
capability — and the two diverge at the frontier.

**2. Frontier saturates at the top.**
Kimi and Claude produce *numerically identical* per-task scores across all 10 tasks: 98.7 / 98.7 /
98.7 / 95.7 / 96.3 / 94.7 / 98.7 / 97.3 / 97.5 / 98.7. Same task structure, same deterministic
judge, same solve-path → same score. The residual gap below perfect is a rubric ceiling
(Robustness 12/15 on T4/T5, Observability ~8.67/10 by design), not a capability gap between the
two models. ComtradeBench today cannot fine-rank two execution-optimised frontier models.

**3. Sub-frontier is high-variance, not uniformly weak.**
Multi-seed Kimi T9 = 97.5 with std 0.0. Multi-seed Llama T9 spans 18.7 – 97.5 depending on seed
(and hosted non-determinism). The discriminative signal is *reliability*, not capability: Llama
can sometimes match frontier, just not *consistently*. Production agent deployment needs the
consistent half.

**4. ⭐ A mid-size instruction-tuned 7B matches closed frontier — without training.**
Qwen2.5-7B-Instruct, zero-shot (no fine-tuning, no GRPO), scores **97.2** — within 0.3 points of
Kimi/Claude (97.5), above GPT-5 (93.2), and *well* above Llama 3.3 70B (89.3). This is **not** a
blanket "open-source matches closed-source" claim: Llama 3.3 70B is *also* open-source and scores
7.9 points lower. The relevant axis is **instruction-tuning quality at the 7B class**, not
licensing. The honest reframing: *this benchmark is solvable by strongly-instructed mid-size
models without any training*. A direct implication is that the GRPO 7B saturation we document
below is *structural* — the base model genuinely clears the benchmark, so there is no gradient
for GRPO to exploit. Qwen2.5-7B closing the gap without training validates that finding rather
than undermining it.

The honest takeaway: ComtradeBench produces four independent discriminative findings —
execution-vs-reasoning at the frontier (Kimi/Claude vs GPT-5 on T9), saturation at the ceiling
(Kimi = Claude), reliability at the sub-frontier (Llama variance), and 7B-class instruction-tuned
parity (Qwen2.5-7B ≈ frontier without training). Full per-task breakdowns are in
`llm_results_{kimi,claude,gpt5,qwen7b_zeroshot,llama}.json` and multi-seed summaries.

### Ablation — context window dominates prompt engineering

We originally claimed the T4/T5 Robustness gap could be closed with an explicit **EVENTS
scratchpad** prompt pattern. The data told a different story. Three conditions on Kimi, same
model family, same agent loop, same seed:

| Condition | Context | Prompt | T4 Robustness | T5 Robustness |
|---|-------|--------|--------------:|--------------:|
| A | 8k   | default | 0 / 15  | 0 / 15  |
| B | 128k | default | 12 / 15 | 12 / 15 |
| C | 128k | EVENTS scratchpad (enhanced) | 12 / 15 | 12 / 15 |

- **A → B (context effect):** +12 Robustness on both tasks, purely from enlarging the context
  window. No prompt change.
- **B → C (prompt effect):** zero additional gain. Explicit "log before you retry" scaffolding
  on top of 128k produced no measurable improvement.

The original T4/T5 = 0 Robustness result at 8k was not a narration failure. It was a
**context-truncation failure** — the retry narration fell off the back of the buffer before it
could land in `run_log`. At 128k, the same model with the same prompt captures everything it
needs. Adding an explicit EVENTS scratchpad on top of 128k changes nothing.

**Takeaway for agent builders:** on tool-use benchmarks with long trajectories, **size the
context to the episode length before reaching for prompt engineering**. A prompt cannot recover
narration that was never written because the buffer filled up. This is a null result — but a
genuinely useful one, because it contradicts the intuition (which we had!) that prompt
scaffolding should fix the observability gap.

### How ComtradeBench compares to existing tool-use benchmarks

| Benchmark | Adversarial faults in env | Within-episode non-stationarity | Multi-dim execution scoring | Budget constraints |
|---|:---:|:---:|:---:|:---:|
| ToolBench (Qin et al., 2023) | — | — | — | — |
| τ-bench (Sierra / Anthropic) | partial (policy violations) | — | ✓ | — |
| BFCL (Berkeley) | — | — | — | — |
| API-Bank | — | — | — | — |
| **ComtradeBench** | **✓** (429/500/drift/dupes/totals) | **✓** (T9) | **✓** (6 dimensions) | **✓** (T10) |

Closest relative is τ-bench — it also scores beyond "did the final answer match" and injects
policy-level adversarial conditions. ComtradeBench's unique combination is **environment-level
fault injection plus within-episode escalation (T9) plus budget-aware rollouts (T10)**. The
adversarial bits live in the environment, not in the prompts or labels, so an agent cannot route
around them by rephrasing.

### Scoring weight rationale

The six-dimensional rubric weights are 30 / 15 / 15 / 15 / 15 / 10. The design principle:
**correctness is necessary but not sufficient**. Correctness gets the largest single weight (30),
but the combined weight of "execution quality under adversity" dimensions
(Completeness + Robustness + Efficiency + Data Quality = 60) exceeds Correctness. This forces the
score to reward agents that do the job right, not just return something plausible. Observability at
10 is intentionally lower — it is an audit requirement, not a core task, but non-zero because an
un-auditable pipeline is not a production-ready pipeline.

### GRPO training — operating envelope empirically mapped

The real finding is not "we trained an agent", it is **where GRPO on this benchmark works and
where it fails**. We have empirical evidence at both ends of the envelope:

**Lower bound — Qwen2.5-1.5B, 50 iter full-parameter GRPO on Lambda A100 40GB.**
Mean reward oscillates in the 0.22 – 0.94 range with no net upward trend over 50 iterations. The
model lacks the capacity to stably solve T9 / T10 (max_reward drops to 0.24 on batches that
sample those tasks, confirming it is a capacity ceiling, not sampling noise). The training loop
itself is correct — loss decreases smoothly, KL stays bounded — but the signal it optimises is
noise-dominated because 1.5B cannot find a stable policy. Data: `grpo_gradient_training.jsonl`,
`grpo_gradient_training_summary.json`.

**Middle point — Qwen2.5-3B + LoRA (r = 16) on Lambda A100 40GB. Learns, then collapses.**
Iters 1-2 are curriculum warmup on T1-T8, with LoRA init producing zero-variance rollouts (by
design). **Iters 3-14 enter the GRPO learning window**: reward_std oscillates 0.46 – 0.55, KL
grows monotonically from 8e-6 to 5.6e-4, and the adapter is clearly receiving gradient signal.
Mean reward bounces 0.0 – 0.73 due to heterogeneous task difficulty sampled each iter. Then
**iter 15 hits policy collapse**: three consecutive iterations (15, 16, 17) produce ZERO valid
rollouts — all 4 rollouts per iter fail to produce parseable tool calls. Iter 18 recovers with
mean_reward = 0.027 and max_reward = 0.107, confirming the LoRA adapter drifted into a
degenerate output region. This is a textbook RL policy-collapse / reward-hacking failure mode.
The run proves ComtradeBench + GRPO on 3B *can* learn, but the window is *fragile* —
stability requires careful KL-penalty tuning and trust-region clipping that we did not apply.
Data: `grpo_3b_lora_collapse.json`, `grpo_gradient_training_3b.jsonl`.

**Upper bound — Qwen2.5-7B + LoRA (r = 16) on Lambda A100 40GB.**
Mean reward at iteration 1 is already **0.987** (above the 0.968 rule-based baseline). Across the
5 completed iters, loss stays at 0 and KL stays at 0 because reward_std across each group of
rollouts is near-zero — GRPO's group-relative advantage normalization produces zero advantage
when rollouts are indistinguishable, so no gradient signal propagates to the LoRA adapter.
This is not a bug. It is saturation: the base model already exceeds the task threshold, so there
is nothing for GRPO to optimise against. Data: `grpo_7b_lora_5iter_saturation.json`.

**Implication.** GRPO's useful training band on ComtradeBench exists — the 3B learning phase
(iters 3-14) is empirical proof — but the band is **narrow and fragile**. All three configurations
failed in different ways: 1.5B under-capacity / noise-dominated, 3B+LoRA learns then collapses,
7B+LoRA saturates. This is a more actionable finding than "training converged on some model":
it names a concrete failure mode (policy collapse at iter 15) and specifies the engineering work
required to avoid it — adaptive KL penalty, stricter trust-region clipping, early-stop on
reward-variance collapse, or a combination. The training pipeline itself is validated by a local
CPU smoke test (`grpo_smoke/`, `grpo_smoke_lora/`) — iter 1 produces loss = 0 (expected;
π_old = π_new at step 0) and iter 2 produces kl > 0 (confirming the policy actually updated
between rollouts), so the pipeline plumbing is sound. The envelope itself is the finding.

<p align="center">
  <img src="/blog/assets/comtradebench/training_curve.png" width="80%" alt="GRPO Training Curve — Qwen2.5-1.5B, 50 iter, full-parameter"/>
</p>

### A latent GRPO bug, found and fixed en route

While setting up the training pipeline, we discovered that `train_grpo.py` in HF-training mode
was loading the *rollout actor* and the *trainable model* as two separate objects: `pipeline()`
instantiated its own copy of the base model, while `AutoModelForCausalLM.from_pretrained(...)`
loaded the trainable copy. `optimizer.step()` updated the trainable copy; rollouts used the
frozen one. Gradient updates never propagated to the rollout actor — the training loop ran, loss
looked fine, and *absolutely nothing was learned*. The existing 8-iter API-mode metrics we had
published earlier did not surface this bug because API mode sets `use_gradient_update = False`
and skips the whole path.

The fix is two lines (`llm._pipe.model = model` plus building the pipeline directly from the
trainable model to avoid a 14 GB duplicate load on 7B), plus a small adjustment so `ref_model`
can be omitted when training with LoRA (we use `peft_model.disable_adapter()` for the reference
policy instead of a deep-copied frozen model, saving another 14 GB of GPU memory). The commit
tells the full story in-repo; the smoke test confirms end-to-end that the adapter now receives
gradient updates between iters.

This is the kind of bug that *only* surfaces when you actually run the full pipeline and look at
KL and reward together. It is the primary reason we recommend anyone running GRPO in a new
codebase always verify KL > 0 after the first gradient step — silent policy/actor desync is
the default failure mode.

**Recommendation for GRPO practitioners, general enough to use anywhere.**
Silent policy/rollout-actor desync is the single worst failure mode we hit because *the training
loop runs, the loss number looks reasonable, and the gradient step returns no error — yet no
learning actually happens*. The only way to catch it is to instrument KL divergence between the
updated policy and a fresh reference pass, and verify `KL > 0` after the first gradient step.
Any GRPO practitioner setting up a new codebase should treat this as standard hygiene — adding
a single assertion at the top of the training loop (e.g. `assert kl > 1e-6, "policy-actor
desync"` after iter 1's gradient step) would have saved us the ~8 hours we spent diagnosing.
This is independent of ComtradeBench — it applies to any GRPO implementation, whether you are
using `train_grpo.py` from this repo, TRL's GRPOTrainer, or a custom loop. Add the check; you
will eventually be glad you did.

---

## What this benchmark reveals

ComtradeBench is designed to expose a gap that clean evaluations often miss: agents can appear capable in idealized settings while remaining brittle under operational noise.

The hardest problems are not "knowing what the API is." They are:

- continuing correctly **after an interruption**
- maintaining data integrity **across many pages**
- adapting when **conditions shift mid-episode**
- balancing **coverage against cost**

This is where reliable agents differ from merely fluent ones.

---

## Benchmark and training substrate

ComtradeBench is not just an evaluation harness — it is built to support agent improvement.

The environment ships with a full **GRPO training pipeline**: reproducible rollouts, group-relative advantage normalization, and reward-only optimization. No human labels needed. No separate reward model.

This is an intentional design choice: if robust tool-use is a real bottleneck for agentic AI, we need environments that can **both measure and train** that capability — with identical conditions in evaluation and training.

---

## Quick start

```bash
# No LLM, no GPU, no API key required
git clone https://github.com/yonghongzhang-io/comtrade-openenv
pip install openenv-core[core]
python agent/smoke_test.py --task T1_single_page
python agent/smoke_test.py --task T9_adaptive_adversary

# GRPO training via local Ollama (CPU-capable)
python agent/train_grpo.py \
    --api-url http://localhost:11434/v1 \
    --api-model qwen2.5:7b \
    --num-iterations 200 --group-size 4
```

All benchmark data is generated procedurally from a seeded PRNG — no external fixtures, no live API dependency. Every result is fully reproducible from a task ID and a random seed.

---

## Limitations and next steps

This release is honest about what it does not yet do:

- **Frontier saturation at the ceiling.** Kimi-128k and Claude Sonnet 4.6 produce numerically identical
  scores across all 10 tasks; open-source Qwen2.5-7B zero-shot lands within 0.3 points of them (97.2).
  ComtradeBench today measures execution reliability well but cannot fine-rank three execution-optimised
  models against each other at the ceiling. A harder T9 variant with steeper mid-episode escalation,
  and T11+ tasks targeting frontier-specific behaviours, would reopen the discrimination.
- **T4/T5 Robustness ceiling at 12/15 is a rubric string-matching artifact.** Reading `server/judge.py`
  L293-336, the +3 bonus on rate-limit tasks requires the literal keyword `"exponential"` or
  `"backoff"` in `run.log`; on server-error tasks it requires `"max"` or `"limit"`. The retry logic
  itself is correct; the ceiling is a rubric artifact, not a model capability gap. Future work is to
  broaden the keyword set or move to a semantic check.
- **Five LLMs evaluated.** Kimi Moonshot V1-128k, Claude Sonnet 4.6, GPT-5, Llama 3.3 70B, and
  Qwen2.5-7B-Instruct (open-source, zero-shot). Adding Gemini, Qwen2.5-72B, and DeepSeek would
  broaden the cross-model story further, though the current data already exposes four independent
  discriminative findings (execution-vs-reasoning at the frontier, saturation, sub-frontier
  reliability, open-source parity).
- **GRPO training stability engineering is future work.** The 3B + LoRA collapse at iter 15 is a
  diagnosable instability: adaptive KL penalty, stricter trust-region clipping, or early-stop
  on reward-variance collapse would likely stabilise the learning window past iter 14. We did
  not perform this hyperparameter engineering in the submission window.
- **Single-seed evaluation for most LLMs.** Kimi and Llama have multi-seed data on T9
  (`multiseed_*_summary.json`). Claude, GPT-5, Qwen2.5-7B, and all other tasks use seed=42 only.
- **Benchmark comparison is qualitative.** We describe the feature matrix vs. τ-bench / BFCL /
  ToolBench but have not yet run the same LLM across all four benchmarks side-by-side.

None of these block using ComtradeBench as a tool-use benchmark today; they are the research
directions we think make the environment more useful to the field.

## Conclusion

<p align="center">

---

### 💬 *Can an agent still finish the job when the API fights back?*

---

</p>

That question matters far beyond trade data. It applies to any agent expected to operate against real interfaces with pagination, retries, noisy outputs, and resource limits.

If we want more reliable agents, we need environments that reward reliability directly.  
That is the role ComtradeBench is designed to play.

---

<p align="center">
  <a href="https://github.com/yonghongzhang-io/comtrade-openenv">GitHub</a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/spaces/yonghongzhang/comtrade-env">HF Space</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/meta-pytorch/OpenEnv">OpenEnv Framework</a>
</p>
