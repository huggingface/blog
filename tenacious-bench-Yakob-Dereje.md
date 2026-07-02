---
title: "I Broke My Own Sales Agent, Then Built the Benchmark to Prove It"
thumbnail: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/thumbnails/default.png
authors:
- user: yakobd
---

# I Broke My Own Sales Agent, Then Built the Benchmark to Prove It

*Yakob Dereje | TenX Academy TRP1 | May 2026*

---

There is a moment in every engineering project where you realise the thing you built works — and you have no idea whether it works *well*.

That moment came for me in Week 10, when I ran my B2B sales agent through τ²-Bench retail and got a score of 0.333. The number meant almost nothing. My agent had just composed 23 outreach emails, qualified 13 prospects, and booked 9 discovery calls. It had done every step of a real sales workflow. And τ²-Bench retail — designed to test whether an AI can navigate a retail purchase — was measuring the wrong thing entirely.

So I built Tenacious-Bench v0.1. Here is what I learned.

---

## The Problem With Benchmarks Built for Someone Else

τ²-Bench retail is a well-engineered benchmark. I want to be clear about that before I criticise it. It tests task completion in a structured environment, covers tool use and multi-turn reasoning, and has rigorous contamination controls. For what it measures, it measures well.

But my agent wasn't doing retail tasks. It was doing B2B sales outreach for Tenacious Intelligence Corporation — a specific company with a specific style guide, 40 banned phrases, five tone markers, and a hard requirement that every claim in every email trace back to a real public signal. τ²-Bench retail has no rubric for any of this.

The failure I kept seeing in my Week 10 traces was not "the agent failed to complete the task." It was "the agent completed the task in a way that would get a Tenacious Research Partner fired."

Probe P16: email used "top talent" — banned phrase, first offence.
Probe P05: email claimed "your team has been actively expanding engineering capacity" for a company with `confidence: very_low` — signal over-claiming.
Probe P09: outreach hook referenced "recent activity" without naming which signal — weak grounding.

These failures were invisible to τ²-Bench retail. They were also systematic — appearing in 7 of 13 traced emails for Segment 1 (recently-funded startups). That is not noise. That is a distributional failure that a domain-specific benchmark should catch.

The gap was real and provable. The audit memo took 589 words, 15 probe IDs, and 6 trace examples to document it. Then I had to build the benchmark to close it.

---

## The Hardest Design Constraint: Machine-Verifiability

The binding constraint in Tenacious-Bench v0.1 was not the number of tasks or the coverage of failure modes. It was this: **every task had to be scoreable by a script**.

No human in the loop. No "sounds professional" rubric. No vibes.

This sounds obvious until you try to do it. My first draft rubric had criteria like "the email should feel grounded in the prospect's situation." That is not a rubric. That is a hope. I revised it until every dimension had a mechanical check:

- Zero of these 40 banned phrases in subject + body
- At least one token from the `required_signals` list appears in the body
- Subject line ≤ 60 characters
- Body ends with a Cal.com booking URL
- LLM-judge scores ≥ 4/5 on each of the five Tenacious tone markers

Five checks. Each returns 0 or 1. The overall score is pass@1 — all five must pass. A script reads a task and a candidate output and returns a number in under two seconds. That is a benchmark.

The LLM-as-a-judge component deserves a note here. I read Gu et al.'s survey on LLM-as-a-judge carefully and came away with one strong disagreement with common practice: most implementations use the same model to generate *and* judge the same output. Gu et al. warn against this — Li et al. (2025) call it "preference leakage" and show it systematically inflates scores. My pipeline rotates model families: Claude Sonnet generates hard seeds, Qwen3 generates bulk variations, and a separate DeepSeek instance judges quality. The rotation policy is committed to `methodology.md`.

---

## Building 293 Tasks From a Seed of 24

Tenacious didn't hand me a labeled dataset. They handed me 12 "good" and 12 "bad" hand-labeled outreach emails, a style guide, and a pricing sheet. Everything else I had to build.

I used four authoring modes simultaneously:

**Trace-derived (27%).** Real Week 10 agent outputs, redacted and restructured as evaluation tasks. These are the most valuable tasks in the benchmark — they test the actual distributional behavior of my system, not a synthetic approximation of it. They were also free: already in `trace_log.jsonl`.

**Programmatic with parameter sweeps (32%).** A single "bench over-commitment" probe became 20 tasks by varying company size, segment, signal confidence, and stack. This is cheap to generate and covers systematic variation efficiently. The weakness is surface-level diversity — the tasks look different but share deep structure.

**Multi-LLM synthesis (31%).** This is where the interesting engineering happened. I routed hard case generation across LLM families: Claude Sonnet authored the 30 hardest seeds anchored directly to my failure taxonomy. Qwen3 generated bulk variations. Everything passed through a three-dimension judge filter (input coherence, ground-truth verifiability, rubric-application clarity) before inclusion. Acceptance threshold: all three dimensions ≥ 3/5. 

**Hand-authored adversarial (10%).** The 30 tasks I wrote myself to defeat my own system. These carry the most originality weight. Writing them required me to think like an adversary — what inputs would cause my agent to produce output that looks correct but isn't? TB-HA015 (maximum violation test: emoji subject line, every banned phrase, zero grounding) came from this mode.

**The contamination protocol** was non-negotiable. Three checks before any task entered the held-out partition: n-gram overlap below 8-gram, embedding cosine similarity below 0.85 (using a cheap sentence-transformer), and time-shift verification for any task referencing public data. The contamination-check script is a committed deliverable and runs in the dataset publication pipeline. 0 violations across all 3 checks.

**Inter-rater agreement** reached 93% after one rubric revision. The protocol: hand-label 30 tasks, wait 24 hours, re-label without looking at the first pass. If agreement falls below 80% on any dimension, the rubric is ambiguous — revise before proceeding. I revised once, on the grounding dimension, where my first formulation was underspecified about what "traces to the brief" actually required.

---

## Why I Chose to Augment Rather Than Generate

The challenge target for Path A is 1,000–3,000 training pairs. My benchmark had 293 tasks — so after partitioning, I had 178 in the training split. After quality filtering, 128 passed the scorer. That's a real gap.

I had two options:

**Option A:** Generate more benchmark tasks via OpenRouter, pushing total tasks beyond 293.

**Option B:** Augment the existing 128 pairs — paraphrase and vary each one 20 times — to reach the target without touching the benchmark size.

I chose Option B. Here is why this was the right call, not just the safe one: generating new benchmark tasks would have changed the *evaluation domain*, not just the training data. I would have been training on a distribution that differed from what I was evaluating against. Augmentation keeps the training distribution anchored to the same 128 high-quality, scorer-verified pairs. The 2,413 augmented variations are surface-level rewrites — same signals, same constraints, different phrasing. LIMA (Zhou et al., NeurIPS 2023) suggests this is exactly the right strategy for single-task domains: quality and constraint coverage matter more than surface diversity.

The augmentation pipeline accepted 94.3% of generated variations after quality filtering. Total training pairs: 2,541. Training ran on a free Google Colab T4 in 19.1 minutes at $0.00.

---

## The Results — Including the Part That Didn't Work

**Delta A = +0.263, 95% CI [0.140, 0.386], p < 0.0001.**

The trained adapter scored 0.754 on the sealed 57-task held-out partition. The Week 10 baseline scored 0.491. The confidence interval is entirely above zero. The result is robust across 10,000 bootstrap resamples with seed=42.

**Delta B = +0.140.** This is the result I was least certain about before running the ablation. The prompt-engineered baseline — the identical Qwen2.5-0.5B backbone with the complete Tenacious style guide embedded in the system prompt — scored 0.614. The trained adapter scored 0.754. Training beat prompt engineering.

This matters because Delta B positive means the learned weights encode style constraints more reliably than a long system prompt. At scale, this translates to shorter prompts, lower token cost per call, and more consistent output under off-distribution inputs where prompt instructions degrade faster than fine-tuned weights.

**The honest failure.** On TB-SG23 — a hand-authored adversarial task containing an invented pricing claim ($1.2M for a 12-month engagement that Tenacious never quoted) — the trained adapter scored 1.0 when the expected score was 0.0.

The model satisfied every measurable style constraint: no banned phrases, subject under 60 characters, grounding check passed on non-numeric signals, CTA present. But it generated a fabricated price and passed anyway.

The root cause is a training data gap. My 2,541 training pairs were all *correct* emails — positive examples only. The model never saw a (fabricated-claim, rejection) pair. It learned what correct looks like. It did not learn that numeric invention is a violation independent of other constraints. This is a known SFT limitation. The resolution is a DPO or rejection-sampling component in v0.2, with fabricated pricing as the explicit rejected output.

I am reporting this because it is real and because it matters for production deployment. The kill-switch condition for this exact failure is specified in the memo.

---

## What Tenacious-Bench v0.1 Still Can't Grade

Building a benchmark teaches you more about what you didn't build than what you did. Four things are missing from v0.1 that v0.2 needs to address:

Multi-turn objection handling. Every task in v0.1 is a single cold-outreach turn. The benchmark has no opinion on what happens when a prospect objects.

Reply classification accuracy. The Conversion Engine classifies replies into five categories and routes accordingly. None of that is tested.

Zero-signal fabrication. Every task supplies at least one hiring signal. No task tests the hardest condition: agent behavior when the brief is empty.

Segment 4 (AI capability gap) generalization. Twenty training pairs. Zero dedicated held-out tasks. Unvalidated.

These are not afterthoughts. They are the next four probes in `probe_library.md`, waiting for v0.2.

---

## The Artifacts

Everything described in this post is publicly available, reproducible in under one hour, and licensed CC-BY-4.0.

**Dataset:** https://huggingface.co/datasets/yakobd/tenacious-bench

**Model (LoRA adapter):** https://huggingface.co/yakobd/tenacious-bench-adapter

**Code:** https://github.com/yakobd/The_Sales_Agent_Evaluation_Bench

```bash
git clone https://github.com/yakobd/The_Sales_Agent_Evaluation_Bench
cd The_Sales_Agent_Evaluation_Bench
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 ablations/score_held_out.py
```

If you are building a sales agent and tired of benchmarks that don't match your domain — clone the repo, swap in your own style guide and failure taxonomy, and build your own v0.1. The dataset authoring pipeline is in `generation_scripts/`. The contamination checker runs automatically. The scoring evaluator is 200 lines of Python.

The benchmark is the hard part. The training run is 19 minutes.

---

*Yakob Dereje is a TenX Academy TRP1 trainee. All experiments were run in May 2026 as part of the TRP1 Week 11 project.*
