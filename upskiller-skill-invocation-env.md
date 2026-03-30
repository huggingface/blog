---
title: "Teaching LLMs When to Use Their Skills: An RL Environment for Skill Invocation"
thumbnail: /blog/assets/upskiller/thumbnail.png
authors:
- user: mpnikhil
  guest: true
  org: ""
date: March 24, 2026
---

## TL;DR

We built an RL environment where an agent faces a task and a skill catalog, decides which skills to load into a limited context budget, optionally unloads irrelevant ones, and submits an answer. A five-component rule-based reward trains the agent to be selective and correct, mapping cleanly onto GRPO without a learned reward model.

## See It in Action

```python
from skill_invocation_env import SkillInvocationEnv, SkillInvocationAction

with SkillInvocationEnv(base_url="https://mpnikhil-skill-invocation-env.hf.space") as env:
    result = env.reset()
    skill_id = result.observation.skill_catalog[0]["id"]
    result = env.step(SkillInvocationAction(action_type="load", skill_id=skill_id))
    result = env.step(SkillInvocationAction(action_type="submit", answer="solution"))
    print(f"Reward: {result.reward}")
```

The environment is deployed as a public HF Space using the [OpenEnv protocol](https://github.com/huggingface/openenv). Any training loop that speaks HTTP can connect and sample episodes. Three actions — **load**, **unload**, **submit** — and a five-component reward that scores correctness, skill selection precision, recall, context bloat, and cumulative token waste.

## Introduction

Skills give LLM agents something tools cannot: procedural knowledge at a higher level of abstraction. Where tools execute discrete actions (call an API, run a query), Skills teach an agent how to reason through entire workflows — proprietary protocols, binary formats, deployment sequences — through in-context learning from curated procedural text.

The challenge is reliable invocation. Without it, developers stuff knowledge directly into the system prompt — AGENTS.md, CLAUDE.md — loading everything up front so the agent never has to decide what to retrieve. Vercel's engineering blog (["AGENTS.md outperforms skills in our agent evals"](https://vercel.com/blog/agents-md-outperforms-skills-in-our-agent-evals)) documented exactly this: agents failed to invoke available skills 56% of the time, producing zero improvement over baseline. Even with explicit instructions, skills maxed out at 79%. The workaround — compress everything into an always-loaded system prompt — hits 100% but pushes context cost onto every turn. Progressive disclosure via Skills is the better architecture. Current models just aren't trained to use it.

That's the gap our environment closes. We also explored DSPy's GEPA optimizer (suggested by Sanyam Bhutani at Meta) as a gradient-free alternative to GRPO for the same training signal.

## Why Skills Matter: The SkillsBench Evidence

The SkillsBench benchmark (Li et al., 2026, [arXiv:2602.12670](https://arxiv.org/abs/2602.12670)) measured skill efficacy across 84 tasks, 11 domains, and 7,308 trajectories. Curated Skills raise average pass rate by **+16.2pp**, but effects vary widely (+4.5pp in Software Engineering to +51.9pp in Healthcare), 16 tasks saw degradation, and self-generated Skills provided no benefit (–1.3pp). Skills work when they fill concrete procedural gaps — and models cannot author the procedural knowledge they benefit from consuming.

This points to three conditions where Skills are irreplaceable — conditions we designed every task around:

- **Cannot be derived from training data.** Proprietary APIs, custom binary formats — a model has no priors here. This is why our synthetic domains (Zephyr-3, NovaBin, HelixLang, ArcDeploy) are entirely fictional.
- **Has precise, non-obvious specifications.** Exact byte layouts, HMAC signing formats — being approximately right is just as wrong as being completely wrong. Our verifiers use code execution, not keyword matching.
- **Would be impossible to guess.** Flag bit positions, deployment phase configs — you either have the reference material or you don't. This is where SkillsBench found +51.9pp in Healthcare.

The challenge isn't just having curated Skills — it's knowing which ones matter and when. That's the behavior our environment trains.

## The Skill Invocation Environment

Every episode presents the agent with a task and a skill catalog (short descriptions only — no full content). The agent decides which skills to load, reads their content, optionally unloads ones that don't apply, and submits a solution. Loading costs context budget. The reward is computed entirely from rule-based checks.

The catalog gives the agent a lightweight index of available Skills. Full content arrives on demand — loaded when the agent determines a skill is relevant, releasable via unload when it determines it is not. Instead of every skill occupying context on every turn, the agent manages a limited budget and allocates it precisely.

### Actions

- **load(skill_id)** — Load a skill's full content into context. Costs 1 unit of context budget.
- **unload(skill_id)** — Remove a skill from context, freeing budget.
- **submit(answer)** — Submit the final solution. Reward is computed on skills loaded at this moment, plus a penalty for unnecessary loads across the episode.

### Reward Function

The reward weights reflect a deliberate priority ordering. Correctness dominates at 0.6 because getting the answer right is the primary objective — without it, efficient skill selection is meaningless. Precision is weighted second at 0.3 because context bloat is the central problem: an agent that loads everything and happens to answer correctly has not learned the behavior we're training for. Recall is lowest at 0.1 because most tasks require only one or two skills, so recall naturally stays high once the agent learns to load at all. The two penalty terms operate on different timescales: bloat (-0.15) penalizes state at submit time; token waste (-0.05) penalizes cumulative load history, discouraging speculative loading even if the agent unloads before submitting.

```
correctness  = 0.6 if answer is correct
precision    = 0.3 × (relevant loaded / total loaded)
recall       = 0.1 × (relevant loaded / total relevant)
bloat        = -0.15 per unnecessary skill at submit
token_waste  = -0.05 per skill ever loaded but irrelevant
total        = max(sum of above, -1.0)
```

The optimal policy is clear: load exactly the right skills, synthesize the correct answer, and submit. The unload action allows exploratory loading without permanent context pollution, as long as the agent cleans up before submitting.

### Tasks: 13 Across 9 Domains

Three tasks adapted from SkillsBench (Apache 2.0) — Flood Detection, Economics Detrending, Dialogue Parsing. The remaining 10 are synthetic. A **TaskGenerator** creates unlimited unique tasks at runtime via two templates (auth_protocol, binary_format) to prevent memorization.

## Training Approaches

### GRPO with TRL's environment_factory

We use [TRL's GRPOTrainer](https://huggingface.co/docs/trl) with **environment_factory**, which natively supports multi-turn tool calling. The environment's methods are exposed as tool calls the model invokes during rollouts. We trained Qwen3-8B with 4-bit QLoRA (NF4, fp16 compute, LoRA r=128, alpha=256, all projection layers, dropout=0.0).

### DSPy GEPA — Gradient-Free Alternative

Sanyam Bhutani (Meta) suggested [DSPy's GEPA optimizer](https://dspy.ai/api/optimizers/GEPA/overview/) (Genetic-Pareto, [arXiv:2507.19457](https://arxiv.org/abs/2507.19457), ICLR 2026 Oral). GEPA reads full execution traces and diagnoses why a candidate failed — our reward function already produces exactly this kind of interpretable feedback. GEPA can use it as Actionable Side Information to evolve the system prompt without weight updates. It needs 100–500 evaluations vs. 10,000+ for RL, and requires no GPU. GEPA optimizes inference-time behavior rapidly; RL then encodes it into weights for scale.

## Design Insights and Open Hypotheses

We didn't complete a full training run at meaningful scale within the hackathon timeframe — these are design hypotheses rather than empirical results.

**The unload action creates a read-then-decide loop.** The agent can read a skill to assess relevance, then release it before submitting. Whether this trains robustly is something we'd want to verify with longer runs.

**Context budget beats binary invoke/don't-invoke.** A budget allocation model better reflects real constraints and should produce a richer reward signal.

**GEPA and RL are complementary.** The natural workflow is to use GEPA first to discover effective prompt structures, then RL to encode that behavior into weights. We'd like to measure whether the two stages compound.

## upskill — Closing the Skill Generation Loop

Ben Burtenshaw's [upskill](https://github.com/huggingface/upskill) (Hugging Face) handles skill authoring: a powerful model writes and refines SKILL.md files, then validates them on smaller models. Robin Hood — expensive frontier models write the skills, cheaper open models consume them.

**upskill generate → validated SKILL.md → Skill Invocation Environment (GRPO) → GEPA (prompt optimization) → smaller model that reliably invokes skills on demand**

upskill handles the knowledge authoring problem. Our environment handles the behavioral problem. GEPA connects the two at inference time.

## Open Directions

- **More task domains:** SkillsBench's 84 tasks reformatted into OpenEnv schema
- **Richer procedural generation:** Database schemas, network protocols, configuration formats
- **Multi-skill tasks:** 3+ coordinated skills per task
- **GEPA integration:** Wiring our reward as Actionable Side Information
- **Held-out evaluation:** Does the model generalize to unseen SkillsBench tasks?

## Acknowledgements

Thanks to **Sanyam Bhutani** (Meta) for the GEPA direction and for organizing the hackathon. Thanks to **Ben Burtenshaw** and the Hugging Face team for the encouragement and the OpenEnv protocol. Thanks to the SkillsBench team (Li et al.) for open-sourcing their benchmark.

## References

- Li et al. (2026). SkillsBench. [arXiv:2602.12670](https://arxiv.org/abs/2602.12670)
- Agrawal et al. (2025). GEPA. ICLR 2026 Oral. [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- [dspy.GEPA Optimizer](https://dspy.ai/api/optimizers/GEPA/overview/)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl)
- [OpenEnv Protocol](https://github.com/huggingface/openenv)
- [Upskiller](https://github.com/mpnikhil/Upskiller)
- Burtenshaw et al. (2026). [We Got Claude to Build CUDA Kernels and Teach Open Models](https://huggingface.co/blog/upskill)
- [huggingface/upskill](https://github.com/huggingface/upskill)
