---
title: "Harness, Scaffold, and the AI Agent Terms Worth Getting Right"
thumbnail: /blog/assets/agent-glossary/thumbnail.png
authors:
- user: sergiopaniego
- user: ariG23498
---

# Harness, Scaffold, and the AI Agent Terms Worth Getting Right

When a field evolves quickly, its vocabulary often evolves faster than its shared understanding. Terms start to blur, get reused in different contexts, or become shorthand for ideas that are never fully explained. We are currently seeing this happen in the field of AI Agents, where concepts are getting mixed together, some are renamed, and others are widely used for a few months before quietly disappearing.

This can be overwhelming for newcomers, and even for practitioners trying to keep up with the latest developments. After ICLR 2026, one of us ([@ariG23498](https://x.com/ariG23498/status/2049668725511737663)) posted a question that captured this confusion well:

> *"What do you mean by the terms 'harness' and 'scaffold' in the context of agents? I have heard a lot of explanations while I was at ICLR, but I could not understand why they did not converge to a single explanation."*

This glossary is our attempt to ground the terms that keep coming up without clear, consistent explanations. It is not meant to be a comprehensive dictionary of every term in the field. Instead, we focus on the concepts that are often mixed up, reused in different ways, or assumed to be obvious when they are not.

Most of these terms come up whether you're building an agent, deploying one, or just using tools like Claude Code, Codex, or Hermes Agent. The last section covers concepts specific to training models, which is more relevant if you work on that side of things.

> [!NOTE]
> Many of these terms don't have universally accepted definitions yet, and different frameworks use the same word differently. The goal here is not to enforce one correct vocabulary, but to provide a practical mental model that makes discussions easier to follow.

Let's get started.

## Table of Contents

- [Model](#model)
- [Scaffolding](#scaffolding)
- [Harness](#harness)
- [Agent](#agent)
- [Context Engineering](#context-engineering)
- [Policy](#policy)
- [Tool Use](#tool-use)
- [Skills](#skills)
- [Sub-agents](#sub-agents)
- [Training](#training)
  - [RL Environment](#rl-environment)
  - [Trainer](#trainer)
  - [Rollout](#rollout)
  - [Reward](#reward)
- [Learn More](#learn-more)

## Model

The model is the LLM: it takes text in and produces text out (e.g., Claude, Qwen, GPT, Kimi, DeepSeek…). On its own, it has no memory between calls, and no loop. The model can express the intent to call a tool, but it needs a harness to actually execute it. It answers one prompt and stops. Wrap it in scaffolding and a harness and it becomes an agent.

## Scaffolding

The behavior-defining layer around the model: system prompt, tool descriptions, how the model's responses get parsed, what it remembers across steps (context management). It shapes how the model sees the world and acts in it, whether during training or at inference.

Products like Claude Code, Codex, and Antigravity CLI call the whole thing a harness. Claude Code's [own docs](https://code.claude.com/docs/en/how-claude-code-works) say it directly: "Claude Code serves as the agentic harness around Claude." That's the broad use: harness means everything that isn't the model. The scaffold/harness distinction matters most when you need to reason about them separately, as in a training pipeline. You'll also hear "scaffold" used more broadly to cover any infrastructure the harness relies on: hooks, runtime configuration, even directory structure.

Some products like Claude Code and Codex are tightly coupled to their provider's models. Others like Antigravity CLI and Hermes Agent let you plug in any model.

## Harness

The execution layer inside the agent: it calls the model, handles its tool calls, decides when to stop. The harness is what makes the agent run. Scaffolding, defined above, is what the model works from: its instructions, its tools, its format.

**Harness engineering** is the discipline of designing this layer well: deciding when the agent should stop, how errors get handled, and what guardrails keep it on track. It applies at both training and inference. [Addy Osmani's piece](https://www.oreilly.com/radar/agent-harness-engineering/) and [OpenAI's account of building with Codex](https://openai.com/index/harness-engineering/) both cover this from the inference side.

At evaluation time, the same pattern shows up as an **eval harness**: instead of collecting training data, it runs a fixed set of scenarios at a model checkpoint and records metrics rather than updating weights.

## Agent

The term comes from reinforcement learning, where an agent is simply a function that takes an observation and returns an action. The environment takes that action and returns a new observation, and the loop repeats. That loop is still at the core of how LLM agents work.

In the LLM world, the term has expanded. An agent is a model plus everything around it that lets it act, not just respond. It turns raw text generation into something that can act in a loop: taking in information, deciding what to do, and acting on the results.

Take a coding agent as a concrete example. The system prompt, tool descriptions, and the output format the model follows form the scaffolding. The loop that calls the model, handles its tool calls, and decides when to stop is the harness. At training time, the harness also runs many of these loops in parallel and feeds the results back to update the model.

![Agent diagram showing Harness, Scaffold, and Model as components inside Agent, with Sub-agent below](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agent-glossary/agent-diagram.png)

In the community, it's usually put as **Agent = Model + Harness** ([@Vtrivedy10](https://x.com/Vtrivedy10/status/2031408954517971368) and [Will Brown's tweet](https://x.com/willccbb/status/2049844685095715289) for reference). If you're not the model, you're the harness. The subtle distinction between harness and scaffold that creates most of the confusion is what the two sections above address.

When people talk about products like Claude Code, Codex, or Cursor, they're referring to a specific harness built on top of a specific model, designed and optimized together. Two products using the same underlying model can feel completely different because their harnesses make different choices. And swapping a better model into the same harness also changes the experience. The model, the harness, and the product are three different things.

## Context Engineering

Designing what goes into the agent's context window: what the model sees at each step, system prompt, tool descriptions, conversation history, retrieved knowledge. It's not a one-time decision: as the model runs, previous turns shape what goes into future calls, and the harness actively manages this throughout the run. It applies at both training and inference, but the cost of getting it wrong is very different. At training, what the model sees shapes what gets learned. Get it wrong and you're retraining. At inference, it's just text: change a prompt and redeploy. The [HF Context Engineering Course](https://huggingface.co/learn/context-course/en/unit0/introduction) covers this in depth.

Memory is part of this picture. **Short-term memory** is what stays in the context window during a single run: conversation history, tool results, previous reasoning. **Long-term memory** persists across sessions, stored externally and retrieved on demand, then injected back into context when relevant.

## Policy

A policy is the behavior a model has learned: given any situation, it defines the probability of taking each possible action. The model weights encode this behavior, but the policy is what those weights produce, not the weights themselves. A policy is not an agent. Wrap a checkpoint in scaffolding and a harness and deploy it, and you get an agent.

## Tool Use

How agents reach outside themselves: APIs, code interpreters, databases, web search, file systems. The model expresses the intent to use a tool in a structured format. Modern inference APIs surface this as a first-class object: the harness receives the call directly and routes it to the right function. The result gets fed back into context and the loop continues.

## Skills

Reusable, structured packages of knowledge that enable multi-step tasks. Where a **tool** is an action ("run this command"), a **skill** bundles everything needed to accomplish a goal ("investigate this bug, form a hypothesis, write a fix"). They are portable across agents and loaded on demand. The line between tool, skill, and sub-agent shifts across frameworks. The [HF Context Engineering Course](https://huggingface.co/learn/context-course/en/unit1/introduction) covers skills in depth.

## Sub-agents

An agent called by another agent to handle a specific subtask. It has its own model and scaffold, reasons independently, and returns a result. The calling agent doesn't need to know how it works internally. This is what separates a **sub-agent** from a **tool** (a function call) or a **skill** (packaged knowledge): a sub-agent can itself reason, use tools, and call further sub-agents.

## Training

The terms above apply whether you're training or deploying. These four are specific to training, where the agent runs through tasks, gets scored, and its model's weights get updated. Every RL training system for LLMs is built around the same pipeline:

![RL training pipeline showing RL Environment, Trainer, and Reward connected by rollout and updated policy](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agent-glossary/rl-pipeline.png)

### RL Environment

The environment is anything you can interact with: a stateful object that takes an action as input, updates its internal state, and returns an observation. In the LLM context, actions are typically tool calls. A filesystem is a simple example: the action `touch foo.txt` updates the state by creating the file, and the observation might be the updated file listing. Definitions vary across frameworks.

We recently published a dedicated guide on this, so rather than compress it here, see [The Ultimate Guide to RL Environments](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide) for a complete breakdown of types, frameworks, and examples.

### Trainer

The trainer is what makes the agent better: it runs many agent episodes, scores the results and uses them to update the inner model's weights. [TRL's GRPOTrainer](https://huggingface.co/docs/trl/main/en/openenv) is a concrete example: a single class that handles episode generation, reward scoring, and weight updates.

### Rollout

A rollout is one full agent run from start to finish: what the agent saw, what it did, and what reward it got at each step. It's also called a *trajectory* or a *trace*, depending on the context. This is the raw data RL algorithms learn from.

### Reward

The score that tells the training algorithm whether the model is getting better. It can be *verifiable* (tests pass/fail, answer matches), *learned* (human preferences, LLM-as-judge), *sparse* (one score at the end of an episode), or *dense* (a score at each step). This is what the trainer uses to actually update the inner model's weights. For a thorough breakdown of each type, see the [Reward Architecture](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide#dimension-4-reward-architecture) section in Adithya's guide.

**Rubrics** break the reward into explicit dimensions with weights, rather than a single number. [OpenEnv](https://github.com/meta-pytorch/OpenEnv) and [Verifiers](https://github.com/willccbb/verifiers) implement rubrics as objects you can combine (`WeightedSum`, `Sequential`, `Gate`).

## Learn More

- [@Vtrivedy10: The Anatomy of an Agent Harness](https://x.com/Vtrivedy10/status/2031408954517971368): detailed breakdown of harness components and why each exists
- [Agent Harness Engineering](https://www.oreilly.com/radar/agent-harness-engineering/): convergent framing on Agent = Model + Harness, with coding agent examples
- [Harness Engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/): real-world account of building a product entirely with Codex agents, covering scaffolding, feedback loops, and context management at inference
- [Tool Schema Rendering Atlas](https://huggingface.co/spaces/evalstate/tool-research) (evalstate): how tool schemas become prompt text across models, showing what each model actually sees after provider templates are applied
- [Simon Willison's How coding agents work blog](https://simonwillison.net/guides/agentic-engineering-patterns/how-coding-agents-work/): how a coding agent works as a harness
- [AI Engineer talks like Harnesses in AI: A Deep Dive](https://www.youtube.com/watch?v=C_GG5g38vLU): what a harness is and how to build one.
- [The Ultimate Guide to RL Environments](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide): framework-by-framework comparison and vocabulary translation
- [Continually improving our agent harness](https://cursor.com/blog/continually-improving-agent-harness): how Cursor iterates on its harness as a product.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): the canonical eval harness

*If any definition feels imprecise or you've encountered a term we've missed, we'd love to hear from you.*

*Thanks to [@pcuenca](https://github.com/pcuenca), [@qgallouedec](https://github.com/qgallouedec), [@evalstate](https://github.com/evalstate), and [@adithya-s-k](https://github.com/adithya-s-k) for reviewing this post.*
