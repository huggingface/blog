---
title: "The Open Source Community is backing OpenEnv for Agentic RL"
thumbnail: /blog/assets/openenv/thumbnail3.png
authors:
- user: burtenshaw
- user: spisakjo
- user: lysandre
- user: darktex
- user: willcb
- user: qjoy
- user: pawalt
- user: cwing-nv
- user: danielhanchen
- user: andrewzhou
- user: thegovind
- user: shimmyshimmer
- user: Hamid-Nazeri
- user: Sanyam
- user: zkwentz
- user: emre0
- user: lewtun
- user: sergiopaniego
- user: banghua
---

# The Open Source Community is backing OpenEnv for Agentic RL

![Thumbnail for the blog post](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv-expansion/banner.png)

OpenEnv is a tool for creating an agentic execution environment like terminals, browsers, or anything an agent can interact with. And today, we’re excited to announce that OpenEnv is becoming even more open, to make the future of training agents open source.

Starting today, OpenEnv will be coordinated by a committee that so far includes Meta-PyTorch, Reflection, Unsloth, Modal, Prime Intellect, Nvidia, Mercor, Fleet AI, Microsoft, and Hugging Face. `OpenEnv` now lives at [`huggingface/OpenEnv`](https://github.com/huggingface/OpenEnv)

OpenEnv project is supported and adopted by some of the leading organizations in the AI ecosystem, including PyTorch Foundation, vLLM, SkyRL (UCB), Lightning AI, Axolotl AI, Stanford Scaling Intelligence Lab, Mithril, OpenMined, Scaler AI Labs, Scale AI, Patronus AI, Surge AI, Halluminate, Turing, Scorecard, and Snorkel AI.

## Why we need OpenEnv to train open source agents

Agent harnesses like Claude Code, Codex, OpenClaw, and Hermes just keep improving. One reason for their improvement is that models like GPT-5.5 and Opus 4.8 are trained to use their respective harnesses.

We want those gains with open source models too: training local models that use harnesses effectively, and saving compute by specializing models for specific tasks.

## Why we need to be (even) more open

Frontier labs train models and harnesses that, for the most part, work like hand in glove. The model is trained to use the harness and optimised for its characteristics. Models can generalise beyond these harnesses, to some extent, but nothing beats the efficiency of training.

![the open source reinforcement learning ecosystem](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv-expansion/diagram.png)

In the open, this isn’t the case. Developers use any harness, any model, any inference engine, on whatever use case they value. This is fundamental to the community, but it’s also a challenge that requires infrastructure and tooling to tackle.

That’s where OpenEnv comes in. It’s a library to interface between harness, environment, and trainer, which works on any model. For this to stick, it will need to be owned by all the major stakeholders.

## A protocol layer, not a reward framework

Alongside the governance change, we're tightening what OpenEnv *is*.

In recent releases, OpenEnv has become an **interoperability layer for RL environments**. Its job is to standardize how environments are published, deployed, and consumed by agents. It will not dictate how rewards are defined or how training loops work. Reward definition, scoring rubrics, and trainer-specific logic belong in the libraries that specialize in them. OpenEnv is the common socket they can all plug into.

In practice this means:

One interface, many environments which all expose the familiar Gymnasium-style API (`reset()`, `step()`, `state()`) running on a client/server architecture. A trainer that speaks OpenEnv can drive any compliant environment without bespoke code.

Familiar protocols and canonical packaging. Environments are served over standard protocols like HTTP and WebSocket and packaged with Docker. MCP is a first-class citizen, so OpenEnv environments are instantly compatible with MCP servers and the same environment behaves consistently in both simulation (train/eval) and production modes.

Interop across env libraries. You can define and consume environments across different ecosystems (verifiers, harbor, and others) and on the infrastructure and hub of your choice. OpenEnv is the deployment and interface layer underneath them, rather than a competitor to them.

## What's next

Over the coming months we will focus on the things that turn OpenEnv from a fast-growing project into a dependable standard:

1. External rewards: letting rewards be defined in whichever library you already use, with OpenEnv as the deployment layer ([RFC 006](https://github.com/huggingface/OpenEnv/pull/794)).
2. Tasksets via datasets: wiring environment tasks to Hugging Face datasets so environments and benchmarks compose cleanly ([RFC 007](https://github.com/huggingface/OpenEnv/pull/795)).
3. Continued Harness integration: first-class support for agentic harnesses.
4. End-to-end examples: full training and evaluation walkthroughs in TRL, Unsloth, and beyond.
5. Auto-validation: measure environment quality and contribution to model learning. This will give the community a scalable way to evaluate their environments and drive up quality (think hackathons!). [RFC 008](https://github.com/huggingface/OpenEnv/issues/778).

## Get involved

OpenEnv is community-centric by design, and it's still early — expect rough edges, and help us smooth them. Check out the code and RFCs: [github.com/huggingface/OpenEnv](https://github.com/huggingface/OpenEnv)

Thanks to everyone who helped make this transition happen. Let's build the common substrate for open-source agentic RL together.
