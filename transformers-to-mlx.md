---
title: "The PR you would have opened yourself" 
thumbnail: /blog/assets/transformers-to-mlx/thumbnail.png
authors:
- user: pcuenq
- user: lysandre
---

# The PR you would have opened yourself

_`transformers-to-mlx`: a Skill and test harness to support porting models from transformers to MLX_

## tldr

We provide a **Skill** and a **test harness** to help port language models from transformers to mlx-lm, so they become (almost) instantly available the moment they are added to transformers. The Skill is designed to support contributors and reviewers as an aide, not an automation. We explain why we did it, how, and comment about how to meaningfully contribute to open source in the age of agents.

## The advent of code agents

In 2026, code agents suddenly started to work. What used to be a distracting auto-completion tool at the side of your editor has turned into a system that will one-shot a solution based on some brief specifications. The generated code usually works on the first try, covers what you asked for, and has reasonable assumptions about details you didn't specify. This is great. As Jensen Huang puts it, [we've instantly gone from 30 million to one billion coders](https://www.youtube.com/watch?v=vif8NQcjVf0&t=7324s) in the world. Creative minds are unleashed.

We have to adapt and learn how to work in this environment. I'm not referring to the impact on those of us who write code for a living, but to all the assumptions and relationships that we take for granted. Open source projects are one of the first arenas where change is happening, and we have to rethink what it means to _help_ as a contributor.

Take the example of the transformers library. It has received contributions from hundreds of people, is used massively and has been downloaded more than a billion times. We are super lucky that this is the case and I'm not mentioning it to sound authoritative, I just want to convey _volume_. Suddenly, anyone with an agent can instruct it to find some open issue, fix it, and submit a PR. And that's exactly what's happening. Those people feel happy because they are contributing to a great library, but the sad reality is that, most of the time, they don't realize they are not.

<!-- TODO: clem's screenshot or something similar -->

Why not?

Let's just consider the sheer amount of PRs received. There's a very small number of transformers maintainers that have to go through all open PRs, understand them, decide if the design direction is correct, identify potential issues, incompatibilities, API changes, side effects, and propose feedback to the author based on this analysis. The amount of PRs has increased ten-fold, but the amount of maintainers has not (and cannot, because team coordination does not scale).

But why is that a problem, Pedro? Let's just accept those PRs and build a better and larger transformers library faster than we ever could! Or, even better, let's have an agent review those PRs so you all don't have to!

There are two underlying assumptions that agent-generated PRs don't usually take into account.

- Codebases like transformers care deeply about the code. It's cool to build projects where it doesn't matter what the code looks like, but transformers can't afford that. Transformers is used by thousands of users and incorporated as a dependency in hundreds of libraries. Any change in the coding standards, APIs, design decisions, will immediately affect thousands of people, that will have to re-learn how to work with the library and rebuild their mental models of what it does and _how_ it does it. Transformers, and many other projects, are primarily built as a human-to-human communication method, through code. One of the most important goals of transformers is that a model implementation file should be self-contained, so it can be understood by a practitioner who reads it top to bottom. This permeates throughut the library design and is the reason why we favor flat hierarchies, as we've already explained [link pending].
- These decisions and goals are usually not explicit, and agents don't have all the context. They are happy to suggest refactors to "improve" the codebase by following "best practices", without realizing those refactors break the implicit contract between the library and its users. They are verbose, generalize solutions too early, don't always notice when a change affects other areas of the library, introduce subtle bugs, break performance. They are also largely sycophantic, and accept any idea as a good one and follow it through diligently. They are indulgent with their approach and can happily accept superfluous patches.

## What does this have to do with MLX?

Transformers is a special case because of its volume, so we are one of the first projects that need to learn how to cooperate with agents efficiently. (Sidebar: a different example in another domain is the App Store - reviewers are swamped because anyone can now build and submit an app, so many do).

The change is spreading everywhere. Like transformers, MLX cares deeply about the code, and their maintainers need to carefully read all open PRs to decide which ones are good for the library, and iterate with their authors to bring them home.

We set out to build an agent-assisted model porting tool for mlx-lm that helps contributors create high-quality PRs, and to provide value to both the contributor and the reviewer. As we'll see in a moment, it helps with testing, pays attention to edge cases and knows about frequent porting pitfalls.

I believe that many of the PRs to mlx-lm that contribute new models are agent-assisted these days. Rather than hiding this fact under the rug, we take this as an opportunity to explore how agents can effectively support open source contributions.

[I haven't explained "transformers as the source of truth" yet. The intro is too large, we should reduce but add this point]

## What we did

We are sharing a Skill with the community that mlx-lm contributors can use to assist them when they want to port a model from transformers to MLX. These are our driving goals:

- Help mlx-lm contributors land model ports fast, reducing the time it takes to build a port since it's added to the transformers codebase until the architecture is available in mlx-lm.
- Always respect the idiosyncrasies of the mlx-lm library: code style, idiomatic solutions, design directions, and any other explicit and implicit convention encoded in the codebase.
- Pay attention to the details, and test for potential failure cases. For example, bugs in the implementation of RoPE may not prevent a model from generating seemingly coherent output, but quality degrades in long sequences. This is an example of something difficult to test, time-consuming, and that only people with LLM experience would test for.
- Help the contributor with easy and advanced tasks alike. For example:
  * Model repositories are automatically found and downloaded from the Hugging Face Hub.
  * Configuration differences across models of the same architecture are detected and explicitly tested for. The agent will look at the transformers reference implementation to verify how to deal with these variants.
  * Common pitfalls and non-subtle failures are considered.
  * Run tests to verify generation coherence and adherence to the transformers reference implementation.
  * Verify the output dtype is the expected one (too many times this has been the reason why a port appears to be too slow).
- Support the reviewers as well.
  * The generated PRs contain a comprehensive report that includes generation examples, notable architecture details, models that were tested, etc.
  * Test results are shared as part of the PR. The Skill will not open a PR until the agent and the contributor consider the implementation correct.
  * PRs are upfront about the use of agents and disclose exactly what they did (everything's that's in the report).
  * A non-agentic test harness can be run on Hugging Face's infra for verification. It includes slow, long-context generation tests that may surface subtle bugs, and supports large models that require considerable amounts of RAM to run. These tests complement the ones performed by the agent, but they are, by design, easily reproducible and not subject to hallucinations or complacency.

We are committed to updating this Skill as contributors use it and the mlx-lm library evolves.

## How we did it

Skills are recipes for agents. They are not magic –you can achieve the same results via prompting and iteration–; in fact, they are just a text file with instructions. But they provide _reproducibility_ and _coherence_: the Skill contains comprehensive guidelines intended to steer the model to produce the desired output and resolve ambiguity. They are also great for documentation: anyone can read the skill to learn what the instructions look like, and suggest improvements when unexplored cases arise. 

We built the Skill through a few conversations with Claude. First, I asked Claude to port a language model (GLM 4.7) from transformers to mlx-lm, and gave instructions on how to do it, just as I would during a normal session. I cheated a bit: I pointed Claude to a local checkout of mlx-lm from which I had deleted the already-existing implementation. After a few iterations with Claude, I had a conversation that revealed how it thought about the problem, a working implementation, and a diff I could compare with the "ground truth" from the mlx-lm project. I then asked Claude to write a summary of these learnings, which became the first version of the Skill file. I went through it and modified to my taste, then I added the learnings from [this contributor](https://github.com/ml-explore/mlx-lm/pull/442#issue-3399360107), who kindly shared his own model-port conversation for a different model (thanks [@gabegoodhart](https://huggingface.co/gabegoodhart) 🙌).

We repeated this process a few times, and incorporated lots of feedback to the Skill: failure cases were generalized, we taught the model how to deal with super large models (use distributed inference across various computers), how to identify candidate repos to be tested, how to download them with the `hf` CLI, and many other useful pieces of knowledge. We paid a lot of attention to the verifications performed during conversion, and asked to explicitly include long-sequence generation tests. The devil is always in the details, so we did our best to imbue the Skill with a sense of problematic areas to be super careful about, [such as RoPE](https://x.com/Prince_Canuma/status/1982913823888814334), or precision contamination to float32 that kills performance. The Skill is tasked with finding out novel architecture details, and focuses on those.

In addition to all these technical and "correctness" details, we also explained _softer_ characteristics: do not use comments to explain code (the reviewer has to parse the comment _and_ the code 🤦‍♂️), never propose refactors, do not modify common utility functions shared with other models unless the user approves.

[todo: find a couple of interesting conversation snippets to include]

The goal is that the user provides a prompt like "Please, convert the olmo_hybrid architecture to MLX", and the Skill produces a PR like [this one](https://github.com/ml-explore/mlx-lm/pull/1023), plus a test descriptor file to be run by an external test harness.

## The test harness

The Skill we created knows how to implement a new architecture, how to identify models to convert and how to test them, and it produces a quite comprehensive results report that is shared as part of the PR. Tests are run by the LLM itself, but we wanted to go a step forward, and created a separate test harness to further validate the new implementations. This serves a couple of purposes:

- Removes uncertainty about the LLM hallucinating results, or being too complacent about them.
- Persists tests and results in a separate repo, so anyone can review them.
- Provides an straightforward way to reproduce results. Tests and results are uploaded on a per-model basis, with all dependencies documented. Using `uv`, any user is a command-line away from running them on their own computer for verification, or to test potential regressions.

This is not meant as an additional CI gate that we impose on mlx-lm, but as a verification tool for the conversions implemented by the agent Skill. We also provide some infra of our own that contributors can use: one maxed-out M3 Ultra with 512 GB of RAM. Because this is not a CI, and because of security considerations, running the external test harness on our infra is not automated – you can run on your own computer, or ping `@pcuenca` in your GitHub PR so I can run it for you. Regular contributors will be whitelisted as usage picks up.

## How to Use the Skill

The Skill has been designed for model contributors, and our goal is to help them and mlx-lm reviewers by taking advantage of what agents can offer. The target audience, therefore, is the same people that are opening PRs right now (or those who would open them manually on their own). The reason this is not intended for _mass_ consumption is that, as we explained, maintainers care deeply about the library and will only accept contributions after they have spent time making sure that the PR works as expected and the contribution is actually useful. It's very rare that these complex PRs are accepted on sight, even when submitted by experts. The typical scenario is that a contributor opens a PR, then the reviewers point out deficiencies or improvements, and they both iterate for a while, until the desired level of quality is met. We expect this process to continue for the foreseeable future, so if you're not prepared to engage in this iteration cycle, then it's a sign that you probably shouldn't be opening a PR [^1]. The reviewers will make an effort to understand your code (even if it was assisted by an agent), so you should at least do the same, own the code, and be ready to incorporate their feedback.

[^1]: We don't think it's productive to ask an agent to deal with reviewers' feedback: LLMs will double-down on their decisions, go through tangents, not refute effectively. This is something you should do as a contributor. And it's respectful of the time the reviewer put in.

You can also use the Skill to learn. One effective way to do it is just by reading it to identify problem areas you may have not be aware of. Another way is to tell the Skill to use your own fork of mlx-lm and try your hand at a conversion, but without submitting a PR to the official repo. You can then study the generated code and compare it with the accepted implementation once it becomes available in the official repo.

If you are an mlx-lm contributor or ready to become one, you can install the Skill using:

[to do]

Disclaimer: we have developed and tested the Skill using Claude Code. The same approach would have worked with Codex or other coding agents, but we haven't tested them. If you use this Skill in other environments, please let us know how it goes!

## Next Steps

mlx-vlm etc.