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

We built a Skill that mlx-lm contributors can use to port a model from transformers to MLX. Given a prompt like "convert the olmo_hybrid architecture to MLX", the Skill sets up a virtual environment to work on, discovers and downloads the relevant models from the Hub, reads the transformers modeling code, writes the MLX implementation, and runs a battery of tests. If results don't look right, it debugs and iterates, and does not declare success until it's satisfied.

We designed it to be useful to reviewers as much as contributors.

**For the contributor**, the Skill of course handles all the scaffolding: finding model variants on the Hub, diffing their configs to spot parameters that vary across model variants, downloading checkpoints, setting up editable installs of both mlx-lm and transformers. But it also handles the more difficult modeling tasks. It pays attention to salient architecture details and verifies sensitive areas, like RoPE configurations, that may result in hard-to-find bugs. It detects when the config doesn't declare a dtype and infers it from the safetensors metadata header. It runs per-layer comparisons between transformers and MLX to pinpoint exactly where divergence occurs. These are the kinds of checks that only someone with porting experience would think to run.

**For the reviewer**, the Skill produces a PR that is upfront about being agent-assisted, but does look like a careful human submission. Reviewers will see that the code follows mlx-lm conventions: idiomatic solutions, no unnecessary comments, no speculative abstractions, no modifications to shared utilities without explicit approval. Given that the code is agent-assisted, we try to include _more_ data than the median PR, to provide as much signal as possible. The PR body includes a report with a summary of the variants and their architectural differences, generation examples, numerical comparisons, dtype verification, per-layer comparisons against the transformers baseline. The PR always discloses that it was agent-assisted, and the Skill will not open it until the contributor has accepted the results.

**For verification**, the Skill generates a test manifest for a separate, non-agentic test harness that is, by design, easily reproducible and not subject to LLM hallucinations or complacency ([more on this below](#the-test-harness)).

## How we did it

Skills are recipes for agents: simple text files with guidelines that steer the model through a complex task. They are not magic; you can achieve the same results via prompting and iteration. But they provide _consistency_ (every run follows the same process, whereas different people would prompt differently), minimize ambiguity and serve as documentation: anyone can read the Skill to understand what it does, identify missing cases and suggest improvements.

We bootstrapped the Skill by porting a model ourselves, in conversation with Claude. I asked it to port GLM 4.7 from transformers to mlx-lm, giving instructions as I would during a normal session. One trick: I pointed Claude at a checkout of mlx-lm from which I had deleted the already-existing implementation, so I could compare the output against the ground truth. After a few iterations I had a working implementation, a conversation that revealed how Claude approached the problem, and the first draft of the Skill, which Claude created as a summary of the process. I edited it heavily, and incorporated the learnings from [@gabegoodhart](https://huggingface.co/gabegoodhart), who kindly shared [his own porting conversation](https://github.com/ml-explore/mlx-lm/pull/442#issue-3399360107) for a different model 🙌.

We repeated this loop several times and the Skill grew. On the technical side, we covered stuff such as [RoPE bugs](https://x.com/Prince_Canuma/status/1982913823888814334) that may produce plausible output that degrades with long sequences, float32 precision contamination that silently kills inference speed (you'd be surprised how frequent these things happen!), config fields that vary across model variants in ways the implementation must handle, distributed inference for super large models that don't fit on a single machine. We taught it how to invoke the `hf` CLI to discover and download models. Most importantly, we instructed it to run the tests that experienced porters would, and to not declare success until they pass.

On the cultural side, we covered _softer_ characteristics and explained the conventions that make a PR easy to review: don't use comments to explain code (the reviewer has to parse the comment _and_ the code 🤦‍♂️), never propose refactors, don't touch shared utilities without asking. These rules cost the agent nothing but save the reviewer lots of time.

[todo: conversation snippets]

The end result: the contributor types a prompt, and the Skill produces a PR like [this one](https://github.com/ml-explore/mlx-lm/pull/1023), plus a test manifest for the external test harness.

## Test harness

The Skill shares a comprehensive results report as part of the PR. All these come from tests the agent runs during conversion, but we didn't want the reviewer to take a leap of faith to accept them. To go a step forward, we created a separate, non-agentic test harness that runs systematic tests on the converted code. This brings a couple of benefits:

- Removes uncertainty about the LLM hallucinating results, or being too complacent about them.
- Guarantees reproducibility: anyone can download the test harness repo and run the tests.
- Documentation and transparency. All results are saved at various levels: [summary reports](https://github.com/pcuenca/mlx-lm-tests/blob/main/results/pr-5/2026-04-14T122120-7ce7a68/summary.md#layers--ran), [per-model details](https://github.com/pcuenca/mlx-lm-tests/blob/main/results/pr-5/2026-04-14T122120-7ce7a68/summary.md#allenaiolmo-hybrid-instruct-sft-7b), [raw inputs/outputs](https://github.com/pcuenca/mlx-lm-tests/tree/main/results/pr-5/2026-04-14T122120-7ce7a68/allenai--Olmo-Hybrid-Instruct-SFT-7B) saved as JSON files. The [tests](https://github.com/pcuenca/mlx-lm-tests/tree/main/results/pr-5/2026-04-14T122120-7ce7a68/scripts) are also copied to results folders so we know what we ran even if we make changes to the harness in the future.

The test harness is not a CI gate. Some checks are straightforward (is the output dtype correct?), but most are qualitative. Is it normal that a pre-trained model repeats itself in long sequences? Is a 4% relative logits difference against the transformers baseline acceptable? These are judgement calls based on experience with similar architectures. The harness provides useful signal, but it's the reviewer and contributor who still have to make the call.

## How to use the Skill

The Skill is designed for the people who are already opening mlx-lm model PRs, or who would do it manually on their own. It's not meant for mass consumption, because PRs to mlx-lm are rarely accepted on sight. The typical cycle is: contributor opens a PR, reviewers point out improvements, both sides iterate until the quality bar is met. If this is true for expert submissions, it will remain true for agent-assisted ones.

If you're not prepared to engage in that cycle, you probably shouldn't be opening a PR. The reviewers will make an effort to understand your code (even knowing it was agent-assisted), so you should do the same. Own the code, and be ready to incorporate their feedback. In particular, don't hand reviewer comments back to an agent and post whatever it produces. LLMs double down on their decisions, go on tangents, and don't push back effectively. Once you engage with the reviewer, this becomes a person-to-person conversation, so it's your turn to discuss and be respectful of the time they put in.

You can also use the Skill to learn; you don't need to submit anything until your confidence and experience build. Read the Skill to identify problem areas you weren't aware of. Point it to your own fork of mlx-lm, try a conversion, and compare your output with the accepted implementation once it lands in the official repo. If you do this a few times, you'll learn a lot about transformers, MLX, and language model architectures.

If you're ready:

```bash
uv run https://raw.githubusercontent.com/huggingface/transformers-to-mlx/main/install_skill.py
uvx hf skills add --claude
```

We developed and tested the Skill using Claude Code. The same approach would work with Codex or other coding agents, but we haven't tested them. If you try the Skill in a different environment, please let us know how it goes!

## Next steps and known shortcomings

The Skill works well for LLMs in mlx-lm, but there's plenty of room to grow.

### What's next

- **mlx-vlm**. Vision-language models live in a [separate repo](https://github.com/Blaizzy/mlx-vlm) with different conventions. Beyond the modeling code, mlx-vlm requires _processors_ to handle image pre-processing before the LLM sees the input. We're looking forward to collaborating with [Prince Canuma](https://huggingface.co/prince-canuma) to help him do what he does.
- **llama.cpp**. Some of the same challenges apply. Processors require image processing algorithms to be replicated in C++, and numerical differences are unavoidable. This is an area where a tightly scoped agent might help.
- **The test harness**. We want to expand the test battery and potentially explore safe automation to run tests automatically on our infra.

### What doesn't work yet

- **Shared utilities in mlx-lm**. mlx-lm is less strict than transformers about extracting common patterns into shared functions. The Skill is purposefully biased towards self-contained model files (same as transformers), but reviewers regularly ask for refactors to move repeated code into shared modules.
- **VLMs and other architectures**, as noted above.
- **Quantized model uploads**. The Skill tests quantization but doesn't upload quantized models to the Hub. We think it doesn't make sense to upload while the PR is being reviewed, but we could create a flow to do it later.
- **Thinking tests**. No thinking-specific tests have been designed yet. The Skill will convert and verify generations from these models, but won't validate the thinking structure.

## Conclusion

The bottleneck in open source is not typing speed: it's understaning the codebase to change it without breaking the implicit and explicit contracts with users. Agents can help in this process, if we teach them what matters. We explored what this looks like in the context of mlx-lm, and hope it's useful for contributors and reviewers to land high-quality model conversions faster!

## Resources

Contributions:
- [transformers-to-mlx Skill repo](https://github.com/huggingface/transformers-to-mlx)
- [Test Harnes repo](https://github.com/pcuenca/mlx-lm-tests)
- [Example agent-assisted conversion against a fork](https://github.com/pcuenca/mlx-lm/pull/5)

The libraries:
- [mlx-lm, the target library](https://github.com/ml-explore/mlx-lm)
- [transformers, the source of truth for modeling code](https://github.com/huggingface/transformers)

Background:
- [Claude Code Skills docs](https://code.claude.com/docs/en/skills)
- [Transformers design philosophy](https://huggingface.co/spaces/transformers-community/Transformers-tenets)
