---
title: "Introducing harness adapters: adapt your favorite model to your harness to improve performance and cost"
thumbnail: /blog/assets/harness-adapters-at-scale/thumbnail.png
authors:
- user: ybelkada
- user: burtenshaw
- user: SaylorTwift
- user: lysandre
- user: aminediroHF
- user: AdithyaSK
- user: evalstate
- user: sergiopaniego
- user: joelniklaus
- user: lewtun
---

# Introducing harness adapters: adapt your favorite model to your harness to improve performance and cost

Combining RL on a custom task distribution with your favorite agent harness yields more accurate and efficient models for that target harness and task distribution.

Models are not fully optimized for all harnesses and the tasks you’re doing. Instead, they target general performance. So when you combine a model and harness, you’re leaving performance and efficiency on the table for the stuff you’re actually doing. With RL, you can train a custom harness adapter that squeezes everything out of your model for your harness and task combination.

Take Qwen3.8-27B, its Terminal-Bench 2.1 score with the pi harness is 39% which uses on average 2.9K tokens per task. If we freeze those weights and add a custom RL-trained LoRA adapter we can get 55% pass@1, with average token usage of 1.5K. Most of this is down to the model reasoning less, calling less tools, and performing better first time.

Our earlier work ([multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl), [harness adapters](https://huggingface.co/blog/harness-hub/harness-adapters)) showed that RL on your own task distribution, inside your own harness, gives a more accurate and more efficient model. Those experiments covered data-analysis and code-building tasks. Here we wanted to push this further by exploring whether the result holds at 27B, on harder real-world software engineering tasks, across three harnesses: Pi, OpenCode and Hermes Agent. And most of all, share a fully reusable recipe, so you can reproduce this on your own task, harness, model combinations.

## TL;DR

- We trained one LoRA adapter for Qwen3.8-27B per agent harness (pi, Hermes Agent, OpenCode), with RL inside the harness itself.
- Datasets, training and agent rollouts all happen on the Hub in Jobs using TRL's `AsyncGRPOTrainer`, OpenEnv and harbor.
- Each harness adapter beats the base model under its own harness on both benchmarks, for both performance and efficiency.
- The three LoRAs are also merged into a single super adapter that runs unchanged under pi, Hermes Agent and OpenCode.
- All artifacts (adapters, GGUFs, RL tasks, safeagents plugin) can be browsed through this [Hugging Face collection](https://huggingface.co/collections/harness-hub/qwen38-27b-harness-adapters).

## Table of Contents

- [Training inside agent harnesses](#training-inside-agent-harnesses)
- [Generating custom RL tasks on Hugging Face](#generating-custom-rl-tasks-on-hugging-face)
- [Testing the filters with short training runs](#testing-the-filters-with-short-training-runs)
- [`AsyncGRPOTrainer` for more efficient GRPO training](#asyncgrpotrainer-for-more-efficient-grpo-training)
- [Training setup](#training-setup)
- [Super-adapter: merging all adapters into one](#super-adapter-merging-all-adapters-into-one)
- [Final results](#final-results)
- [Open question: What did the model really learn to do?](#open-question-what-did-the-model-really-learn-to-do)
- [Closing words and future directions](#closing-words-and-future-directions)

## Training inside agent harnesses

We leverage the [Harbor integration in OpenEnv](https://huggingface.co/docs/openenv/environments/harbor) and [TRL’s OpenEnv](https://huggingface.co/docs/trl/openenv) guide to train the model inside the harnesses we chose for the final training. We have built ready-to-use docker images for all our RL tasks that come with the target harnesses already pre-installed inside. In our final docker images, we use Pi `1.5.2`, OpenCode `2.2.7` and Hermes Agent `v0.22.6`. The diagram below from the aforementioned blogpost explains well the entire protocol:

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/loop.html" width="100%" height="520" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

Most RL frameworks train a re-implementation of the agent loop: the agent is written against the framework's own interface, such as [verl's AgentLoop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html), [SkyRL-Gym](https://github.com/NovaSky-AI/SkyRL) environments, [rLLM](https://github.com/rllm-org/rllm)'s agent and environment classes, [slime](https://github.com/THUDM/slime)'s custom rollout functions, or [ART](https://github.com/OpenPipe/ART)'s client inside your own Python agent. Here the harness runs exactly as it ships, and the trainer learns from the exact tokens it produced, recorded by a proxy between the harness and vLLM. Recent work moves in the same direction: [Agent Lightning v1.0](https://huggingface.co/papers/2608.17528) keeps the deploy-time harness in the loop through a proxy ("harnessed agentic RL"), and [SkyRL's Harbor integration](https://novasky-ai.notion.site/skyrl-harbor) trains on Harbor trials, converting their chat histories into tokens for the trainer.

## Generating custom RL tasks on Hugging Face

For this recipe we created a custom dataset for agentic coding with the Hub, OpenEnv, and [Repo2RLEnv](https://github.com/huggingface/Repo2RLEnv). We combined together task retrieval of existing RL envs, task generation with Repo2RLEnv, task validation with OpenEnv, and sampling into a single data pipeline. This gave us the highest quality set of validated and relevant tasks to improve Qwen3.8-27B on agentic coding. In this section, we’ll walk through how we built it, so you can try it out on your own domains.

RL training needs tasks the agent can attempt, a verifier that scores each attempt, and evidence that the verifier is right. A verifier that accepts wrong answers teaches the wrong behavior. A task that is already feasible will not teach the model anything.

This section describes how we built the task set from three sources: two on the Hugging Face Hub and a third generated with Tasksmith, a pipeline in Repo2RLEnv. Every task passes an eligibility screen with `openenv validate`, and the same difficulty check against the base model, Qwen3.8-27B.

<p align="center">
    <img src="/blog/assets/harness-adapters-at-scale/task-pipeline.png" alt="Task pipeline: Tasksmith, SWE-smith and SWE-Gym sources pass through an eligibility screen, Harbor packaging, openenv validate on Hugging Face Jobs, and a difficulty check with Qwen3.8-27B" width="80%">
</p>

### Three sources

Each task uses the Harbor format, which Terminal-Bench 2.1 also uses. A broad search of the Hub for Harbor environments returned about 59,000 tasks, but they included duplicates, evaluation sets and tasks marked as not for training. We filtered down to just two Hub sources built for training and generated a third set ourselves:

- SWE-smith ([Python](https://huggingface.co/datasets/SWE-bench/SWE-smith-py), [Go](https://huggingface.co/datasets/SWE-bench/SWE-smith-go)) injects bugs into real Python and Go repositories, with an LLM rewrite or a mutation such as a flipped operator. The tests that catch each bug become the task's tests. It adds volume.
- SWE-Gym, packaged as Harbor tasks in [TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove), turns real GitHub issues and their merged fixes into tasks. It adds real-world bugs.
- [Tasksmith](https://huggingface.github.io/Repo2RLEnv/pipelines/tasksmith/) generates tasks from merged pull requests. An LLM agent writes the instruction and tests, and Tasksmith checks the result: the reference and a valid alternative must pass, while the unfixed code and a known-wrong solution must fail.

We used Tasksmith to target gaps. We mapped the skills that our target benchmarks test (Terminal-Bench 2.1, the hard subset of SWE-bench Pro, and DeepSWE), then chose pull requests that exercise them, such as HTTP streaming, filesystem operations and serialization. From 30 merged pull requests, Tasksmith produced 24 tasks in 10 Python repositories that passed its checks, 17 of them on a second run. Its checks do not replace ours, so these tasks go through the same pipeline as the rest.

### Validation with `openenv validate`

A metadata screen runs first. It keeps tasks from training splits with permissive licenses. It drops duplicates, tasks marked as not for training, the 253 tasks in our target benchmarks, and any task from a repository those benchmarks use. We then ran the semantic checks from `openenv validate` on every task in fresh trials:

- The reference solution earns the maximum reward (`semantic.oracle_max`).
- Doing nothing scores clearly lower (`semantic.floor_gap`).
- No canary scores more than doing nothing (`semantic.canary_floor`).
- A replay of the reference gives the same reward (`runtime.episode_determinism`).

Each trial ran as its own Hugging Face Job, which let us validate dozens of tasks at once. `openenv validate` offers the same option for any Harbor task:

```bash
openenv validate path/to/task --sandbox hf-sandbox
```

858 of 1,127 Hub candidates passed. Wrong-answer controls caught defects, including a Tasksmith task that accepted an incomplete implementation and a Terminal-Bench 2.1 verifier (regex-log) that accepted known-wrong answers. Most failures were infrastructure problems, such as images that did not start. Failed tasks stay in a rejected split with the failing check recorded.

### Difficulty check with the base model

A valid task can still teach nothing. GRPO (group relative policy optimization) scores each attempt against the other attempts at the same task, so a task where every attempt passes, or every attempt fails, gives no learning signal. Before training, Qwen3.8-27B attempted each task 4 times, or 8 times for a stratified sample of six tasks per repository.

Of 442 scored tasks, Qwen3.8-27B solved 151 every time and 120 never. The 171 tasks it solved only sometimes form the qualified set. The figure below shows how the scores pile up at both ends and how they differ by source. The model solves most Go tasks every time, fails every attempt on more than a third of the SWE-smith Python tasks, and lands in the middle on 73% of the SWE-Gym tasks.

<p align="center">
    <img src="/blog/assets/harness-adapters-at-scale/difficulty-check.png" alt="Qwen3.8-27B results on 442 scored tasks, split by attempts solved and by source" width="90%">
</p>

*Qwen3.8-27B results on 442 scored tasks. Green marks the qualified tasks, which the model solves on some attempts but not all.*

This example shows how we filtered and validated 1,127 tasks to 278 which we included directly in the final dataset. The pipeline then took the remaining 522 rejected tasks, repaired and complexified them with Tasksmith and repeated validation.

## Testing the filters with short training runs

The pipeline applies two filters: validation removes broken tasks, and the difficulty check keeps tasks the base model solves only sometimes. This section tests whether those filters change what the model learns. We trained four short runs of up to 50 steps, each on tasks from a different stage of the pipeline, and compared them with the base model on Terminal-Bench 2.1.

Each run trains Qwen3.8-27B with LoRA and GRPO, using TRL's `AsyncGRPOTrainer`, on 32 tasks. All four use the same agent harness (OpenCode with up to 48 model calls), the same settings and one 8×H200 Hugging Face Job, and the reward is the verifier's pass or fail. The task sets differ in how they were filtered:

- **Unvalidated**: 24 accepted tasks plus 8 that failed validation.
- **Validated**: 32 accepted tasks, chosen without regard to difficulty.
- **Unqualified**: 32 accepted tasks that the base model always or never solves.
- **Qualified**: 32 accepted tasks that the base model solves on some attempts.

The figure below shows how the tasks impacted model learning. The table lists each run's steps and the checkpoint we evaluate. The chart tracks training reward across subsets of tasks with differing validation status.

<iframe src="https://harness-hub-tb21-lift-status.static.hf.space" title="TB2.1 lift experiment status" width="100%" height="760" style="border:0;max-width:880px" loading="lazy"></iframe>

### The dataset

`harness-hub/agentic-coding-envs` holds 871 tasks from 35 repositories. Hub tasks are capped at 40 per repository, so no single project dominates. The table below shows what each source contributes. The dataset has two views: every accepted task, and the qualified subset for GRPO training. Rejected attempts stay in a separate config.

| Source | Task type | Tasks | Qualified |
|---|---|---:|---:|
| SWE-smith | Injected bugs in Python and Go | 717 | 144 |
| SWE-Gym (TaskTrove) | Real GitHub issues | 130 | 27 |
| Tasksmith | Generated from merged pull requests | 24 | 22 |
| **Total** | | **871** | **171** |

## `AsyncGRPOTrainer` for more efficient GRPO training

We leveraged the stable version of `AsyncGRPOTrainer` from TRL, with the main advantage of the trainer being that it does not wait until all rollouts are completed before moving to the next training optimization step.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/asyncgrpo-sync-async.html" width="100%" height="390" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

For more detailed reading on the trainer and its building blocks, please refer to [the AsyncGRPO blogpost](https://huggingface.co/spaces/trl-lib/async-grpo-stable).

## Training setup

To obtain our final harness adapters, we trained LoRA adapters on top of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) using TRL's `AsyncGRPOTrainer`. The adapters were attached to the model using the `all-linear` flag. To maximize the training efficiency, thinking mode was turned off during rollout stages, and we evaluated our final adapter with and without thinking for broader comparison.

The final reward function consists of the verifier's correctness, combined with a small tool-efficiency bonus (0.1, derived from [early experiments from the previous iteration of harness adapters](https://huggingface.co/blog/harness-hub/harness-adapters) which showed that 0.1 worked best).

We performed a small LR sweep on 50 steps using Termgrade environments (1e-5, 5e-5 and 1e-4) and noticed that 1e-5 barely moved training correctness (from 0.52 to 0.50), 1e-4 collapsed the rollouts instead of improving them (turns per rollout moved from 18 to 6, with entropy rising), and only 5e-5 clearly improved (from 0.49 to 0.60 by step 40). Longer 5e-5 runs then showed signs of instability (collapsing rollout lengths, rising entropy), so we settled on 2e-5 for the final runs.

### Training infrastructure

Each harness adapter was trained using NVIDIA 2xB200 GPUs, with one GPU allocated for the vLLM server and the other allocated for the training process. We first performed some stress testing to maximize the amount of RL tasks that can be trained within a single optimization step. For that, we ‘maxed out’ the `--token-budget` argument from `AsyncGRPOTrainer` to its limit for the setup that we had. The chart below shows the evolution of the number of groups per step for each harness (in blue: Pi, green: OpenCode and orange: Hermes).

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/groups.html" width="100%" height="500" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

As the model gets better and better during training, it is able to fit more and more groups during each training step, which diversifies the signals that the model gets at each optimization step.

### Final mixture

The final data mixture consists of 750 tasks from [harness-adapters RL environments](https://huggingface.co/datasets/harness-hub/harness-adapters) and 300 tasks from [termgrade](https://huggingface.co/datasets/ai-and/termgrade-environments). In order to maximize the diversity of the sampled tasks during training, we have slightly customized the data sampling strategy, which will be explained below:

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/sampling.html" width="100%" height="430" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

Before training, the tasks are put in a fixed order that interleaves them by family (i.e., their repository for our generated RL environments. For Termgrade, we just sorted them by name since each task is considered as its own family), so every training step mixes as many repositories as possible while following the distribution of the mixture.

### safeagents in the loop

We also plug the [safeagents](https://github.com/huggingface/safeagents) tool, in order to block any misbehavior from the model during training. The final plugin (which consists of the regex and pattern rules specific to the RL tasks and test path) is hosted in [harness-hub/qwen-safeagents-plugin](https://huggingface.co/harness-hub/qwen-safeagents-plugin) and can be used out of the box using the latest version of safeagents in case anyone wants to reproduce the training setup.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/protocol.html" width="100%" height="430" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

The number of blocked tools from safeagents decreases consistently during training, which indicates that the model gets ‘safer and safer’ after each optimization step. Below is a diagram of the safeagents blocked calls per 1000 tool calls, by training step, for each harness adapter.

<p align="center">
    <img src="/blog/assets/harness-adapters-at-scale/safeagents-blocked.png" alt="safeagents blocked calls per 1,000 tool calls by training step for the pi, hermes and opencode adapters" width="100%">
</p>

## Super-adapter: merging all adapters into one

Merging models often combines their capabilities while keeping most of what each one could do. Once the three harness adapters were trained, we asked two questions. Does merging them cost accuracy? And can a single "super-adapter" be accurate and efficient under every harness?

We started by comparing the weight updates of the three adapters, layer by layer. The cosine similarity between any two of them is about 0.01–0.02: each adapter pushed the model in a nearly orthogonal direction during training. That's the best case for merging, since the updates barely interfere.

We concatenate the three LoRA adapters along the rank dimension (PEFT's `combination_type="cat"`). The merged adapter has rank 96 (3 × 32) and applies exactly w · (ΔW_pi + ΔW_hermes + ΔW_opencode). Prior work on merging ([task arithmetic](https://huggingface.co/papers/2212.04089), [TIES](https://huggingface.co/papers/2306.01708), [KnOTS](https://huggingface.co/papers/2410.19735)) finds that scaling each update below 1 works best, so we compared three weights on all 89 Terminal-Bench 2.1 tasks, under each harness, with the step-120 adapters:

- ⅓ (the average): each harness keeps only a third of its own update.
- ½: each update at half strength.
- 1 (the plain sum): every update applied in full.

| Weight | pi | hermes | opencode | All three harnesses | Tool calls per task |
|---|---:|---:|---:|---:|---:|
| base model | 39.1% | 46.0% | 44.8% | 43.3% | 49.2 |
| ⅓ (average) | 49.4% | 51.2% | 47.1% | 49.2% | 35.6 |
| **½** | 48.8% | 53.5% | 55.2% | **52.5%** | 25.9 |
| 1 (sum) | 40.2% | 62.8% | 47.1% | 50.0% | 13.4 |

The plain sum over-applies the updates. The agents become very short (13 tool calls per task, against 49 for the base model), and while Hermes peaks at 62.8%, pi falls back to the base model's level. The average dilutes each harness's own update. ½ gives the best balance, so we built the final super-adapter from the released checkpoints (pi step 160, hermes step 160, opencode step 150) at weight ½.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/super-results.html" width="100%" height="560" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

### Adapter merging evaluation results

Under each harness, on the tasks every run graded (50–51 on SWE-bench Pro, 82–85 on Terminal-Bench 2.1), one attempt per task (in parentheses, the efficiency gain with respect to the base model):

| Harness | Benchmark | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Qwen3.8-27B-\<harness\>-adapted | [Qwen3.8-27B-super-adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter) | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), thinking | [Qwen3.8-27B-super-adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter), thinking |
|---|---|---:|---:|---:|---:|---:|
| pi | SWE-bench Pro (hard 51) | 40.0% | 64.0% (−60%) | 46.0% (−73%) | 74.0% | 74.0% (−48%) |
| hermes | SWE-bench Pro (hard 51) | 29.4% | 43.1% (−56%) | 47.1% (−60%) | 45.1% | 60.8% (−36%) |
| opencode | SWE-bench Pro (hard 51) | 35.3% | 51.0% (−72%) | 45.1% (−63%) | 47.1% | 66.7% (−31%) |
| pi | Terminal-Bench 2.1 | 40.0% | 56.5% (−56%) | 48.2% (−56%) | 52.9% | 57.6% (−25%) |
| hermes | Terminal-Bench 2.1 | 44.7% | 58.8% (−70%) | 55.3% (−53%) | 58.8% | 63.5% (−19%) |
| opencode | Terminal-Bench 2.1 | 47.6% | 52.4% (−53%) | 52.4% (−68%) | 54.9% | 58.5% (−22%) |

"Thinking" is the chat template's thinking mode at its default effort (xhigh). The specialists were trained and served with thinking off.

With thinking off, the super-adapter improves on the base model on all six harness–benchmark pairs (mean pass@1 39.5% → 49.0%; 71 tasks newly solved vs 34 lost, p < 0.001).

Despite not being trained at all with thinking on, it's the strongest model we evaluated with thinking mode. Against the base model thinking at the same effort, it wins on five of six pairs and ties the sixth (mean 55.5% → 63.5%; 67 newly solved vs 37 lost, p = 0.004). The largest gains are on SWE-bench Pro: +16 points under hermes and +20 under opencode. Under pi on SWE-bench Pro, where thinking alone already reaches 74%, it ties.

Use the specialist harness adapters when thinking off. Each harness's own adapter is ahead on four of six pairs, by the most under pi (64.0% vs 46.0% on SWE-bench Pro, 56.5% vs 48.2% on Terminal-Bench). Splitting one model across three harnesses costs pi the most of its gain. If you only use one harness and keep thinking off, take its specialist.

The super adapter is also much more efficient. On the tasks both the base model and the super-adapter solve, with thinking off, it makes 53–73% fewer tool calls and generates 44–60% fewer tokens per task, with fewer tool calls on 111 of 130 tasks. With thinking on, it still cuts tool calls by 31–48% and tokens by 24–43% on SWE-bench Pro. On Terminal-Bench, tool calls drop by 19–25% and token counts stay about the same.

The ½ weight was chosen on Terminal-Bench 2.1 with the step-120 adapters, so the Terminal-Bench numbers are slightly optimistic; SWE-bench Pro played no part in the choice. Full paired results are in the [super-adapter's model card](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter).

## Final results

This section reports what happened during training, then how the adapters score on held-out benchmarks inside and outside their own harnesses.

### Training results

Training ran for a cumulative total of 598 steps, with a total of 78,114 samples trained, which consists exactly of 167M trained tokens.

During training, we see a clear signal that the model learns to solve tasks in fewer and fewer steps, with no clear sign of saturation. Despite having some clear increasing trend sometimes, training correctness seems to overall stay flat. This can be explained by the fact that the model meets new tasks during training, but the number of groups that fit in a step’s token budget more than doubles as trajectories get shorter. The accuracy gains show up on the held-out benchmarks in the section below.

On the other hand, safeagents was able to block multiple suspicious attempts. Across the three training runs, a total of 1.3M tool calls have been checked, and a total of 647 tool calls have been blocked.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/reward.html" width="100%" height="500" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/toolcalls.html" width="100%" height="500" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

### Evaluation results

This section covers the evaluation setup, the in-harness results, transfer across harnesses, and an out-of-distribution check.

#### Evaluation setup

We evaluate each adapter inside the harness it was trained in, on two agentic coding benchmarks: all 89 tasks of [Terminal-Bench 2.1](https://huggingface.co/datasets/harborframework/terminal-bench-2.1) (1 hour and 100 agent steps per task) and the 51 hardest tasks of SWE-bench Pro (2 hours and 100 agent steps). We cap the maximum number of tokens per task to be 131072. We do not tune the agent settings and use the default setup for all harnesses. For example, Pi does not keep previous reasoning turns while OpenCode and Hermes do, which mainly affects the SWE-bench Pro results due to rapid context overflow. Therefore you cannot compare the results across different harnesses (using Pi does not necessarily lead to better results than Hermes or OpenCode), but it is fair to compare results within the same harness and draw conclusions.

Every rollout runs through the same Harbor and OpenEnv stack as training, in HF sandboxes, with one attempt per task at temperature 0.8 and thinking off. Each LoRA checkpoint is merged into the base weights and served alone with vLLM on a single GPU, at the same concurrency as the base model's eval.

For all the evaluation benchmarks, we report the pass@1 as well as the number of tool calls, total turns, for all tasks and for solved tasks.

For openness, all model trajectories for final evaluations are pushed in this public bucket: [harness-hub/harness-adapters-trajectories](https://huggingface.co/buckets/harness-hub/harness-adapters-trajectories).

#### Final evaluation results

All three specialist adapters beat the base model under their own harness, on both benchmarks. On SWE-bench Pro (hard 51), pi goes from 39.2% to 64.7% (+25.5 points: 15 tasks newly solved vs 2 lost, p = 0.002), hermes from 29.4% to 43.1%, and opencode from 35.3% to 51.0%. On the full Terminal-Bench 2.1, pi gains +16.3 points (16 vs 2, p = 0.001) and hermes +14.1 (19 vs 7, p = 0.03). Opencode gains +6.0 overall, and +13.6 on the 66 tasks that played no part in choosing its checkpoint (12 vs 3, p = 0.035).

The adapters also solve tasks much more efficiently: on the tasks both they and the base model solve, they make 56–72% fewer tool calls per task on SWE-bench Pro and 53–70% fewer on Terminal-Bench 2.1. The single super-adapter beats the base model under all three harnesses too (+6 to +18 points on SWE-bench Pro, +5 to +10 on Terminal-Bench 2.1), with just as large a cut in tool calls (53–73%). It keeps less of each specialist's gain with thinking off, most under pi, but with thinking on it beats the base model thinking at the same effort on five of six harness–benchmark pairs (see the [super-adapter section](#super-adapter-merging-all-adapters-into-one)). Each model card has the full paired results.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/results.html" width="100%" height="470" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

#### Does the efficiency and performance ability transfer across harnesses?

To answer this question, we also evaluated the pi harness adapter under Hermes Agent and OpenCode.

Under OpenCode it matches OpenCode's own specialist on Terminal-Bench 2.1 (44.8% → 54.0%, against 46.4% → 52.4%). Under Hermes it keeps about two-thirds of the specialist's gain (46.3% → 56.1%, against 44.7% → 58.8%), with the same ~45% cut in tool calls in both. Therefore it seems that the newly learned capabilities are transferred across harnesses.

#### Evaluation on out-of-distribution tasks and setup

What happens if we evaluate in out-of-distribution tasks, outside the harnesses the adapters have been trained with?

We ran all five models on 300 [AutomationBench tasks](https://github.com/zapier/AutomationBench) which consists of business-automation workflows evaluations, with AutomationBench's own agent loop instead of Pi, Hermes or OpenCode. This way, we can cross check two different questions: can the model solve problems outside SWE-related tasks? What happens when the model is being put outside the harness it has been trained with?

Results show that the adapters stay close to the base model. Pi and OpenCode are slightly ahead (27.0% and 25.3% pass@1 against 23.0%), the merged adapter is level (23.7%), and Hermes is a few points behind (17.7%). The differences are within noise, except perhaps Hermes, therefore RL inside the harnesses neither transfers clearly to this very different setting nor breaks the model on it.

| Model | pass@1 (all assertions met) | vs base | Partial credit (share of assertions met) | vs base |
|---|---:|---:|---:|---:|
| base | 23.0% | | 62.2% | |
| pi 160 | 27.0% | +4.0 | 66.2% | +4.0 |
| opencode 150 | 25.3% | +2.3 | 65.2% | +3.0 |
| merged | 23.7% | +0.7 | 60.6% | −1.6 |
| hermes 160 | 17.7% | −5.3 | 56.6% | −5.6 |

## Open question: What did the model really learn to do?

We appreciate the feedback from the users and the community, initial testing and results from the community seemed overall extremely positive, and we wish to continue in this direction in the future. We also saw some surprising results, such as users reporting that the [harness adapters work really well under Qwen3.8 derivatives, such as Swift-1.5](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted-GGUF/discussions/2).

This question is an open question we thought about, post-release. After inspecting carefully the generated trajectories from the evaluation benchmarks, it turns out the model, under non-thinking mode, tends to generate some short text before proposing a tool call, as opposed to the base model which generates the tool calls barely without any text beforehand.

This phenomenon can be seen as a sort of ‘reasoning leakage’, where the model learns to ‘escape’ from its non-thinking version to its thinking version by generating some text that can sometimes look like reasoning traces (e.g.: ..wait? ..let me see, etc.). [A recent paper](https://huggingface.co/papers/2610.14270) also describes behavior from LLMs during the RL stage very well.

Does this invalidate our models? We do not think so, as the final results are still valid; under the exact same conditions (fixed token budget) our adapters do outperform the base non-thinking model (by far) and match the performance of the thinking version, while being much more token efficient. This has been further confirmed by the positive feedback from the community.

However this raises the need of a proper follow up work, where we will try to re-apply the same pipeline for pure thinking models, as well as exploring hybrid thinking / non-thinking training, to better understand the behavior of RL training on hybrid thinking models. We will share with the community our next findings (and harness adapters), once we get conclusive results.

## Closing words and future directions

Throughout this work, we demonstrated that RL training your model inside your target harness leads to more efficient and therefore more accurate final models, when evaluated under the same conditions. This observation is in line and in continuation with several previous work from Hugging Face teams, such as [multi-harness-RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) or the [early iterations of harness adapters](https://huggingface.co/blog/harness-hub/harness-adapters).

In the future, we will explore applying this recipe to more models, while better studying the training strategies for hybrid thinking / non-thinking models.

We also share to the community openly the full recipe and pipeline to reproduce everything, and invite everyone to build their own RL tasks, derived from their own codebase, or any relevant materials, and apply the same training recipe and ideas to their favorite model and harness for more efficient agentic models. We hope this blogpost will serve as a starting point for builders to build and share their own harness adapters in the future.
