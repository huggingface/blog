---
title: "Welcome RL Environments to the hub"
thumbnail: /blog/assets/datasets-filters/thumbnail.png
authors:
  - user: burtenshaw
---

# Welcome RL Environments to the hub

Reinforcement Learning environments give new capabilities to agentic AI systems, and they’re a great way to measure and improve performance in your agents. Therefore, the hugging face hub now has a special place for RL Environments.

\<screenshot\>

An RL environment on the Hub is a dataset repo that shows up in the new [RL Environments filter](https://huggingface.co/datasets?other=rl-environment). The **Use this dataset** button gives you the command to run it in that framework. There is no new repo type, no registry, and no sign-up. There are already environments in Harbor, Verifiers, and Nemo Gym.

## Stop building environment registries

Every RL paper or framework ships its own way to find environments. A hub here, a registry there, a GitHub list of tasks with a custom loader. Each one is a small walled garden. If you publish an environment for one framework, users of the other three can’t load it. If you want to train on an environment from another framework or a new paper, you’ll need to port it by hand.

We think this is the wrong shape. An environment is tasks, tests, containers, and a reward rule. That is just data with a runtime attached. The Hub already stores data, versions it, gates it, previews it, and serves it to millions of people. It does not need a second system to hold environments. It needs a way to say "this data is an environment, and here is how you run it."

The frameworks keep doing what they are good at. The Hub does what it is good at, which is hosting, discovery, and versioning. Nobody has to own the catalogue. In fact, catalogues can run on other platforms too, powered by the hub.

Rl Environments are not a new repository type. An environment is a dataset repo, so it gets everything a dataset repo gets: gating, versioning, the viewer, discussions, and PRs.

The Hub does not have to run your environment, but you can as jobs, if you need. The tags just describe compatibility and generate loading commands. Execution stays in the framework, on your hardware or your sandbox provider.

## What shipped

**The RL Environments filter.** Go to [huggingface.co/datasets?other=rl-environment](https://huggingface.co/datasets?other=rl-environment). Every dataset with the `rl-environment` tag appears there, whatever framework it works with.

**Framework tags.** Four environment frameworks are registered as dataset libraries:

| Tag | Framework |
| :---- | :---- |
| `harbor` | [Harbor](https://github.com/harbor-framework/harbor) |
| `verifiers` | [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) |
| `openenv` | [OpenEnv](https://github.com/huggingface/OpenEnv) |
| `nemo-gym` | [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) |

Each framework tag puts the framework's icon on the dataset page and adds a generated snippet to **Use this dataset**.

A dataset can carry more than one framework tag. That is the point. Tags describe compatibility, and compatibility is not exclusive.

Take a dataset of Harbor task directories. Harbor runs it directly:

```sh
harbor run \
    --dataset hf://datasets/harborframework/terminal-bench-2.1 \
    --agent oracle
```

Verifiers can read the same task directories, so the same repo also gets a Verifiers snippet:

```py
import verifiers as vf

taskset = vf.HarborTaskset(
    config=vf.HarborTasksetConfig(
        dataset="hf://datasets/<org>/<dataset>",
        split="train",
    )py
)

env = vf.Env(taskset=taskset, harness=vf.OpenCode())
```

OpenEnv loads an environment straight from the repo:

```py
from openenv import AutoEnv

env = AutoEnv.from_env("<org>/<dataset>", trust_remote_code=False)
```

And NeMo Gym pulls the data down and evaluates against it:

```
[TODO]
```

The best part is that this gives one repo and one discussion tab where people report broken tasks from all major frameworks. So when an author fixes a bad test, every framework gets the fix on the next pull.

## Tag your environment

Open your dataset card and add this to the YAML header:

```
---
pretty_name: Terminal-Bench 2.0
tags:
- rl-environment
- harbor
- verifiers
---
```

That is the whole integration. Keep `rl-environment`, then list every framework that can load your files. If your environment works with a framework we have not registered yet, open a PR to the [list of supported libraries](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/dataset-libraries.ts).

The [docs](https://huggingface.co/docs/hub/datasets-cards#declare-an-rl-environment-dataset) have the full reference.

## Already on the Hub

We opened PRs to tag some of the environments people already train on. If you maintain one of these, merge the PR and your environment shows up in the filter.

**Harbor**

* [BeyondSWE](https://huggingface.co/datasets/AweAI-Team/BeyondSWE-harbor/discussions/1): the BeyondSWE benchmark as Harbor task directories, one folder per instance.  
* [Terminal-Lego](https://huggingface.co/datasets/Lego-X/Terminal-Lego-15k/discussions/1): Terminal-Bench-style tasks built from real StackOverflow issues, kept only after Docker round-trip verification.  
* [Harbor-Mix](https://huggingface.co/datasets/harborframework/harbor-mix/discussions/2): 100 hard agentic tasks picked from the Harbor adapters pool, cheaper to run than a full multi-benchmark sweep.  
* [NatureBench](https://huggingface.co/datasets/FrontisAI/NatureBench-Harbor/discussions/2): the 90 NatureBench tasks prebuilt for Harbor.

**Verifiers**

* [Reverse-Text-RL](https://huggingface.co/datasets/PrimeIntellect/Reverse-Text-RL/discussions/2): the small reversal task prime-rl uses in CI to debug RL training.  
* [Multi-SWE-RL-Verified](https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL-Verified/discussions/2): 2,232 of 4,703 Multi-SWE-RL rows that pass gold-patch validation across C, Go, Java, JavaScript, Rust, and TypeScript.  
* [R2E-Gym-Subset-Verified](https://huggingface.co/datasets/PrimeIntellect/R2E-Gym-Subset-Verified/discussions/1): a verified R2E-Gym subset.  
* [Scale-SWE-Verified](https://huggingface.co/datasets/PrimeIntellect/Scale-SWE-Verified/discussions/2): 17,202 of 20,181 Python issue-resolving tasks that give a clean reward signal end to end.

**NeMo Gym**

* [Workplace Assistant](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant/discussions/3): a multi-step tool-use sandbox with five databases, 26 tools, and 690 business tasks.  
* [Structured Outputs](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs/discussions/2): instruction following with structured outputs.  
* [CFBench](https://huggingface.co/datasets/nvidia/Nemotron-RL-CFBench-v1/discussions/2): multilingual constraint following.  
* [SysBench](https://huggingface.co/datasets/nvidia/Nemotron-RL-SysBench-v1/discussions/1): multi-turn system message following.

## What comes next

The first version generates one `default` snippet per framework. Per-config snippets are next, so a repo with several task sets can show the right command for each. After that we will look at structural detection for frameworks with strict layouts. It would also be really cool build custom task UIs, we’ve been experimenting with this here:

\<harbor ui embedded space\>

The bigger goal is for framework tagging to be automatic everywhere. OpenEnv already does it on upload. If you maintain Harbor, Verifiers, Nemo Gym, or any other environment framework, add the tags in your push path. It is a few lines, and every environment your users publish becomes visible to everyone else.

If you train agents, go browse the [filter](https://huggingface.co/datasets?other=rl-environment). If you build environments, tag them. If you think the tag list is missing a framework, open the PR.