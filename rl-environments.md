---
title: "Welcome RL Environments to the hub"
thumbnail: /blog/assets/datasets-filters/thumbnail.png
authors:
  - user: burtenshaw
  - user: xeophon
    guest: true
  - user: ryanmarten
    guest: true
  - user: merve
  - user: lhoestq
  - user: julien-c
---

# Welcome RL Environments to the hub

Reinforcement Learning environments give new capabilities to agentic AI systems, and they’re a great way to measure and improve performance in your agents. Therefore, Hugging Face Hub now has a special place for RL Environments.

An environment gives an agent a task, responds to its actions with observations, and scores the outcome. The resulting rewards can measure an agent's performance during evaluation or provide a learning signal during training. For an introduction to this interaction loop, see [our blogpost on environments](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide). Within the environment, the agent will perform a set of tasks that are represented as datasets. Therefore, environments can be split into broadly two parts: tasksets and runtimes. In this release, we are focusing on the tasksets.

<figure class="image text-center">
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rl-environments/rl-environments-filter.gif"
    alt="Browsing the RL Environments filter on the Hugging Face Hub"
    width="100%"
  >
</figure>

An RL environment on the Hub is a dataset repo that shows up in the new [RL Environments filter](https://huggingface.co/datasets?other=rl-environment). The **Use this dataset** button gives you the command to run it in that framework. There is no new repo type, no registry, and no sign-up. There are already environments in Harbor, Verifiers, and NVIDIA NeMo Gym.

## Stop building environment registries

Every RL paper or framework uses its own way to find environments. Custom hubs, runtime registries, independent task datasets, or a GitHub list of tasks with a custom loader. This means that many of the published environments are siloed: if you publish an environment for one framework, users of the other three can’t load it. If you want to train on an environment from another framework or a new paper, you’ll need to port it by hand.

We think this is the wrong shape. An environment is tasks, tests, containers, and a reward rule, which are data with a runtime on top. The Hub already stores data, versions it, gates it, previews it, and serves it to millions of people. It does not need a second system to hold environments. It needs a way to say "this data is an environment, and here is how you run it."

<figure class="image text-center">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 520" width="100%" role="img" aria-labelledby="environment-diagram-title environment-diagram-description">
    <title id="environment-diagram-title">From a dataset repository to an agent run</title>
    <desc id="environment-diagram-description">The Hub stores task data but the frameworks store runtime and verfier code. A framework loads the data and supplies runtime or verifier implementations when they are not included in the repository. At runtime an agent exchanges actions and observations with an environment. A verifier scores the outcome and produces rewards for evaluation or training.</desc>
    <rect x="2" y="2" width="876" height="516" rx="16" fill="#ffffff" stroke="#e5e7eb"/>
    <g font-family="Arial, sans-serif" fill="#111827" text-anchor="middle">
      <rect x="80" y="24" width="720" height="104" rx="12" fill="#fff7d6" stroke="#d4a72c"/>
      <text x="440" y="58" font-size="24" font-weight="bold">Dataset repository on the Hub</text>
      <text x="440" y="92" font-size="19">Tasks and data · Runtime and verifier files, when included</text>
      <path d="M440 132 V174 M433 164 L440 174 L447 164" fill="none" stroke="#6b7280" stroke-width="2"/>
      <text x="612" y="161" font-size="17" fill="#4b5563">Framework loads the files</text>
      <rect x="24" y="186" width="832" height="310" rx="12" fill="#f9fafb" stroke="#9ca3af" stroke-dasharray="6 5"/>
      <text x="440" y="218" font-size="19" fill="#4b5563">Execution on your machine or a supported cloud backend</text>
      <rect x="64" y="264" width="180" height="80" rx="10" fill="#dbeafe" stroke="#60a5fa"/>
      <text x="154" y="310" font-size="23" font-weight="bold">Agent</text>
      <rect x="480" y="264" width="324" height="80" rx="10" fill="#dcfce7" stroke="#4ade80"/>
      <text x="642" y="298" font-size="23" font-weight="bold">Environment</text>
      <text x="642" y="324" font-size="17">State, tools, and task execution</text>
      <path d="M250 286 H470 M460 279 L470 286 L460 293 M474 326 H254 M264 319 L254 326 L264 333" fill="none" stroke="#4b5563" stroke-width="2"/>
      <text x="360" y="273" font-size="18">Actions</text>
      <text x="360" y="354" font-size="18">Observations</text>
      <path d="M642 350 V389 M635 379 L642 389 L649 379" fill="none" stroke="#4b5563" stroke-width="2"/>
      <text x="725" y="378" font-size="17">Outcome</text>
      <rect x="538" y="398" width="208" height="60" rx="10" fill="#ede9fe" stroke="#a78bfa"/>
      <text x="642" y="434" font-size="22" font-weight="bold">Verifier</text>
      <path d="M528 427 H362 M372 420 L362 427 L372 434" fill="none" stroke="#4b5563" stroke-width="2"/>
      <text x="444" y="415" font-size="17">Reward</text>
      <text x="205" y="424" font-size="21" font-weight="bold">Evaluate or train</text>
      <text x="205" y="451" font-size="17">Score runs or update the model</text>
    </g>
  </svg>
  <figcaption>Task data lives on the Hub. Runtime configuration and verifier code can live in the repo or the framework.</figcaption>
</figure>

The frameworks keep doing what they are good at. The Hub does what it is good at, which is hosting, discovery, and versioning. Nobody has to own the catalogue. In fact, catalogues can run on other platforms too, powered by the hub.

The dataset repository hosts your environment files. The framework runs them locally or on a supported cloud backend. [Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs) can run cloud workloads, and [Hugging Face Sandboxes](https://huggingface.co/docs/huggingface_hub/main/guides/sandbox), built on Jobs, provide interactive command execution. The tags describe compatibility and generate loading commands; adding a tag does not start a job or sandbox.

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

A dataset can carry more than one framework tag. That is the point. Tags describe compatibility, and compatibility is not exclusive. Each listed framework must support the files in the repository; adding a tag does not convert them.

## Run an environment and inspect its reward

Choose the example for your framework and run it in a separate Python environment with the prerequisites listed below.

### Harbor: run a reference solution

[Harbor](https://www.harborframework.com/docs/datasets) can load task directories from a Hub repository. The oracle agent runs the task's reference solution, then the verifier scores the result. It does not call a model.

```sh
uv tool install --python 3.13 'harbor==0.21.0'
harbor run \
    --repo https://huggingface.co/datasets/harborframework/terminal-bench-2.1 \
    --dataset terminal-bench-2.1@2.1.0 \
    --include-task-name '*regex-log' \
    --agent oracle --env docker --jobs-dir results/harbor
harbor view results/harbor
```

The viewer shows the task's reward, verifier output, and logs. This checks the task and its reference solution before you try a model agent.

### Verifiers: run a model on the same task

The Harbor integration of [verifiers v1](https://www.primeintellect.ai/blog/verifiers-v1) can run the same task directories in different runtimes, such as Docker. It also supports different harnesses, including a minimal bash harness.

```sh
uvx --python 3.13 --from 'verifiers[harbor]' eval harbor \
    --env.taskset.repo https://huggingface.co/datasets/harborframework/terminal-bench-2.1 \
    --env.taskset.dataset terminal-bench-2.1@2.1.0 \
    --env.taskset.tasks '["regex-log"]' \
    --env.agent.runtime.type docker \
    --env.agent.harness.id bash \
    --model "$MODEL" \
    --client.base-url "$LLM_URL"
```
Here repo is the full Hugging Face Git URL, while dataset is the name and version in that repo's registry.json. This loader uses Harbor's registry conventions, so a bare Hub repo ID cannot replace both values.

### OpenEnv: run an agent and inspect its reward

[OpenEnv's Harbor integration](https://huggingface.co/docs/openenv/main/environments/harbor) can run the same task directories with an agent such as OpenCode and return the verifier's reward alongside the agent's trace. 

```sh
pip install "openenv[harbor]==0.7.0"

openenv harbor rollout \
    --llm-url "$LLM_URL" \
    --model "$MODEL" \
    --dataset harborframework/terminal-bench-2.1 \
    --task-index 0 \
    --harness opencode \
    --sandbox docker \
    --out rollout.json
```

The command downloads the dataset's `tasks/` directories, runs one task in Docker, and writes the result. The default connection uses a temporary Gradio tunnel so the sandboxed agent can reach OpenEnv's model proxy. Read the verifier result and the number of model calls:

```py
import json
from pathlib import Path

result = json.loads(Path("rollout.json").read_text())[0]
print("Reward:", result["reward"])
print("Model calls:", result["n_turns"])
print("Error:", result["error"])
```

A reward of `None` means no verifier reward was produced; inspect `error` before interpreting the run as a model failure. This path expects Harbor task directories.

### NeMo Gym: generate responses and inspect rewards

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) supports evaluation and RL training: its environments collect trajectories and compute rewards, while a training framework updates model weights. For example, the [Structured Outputs dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs) pairs prompts with JSON schemas. Its verifier rewards schema adherence; it does not check whether the generated content is factually correct.

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

<figure class="image text-center">
  <iframe
    src="https://fineenvs-rl-explorer.hf.space/"
    title="RL Environment Explorer"
    width="100%"
    height="700"
    frameborder="0"
    loading="lazy"
  ></iframe>
</figure>

The bigger goal is for framework tagging to be automatic everywhere. OpenEnv already does it on upload. If you maintain Harbor, Verifiers, Nemo Gym, or any other environment framework, add the tags in your push path. It is a few lines, and every environment your users publish becomes visible to everyone else.

If you train agents, go browse the [filter](https://huggingface.co/datasets?other=rl-environment). If you build environments, publish and tag them, whether they cover coding, tool use, games, robotics, or another task. Include the files, a working run command, and the rule that produces the reward so others can use them. If your framework is missing, contribute it to the [list of supported libraries](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/dataset-libraries.ts).
