---
title: "We made an AI lab of agents to train models"
thumbnail: /blog/assets/multi-agent/thumbnail.png 
authors:
- user: burtenshaw
---

# We made an AI lab of agents to train models

tl;dr: a team of agents research papers, propose hypotheses, run parallel experiments, review runs, make improvements, and repeat.

Most ML experiment infrastructure solves the execution problem; i.e. how to get code onto GPUs. An in many cases, code agents (opencode, claude, codex, etc.) have solved this problem well. If you're curious about running a single agent for ML experiments, check out [this blog post](https://huggingface.co/blog/hf-skills-training).

The harder problem kicks off after the run finishes; i.e. what happened, why did it fail, and what should we do next?

This post tackles that problem by walking through a system to define a research lab. We focus on the retrospective problem as the main concern, and structure the stack with three layers: 
- a **control plane** which we implemented in Codex subagents, Claude Code subagents, and GasTown.
- an **execution plane** which we implemented in Hugging Face Jobs with managed H200 GPU based on `uv` scripts.
- an **observability layer** which we implemented in Trackio to turn events and metrics into an open data layer and a dashboard.

In practice, you could implement this in a number of different ways, what's important is an open infrastructure layer that can be used by any agent (like the hub).

The code is open at [burtenshaw/autolab-gastown](https://github.com/burtenshaw/autolab-gastown) where we optimized a language model training benchmark based on Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) project. Everything below comes from real waves of experiments (check out `2026-03-28-wave2` and `2026-03-29-wave3`).

## What the experiment actually is

`autoresearch` is a short, fixed-budget language model training run. The target metric is `val_bpb` (validation bits-per-byte). The current best score lives on a hosted Autolab service, and the goal is to beat it by making small, isolated changes to `train.py`.

Each experiment follows one strict rule: **one hypothesis, one edit, one run**. You refresh from the hosted master, change exactly one thing in `train.py`, launch a managed H200 job, and submit the diff only if the result improves on the baseline.

For education's sake, this is what the agents need to run (in reality, agents run these scripts):

```bash
git checkout <branch>
git merge origin/main

# Launch the run
hf jobs uv run scripts/hf_job.py launch --mode experiment ...

# Submit only if it wins
uv run scripts/submit_patch.py --comment "warmdown ratio 0.925, val_bpb 0.958973"
```

That sequence is the core experiment loop for one hypothesis. The main questions are; what improvements do we make? how do we run experiments in parallel? how do we improve our understanding of the results?

We ran this experiment for 3 hours and started 126 runs in parallel. The figure below shows the jobs running in the experiment and the results of the runs.

![job overview](https://huggingface.co/buckets/burtenshaw/autolab-blog-assets/resolve/autolab-rig/images/gastown_wave2_running_jobs.png)

The agent made a series of improvements to the experiment, each time running a new wave of experiments and improving the results. 

After the wave we can explore the results and understand what happened through the trackio dashboard:

## The control plane in Gastown

We first implemented the control plane in Gastown, a multi-agent workspace manager. This was a pretty wild experience which mostly taught us a lot about multi-agent systems and how to manage them. I would not reccomend that everybody go out and try gastown for their next project, but it was certainly a learning experience.

If you starting out on this journey, you might choose to use Codex subagents or Claude Code instead. They're simpler with less sharp edges and moving parts. Nonetheless, Gastown is a useful tool to to unpack the problem of running experiments in parallel.

In this setup Gastown has four primitives that matter:

| Primitive | Gastown Terms |  Application in this experiment |
|--|--|--|
| Batch of work | **Convoy** |  A batch of experiments dispatched together |
| Unit of work | **Bead** | One tracked hypothesis through its entire lifecycle |
| Worker | **Polecat** | An isolated worker agent with its own git worktree that runs the experiment |
| Dispatch | **Sling** | The dispatch event that assigns work to a worker |

Gastown has distinctive terminology and it's unlikely that the field is going to adopt these terms. But most other tools are not exposing the primitives of multi-agent systems, so they're useful for exploring the problem.

Here's how you can inspect the experiment. For example, we can get an overview of the convoys:

```bash
# What was in the batch?
gt convoy status
```
![convoy](https://huggingface.co/buckets/burtenshaw/autolab-blog-assets/resolve/autolab-rig/images/convoys.png)

We can deep dive back into a single item of work by showing the details:

```
bd show --long --id au-2rf
```

![bead](https://huggingface.co/buckets/burtenshaw/autolab-blog-assets/resolve/autolab-rig/images/terminal_vid.gif)

Or we can see the atomic state of a single worker:

```
# Who ran it?
gt polecat status autolab/turquoise
```

![polecat](https://huggingface.co/buckets/burtenshaw/autolab-blog-assets/resolve/autolab-rig/images/polecat_status.png)

Each bead carries the full context: the hypothesis text, the parent master hash it branched from, the polecat that executed it and its context, the HF Job ID, link to the trackio dashboard, and the final metric. That chain is what makes the process legible after the fact.

We can also aggregate this state and create a summary of the experiment. For example, this is a gantt chart of the workers running in the experiment:

![gantt](https://huggingface.co/buckets/burtenshaw/autolab-blog-assets/resolve/autolab-rig/images/gastown_wave2_running_jobs.png)

### Creating a wave

A typical wave starts with the planner creating beads from a mix of exploit brackets (tightening around known winners) and paper-derived exploration (new ideas from recent research):

In practice, each example is created by the planner and dispatched to a worker like this:

```bash
# Exploit bracket: tighten around a known warmdown winner
bd create --type task --priority 1 \
  --labels autolab:experiment,autolab:scheduler \
  --title "scheduler: warmdown ratio 0.925 hub-master bracket" \
  --description "$(cat <<'EOF'
Hypothesis:
- The strong warmdown win at 0.90 suggests the optimum may sit
  slightly higher on current hub master.
Single change:
- WARMDOWN_RATIO = 0.925
Parent:
- current hub master 765a36b0700b
EOF
)"
```

## Codex subagents

Let's leave Gastown land for now and switch to Codex subagents, a lighter control plane.

We'll use the comparable Codex-native variant keeps the same experimental structure, but it replaces Gastown's explicit work objects with project-scoped custom agents and markdown state inside the repo.

The role mapping is direct:

| Primitive | Codex subagents |
|--|--|
| Batch of work | `research/campaigns/*.md` campaign note |
| Unit of work | `research/experiments/*.md` experiment note |
| – | `planner` subagent |
| Worker | `experiment_worker` subagent |
| – | `reviewer` or `memory_keeper` subagent |
| Dispatch | Parent Codex session spawning a worker |

The user starts a parent Codex session in the repo which uses the checked-in `.codex/config.toml`, and delegates to custom subagents that
already know the autoresearch task based on instructions in the repo.

### Running the same experiment with Codex subagents

Here's the same kind of warmdown-ratio follow-up as a Codex-native session. Inside that parent session, the flow is lighter-weight but structurally similar:

```text
Read AGENTS.md, research/notes.md, research/do-not-repeat.md,
research/reference/master.seed.json, and research/reference/dag.seed.json.

Spawn the `planner` subagent and ask for up to 3 fresh scheduler experiments for
the current master.

Create `research/campaigns/scheduler-warmdown.md` from
`codex/templates/campaign.md`.

Create `research/experiments/warmdown-0925.md` from
`codex/templates/experiment-task.md` and assign:
- Hypothesis: WARMDOWN_RATIO = 0.925
- Parent master hash: 765a36b0700b
- Single variable: warmdown ratio only

Spawn one `experiment_worker` for GPU 0 and tell it to:
- refresh from current master
- edit `train.py` only
- run `CUDA_VISIBLE_DEVICES=0 ./run-local.sh /tmp/autolab-run.log`
- parse `python3 scripts/parse_metric.py /tmp/autolab-run.log`
- submit only if the local `val_bpb` beats master
- record the result in `research/notes.md` and the experiment note
```

That gives you a planner/worker split as Gastown, but without requiring a
separate rig, convoy state, or named worktrees.

In practice, codex was able to start off many runs and improved scores. However, it had nowhere near the autonomy of the Gastown approach nor was the process as legible after the fact. 

I expect that many of Gas Town's features may one day appear in other tools, but for now the experiment was a useful proof of concept.

## Claude Code

Let's switch to Claude Code which uses a repo-local control plane with worktree isolation ans sits between the other two approaches. Like the Codex path, it keeps the durable notebook in the repo. Unlike the Codex path, the checked-in `experiment-worker` is designed to run in the background and in its own worktree, so parallel workers get stronger checkout isolation. 

The role mapping is also direct:

| Primitive | Claude Code |
|--|--|
| Batch of work | `research/campaigns/*.md` campaign note |
| Unit of work | `research/experiments/*.md` experiment note |
| – | `planner` subagent |
| Worker | `experiment-worker` subagent |
| – | `reviewer` or `memory-keeper` subagent |
| Dispatch | Parent Claude session spawning a worker |

Once again, the user starts Claude Code in the repo which loads `CLAUDE.md`, checks the project agents under `.claude/agents/`, and then delegates to custom subagents that already know the autoresearch task based on instructions in the repo.

### Running the same experiment with Claude Code

Inside that parent session, the flow looks like the prompt below:

```text
Read CLAUDE.md, AGENTS.md, research/notes.md, research/do-not-repeat.md,
research/reference/master.seed.json, and research/reference/dag.seed.json.

Run `/agents` and confirm the checked-in project agents are loaded.

Ask the `planner` subagent for up to 3 fresh scheduler experiments for the
current master.

Create `research/campaigns/scheduler-warmdown.md` from
`claude/templates/campaign.md`.

Create `research/experiments/warmdown-0925.md` from
`claude/templates/experiment-task.md` and assign:
- Hypothesis: WARMDOWN_RATIO = 0.925
- Parent master hash: 765a36b0700b
- Single variable: warmdown ratio only
- Log path: research/live/warmdown-0925.log

Spawn one background `experiment-worker` for GPU 0 and tell it to:
- refresh from current master
- edit `train.py` only
- run `CUDA_VISIBLE_DEVICES=0 ./run-local.sh research/live/warmdown-0925.log`
- parse `python3 scripts/parse_metric.py research/live/warmdown-0925.log`
- leave a short result summary for `memory-keeper`

After the worker finishes, use `memory-keeper` in the main checkout to update
`research/notes.md`, the experiment note, and `research/do-not-repeat.md`.
```

That gives you a single parent session like the Codex flow, but with a more explicit worktree story for parallel workers.

## The Hub as Agent Infrastructure

We use the Hub as an infrastructure layer for the agents:
- buckets are our storage layer
- jobs are our execution layer
- trackio is our observability layer
- datasets are our data layer
- papers are our research layer

Because the hub is extremely open, agents can use it to store their state and execute their work. But they are not bound by its API surface. They can take the primitives and build their own tools on top of it.

### HF Jobs: the execution plane

Once the control plane decides what to run, Hugging Face Jobs handles the execution as a self-contained UV script from the current workspace, bundles it with the experiment's `train.py`, and submits it as a managed job:

Under the hood, this calls `hf jobs uv run` with explicit hardware, timeout, secrets, and a mounted cache bucket:

```bash
hf jobs uv run \
  --flavor h200 \
  --timeout 90m \
  --namespace burtenshaw \
  --detach \
  --label autolab \
  --label bead=au-2rf \
  --label hypothesis=warmdown_ratio_0_925 \
  --secrets HF_TOKEN \
  --volume "hf://buckets/burtenshaw/autolab-cache:/autolab-home/.cache/autoresearch" \
  .runtime/autolab-hf-job.py
```

Labels tie each job back to the bead that created it. Each run takes roughly 5 minutes of actual training (~300 seconds on H200), and the final summary block contains everything needed to evaluate the result:

```
val_bpb:          0.958973
training_seconds: 300.0
peak_vram_mb:     33609.6
mfu_percent:      46.79
total_tokens_M:   324.8
num_steps:        2478
```

The shared HF bucket means you only pay the data bootstrap cost once. After the first `--mode prepare` job primes the cache, every subsequent experiment reuses it.

The agents can inspect the jobs and stream the logs:

```bash
# Check a specific job
hf jobs inspect 69c7b085bf20ec90acee3a4f

# Stream logs from a running job
hf jobs logs 69c7b085bf20ec90acee3a4f

# List all autolab-tagged jobs
hf jobs ps --namespace burtenshaw --filter tag:autolab
```

### Trackio observability

Trackio turns the control-plane events and execution metadata into something you can browse after the wave is over. 

![alerts](https://huggingface.co/buckets/burtenshaw/autolab-blog-assets/resolve/autolab-rig/images/alerts.png)

The reporter script syncs HF Jobs status, bead state, and metrics into a local Trackio project:

```bash
# Sync a specific wave
uv run scripts/trackio_reporter.py sync \
  --project autolab \
  --wave-id 2026-03-28-wave2

# Generate the deck summary
uv run scripts/trackio_reporter.py deck-summary \
  --wave-id 2026-03-28-wave2

# Export investigation assets
uv run scripts/trackio_reporter.py deck-assets \
  --wave-id 2026-03-28-wave2
```

This produces three things:

1. **A wave report** — master baseline, hypothesis themes, win/loss counts, and the ranked leaderboard
2. **A timeline table** — every control-plane event in chronological order (bead created, polecat assigned, job launched, job completed, result recorded)
3. **A job board** — every HF Job with its bead, status, hypothesis, and final metric

You can also run the live dashboard:

```bash
trackio show --project autolab
```

The dashboard surfaces anomalies that matter during active waves: duplicate jobs burning the same hypothesis, stale beads still marked as running, prepare jobs tied to closed beads. But the more important use case is after the wave finishes — when someone who wasn't watching the system live needs to understand what happened.

The final step pushes the Trackio project to a Hugging Face Space, which becomes the durable retrospective interface:

```bash
trackio sync \
  --project autolab \
  --space-id burtenshaw/autolab-trackio
```

## Try it out

The repo is at [burtenshaw/autolab-gastown](https://github.com/burtenshaw/autolab-gastown). The direct operator path (no Gastown required) is:

```bash
git clone https://github.com/burtenshaw/autolab-gastown.git
cd autolab-gastown
uv sync
cp .autolab.credentials.example ~/.autolab/credentials
# Edit credentials with your Autolab endpoint and HF namespace
hf auth login
```

From there you can choose whichever control plane fits the wave we would suggest trying Open Code, Claude Code, or codex with guide in docs.

<!-- TODO: add links to the guides -->
