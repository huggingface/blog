---
title: "How we rebuilt the hf CLI for AI Agents"
thumbnail: /blog/assets/hf-cli-for-agents/thumbnail.png
authors:
- user: celinah
- user: Wauplin
---

# How we rebuilt the hf CLI for AI Agents

`hf` is the official command-line entrypoint to the Hugging Face Hub. Anything you can do on the Hub from the Python SDK, you can do from your terminal: download and upload models, datasets and Spaces; create and manage repos, branches, tags and pull requests; run Jobs on HF infrastructure; manage Buckets, Collections, webhooks and Inference Endpoints.

The `hf` CLI has been primarily built for our users over the years. But it's now increasingly used by **coding agents**: Claude Code, Codex, Cursor and more. So we rebuilt it to make it work for both audiences at once. This blog post summarizes what we did, and how we benchmarked it.

## AI agent traffic on the Hub

We started tracking agent usage of the Hub in April 2026. The `hf` CLI (and the `huggingface_hub` Python SDK it's built on) detects when a coding agent is driving it by reading the environment variables agents set: `CLAUDECODE`/`CLAUDE_CODE` for Claude Code, `CODEX_SANDBOX` for Codex, plus Cursor, Gemini, Pi, and the universal `AI_AGENT`. That single signal does two jobs: it shapes the CLI's output (more on that below) and it tags each Hub request with an `agent/<name>` user-agent, so we can attribute traffic to the agent driving it. The two largest by distinct users are **Claude Code and Codex**, well ahead of everything else, and they're the two agents we benchmark later in this article.

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-users.png" alt="Distinct users of the Hugging Face Hub by coding agent since April 2026. Claude Code leads with 39.5k users and 48.6M requests, then Codex with 34.8k users and 36.4M requests, followed by antigravity, cursor-cli, openclaw, cursor, gemini and pi." width="100%"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-users-dark.png" alt="Distinct users of the Hugging Face Hub by coding agent since April 2026. Claude Code leads with 39.5k users and 48.6M requests, then Codex with 34.8k users and 36.4M requests, followed by antigravity, cursor-cli, openclaw, cursor, gemini and pi." width="100%"/>
</div>

The bars count distinct users per agent; request volume is the sub-label. Claude Code alone is ~40k users and nearly 49M requests, with Codex close behind. These are early numbers (we only began attributing agent traffic in April 2026), but the scale is already significant, and we expect it to keep growing as coding agents become a standard way to work with the Hub.

## Built for humans and agents

The `hf` CLI serves two users from the same commands: humans and coding agents, and they want opposite things. A human wants rich terminal output: ANSI color, padded tables truncated to fit the screen, a green `✓` on success, `✔` for booleans, progress bars, prose hints. An agent wants
the inverse: no ANSI, nothing truncated, every value in full since an agent can handle far denser output than a human, kept compact and structured to stay light on tokens. It also can't answer a CLI prompt and will happily re-run a command after a timeout. The rest of this section is how `hf` gives each side what it needs.

### One command, multiple renderings

Because `hf` detects when an agent is driving it (the same signal behind the traffic numbers above), it renders the **same command** two ways, so an agent gets agent-shaped output without passing a flag:

```text
# human (default in a terminal): aligned table, truncated to fit, with a hint
> hf models ls --author Qwen --sort downloads --limit 3
ID                       CREATED_AT DOWNLOADS LIBRARY_NAME LIKES PIPELINE_TAG    PRIVATE TAGS
------------------------ ---------- --------- ------------ ----- --------------- ------- -------------------------
Qwen/Qwen3-0.6B          2025-04-27  21156913 transformers  1285 text-generation         transformers, safetens...
Qwen/Qwen2.5-1.5B-Ins... 2024-09-17  15143953 transformers   725 text-generation         transformers, safetens...
Qwen/Qwen3-4B            2025-04-27  14808352 transformers   625 text-generation         transformers, safetens...
Hint: Use `--no-truncate` or `--format json` to display full values.

# agent (auto-detected): TSV, full ids + ISO timestamps + every tag, nothing truncated
$ hf models ls --author Qwen --sort downloads --limit 3
id      created_at      downloads       library_name    likes   pipeline_tag    private tags
Qwen/Qwen3-0.6B 2025-04-27T03:40:08+00:00       21156913        transformers    1285    text-generation False   ['transformers', 'safetensors', 'qwen3', 'text-generation', 'conversational', 'arxiv:2505.09388', 'base_model:Qwen/Qwen3-0.6B-Base', 'base_model:finetune:Qwen/Qwen3-0.6B-Base', 'license:apache-2.0', 'text-generation-inference', 'endpoints_compatible', 'deploy:azure', 'region:us']
Qwen/Qwen2.5-1.5B-Instruct      2024-09-17T14:10:29+00:00       15143953        transformers    725     text-generation False['transformers', 'safetensors', 'qwen2', 'text-generation', 'chat', 'conversational', 'en', 'arxiv:2407.10671', 'base_model:Qwen/Qwen2.5-1.5B', 'base_model:finetune:Qwen/Qwen2.5-1.5B', 'license:apache-2.0', 'text-generation-inference', 'endpoints_compatible', 'deploy:azure', 'region:us']
Qwen/Qwen3-4B   2025-04-27T03:41:29+00:00       14808352        transformers    625     text-generation False   ['transformers', 'safetensors', 'text-generation', 'arxiv:2309.00071', 'arxiv:2505.09388', 'base_model:Qwen/Qwen3-4B-Base', 'base_model:finetune:Qwen/Qwen3-4B-Base', 'license:apache-2.0', 'endpoints_compatible', 'deploy:azure', 'region:us']
```

A **human** gets an aligned table, truncated to fit the terminal, plus a hint on how to see more, with color cues for status (a green `✓` on success, red on error). An **agent** gets the complete record as TSV: full repo ids, full ISO timestamps, every tag, no ANSI codes, nothing truncated, clean to parse and light on tokens.

In practice, we've implemented logging methods like `.table(...)`, `.result(...)`, `.json()`, etc., which take raw data as input and handle the formatting. In addition to human and agent modes, we've introduced `--json` and `--quiet` options to make it easier to pipe commands together. The default mode is automatically chosen based on context, but users can always force the format of their choice with `--format human | agent | json | quiet`.

### Next-command hints

CLI commands rarely run in isolation: one step usually implies the next (`git add`, then `git commit`). Many `hf` commands now end with a **hint**: the exact next command to run, pre-filled with the IDs you just used, so a user or agent can chain straight to the next step instead of working it out from scratch. Start a Job in the background and it points you to its logs; create a Space and it points you to its boot status:

```text
$ hf jobs run --detach python:3.12 python train.py
✓ Job started
  id: 6f3a1c2e9b
  url: https://huggingface.co/jobs/celinah/6f3a1c2e9b
Hint: Use `hf jobs logs 6f3a1c2e9b` to fetch the logs.
```

For a human that's a convenience. For an agent it's a rail: the next action is named, parameterized with the right ids, and ready to run, so it takes fewer steps working out what to do. Errors behave the same way, naming the fix instead of just failing:

```
Error: Not logged in. Run `hf auth login` first.
```

Hints, warnings and errors all go to stderr while data goes to stdout, so none of this guidance pollutes the output the agent is parsing.

### Non-blocking and safe to retry

`hf` never sits on an interactive prompt waiting for a key an agent can't press. A destructive command still asks a human to confirm, but in agent mode it *fails fast* with the fix in the message (`Use --yes to skip confirmation.`), and `-y`/`--yes` skips it. And because agents retry on timeouts and lost context, operations are built to be safe to repeat: `hf repos create --exist-ok` is a no-op if the repo already exists, and re-running an upload re-commits cleanly. Separately, the commands that move real data take a `--dry-run` that shows exactly what they'll transfer before they run, which proves handy for a human and an agent alike, since neither has to commit to a long download or sync blind:

```text
# agent mode: a destructive command without --yes refuses, with the fix in the message
$ hf repos delete my-org/old-model
Error: You are about to permanently delete model 'my-org/old-model'. Proceed? Use --yes to skip confirmation.

# commands that move data take --dry-run to preview the transfer first
$ hf download deepseek-ai/DeepSeek-V4-Pro config.json --dry-run
[dry-run] Will download 1 files (out of 1) totalling 1.8K.
file         size
config.json  1.8K
```

### Discoverable, predictable commands

`hf` is built to be probed: run `hf` to see the resource groups, run `--help` on the one you need, and every `--help` ends with real, copy-pasteable examples (which an agent matches against far faster than it parses a description):

```
$ hf models ls --help
...
Examples
  $ hf models ls --sort downloads --limit 10
  $ hf models ls --search "qwen" --author Qwen
  $ hf models ls Qwen/Qwen3-4B --tree
```

The command tree is consistent, **resource + verb** with the obvious aliases (`hf models ls`, `hf repos create`, `hf jobs ps`, `hf collections delete`; `list`/`ls`, `remove`/`rm`), so once an agent learns one command it can guess the rest. And the output composes: `-q` prints one id per line to pipe into the next command, `--json` gives you something to hand to [`jq`](https://jqlang.org/).

```text
$ hf models ls --author Qwen -q | head -3
Qwen/Qwen3-0.6B
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen3-4B
```

## Benchmarking the hf CLI for Coding Agents

To find out whether the `hf` CLI is really more efficient for agents, we measured it. We built a small evaluation harness and ran the same set of Hub tasks through each way of driving the Hub, many times over, grading every run against the live Hub.

### The setup

We defined **18 non-trivial Hub tasks**. Not "download a file", but the kind of thing you'd actually ask for: aggregate a trending org's models, inspect a repo's files and their sizes, upload a folder with include/exclude rules, delete files, copy files across repos, open a PR that adds a license, create a repo with a branch and a tag, sync and prune a bucket, build a collection. Each task goes to a fresh coding agent with exactly **one** way to talk to the Hub:

- the `hf` CLI, or
- **curl / the Python SDK**: no `hf` CLI at all, so the agent falls back to `curl` against the REST API or the `huggingface_hub` Python library.

We run the `hf` CLI in two configurations, with and without its skill - a generated command reference we come back to in [its own section](#the-hf-cli-skill). But the headline comparison below is simply **`hf` CLI vs curl / the SDK**; the skill's incremental effect is small enough that we break it out on its own rather than crowd it into the main results.

The config is deliberately clean: a fresh instance per run, no custom MCP servers, no `CLAUDE.md` or `AGENTS.md`, nothing in context to nudge behavior. The task and the tool go into a single prompt, and the agent finishes with a `TASK_COMPLETE` or `TASK_FAILED` marker, but we don't trust that marker (an agent will report success on work that never landed), so we grade every run independently by **re-querying the live Hub**: did the branch really get created, is the file actually gone, does the bucket exist? Each task/tool combination is run **10 times**, since coding agents are non-deterministic, about **520 runs per agent** (18 tasks × 3 tools × 10 reps, minus a cap on one billable Jobs task) and ~1,000 graded runs in total. We ran the whole thing twice, on the two most popular coding agents (**Claude Code** with Sonnet 4.6 and **OpenAI Codex** with GPT-5.5).

### The results

The ranking came out **identical on both agents**:


| agent                        | tool        | mean score | token tax | false "done" |
| ---------------------------- | ----------- | ---------- | ---------------- | ------------ |
| **Claude Code (Sonnet 4.6)** | `hf` CLI    | **0.94**   | baseline         | **2 / 163**  |
|                              | curl / Python SDK | 0.84       | **1.6× tokens**  | 11 / 163     |
| **Codex (GPT-5.5)**          | `hf` CLI    | **0.93**   | baseline         | **3 / 163**  |
|                              | curl / Python SDK | 0.92       | **1.8× tokens**  | 10 / 163     |


*(false "done" = the agent reported success on the 17 solvable tasks but the Hub said otherwise.
The `hf` CLI rows are the CLI with its skill installed; what the skill adds on top of the bare CLI
(chiefly fewer tool calls) is broken out in [the skill section](#the-hf-cli-skill) below. Every run's
full transcript is published [in this bucket](https://huggingface.co/buckets/celinah/hf-cli-agent-benchmark).)*

Two pictures carry the result. First, **task success on Sonnet**, the agent where curl and the SDK struggle most:

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-success.png" alt="Task success on Claude Code with Sonnet 4.6: hf CLI 94%, curl / Python SDK 84%." width="100%"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-success-dark.png" alt="Task success on Claude Code with Sonnet 4.6: hf CLI 94%, curl / Python SDK 84%." width="100%"/>
</div>

Without the CLI, curl and the SDK trail by ten points, because on Sonnet they simply can't finish parts of the job (the writes, mostly), while the `hf` CLI clears them. Second, and this is the one to look at, **their token tax on GPT-5.5**, broken down per task. Each bar is the curl/SDK tokens divided by the CLI's on the same task, so `2.4×` means they burned 2.4 times as many tokens to do the same thing:

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-tokens.png" alt="Per-task token ratio of curl/Python SDK divided by the hf CLI on GPT-5.5, sorted high to low. Multi-step tasks cost curl/Python SDK far more: bucket create+sync+prune 6.0x, rank orgs by trending models 4.1x, repo create+branch+tag / delete files / copy files across repos 2.4x each. Simple one-shot reads sit near parity or cheaper: batch model metadata 0.5x, count dataset rows 0.3x." width="80%"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-tokens-dark.png" alt="Per-task token ratio of curl/Python SDK divided by the hf CLI on GPT-5.5, sorted high to low. Multi-step tasks cost curl/Python SDK far more: bucket create+sync+prune 6.0x, rank orgs by trending models 4.1x, repo create+branch+tag / delete files / copy files across repos 2.4x each. Simple one-shot reads sit near parity or cheaper: batch model metadata 0.5x, count dataset rows 0.3x." width="80%"/>
</div>

The shape is the whole point. On a one-shot read (count dataset rows, batch metadata) curl and the SDK are fine, sometimes even lighter. But the moment a task is *real work*, several dependent calls that each need the previous result, the agent has to hand-roll the entire chain of REST calls (or dig through the SDK) and the cost blows up: **2.4× to 6× the CLI** on creating a repo with a branch and tag, deleting files, copying across repos, syncing a bucket. The `hf` CLI folds that chain into one command. This is also exactly why a benchmark built on easy reads makes curl and the SDK look better than they really are: it never reaches the tasks where the CLI pulls away.

### Key findings

- **The `hf` CLI is far leaner than curl or the SDK.** For the same task, at equal-or-better success, curl and the SDK burn **1.6× the tokens on Sonnet (302k vs 194k) and 1.8× on GPT-5.5 (346k vs 191k)**. On easy reads they're fine, but on real multi-step work they pay **2× to 6×**: the CLI folds a chain of REST calls into one command, while curl or the SDK re-derives that chain by hand every run.
- **On a stronger model, curl and the SDK work but stay wasteful.** On Sonnet they can't finish parts of the job (0.84; they fumble the writes). On GPT-5.5 they mostly work (0.92), hand-rolling the REST calls (or using the SDK) correctly, but still pay ~1.8× the tokens.

## The hf-cli skill

`hf` ships a **skill**: a compact reference of the whole command surface that an agent loads as context. It's **auto-generated** from the live `hf` command tree, one line per command (its signature, a one-line description, and the flags that matter), grouped by resource, with a short glossary of common options. It deliberately skips the self-explanatory flags so it stays terse and light on context, and it's regenerated every release. Run `hf skills preview` to print it, or install it with:

```bash
# for Codex, Cursor, OpenCode, Pi and other agents that load skills from `.agents/skills`
hf skills add
# includes the above + Claude Code
hf skills add --claude
```

What does it buy you? Mostly, the agent stops guessing. The clearest single view is how many commands each run takes, with the skill and without:

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-skill.png" alt="Mean commands (tool calls) per run, with and without the hf-cli skill, on both agents. Claude Code (Sonnet 4.6): 10.4 without the skill, 6.9 with it. Codex (GPT-5.5): 10.1 without, 7.3 with. Fewer is better." width="100%"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-skill-dark.png" alt="Mean commands (tool calls) per run, with and without the hf-cli skill, on both agents. Claude Code (Sonnet 4.6): 10.4 without the skill, 6.9 with it. Codex (GPT-5.5): 10.1 without, 7.3 with. Fewer is better." width="100%"/>
</div>

On both agents that's about ten commands per task down to about seven, roughly 30% fewer tool calls - because the agent isn't probing `--help` to find the right command and argument; it already has the map. The skill won't cut your token bill (it's a fixed slice of context the agent loads up front, so tokens hold steady or tick up) and it won't make an already-reliable CLI much more reliable. What it buys is fewer wrong turns: the agent spends its budget doing the task instead of rediscovering the tool. This could be particularly helpful when using `hf` with local models.

## Try it yourself

We benchmarked all this because we think it matters. Agents are becoming real users of the Hub: they train models, build and clean datasets, and ship demos as Spaces, almost always on behalf of a person. A Hub that works well for agents is also a Hub that works better for the people using them. The better an agent's tools are, the more it can do for you.

If your agent interacts with the Hugging Face Hub, we recommend giving it the `hf` CLI:

```bash
# macOS / Linux
curl -LsSf https://hf.co/cli/install.sh | bash

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Then hand it the skill, so it knows the whole command surface from the first turn:

```bash
hf skills add            # Codex, Cursor, OpenCode, Pi and other agents that load skills from .agents/skills
hf skills add --claude   # the above + Claude Code
```

Then point your agent at the Hub and let it work. Make sure you're logged in (`hf auth login`), then hand it a prompt like:

```text
Run `hf --help`, then use it to list my models, datasets, and Spaces.
Take a look at how I use the Hub and suggest a few ways you could help me.
```

It'll work out the commands on its own and come back with something useful.

The full command reference lives in the [`hf` CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).

Building an agent harness? **Get it registered!** That's how `hf` learns to detect it, and how the Hub attributes its traffic to you. You simply need to open a small PR adding an entry to [`[agent-harnesses.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/agent-harnesses.ts)`](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/agent-harnesses.ts). Read the [[Register your agent harness](https://huggingface.co/docs/hub/agents-overview#register-your-agent-harness)](https://huggingface.co/docs/hub/agents-overview#register-your-agent-harness) guide for more details.
