---
title: "How we rebuilt the hf CLI for AI Agents"
thumbnail: /blog/assets/hf-cli-for-agents/thumbnail.png
authors:
- user: celinah
- user: Wauplin
---

# How we rebuilt the `hf` CLI for AI Agents

`hf` is the official command-line entrypoint to the [Hugging Face Hub](https://huggingface.co). Anything you can do on the Hub from Python, you can do from your terminal: download and upload models, datasets and Spaces; create and manage repos, branches, tags and pull requests; run Jobs on HF infrastructure; manage Buckets, Collections, webhooks and Inference Endpoints; log in and manage your cache. One tool, the whole Hub.

It's also, increasingly, used by **coding agents**: Claude Code, Codex, Cursor and more. So we rebuilt `hf` to make it work for both audiences at once. This is what we did, and how we benchmarked it.

## AI agent traffic on the Hub

We started tracking agent usage of the Hub in April 2026: both the `huggingface_hub` Python SDK and the `hf` CLI built on it tag each request with an `agent/<name>` user-agent, so we can attribute Hub traffic to the coding agent driving it. The two largest by distinct users are **Claude Code and Codex**, well ahead of everything else, and they're the two agents we benchmark later in this article.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-users-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-users.png">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-users.png" alt="Distinct users of the Hugging Face Hub by coding agent since April 2026." width="100%">
  </picture>
</p>

The bars count distinct users per agent; request volume is the sub-label. Claude Code alone is ~40k users and nearly 49M requests, with Codex close behind. These are early numbers (we only began attributing agent traffic in April 2026), but the scale is already significant, and we expect it to keep growing as coding agents become a standard way to work with the Hub.

## Install

The recommended way is the standalone installer (no Python environment required):

```bash
# macOS / Linux
curl -LsSf https://hf.co/cli/install.sh | bash

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

## How we rebuilt the hf CLI for coding agents

The `hf` CLI serves two users from the same commands: humans and coding agents, and they want opposite things. A human wants rich terminal output: ANSI color, padded tables truncated to fit the screen, a green `✓` on success, `✔` for booleans, progress bars, prose hints. An agent wants
the inverse: no ANSI, nothing truncated, every value in full, compact and structured to stay light on tokens. It also can't answer a prompt and will happily re-run a command after a timeout. The rest of this section is how `hf` gives each side what it needs.

### Coding agents detection

We added an agent-detection mechanism inside `huggingface_hub` so that the `hf` CLI knows whether a human or a coding agent is driving it, by reading the environment variables agents set (`CLAUDECODE`/`CLAUDE_CODE` for Claude Code, `CODEX_SANDBOX` for Codex, plus Cursor, Gemini, Pi, etc.. and the universal `AI_AGENT`). It renders the **same command** accordingly, so an agent gets agent-shaped output without passing a flag.

The same command, two renderings:

```text
# human (default in a terminal): aligned table, truncated to fit, with a hint
$ hf models ls --author Qwen --sort downloads --limit 3
ID                  CREATED_AT DOWNLOADS LIBRARY_NAME LIKES PIPELINE_TAG    PRIVATE TAGS
------------------- ---------- --------- ------------ ----- --------------- ------- --------------------
Qwen/Qwen3-0.6B     2025-04-27  21156913 transformers  1283 text-generation         transformers, saf...
Qwen/Qwen2.5-1.5... 2024-09-17  15143953 transformers   725 text-generation         transformers, saf...
Qwen/Qwen3-4B       2025-04-27  14808352 transformers   624 text-generation         transformers, saf...
Hint: Use `--no-truncate` or `--format json` to display full values.

# agent (auto-detected): TSV, full ids + ISO timestamps + every tag, nothing truncated
$ hf models ls --author Qwen --sort downloads --limit 3
id      created_at      downloads       library_name    likes   pipeline_tag    private tags
Qwen/Qwen3-0.6B 2025-04-27T03:40:08+00:00       21156913        transformers    1284    text-generation False   ['transformers', 'safetensors', 'qwen3', 'text-generation', 'conversational', 'arxiv:2505.09388', 'base_model:Qwen/Qwen3-0.6B-Base', 'base_model:finetune:Qwen/Qwen3-0.6B-Base', 'license:apache-2.0', 'text-generation-inference', 'endpoints_compatible', 'deploy:azure', 'region:us']
Qwen/Qwen2.5-1.5B-Instruct      2024-09-17T14:10:29+00:00       15143953        transformers    725     text-generation False   ['transformers', 'safetensors', 'qwen2', 'text-generation', 'chat', 'conversational', 'en', 'arxiv:2407.10671', 'base_model:Qwen/Qwen2.5-1.5B', 'base_model:finetune:Qwen/Qwen2.5-1.5B', 'license:apache-2.0', 'text-generation-inference', 'endpoints_compatible', 'deploy:azure', 'region:us']
Qwen/Qwen3-4B   2025-04-27T03:41:29+00:00       14808352        transformers    625     text-generation False   ['transformers', 'safetensors', 'text-generation', 'arxiv:2309.00071', 'arxiv:2505.09388', 'base_model:Qwen/Qwen3-4B-Base', 'base_model:finetune:Qwen/Qwen3-4B-Base', 'license:apache-2.0', 'endpoints_compatible', 'deploy:azure', 'region:us']
```

A **human** gets an aligned table, truncated to fit the terminal, plus a hint on how to see more, with color cues for status (a green `✓` on success, red on error). An **agent** gets the complete record as TSV: full repo ids, full ISO timestamps, every tag, no ANSI codes, nothing truncated, clean to parse and light on tokens. You can always force the choice with `--format human | agent | json | quiet` (`--json` and `-q` are shorthands).

### Next-command hints

We made `hf` end some commands with a **hint**: the exact next command to run, pre-filled with the ids you just used, so an agent can chain straight to its next step instead of working it out from scratch. A `--dry-run` hands you the line that applies it, creating a Space tells you how to watch it boot, extracting a model card tells you how to fetch just the metadata:

```text
$ hf models card deepseek-ai/DeepSeek-V4-Pro
... (prints the model card) ...
Hint: Use `hf models card deepseek-ai/DeepSeek-V4-Pro --metadata` to extract only the card metadata.
```

For a human that's a convenience. For an agent it's a rail: the next action is named, parameterized with the right ids, and ready to run, so it takes fewer steps working out what to do. Errors behave the same way, naming the fix instead of just failing:

```
Error: `hf repos cp` only works with repositories. Use `hf cp` or `hf buckets cp` for buckets.
```

Hints, warnings and errors all go to stderr while data goes to stdout, so none of this guidance pollutes the output the agent is parsing.

### Non-blocking and safe to retry

`hf` never sits on an interactive prompt waiting for a key an agent can't press. A destructive command still asks a human to confirm, but in agent mode it *fails fast* with the fix in the message (`Use --yes to skip confirmation.`), and `-y`/`--yes` skips it. And because agents retry on timeouts and lost context, operations are built to be safe to repeat: `hf repos create --exist-ok` is a no-op if the repo already exists, re-running an upload re-commits cleanly, and anything destructive can be previewed with `--dry-run` first:

```text
# agent mode: a destructive command without --yes refuses, with the fix in the message
$ hf repos delete my-org/old-model
Error: You are about to permanently delete model 'my-org/old-model'. Proceed? Use --yes to skip confirmation.

# and you can preview anything first with --dry-run
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

The command tree is consistent, **resource + verb** with the obvious aliases (`hf models ls`, `hf repos create`, `hf jobs ps`, `hf collections delete`; `list`/`ls`, `remove`/`rm`), so once an agent learns one command it can guess the rest. And the output composes: `-q` prints one id per line to pipe into the next command, `--json` gives you something to hand to `jq`.

```text
$ hf models ls --author Qwen -q | head -3
Qwen/Qwen3-0.6B
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen3-4B
```

## Benchmarking the `hf` CLI for Coding Agents

To find out whether the `hf` CLI is really more efficient for agents, we measured it. We built a small evaluation harness and ran the same set of Hub tasks through each way of driving the Hub, many times over, grading every run against the live Hub.

### The setup

We defined **18 non-trivial Hub tasks**. Not "download a file", but the kind of thing you'd actually ask for: aggregate a trending org's models, inspect a repo's files and their sizes, upload a folder with include/exclude rules, delete files, copy files across repos, open a PR that adds a license, create a repo with a branch and a tag, sync and prune a Bucket, build a Collection. Each task goes to afresh coding agent with exactly **one** way to talk to the Hub:

- the `hf` CLI with its skill installed,
- the bare `hf` CLI, no skill, and
- `curl` against the REST API.

The config is deliberately clean: a fresh instance per run, no custom MCP servers, no `CLAUDE.md` or `AGENTS.md`, nothing in context to nudge behavior. The task and the tool go into a single prompt, and the agent is told to finish with a `TASK_COMPLETE` or `TASK_FAILED`
marker. Each task/tool combination is run **10 times**, since coding agents are non-deterministic. That's about **520 runs per agent** (18 tasks × 3 tools × 10 reps, minus a cap on one billable Jobs task). And we ran the whole thing twice, on the two most popular coding agents (**Claude Code** with Sonnet 4.6 and **OpenAI Codex** with GPT-5.5), to make sure the result wasn't an artifact of one model.

An agent will report success on work that never landed, so we don't trust the marker; we grade every run independently by **re-querying the Hub**: did the branch really get created, is the file actually gone, does the bucket exist? That's ~1,000 graded runs in total.

### The results

The ranking came out **identical on both agents**:

| agent | tool | mean score | mean commands | curl's token tax | false "done" |
|---|---|:--:|:--:|:--:|:--:|
| **Claude Code (Sonnet 4.6)** | `hf` CLI | **0.94** | **6.9** | baseline | **2 / 163** |
| | curl (REST) | 0.84 | 12.8 | **1.6× tokens** | 11 / 163 |
| **Codex (GPT-5.5)** | `hf` CLI | **0.93** | **7.3** | baseline | **3 / 163** |
| | curl (REST) | 0.92 | 11.0 | **1.8× tokens** | 10 / 163 |

*(false "done" = the agent reported success on the 17 solvable tasks but the Hub said otherwise.
Mean commands = tool calls per run; we don't report turns, since Codex counts a whole run as one.
The `hf` CLI rows are the CLI with its skill installed; what the skill adds over the bare CLI is
shown separately in [the skill section](#a-skill-you-can-hand-your-agent) below. Every run's full
transcript is published [in this bucket](https://huggingface.co/buckets/celinah/hf-cli-agent-benchmark).)*

Two pictures carry the result. First, **task success on Sonnet**, the agent where curl struggles most:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-success-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-success.png">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-success.png" alt="Task success on Claude Code with Sonnet 4.6: hf CLI 94%, curl 84%." width="100%">
  </picture>
</p>

curl trails by ten points, because on Sonnet it simply can't finish parts of the job (the writes, mostly), while the `hf` CLI clears them.

Second, and this is the one to look at, **curl's token tax on GPT-5.5**, broken down per task. Each bar is curl's tokens divided by the CLI's on the same task, so `2.4×` means curl burned 2.4 times as many tokens to do the same thing:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-tokens-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-tokens.png">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-tokens.png" alt="Per-task token ratio of curl divided by the hf CLI on GPT-5.5, sorted high to low. Multi-step tasks cost curl far more: bucket create+sync+prune 6.0x, rank orgs by trending models 4.1x, repo create+branch+tag / delete files / copy files across repos 2.4x each, build a bg-removal Space 2.1x. Simple one-shot reads sit near parity or cheaper: list a repo's files 0.9x, batch model metadata 0.5x, count dataset rows 0.3x." width="80%">
  </picture>
</p>

The shape is the whole point. On a one-shot read (count dataset rows, batch metadata) curl is fine, sometimes even lighter. But the moment a task is *real work*, several dependent calls that each need the previous result, curl has to hand-roll the entire chain of REST calls and the cost blows up: **2.4× to 6× the CLI** on creating a repo with a branch and tag, deleting files, copying across repos, syncing a bucket. The `hf` CLI folds that chain into one command. This is also exactly why a benchmark built on easy reads makes curl look better than it really is: it never reaches the tasks where the CLI pulls away.

### Key findings

- **The `hf` CLI is far leaner than curl.** For the same task, at equal-or-better success, curl burns **1.6× the tokens on Sonnet (302k vs 194k) and 1.8× on GPT-5.5 (346k vs 191k)**, runs about 1.5-1.9× as many commands, and is roughly 1.8× slower. On easy reads curl is fine, but on real multi-step work it pays **2× to 6×**: the CLI folds a chain of REST calls into one command, while curl re-derives that chain by hand every run.
- **On a stronger model, curl works but stays wasteful.** On Sonnet it can't finish parts of the job (0.84; it fumbles the writes). On GPT-5.5 it mostly works (0.92), hand-rolling the REST calls correctly, but still pays ~1.8× the tokens and runs slower.

## A skill you can hand your agent

`hf` ships a **skill**: a compact, auto-generated reference of the whole command surface that an agent loads as context. You can install it using these commands:

```bash
# for Codex, Cursor, OpenCode, Pi and other agents that load skills from `.agents/skills`
hf skills add
# includes the above + Claude Code
hf skills add --claude
```

It installs a compact map of the whole command surface, so the agent doesn't have to rediscover it command by command.

In the benchmark above, the skill's value over the bare CLI is **reliability**: the two cost about the same in tokens, but the skill lifts success (0.92 to 0.94 on Sonnet, 0.92 to 0.93 on GPT-5.5), roughly halves the false "done"s, and removes guesswork about which command and argument to reach for. The clearest single view is how many commands each run takes, with the skill and without:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-skill-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-skill.png">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-skill.png" alt="Mean commands (tool calls) per run, with and without the hf-cli skill, on both agents. Claude Code (Sonnet 4.6): 10.4 commands without the skill, 6.9 with it. Codex (GPT-5.5): 10.1 without, 7.3 with. Fewer is better." width="100%">
  </picture>
</p>

About ten commands per task without the skill, seven with it, on both agents: fewer wrong turns and retries for the same result.


