---
title: "Evaluating agents entirely on the Hub"
thumbnail: /blog/assets/agent-evals-on-the-hub/thumbnail.png
authors:
- user: nielsr
---

# Evaluating agents on the Hub: a benchmark run with no machine of your own

Agent benchmarks are no longer a simple prompt and an answer. Nowadays, each task on which an agent is evaluated is a [container](https://www.docker.com/resources/what-container/) with a project
in it, sometimes a whole suite of running apps, a [harness](https://en.wikipedia.org/wiki/Agent_harness) that drives a model through tens or hundreds of
tool calls, and a grader that may need a container of its own and an [LLM judge](https://en.wikipedia.org/wiki/LLM-as-a-Judge). That usually
means a workstation with Docker, a few API keys, and a laptop that has to stay awake for a day.

We ran 2 full agent evaluations of [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) on Google's [Physical-Engineering's Last Exam
(PLE)](https://huggingface.co/datasets/google/physical-engineerings-last-exam) entirely on the Hugging Face Hub.

This post explains how, what broke on the way, and what the numbers say. PLE is one of the natively supported [benchmarks](https://huggingface.co/datasets?benchmark=benchmark:official&sort=trending) on the hub (learn more about this new feature [here](https://huggingface.co/docs/hub/eval-results)).

If you're designing an evaluation of your own (choosing the claim, checking the verifier,
reporting uncertainty, deciding what to open), start with
[How to build and maintain an agentic evaluation](https://huggingface.co/blog/agentic-evals).
This post is the other half: running a large, hard and private benchmark at scale on the Hub,
with harnesses that weren't built for it.

## In short

- **Everything on the Hub.** The tasks of the benchmark are available as a [gated dataset](https://huggingface.co/datasets/google/physical-engineerings-last-exam). GLM-5.3-Flash is accessed via [Inference Providers](https://huggingface.co/docs/inference-providers). As for evaluation framework, [Harbor](https://www.harborframework.com/) is leveraged, which is run inside [HF Jobs](https://huggingface.co/docs/hub/jobs-overview) sandboxes, which is serverless compute (CPU and GPU). Each task of the PLE benchmark comes with a Docker image. Complex images are stored in the Hub's [container registry](https://huggingface.co/docs/hub/en/container-images). Finally, each finished trial is pushed to a private dataset as a [session trace](https://huggingface.co/docs/hub/en/session-traces-format).
  As everything runs on the hub, you can close your laptop after running the `launch_job.sh` script.
- **Harness adapters, the infrastructure kind.** We evaluated GLM-5.3-Flash using 2 harnesses (Harbor's `terminus-2` and
  OpenAI's `codex`). Three thin adapters were build to connect them seamlessly to the Hub: an *environment* adapter (Harbor ↔ HF Jobs sandboxes), an *image* adapter (multi-stage task images), and a *model* adapter (a loopback relay that lets an in-sandbox agent reach Inference Providers without ever seeing a token). These are explained in detail below.
- **About $50–60 per full run.** PLE consists of 29 tasks, and the agent gets 3 attempts (trials) for each. This costs about $47–60 per harness,
  almost all of it inference; sandboxes add about two dollars.
- **The harness matters as much as anything.** GLM-5.3-Flash scores **0.273** with
  terminus-2 and **0.152** with codex on the same tasks, same model, same judge. See also the
  [Harness Leaderboard](https://huggingface.co/spaces/joelniklaus/harness-leaderboard) which
  compares the performance of various agent harnesses on the same benchmarks.

## Why run evaluations on the Hub

Three things are hard about agent evals, and the Hub already has an answer for each:

| Problem | Usual setup | On the Hub |
|---|---|---|
| Where the task runs | Docker on a workstation, or a sandbox vendor account | HF Jobs sandboxes, sized per task, billed to your org |
| Where the model runs | One API key per provider | Inference Providers: access to a wide variety of models, [billed to your org](https://huggingface.co/docs/inference-providers/main/en/pricing#billing-for-team-and-enterprise-organizations) |
| Where the results go | A local `jobs/` folder | A private dataset: one trace per trial, live, browsable |

The fourth piece is the benchmark itself. PLE is published as Harbor task folders in a gated
dataset. Its terms forbid redistributing the tasks, so everything that contains task content (agent traces, run archives, images)
stays private in the evaluating org. That's the protected column of the release-strategy table in
[the evaluation guide](https://huggingface.co/blog/agentic-evals): with a gated benchmark you can still open the method, the code and
the scores, but the traces stay with the people who agreed to the terms.

## The stack

The figure below summarizes the overall architecture:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/buckets/nielsr/hub-agent-evals-blog-figures/resolve/hub-eval-stack-dark.png">
  <img src="https://huggingface.co/buckets/nielsr/hub-agent-evals-blog-figures/resolve/hub-eval-stack-light.png" alt="The stack: a laptop launches a CPU Job running Harbor; it reads tasks from the gated dataset, calls Inference Providers (terminus-2), drives agent and grader sandboxes on HF Jobs, and pushes traces and results to a private dataset. World images come from the container registry; codex reaches the model through a relay inside the agent sandbox; graders reach only the LLM judge." width="100%">
</picture>

The launcher can be ran like so:

```bash
export HF_BILL_TO=<your-org> TRACES_REPO=<your-org>/<private-dataset>
AGENT=codex TASKS=<task ids> N_ATTEMPTS=3 N_CONCURRENT=29 scripts/launch_job.sh
```

Some explanation here: the launcher script commits the code to a CPU job as an archive. Harbor, the evaluation framework, runs there, and creates a
sandbox per trial (remember that PLE consists of 29 tasks with 3 trials per task as also seen by the environment variables above). A small watcher pushes each finished trial's [trace](https://huggingface.co/docs/hub/en/agent-traces) and an updated per-task results file to a private HF dataset while the run continues. Progress lines (`TRIAL … reward=…`, `ALERT …`) show up in the job logs, so a run can even be followed from a phone. The `HF_BILL_TO` environment variable sets the Hugging Face org to which you can to set up the billing. The script is run for a given agent harness (above we set this to "codex" via the `AGENT` environment variable).

A run that partly failed doesn't have to be repeated: a regrade job takes the run archive
from the dataset and re-runs only the graders of the trials whose grading failed.

## Adapter 1: Harbor environments on HF Jobs sandboxes

Hugging Face provides a [Sandbox](https://huggingface.co/docs/huggingface_hub/en/guides/sandbox): a Job running a small exec and file server, created in
seconds from any public image. Recent Harbor releases include a built-in environment on top of it (`-e hf-sandbox`), which is what [the evaluation guide](https://huggingface.co/blog/agentic-evals) uses, and it is the simplest way to run tasks with a prebuilt image and open network. PLE needs two things it doesn't have:
network policies (agents with no network, graders that may reach only the judge) and images that have to be built from multi-stage Dockerfiles. To deal with these, we wrote our own environment for Harbor 0.20, which implements Harbor's environment interface on the same `Sandbox`:

- **Sizing.** Each task declares CPUs and memory; the adapter picks the cheapest CPU flavor
  that covers them (most PLE tasks fit `cpu-basic` or `cpu-upgrade`).
- **Images without a builder.** For single-stage Dockerfiles there is no build at all: the
  sandbox starts from the `FROM` image and the Dockerfile is replayed inside it (`RUN`, `ENV`,
  `WORKDIR`, `COPY`, …). Anything that can't be replayed faithfully fails loudly instead of
  building something different.
- **Network policies without NET_ADMIN.** Benchmarks need agents that can't reach the internet,
  and graders that can reach only their judge. Sandboxes have no network namespaces or
  firewall capabilities, so the adapter uses [Landlock](https://docs.kernel.org/userspace-api/landlock.html):
  every command after setup runs under a ruleset that denies outbound TCP `connect`, inherited
  by every child and irreversible, even for root. `no-network` denies all TCP; `allowlist`
  starts a small CONNECT proxy *before* confinement that forwards only to the allowed hosts,
  and confined processes may only connect to the proxy's port. That is how PLE's judge-graded
  verifiers reach exactly one host, the judge's API.
- **Long commands.** Agents like codex run as one command that lasts hours. A single
  streaming exec request for that long gets cut by the transport, so commands that may run
  longer than a few minutes are started detached and polled with short requests.

Landlock filters by port, not by address, and doesn't cover UDP. For tasks with no network at
all that's exact; for tasks with local services (below) it's a port allowlist, which we report
as a caveat.

## Adapter 2: task images that a Space can't build

Twelve PLE tasks are *worlds*: a 14-stage Dockerfile with FreeCAD, LibreOffice, MySQL,
PostgreSQL, Redis, Nextcloud, OpenProject, Wiki.js, Collabora and a gateway in front of a
dozen MCP servers, started by a health check. Up to 12 GB of RAM at runtime, ~8 GB of image.

- **Build.** A Docker Space builder ran out of memory at OpenProject's asset precompile on
  every hardware tier. [kaniko](https://github.com/GoogleContainerTools/kaniko) in a
  `cpu-performance` Job (32 vCPU / 256 GB) builds them in 36–86 minutes each and pushes to a
  private repo in the Hub's container registry. All 12 images were built in parallel.
- **Run.** Jobs pull public images, but a sandbox can't be given credentials for a private
  one. Kaniko's single-snapshot builds are "the public base image + a few build layers", so
  the image adapter starts the sandbox from the public base and streams the build layers into
  it (applying OCI whiteouts), using a short-lived, pull-only token for that one repository.
  The HF token never enters the sandbox. 7–8 GB of layers apply in 3–7 minutes.
- **Services under confinement.** The task's health check starts the apps; the apps, the
  agent and the grader's evidence-collection hook all share one Landlock domain that allows TCP
  only to the world's loopback ports (MCP gateway, MCP servers, web apps, databases), measured
  per image with a probe. Ports 80 and 443 are never allowed. In Docker terms: `--network none`
  plus loopback.

## Adapter 3: models from inside the sandbox

terminus-2 calls the model from the orchestrator and only sends keystrokes into the sandbox,
so the sandbox never needs network. codex, like most coding agents, runs *inside* the sandbox
and calls the model API itself. That's how the PLE leaderboard runs it, so a leaderboard-style
run needs a model path out of an otherwise sealed sandbox, without handing the agent a
credential.

The model adapter is a ~350-line, standard-library-only relay that starts on `127.0.0.1`
before confinement:

- **The token stays out of reach.** The relay reads the token from a file it deletes, marks
  itself non-dumpable, and lives outside the agent's Landlock domain; even root in the sandbox
  gets `EACCES` on its memory and environment. The agent gets a dummy key, and its only route
  out is the relay's port, which forwards only `/v1/*` to the router and pins the model id.
- **Protocol translation.** codex speaks the OpenAI Responses API and resends the whole
  conversation every turn. The router's Responses endpoint caps request bodies at about
  100 KB, which a real task exceeds within a few turns, while its Chat Completions endpoint
  accepts megabytes. The relay therefore serves Responses on top of Chat Completions,
  streaming in both directions: messages, tool calls and outputs, images, reasoning effort,
  usage with cached tokens.
- **What codex expects.** codex groups MCP servers and its sub-agent tools into `namespace`
  tools, which Chat Completions doesn't have; the relay flattens them and maps calls back. It
  also registers the relay as a custom provider so codex compacts long contexts locally
  instead of asking for OpenAI-only server-side compaction.

codex itself (the leaderboard's version, 0.162.1) is installed while the sandbox still has
network, before confinement, in a private Node.js so the world's own apps keep theirs.

## Results on PLE

Model: [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) via Inference
Providers. Judge for rubric and checks-judge tasks: Claude Opus 5.5, as in PLE's own setup.
Score: mean over the 29 dev tasks of each task's mean reward over 3 attempts.

| Harness | All 29 tasks | 12 world tasks | 17 other tasks |
|---|---|---|---|
| terminus-2 (150-turn cap) | **0.273** | 0.333 | 0.230 |
| codex 0.162.1 (reasoning `high`, 6 h limit) | **0.152** | 0.214 | 0.108 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/buckets/nielsr/hub-agent-evals-blog-figures/resolve/ple-per-task-by-harness-dark.png">
  <img src="https://huggingface.co/buckets/nielsr/hub-agent-evals-blog-figures/resolve/ple-per-task-by-harness-light.png" alt="Per-task reward of GLM-5.3-Flash on the 29 PLE dev tasks with terminus-2 and with codex, mean of 3 attempts, in two panels: 12 world tasks and 17 other tasks. terminus-2 is higher on 24 tasks, codex on 3, and 2 are tied at zero." width="100%">
</picture>

A few things stand out:

- **The harness is a first-class variable.** Same model, same tasks, same judge: terminus-2
  beats codex on 24 of 29 tasks (codex wins 3, 2 are tied at zero), and the gap holds when the
  task set is resampled. The PLE
  leaderboard shows the same effect across models; for some open models, the best and worst
  harness differ by a factor of four.
- **Short sessions.** GLM-5.3-Flash under codex works for a median of 4.4 minutes per task and
  then writes a completion summary, rather than running out of time. Under terminus-2 the
  median is about 15 minutes on the world tasks.
- **Worlds are not the hardest part.** Both harnesses score higher on the world tasks than on
  the others, even though terminus-2 has no MCP client and reached the apps only with `curl`.
- **Variance within a task is real.** Three attempts of one task range from 0.14 to 0.61 in
  places, so single-attempt numbers on 29 tasks would be noisy; the 3-attempt mean is stable to
  about ±0.015 for this setup.

**What these numbers are not:** leaderboard submissions. The full list of caveats is in
[Limitations](#limitations).

## What it cost

Everything below is billed to one org: Jobs and sandboxes by the minute, the model per token
through Inference Providers, the judge per token on its own API. The table covers every trial
that went into the two scores, including the ones that failed and were re-run; it leaves out
development probes and smoke tests.

| | terminus-2 (29 tasks × 3) | codex (29 tasks × 3) |
|---|---|---|
| Model: GLM-5.3-Flash via Inference Providers | $43–52 | $41–43 |
| Sandboxes: `cpu-basic` / `cpu-upgrade` | $1.93 (76 sandbox-hours) | $0.73 (31 sandbox-hours) |
| Judge: Claude Opus 5.5 (estimated) | ≈ $6 | ≈ $5 |
| **Total** | **≈ $51–60** | **≈ $47–49** |
| Per task (3 attempts) | ≈ $1.80–2.10 | ≈ $1.60–1.70 |
| Median agent time per attempt | 32 min | 4.6 min |

Shared, one-time: building the 12 world images took 13 kaniko jobs, 11.7 hours on
`cpu-performance`, **$22.21** (including one failed build). The orchestrator jobs ran on
`cpu-basic` for 17 hours in total, **$0.17**.

Some notes on the numbers:

- **Inference dominates, and caching makes it cheap.** Agents resend their whole context
  every turn: terminus-2 read 895M input tokens and codex 1.23B, but 96% and 99% of them were
  cache hits. The ranges come from the router's own per-token prices, which varied between
  calls in our measurements ($0.15–0.225 per million uncached input tokens, about $0.03 cached,
  $0.50–0.75 output).
- **Sandboxes are almost free.** CPU sandboxes cost $0.01–0.03 an hour; even 29 trials in
  parallel, each with its own grader sandbox, add up to about two dollars.
- **The judge is an estimate.** The graders don't record token usage, so we estimated it from
  the size of what each rubric call reads (Opus 5.5 at $4 / $20 per million tokens).
- **Wall clock.** 29 tasks × 3 attempts with codex, 29 at a time, took 98 minutes; the
  12 world tasks × 3 with terminus-2, 12 at a time, took about 2 hours. The 6-hour task limit
  was never close.

For scale, the PLE leaderboard lists about $14 per task for GLM 5.2 with codex at list prices;
GLM-5.3-Flash is a smaller, much cheaper model, and with this harness it also stops much
earlier.

## Limitations

- **Not leaderboard submissions.** terminus-2 isn't one of the leaderboard's harnesses, and it
  ran with a 150-turn cap. codex ran with reasoning effort `high` (the highest the router
  accepts) rather than `max`, with web search disabled, and through the relay's
  Responses-to-Chat translation, which does not send the model's reasoning back to it between
  turns. codex also has no built-in settings for GLM-5.3-Flash, so it used fallback values,
  including a context window smaller than the model's, which makes it compact earlier.
- **Network isolation is weaker for world tasks.** Landlock filters by port, not by address,
  so a process in a world task could reach any internet host on one of the ~40 allowed app
  ports (never 80 or 443). UDP isn't blocked in any task.
- **Images are rebuilt on today's base.** World images are layered onto the current public
  `python:3.13-slim-trixie`; if that tag moves, the base under the build layers changes.
- **Re-runs and exclusions.** Trials that failed because of our harness, not the agent, were
  re-run with the same attempt count: 7 in the terminus-2 runs (6 hit a port gap in two world
  images, 1 lost its terminal), 10 in the codex run (cut exec streams, remote compaction).
  Graders alone (agents not repeated) were re-run for the first terminus-2 run, after a
  working-directory bug in our backend, and for two codex trials (a DNS failure, a timeout).
  One codex trial that crashed but was still graded 0.0 was excluded in favour of its re-run.
- **Three attempts per task.** Attempts of the same task vary a lot (0.14 to 0.61 in one
  case). The 29-task means are stable to about ±0.015 over attempts, but a different set of
  tasks could move them by ±0.08.
- **terminus-2 has no image input and no MCP client.** It scored near zero on the three floor-plan
  tasks, where it reads images pixel by pixel until it runs out of turns, and it used the world apps
  only through `curl`.

## What broke, and what we changed

Most of the work was not the happy path. In case you're building something similar:

| Symptom | Cause | Fix |
|---|---|---|
| Space builds OOM-killed | builder memory is fixed | kaniko in a large CPU Job |
| `pull access denied` for a private image | Jobs take no registry credentials | apply build layers inside the sandbox |
| kaniko fails on `rm -rf /tmp/*` | it runs RUN steps in the Job's own filesystem | skip the Job's mounted file in cleanups |
| world never becomes healthy | apps talk to each other on ports we hadn't allowed | probe each image's listening ports |
| `413 Payload Too Large` | Responses endpoint caps bodies at ~100 KB | serve Responses over Chat Completions |
| `RemoteProtocolError` after hours | one exec stream for a 2-hour command | detached commands, short polls |
| codex sees no MCP tools | `namespace` tool type dropped | flatten and map back |
| "remote compaction v2" errors | codex treats the relay as OpenAI | register it as a custom provider |

Each of these turned into a failed trial first. Because every trial's trace and the run
archive land in the dataset as soon as they exist, a failure is an archive to read rather
than a log that scrolled away, and a partial failure is a regrade rather than a re-run.

## What would make the adapters unnecessary

Each adapter exists because of one missing feature. In order of impact:

- **Pull private registry images in Jobs.** A Job in the org that owns a private `cr.hf.co`
  image can't pull it today (`pull access denied`), and Jobs take no registry credentials.
  Supporting that removes adapter 2's layer streaming, its 3–7 minutes per sandbox, and the
  stream failures we hit under load.
- **Larger request bodies on the router's Responses endpoint.** `/v1/responses` rejects bodies
  over about 100 KB, while `/v1/chat/completions` accepts megabytes. With that limit raised, and
  codex's `namespace` tools accepted, codex could call the router directly and adapter 3's
  translation goes away.
- **Network policies on sandboxes.** A `network="none"` or `allowlist=[...]` option, enforced
  outside the container, would replace the Landlock wrapper and the CONNECT proxy, and close the
  port-based and UDP gaps above.
- **Detached commands that survive disconnects.** Start a process, read its output from an
  offset, get its exit code later. We built this from files and polling after cut streams
  killed 8 of 87 codex trials.
- **Image builds and scoped tokens.** A build job type with BuildKit and enough memory, and
  short-lived inference tokens limited to one model and one org's billing, so an in-sandbox
  agent could hold its own key.

## What's next

- **Upstream.** The network policies and private-image support described here are what PLE
  needs on top of Harbor's built-in `hf-sandbox` environment, and we'd like them to land there,
  so that a benchmark like PLE runs with `-e hf-sandbox` too.
- **More harnesses.** The model relay is harness-agnostic: anything that runs inside the
  sandbox and speaks an OpenAI-style API can use it, so pi, opencode and hermes are next, which
  would also connect this evaluation path to the harness adapters trained in
  [Introducing harness adapters](https://huggingface.co/blog/harness-adapters-at-scale).
- **Leaderboard-grade codex.** A small A/B with reasoning passed back to the model, to measure
  how much the protocol translation costs.

## Openings: run your own benchmark this way

Nothing here is specific to PLE:

1. **Publish tasks in Harbor format** in a dataset (gated if the terms require it).
2. **Point the orchestrator at it** with an org to bill and a private dataset for traces.
3. **Build images that can't be replayed** with one kaniko Job per task.
4. **Pick a harness and a model** on Inference Providers, and read the traces in the dataset
   viewer while the run is still going.

If you evaluate a model or a harness this way, we'd love to hear what broke for you.

## Acknowledgments and sources

Thanks to the authors of PLE for releasing a benchmark of real engineering work, with a
public dev set, graders and a documented route for running it with Harbor, and to the Harbor
maintainers for a task format and runtime that made an HF Jobs backend a matter of one
environment class.

**Benchmark**

- R. Han, Y. Zhuang, A. Ray, H. Yu, Z. CuiZhu, G. Lee, Y. Ming, B. Roy, V. Sakalkar,
  V. Tirumalashetty, B. Gokturk, T. Pfister, C.-Y. Lee. *PLE: Physical-Engineering's Last
  Exam*, 2026. [arXiv:2610.30537](https://arxiv.org/abs/2610.30537)
- PLE dev set (29 tasks, gated):
  [google/physical-engineerings-last-exam](https://huggingface.co/datasets/google/physical-engineerings-last-exam),
  pinned at commit `e0c21fa`
- PLE dev leaderboard, used for the harness versions, reasoning settings and reference
  scores quoted here: [ple-bench.com/leaderboard](https://ple-bench.com/leaderboard?set=dev)

**Harnesses and framework**

- [Harbor](https://github.com/laude-institute/harbor) 0.20.0: task format, orchestrator and
  the `terminus-2` agent
- [codex](https://github.com/openai/codex) 0.162.1, the version on the PLE leaderboard

**Models**

- Evaluated: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), served by
  fireworks-ai through Inference Providers
- Judge: Claude Opus 5.5 ([Anthropic](https://www.anthropic.com/claude)), the judge PLE's
  rubric and checks-judge graders are calibrated with

**Hugging Face infrastructure**

- [HF Jobs](https://huggingface.co/docs/hub/jobs-overview) and the `Sandbox` API in
  [huggingface_hub](https://github.com/huggingface/huggingface_hub) 2.4.0
- [Inference Providers](https://huggingface.co/docs/inference-providers)
- [Session Trace format](https://huggingface.co/docs/hub/en/session-traces-format) for the
  per-trial traces
- The Hub's container registry (`cr.hf.co`) for the world-task images

**Other tools**

- [kaniko](https://github.com/GoogleContainerTools/kaniko) for building images inside a Job
- [Landlock](https://docs.kernel.org/userspace-api/landlock.html) for network confinement
  without extra privileges

**Related**

- [Introducing harness adapters](https://huggingface.co/blog/harness-adapters-at-scale), the companion post on training per-harness
  adapters

All scores in this post come from our own runs (private traces and run archives in the
evaluating org) and are reproducible from them with the scoring script in the code release.
