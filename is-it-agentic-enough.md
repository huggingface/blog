---
title: "Is it agentic enough? Benchmarking open models on your own tooling"
thumbnail: /blog/assets/is-it-agentic-enough/thumbnail.png
authors:
- user: lysandre
- user: SaylorTwift
- user: pcuenq
---

# Is it agentic enough? Benchmarking open models on your own tooling

![Benchmarking transformers revisions across different metrics](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_6.png)
<p align="center">
  <em>Benchmarking transformers revisions across different metrics</em>
</p>
<br/>

> This is a human-made, agent-focused blogpost.

Coding agents increasingly work with our software instead of us: describe a task, and the agent picks the library,
writes the calls, runs them, and debugs its own mistakes. When the library gets in the way, it will
happily bypass it and rewrite the logic from scratch. This introduces a new concept in library development:
the code should not only be correct and fast, but should be designed so that an agent can drive it effectively. A clunky API
or stale docs annoy us developers, but it now also sends the agent down a longer, more expensive path.

Most benchmarks just look at the final answer. We wanted the whole process instead: not just whether the agent got
it right, but how much work it took to get there, and how that shifts across models, library revisions, and
tasks. We measured exactly that, using `transformers` as our case study.

We're entering an era where open models, open-source tooling, and our open-source libraries make up a stack we
can shape end to end. That changes the question a maintainer should ask: not "can the agent get the right
answer?" (it usually can) but "how do we shape our tooling so it gets there with less work?". 

Here, we will introduce a tool specific benchmark focusing on how the answer was found, and provide a simple 
implementation of one such harness, running entirely on open models driven by the
[pi](https://www.npmjs.com/package/@mariozechner/pi-coding-agent) coding agent, with the full sweep of
models × revisions × tasks fanned out across [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
so every run sees identical hardware.

<br/>
<p align="center">
    But, <b><i>how do you optimize software for agents?</i></b>
</p>
<br/>

We're strong believers in the following two software principles:

- If it isn't tested, then it doesn't work
- If it isn't documented, then it doesn't exist

This remains the same within the realm of agentic-optimized tooling, and, for once, the two are directly tied to
each other.

You want your tool to exist for an agent: it needs to be discoverable. The API needs to be clear and the docs
need to be extensive. They need to be structured in a way that the agent has rapid access to the useful files and
examples. If you want your tool to work for an agent, then you should test it for agentic-use. 

## Testing software for agentic-use

We'll use `transformers` as an example throughout this blogpost: agents *using* it to solve ML tasks (classifying
text, captioning images, transcribing audio), not contributing code to it; though the harness was designed to work
with any tool that can be operated from the command line.

Our intuition on `transformers` was that usage could be dramatically simplified
with a few changes: a CLI, a Skill, and self-contained, task-specific examples. This is the same recipe
recently applied to the [`hf` CLI, redesigned to be agent-optimized](https://huggingface.co/blog/hf-cli-for-agents),
where agents used 1.3–1.8× (and up to 6×) fewer tokens. We wanted to know whether that kind of win generalizes, and
whether it could be useful for transformers as well.

Intuition is a powerful tool, but we wanted more evidence before we opened PRs that add several thousand lines of code to such a widely used codebase as `transformers`. We set out to measure what success looks like.

### Not all successes are equal

Two agents can both produce the correct label for a sentiment-classification task,
but one:

- writes a 40-line Python script, imports `transformers`, debugs a shape error,
  re-runs twice, and finally prints the answer;

while the other

- types `transformers classify --model ... --text "..."` and is done in one call.

Both reach `POSITIVE (0.9999)`, and here are the two paths an agent actually took on this exact task:

```diff
# Task: classify the sentiment of "I absolutely loved the movie, it was fantastic!"

- # one agent: pipe a script into python and parse the output
- python - <<'PY'
- from transformers import AutoTokenizer, AutoModelForSequenceClassification
- import torch
- import torch.nn.functional as F
-
- model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
- tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
- inputs = tokenizer("I absolutely loved the movie, it was fantastic!", return_tensors="pt")
- with torch.no_grad():
-     logits = model(**inputs).logits
- probs = F.softmax(logits, dim=1)
- idx = torch.argmax(probs, dim=1).item()
- print(model.config.id2label[idx], probs[0][idx].item())
- PY

+ # the other agent: one command
+ transformers classify \
+   --model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
+   --text "I absolutely loved the movie, it was fantastic!"
```

Both methods reach the same result. But they have very different profiles in **cost, latency, token usage, and failures**.

If your evaluation only checks the final string, you're blind to these as well as
whether a change you shipped to the library (a CLI improvement, better error messages, a
Skill) actually helped agents.

Our goal with this harness is to evaluate how much work
an agent has to do to perform a given task, and whether changes to the library improve performance.

### How do we run evaluations?

A few words on how we'll evaluate agents here.

We run every task under three variants (or "tiers"); three different ways an agent can come at `transformers`:

```text
bare     pip install transformers, and nothing else
clone    the full transformers source, checked out in the working directory
skill    a packaged Skill: the CLI's docs + task examples, loaded in context
```

These aren't nested: `skill` doesn't contain `clone` (it ships curated docs, not the source tree), and neither
strictly contains the other, each gives the agent a different kind of help. As we'll see, a model can sometimes 
do better on `clone` than on `skill`.

A few more choices:

- For now we only focus on deterministic tasks which can provide an exact match, as they provide a very nice ground for experimentation. Model-as-a-judge and other schemes are the obvious next steps for other tasks.
- Every run is its own Hugging Face Job: one per (model × revision × task), so the whole sweep runs in parallel on identical hardware, which keeps the comparison fair at scale.
- Results and traces land in a Hugging Face Bucket: fast, no versioning needed, and handles very high write concurrency.

### Which models to benchmark against?

Not all models driving agents are equal, and their difference changes what you should look at when running them.

*Large open models*

At one end, you have the largest, most capable open models. On reasonably common tasks,
these should get the right answer, eventually. For them, __"match %"__ saturates near
100% and stops telling you much about your tool; a more relevant benchmark is the effort
it took the agent to get there: how many turns, tokens and seconds it took, and whether they walked a clean
path or used deprecated APIs.

*Local*

Local models vary widely in size, and so do their abilities.
Metrics such as __"match %"__ are more relevant than for their larger counterparts,
as you can see how model sizes/capabilities affect results on your specific
tool.

This harness not only provides guidance to library maintainers on how to improve a repository for agent interactions, it also helps assess how different agents and models perform on the tasks users care about.

The harness scores every run on several axes, so that you can ask what actually matters
for each class of model:

- **match %**: did the final answer contain the expected result (per-task,
  case-insensitive substring / regex / exact, all explicit in the report);
- **median time** and **median tokens** (new vs. cached vs. generated);
- **runs with error %**: including a guard that flags runs which produced *nothing*
  (0 output tokens, no tool calls, no answer) so silent failures don't masquerade
  as "0";
- **marker adoption**: tool-defined behavior markers; see below for an explanation
  of what this is.

All of it lands in a report you can directly examine:

<p align="center">
<iframe
	src="https://transformers-community-is-transformers-agentic.static.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
<br>
  <em>The live report: Overview, Coverage, and Results, all client-side.
  (Not loading? <a href="https://transformers-community-is-transformers-agentic.static.hf.space">open it in a new tab</a>.)</em>
</p>

And because it captures the native agent trace of every run, numbers are just the beginning: you
can read exactly what the agent did, command by command. The traces are shareable
through the Hub's [agent-traces viewer](https://huggingface.co/docs/hub/agent-traces):

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_11.png" alt="A run rendered in the Hub's agent-traces viewer: MiniMax-M2.7 on the answer-question task" width="85%"><br>
  <em>A run rendered in the Hub's agent-traces viewer: MiniMax-M2.7 on the answer-question task.</em><br>
  <a href="https://huggingface.co/buckets/lysandre/transformers-agentic-use/tree/traces/22404f7951/pi/MiniMaxAI--MiniMax-M2.7/bare__answer-question__run1.jsonl"><b>Open this trace on the Hub ↗</b></a>
</p>

Before the results, a quick recap of the setup. Each run varies four things: the **model** driving the agent,
the **`transformers` revision** it runs against, the **task**, and the **tier** (`bare` / `clone` / `skill`).
As discussed, we look at different metrics for the two different model categories.

### Large open models: hold the model, vary the revision

Since a large open model will usually get to the correct result, what you're really
measuring is the effort it took to do so. Did it take ten turns or one? Did it follow
an API path you deprecated because it trusted obsolete documentation? Did it hit an
error you hadn't foreseen?

The natural experiment is to fix one strong model and vary the tool's
revisions: the successive git versions of `transformers` we test against, from released tags like
`v5.8.0` and `v5.9.0` to the specific commit that introduces the CLI and Skill. We want to watch whether the load
it puts on the agent goes up or down. We used the harness on `transformers` to check
whether adding a dedicated CLI and Skill actually lightened the agents' work.

For the three large models we used in our tests, the average time spent on all tasks indicates that the Skill commit
results in less time spent working on the tasks:

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_13.png" alt="Median time per revision, by tier" width="85%"><br>
  <em>Median time per revision, by tier: the skill commit (green dot) is the fastest.</em>
</p>

On the other hand, in the experiments in which we cloned the repository, we can see a significant increase
in token consumption due to the commit that introduced the CLI and examples, as we'll see in a moment.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_12.png" alt="Median new tokens per revision, by tier" width="85%"><br>
  <em>Median new tokens per revision, by tier: the clone variant jumps once the CLI lands in the repo.</em>
</p>

Reading the clone-variant traces explains why. The commit adds a command, but it also ships the
CLI's implementation and a set of `cli/agentic/*.py` usage examples into the repository directly. 

On the `clone` variant the agent has a full transformers checkout in front of it, and roughly a third of the runs go read the new
surface (the `/cli/` tree and the example scripts) to learn the interface before calling it. This raises the
median input from ~4k to ~6.4k tokens. 

The two charts are then two sides of one tradeoff: the commit buys the large models less time 
(they reach for the CLI instead of debugging Python) at the cost of more
tokens (they read the code that taught them the CLI). A tradeoff worth knowing about before merging PRs.

One caveat works in the CLI's favor, though, which isn't benchmarked yet: the cost of reading it is amortized 
with successive runs. Our setup is built for one-off experiments. Each run is a fresh agent that rediscovers 
the CLI from scratch, so it pays the discovery cost every time. In real usage an agent learns the interface 
once and then solves task after task within the same session, amortizing that cost across many requests. The
token bump we measure here is closer to a worst case than to what a user would see day to day.

### Small models: hold the revision, vary the model

Open models give us fine-grained control over the variables that matter most here: size, configuration, quantization,
provider, training, and anything that would differ from one model to the next. They're also where a good tool surface
matters most: a small model asked to "use `transformers` to do X" on a `bare` environment can
guess an API that changed some releases ago, may do unnecessary tool calls, and can
get the wrong answer.

So here the experiment is the opposite of the above: hold the revision and sweep the model. This helps
see which models actually take care of the task, not just by token count and time,
but down to which ones can't reliably handle the tool calls. Our intuition is
that the smaller the model, the harder both tool use and the task get; we ran the
harness across a range of model sizes to test exactly that:


<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_14.png" alt="Match % across models, by tier" width="85%"><br>
  <em>Match % across models, by tier: the skill tier lifts the larger models but drops the smaller ones.</em>
</p>

which also seems to be correlated with the number of tokens ingested

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_15.png" alt="Median new tokens across models, by tier" width="85%"><br>
  <em>Median new tokens across models, by tier.</em>
</p>

> A note on fair comparison: naively averaging across tasks is misleading when
> coverage is uneven (a model that only finished the quick tasks looks fast). The
> report has a **"shared tasks only"** toggle (across models and/or revisions) so
> you compare like-for-like, and a **Coverage** heatmap so you can see exactly which
> task × revision × model cells actually ran.

## Tweaking the tool: markers and results

Two things come together here: how to look past whether the agent succeeded to what it did and how it
did it; as well as the first results we pulled out of the harness.

### What's a marker?

Match %, tokens, and time tell you the cost of a run but don't tell you much about what happened under the hood.

This is why we've introduced the concept of markers. A marker is a named pattern the profile (the
small per-tool plugin that teaches the harness how to build and drive a given library) matches against
a run. 

It is a one-line label for a behavior you care about, checked against the shell
commands the agent ran, the code it wrote, the files it read, or its final
answer. A run can fire several markers or none; the report shows how often each one fired,
per model and per revision.

For `transformers` we declare a handful but we'll only look at the two most relevant ones:

- **`cli`**: the agent invoked the `transformers` command-line tool (e.g.
  `transformers classify …`) instead of writing Python.
- **`pipeline`**: it reached for the high-level `pipeline(...)` Python API.

These are what we watch to see whether a change actually shifted the agent's behavior. Interestingly here,
the larger the model, the more it leverages the new context instead of using its memory; therefore leveraging
the newly introduced CLI.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_16.png" alt="CLI adoption by tier across models" width="85%"><br>
  <em>CLI adoption by tier across models: only the skill tier reaches for it, and more so as models grow.</em>
</p>

CLI adoption is new: the CLI lands in a single commit, isn't in any model's training data, and is only
lightly documented. The effect is clear: it's the Skill variant, the one that ships the CLI's
documentation, that actually reaches for it, at 55.3%.

### Is the CLI + Skill commit helping?

Comparing the commit across model sizes, the CLI + Skill helps the bigger models: on the `skill` tier, Kimi and the other large agents reach for the CLI and finish in fewer turns. (On `clone` they spend *more* input tokens first, reading the new CLI code, as we saw above, so the win shows up in time and turns, not raw tokens.)

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_5.png" alt="Kimi-K2.6, GLM-5.1, and MiniMax-M2.7 across revisions" width="85%"><br>
  <em>Kimi-K2.6, GLM-5.1, and MiniMax-M2.7 across revisions</em>
</p>

But in some smaller-model settings, it appears to hurt performance. One plausible explanation is that small
models lean on memorized API patterns, reproducing `pipeline(...)` snippets
they've seen in their training data. The new concepts are then a larger
surface for them to get wrong. You can watch this directly on the harness: lower
match %, more retries, the `cli` marker barely firing. It is particularly striking on the Qwen3-4B model:
the Skill barely changes its match rate yet its cost distribution is significantly affected. 

Almost all of that comes from the `clone` tier. The checkout now contains
the CLI's implementation and `cli/agentic/*.py` examples, and the 4B agent reads them in bulk: its median new
tokens jump from ~2.4k to ~23k, with time and output skyrocketing as well, for no gain in
accuracy.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_7.png" alt="Qwen3-4B cost distributions across revisions: elapsed, new tokens, repeat tokens, out tokens" width="85%"><br>
  <em>Qwen3-4B across revisions. The CLI + Skill commit fans the cost distribution wide open, on the <code>clone</code> tier the agent reads the newly-shipped CLI source in bulk (~10× the new tokens), for no gain in match %. (<code>repeat tokens</code> stays flat: this setup uses no prompt caching.)</em>
</p>

Sometimes, though, the Skill breaks correctness outright. Reading the traces shows how, for example for Qwen3-14B: 
adding the Skill drops its overall match rate from 67% (bare) to 43%, and on the simplest tasks the collapse is very
visible: `classify-sentiment` goes from 100% on the `clone` variant to **0%** with the Skill. 

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_17.png" alt="Qwen3-14B classify-sentiment match % by tier across revisions" width="85%"><br>
  <em>Qwen3-14B on <code>classify-sentiment</code>, by tier: <code>clone</code> (blue) holds at 100% across revisions, but the Skill variant (green) collapses to 0% at the CLI + Skill revision.</em>
</p>

Looking at the traces, the model mistakes the CLI for a *tool it can call directly* (as in an agentic-harness
tool, like web-search). The Skill is **not** an executable tool: it's documentation loaded
into the agent's context, and the `transformers` CLI is only ever meant to be run from the shell (via `bash`); so this
will not work.

Qwen3-14B reads the Skill and, in 39 of its 56 Skill runs, either emits a `transformers(command="classify", ...)`
tool call (a tool that was never registered) or, finding nothing like it among its `read`/`bash`/`edit`/`write`
tools, concludes it *can't* run a model and gives up. Either way, rather than fall back to the one-line
`pipeline(...)` that scored 100% on the `clone` checkout, it declares the task impossible.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_10.png" alt="Qwen3-14B gives up on classify-sentiment under the Skill variant" width="85%"><br>
  <em>Qwen3-14B on classify-sentiment (Skill variant): it reasons that read/bash/edit/write can't run a model, and gives up.</em>
</p>

This is exactly what we built the harness to catch: the same change that speeds the large models
ends up breaking the small ones, which seemed a bit counterintuitive to us at first and something we'd likely have
shipped as-is. The takeaway for maintainers: **agent-facing APIs should be evaluated across model sizes, because a
new affordance can reduce work for strong models while adding ambiguity for smaller ones.** It also hints at a fix:
rather than hand-write a Skill and check it after the fact, you could generate and validate one against the weaker
models up front. 

This is exactly what [Upskill](https://huggingface.co/blog/upskill) does: it turns a strong model's solution into 
a Skill only when it measurably helps the smaller ones.

## Trying it yourself

The harness is one CLI, `agent-eval`. Install it, run a suite, fan it out across models × revisions on HF Jobs, and publish the
report as a Hugging Face Space. 

> [!WARNING]
> **Trusted local use only.** The harness runs a coding agent with bypassed permissions and executes code from
> whatever revision you point it at, and traces can contain prompts, output, and local paths. See
> [SECURITY.md](https://github.com/huggingface/is-it-agentic-enough/blob/main/SECURITY.md) before pointing it at code you didn't write or sharing results.

The full, kept-current setup and usage instructions live in the [README](https://github.com/huggingface/is-it-agentic-enough).

## Closing

Checking the final answer tells you whether an agent *can* use your library. It
doesn't tell you what it costs: the turns, tokens, errors, and the path it took to
get there. This harness measures that, across the revisions and models you pick.

On `transformers`, it caught something we'd have shipped on faith: the CLI + Skill
helps the largest open models and hurts the smallest ones. Worth knowing before merging!

It's profile-based, and designed to be adaptable: point it at your own library, define
a few tasks and their expected answers, get the same report. Code and tasks are in
the [repo](https://github.com/huggingface/is-it-agentic-enough), traces are on the Hub.
Let us know if you use it for your project!

## Acknowledgements

This harness stands entirely on
[pi](https://www.npmjs.com/package/@mariozechner/pi-coding-agent), Mario
Zechner's coding-agent CLI: it drives every open-model run, and only needs an
`HF_TOKEN` to serve a model, which is what made the open-model sweep practical at
all.

Thanks to the model builders and inference providers behind the models we
swept. Across the board they performed well above what the `bare` baseline would
suggest.