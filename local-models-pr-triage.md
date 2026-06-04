---
title: "We got local models to triage the OpenClaw repo for FREE!"
thumbnail: /blog/assets/local-models-pr-triage/thumbnail.png
authors:
- user: osolmaz
- user: burtenshaw
- user: evalstate
- user: pcuenq
- user: lysandre
---

# We got local models to triage the OpenClaw repo for FREE!

(Minor concern: this is not being used in production but only in my local setup, in case that is understood)

Alt title: I got my GPU to triage my OpenClaw PRs for FREE!

---

OpenClaw gets hundreds of issues and PRs every day, which need to be triaged, prioritized and routed to maintainers. I, Onur, am working to make local models work well with OpenClaw. 

I also happen to have 128 GB of unified memory, namely an NVIDIA GB10, at my disposal, so I took on the task:

> Can I build a real-time notification system that filters and notifies me for only the issues that I am responsible for... with local models?

<figure class="image table text-center m-0 w-full" style="text-align: center;">
  <img src="https://i.imgur.com/3cGIhZd.png" alt="NVIDIA DGX Spark" style="display: block; width: 50%; min-width: 280px; margin: 0 auto;" />
  <figcaption>This tiny box, a.k.a. DGX Spark, can run 4-6 Gemma 4 E4B generations at once.</figcaption>
</figure>

We can of course set up my OpenClaw main agent running on my $200/mo ChatGPT pro plan to trigger a job on every new issue or PR. But then that might use up my quota too quickly—so we might instead set it to run every 2 hours, or 6 hours. Since we would be batching a large number of issues, we would be trading real-time notifications for cheaper and lower quality processing.

But a better approach would be to use the hardware we already have up and running to do this for free (or rather, for the cost of electricity).

How would that work? See below.

## Categorizing issues and PRs

Basically, we came up with a finite set of labels representing the categories of issues we need to triage, and then use a local model to classify each issue into one of those categories, like `local_models`, `self_hosted_inference`, `acp`, `agent_runtime`, `codex`, `ui_tui` and so on.[^1]

But how to do the classification though? A simple single request to a Chat Completions endpoint with a tool JSON schema, with the topics as an enum?

Kind of. But this is 2026, not 2023, and we have AGENTS. We can do better!

For the local mode of choice, use [`gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it), because it makes it possible to make 3 concurrent requests safely with the hardware we have, giving us a lot of throughput!

We use an agent harness to drive the classification run. For this, we bundle [pi](https://pi.dev) as a harness that can call local model endpoints.

The agent by default receives the PR title, body and a truncated excerpt of the PR diff in the first prompt. Then, it can choose to use the `bash` tool to perform read-only operations on the OpenClaw repo (in case it needs to look at the codebase), or the `final_json` tool to submit the final classification result.

You wouldn't want to give full bash access to a model like Gemma 4, because there is a higher likelihood of getting prompt injected, due to the model being small!

For that reason, we use [`reposhell`](https://github.com/osolmaz/localpager/tree/main/reposhell) instead of `bash`: a restricted `bash`-like shell that only allows read-only operations (`ls`, `find`, `cat`, `grep`, etc.) on the OpenClaw repo. The model thinks it is using `bash`, but any operation that is not allowed is rejected:


```
reposhell bound cwd=/repo/openclaw repos=openclaw
type help for allowed commands; exit or quit to leave

reposhell /repo/openclaw> help
allowed: pwd, ls, find, rg, grep, sed -n, cat, head, tail, wc -l, git status --short, git show --name-only, git grep, git ls-files
search: rg -n -i "lm studio" or grep -R -n -i "lm studio" .
files: rg --files -g "*.ts" or git ls-files src
examples: rg -n reposhell README.md | sed is not allowed; use one simple command at a time

reposhell /repo/openclaw> head README.md
# 🦞 OpenClaw — Personal AI Assistant

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/openclaw/openclaw/main/docs/assets/openclaw-logo-text-dark.svg">
        <img src="https://raw.githubusercontent.com/openclaw/openclaw/main/docs/assets/openclaw-logo-text.svg" alt="OpenClaw" width="500">
    </picture>
</p>

<p align="center">

reposhell /repo/openclaw> curl localhost
reposhell policy denied command: unsupported command "curl"
exit_code=2

reposhell /repo/openclaw>
```

We have mentioned earlier that we bundle a specific `pi` configuration that can only perform read-only operations and return classification output. We simply call it [`localpager-agent`](https://github.com/osolmaz/localpager/tree/main/localpager-agent), named after `localpager`, the main project. Each PR and issue generates a prompt, which is then passed to the CLI like below, alongside other args:

```bash
localpager-agent \
  --model "<model-id>" \
  --base-url "<openai-compatible-base-url>" \
  --session-dir "<session-output-dir>" \
  --final-schema "<runtime-schema.json>" \
  --tools bash,final_json \
  --reposhell-socket "<reposhell.sock>" \
  --reposhell-default-repo "<repo-id>" \
  --reposhell-visible-repos "<repo-id>[,<repo-id>...]" \
  -p "$(cat <rendered-prompt.md>)"
```


## Processing incoming PRs and issues

So then what orchestrates everything in between the incoming PR/issue and the final notification on Discord?

<figure class="image table text-center m-0 w-full" style="text-align: center;">
  <img src="https://i.imgur.com/gPf3nSrh.jpg" alt="Localpager Discord notification" style="display: block; width: 100%; min-width: 300px; margin: 0 auto;" />
  <figcaption>This is what the final filtered Discord notification looks like: a PR about the desired vertical gets routed to me.</figcaption>
</figure>

This part is very simple and does not involve any LLMs:

1. We use [openclaw/gitcrawl](http://github.com/openclaw/gitcrawl) to act as a local mirror for the repo. Whenever there is a new PR or issue, each item is normalized into the same shape and written into localpager's own SQLite database. If the item is new, localpager creates a classification job for it.
2. A worker then claims jobs from that queue. It builds a GitHub context object containing the issue or PR title, body, labels, author, state, and optionally comments, changed files, and selected diff excerpts. That means the Gemma does not need to browse GitHub or open the URL itself most of the time. It is handed all the relevant context.
3. The context object is rendered into a prompt and passed to `localpager-agent` as described in the previous section. The agent could thinks, use reposhell, but must eventually output a classification result in the defined schema.
4. The output is stored back in localpager SQLite database, and relayed to Discord based on the notification policy configured by the user (i.e. notify me for these topics, but not these other ones).

Below is a figure showing the overall architecture of localpager:

<figure class="image table text-center m-0 w-full" style="text-align: center;">
  <img src="assets/local-models-pr-triage/localpager-architecture.svg" alt="Localpager architecture" style="display: block; width: 70%; min-width: 300px; margin: 0 auto;" />
</figure>

## Making small models not classify horribly

Let's be frank: `gemma-4-e4b-it` was designed to run on limited hardware, and by default it has a tendency to put too many unrelated labels on a PR or issue. But being small, it can run 10-15x faster than a larger model like [DeepSeek-V4-Flash](https://huggingface.co/antirez/deepseek-v4-gguf) and with 4x less memory, which lets me run 3 of them concurrently. And for such triage tasks, we can use the larger DS4 to be the teacher: create a dataset of more correct classifications, and then iterate on the prompt for Gemma to maximize accuracy over the teacher-generated dataset.

That is exactly what we did, and saved the results in [openclaw-classification-dataset](https://huggingface.co/datasets/dutifuldev/openclaw-classification-dataset).

For example, [`PR #72404 fix(models): default input=[text,image] for vision-capable explicit-only models`](http://github.com/openclaw/openclaw/pull/72404) was originally labeled by DS4 as `[config]`, but the same prompt with Gemma 4 had given `[local_model_providers, reliability]`. After optimizing the prompt, however, Gemma 4 also gives `[config]` as the correct label.

In another case, [`PR #84549 fix(deepinfra): load all DeepInfra models when user wants to browse...`](http://github.com/openclaw/openclaw/pull/84549) shows that the "correct" label can still be a bit subjective. DS4 labeled it as `[model_serving]`, while the optimized Gemma prompt gave `[model_releases, chat_integrations]`. Whereas this assignment is not exactly right, it is not exactly wrong either.

You can see the original prompt for DS4 and the optimized Gemma prompt [here](https://huggingface.co/datasets/dutifuldev/openclaw-classification-dataset/blob/main/prompts/README.md).

[^1]: See full list of topics and other configuration [here](https://github.com/osolmaz/localpager/blob/main/examples/profiles/openclaw-routing-topics.json)
