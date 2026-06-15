---
title: "We got FREE local models to triage the OpenClaw repo!"
thumbnail: /blog/assets/local-models-pr-triage/thumbnail.png
authors:
- user: osolmaz
- user: burtenshaw
- user: evalstate
- user: pcuenq
- user: lysandre
---

# We got FREE local models to triage the OpenClaw repo!

These last few days have shown us how important it is to own your AI stack and be able to run models locally, especially if you are building your business on top of AI. In that light, we wanted to share how we use local models like DeepSeek-V4-Flash and Gemma-4-E4B in an agent harness, to run classification tasks[^1]. This approach is different from using a model like BERT for classification. A small model in an agent harness like Pi can be used in tandem with structured outputs, to assign labels. We chose this approach, because we already had small models and harness on hand, and have conviction that this setup will increase in popularity as small models improve in capability.[^2]

Our starting point was open source contributions in the OpenClaw repo. OpenClaw gets hundreds of issues and PRs every day, which need to be triaged, prioritized and routed to maintainers. I, Onur, am working to make local models work well with OpenClaw. Being a maintainer of this specific vertical, I need to react quickly to any P0 issues.

With SOTA closed models like GPT-5, Opus, or Sonnet, this is a pretty straightforward task. But I happen to sit on 128 GB of unified memory, namely an NVIDIA GB10. So I took on the task:

> Can I build a real-time notification system that filters and notifies me for only the issues that I am responsible for... with local open-weight models?

<figure class="image table text-center m-0 w-full" style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/local-models-pr-triage/dgx-spark.png" alt="NVIDIA DGX Spark" style="display: block; width: 50%; min-width: 280px; margin: 0 auto;" />
  <figcaption>This tiny box, a.k.a. DGX Spark, can run 3 to 5 Gemma 4 E4B generations at once.</figcaption>
</figure>

If I set up my OpenClaw main agent running on a $200/mo ChatGPT Pro plan to trigger a job on every new issue or PR, that would use up my quota too quickly. I might instead set it to run every 2 hours, or 6 hours. Since we would be batching a large number of issues, we would be trading real-time notifications for cheaper and lower quality processing.

If I were to run this on a local model on the hardware I already have up and running, I would not only have near-instantaneous notifications, I would also be able to do it for free (or rather, for the cost of electricity).

How would that work? We show below.

## Categorizing issues and PRs

We came up with a finite set of labels representing the categories of issues we need to triage, and then use a local model to classify each issue into one of those categories, like `local_models`, `self_hosted_inference`, `acp`, `agent_runtime`, `codex`, `ui_tui` and so on.[^3]

But how to do the classification though? A simple single request to a Chat Completions endpoint with a tool JSON schema, with the topics as an enum?

Kind of. But this is 2026, not 2023, and we have AGENTS. We can do better!

For the local model of choice, use [`gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it), because it makes it possible to make at least 3 concurrent requests safely with the hardware we have, giving us a lot of throughput!

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

We have mentioned earlier that we bundle a specific `pi` configuration that can only perform read-only operations and return classification output. We call it [`localpager-agent`](https://github.com/osolmaz/localpager/tree/main/localpager-agent), named after `localpager`, the main project here. Each PR and issue generates a prompt, which is then passed to the CLI like below, alongside other args:

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
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/local-models-pr-triage/discord-notification.jpg" alt="Localpager Discord notification" style="display: block; width: 100%; min-width: 300px; margin: 0 auto;" />
  <figcaption>This is what the final filtered Discord notification looks like: a PR about the desired vertical gets routed to me.</figcaption>
</figure>

This part is very simple and does not involve any LLMs:

1. We use [openclaw/gitcrawl](http://github.com/openclaw/gitcrawl) to act as a local mirror for the repo. Whenever there is a new PR or issue, each item is normalized into the same shape and written into localpager's own SQLite database. If the item is new, localpager creates a classification job for it.
2. A worker then claims jobs from that queue. It builds a GitHub context object containing the issue or PR title, body, labels, author, state, and optionally comments, changed files, and selected diff excerpts. That means the Gemma does not need to browse GitHub or open the URL itself most of the time. It is handed all the relevant context.
3. The context object is rendered into a prompt and passed to `localpager-agent` as described in the previous section. The agent can think and use reposhell, but must eventually output a classification result in the defined schema.
4. The output is stored back in localpager SQLite database, and relayed to Discord based on the notification policy configured by the user (i.e. notify me for these topics, but not these other ones).

Below is a figure showing the overall architecture of localpager:

<figure class="image table text-center m-0 w-full" style="text-align: center;">
  <img src="assets/local-models-pr-triage/localpager-architecture.svg" alt="Localpager architecture" style="display: block; width: 70%; min-width: 300px; margin: 0 auto;" />
</figure>

The architecture is semi-agentic. Labeling is done agentically, while sending a notification is handled by deterministic rules. This is to make the notification pipeline faster by removing the need for inference for the most straightforward parts of the task. Local inference is free but each task has a resource contention cost: GPU bandwidth should be reserved for tasks where inference is absolutely needed.

Separating it this way also reduces the rate of error: the model would otherwise have two chances to make an error, `classify + notify`. Now it has only one: `classify`.

## Making small models not classify horribly

Let's be frank: `gemma-4-e4b-it` was designed to run on limited hardware, and by default it has a tendency to put too many unrelated labels on a PR or issue. But being small, it can run 10-15x faster than a larger model like [DeepSeek-V4-Flash](https://huggingface.co/antirez/deepseek-v4-gguf) locally and with 4x less memory, which lets us run 3 of them concurrently. For such triage tasks, we can use the larger DeepSeek-V4-Flash model (I will refer to it as DS4) as the teacher: create a dataset of reference labels, and then iterate on the prompt for Gemma to maximize accuracy over the teacher-generated dataset.

We selected over 700 PRs and issues, and came up with a set of 40 labels using Codex. We then constructed a prompt that gives context on what each label means, and did an independent DS4 with `localpager` on each PR/issue to generate the reference labels.

The results are saved in [openclaw-classification-dataset](https://huggingface.co/datasets/dutifuldev/openclaw-classification-dataset). The iteration process did not follow an automated approach like GEPA, and was done in an interactive Codex session semi-manually. Despite that, we were able to get rid of 20% of the false positives and 15% of the false negatives, mostly with the labels that we were concerned about, e.g. `local_models`. We were able to do that without making the prompt any more complex or longer, like adding a new rule for each case where it failed.

For example, [`PR #72404 fix(models): default input=[text,image] for vision-capable explicit-only models`](http://github.com/openclaw/openclaw/pull/72404) was originally labeled by DS4 as `[config]`, but the same prompt with Gemma 4 had given `[local_model_providers, reliability]`. After optimizing the prompt, however, Gemma 4 also gives `[config]` as the correct label.

In another case, [`PR #84549 fix(deepinfra): load all DeepInfra models when user wants to browse...`](http://github.com/openclaw/openclaw/pull/84549) shows that the "correct" label can still be a bit subjective. DS4 labeled it as `[model_serving]`, while the optimized Gemma prompt gave `[model_releases, chat_integrations]`. This assignment is not exactly right, but it is not exactly wrong either.

You can see the original prompt for DS4 and the optimized Gemma prompt [here](https://huggingface.co/datasets/dutifuldev/openclaw-classification-dataset/blob/main/prompts/README.md). We went through multiple iterations, and ended up reducing the number of false positives and false negatives. For this notification system, false positives were more important to get rid of for us, because they waste one's attention and time when an issue is labeled incorrectly. For that reason, we were biased towards reducing false positives more than false negatives.

## Tracking and validating real time performance using OpenClaw

We have mentioned earlier that instead of running a job with a local model for every new issue or PR, we can run a batch job with a SOTA cloud model, like GPT-5.5 running in OpenClaw, every n hours (e.g. every 2 hours) to achieve the same end.

In that case, we would need a ChatGPT Pro plan. Since the model is SOTA, we can still expect it to perform reasonably well, despite batching 2 hours of issues/PRs together.

Because we want to see how well our prompt-optimized Gemma 4 solution performs against GPT-5.5, we run both simultaneously, and let GPT-5.5 be the judge of false positives and negatives, every 2 hours.

To be safe, we run the OpenClaw job in a sandbox, with only access to the [public repo](https://github.com/osolmaz/onurclaw) we report results to. In our case, we let the OpenClaw job update a machine-readable file, then a simple script reads the Codex-assigned labels and computes the false positive/negative status. Example output:

> False negatives
> 
> - Issue #88499 openai-responses provider: 404 on previous_response_id when store=false (default)
>   - inventory area: OpenAI-compatible/proxy; notifier topics: agent_runtime, api_surface, sessions; notification: none
> 
> False positives
> 
> - PR #88275 fix(models-config): allow self-hosted providers without apiKey in models.json (#88267)
>   - notifier interest: i0; topics: self_hosted_inference, local_model_providers, config; notification: sent
> - PR #88266 refactor: extract model catalog core package
>   - notifier interest: i1; topics: config, api_surface, local_model_providers; notification: sent
> - PR #88247 feat: add hosted model providers
>   - notifier interest: i0; topics: local_model_providers, model_serving, docs, api_surface; notification: sent

The instructions on how to classify, edit the machine-readable file, get the false positives and false negatives using a script are present in an [agent skill](https://github.com/osolmaz/onurclaw/blob/main/.agents/skills/openclaw-onur-inventory/SKILL.md) which is referenced in an [OpenClaw cron job](https://docs.openclaw.ai/automation/cron-jobs) that runs every 2 hours. The OpenClaw agent then ingests any new issues or PRs, adds them to the JSON file with appropriate labels, runs the scripts and reports back in the same Discord channel. This way, we can observe the local model's performance every few hours, and get notified of the misses.

## Conclusion

We think that the issue/PR triage task is a specific case of a broader set of tasks which we call "high throughput triage". This post explored the idea of using a local model to filter out information in real time in only one domain, that is, open source contributions.

However, the same approach can be applied to other domains as well:

- News categorization in journalism
- Filtering for posts of interest in social media and forums like X or Reddit
- Triaging customer support tickets
- Triaging content moderation appeals
- Filtering potential outreach while doing sales
- Filtering for certain topics on arXiv while doing research

The list can be extended, but we think that the idea should be clear.

Besides triaging, we have also explored how classification can be performed with agent harnesses running fast local models in a secure manner. We called this approach *agentic classification*: the model is not fed the entire body of information upfront, but can search for more context before returning structured data.

[^1]: For the use case in this post, we have discovered that breaking down a PR/Issue in a way that means the product surface is understood and labelled correctly is a hard problem.
[^2]: Although in our testing we didn't---it would be quite reasonable for a model to conclude a next-step to gather info, use an external classifier. The agentic approach and the traditional approach are not mutually exclusive.
[^3]: See full list of topics and other configuration [here](https://github.com/osolmaz/localpager/blob/main/examples/profiles/openclaw-routing-topics.json)
