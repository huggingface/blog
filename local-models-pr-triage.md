---
title: "Using Local Models to Triage and Assign Hundreds of PRs Every Day"
thumbnail: /blog/assets/local-models-pr-triage/thumbnail.png
authors:
- user: osolmaz
- user: burtenshaw
- user: evalstate
- user: pcuenq
- user: lysandre
---

# I got my GPU to triage my OpenClaw PRs for FREE!

Alt title: We got local models to triage the OpenClaw repo for FREE!

OpenClaw gets hundreds of issues and PRs every day, which need to be triaged, prioritized and routed to maintainers. I, Onur, am working to make local models work well with OpenClaw. 

I also happen to have 128 GB of unified memory, namely an Nvidia GB10 a.k.a. DGX Spark, at my disposal, so I took on the task:

> Can I build a real-time notification system that filters and notifies me for only the issues that I am responsible for... with local models?

I can of course set up my OpenClaw main agent running on my $200/mo ChatGPT pro plan to trigger a job on every new issue or PR. But then that might use up my quota too quickly—so I might instead set it to run every 2 hours, or 6 hours. Since I would be batching a large number of issues, I would be trading real-time notifications for cheaper and lower quality processing.

But a better approach would be to use the hardware I already have up and running to do this for free (or rather, for the cost of electricity).

How would that work?

Basically, I came up with a finite set of labels representing the categories of issues I need to triage, and then use a local model to classify each issue into one of those categories, like `local_models`, `self_hosted_inference`, `acp`, `agent_runtime`, `codex`, `ui_tui` and so on.

But how to do the classification though? A simple single request to a Chat Completions endpoint with a tool JSON schema with the topics as an enum?

Kind of. But this is 2026, not 2023. And we have AGENTS. We can do better!
