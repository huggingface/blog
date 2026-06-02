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

# We got local models to triage the OpenClaw repo for FREE!

OpenClaw gets hundreds of issues and PRs every day, which need to be triaged, prioritized and routed to maintainers. I, Onur, am working to make local models work well with OpenClaw. I also have 128 GB of unified memory, namely Nvidia GB10 at my disposal, so I took on the task:

> Can I build a real-time notification system that filters and notifies me for only the issues that I am responsible for... with local models?

I can of course set up my OpenClaw main agent running on my $200/mo ChatGPT pro plan to trigger a job on every new issue or PR. But then that might use up my quota too quickly—so I might instead set it to run every 2 hours, or 6 hours. Since I would be batching a large number of issues, I would be trading real-time notifications for cheaper and lower quality processing.

But a better approach would be to use the hardware I already have up and running to do this for free, or rather, the cost of electricity.
