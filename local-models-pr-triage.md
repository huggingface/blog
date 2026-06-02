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

# Using Local Models to Triage and Assign Hundreds of PRs Every Day

OpenClaw gets hundreds of issues and PRs every day, which need to be triaged, prioritized and routed to maintainers. Being personally responsible for local models at Openclaw and having [128 GB of unified memory](https://www.techpowerup.com/gpu-specs/gb10.c4342) at my disposal, I took on the task:

> Can I build a real-time notification system that filters and notifies me for only the issues that I am responsible for?

I can of course set up my OpenClaw main agent running on my $200/mo ChatGPT pro plan to trigger a job on every new issue or PR. But then that might use up my quota too quickly---so I might instead set it to run every 2 hours, or 6 hours. Since I would be batching a large number of issues, I would be trading real-time notifications for cheaper and lower quality processing.
