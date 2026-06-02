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
