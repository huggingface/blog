---
title: "Security incident disclosure — July 2026"
thumbnail: /blog/assets/security-incident-july-2026/thumbnail.png
authors:
- user: huggingface
---

# Security incident disclosure — July 2026

Earlier this week, we detected and responded to an intrusion into part of our production infrastructure. This one was different from anything we had handled before in one important way: it was driven, end to end, by an autonomous AI agent system - and we detected and dissected it largely with AI of our own.

We identified unauthorized access to a limited set of internal datasets and to several credentials used by our services. We are still completing our assessment of whether any partner or customer data was affected, and we will contact any affected parties directly as required. We have found no evidence of tampering with public, user-facing models, datasets, or Spaces, and our software supply chain (container images and published packages) was verified clean.

## What happened

The intrusion started where AI platforms are uniquely exposed: the data-processing pipeline. A malicious dataset abused two code-execution paths in our dataset processing (a remote-code dataset loader and a template-injection in a dataset configuration) to run code on a processing worker. From there, the actor escalated to node-level access, harvested cloud and cluster credentials, and moved laterally into several internal clusters over a weekend.

The campaign was run by an autonomous agent framework (appearing to be built on an agentic security-research harness - used LLM still not known) executing many thousands of individual actions across a swarm of short-lived sandboxes, with self-migrating command-and-control staged on public services. This matches the "agentic attacker" scenario the industry has been forecasting.

## What we did

- Fixed the root vulnerability: the dataset code-execution paths used for initial access are closed.
- Eradicated the attacker's foothold across the affected clusters and rebuilt the compromised nodes.
- Revoked and rotated the affected credentials and tokens, and began a broader precautionary rotation of secrets.
- Deployed additional guardrails and stricter admission controls on our clusters.
- Improved our detection and alerting so a high-severity signal pages a responder in minutes, any day of the week.

We are working with outside cybersecurity forensic specialists to investigate the issue and review our security policies and procedures. Finally, we have also reported this incident to law enforcement agencies.

## For our community

As a precaution, we recommend rotating any access tokens and reviewing recent activity on your account. If you believe you are affected, or want to report a security concern, contact us at security@huggingface.co.

We are grateful to the teams across Hugging Face who responded around the clock, and we are sorry for any disruption this caused. Security is never finished; we will keep raising the bar.

## Analyzing an AI-driven intrusion

The attack was initially surfaced through AI-assisted detection. Our anomaly-detection pipeline uses LLM-based triage over security telemetry to separate real signals from the daily noise, and it was the correlation of those signals that flagged the compromise.

To understand what a swarm of tens of thousands of automated actions did, we ran LLM-driven analysis agents over the full attacker action log, comprised of more than 17,000 recorded events. We ran this analysis on models hosted on our own infrastructure, so that no attacker data - and none of the credentials it referenced - left our environment. This allowed us to reconstruct the timeline, extract indicators of compromise, map the credentials touched, and separate genuine impact from decoy activity, doing in hours what would usually take days.

## What this means

Autonomous, AI-driven offensive tooling is no longer theoretical. It lowers the cost of running a broad, patient, multi-stage campaign, and it operates at machine speed. Defending an online platform now means treating the data and model surface as a first-class attack surface, and using AI on defense to keep pace. We will keep investing there, and keep sharing what we learn.

We will publish a more detailed technical write-up of how we detected and analyzed this campaign, and what we learned using AI for incident response, in a follow-up post.
