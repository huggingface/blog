---
title: "The thief's analyst: when AI ranks victims, not just breaks in"
thumbnail: /blog/assets/volta-ai-victim-ranking/thumbnail.png
authors:
- user: relayshieldadmin
  guest: true
---

# The thief's analyst: when AI ranks victims, not just breaks in

Agent builders publish the pieces agents are built from: models, Spaces, and MCP servers. Talos reports VOLTA uses those same pieces in reverse: a stolen bundle posted to OpenRouter, ranked by a model for monetization value, with a three-model fallback when one refuses. Those agents rarely break in through a login. They break through a tool call, a drifted intent, or a document they read, which is where identity, intent, and content checks have to sit, together, at the payoff call.

An infostealer used to steal first, and sort later. VOLTA sorts first, by value, with a model, and it does not stop when a model refuses.

That is the shift in the Talos Intelligence report on VOLTA, published as "Appraisal by AI: Model-Driven Victim Prioritization in the VOLTA Infostealer". VOLTA posts a structured bundle of stolen data to OpenRouter, so a model can rank the victim's assets by monetization value. If that model refuses, the request moves down a three-model chain, instead of stopping.

Be clear about the source and the limit here. This post is based on the Talos report, not on a RelayShield catch, and it does not claim RelayShield caught VOLTA, or that VOLTA indicators are in the RelayShield corpus, unless that is verified separately. What the report shows, if it is accurate, is a change in the attacker's job, from manual triage to model-driven triage, and that change matters for defense, even without a live RelayShield case.

## From breaking in to deciding who is worth hitting

Most infostealer stories focus on the break-in, the stealer, the dump, the sale. VOLTA focuses on the next step, appraisal.

A stolen bundle is not equally valuable in every case. One victim has crypto wallets and high-value accounts, another has low-value credentials, a third has corporate access that is worth more than either. Sorting that by hand takes time, and time is where a victim can change passwords, revoke sessions, and contain the damage.

A model ranking by monetization value shortens that window. It tells the attacker, in structured form, which victim to work first, which assets to monetize first, and which access to sell first. That is not a new break-in technique. It is a new prioritization technique, and prioritization is what turns a large dump into a fast payout.

The three-model fallback matters for the same reason. Refusal, in this design, is treated as a routing problem, not as a block. If the first model refuses to rank stolen assets, the request moves to the next model in the chain, and then to the next, until a model answers. That is a deliberate design choice, and it shows why a single model's refusal is not a defense, when the attacker controls the routing.

## Why this is a content and exfiltration problem, not just a malware problem

VOLTA's ranking step is also an exfiltration step, and that is where defense should look.

A structured bundle of stolen data, posted to an AI API, is a tool call and a tool result, in the same shape as the agent calls RelayShield screens. The bundle goes out, the ranking comes back, and the next action, monetization, sale, or further access, follows from that ranking. If you only screen the initial stealer, and not the outbound bundle and the inbound ranking, you miss the payoff call.
That is why RelayShield's approach, screening tool lists, tool calls, and tool results in both directions, with flagging, quarantine, and blocking at the tool call, as policy dictates, matters here, in principle, not as a claim about VOLTA specifically. Discovery screening is flag-only, with the response forwarded unchanged. A single flag alone does not block. The block, when policy dictates it, lands at the payoff call, which, in a VOLTA-like chain, is the outbound bundle, or the inbound ranking that drives the next monetization step.

Sequence matters too. A single outbound call to an AI API, on its own, can look benign. A sequence, steal, bundle, post to OpenRouter, rank, monetize, sell, is not benign, and that sequence is what kill-chain and sequence scoring are built to surface, across calls, in a session, not from one call, in isolation.

## The full set of dots, connected: TI corpus, MCP Proxy Firewall, and intent

This is the first post where the full RelayShield set connects, in one chain, and it is worth naming each dot, and what it does, without claiming a VOLTA catch.

First dot, threat intelligence. RelayShield's corpus consists of 136 monitored Telegram marketplaces, with 785K indicators and over 10 million citations, plus deterministic scam-kit fingerprinting, with kit identifiers that can be matched per call. That corpus is what lets a screen know, not just that a string looks suspicious, but that a domain, a wallet, a link, or a kit has been seen in threat-actor traffic before, with citations, not just a heuristic guess. For a VOLTA-like chain, that means the outbound bundle, the AI API destination, and the next monetization destination can be checked against known indicators, not just against a blocklist of stealer binaries.

Second dot, the MCP Proxy Firewall, in the traffic path. The proxy sits between the agent and its tools, and it screens tool lists, tool calls, and tool results in both directions, in memory, in the traffic path, not by re-reading files from disk. That placement is the point. A prompt filter on the agent side can be bypassed by the agent itself, or by an injected instruction in a document it reads. A gate in the traffic path cannot be bypassed by an agent, because the call has to pass through it to reach the tool, and the result has to pass through it to reach the agent.

Third dot, intent binding, identity, and the verdict. An agent that posts a stolen bundle to an AI API is making a state-changing call, sending data out, and that call should be bound to intent, for that session, for that amount, for that recipient, and for that scope, or it should be blocked, with a reason of missing_binding, intent_drift, or a revoked or expired passport, as policy dictates. A clean read-only call passes with screening only. A state-changing call with no binding, or with a drifted recipient or amount, does not. That is the difference between a gate and a log, identity, intent, and content, checked together, with a verdict that records all three, and enforcement that follows the verdict.

Fourth dot, the audit and the console, so the gate is not invisible. Proxy gate verdicts are collected, with verdict, policy action, system outcome, reason, tool, binding, and passport, and pushed to the production console, where they appear as live rows in the Verifiers tab, with KPI totals for screened, allows, flags, quarantines, and blocks. That means a VOLTA-like outbound bundle, if it were run through the gate, would not just be blocked in silence, it would be recorded, with a reason, in a system record other than the agent's report, which is the record you need when you ask, who acted, what was authorized, and what actually went out and came back.

Be clear about the limit here. The proxy gate, the intent binding, and the collector to the console are verified locally, in Enforce, and the collector to the production console is live for test verdicts, as of November 1. This post does not claim that the gate blocked VOLTA, or that VOLTA traffic ran through a production gateway, because it did not. It claims, in principle, that this is where a gate belongs, and that the pieces to build that gate, corpus, proxy, intent, verdict, and audit, now connect, in one chain, for the first time in a RelayShield post.

## What this does not change, and what it does

VOLTA does not change the basic hygiene for infostealers. Change passwords, revoke sessions, rotate credentials, check for new sessions and new devices, and treat a stealer infection as a credential compromise, not just a malware removal.

What it does change is the urgency, and the order of work. A victim with high-value assets should not wait in a queue behind a victim with low-value assets, because the attacker will not wait. The ranking step means the highest-value victim is the first victim to be monetized, so the highest-value victim should be the first victim to be contained.

If an attacker can rank victims by value in minutes, then the time between infection and monetization is not a grace period, it is a race. That means the screen has to be in the traffic path, before the ranking call, not after the sale, and it has to be able to block, not just log, when the pattern matches.

It also changes what to screen for. A screen that only looks for a stealer binary, or for a known command-and-control domain, will miss a ranking call to a legitimate AI API, because that call, on its own, looks like a model call. A screen that looks for a sequence, a bundle out, a ranking back, a monetization step next, will see the payoff, even when each call, alone, looks clean.

## What to take from this

AI is not only changing how attackers break in, it is changing how they decide who to hit first, and how fast they move from a dump to a payout. That is a shift in the attacker's workflow, and a shift in workflow needs a shift in defense, from the break-in, to the ranking, to the payoff.

If you run agents, or if you defend users who do, ask where your gate sits. Does it see the tool list, the tool call, and the tool result, in both directions, or does it only see the prompt, or only see the file? Does it bind the call to the identity and the intent that started it, or does it judge each call alone? Does it record its verdict, with a reason, in a system record other than the agent's report, or does it only log, and hope someone reads the log?

And the next time you see a stealer report, do not only ask, how did it break in. Ask, what did it rank, how did it rank it, and what did it do next, because the ranking step is where the victim is chosen, and the payoff step is where the damage is done.

Source: Talos Intelligence, "Appraisal by AI: Model-Driven Victim Prioritization in the VOLTA Infostealer" https://blog.talosintelligence.com/appraisal-by-ai-model-driven-victim-prioritization-in-the-volta-infostealer/

*Originally published at [blog.relayshield.net/volta-ai-victim-ranking](https://blog.relayshield.net/volta-ai-victim-ranking).*
