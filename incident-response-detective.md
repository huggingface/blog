---

title: "Incident Response Detective: Teaching AI to Resist Social Engineering in SRE Operations"
thumbnail: /blog/assets/incident-response-detective/thumbnail.png
authors:

- user: Shiggii

---

# Incident Response Detective: Teaching AI to Resist Social Engineering in SRE Operations

## The Hook

Large language models fail catastrophically under adversarial social pressure. Llama 3.3 70B achieves only 0.001 success rate on our adversarial_easy task—a 99.9% failure rate on scenarios that human SREs handle routinely. The same runbook and logs are on screen, but a confident on-call lead and a second voice push the *wrong* fix—*scale* when *rollback* is right—and the model follows *them*, not the evidence. That is not a corner case. It is what happens when confident human language outranks sparse log signal while the PagerDuty clock ticks.

## The Problem

Production incident response is a social-technical system. PagerDuty fires, Slack fills with hot takes, and the logs are a haystack. Runbooks are supposed to be law—yet under pressure, people still lobby for scale-ups, flushes, and rollbacks because someone *sounds* sure. AI-assisted SRE tooling inherits that risk: an agent that obeys the chat instead of the runbook can automate the wrong action at the worst time. The bug is not vocabulary—it is **authority** and **urgency** overwhelming ambiguous evidence. We needed a benchmark where only the *chat* is adversarial, logs and runbook unchanged, to measure that failure directly.

## The Solution

**Incident-Response-Detective** is an OpenEnv triage environment: each episode blends **server logs**, **Slack-style chat**, and a **runbook**—the same triad real on-calls juggle. There is no single keyword win; the agent must read prohibitions, trace timestamps, and sometimes ignore a wall of bad advice. The reward is **shaped**: **safety** (runbook-consistent, non-dangerous actions) plus **efficiency** (resolve fast on the right path), with a hard floor for genuinely dangerous actions. **Adversarial** mode only swaps the chat, so you isolate social engineering from the rest of the signal.

## Training Approach

We fine-tuned with **GRPO** and **LoRA** on **Qwen 2.5-0.5B-Instruct** in a Kaggle pipeline—about **400 optimizer steps** to align a small model with the environment's reward, so it learns a policy, not a script. (We also keep a 384-step **Groq evaluation harness** in the repo for curves and ablations; the Kaggle run is what updates real weights and carries the main headline on adversarial easy.)

## Results

The behavioral win is the **0.201 → 0.999** jump on the **adversarial** easy regime, where the only attack is *social*—engineered agreement for the wrong remediation. Table from harness before/after eval:


| Task                      | Before Training | After Training | Improvement        |
| ------------------------- | --------------- | -------------- | ------------------ |
| task_easy (adversarial)   | 0.2006          | 0.999          | +0.798             |
| task_medium (adversarial) | 0.999           | 0.999          | — (already strong) |
| task_hard (adversarial)   | 0.999           | 0.999          | — (already strong) |


**Easy + adversarial** is the real stress test: the room is wrong on purpose. Medium and hard often hit ceiling when the runbook or logs give a clear hook—the headline is a tiny model learning to *withstand* a chat pile-on.

## Try It

- **Hugging Face Space:** [Shiggii / incident-response-detective](https://huggingface.co/spaces/Shiggii/incident-response-detective)
- **Kaggle notebook (GRPO + LoRA):** [notebookb5136cd284](https://www.kaggle.com/code/shikharkumarsanjay/notebookb5136cd284)
- **Trained adapter:** [Shiggii / qwen-incident-response-grpo](https://huggingface.co/Shiggii/qwen-incident-response-grpo)

## Future Work

We have a **procedural** task generator that varies names, services, and timestamps under fixed causal structure—the next move is to feed that pool into training and evaluation at scale, not only three hand-authored scenarios.
