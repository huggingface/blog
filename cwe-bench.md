---
title: "CWE-bench v2: More coverage, quality, and transparency"
thumbnail: /blog/assets/cwe-bench/thumbnail.jpg
authors:
- user: chalo2000
  guest: true
  org: collinear-ai
- user: nazneen
  guest: true
  org: collinear-ai
- user: SaylorTwift
---

# CWE-bench v2: More coverage, quality, and transparency

[CWE-bench](https://cwe-bench.com/) is designed to test the limits of frontier cyber capabilities. In our continued pursuit to push the horizon at [Collinear AI](https://collinear.ai/), CWE-bench v2 opens up a [public set of 100 tasks](https://huggingface.co/datasets/collinear-ai/cwe-bench) that anyone can run and evaluate their models on. The setup is the same as before. An agent gets a checkout of a real open-source repository and a short instruction, with the goal of finding and fixing a security weakness without breaking existing functionality. Scoring is fully programmatic, so results are deterministic and anyone can reproduce them without extra cost. All our programmatic checks have been curated with the help of LLM judge panels that shape them towards the direction of hardness and fairness. A private set is coming soon, so we can keep tracking progress on tasks no model has seen.

Our goal is to be exhaustive. A defender has to anticipate the ways an attacker could get in, and get it right every single time. These 100 tasks cover 91 different CWEs, spanning a wide range of vulnerability classes. A model that does well here has to understand the multiple weakness surfaces that a potential attacker can exploit. Recognising a familiar pattern is not enough.

As part of our meticulous effort to keep improving quality, for every task we checked that the following questions were true:

- Does the instruction read like a realistic instruction and not an evaluation setting?
- Can every graded behaviour be inferred from the instruction?
- Does the verifier check for correct behaviour and not for specific structure of one fix?
- Does the starter code fail and the reference fix pass?
- Does the original functionality still work with the new fix?
- Do the verifiers fail memorized solutions like the known upstream patch?

Every task was also reviewed by hand. Open-sourcing the set is part of the same effort to improve quality. We welcome feedback on any task, and contributions to the public se<!-- TODO(Collinear): this sentence is cut off in the Google Doc draft -->

## Results

![CWE-bench v2 results: pass@1 and pass@4 for 10 models, with cost per rollout](/blog/assets/cwe-bench/results.png)

Of the 10 models evaluated, Claude Opus 5.5 leads with 67.2% pass@1 and 78.0% pass@4. 95 tasks solved by at least one model, an indication that the tasks are indeed solvable. However, even the leader fails 22% of the set on every attempt, so the long tail is far from solved. The public set, the grader and the harness are open. Run your model, and tell us where it stands.

## Run it from the Hub

Every task is a self-contained [Harbor](https://harborframework.com/) task (a Dockerized environment, an instruction, and a deterministic verifier), and the dataset ships a `registry.json`. Harbor can therefore run the whole benchmark straight from the Hugging Face repo, with nothing to clone:

```bash
harbor run --repo "https://huggingface.co/datasets/collinear-ai/cwe-bench" -d cwe-bench@2.0 -k 4 \
  --agent <agent> --model <model> --env docker \
  --allow-agent-host <model API host>
```

`-k 4` runs 4 attempts per task, and the reported score is the mean reward over all 400 trials. The dataset is gated, so accept its access conditions first and make sure git can authenticate to huggingface.co with your HF token.

The tasks are also tagged `rl-environment`: the same environments and verifiers can provide rewards for training defensive agents, not just for evaluating them.

## A leaderboard on the Hub

CWE-bench is an [official benchmark on the Hub](https://huggingface.co/datasets?benchmark=benchmark:official). Its dataset page shows a leaderboard built from scores published in model repos, and the same scores appear on each model's page. To add your model, open a PR on its repo with a results file, as described in the [eval results docs](https://huggingface.co/docs/hub/eval-results#adding-evaluation-results). The Collinear team curates the leaderboard and can hide results they can't verify.
