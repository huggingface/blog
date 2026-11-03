---
title: "CWE-bench v2: More coverage, quality, and transparency"
thumbnail: /blog/assets/cwe-bench/thumbnail.jpg
authors:
- user: chalo2000
  guest: true
  org: collinear-ai
- user: ajayjagan2511
  guest: true
  org: collinear-ai
- user: nazneen
  guest: true
  org: collinear-ai
- user: SaylorTwift
---

# CWE-bench v2: More coverage, quality, and transparency

[CWE-bench](https://cwe-bench.com/) is designed to test the limits of frontier cyber capabilities. In our continued pursuit to push the horizon at [Collinear AI](https://collinear.ai/), CWE-bench v2 opens up a [public set of 100 tasks](https://huggingface.co/datasets/collinear-ai/cwe-bench) that anyone can run and evaluate their models on. The setup is the same as before. An agent gets a checkout of a real open-source repository and a short instruction, with the goal of finding and fixing a security weakness without breaking existing functionality. Scoring is fully programmatic, so results are deterministic and anyone can reproduce them without extra cost. All our programmatic checks have been curated with the help of LLM judge panels that shape them towards the direction of hardness and fairness. We also maintain a [held-out set](https://www.cwebench.com), so that we can keep tracking progress on tasks no model has seen. To evaluate your model on the held-out set, contact [cyber@collinear.ai](mailto:cyber@collinear.ai).

Our goal is to be exhaustive. A defender has to anticipate the ways an attacker could get in, and get it right every single time. These 100 tasks cover 91 different CWEs, spanning a wide range of vulnerability classes. A model that does well here has to understand the multiple weakness surfaces that a potential attacker can exploit. Recognising a familiar pattern is not enough.

As part of our meticulous effort to keep improving quality, for every task we checked that the following questions were true:

- Does the instruction read like a realistic instruction and not an evaluation setting?
- Can every graded behaviour be inferred from the instruction?
- Does the verifier check for correct behaviour and not for specific structure of one fix?
- Does the starter code fail and the reference fix pass?
- Does the original functionality still work with the new fix?
- Do the verifiers fail memorized solutions like the known upstream patch?

Every task was also reviewed by hand. Open-sourcing the set is part of the same effort to improve quality. We welcome feedback on any task, and contributions to the public set.

## Results

Results are tracked in two places on the Hub. The [leaderboard on the CWE-bench dataset page](https://huggingface.co/datasets/collinear-ai/cwe-bench) covers open-weight models and updates on its own: labs and the community add a result by opening a PR on the model's repo with a results file ([how](https://huggingface.co/docs/hub/eval-results#adding-evaluation-results)). For the full comparison, including API-only models such as Gemini, Claude and GPT, see the [CWE-bench leaderboard Space](https://huggingface.co/spaces/collinear-ai/cwe-bench-leaderboard), where new models are added through a PR on the Space.

Open-weight models, from the dataset page:

<iframe
  src="https://huggingface.co/datasets/collinear-ai/cwe-bench/embed/leaderboard?leaderboard_is_expanded=true"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

All evaluated models, including API-only ones:

<iframe
  src="https://collinear-ai-cwe-bench-leaderboard.static.hf.space"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Of the 11 models evaluated, Gemini 4 Argon leads with 67.8% pass@1 and 81.0% pass@4, just ahead of Claude Opus 5.5 (67.3% pass@1, 78.0% pass@4). 95 tasks are solved by at least one model, an indication that the tasks are indeed solvable. However, even the leader fails 19% of the set on every attempt, so the long tail is far from solved. The public set, the grader and the harness are open. Run your model, and tell us where it stands.

![CWE-bench v2: pass@1 against average cost per rollout, with the Pareto frontier](/blog/assets/cwe-bench/pareto.png)

On cost, DeepSeek V4.1 Flash (1017), Claude Opus 5.5 and Gemini 4 Argon form the Pareto frontier: DeepSeek reaches 57.0% pass@1 at $0.04 per rollout, Opus adds ten points at $1.83, and Gemini edges ahead at $3.92.

## Run it from the Hub

Every task is a self-contained [Harbor](https://harborframework.com/) task (a Dockerized environment, an instruction, and a deterministic verifier), and the dataset ships a `registry.json`. Harbor can therefore run the whole benchmark straight from the Hugging Face repo, with nothing to clone:

```bash
harbor run --repo "https://huggingface.co/datasets/collinear-ai/cwe-bench" -d cwe-bench@2.0 -k 4 \
  --agent <agent> --model <model> --env docker \
  --allow-agent-host <model API host>
```

`-k 4` runs 4 attempts per task, and the reported score is the mean reward over all 400 trials. The dataset is gated, so accept its access conditions first and make sure git can authenticate to huggingface.co with your HF token.

The tasks are also tagged `rl-environment`: the same environments and verifiers can provide rewards for training defensive agents, not just for evaluating them.

