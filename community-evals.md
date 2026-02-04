---
title: "Community Evals: Because we're done trusting black-box leaderboards over the community"
thumbnail: /blog/assets/community-evals/thumbnail.png
authors:
- user: burtenshaw
- user: SaylorTwift
- user: kramp
- user: merve
- user: julien-c
- user: davanstrien
---

<!-- TODO: add your name to the authors list -->
<!-- we need a thumbnail for this blog post -->

# Community Evals: Because we're done trusting black-box leaderboards over the community

**TL;DR:** Benchmark datasets on Hugging Face can now host leaderboards. Models store their own eval scores. Everything links together. The community can submit results via PR. Verified badges prove that the results can be reproduced.

## **Evaluation is broken**

Let's be real about where we are with evals in 2026\. MMLU is saturated above 91%. GSM8K hit 94%+. HumanEval is conquered. Yet some models that ace benchmarks still can't reliably browse the web, write production code, or handle multi-step tasks without hallucinating, based on usage reports. There is a clear gap between benchmark scores and real-world performance.

Furthermore, there is a gap within reported benchmark scores. Multiple sources report different results. From Model Cards, to papers, to evaluation platforms, there is no alignment in reported scores. The result is that the community lacks a single source of truth. 

## **What We're Shipping**

**Decentralized and transparent evaluation reporting.**

We are going to take evaluations on the Hugging Face Hub in a new direction by decentralizing reporting and allowing the entire community to openly report scores for benchmarks. At first, we will start with a shortlist of 4 benchmarks and over time we’ll expand to the most relevant benchmarks. 

**For Benchmarks:** Dataset repos can now register as benchmarks ([MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa), [HLE](https://huggingface.co/datasets/cais/hle) are already live). They automatically aggregate reported results from across the hub and display leaderboards in the dataset card. The benchmark defines the eval spec via `eval.yaml`, based on [Inspect AI](https://inspect.aisi.org.uk/), so anyone can reproduce it. The reported results need to align with the task definition. 

![benchmark image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/eval-results-blog/benchmark.png)

**For Models:** Eval scores live in `.eval_results/*.yaml` in the model repo. They appear on the model card and are fed into benchmark datasets. Both the model author’s results **and open pull requests** for results will be aggregated. Model authors will be able to close score PR and hide results. 

**For the Community:** Any user can submit evaluation results for any model via a PR. Results get shown as "community", without waiting for model authors to merge or close. The community can link to sources like a paper, Model Card, third party evaluation platform, or `inspect` eval logs. The community can discuss scores like any PR. Since the Hub is Git based, there is a history of when evals were added, when changes were made, etc. The sources look like below.

![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/eval-results-blog/model.png)

To learn more about evaluation results, check out the [docs](https://huggingface.co/docs/hub/eval-results).

## **Why This Matters**

Decentralizing evaluation will expose scores that already exist across the community in sources like model cards and papers. By exposing these scores, the community and we can build on top of them to aggregate, track, and understand scores across the field. Also, all scores will be exposed via hub APIs, making it easy to aggregate and build curated leaderboards, dashboards, etc.

Community evals do not replace benchmarks and leaderboards and closed evals with published leaderboards are still crucial. However, we want to contribute to this with open eval results based on reproducible eval specs. 

This won't solve benchmark saturation or close the benchmark-reality gap. Nor will it stop training on test sets. But it makes the game visible by exposing what is evaluated, how, when, and by whom. 

Mostly, we hope to make the Hub an active place to build and share reproducible benchmarks. Particularly focusing on new and challenging domains that challenge SOTA models more,

## **Get Started**

**Add eval results:** Drop a YAML file in `.eval_results/` on any model repo.

**Check out the score** on the [benchmark dataset](https://huggingface.co/datasets?benchmark=benchmark:official&sort=trending).

**Register a benchmark:** Add `eval.yaml` to your dataset repo and [contact us to be included in the shortlist.](https://huggingface.co/spaces/OpenEvals/README/discussions/2)

*The feature is in beta. We're building in the open. [Feedback welcome.](https://huggingface.co/spaces/OpenEvals/README/discussions/1)*