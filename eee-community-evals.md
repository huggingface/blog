---
title: "Featuring Every Eval Ever Results on Hugging Face Model Pages"
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6413251362e6057cbb6259bd/czIJDDShvtMBs2M2T7B45.gif
authors:
- user: deepmage121
  guest: true
  org: evaleval
- user: evijit
- user: saylortwift
- user: janbatzner
  guest: true
  org: evaleval
- user: borgr
  guest: true
  org: evaleval
- user: irenesolaiman
---

# Featuring Every Eval Ever Results on Hugging Face Model Pages

Every Eval Ever (EEE) and Hugging Face Community Evals are now intercompatible. We enable cross-posting and interpreting evaluation results, while linking to open models, leaderboards, and a unified standardized metadata store.

EEE <a href="https://evalevalai.com/infrastructure/2026/02/17/everyevalever-launch/">launched</a> in February 2026 as a project of the <a href="https://evalevalai.com/">EvalEval Coalition</a>, the first cross-institutional effort to improve how AI evaluation results get reported by both first and third party evaluators. Hugging Face launched <a href="https://huggingface.co/blog/community-evals">Community Evals</a> in February 2026 to decentralize how benchmark scores get reported on the Hub. Combined, they patch gaps in how users, researchers, and policymakers trust, understand, and choose evaluations and models.

Evaluation results are how we measure model capabilities, compare models against each other, and reason about safety and governance, and yet they are scattered and hard to compare. They live in papers, leaderboards, blog posts, and harness logs, among others, each in its own format. The same model on the same benchmark often returns different scores depending on who ran it and how; LLaMA 65B, for one, has been reported at both 63.7 and 48.8 on <a href="https://huggingface.co/blog/open-llm-leaderboard-mmlu">MMLU</a>. These gaps can arise from <a href="https://arxiv.org/abs/2606.14516">evaluation settings that we found are commonly unreported</a>.

EEE is our fix for the reporting side. It's one JSON schema for an evaluation result that records:

- who ran it
- which model
- how it was accessed
- generation settings
- what the metric actually means
- \[recommended\] companion JSONL file for per-sample outputs.

The schema was built with feedback from researchers and policy researchers, and it takes in results from any source, so harness logs, leaderboard scrapes, and paper numbers all end up in the same shape. The <a href="https://github.com/evaleval/every_eval_ever">GitHub repository</a> has the converters, examples, and a contributor guide. Since launching, the <a href="https://arxiv.org/abs/2606.14516">datastore</a> on Hugging Face has grown to around 229,000 evaluation results across more than 22,000 models and 2,200 benchmarks, pulled from 31 different reporting formats. Reproducing just those runs from scratch would cost somewhere in the hundreds of thousands of dollars, which is a reasonable argument for not letting the data scatter once someone has paid to generate it.

Learn more about the schema and how to contribute <a href="https://evalevalai.com/infrastructure/2026/02/17/everyevalever-launch/">here</a>.

Now, it comes with better integration and attribution. Contributors can now send EEE results to Hugging Face Community Evals. We built a converter that takes your EEE records and writes the small YAML files Hugging Face expects, so you don't have to keep the same result in two formats by hand.

![EEE and Hugging Face Community Evals integration](https://cdn-uploads.huggingface.co/production/uploads/6413251362e6057cbb6259bd/czIJDDShvtMBs2M2T7B45.gif)

This is new functionality for everyone who reports or reads evaluations, not only existing EEE contributors. First-party evaluators reporting on their own models and third-party evaluators reporting on someone else's can both submit to Community Evals and to EEE, and anyone browsing the Hub gets results that trace back to a full record. When you submit your data through your organization's official Hugging Face account, your results show up with a <a href="https://evalcards.evalevalai.com/help/get-verified">verified</a> checkmark on EvalEval, a signal to readers that the numbers come straight from the source. The rest of this post walks through what Community Evals are and what the converter does.

## How Hugging Face Community Evals works together with EvalEval

Hugging Face Community Evals has two sides.

A benchmark lives in a dataset repo that registers itself by adding an `eval.yaml`. Once registered, that dataset page collects and displays a leaderboard of every score reported against it across the Hub. The list of <a href="https://huggingface.co/datasets?benchmark=benchmark:official&amp;sort=trending">official benchmarks</a> grows over time.

A model's scores live in `.eval_results/*.yaml` inside the model repo. They show up on the model card and feed into the matching benchmark leaderboard. Both the model author's own results and results submitted by anyone else through a pull request get aggregated, and each score carries a badge saying whether it was author-submitted, community-submitted, or independently verified. Anyone can add a score to any model by opening a PR with the right YAML file, and the model author can close PRs or hide results on their own repo.

Here is what one of these leaderboards looks like, embedded from the HLE benchmark dataset:
