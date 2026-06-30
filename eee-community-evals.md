---
title: "Featuring Every Eval Ever Results on Hugging Face Model Pages"
thumbnail: /blog/assets/eee_commevals/eee_commevals_banner.png
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
- user: julien-c
---

# Featuring Every Eval Ever Results on Hugging Face Model Pages

Every Eval Ever (EEE) and Hugging Face Community Evals are now intercompatible. We enable cross-posting and interpreting evaluation results, while linking to open models, leaderboards, and a unified standardized metadata store.

EEE [launched](https://evalevalai.com/infrastructure/2026/02/17/everyevalever-launch/) in February 2026 as a project of the [EvalEval Coalition](https://evalevalai.com/), the first cross-institutional effort to improve how AI evaluation results get reported by both first and third party evaluators. Hugging Face launched [Community Evals](https://huggingface.co/blog/community-evals) in February 2026 to decentralize how benchmark scores get reported on the Hub. Combined, they patch gaps in how users, researchers, and policymakers trust, understand, and choose evaluations and models.

Evaluation results are how we measure model capabilities, compare models against each other, and reason about safety and governance, and yet they are scattered and hard to compare. They live in papers, leaderboards, blog posts, and harness logs, among others, each in its own format. The same model on the same benchmark often returns different scores depending on who ran it and how; LLaMA 65B, for one, has been reported at both 63.7 and 48.8 on [MMLU](https://huggingface.co/blog/open-llm-leaderboard-mmlu). These gaps can arise from [evaluation settings that we found are commonly unreported](https://arxiv.org/abs/2606.14516).

EEE is our fix for the reporting side. It's one JSON schema for an evaluation result that records: 

- who ran it  
- which model  
- how it was accessed  
- generation settings  
- what the metric actually means  
- \[recommended\] companion JSONL file for per-sample outputs.

The schema was built with feedback from researchers and policy researchers, and it takes in results from any source, so harness logs, leaderboard scrapes, and paper numbers all end up in the same shape. The [GitHub repository](https://github.com/evaleval/every_eval_ever) has the converters, examples, and a contributor guide. Since launching, the [datastore](https://arxiv.org/abs/2606.14516) on Hugging Face has grown to around 229,000 evaluation results across more than 22,000 models and 2,200 benchmarks, pulled from 31 different reporting formats. Reproducing just those runs from scratch would cost somewhere in the hundreds of thousands of dollars, which is a reasonable argument for not letting the data scatter once someone has paid to generate it.

Learn more about the schema and how to contribute [here](https://evalevalai.com/infrastructure/2026/02/17/everyevalever-launch/).

Now, it comes with better integration and attribution. Contributors can now send EEE results to Hugging Face Community Evals. We built a converter that takes your EEE records and writes the small YAML files Hugging Face expects, so you don't have to keep the same result in two formats by hand.

![Verified Evaluators on Eval Cards](https://cdn-uploads.huggingface.co/production/uploads/6413251362e6057cbb6259bd/czIJDDShvtMBs2M2T7B45.gif)

This is new functionality for everyone who reports or reads evaluations, not only existing EEE contributors. First-party evaluators reporting on their own models and third-party evaluators reporting on someone else's can both submit to Community Evals and to EEE, and anyone browsing the Hub gets results that trace back to a full record. When you submit your data through your organization's official Hugging Face account, your results show up with a [verified](https://evalcards.evalevalai.com/help/get-verified) checkmark on EvalEval, a signal to readers that the numbers come straight from the source. The rest of this post walks through what Community Evals are and what the converter does.

## How Hugging Face Community Evals works together with EvalEval

Hugging Face Community Evals has two sides.

A benchmark lives in a dataset repo that registers itself by adding an `eval.yaml`. Once registered, that dataset page collects and displays a leaderboard of every score reported against it across the Hub. The list of [official benchmarks](https://huggingface.co/datasets?benchmark=benchmark:official&sort=trending) grows over time.

A model's scores live in `.eval_results/*.yaml` inside the model repo. They show up on the model card and feed into the matching benchmark leaderboard. Both the model author's own results and results submitted by anyone else through a pull request get aggregated, and each score carries a badge saying whether it was author-submitted, community-submitted, or independently verified. Anyone can add a score to any model by opening a PR with the right YAML file, and the model author can close PRs or hide results on their own repo.

Here is what one of these leaderboards looks like:

<iframe src="https://huggingface.co/datasets/cais/hle/embed/leaderboard" frameborder="0" width="100%" height="560px"></iframe>

This is where EEE and Community Evals fit together. When you send a result to both, two things happen: First, your score appears on the Hugging Face model page and gets pulled into the benchmark's leaderboard. And second, it carries a source badge that links straight back to the full EEE record, where the generation config, the harness version, the reproducibility notes, and any instance-level data live.

<iframe src="https://evaleval-general-eval-card.hf.space/embed/eval/frontier/mmlu-pro/mmlu-pro" width="100%" height="420" frameborder="0" style="border:1px solid #e5e5e5;border-radius:4px;" loading="lazy" title="Score distribution — Frontier"></iframe>

![EvalEval as source on SmolLM2 Model Page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/eee_commevals/smollm2.png)


*An Evaluation (MMLU-Pro) from EEE Datastore (a) cross-linked at the file level to a Hugging Face model card (b). The Source EvalEval badge links to the full JSON record.*

**The two destinations do different jobs toward the same goal. Hugging Face puts your result where people look at models, with a link back to the source. EEE keeps the full structured record that makes the result interpretable, and powers [Eval Cards](https://evalevalai.com/projects/eval-cards/) on top of it.** Send your data to both and the same evaluation ends up visible and legible at once, which is the point of reporting one at all.

You can see that cross-compatibility below. The same GPQA scores that surface on the model card above also render in Eval Cards, which composes the EEE run data with benchmark and model metadata into one interpretable record. Same evaluation, a different surface:

<iframe src="https://evaleval-general-eval-card.hf.space/embed/eval/leaderboard/openeval/gpqa" frameborder="0" width="100%" height="560px">
</iframe>

## How it works

Hugging Face stores eval scores in the model repo as a YAML under `.eval_results/`. The required fields are just the benchmark dataset, the task, and the value. The source block is the part that creates the backlink to EEE.

```
- dataset:
    id: openai/gsm8k
    task_id: gsm8k
  value: 96.8
  date: '2024-07-16'
  notes: '8-shot CoT'
  source:
    url: https://huggingface.co/datasets/evaleval/EEE_datastore/blob/main/flat/objects/<xx>/<yy>/<uuid>.json
    name: EvalEval
```

**The converter fills this in from your existing records.** It maps `source_data.hf_repo` to `dataset.id`, `evaluation_name` to `task_id`, `score_details.score` to `value`, and `evaluation_timestamp` to `date`, then drops in the datastore object URL as the source link to the per-record EEE JSON. It currently handles four of the official benchmarks: MMLU-Pro, GPQA, HLE, and GSM8K.

**The converter does more than reshape fields.** You point it at one EEE datastore collection and it downloads that collection along with the records it references, checks the object hashes, and finds the scores that map to a supported benchmark. Before it writes anything live it audits what already exists: it reads every `.eval_results` YAML on the model's main branch and in open PRs, and compares by dataset and task rather than by filename. If a score is already there it is marked `already_present`, if a different score is there it is flagged as a `score_conflict`, and if the model repo doesn't resolve on the Hub it is marked `missing_hf_model`. Everything else is marked `ready`.

**Nothing gets pushed without your sign-off.** The tool writes local YAML previews and a review file you can inspect, shows a report of what is ready and what needs attention, and only opens PRs after you type `OPEN PRS` and enter a commit message. Reruns reuse the cached results for a collection unless you pass `--force`.

![TUI of the Converter](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/eee_commevals/terminal%20shot.png)

*The converter's review step. Excluded entries (here, models with no matching Hub repo) are listed with their EEE source URLs, and the ready PRs wait on an explicit OPEN PRS confirmation.*

## Start here

Submit your full records to [the EEE datastore](https://huggingface.co/datasets/evaleval/EEE_datastore). 

Utilizing EEE requires only one additional step, which the converter largely automates. The [community eval converter tool](https://github.com/evaleval/every_eval_ever/tree/main/tools/hf-community-evals) can be found in the GitHub repository. To process a collection, execute the following:

```shell
uv run tools/hf-community-evals/community_evals_converter.py MMLU-Pro \
  --datastore evaleval/EEE_datastore@main
```

Review the previews and the report it generates, then type `OPEN PRS` when you're ready to submit. Full documentation for the schema, CLI, and converters is at [evalevalai.com/every\_eval\_ever/hf-community-evals](https://evalevalai.com/every_eval_ever/hf-community-evals/).
