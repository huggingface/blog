---
title: "How to build and maintain an agentic evaluation"
thumbnail: /blog/assets/agentic-evals/thumbnail.png
authors:
- user: evijit
- user: burtenshaw
- user: irenesolaiman
---

# How to build and maintain an agentic evaluation

## Overview

AI evaluations help shape which systems people build, adopt, and trust. An evaluation gives a system tasks, checks its answers or actions, and produces a score. Building and maintaining a useful evaluation therefore takes real work: defining tasks that reflect domain expertise, scoring them reliably, and checking that the results support the claims made. When methods, code, and data stay private, that work is harder for others to scrutinize or build on.

Open source offers a means to build stronger, more frequently updated, and more comparable evaluations and the infrastructure around them.  Existing tasks, environments, and tools give builders a starting point, so their effort can go to domain expertise and validation. Sharing how an evaluation was built, including what went wrong, lets others find errors, improve it, and reuse it.

The Hugging Face hub is the home for AI evaluation: we host around 3500 eval leaderboards and have seen about a [16X growth in eval datasets on the hub in the past year](https://huggingface.co/spaces/huggingface/hub-stats), to about 17000 datasets. In recent months, we have also incorporated support for agentic abstractions that are relevant to evaluation. For example, [Traces](https://huggingface.co/docs/hub/en/agent-traces) that store an agent’s work in an easily reviewable and sharable way; [Benchmarks](https://huggingface.co/datasets?benchmark=benchmark%3Aofficial) that highlight datasets used to evaluate models with leaderboards; and [Environments](https://huggingface.co/blog/rl-environments) that show datasets used as agentic RL or evaluation environments.

This guide covers four things:

- [**Three eras of evaluation**](#1-introduction-how-evaluation-design-has-changed) (section 1). Evaluations moved from checking answers, to checking what an agent did, to checking what a team of agents produces. Each step adds software that can change the score.
- [**The steps in an agentic evaluation**](#2-building-an-eval-what-to-reuse-adapt-and-validate) (section 2). We follow one task through eight steps, from the claim to the reported score. The full walkthrough, with commands, is in [Appendix A](#appendix-a-practical-example-agentic-deployment-in-a-physics-lab).
- [**When a benchmark is wrong**](#3-bitter-lessons-when-a-benchmark-is-wrong) (section 3). Errors in tasks and graders are common. We look at how they are found and corrected.
- [**What to open strategically and why**](#4-open-sourcing-an-evaluation-strategically) (section 4). A table sets out, component by component, what to share and what can stay protected.

> [!NOTE]
> In the research community, AI evaluation terminology is [notoriously inconsistent](https://ai.scal.ing/p/evals-intro-1). The [glossary](#glossary) at the end defines each term we use and notes the other names you may come across.

## 1. Introduction: how evaluation design has changed

AI evaluation design has evolved in three forms as language models have become more capable and able to perform more complex work: question answer, agentic, and teams or swarms. Question-answer benchmarks such as [MMLU](https://arxiv.org/abs/2009.03300) (2020) and [GPQA](https://arxiv.org/abs/2311.12022) (2023) measure knowledge and reasoning through the answers models produce. As models became capable of planning and using tools, agentic benchmarks such as [SWE-bench](https://arxiv.org/abs/2310.06770) and [WebArena](https://arxiv.org/abs/2307.13854) expanded the scope to resolving software issues and completing tasks on websites. These tasks require evaluation setups that account for the environment in which the system operates and the changes it makes.

Lately, research on multi-agent systems has started to add another dimension: how agents coordinate and what they accomplish together. This includes [decentralized swarms](https://arxiv.org/abs/2505.04364), [teams managed by an orchestrator](https://aclanthology.org/2026.findings-emnlp.662/), and [specialized agents collaborating on scientific work](https://arxiv.org/abs/2610.07763). Figure 1 shows a rough timeline of these three eras of evaluations.

<img src="https://huggingface.co/buckets/huggingface/agentic-evals-blog-assets/resolve/images/eval-eras.png" alt="Timeline of three types of language model evaluations: question-and-answer, agentic tasks, and teams and swarms." width="624" style="max-width: 100%; height: auto; background-color: white;">

*Figure 1. Three types of language model evals. Question-and-answer benchmarks check a response against a key. Agentic benchmarks check what an agent changed in an environment. Team and swarm benchmarks score what a group of agents produced.*

We’ll focus on agentic evaluations in this article for two reasons. First, they are becoming increasingly the most prominent type in [model release announcements](https://claude.ai/artifact/22cBPJt6Aos6bYpjmB1BHw#distribution) due to high user and enterprise interest in agentic deployment. Second, they offer a useful starting point for understanding how evaluation design affects model performance. Team and swarm evaluations introduce further complexity through interactions between agents, which we leave outside the scope of this guide. Once an agent uses tools to complete a task, each part of the surrounding software can affect its score. For instance, in an [audit of Terminal-Bench](https://icml.cc/virtual/2026/82793), researchers kept the model fixed and changed the agent harness. Accuracy varied by up to 29.9 percentage points. To understand differences like that, we examine the anatomy of an agentic evaluation and follow one task from the instructions the agent receives to the checks that determine whether it passed.

### The anatomy of an agentic evaluation

<img src="https://huggingface.co/buckets/huggingface/agentic-evals-blog-assets/resolve/images/eval-anatomy.png" alt="Components of an agentic evaluation: task, agent harness, environment, rollout, trace, evaluation harness, and verifier." width="624" style="max-width: 100%; height: auto; background-color: white;">

*Figure 2. The different components of an agentic evaluation.*

There are various parts to an agentic evaluation. A task gives the agent its instruction. The agent harness wraps the model and its tools, letting it work inside a pinned environment. During one rollout, the agent attempts the task from start to finish: it may run several commands, inspect their output, and try again before it stops. The evaluation harness runs several rollouts for each task. When a rollout ends, the verifier grades the environment's final state. Each rollout also leaves a trace, and traces and scores together become part of the final published evidence. Each part can be published, inspected, and reused separately.

### Why benchmark release decisions matter

Demand for independent evaluation is growing. Policymakers, enterprises, and researchers want evaluations they can verify, and today they mostly cannot: [98.2% of model-benchmark scores are reported by only one party](https://arxiv.org/pdf/2606.09809), and researchers have argued that [in-house evaluation is not enough](https://arxiv.org/abs/2503.16861).

At the same time, more benchmarks are holding back test items to reduce contamination: questions or solutions entering training data, so scores may reflect prior exposure. Artificial Analysis [says 60% of its Intelligence Index v5](https://x.com/ArtificialAnlys/status/2115265133467635877), announced in October 2026, rests on evaluations with private test sets or held-out answers. Google DeepMind recently piloted [double-blind evaluations](https://deepmind.google/blog/piloting-the-worlds-first-double-blind-ai-evaluations/), where a secure enclave hides an outside evaluator's test prompts from the model provider and the model's weights from the evaluator. While that can reduce contamination, privacy by itself does not guarantee quality, and withholding supporting materials can make expert review harder. We argue that builders can and should still share infrastructure, grading methods, and evidence about task design. Scrutiny depends on giving others a way to question the measurement and correct mistakes, including controlled review where public release is unsuitable.

## 2. Building an eval: what to reuse, adapt, and validate

With the anatomy in place, we now build one agentic evaluation for a real use case: an agent that analyzes measurements for a physics lab. Rather than starting from scratch, we adapt an open task, so most of our effort goes into domain expertise and validation.

We work through eight steps: [define the claim](#step-1-define-the-claim), [find a task](#step-2-find-a-starting-task), [run it unchanged](#step-3-run-the-task-unchanged), [adapt it](#step-4-adapt-the-task), [check the evaluation](#step-5-check-the-evaluation), [read the traces](#step-6-run-it-and-read-the-traces), [report the score](#step-7-report-the-score), and [compare models](#step-8-compare-models). We then take stock of what we reused. The full walkthrough is in [Appendix A](#appendix-a-practical-example-agentic-deployment-in-a-physics-lab). Figure 3 maps the steps and what each produced in our example.

<img src="https://huggingface.co/buckets/huggingface/agentic-evals-blog-assets/resolve/images/eval-steps.gif" alt="Animated walkthrough of the eight steps of building an agentic evaluation, with what each step produced in the raman-lab example." width="624" style="max-width: 100%; height: auto; background-color: white;">

*Figure 3. The eight steps of building an agentic evaluation. The left panel highlights each step in turn, and the right panel shows what it involved in our example. Yellow steps work on the evaluation itself, and orange steps run models and read the results.*

### Method

**Claim.** Consider a physics lab that wants an agent to fit peaks in a Raman spectrum. A Raman spectrum is a measurement whose peaks reveal properties of a material, here graphene. The agent's job is ordinary coding: read the data file, fit a curve to each peak, and save the results (Figure 5 in Appendix A). For the purposes of this article, our claim is deliberately narrow: fit the specified peaks, calculate the requested D-to-G ratio (the intensity of one peak relative to another, which can indicate defects in the material; we also call it the defect ratio), and save accurate results consistently. Reviewing a whole lab report would need additional skills and evidence.

**Starting task and baseline.** We reused the [Harbor](https://docs.harborframework.com/) task format and Terminal-Bench's [raman-fitting](https://hub.harborframework.com/tasks/terminal-bench/raman-fitting) task. Harbor is an open source evaluation harness, and Terminal-Bench is an open benchmark of tasks that an agent completes in a terminal. We first ran the task's reference solution, which passed, and then ran the original task three times with [Terminus-2](https://docs.harborframework.com/agents/pre-integrated-agents), a terminal agent harness, and [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), an open-weight model from OpenAI. Harbor ran inside an HF Job, each rollout used its own HF Sandbox, and Inference Providers supplied the model responses. Reusing the runner and task format let us concentrate on the physics, grading, and review. Only the agent harness needed no new work. Every other component needed checking or new work, which we list part by part at [the end of Appendix A](#how-we-reused-existing-open-components).

**Adaptation.** We built [raman-lab](https://huggingface.co/datasets/burtenshaw/raman-lab): eight synthetic spectra with known parameters and varied noise and defect levels. The adaptation fits three peaks instead of two and reports a defect ratio. We used an agent to help generate the spectra, reviewed them by hand, and wrote new instructions, reference fits, and output checks. The new instructions state the units and parameter definitions that the original left implicit. These eight tasks are a development set, meaning tasks we use while building the evaluation; broader coverage should follow the variation in real measurements.

**Checks.** Before executing more model runs, we tested the evaluation itself. We ran the reference solution, an agent that does nothing, and a shortcut that skips the required work. The reference solution should pass every task, and the other two should fail. Our shortcut read the height of each peak straight from the data and guessed the widths. It returned the expected defect ratio on all eight tasks without fitting the spectra, so we added checks on the fit itself.

**Shortcut.** We also tested a shortcut to check how robust the evaluation is against cheating. Our shortcut is a short script that skips the actual fitting. It reads the tallest point of each peak straight from the data, guesses the peak widths, and divides the D height by the G height. Even so, it returned the expected defect ratio on all eight tasks, so a grader that checked only the ratio would have given it full marks. We added checks on each fitted peak, and the final verifier rejects it.

### Results

<img src="https://huggingface.co/buckets/huggingface/agentic-evals-blog-assets/resolve/images/what-we-learned.png" alt="Three panels: the shortcut fooled a ratio-only grader on 8 of 8 tasks; gpt-oss-120b's 24 attempts ended as 11 passed, 11 with no results file, and 2 failed fits; pass counts out of 24 for six models." width="624" style="max-width: 100%; height: auto; background-color: white;">

*Figure 4. What we learned by measuring different aspects while building the evaluation. (A) Testing the grader with a shortcut showed that a ratio-only check could be fooled. (B) Reading gpt-oss-120b’s traces showed that most of its failures came from stopping early. (C) Running other models showed that the tasks can be solved.* [*Appendix A*](#appendix-a-practical-example-agentic-deployment-in-a-physics-lab) *has the detailed results in Figure 7 (how gpt-oss-120b’s attempts ended), Figure 8 (all six models) and Figure 9 (when each model said it was done).*

**Baseline.** On the original task, gpt-oss-120b passed 0 of 3 rollouts. In two it declared completion before observing relevant output; in the third it could not read the data file, which uses decimal commas that the reference solution converts before loading. The score alone could not tell these failures from an incorrect fit. The traces could.

**Traces and score.** On the eight adapted tasks, we ran three rollouts per task. The adapted setup passed 11 of 24 rollouts. In 19, the model called `mark_task_complete` in its first response; 11 of those attempts produced no results file. Several of these began with a script failing because NumPy or SciPy was missing, which the task expects the agent to install. Most failures therefore came from stopping early or from a missing package, and only two of the 13 failed the fit check (Figure 7 in Appendix A). Our pass@3 was 7/8: seven tasks succeeded at least once. Our pass^3 was 1/8: one task succeeded every time. Retries may be acceptable when a person reviews the output. Unattended processing needs consistency, so the second measure matters more. The mean pass rate was 0.46, with a standard error of about 0.11 when rollouts are clustered by task. A repeat run passed 10 of 24. With eight development tasks, we would be cautious about generalizing either result to new spectra.

**Model comparison.** We then compared six models on the same eight tasks (Figure 4). Several providers can serve the same open-weight model with different quantization and context limits, so we pinned one provider per model. With DeepInfra pinned as the provider, gpt-oss-120b passed 13 of 24 rollouts, compared with 11 of 24 in the earlier run, which used the default provider. Only gpt-oss-120b declared completion in its first response, in 22 of 24 rollouts. None of the other five models did so in any rollout (Figure 9 in Appendix A). The comparison also shows the limits of the development set. [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), and [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) passed all 24 raman-lab rollouts, so these tasks cannot rank them. That will take harder tasks.

## 3. Bitter lessons: when a benchmark is wrong

Our shortcut returned the expected defect ratio without fitting a spectrum. A verifier that checked only the ratio would have given it full marks on all eight tasks. We found it by trying to break the grader with a solution we knew was incomplete, then added checks on the fitted spectrum ([Appendix A, Step 5](#step-5-check-the-evaluation)).

The shortcut got the right answer for the wrong reason, like a student who copies the final number without the working. The lesson for us was to test what the output demonstrates. A faster valid method should pass; a method that skips a required skill should not. Domain experts can help set tolerances so the verifier accepts alternative correct fits without accepting work that misses the point of the task.

Such errors are common, and they move scores considerably. A [review of agentic benchmarks](https://arxiv.org/abs/2507.02825) found that flaws in task setup or reward design can shift measured performance by up to 100% in relative terms.

### Public review is essential for correcting benchmarks

Public review can expose errors in instructions and tests that benchmark builders missed. Terminal-Bench 2.1 fixed [28 of its 89 tasks](https://www.tbench.ai/news/terminal-bench-2-1) after user reports, including one whose tests expected Spark SQL while its instructions asked for PostgreSQL, and [agent rankings stayed stable](https://icml.cc/virtual/2026/82793) through the correction.

Access to the evidence also determines how much work an outside audit takes. CritPt keeps its reference solutions private, so the [physicists who audited it](https://arxiv.org/abs/2609.13009) had to solve the problems themselves before finding errors in 21 of the 56 challenges they reviewed. The authors then released a [corrected version](https://artificialanalysis.ai/evaluations/critpt-v2). OpenAI's audit of [SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) found material issues in 59.4% of the 138 problems it examined, working from tasks and tests that anyone can read. For a builder, outside scrutiny is part of validation: publish what reviewers need to challenge a task, give them a way to report problems, and record what changed.

Similar flaws emerged in SWE-bench Pro. In July 2026 OpenAI [estimated that about 30% of SWE-Bench Pro tasks were broken](https://openai.com/index/separating-signal-from-noise-coding-evaluations/) and withdrew its recommendation to adopt it. Benchmark validation needs to continue after release.

### Check the claim as well as the code

A consistent score can still measure the wrong thing. [Measuring what Matters](https://arxiv.org/abs/2511.04703), a review of 445 LLM benchmarks by 29 expert reviewers, found recurring patterns in tasks and metrics that undermine construct validity: whether the tasks and metrics support claims about the intended capability.

For example, fitting synthetic spectra does not establish that an agent can review lab reports. That may also require recognizing a poor measurement or asking for help. If every task supplies a usable spectrum, the test cannot show whether the agent knows when a fit would be inappropriate.

[Harder Than the Task](https://arxiv.org/abs/2611.03494) extends this concern to organizational work, where authority, distributed knowledge, and dependencies can decide success. Its authors find that fewer than one quarter of prominent benchmarks claim to cover any such dimension. An agent may need calibration data from another team or permission to replace a measurement, and supplying everything at the outset removes those difficulties. If they matter to the claim, they belong in the evaluation. If they do not, it is enough to say that they remain untested.

Each error is worth a short record: the failed assumption, how it was found, and what changed. The record should also name the affected versions and say whether earlier scores remain comparable.

### Contamination vs saturation

The SWE-bench Verified audit described above also found contamination: every frontier model tested could reproduce the reference patch or verbatim problem details for some tasks. SWE-bench's public repositories and wide use exposed tasks to training. The openness that enabled scrutiny of bad tests also made items easier to encounter before evaluation.

However, if contamination is carefully controlled for during training, hiding the test set by itself does not prevent saturation. A [study of 60 language model benchmarks](https://arxiv.org/abs/2602.16763) found similar saturation in public and private ones, and concluded that hiding test data does not prevent saturation once a benchmark is widely adopted, meaning a good benchmark falls out of use due to saturation without developers being able to learn from and build on it. What did correspond to lower saturation was expert curation and a larger test set.

This leaves a practical release question: what can we expose for review while protecting active test items?

## 4. Open sourcing an evaluation strategically

Balancing contamination risks against saturation requires a strategic approach to what content is made public. Rather than treating openness as an all-or-nothing decision, benchmark creators can decide what to release part by part, protecting active test tasks from data leakage while still sharing the underlying infrastructure, grading methods, and reference examples that allow external researchers to audit and build upon the evaluation.

Existing benchmarks show a range of openness. [ARC-AGI-2](https://arcprize.org/arc-agi/2) publishes a training set and a 120-task evaluation set, and holds back semi-private and private sets of 120 tasks each. [AILuminate](https://mlcommons.org/ailuminate/faq/) keeps its official prompts confidential, publishes its methods and taxonomy, and keeps the underlying evidence for controlled review by authorized auditors and trusted researchers. Terminal-Bench publishes its tasks and marks their files with a [canary string](https://huggingface.co/datasets/harborframework/terminal-bench-2.1/blob/main/tasks/raman-fitting/solution/solve.sh) asking that they stay out of training corpora. No public reviewer can check a protected item, so whatever stays closed needs a review route of its own.

Table 1 outlines the core considerations of open sourcing an evaluation: what can be open, where it lives on the Hugging Face Hub, and what may stay protected. It has one row for each component in Figure 2, so you can go straight to the part you are about to release.

| Part | What can be open | How to share on the Hugging Face Hub | What can stay protected with alternatives for public review |
| --- | --- | --- | --- |
| Task | Specifications, categories, construction methods, and a public development split | A dataset repository, with a [dataset card](https://huggingface.co/docs/hub/datasets-cards) recording the source task, changes, license, and limitations | Active test items and answers, kept in a separate [gated dataset](https://huggingface.co/docs/hub/datasets-gated) with manual approval of each access request. Release retired items where permissible. |
| Environment | Container definition, build files, pinned dependencies, and synthetic fixtures | In the dataset repository, declared as an [environment type](https://huggingface.co/datasets?other=rl-environment&sort=trending) in the dataset card | Private data, credentials, licensed assets, or dangerous details. Document how public substitutes differ. |
| Agent harness and tools | Agent-loop code, tool interfaces, adapters, permissions, and budgets | A link to the harness source and its pinned version, kept in the run record with the configuration used. Give the pinned harness version in the notes field of each `.eval_results` entry, so it’s next to the score | Secrets, private services, and sensitive tool implementations. Provide safe substitutes and authorized inspection. |
| Rollout and trace | Public-task traces, reviewed excerpts, and analysis code | A dataset or storage bucket. Harbor's [trace exporter](https://github.com/harbor-framework/harbor/blob/26106c6d9c66848a1440faf5724d336affac52aa/docs/content/docs/training-workflows/sft.mdx) pushes rollout records, and the Hub's [trace viewer](https://huggingface.co/docs/hub/agent-traces) opens supported session formats. Link scores to these traces through the source field of each `.eval_results` entry | Traces exposing tasks, private data, or restricted model behavior. Offer protected analysis and case inspection. |
| Reference solution | Solution method and code for public examples | The dataset repository, with each task | Solutions that reveal protected items. Let authorized reviewers check them against the task. |
| Evaluation harness | Rollout orchestration, model configuration, logging, score aggregation, and version records | A run record kept with the dataset: the job script, model and harness configuration, and package versions. Inspect writes its logs straight to an `hf://buckets/` path | Credentials and item-identifying configuration. Keep a reviewable record of the exact restricted run. |
| Verifier | Rubrics, grader code, tolerances, and worked examples | The dataset repository, with each task | Reference answers and logic that reveal protected items. Authorized experts should inspect disputed judgments. |
| Claims and task design | Intended use, selection rationale, contextual assumptions, and validation evidence | The dataset card. Bundle the task dataset, trace dataset, any harness Space, and the paper into one Hub Collection | Sensitive source material may need redaction. |
| Results | Scores, uncertainty, costs, exclusions, and limitations, with the model setup and other reproducibility details | Register the dataset as a Hub Benchmark with an `eval.yaml`. Scores go in each model repo's `.eval_results/` folder, opened as a [pull request](https://huggingface.co/docs/hub/eval-results), with source linking to the trace dataset. Optionally, also convert them to the [EEE schema](https://github.com/evaleval/every_eval_ever) | Task-revealing detail in results. Publish safe summaries and arrange review of the underlying evidence. |
| Corrections | A correction history that names the affected versions | [Pull requests and discussions](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) on the repository, with a change log in the card | The protected example behind a correction. Still record which versions it affected. |

*Table 1. What to open, how to share it on the Hugging Face Hub, and what can stay protected, for each evaluation component.*

Our own example follows this approach. We released raman-lab as a [dataset on the Hub](https://huggingface.co/datasets/burtenshaw/raman-lab). It includes eight synthetic graphene spectra in Harbor task format, each with its environment, reference fit, verifier, and tolerances. The raw rollouts behind our results are shared as [a separate dataset](https://huggingface.co/datasets/burtenshaw/raman-lab-runs) including Harbor trial folders with trajectories and verifier output for all six models. The spectrum generator, analysis code, and results are in a companion [GitHub repository](https://github.com/burtenshaw/raman-lab-evals), so both the agent harness and the evaluation harness can be inspected and rerun.

In a more sensitive domain, a clinical evaluation could open its environment definition, harness, tool interfaces, and grading framework, with synthetic cases for development use. Patient records and reference answers would stay under controlled access while authorized reviewers inspect disputed judgments. Each restriction should come with a condition for revisiting it.

### Strategic open sourcing checklist

We recommend a short checklist, drawn in part from the more comprehensive [Agentic Benchmark Checklist](https://arxiv.org/abs/2507.02825). Before releasing or retiring an evaluation, check that another builder can:

- Identify what it measures, its intended use, and the conditions it leaves out.
- Find the reusable artifacts, their licenses and versions, and a way to report problems.
- See how a do-nothing agent scores and, where possible, how a human expert does.
- Read each score with its uncertainty, and with guidance on what known flaws mean for it.
- Learn which flaws were found, how they were handled, and how far they could move the results.
- Know how and when tasks will be refreshed to limit contamination and overfitting.

These release decisions also determine what remains useful when the benchmark no longer separates the systems it was built to compare.

## 5. Conclusion: giving the evaluation a useful afterlife

Benchmarks have a lifecycle. They are built, adopted, optimized against, and in time stop separating the systems they were made to compare. The [saturation study](https://arxiv.org/abs/2602.16763) from section 3 rated 29 of its 60 language model benchmarks as highly saturated, with rates rising with age. Saturation is not a failure in itself, since it can mean the capability has arrived.

What a benchmark leaves behind depends on how open it was. Publishing only scores leaves future builders little they can run or adapt. When the tasks, environments, verifiers, and traces are open, each keeps working after the leaderboard stops. A solved task becomes a regression check for a new harness. Its environment and verifier can seed a harder successor. Open benchmarks also get repaired and reissued: MMLU's questions became [MMLU-Redux](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0), and Terminal-Bench 2.0 became [2.1](https://www.tbench.ai/news/terminal-bench-2-1) through community contributions. Our own walkthrough is an afterlife of this kind: raman-lab began as one task in a benchmark built for general terminal work.

That is the promise of building evaluations openly. The work that goes into an evaluation can outlive the benchmark itself, giving others a foundation to ask new questions, test new capabilities, and improve on what came before. The more we share what we build and learn, the less each new evaluation has to start from scratch.

If you’re curious what this looks like in practice, then follow our practical example.

## Appendix A. Practical example: agentic deployment in a physics lab

Consider a physics lab that wants an agent to fit peaks in a Raman spectrum. Raman spectroscopy shines a laser on a sample and measures shifts in the frequency of the scattered light; the resulting peaks reveal properties of the material. For graphene, an atom-thick sheet of carbon, the G and 2D peaks near 1580 and 2700 cm⁻¹ [identify the number of layers](https://arxiv.org/abs/cond-mat/0606284). A third peak, D, appears when the lattice is disordered, and its intensity relative to G [can indicate defects](https://arxiv.org/abs/1306.5856). Figure 5 shows the setup.

<img src="https://huggingface.co/buckets/huggingface/agentic-evals-blog-assets/resolve/images/raman-setup.png" alt="A laser lights a graphene sample, a spectrometer records the Raman spectrum, the agent fits each peak, and the verifier checks the saved values." width="624" style="max-width: 100%; height: auto; background-color: white;">

*Figure 5. Our intended setup. A laser lights a graphene sample, and a spectrometer records how the scattered light has shifted. The agent receives that spectrum as a file, fits a curve to each peak, and saves the fitted values. The verifier checks the saved values against reference fits. The curve shown is a sketch, not measured data.*

We start by building on a task from an open benchmark. [Terminal-Bench 2](https://huggingface.co/datasets/harborframework/terminal-bench-2.0) already includes [raman-fitting](https://hub.harborframework.com/tasks/terminal-bench/raman-fitting): fit a Lorentzian curve, a mathematical peak shape, to each of those two peaks and save eight parameters to `/app/results.json`. That gives us a runnable starting point for a narrower lab evaluation. We run these on the open source evaluation harness, [Harbor](https://docs.harborframework.com/). Table 2 lists what the task provides and what to check before reusing each part.

| Part | In raman-fitting | Check before reuse |
| --- | --- | --- |
| Task | `instruction.md` asks for two peak fits saved to `/app/results.json`. | Are units, parameter definitions, and the output format explicit? |
| Environment | A pinned container image holding the spectrum; `task.toml` sets resources and limits. | Which packages must the agent install? Are resources and permissions suitable? |
| Agent harness and tools | Terminus-2 connects the model to shell commands. | Can the agent inspect what its commands produced? |
| Rollout and trace | One attempt, recorded in `agent/trajectory.json`. | Does the record show what happened before the agent stopped? |
| Reference solution | `solution/solve.sh` runs the known fit. | Does it pass in the supplied environment? |
| Evaluation harness | Harbor runs rollouts and collects scores and traces. | Are versions and execution settings recorded? |
| Verifier | `tests/test.sh` checks eight fitted values against reference tolerances. | Does it grade the saved file and accept valid alternative fits? |

*Table 2. What to check before reusing each component.*

### Step 1. Define the claim

A good place to start is a *claim*: a plain statement of what the agent can do, which the evaluation is meant to support. "The agent can fit a Raman spectrum" is a claim, and so is "the agent can process a lab report." From there, you can break the claim into smaller skills that can be checked one at a time, such as reading the file, locating the peaks, and writing the output.

For the purposes of this article, our claim is deliberately narrow: fit the specified peaks, calculate the requested D-to-G ratio, and save accurate results consistently. Reviewing a whole lab report would need additional skills and evidence.

Domain experts can help here. We suggest asking them to choose realistic noise levels, tolerances, and available software, then score examples independently and investigate disagreements. From there, you can break the claim into parts you can check and map each part to what each task tests.

### Step 2. Find a starting task

For your own domain, the Hugging Face Hub is a good place to look, under its Benchmark and Environment dataset types. When you find a candidate, it is worth inspecting its instructions, environment, [reference solution](https://huggingface.co/datasets/harborframework/terminal-bench-2.1/blob/main/tasks/raman-fitting/solution/solve.sh), and tests. Even a task that models already solve can provide useful infrastructure. From raman-fitting we inherit a working task format. It has only one spectrum, though, and a lab's measurements vary, so we expand on this task further as described in Step 4.

### Step 3. Run the task unchanged

Before running any model, it helps to run the reference solution (also called oracle). If it fails, the fault lies in the task, environment, or verifier, and needs to be fixed before you read anything into model failures. After that, you can run the model several times without changing the setup.

We ran Harbor inside an [HF Job](https://huggingface.co/docs/hub/jobs-overview), which supplied compute for the runner. Each rollout used its own [HF Sandbox](https://huggingface.co/docs/huggingface_hub/guides/sandbox) for commands and files, and [Inference Providers](https://huggingface.co/docs/inference-providers/index) supplied the model responses. With Harbor's `hf-sandbox` extra installed and `HF_TOKEN` set, these commands test the oracle, then run three rollouts with [Terminus-2](https://docs.harborframework.com/agents/pre-integrated-agents), a terminal agent harness, and [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), an open-weight model from OpenAI.

```bash
harbor run -t terminal-bench/raman-fitting \
  -a oracle -e hf-sandbox

OPENAI_API_KEY="$HF_TOKEN" harbor run \
  -t terminal-bench/raman-fitting -k 3 -e hf-sandbox \
  -a terminus-2 -m openai/openai/gpt-oss-120b \
  --ak api_base=https://router.huggingface.co/v1
```

In these commands, `-t` names the task, `-a` the agent, `-e` the environment, `-m` the model, and `-k` the number of rollouts per task.

The model passed 0 of 3 rollouts. In two it declared completion before observing relevant output; in the third it could not read the data file, which uses decimal commas that the reference solution converts before loading. The score alone could not tell these failures from an incorrect fit. The traces could.

### Step 4. Adapt the task

We built [raman-lab](https://huggingface.co/datasets/burtenshaw/raman-lab): eight synthetic spectra with known parameters and varied noise and defect levels. The adaptation fits three peaks instead of two and reports a defect ratio. We used an agent to help generate the spectra, reviewed them by hand, and wrote new instructions, reference fits, and output checks. These eight tasks are a development set, meaning tasks we use while building the evaluation; broader coverage should follow the variation in real measurements.

The D-to-G intensity ratio can indicate defects, but its interpretation [depends on the laser excitation energy and the disorder regime](https://pubmed.ncbi.nlm.nih.gov/21696186/). An evaluation asking for a "defect ratio" must say which peaks, whether intensity means height or integrated area, and how the result is checked.

The original [instruction.md](https://huggingface.co/datasets/harborframework/terminal-bench-2.1/blob/main/tasks/raman-fitting/instruction.md) shows why such definitions matter: it asks for x0, gamma, amplitude, and offset without defining them or their units. The reference solution treats them as peak center, half-width at half-maximum, height above background, and background level, and converts the x-axis from nanometres to Raman shift. We put choices like these into the adapted instructions, so that the verifier's requirements are visible. Figure 6 shows the two files side by side.

<img src="https://huggingface.co/buckets/huggingface/agentic-evals-blog-assets/resolve/images/instruction-diff.png" alt="The raman-fitting and raman-lab instruction.md files side by side, with the additions highlighted." width="624" style="max-width: 100%; height: auto; background-color: white;">

*Figure 6. The `instruction.md` files for* [*raman-fitting*](https://huggingface.co/datasets/harborframework/terminal-bench-2.1/blob/main/tasks/raman-fitting/instruction.md) *and* [*raman-lab*](https://huggingface.co/datasets/burtenshaw/raman-lab/blob/main/graphene-s01/instruction.md)*, side by side. The highlighted additions cover the new parts of the claim: the D peak, the defect ratio, and both in the saved results. The remaining part, handling varied measurements, lives in the data: all eight raman-lab tasks share this instruction, and only the spectrum, with its noise and defect level, changes.*

The verifier is guided by three rules:

- Grade what the agent produced: inspect the saved output, not its claim to have finished. [τ-bench](https://arxiv.org/abs/2406.12045) does this by comparing the final database state with an annotated goal state.
- State every checked fact in the instruction, including units, parameter conventions, tolerances, and required fields.
- Generate synthetic data from known parameters so the reference answer is available, then review whether the examples represent the intended problem.

The shared format can stay as it is while the data and checks change. This command creates a task scaffold:

```bash
harbor task init my-org/task-01 -p datasets/my-evals
```

### Step 5. Check the evaluation

Before executing more model runs, test the evaluation itself. Here `datasets/my-evals` contains the adapted tasks.

```bash
harbor run -p datasets/my-evals -a oracle -e hf-sandbox
harbor run -p datasets/my-evals -a nop -e hf-sandbox
```

- The oracle should pass every task, i.e., the known solution must work in the supplied environment.
- The no-op agent does nothing, so it should fail every task. A blank exam paper should not pass. A [review of agentic benchmarks](https://arxiv.org/abs/2507.02825) found one that counted empty responses as successes.
- A shortcut that skips required work should fail. One way to test this is with a deliberately lazy solution, such as one that outputs a plausible answer without doing the work, to see whether the verifier accepts it. Also try plausible wrong answers and alternative correct methods, to catch both false passes and false failures.

Our shortcut returned the expected defect ratio on all eight tasks without fitting the spectra, so we added checks on the fit itself. Section 3 explains what that means for verifier design.

### Step 6. Run it and read the traces

It is helpful to read both successful and failed rollouts. Harbor records messages, tool calls, and observations in `agent/trajectory.json` using [ATIF](https://docs.harborframework.com/agents/atif), a shared trace format. Grouping failures by cause makes the patterns easier to see: parsing the data, executing a tool, stopping too early, or grading the output.

The adapted setup passed 11 of 24 rollouts. In 19, the model called `mark_task_complete` in its first response; 11 of those attempts produced no results file.

<img src="https://huggingface.co/buckets/huggingface/agentic-evals-blog-assets/resolve/images/gpt-oss-outcomes.png" alt="Outcomes of gpt-oss-120b's 24 raman-lab attempts, split by whether it called mark_task_complete in its first response." width="624" style="max-width: 100%; height: auto; background-color: white;">

*Figure 7. On 19 of 24 attempts, the agent called `mark_task_complete` in its first response. Eleven produced no results file, seven passed, and one failed the fit check. Of the five remaining attempts, four passed and one failed the fit check.*

Several missing-file failures began with missing NumPy or SciPy. The published task expects the agent to install them; its reference solution starts with a pip install. A lab evaluation of fitting with supplied tools could install and pin them in advance. Dependencies also drift: Terminal-Bench 2.1 [repaired nine tasks](https://www.tbench.ai/news/terminal-bench-2-1) whose external dependencies had changed after release.

Preinstalling changes the evaluation condition, so keep the model, harness, tasks, and rollout count fixed when comparing the two setups.

Finally, we adapt the setup, repeat the checks in step 5, and rerun. Harbor's [trace exporter](https://github.com/harbor-framework/harbor/blob/26106c6d9c66848a1440faf5724d336affac52aa/docs/content/docs/training-workflows/sft.mdx) can prepare records for sharing on the Hub. It is worth reviewing them for protected content first, and keeping both successes and failures so others can investigate the result.

### Step 7. Report the score

The right metric depends on the decision you need to make:

- [pass@k](https://arxiv.org/abs/2107.03374) asks whether at least one of k rollouts succeeds. Our pass@3 was 7/8: seven tasks succeeded at least once.
- [pass^k](https://arxiv.org/abs/2406.12045) asks whether all k rollouts succeed. Our pass^3 was 1/8: one task succeeded every time.

Retries may be acceptable when a person reviews the output. Unattended processing needs consistency, so the second measure matters more.

A mean is more useful with its uncertainty beside it. Rollouts of the same spectrum share its difficulty, so they are not independent, and the standard error should be [clustered by task](https://arxiv.org/abs/2411.00640). Clustering gave a mean success rate of 0.46 with a standard error of about 0.11. A repeat run passed 10 of 24 rollouts, compared with 11 of 24 in the first run. With only eight development tasks, we would be cautious about generalizing either result to new spectra.

Harbor writes a `result.json` per rollout, which it calls a trial. We group these by task, check that each task has three rollouts, and record errors or exclusions. Reporting tokens, elapsed time, and tool calls beside the score lets readers compare setups that reach similar results at different costs.

### Step 8. Compare models

When a single model fails a task, that run alone cannot tell us what went wrong. It could be the model, the agent harness, or the evaluation itself. Running more models on the same tasks helps narrow the possibilities. When several models fail the same task, inspect the instructions, environment, and verifier for a shared cause. When one model repeatedly fails in a distinctive way, compare its traces with successful attempts.

Inference Providers serves many open-weight models through the same router, so each new model is one more value for `-m`. Several providers can serve the same model with different quantization and context limits, so it helps to pin one with a suffix such as `:deepinfra` and record it beside the score.

```bash
OPENAI_API_KEY="$HF_TOKEN" harbor run -p datasets/my-evals -k 3 \
  -e hf-sandbox -a terminus-2 -m openai/Qwen/Qwen3.8-27B:deepinfra \
  --ak api_base=https://router.huggingface.co/v1
```

Figure 8 shows the raman-lab results for each model.

<img src="https://huggingface.co/buckets/huggingface/agentic-evals-blog-assets/resolve/images/model-results.png" alt="raman-lab pass^3, pass@3, and mean pass rate with clustered standard error for six models." width="624" style="max-width: 100%; height: auto; background-color: white;">

*Figure 8. raman-lab results per model. The hollow circle is pass^3, the filled circle is pass@3, and the diamond is the mean with its clustered standard error.*

The comparison makes one pattern in gpt-oss-120b clearer: it often declares completion before seeing command output. In this separate run with DeepInfra pinned as the provider, it called `mark_task_complete` in its first response in 22 of 24 rollouts, while none of the other five models did so (Figure 9). The Step 6 run used the router's default provider and recorded 19 of 24 such early completions. It passed 11 of 24 rollouts, compared with 13 of 24 in the DeepInfra run. Both counts use the same test: whether the model called `mark_task_complete` in its first response.

<img src="https://huggingface.co/buckets/huggingface/agentic-evals-blog-assets/resolve/images/completion-timing.png" alt="raman-lab rollout outcomes per model, split by whether the agent observed command output before declaring completion." width="624" style="max-width: 100%; height: auto; background-color: white;">

*Figure 9. raman-lab rollout outcomes by when the agent declared completion. Left: the agent observed command output first. Right: it called `mark_task_complete` in its first response.*

The comparison also shows the limits of the development set. [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), and [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) passed all 24 raman-lab rollouts, so these tasks cannot rank them. That will take harder tasks.

Inference for all six models cost under $5 at the router's list prices. The median raman-lab rollout used about 6,000 tokens for gpt-oss-120b and about 90,000 for MiniMax-M3.1-Flash. The [companion repository](https://github.com/burtenshaw/raman-lab-evals) holds [the traces](https://huggingface.co/datasets/burtenshaw/raman-lab-runs), the per-rollout results, and the scripts to add another model.

### How we reused existing open components

Reusing the runner and task format let us concentrate on the physics, grading, and review. We could reuse them because Harbor and Terminal-Bench publish their code and tasks.

- **Task:** We reused the Harbor task format and Terminal-Bench's raman-fitting task. The new work was a narrower claim, skill coverage, and instructions for raman-lab.

- **Task data:** The original spectrum served as a starting example. We built a generator for eight spectra with known parameters and reviewed them manually.
- **Environment:** Harbor's environment support and HF Sandbox ran the tasks. We checked package requirements and diagnosed missing dependencies.
- **Agent harness and tools:** Terminus-2 with shell access, with Inference Providers for model responses. This part needed no new work.
- **Rollout and trace:** Harbor's rollout records and ATIF captured each run. We reviewed successes and failures and grouped them by cause.
- **Reference solution:** We kept the convention of running `solution/solve.sh` and wrote a reference fit for each new task.
- **Evaluation harness:** Harbor ran in an HF Job. We configured the run and grouped results by task.
- **Verifier:** We reused the `tests/test.sh` entry point and its way of writing the score to a file. The new work was fit and ratio checks, plus tests for shortcuts and incorrect outputs.

## References
- Akhtar, M., Reuel, A., Soni, P., et al. (2026). When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation. arXiv:2602.16763. <https://arxiv.org/abs/2602.16763>
- Ansari, A., Sun, H., Liu, A. Z., et al. (2026). How Good Are Frontier Models at Physics? Expert Re-Grading Reveals Broken Evaluations and Near-Saturation of Leading Benchmarks. arXiv:2609.13009. <https://arxiv.org/abs/2609.13009>
- Anthropic. (2026). Demystifying evals for AI agents. Anthropic Engineering, January 2026. <https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents>
- ARC Prize Foundation. (n.d.). ARC-AGI-2. <https://arcprize.org/arc-agi/2>
- Artificial Analysis. (2026). Announcing Artificial Analysis Intelligence Index v5. Post on X, October 2026. <https://x.com/ArtificialAnlys/status/2115265133467635877>
- Barres, V., Barton, A., Foss, N. J., & Nielsen, H. M. (2026). Harder Than the Task: From Technical to Organizational Difficulty in AI Benchmarking. arXiv:2611.03494. <https://arxiv.org/abs/2611.03494>
- Bean, A. M., Kearns, R. O., Romanou, A., et al. (2025). Measuring what Matters: Construct Validity in Large Language Model Benchmarks. arXiv:2511.04703. <https://arxiv.org/abs/2511.04703>
- Buchanan, E. K., Linderman, S., & Ré, C. (2026). Benchmarks as Software: A Case Study on Terminal-Bench. ICML 2026. <https://icml.cc/virtual/2026/82793>
- Cançado, L. G., Jorio, A., Ferreira, E. H., et al. (2011). Quantifying defects in graphene via Raman spectroscopy at different excitation energies. Nano Letters, 11(8), 3190–3196. <https://pubmed.ncbi.nlm.nih.gov/21696186/>
- Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374. <https://arxiv.org/abs/2107.03374>
- Ferrari, A. C., & Basko, D. M. (2013). Raman spectroscopy as a versatile tool for studying the properties of graphene. Nature Nanotechnology, 8, 235–246. <https://arxiv.org/abs/1306.5856>
- Ferrari, A. C., Meyer, J. C., Scardaci, V., et al. (2006). Raman spectrum of graphene and graphene layers. Physical Review Letters, 97, 187401. <https://arxiv.org/abs/cond-mat/0606284>
- Gema, A. P., Leang, J. O. J., Hong, G., et al. (2024). Are We Done with MMLU? arXiv:2406.04127. <https://arxiv.org/abs/2406.04127>
- Ghosh, A., Reuel, A., Chim, J., et al. (2026). Evaluation Cards: An Interpretive Layer for AI Evaluation Reporting. arXiv:2606.09809. <https://arxiv.org/abs/2606.09809>
- Harbor Framework. (n.d.). Harbor documentation. <https://docs.harborframework.com/>
- Isaac, W., Messing, S., & Lum, K. (2026). Piloting the world's first double-blind AI evaluations. Google DeepMind, August 2026. <https://deepmind.google/blog/piloting-the-worlds-first-double-blind-ai-evaluations/>
- Longpre, S., Klyman, K., Appel, R. E., et al. (2025). In-House Evaluation Is Not Enough: Towards Robust Third-Party Flaw Disclosure for General-Purpose AI. arXiv:2503.16861. <https://arxiv.org/abs/2503.16861>
- Miller, E. (2024). Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640. <https://arxiv.org/abs/2411.00640>
- MLCommons. (n.d.). AILuminate: frequently asked questions. <https://mlcommons.org/ailuminate/faq/>
- OpenAI. (2026). Why we no longer evaluate SWE-bench Verified. February 2026. <https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/>
- OpenAI. (2026). Separating signal from noise in coding evaluations. July 2026. <https://openai.com/index/separating-signal-from-noise-coding-evaluations/>
- Paniego, S., & Gosthipaty, A. R. (2026). Harness, Scaffold, and the AI Agent Terms Worth Getting Right. Hugging Face blog, May 2026. <https://huggingface.co/blog/agent-glossary>
- Skarlinski, M., Laurent, J., Bou, A., & White, A. (2025). About 30% of Humanity's Last Exam chemistry/biology answers are likely wrong. FutureHouse, July 2025. <https://www.futurehouse.org/research/hle-exam>
- Terminal-Bench. (2026). Terminal-Bench 2.1. Release notes, May 2026. <https://www.tbench.ai/news/terminal-bench-2-1>
- Wei, K. (2026). A primer on AI evaluations for policymakers: Part I. Matters of Scale, November 2026. <https://ai.scal.ing/p/evals-intro-1>
- XLANG Lab. (2025). Introducing OSWorld-Verified. July 2025. <https://xlang.ai/blog/osworld-verified>
- Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045. <https://arxiv.org/abs/2406.12045>
- Zhu, Y., Jin, T., Pruksachatkun, Y., et al. (2025). Establishing Best Practices for Building Rigorous Agentic Benchmarks. arXiv:2507.02825. <https://arxiv.org/abs/2507.02825>

## Glossary

**Agent harness:** The software around a model that turns its responses into actions, such as shell commands. Terminus-2 is the agent harness in this post.

**Agentic evaluation:** An evaluation that scores what an agent changed in an environment, not only the text it returned.

**ATIF:** Agent Trajectory Interchange Format, a shared format for recording an agent's messages, tool calls and observations.

**Canary string:** A unique marker in benchmark files that lets model trainers find and exclude them from training data.

**Clustered standard error:** A standard error that treats rollouts of the same task as related, which gives a wider and more honest margin than treating each rollout as independent.

**Container image:** A fixed package of operating system, files and software, so every rollout starts from the same environment.

**Contamination:** Test questions or solutions appearing in training data, so a score may reflect memory rather than skill.

**Context limit:** The maximum amount of text, counted in tokens, that a model can read in one request.

**D, G and 2D peaks:** Features in graphene's Raman spectrum. G and 2D indicate the number of layers, and D appears when the carbon lattice has defects.

**Defect ratio:** The intensity of the D peak divided by that of the G peak, used to indicate defects in graphene. Also called the D-to-G ratio.

**Development set:** Tasks used while building and debugging an evaluation, as opposed to a held-back test set used for final scores.

**Environment:** The software, files and resources an agent works in during a task.

**Evaluation harness:** The software that runs tasks, repeats rollouts, collects traces and aggregates scores. Harbor is the evaluation harness in this post.

**Gated dataset:** A Hugging Face dataset that can be downloaded only after the owner approves an access request.

**Harbor:** An open-source evaluation harness for agentic tasks.

**HF Job:** Compute on Hugging Face that runs a script. Here it runs Harbor.

**HF Sandbox:** An isolated Hugging Face environment where a single rollout runs its commands and stores its files.

**Held-out test set:** Test items kept private so that models cannot be trained or tuned on them.

**Inference Providers:** A Hugging Face service that routes model requests to third-party providers through one API, called the router.

**Lorentzian:** A mathematical curve commonly used to model the shape of a spectral peak.

**No-op agent:** An agent that does nothing. A sound evaluation gives it a score of zero.

**Open-weight model:** A model whose trained parameters are published for anyone to download and run.

**Oracle:** An agent that runs the reference solution, used to check that every task can be solved in the supplied environment.

**pass@k:** The share of tasks where at least one of k rollouts succeeds.

**pass^k:** The share of tasks where all k rollouts succeed.

**Pinning:** Fixing the exact versions of software, images or model providers so a run can be repeated.

**Quantization:** Storing model weights at lower numerical precision to run faster and cheaper, which can change outputs slightly.

**Raman shift:** The difference in frequency between the laser light and the light a sample scatters, measured in cm⁻¹.

**Raman spectroscopy:** A technique that shines a laser on a sample and measures shifts in the scattered light to identify properties of the material.

**Reference solution:** A known correct way to complete a task, stored with the task.

**Rollout:** One attempt by an agent at one task. Harbor calls it a trial.

**Run record:** A record of what ran behind a score: the exact task, environment, model configuration and code versions.

**Saturation:** The point where most models reach top scores on a benchmark, so it no longer separates them.

**Synthetic data:** Data generated from known parameters, so the correct answer is known in advance.

**Terminal-Bench:** An open benchmark of tasks that an agent completes in a computer terminal.

**Terminus-2:** The terminal agent harness used in this post. It connects the model to shell commands.

**Token:** The unit of text a model reads and writes, roughly a short word or part of a word.

**Trace:** The full record of a rollout: the model's messages, its tool calls and what each command returned. Also called a trajectory.

**Verifier:** The code that grades a rollout by checking the saved output or the environment's final state.
