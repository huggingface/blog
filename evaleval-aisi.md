---
title: "How UK AISI and EvalEval Are Making Benchmark Results Reproducible"
thumbnail: /blog/assets/evaleval-aisi/thumbnail.png
authors:
- user: evijit
  org: evaleval
- user: j-chim
  guest: true
  org: evaleval
- user: deeplumiere
  guest: true
  org: evaleval
- user: srishtiy
  guest: true
  org: evaleval
- user: wmmkennedy
  guest: true
  org: evaleval
- user: irenesolaiman
  org: evaleval
- user: mcfadyen-aisi
  guest: true
  org: ai-safety-institute
- user: lynn-aisi
  guest: true
  org: ai-safety-institute
- user: coz-aisi
  guest: true
  org: ai-safety-institute
---

# How UK AISI and EvalEval Are Making Benchmark Results Reproducible

The [EvalEval Coalition](https://evalevalai.com/) is thrilled to share that the [UK AI Security Institute (AISI)](https://www.aisi.gov.uk/) is using EvalEval's infrastructure to openly share evaluation results, supporting more reproducible and verifiable evaluation science.

AISI and EvalEval have previously collaborated on research that began at a [joint workshop alongside NeurIPS 2025](https://evalevalai.com/events/workshop-2025/), and feedback from the Institute has helped shape the [Every Eval Ever (EEE) schema](https://evalevalai.com/projects/every-eval-ever/). This next phase of the collaboration puts that shared infrastructure into practice.

## Why reproducible evaluation reporting matters

As AI deployment accelerates, evaluations are becoming increasingly important sources of evidence about model and system performance. Yet results are reported across many formats, platforms, and outlets, often without enough information to reproduce them. Running the evaluations again may itself be prohibitively expensive.

EvalEval's mission is to improve this ecosystem through a shared reporting schema, [Every Eval Ever](https://evalevalai.com/projects/every-eval-ever/), and an open platform, [Evaluation Cards](https://evalcards.evalevalai.com/), that brings evaluation results and the information needed to interpret them into a common structure.

This builds naturally on AISI's work to make evaluation more efficient through [OptStop](https://arxiv.org/abs/2608.14425), more statistically rigorous through [HiBayES](https://www.aisi.gov.uk/blog/hibayes-improving-llm-evaluation-with-hierarchical-bayesian-modelling), and more standardised in areas including transcript analysis and capability elicitation. Together, AISI and EvalEval are working to diagnose gaps in evaluation reporting and build shared infrastructure to close them.

## What AISI is sharing

Transcript-level transparency matters not only for reproducibility, but also for analysis and diagnosis. In this new phase of the collaboration, AISI is making publicly reported evaluation methods and findings available through Evaluation Cards where appropriate. The release includes verified results, context, and configuration information for the five benchmarks in the paper's main experiment:

- HealthBench
- FrontierMath
- Humanity's Last Exam
- SWE-Bench Pro
- Terminal-Bench 2.0

These results cover six frontier models: Claude Opus 4, Claude Opus 4.5, Claude Opus 4.6, GPT-5, GPT-5.2, and GPT-5.4. The release also includes results from two related cyber evaluations—Cyber CTFs and The Last Ones—which use a different, partially overlapping set of models. The data accompany AISI's paper, [*How Inference Compute Shapes Frontier LLM Evaluation*](https://arxiv.org/abs/2606.17930), which studies how benchmark performance depends on inference-time compute and evaluation protocol.

<iframe src="https://evaleval-general-eval-card.hf.space/embed/eval/trajectories/aisi-inference-scaling/hle?panel=tokens" frameborder="0" width="100%" height="670px"></iframe>

*Performance on Humanity's Last Exam changes with evaluation protocol and inference compute. Each curve shows the cumulative share of attempted tasks solved within a given token count, using the earliest observed success per task. When models received correctness feedback from an oracle after each attempt, they continued to solve additional tasks as token use increased.*

When results are openly released with setup information, researchers and practitioners can examine individual studies more closely and compare findings across the wider ecosystem. Where other reports lack these details, releases like AISI's provide verified reference points for interpreting evaluations in context—for example, by helping researchers understand how setup choices may influence reported performance. As more evaluators adopt EEE, open comparisons like these can support broader and more reliable meta-research.

<iframe src="https://evaleval-general-eval-card.hf.space/embed/eval/distribution/aisi-inference-scaling/terminal-bench-2?view=context" frameborder="0" width="100%" height="500px"></iframe>

*AISI's Terminal-Bench 2.0 results alongside other reported evaluations for the same models, under different evaluation setups.*

We are excited about this adoption and look forward to further standardising and sharing evaluations with AISI and other AI evaluation organisations.

## Contribute to the shared mission

- **Model developers:** [Report verified evaluation results](https://evalcards.evalevalai.com/help/get-verified).
- **Evaluation developers:** Report benchmarks and run data using the [Every Eval Ever schema](https://github.com/evaleval/every_eval_ever).
- **Evaluation, governance, and policy researchers:** [Explore Evaluation Cards](https://evalcards.evalevalai.com/) by benchmark or model, or use it to examine the state of evaluation reporting as a whole.

## About the EvalEval Coalition

The EvalEval Coalition is a research community developing scientifically grounded research and robust deployment infrastructure for the evaluation ecosystem. Its goal is to improve evaluation science, address the lack of consensus around documenting evaluation applicability and utility, and broaden coverage of the impacts that matter for scientific research and policy analysis.

The coalition's flagship projects include [Every Eval Ever](https://evalevalai.com/projects/every-eval-ever/), a shared schema and repository for evaluation results, and [Evaluation Cards](https://evalevalai.com/projects/eval-cards/), which combines benchmark metadata, evaluation-run data, and model metadata into interpretable records. Together, they make it easier to understand when apparently similar scores were produced under meaningfully different conditions.

## About the UK AI Security Institute

The [UK AI Security Institute](https://www.aisi.gov.uk/) is a research organisation within the UK government's Department for Science, Innovation and Technology. Its mission is to equip governments with a scientific understanding of the risks posed by advanced AI. AISI conducts research and builds infrastructure to understand advanced AI capabilities and impacts, develop and test mitigations, and inform policy.

## Further reading

- [*How Inference Compute Shapes Frontier LLM Evaluation*](https://arxiv.org/abs/2606.17930)
- [HiBayES: Improving LLM evaluation with hierarchical Bayesian modelling](https://www.aisi.gov.uk/blog/hibayes-improving-llm-evaluation-with-hierarchical-bayesian-modelling)
- [HiBayES paper](https://arxiv.org/abs/2505.05602)
- [OptStop paper](https://arxiv.org/abs/2608.14425)
- [Every Eval Ever](https://evalevalai.com/projects/every-eval-ever/)
- [Evaluation Cards](https://evalcards.evalevalai.com/)
