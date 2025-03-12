---
title: "Introducting HELMET: Holistically Evaluating Long-context Language Models" 
thumbnail: /blog/assets/helmet/logo.jpeg
authors:
- user: hyen
  guest: true
  org: princeton-nlp

---

<h1 class="subtitle is-3 publication-subtitle">
  <span>Introducing  <img src="./assets/helmet/logo.jpeg" alt="logo" width="30"/><span style="color: #E77500"><b>HELMET</b></span>: Holistically Evaluating Long-context Language Models</span>
</h1>

By Howard Yen<sup><span style="color: #E77500">&spades;</span></sup>, 
Tianyu Gao<sup><span style="color: #E77500">&spades;</span></sup>, 
Minmin Hou<sup><span style="color: #00C7FD">&clubs;</span></sup>, 
Ke Ding<sup><span style="color: #00C7FD">&clubs;</span></sup>, 
Daniel Fleischer<sup><span style="color: #00C7FD">&clubs;</span></sup>, 
Peter Izsak<sup><span style="color: #00C7FD">&clubs;</span></sup>, 
Moshe Wasserblat<sup><span style="color: #00C7FD">&clubs;</span></sup>, 
and Danqi Chen<sup><span style="color: #E77500">&spades;</span></sup>\
<sup><span style="color: #E77500">&spades;</span></sup>Princeton Language and Intelligence (PLI), Princeton University\
<sup><span style="color: #00C7FD">&clubs;</span></sup>Intel\
2025-02-29

Correspondence: hyen@cs.princeton.edu \
Paper: https://arxiv.org/abs/2410.02694 \
Code & Data: https://github.com/princeton-nlp/HELMET \
Website: https://princeton-nlp.github.io/HELMET 

- [Evaluating long-context language models is challenging but important](#evaluating-long-context-language-models-is-challenging-but-important)
- [Existing evaluations overly rely on synthetic tasks](#existing-evaluations-overly-rely-on-synthetic-tasks)
- [Crafting diverse, controllable, and reliable evaluation for LCLMs](#crafting-diverse-controllable-and-reliable-evaluation-for-lclms)
  - [Key improvements over existing benchmarks](#key-improvements-over-existing-benchmarks)
- [LCLMs still have a long way to go on real-world tasks](#lclms-still-have-a-long-way-to-go-on-real-world-tasks)
  - [Diverse long-context applications call for diverse evaluation](#diverse-long-context-applications-call-for-diverse-evaluation)
  - [Model performance across tasks and lengths](#model-performance-across-tasks-and-lengths)
- [Using HELMET for future developments](#using-helmet-for-future-developments)
  - [Diverse domains](#diverse-domains)
  - [Avoid running expensive baselines](#avoid-running-expensive-baselines)
  - [Future works](#future-works)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

Since we first released HELMET last October, there has been more development on long-context language models than ever before, and we are thrilled to see the adoption of HELMET by the community, such as [Microsoft's Phi-4](https://arxiv.org/abs/2412.08905) and [AI21's Jamba 1.6](https://www.ai21.com/blog/introducing-jamba-1-6/).
After the initial release, we have added more models to our evaluation suite and conducted additional analyses. We are excited to share our new results and present HELMET at ICLR 2025!

In this blog, we will describe the construction of HELMET, our key findings, and how practitioners can use HELMET to differentiate between various LCLMs in future research and applications.

## Evaluating long-context language models is challenging but important

From summarizing numerous legal documents to learning new tasks on the fly, long-context language models (LCLMs) have immense potential to change the way we use and interact with language models.
Traditionally, language models have been limited by their context window, which is typically around 2k tokens (e.g., [GPT-3](https://arxiv.org/abs/2005.14165)).
Recently, model developers have been constantly increasing the context window of their models, with recent models like [GPT-4o](https://openai.com/index/hello-gpt-4o/), [Claude](https://www.anthropic.com/news/claude-3-family), and [Gemini](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#ethics-safety) supporting context windows of up to millions of tokens.

<figure>
  <img src="./assets/helmet/teaser.png" alt="logo" width="800"/>
  <figcaption>Figure 1: Existing benchmarks show counterintuitive trends, such as smaller models outperforming larger ones.</figcaption>
</figure>

However, despite the increasing interest in LCLMs, model developers often evaluate on different datasets, which makes it difficult to compare various models.
With longer context windows, previous natural language benchmarks are no longer suitable for evaluating LCLMs.
Consequently, perplexity and synthetic tasks (e.g., needle-in-a-haystack) emerged as the most popular evaluation metrics for LCLMs, but they often **do not reflect real-world performance**.
Furthermore, existing benchmarks for LCLMs may show confusing and counterintuitive results, making it difficult to understand the strengths and weaknesses of different models (Figure 1).
<!-- For downstream users, it may be difficult to choose the right model for their applications, as the evaluation settings between different models are *inconsistent* and often *not reflective of real-world applications*. -->

In this work, we propose HELMET (How to Evaluate Long-Context Models Effectively and Thoroughly), a comprehensive benchmark for evaluating LCLMs that improves upon existing benchmarks in several ways—*diversity, controllability, and reliability*.
We evaluate 59 recent LCLMs and find that simple synthetic tasks, such as needle-in-a-haystack, do not reflect real-world performance, and it is crucial to evaluate models across diverse applications to understand their capabilities.


<!-- Since the initial release, model developers have adopted HELMET for evaluating their models, such as [Microsoft's Phi-4](https://arxiv.org/abs/2412.08905), and we hope that HELMET will be useful for future development of LCLMs. -->

<!-- We evaluate over 50 recent models on diverse, application-centric tasks, which enables researchers and practitioners to compare models across different axes. -->

<!-- However, existing benchmarks for long-context language modeling primarily rely on either perplexity and synthetic tasks, such as needle-in-a-haystack, even though it is unclear how well the performance on these tasks would transfer to real-world applications.  -->

<!-- In this work, we propose HELMET (How to Evaluate Long-Context Models Effectively and Thoroughly), a comprehensive benchmark for evaluating LCLMs.
In contrast to previous benchmarks, HELMET is designed to include diverse, application-centric tasks, complemented with reliable evaluation settings. 
We evaluate over 50 recent models, enabling detail comparisons and understanding of existing models and architectures across diverse axes.
Our experiments reveal key findings: (1) synthetic tasks like needle-in-a-haystack (NIAH) do not reflect real-world performance, (2) diverse types of tasks exhibit distinct trends, and (3) open-source models still lag behind proprietary models on more complex tasks.
Ultimately, we advocate for a holistic evaluation across diverse tasks. -->

## Existing evaluations overly rely on synthetic tasks

With the development of LCLMs across both industry and the open-source community, it is crucial to have a reliable method for evaluating and comparing these models. However, current models are *often evaluated on different benchmarks* (Table 1).

<figure>
  <img src="./assets/helmet/model_eval_comparison.png" alt="logo" width="800"/>
  <figcaption>Table 1: Model developers often evaluate on different sets of datasets. <sup>♭</sup>: Base models. NQA: NarrativeQA, Qspr: Qasper, QALT: QuALITY, SQALT:SQuALTY.</figcaption>
</figure>

A common practice for evaluating long-context language models is to use perplexity or synthetic tasks, such as needle-in-a-haystack (NIAH). However, recent works have shown that perplexity does not correlate well with downstream performance ([Fang et al., 2024](https://arxiv.org/abs/2410.23771)). In our work, we show that synthetic tasks like NIAH do not correlate with real-world performance (Figure 2).

<figure>
  <img src="./assets/helmet/correlation_syn.png" alt="syn" width="500"/>
  <figcaption>Figure 2: Simple synthetic tasks, such as NIAH, do not correlate well with downstream tasks, such as summarization or generation with citations.</figcaption>
</figure>

\
Among the existing benchmarks with realistic applications, such as ZeroScrolls ([Shaman et al., 2023](https://arxiv.org/abs/2308.14508)), LongBench ([Bai et al., 2024](https://arxiv.org/abs/2308.14508)), and InfiniteBench ([Zhang et al., 2024](https://arxiv.org/abs/2402.13718)), there are still crucial limitations:

- Insufficient coverage of downstream tasks: often focused on specific domains
- Inadequate lengths for testing frontier LCLMs: context lengths < 128K tokens
- Unreliable metrics: N-gram matching metrics like ROUGE are noisy
- Incompatibility with base models: require instruction-tuning

Thus, we propose HELMET to address these limitations and provide a comprehensive evaluation of LCLMs.

## Crafting diverse, controllable, and reliable evaluation for LCLMs

We design HELMET with the following desiderata:
1. Diverse coverage of downstream tasks
2. Controllable length and complexity
3. Reliable evaluation for base and instruction-tuned models

Table 2 shows an overview of the benchmark.
In our experiments, we evaluate on input length from 8K to 128K tokens, but HELMET can be easily extended to even longer context lengths.

<figure>
  <img src="./assets/helmet/data_summary.png" alt="logo" width="800"/>
  <figcaption>Table 2: Overview of HELMET datasets. SubEM: Substring Exact Match.</figcaption>
</figure>

### Key improvements over existing benchmarks

***Diverse coverage***: HELMET includes a diverse set of tasks, such as retrieval-augmented generation with real retrieval passages, generation with citations, and summarization. We carefully select datasets with naturally long contexts that reflect real-world applications. These datasets are complemented with reliable evaluation settings, such as model-based evaluations and human studies.

***Reliable evaluation***: Many existing benchmarks still use n-gram-based metrics, such as ROUGE, despite their poor correlation with human judgments ([Goyal et al., 2023](https://arxiv.org/abs/2209.12356)). We employ model-based evaluations that show better distinguishability between models and different input lengths (Figure 3). Furthermore, our human studies show that our metrics have a high agreement with human judgments.

<figure>
  <img src="./assets/helmet/model_eval.png" alt="logo" width="800"/>
  <figcaption>Figure 3: ROUGE cannot differentiate between models and lengths, while model-based evaluations are better at separating models of different capacities.</figcaption>
</figure>

***Robust prompting***: Existing long-context benchmarks often require models to follow instructions, but many model developments revolve around base models, which have to rely on synthetic tasks or perplexity for evaluation. Thus, we support base models for a subset of our tasks via in-context learning examples. This substantially improves the performance of base models, which is more reflective of real-world applications.

***Controllable length and difficulty***: An important dimension to consider when evaluating LCLMs is the input length, as longer inputs can provide more information while challenging the model's ability to process noisy contexts. In our tasks, we can control the input length by changing the number of retrieved passages (RAG, Cite, Re-rank), the number of demonstrations (ICL), or the length of the input document (LongQA, Summ). Although LongQA and Summ cannot be easily extended to longer contexts, we intentionally chose datasets with natural documents of length far greater than 100K tokens, such that they can still be used to evaluate frontier LCLMs.

## LCLMs still have a long way to go on real-world tasks

Our experiments and analyses include a comprehensive set of 59 LCLMs. To our knowledge, this is the most thorough and controlled comparison of long-context models on diverse applications. These models cover both leading proprietary and open-source models, and we also consider models with different architectures (e.g., full-attention transformers, hybrid architectures) and positional extrapolation techniques. In this section, we will highlight a few key findings from our experiments.

### Diverse long-context applications call for diverse evaluation

Long-context benchmarks are often constructed with specific applications in mind, such as summarization or question answering, which limits the understanding of LCLMs in a broader context. We examine model performance over a wide range of real tasks and find that different categories do not always correlate with each other (Figure 4).

<figure>
   <img src="./assets/helmet/correlation_category_inst.png" alt="syn" width="500"/>
   <figcaption>Figure 4: Different categories do not correlate well with each other.</figcaption>
</figure>

While some tasks moderately correlate with each other (e.g., RAG and MS-MARCO) due to their retrieval-based nature, others show little correlation (e.g., Summ and Cite). Notably, ICL has the lowest correlation with other tasks, which suggests that it is a unique task that requires different capabilities from the model. Therefore, model developers should evaluate across these distinct axes to draw a more holistic picture of the model's capabilities.

### Model performance across tasks and lengths

We present the results of the frontier proprietary models as well as a few open-source models on HELMET.
Additional results can be found in the paper and the website.

<figure>
   <img src="./assets/helmet/results_length_main.png" alt="syn" width="800"/>
   <figcaption>Figure 5: HELMET results on selected instruction-tuned models across tasks and input lengths.</figcaption>
</figure>

First, we observe that **open-source models lag behind closed-source models on complex tasks**. Although the gap appears small on simpler tasks, such as Recall, the gap widens on more complex ones, such as Cite.

Furthermore, **performance degradation with increasing lengths is category-dependent**. Even the most advanced models, such as GPT-4o and Gemini, experience a significant decrease in performance on tasks like re-ranking. This change in performance cannot be observed from simply looking at the synthetic task performance.

Finally, **there is no clear winner across all categories**, thereby calling for evaluation across different axes. Additional analysis, such as the performance of different positional extrapolation methods and the lost-in-the-middle phenomenon, can be found in the paper.

## Using HELMET for future developments

Using HELMET is easy! Simply clone our [GitHub repository](https://github.com/princeton-nlp/HELMET), and everything is ready to go after setting up the environment; see the repo for more details! There are many practical reasons to use HELMET as well:

### Diverse domains
With HELMET, practitioners can easily choose the right model for their applications by comparing models across diverse tasks. Given the increasing interest in LCLMs for both applications and other research fields, we hope that HELMET will be a useful tool for the community.

### Avoid running expensive baselines
It is often expensive to run all the baselines for evaluating LCLMs, especially at long contexts given their computational and memory costs. For example, running HELMET at all lengths on a 70B model requires a node with 8 * 80GB GPUs for hundreds of GPU hours, which can be costly. By evaluating on HELMET, researchers can directly compare their models to existing ones simply by referencing our results, which cover 59 models of different sizes and architectures.

### Future works

HELMET is a step towards a more comprehensive evaluation of long-context language models, but there are still many more exciting applications of LCLMs. For example, we recently released [LongProc](https://arxiv.org/abs/2501.05414), a benchmark for evaluating LCLMs on *long-form generation* and *following procedures*. 
Although summarization tasks have long outputs (up to 1K tokens), LongProc focuses on even longer outputs, up to 8K tokens. Similar to HELMET, LongProc is also designed with reliable evaluation settings and diverse tasks. We are working on integrating LongProc into HELMET's evaluation suite, and we hope that this will provide a more comprehensive evaluation of LCLMs on long-form tasks.

## Acknowledgements

We thank Mengzhou Xia, Howard Chen, Xi Ye, Yinghui He, Lucy He, Alexander Wettig, Sadhika Malladi, Adithya Bhaskar, Joie Zhang, and other members of the Princeton Language and Intelligence (PLI) group for their helpful feedback.
This work is gratefully supported by the Microsoft Accelerate Foundation Models Research (AFMR) for Azure OpenAI credits and an Intel grant.

## Citation

If you find HELMET useful, please consider citing our paper:

```
@inproceedings{yen2024helmet,
      title={HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly}, 
      author={Howard Yen and Tianyu Gao and Minmin Hou and Ke Ding and Daniel Fleischer and Peter Izsak and Moshe Wasserblat and Danqi Chen},
      year={2025},
      booktitle={International Conference on Learning Representations (ICLR)},
}
```

