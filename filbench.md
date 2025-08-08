---
title: "🇵🇭 FilBench - Can LLMs Understand and Generate Filipino?"
thumbnail: /blog/assets/filbench/thumbnail.png
authors:
  - user: ljvmiranda921
    guest: true
    org: UD-Filipino
  - user: acocodes
    guest: true
    org: UD-Filipino
  - user: connermanuel
    guest: true
    org: UD-Filipino
  - user: jcblaise
    guest: true
    org: UD-Filipino
  - user: jcblaise
    guest: true
    org: SEACrowd
  - user: josephimperial
    guest: true
    org: SEACrowd
  - user: davanstrien
    guest: false
  - user: SaylorTwift
    guest: false
  - user: clefourrier
    guest: false
---

# 🇵🇭 FilBench - Can LLMs Understand and Generate Filipino?

As large language models (LLMs) become increasingly integrated into our lives, it becomes crucial to assess whether they reflect the nuances and capabilities of specific language communities.
For example, Filipinos are among the most active ChatGPT users globally, ranking fourth in ChatGPT traffic (behind the United States, India, and Brazil [[1](https://blogs.worldbank.org/en/digital-development/who-on-earth-is-using-generative-ai-)] [[2](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4715603)]), but despite this strong usage, we lack a clear understanding of how LLMs perform for their languages, such as Tagalog and Cebuano.
Most of the existing evidence is anecdotal, such as screenshots of ChatGPT responding in Filipino as proof that it is fluent.
What we need instead is a systematic evaluation of LLM capabilities in Philippine languages.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/filbench/filbench-intro.png" style="width: 100%;"/>

That’s why we developed FilBench: a comprehensive evaluation suite to assess the capabilities of LLMs for Tagalog, Filipino (the standardized form of Tagalog), and Cebuano, on fluency, linguistic and translation abilities, as well as specific cultural knowledge.

We used it to evaluate 20+ state-of-the-art LLMs on FilBench, providing a comprehensive assessment of their performance in Philippine languages:

<iframe
	src="https://ud-filipino-filbench-leaderboard.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

- 📄 Paper: https://arxiv.org/abs/2508.03523
- 🖥️ GitHub: https://github.com/filbench/filbench-eval

## FilBench

The FilBench evaluation suite contains four major categories–Cultural Knowledge, Classical NLP, Reading Comprehension, and Generation–divided into 12 tasks.
For example, the Classical NLP category includes tasks such as sentiment analysis, whereas Generation tasks include different aspects of translation.
In order to ensure that these categories reflect the priorities and trends in NLP research and usage, we curate them based on a historical survey of NLP research on Philippine languages from 2006 to early 2024.
(Most of these categories exclusively contain non-translated content to ensure faithfulness to the natural use of Philippine languages.)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/filbench/filbench-main.png" style="width: 100%;"/>

- **Cultural Knowledge:** This category tests a language model's ability to recall factual and culturally specific information. For Cultural Knowledge, we curated a variety of examples that test an LLM's regional and factual knowledge (Global-MMLU), Filipino-centric values (KALAHI), and ability to disambiguate word sense (StingrayBench).
- **Classical NLP:** This category encompasses a variety of information extraction and linguistic tasks, such as named entity recognition, sentiment analysis, and text categorization, that specialized, trained models traditionally performed. In this category, we include instances from CebuaNER, TLUnified-NER, and Universal NER for named entity recognition, and subsets of SIB-200 and BalitaNLP for text categorization and sentiment analysis.
- **Reading Comprehension:** This category evaluates a language model's ability to understand and interpret Filipino text, focusing on tasks such as readability, comprehension, and natural language inference. For this category, we include instances from the Cebuano Readability Corpus, Belebele, and NewsPH NLI.
- **Generation:** We dedicate a large portion of FilBench to testing an LLM's capability to faithfully translate texts, either from English to Filipino or from Cebuano to English. We include a diverse set of test examples ranging from documents (NTREX-128), realistic texts from volunteers (Tatoeba), and domain-specific text (TICO-19).

Each of these categories provides an aggregated metric.
To create a single representative score, we compute the weighted average based on the number of examples in each category, which we call the FilBench Score.

To simplify usage and set up, we built FilBench on top of [Lighteval](https://github.com/huggingface/lighteval), an all-in-one framework for LLM evaluation.
For language-specific evaluation, we first defined translation pairs from English to Tagalog (or Cebuano) for common terms used in evaluation such as "yes" (oo), "no" (hindi), and "true" (totoo) among others.
Then, we used the provided templates to implement custom tasks for the capabilities we care about.

FilBench is now available as a set of community tasks in the official Lighteval repository!

## What did we learn from FilBench?

By evaluating several LLMs on FilBench, we uncovered several insights into how they perform in Filipino.

### Finding #1: Although region-specific LLMs still lag behind GPT-4, collecting data to train these models is still a promising direction

In the past few years, we have seen an increase in region-specific LLMs that target Southeast Asian languages (SEA-specific), such as SEA-LION and SeaLLM.
These are open-weight LLMs that you can freely download from HuggingFace.
We find that SEA-specific LLMs are often the most parameter-efficient for our languages, achieving the highest FilBench scores compared to other models of their size.
However, the best SEA-specific model is still outperformed by closed-source LLMs like GPT-4o.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/filbench/filbench-finding-1.png" style="width: 100%;"/>

Building region-specific LLMs still makes sense, as we observe performance gains of 2-3% when continuously fine-tuning a base LLM with SEA-specific instruction-tuning data.
This suggests that **efforts to curate Filipino/SEA-specific training data for fine-tuning remain relevant**, as they can lead to better performance on FilBench.

### Finding #2: Filipino translation is still a difficult task for LLMs

We also observe that across the four categories on FilBench, most models struggle with Generation capabilities.
Upon inspecting failure modes in Generation, we find that these include cases where the model fails to follow translation instructions, generates overly verbose texts, or hallucinates another language instead of Tagalog or Cebuano.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/filbench/filbench-finding-2.png" style="width: 100%;"/>

### Finding #3: Open LLMs Remain a Cost-Effective Choice for Filipino Language Tasks

The Philippines tends to have limited internet infrastructure and lower average incomes [[3](https://unesdoc.unesco.org/ark:/48223/pf0000393860?posInSet=1&queryId=cb72b22d-9dd3-44cd-9090-c4c89328a09c)], necessitating accessible LLMs that are cost- and compute-efficient.
Through FilBench, we were able to identify LLMs that are on the Pareto frontier of efficiency.

In general, we find that open-weight LLMs, i.e., models that you can freely download from HuggingFace, are way cheaper than commercial models without sacrificing their performance.
If you want an alternative to GPT-4o for your Filipino language tasks, then try Llama 4 Maverick!

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/filbench/filbench-finding-3.png" style="width: 60%;"/>
</div>

We also make this information available in the HuggingFace space of the FilBench leaderboard.

## Does your LLM work on Philippine Languages? Try it on FilBench!

We hope that FilBench provides deeper insights into LLM capabilities for Philippine languages and serves as a catalyst for advancing Filipino NLP research and development.
The FilBench evaluation suite is built on top of Hugging Face's lighteval, allowing LLM developers to easily evaluate their models on our benchmark.
For more information, please visit the links below:

- 📄 Paper: https://arxiv.org/abs/2508.03523
- 🖥️ GitHub: https://github.com/filbench/filbench-eval

## Acknowledgements

The authors would like to thank Cohere Labs for providing credits through the Cohere Research Grant to run the Aya model series, and Together AI for additional computational credits for running several open models.
We also acknowledge the Hugging Face team, particularly the OpenEvals team (Clémentine Fourrier and Nathan Habib) and Daniel van Strien, for their support in publishing this blog post.

## Citation

If you are evaluating on FilBench, please cite our work:

```bibtex
@articleP{filbench,
  title={Fil{B}ench: {C}an {LLM}s {U}nderstand and {G}enerate {F}ilipino?},
  author={Miranda, Lester James V and Aco, Elyanah and Manuel, Conner and Cruz, Jan Christian Blaise and Imperial, Joseph Marvin},
  journal={arXiv preprint arXiv:2508.03523},
  year={2025}
}
```
