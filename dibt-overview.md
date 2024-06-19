---
title: "Data Is Better Together: A Look Back and Forward" 
thumbnail: /blog/assets/dibt-overview/thumbnail.png
authors:
- user: davanstrien
- user: davidberenstein1957
- user: sdiazlor
---

# Data Is Better Together: A Look Back and Forward

For the past few months, we have been working on the [Data Is Better Together](https://github.com/huggingface/data-is-better-together) initiative. With this collaboration between Hugging Face and Argilla and the support of the open-source ML community, our goal has been to empower the open-source community to create impactful datasets collectively.

Now, we have decided to move forward to explore new possibilities while always keeping our main goal in mind. To provide a clear overview of our achievements and the tasks to which everyone can still contribute, we have organized this into two sections: community efforts and cookbook efforts.

## Community efforts

Our first steps in this initiative focused on the **prompt ranking** project. Our goal was to create a dataset of 10K prompts, both synthetic and human-generated, ranked by quality. The community's response was immediate!

- In a few days, over 385 people joined.
- We released the [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) dataset intended for prompt ranking tasks or synthetic data generation.
- The dataset was used to build new [models](https://huggingface.co/models?dataset=dataset:DIBT/10k_prompts_ranked), such as SPIN.

Seeing the global support from the community, we recognized that English-centric data alone is insufficient, and there are not enough language-specific benchmarks for open LLMs. So, we created the **Multilingual Prompt Evaluation Project (MPEP)** with the aim of developing a leaderboard for multiple languages. For that, a subset of 500 high-quality prompts from [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) was selected to be translated into different languages.

- More than 18 language leaders created the spaces for the translations.
- Completed translations for [Dutch](https://huggingface.co/datasets/DIBT/MPEP_DUTCH), [Russian](https://huggingface.co/datasets/DIBT/MPEP_RUSSIAN) or [Spanish](https://huggingface.co/datasets/DIBT/MPEP_SPANISH), with many more efforts working towards complete translations of the prompts.
- The creation of a community of dataset builders on Discord

Going forward, we’ll continue to support community efforts focused on building datasets through tools and documentation. 

## Cookbook efforts

As part of [DIBT](https://github.com/huggingface/data-is-better-together), we also created guides and tools that help the community build valuable datasets on their own.

- **Domain Specific dataset**: To bootstrap the creation of more domain-specific datasets for training models, bringing together engineers and domain experts.
- **DPO/ORPO dataset**: To help foster a community of people building more DPO-style datasets for different languages, domains and tasks.
- **KTO dataset**: To help the community create their own KTO datasets.

## What have we learnt?

- The community is eager to participate in this efforts and there is excitement about collectively working on datasets
- There are existing inequalities that must be overcome to ensure comprehensive and inclusive benchmarks. Datasets for certain languages, domains, and tasks are currently underrepresented in the open-source community.
- We have many of the tools needed for the community to effectively collaborate on building valuable datasets.

## How can you get involved?

You can still contribute to the cookbook efforts by following the instructions in the README of the project you're interested in and share your datasets and results with the community. Or provide new guides and tools for everyone. Your contributions are invaluable in helping us build a robust and comprehensive resource for all.

If you want to be part, please join us in the `#data-is-better-together` channel in the [**Hugging Face Discord**](http://hf.co/join/discord) and let us know what you want to build together!

We are looking forward to building better datasets together with you!