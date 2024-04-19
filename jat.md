---
title: 'Jack of All Trades, Expert of Some, a Multi-Purpose Transformer Agent'
thumbnail: /blog/assets/jat/thumbnail.gif
authors:
- user: qgallouedec
- user: edbeeching
- user: ClementRomac
---

# Jack of All Trades, Expert of Some, a Multi-Purpose Transformer Agent

## Introduction

We're excited to share Jack of All Trades (JAT), a project that aims to move in the direction of a generalist agent. In a nutshell, the project has resulted in:

- the creation of a dataset for generalist agent training, and
- the creation of a multi-task, multi-modal agent based on a transformer.

The JAT dataset contains hundreds of thousands of expert trajectories collected with expert agents in very different environments. We have used this dataset to train a transformer-based agent: JAT. This model is capable of playing video games, controlling a robot to perform a wide variety of tasks, understanding and executing commands in a simple navigation environment and much more!

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F327/blog/jat/global_schema.gif" alt="Global schema"/>

## Datasets & expert policies

Le projet JAT consiste en plusierus elements essentiels indispenspasable pour l’entrainement d’agent generaliste. Chacune de ces composant est complement ouverte, donnant un voie immédiate aux praticien pour travailler par dessus.

### Expert policies

RL traditionally involves training expert policies on single environments. Leveraging these expert policies is a genuine way to build a general-purpose agent. We trained expert agents for each of the environments in this work (with the exception of BabyAI, for which a Bot allows us to have an untrained expert agent). These agents were trained using Sample-Factory, and are all available on the Hub.

[Collapasable list of link to agents ?]

### Dataset

We release the JAT Dataset, the first dataset for generalist agent training. The JAT dataset contains hundreds of thousands of expert trajectories collected with the above-mentioned expert agents. To use this dataset, simply load it like any other dataset from the Hub:

```python
>>> from datasets import load_dataset
>>> dataset = load_dataset("jat-project/jat-dataset", "metaworld-assembly")
>>> first_episode = dataset["train"][0]
>>> first_episode.keys()
dict_keys(['continuous_observations', 'continuous_actions', 'rewards'])
>>> len(first_episode["rewards"])
500
>>> first_episode["continuous_actions"][0]
[6.459120273590088, 2.2422609329223633, -5.914587020874023, -19.799840927124023]
```

In addition to RL data, we include textual datasets to enable a unique interface for the user. That's why you'll also find subsets for [Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Database_download), [Oscar](https://oscar-project.org), [OK-VQA](https://okvqa.allenai.org) and [Conceptual-Captions](https://ai.google.com/research/ConceptualCaptions/).

[dataset_atari_return_distribution.pdf](Jack%20of%20All%20Trades,%20Master%20of%20Some%202ec1606871e94b4abed0967bbe2c72d8/dataset_atari_return_distribution.pdf)


## Reproducing GATO

<!-- [Is a whole section really needed? See below] -->

## Building the JAT architecture

We introduce the JAT model, a multi-modal multi-purpose transformer-based agent. The JAT project was born with the quest of reproducing and open-sourcing [Gato](https://arxiv.org/abs/2205.06175). In addition to reproducing the Gato multi-modal transformer-based architecture, we also improved it by introducing a new unified input/output space where everything is handled as a single token. This results in a simpler design but also allowed us to significantly expand the attention window over previous timesteps compared to Gato.

<!-- [Schema] -->

We also add to JAT an auxiliary objective of predicting the next observation leading to improved performance.

## Experiments and Results

<figure class="image text-center">
  <video style="max-width: 100%; margin: auto;" autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/raw/refs%2Fpr%2F327/blog/jat/jat_hf.mp4"></video>
  <figcaption></figcaption>
</figure>

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F328/blog/jat/kappa_aggregated.svg" height="200" alt="Kappa Aggregated">
  <figcaption>Aggregate measures with 95% CIs for the study on the influence of observation prediction learning for selected tasks. The results presented cover the selected range of κ values and are based on 100 evaluations per task. Optimal \\( \kappa \\) selection can significantly improve agent performance.</figcaption>
</figure>

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F328/blog/jat/score_steps.svg" height="200" alt="Speed Comparison">
  <figcaption></figcaption>
</figure>

## Conclusions

## What's next?

### Call for contributors

Paper page: https://huggingface.co/papers/2402.09844