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

We're excited to share Jack of All Trades (JAT), the first open-source multi-purpose agent! JAT is able to simultaneously play video games, control a robot to do a vast amount of tasks, understand and carry out orders in a simple navigation environment and so much more!

<!-- [Schema] -->

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

[score_steps.pdf](assets/jat/score_steps.pdf)

## Conclusions

## What's next?

### Call for contributors

Paper page: https://huggingface.co/papers/2402.09844