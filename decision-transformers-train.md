---
title: "Train your first Decision Transformer"
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
---

# Train your first Decision Transformer

<div class="blog-metadata">
    <small>Published September 02, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/decision-transformers-train.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/edbeeching"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1644220542819-noauth.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>edbeeching</code>
            <span class="fullname">Edward Beeching</span>
        </div>
    </a>
    <a href="/ThomasSimonini"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1632748593235-60cae820b1c79a3e4b436664.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>ThomasSimonini</code>
            <span class="fullname">Thomas Simonini</span>
        </div>
    </a>
</div>

In a [previous post](https://huggingface.co/blog/decision-transformers), we announced the launch of Decision Transformers in the transformers library. This new technique of **using a Transformer as a Decision-making model** is getting increasingly popular.

So today, **you’ll learn to train your first Offline Decision Transformer model from scratch to make a half-cheetah run.**

<figure class="image table text-center m-0 w-full">
    <video 
        alt="CheetahEd-expert"
        style="max-width: 70%; margin: auto;"
        autoplay loop autobuffer muted playsinline
    >
      <source src="assets/101_decision-transformers-train/replay.mp4" type="video/mp4">
  </video>
</figure>
*An "expert" Decision Transformers model, learned using offline RL in the Gym HalfCheetah environment.*

Sounds exciting? Let's get started!

- [What are Decision Transformers?](#what-are-decision-transformers)
- [Training Decision Transformers](#training-decision-transformers)
  - [Loading the dataset and building the Custom Data Collator](#loading-the-dataset-and-building-the-custom-data-collator)
- [Conclusion](#conclusion)
- [What’s next?](#whats-next)
- [References](#references)

## What are Decision Transformers?

The Decision Transformer model was introduced by **[“Decision Transformer: Reinforcement Learning via Sequence Modeling” by Chen L. et al](https://arxiv.org/abs/2106.01345)**. It abstracts Reinforcement Learning as a **conditional-sequence modeling problem**.

The main idea is that instead of training a policy using RL methods, such as fitting a value function that will tell us what action to take to maximize the return (cumulative reward), **we use a sequence modeling algorithm (Transformer)** that, given the desired return, past states, and actions, will generate future actions to achieve this desired return. It’s an autoregressive model conditioned on the desired return, past states, and actions to generate future actions that achieve the desired return.

**This is a complete shift in the Reinforcement Learning paradigm** since we use generative trajectory modeling (modeling the joint distribution of the sequence of states, actions, and rewards) to replace conventional RL algorithms. It means that in Decision Transformers, we don’t maximize the return but rather generate a series of future actions that achieve the desired return.

The process goes this way:

1. We feed **the last K timesteps** into the Decision Transformer with three inputs:
    - Return-to-go
    - State
    - Action
2. **The tokens are embedded** either with a linear layer if the state is a vector or a CNN encoder if it’s frames.
3. **The inputs are processed by a GPT-2 model**, which predicts future actions via autoregressive modeling.

![https://huggingface.co/blog/assets/58_decision-transformers/dt-architecture.gif](https://huggingface.co/blog/assets/58_decision-transformers/dt-architecture.gif)

*Decision Transformer architecture. States, actions, and returns are fed into modality-specific linear embeddings, and a positional episodic timestep encoding is added. Tokens are fed into a GPT architecture which predicts actions autoregressively using a causal self-attention mask. Figure from [1].*

There are different types of Decision Transformers, but today, we’re going to train an offline Decision Transformer, meaning that we only use data collected from other agents or human demonstrations. **The agent does not interact with the environment**. If you want to know more about the difference between offline and online reinforcement learning, [check this article](https://huggingface.co/blog/decision-transformers).

Now that we understand the theory behind Offline Decision Transformers, **let’s see how we’re going to train one in practice.**

## Training Decision Transformers

In the previous post, we demonstrate how to use a transformers Decision Transformer model and load pretrained weights from the 🤗 hub. 

In this part we will use 🤗 Trainers and a custom Data Collator to training a Decision Transformer model from scratch, using an Offline RL Dataset hosted on the 🤗 hub. You can find code for this tutorial in [this colab notebook]() # ADD LINK 

We will be performing offline RL to learning the following behavior in the [mujoco halfcheetah environment](https://www.gymlibrary.dev/environments/mujoco/half_cheetah/).

<figure class="image table text-center m-0 w-full">
    <video 
        alt="CheetahEd-expert"
        style="max-width: 70%; margin: auto;"
        autoplay loop autobuffer muted playsinline
    >
      <source src="assets/101_decision-transformers-train/replay.mp4" type="video/mp4">
  </video>
</figure>
*An "expert" Decision Transformers model, learned using offline RL in the Gym HalfCheetah environment.*

### Loading the dataset and building the Custom Data Collator

We host a number of Offline RL Datasets on the hub. Today we will be training with the halfcheetah “expert” dataset, hosted here on hub.

First we need to import the `load_dataset` function from the 🤗 datasets package and download the dataset to our machine.

```python
from datasets import load_dataset
dataset = load_dataset("edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2")
```

While most datasets on the hub are ready to use out of the box, sometime we wish to perform so additional processing or modifcation of the dataset. In that case [we wish to match the authors implementation](https://github.com/kzl/decision-transformer), that is we need to:

- Normalize each feature by subtraction the mean and dividing by the standard deviation.
- Pre-compute discounted returns.
- Scaling the rewards and returns by a factor of 1000.
- Augmenting the dataset sampling distribution so it takes into account the length of the expert agent’s trajectories.

```python

```

## Conclusion

This post have demonstrated how to training the Decision Transformer on an offline RL dataset, hosted on 🤗 datasets (link). We have used a 🤗 transformers(link repo) Trainer(link docs) and a custom data collator.

In addition to Decision Transformers, **we want to support more use cases and tools from the Deep Reinforcement Learning community**. Therefore, it would be great to hear your feedback on the Decision Transformer model, and more generally anything we can build with you that would be useful for RL. Feel free to **[reach out to us](mailto:thomas.simonini@huggingface.co)**.

## What’s next?

In the coming weeks and months, **we plan on supporting other tools from the ecosystem**:

- Expanding our repository of Decision Tranformer models with models trained or finetuned in an online setting [2]
- Integrating [sample-factory version 2.0](https://github.com/alex-petrenko/sample-factory)

The best way to keep in touch is to **[join our discord server](https://discord.gg/YRAq8fMnUG)** to exchange with us and with the community.

## References
[1] Chen, Lili, et al. "Decision transformer: Reinforcement learning via sequence modeling." *Advances in neural information processing systems* 34 (2021).

[2] Zheng, Qinqing and Zhang, Amy and Grover, Aditya “*Online Decision Transformer”* (arXiv preprint, 2022)
