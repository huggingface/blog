---
title: "Deep Q-Learning with Space Invaders"
thumbnail: /blog/assets/78_deep_rl_dqn/thumbnail.gif
---

<html>
<head>
<style>
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
</style>
<h1>Deep Q-Learning with Space Invaders</h1>
<h2>Unit 3, of the the <a href="https://github.com/huggingface/deep-rl-class">Deep Reinforcement Learning Class with Hugging Face 🤗</a></h2>

<div class="author-card">
    <a href="/ThomasSimonini">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1632748593235-60cae820b1c79a3e4b436664.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>ThomasSimonini</code>
            <span class="fullname">Thomas Simonini</span>
        </div>
  </a>
</div>

</head>

<body>

*This article is part of the Deep Reinforcement Learning Class. A free course from beginner to expert. Check the syllabus [here.](https://github.com/huggingface/deep-rl-class)*
---

[In the last unit](https://huggingface.co/blog/deep-rl-q-part2), we learned our first reinforcement learning algorithm: Q-Learning, **implemented it from scratch**, and trained it in two environments, FrozenLake-v1 ☃️ and Taxi-v3 🚕.

We got excellent results with this simple algorithm. But these environments were relatively simple because the **State Space was discrete and small** (14 different states for FrozenLake-v1 and 500 for Taxi-v3).

But as we'll see, producing and updating a **Q-table can become ineffective in large state space environments.**

So today, **we'll study our first Deep Reinforcement Learning agent**: Deep Q-Learning. Instead of using a Q-table, Deep Q-Learning uses a Neural Network that takes a state and approximates Q-values for each action based on that state.

And **we'll train it to play Space Invaders and other Atari environments using [RL-Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)**, a training framework for RL using Stable-Baselines that provides scripts for training, evaluating agents, tuning hyperparameters, plotting results, and recording videos.
  
<figure class="image table text-center m-0 w-full">
  <img src="assets/78_deep_rl_dqn/atari-envs.gif" alt="Environments"/>
</figure>
  
So let’s get started! 🚀

[Table des matières]

## From Q-Learning to Deep Q-Learning

We learned that **Q-Learning is an algorithm we use to train our Q-Function**, an **action-value function** that determines the value of being at a particular state and taking a specific action at that state.

<figure class="image table text-center m-0 w-full"> <img src="assets/73_deep_rl_q_part2/Q-function.jpg" alt="Q-function"/> <figcaption>Given a state and action, our Q Function outputs a state-action value (also called Q-value)</figcaption> </figure>

The **Q comes from "the Quality" of that action at that state.**

Internally, our Q-function has **a Q-table, a table where each cell corresponds to a state-action value pair value.** Think of this Q-table as **the memory or cheat sheet of our Q-function.**

The problem is that Q-Learning is a *tabular method*. Aka, a problem in which the state and actions spaces **are small enough for approximate value functions to be represented as arrays and tables**. And this is **not scalable**.

Q-Learning was working well with small state space environments like:

- FrozenLake, we had 14 states.
- Taxi-v3, we had 500 states.

IMG Frozen Lake Taxi-v3

But think of what we're going to do today: we will train an agent to learn to play Space Invaders using the frames as input.

As **Nikita Melkozerov mentioned, Atari environments** have an observation space with a shape of (210, 160, 3), containing values ranging from 0 to 255 si that gives us 256^(210*160*3) = 256^100800 (for comparison, we have approximately 10^80 atoms in the observable universe ).

<img src="assets/78_deep_rl_dqn/atari.jpg" alt="Atari State Space"/>

Therefore, the state space is gigantic; hence creating and updating a Q-table for that environment would not be efficient. In this case, the best idea is to approximate the Q-values instead of a Q-table using a parametrized Q-function $Q_\theta(s,a)$.

This neural network will approximate, given a state, the different Q-values for each possible action at that state. And that's exactly what Deep Q-Learning does.

<img src="assets/63_deep_rl_intro/deep.jpg" alt="Deep Q Learning"/>


Now that we understand Deep Q-Learning, let's dive deeper into the Deep Q-Network.
  
## 
