---
title: "Proximal Policy Optimization (PPO)"
thumbnail: /blog/assets/93_deep_rl_ppo/thumbnail.gif
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
<h1>Proximal Policy Optimization (PPO)</h1>
<h2>Unit 8, of the <a href="https://github.com/huggingface/deep-rl-class">Deep Reinforcement Learning Class with Hugging Face 🤗</a></h2>

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

<img src="assets/93_deep_rl_ppo/thumbnail.jpg" alt="Thumbnail"/>  

---
**[In the last Unit](https://huggingface.co/blog/deep-rl-a2c)**, we learned about Advantage Actor Critic (A2C), a hybrid architecture combining value-based and policy-based methods that help to stabilize the training by reducing the variance:

- *An Actor* that controls **how our agent behaves** (policy-based method)
- *A Critic* that measures **how good the action taken is** (value-based method)

Today we'll learn about Proximal Policy Optimization (PPO), a model that improves our agent's training stability by avoiding too large policy updates. To do that, we use a ratio that will tell us the difference between our new and old policy and clip this ratio from a specific range [1-epsilon, 1+epsilon].

Doing this will ensure **that our policy update will not be too large and that the training is more stable.**

And then, after the theory, we'll code a PPO architecture from scratch using PyTorch and bulletproof our implementation with CartPole-v1 and LunarLander-v2.

Sounds exciting? Let's get started!

[Table des matières]
