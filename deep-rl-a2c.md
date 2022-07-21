---
title: "Advantage Actor Critic (A2C)"
thumbnail: /blog/assets/89_deep_rl_a2c/thumbnail.gif
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
<h1>Advantage Actor Critic (A2C)</h1>
<h2>Unit 7, of the <a href="https://github.com/huggingface/deep-rl-class">Deep Reinforcement Learning Class with Hugging Face 🤗</a></h2>

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

<img src="assets/89_deep_rl_a2c/thumbnail.jpg" alt="Thumbnail"/>  

---

[In Unit 5](https://huggingface.co/blog/deep-rl-pg), we learned about our first Policy-Based algorithm called **Reinforce**. 
Indeed, in Policy-Based methods, **we aim to optimize the policy directly without using a value function**. More precisely, Reinforce is part of a subclass of *Policy-Based Methods* called *Policy-Gradient methods*. This subclass optimizes the policy directly by **estimating the weights of the optimal policy using Gradient Ascent**.

We saw that Reinforce worked well. However, because we use Monte-Carlo sampling to estimate return (we use an entire episode to calculate the return), **we have significant variance in policy gradient estimation**. 

Remember that the policy gradient estimation is **the direction of the steepest increase in return**. Aka, how to update our policy weights so that actions that lead to good returns have a higher probability of being taken. This variance that will study in this unit **leads to slower training since we need a lot of samples to mitigate it**.

So today, we'll study an **Actor-Critic method**, a hybrid architecture combining a value-based and policy-based methods that help to stabilize the training by reducing the variance:
- *An Actor* that controls **how our agent behaves** (policy-based method)
- *A Critic* that measures **how good the action taken is** (value-based method)

We'll study one of these "hybrid methods, " Advantage Actor Critic (A2C), a**nd train our agent using Stable-Baselines3 in robotic environments**. Where we'll train two agents to walk:
- A bipedal walker 🦿
- A spider 🕸️

<img src="https://github.com/huggingface/deep-rl-class/blob/main/unit7/assets/img/pybullet-envs.gif?raw=true" alt="Robotics environments"/>

Sounds exciting? Let's get started!
  
- []



## The Problem of Variance in Reinforce
In Reinforce, we want to **increase the probability of actions in a trajectory proportional to the goodness of the return**.

<img src="https://huggingface.co/blog/assets/85_policy_gradient/pg.jpg" alt="Reinforce"/>
- If the **return is high**, we will **push up** the probabilities of the (state, action) combinations.
- Else, if the **return is low**, it will **push down** the probabilities of the (state, action) combinations.

This return \\(R(\tau)\\) is calculated using a *Monte-Carlo sampling*. Indeed, we collect a trajectory and calculate the discounted return, **and use this score to increase or decrease the probability of every action taken in that trajectory**. If the return is good, all actions will be “reinforced” by increasing their likelihood of being taken.
  
\\(R(\tau) = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...\\) 

The advantage of this method is that **it’s unbiased. Since we’re not estimating the return**, we use only the true return we obtain.
  
But the problem is that **the variance is high, since trajectories can lead to different returns** due to stochasticity of the environment (random events during episode) and stochasticity of the policy. Consequently, the same starting state can lead to very different returns.
And so, **the return starting at the same state can vary significantly across episodes**.
  
<img src="assets/89_deep_rl_a2c/variance.jpg" alt="variance"/>  

The solution is to mitigate the variance by **using a large number of trajectories, hoping that the variance introduced in any one trajectory will be reduced in aggregate and provide a "true" estimation of the return.**

However, increasing the batch size significantly reduces sample efficiency. So we need to find additional mechanisms to reduce the variance.
  
