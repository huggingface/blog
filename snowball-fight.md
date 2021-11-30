---
title: 'Introducing Snowball Fight ☃️, our first ML-Agents environment'
thumbnail: /blog/assets/38_introducing_snowball_fight/thumbnail.png
---

<h1>
    Introducing Snowball Fight ☃️, our First ML-Agents Environment
</h1>

<div class="blog-metadata">
    <small>Published November 30, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/snowball-fight.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/ThomasSimonini"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1632748593235-60cae820b1c79a3e4b436664.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>simoninithomas</code>
            <span class="fullname">Thomas Simonini</span>
        </div>
    </a>
</div>



We're excited to share our **first custom Deep Reinforcement Learning environment**: Snowball Fight 1vs1 🎉.
![gif](assets/38_introducing_snowball_fight/snowballfight.gif)

Snowball Fight is a game made with Unity ML-Agents, where you shoot snowballs against a Deep Reinforcement Learning agent. The game is [**hosted on Hugging Face Spaces**](https://hf.co/spaces/launch). 

The game is **hosted on Hugging Face Spaces**. 

👉 [You can play it online here](https://huggingface.co/spaces/ThomasSimonini/SnowballFight)

In this post, we'll cover **the ecosystem we are working on for Deep Reinforcement Learning researchers and enthusiasts that use Unity ML-Agents**.

## Unity ML-Agents at Hugging Face

The [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) is an open source library that allows you to build games and simulations with Unity game engine to **serve as environments for training intelligent agents**.

With this first step, our goal is to build an ecosystem on Hugging Face for Deep Reinforcement Learning researchers and enthusiasts that uses ML-Agents, with three features.

1. **Building and sharing custom environments.** We are developing and sharing exciting environments to experiment with new problems: snowball fights, racing, puzzles... All of them will be open source and hosted on the Hugging Face's Hub.
2. **Allowing you to easily host your environments, save models and share them** on the Hugging Face Hub. We have already published the Snowball Fight training environment [here](https://huggingface.co/ThomasSimonini/ML-Agents-SnowballFight-1vs1), but there will be more to come!
3. **You can now easily host your demos on Spaces** and showcase your results quickly with the rest of the ecosystem.



## Be part of the conversation: join our discord server!

If you're using ML-Agents or interested in Deep Reinforcement Learning and want to be part of the conversation, **[you can join our discord server](https://discord.gg/YRAq8fMnUG)**. We just added two channels (and we'll add more in the future):

- Deep Reinforcement Learning
- ML-Agents

[Our discord](https://discord.gg/YRAq8fMnUG) is the place where you can exchange about Hugging Face, NLP, Deep RL, and more! It's also in this discord that we'll announce all our new environments and features in the future.


## What's next?

In the coming weeks and months, we will be extending the ecosystem by:

- Writing some **technical tutorials on ML-Agents**.
- Working on a **Snowball Fight 2vs2 version**, where the agents will collaborate in teams using [MA-POCA, a new Deep Reinforcement Learning algorithm](https://blog.unity.com/technology/ml-agents-plays-dodgeball) that trains cooperative behaviors in a team. 

![screenshot2vs2](assets/38_introducing_snowball_fight/screenshot2vs2.png)

- And we're building **new custom environments that will be hosted in Hugging Face**.

## Conclusion

We're excited to see what you're working on with ML-Agents and how we can build features and tools **that help you to empower your work**.

Don't forget to [join our discord server](https://discord.gg/YRAq8fMnUG) to be alerted of the new features.
