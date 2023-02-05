---
title: "Introducing ⚔️ AI vs. AI ⚔️ a deep reinforcement learning multi-agents competition system" 
thumbnail: /blog/assets/128_aivsai/thumbnail.png
authors:
- user: CarlCochet
- user: ThomasSimonini
---
<!-- {blog_metadata} -->
<!-- {authors} -->

![Thumbnail](assets/ai-vs-ai/thumbnail.png)

We’re excited to introduce a new tool we created: **⚔️ AI vs. AI ⚔️, a deep reinforcement learning multi-agents competition system**.

This tool hosted on a Space allows us **to create multi-agent competitions**. It is composed of three elements:

- A *Space* with a matchmaking algorithm that **runs the model fights using a background task**.
- A *Dataset* **containing the results**.
- A *Leaderboard* that gets the **match history results and displays the models’ ELO**.

Then, when a user pushes a trained model to the Hub, **it gets evaluated and ranked against others**. Thanks to that, we can evaluate your agents against other’s agents in a multi-agent setting.

In addition to being a useful tool for hosting multi-agent competitions, we think this tool can also be a **robust evaluation technique in multi-agent settings.** Since by playing against a lot of policies, your agents are evaluated against a wide range of behaviors, and you’ll get a good idea of the quality of your policy.

Let’s see how it works with our first competition host: SoccerTwos Challenge.

![SoccerTwos example](assets/ai-vs-ai/soccertwos.gif)

## How does AI vs. AI works?





## Our first AI vs. AI challenge experimentation: SoccerTwos Challenge ⚽

This challenge is Unit 7 of our [free Deep Reinforcement Learning Course](https://huggingface.co/deep-rl-course/unit0/introduction). It started on February 1st and will end on April 30th.

If you’re interested, **you don’t need to participate in the course to be able to participate in the competition. You can start here** 👉 https://huggingface.co/deep-rl-course/unit7/introduction

In this Unit, readers learned the basics of multi-agent reinforcement learning (MARL), and they need to train a **2vs2 soccer team.**

The environment used was made by the [Unity ML-Agents team](https://github.com/Unity-Technologies/ml-agents). The goal is simple: your team needs to score a goal: to do that, they need to beat the opponent team and collaborate with their teammate.

![SoccerTwos example](assets/ai-vs-ai/soccertwos.gif)

In addition to the leaderboard, we created a Space demo where people can choose two teams and visualize them playing 👉[https://huggingface.co/spaces/unity/SoccerTwos](https://huggingface.co/spaces/unity/SoccerTwos)

This experimentation is going well since we already have 48 models on the leaderboard: [ADD link leaderboard]

![Leaderboard](assets/ai-vs-ai/leaderboard.png)

We also [created a discord channel called ai-vs-ai-competition](http://hf.co/discord/join) so that people can exchange with others and share advice.

### Conclusion and what’s next?

Since the tool we developed **is environment agnostic**, we want to host more challenges in the future with [PettingZoo](https://pettingzoo.farama.org/) and other multi-agents environments. If you have some environments or challenges you want to do, <a href="mailto:thomas.simonini@huggingface.co">don’t hesitate to reach us</a>.

In addition to being a useful tool for hosting multi-agent competitions, we think that this tool can also be **a robust evaluation technique in multi-agent settings: by playing against a lot of policies, your agents are evaluated against a wide range of behaviors, and you’ll get a good idea of the quality of your policy.**

The best way to keep in touch is to [join our discord server](http://hf.co/discord/join) to exchange with us and with the community.

****************Citation****************

Citation: If you found this useful for your academic work, please consider citing our work, in text:

`Cochet, Simonini, "Introducing AI vs. AI a deep reinforcement learning multi-agents competition system", Hugging Face Blog, 2023.`

BibTeX citation:

```
@article{cochet-simonini2023ift,
  author = {Cochet, Carl and Simonini, Thomas},
  title = {Introducing AI vs. AI a deep reinforcement learning multi-agents competition system},
  journal = {Hugging Face Blog},
  year = {2023},
  note = {https://huggingface.co/blog/aivsai},
}
```
