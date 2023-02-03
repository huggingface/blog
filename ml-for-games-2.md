---
title: "AI for Game Development: Creating a Farming Game in 5 Days. Part 2"
thumbnail: /blog/assets/124_ml-for-games/thumbnail2.png
authors:
- user: dylanebert
---

<h1>AI for Game Development: Creating a Farming Game in 5 Days. Part 2</h1>

{authors}
 
</head>

<body>

**Welcome to AI for Game Development!** In this series, we'll be using AI tools to create a fully functional farming game in just 5 days. By the end of this series, you will have learned how you can incorporate a variety of AI tools into your game development workflow. I will show you how you can use AI tools for:

1. Art Style
2. Game Design
3. 3D Assets
4. 2D Assets
5. Story

Want the quick video version? You can watch it [here](https://www.tiktok.com/@individualkex/video/7186551685035085098). Otherwise, if you want the technical details, keep reading!

**Note:** This tutorial is intended for readers who are familiar with Unity development and C#. If you're new to these technologies, check out the [Unity for Beginners](https://www.tiktok.com/@individualkex/video/7086863567412038954) series before continuing.

## Day 2: Game Design

In [Part 1](https://huggingface.co/blog/ml-for-games-1) of this tutorial series, we used **AI for Art Style**. More specifically, we used Stable Diffusion to generate concept art and develop the visual style of our game.

In this part, we'll be using AI for Game Design. In [The Short Version](#the-short-version), I'll talk about how I used ChatGPT as a tool to help develop game ideas. But more importantly, what is actually going on here? Keep reading for background on [Language Models](#language-models) and their broader [Uses in Game Development](#uses-in-game-development).

### The Short Version

The short version is straightforward: ask [ChatGPT](https://chat.openai.com/chat) for advice, and follow its advice at your own discretion. In the case of the farming game, I asked ChatGPT:

> You are a professional game designer, designing a simple farming game. What features are most important to making the farming game fun and engaging?

The answer given includes (summarized):

1. Variety of crops
2. A challenging and rewarding progression system
3. Dynamic and interactive environments
4. Social and multiplayer features
5. A strong and immersive story or theme

Given that I only have 5 days, I decided to [gray-box](https://en.wikipedia.org/wiki/Gray-box_testing) the first two points. You can play the result [here](https://individualkex.itch.io/ml-for-game-dev-2), and view the source code [here](https://github.com/dylanebert/FarmingGame).

I'm not going to go into detail on how I implemented these mechanics, since the focus of this series is how to use AI tools in your own game development process, not how to implement a farming game. Instead, I'll talk about what ChatGPT is (a language model), how these models actually work, and what this means for game development.

### Language Models

ChatGPT, despite being a major breakthrough in adoption, is an iteration on tech that has existed for a while: *language models*.

Language models are a type of AI that are trained to predict the likelihood of a sequence of words. For example, if I were to write "The cat chases the ____", a language model would be trained to predict "mouse". This training process can then be applied to a wide variety of tasks. For example, translation: "the French word for cat is ____". This setup, while successful at some natural language tasks, wasn't anywhere near the level of performance seen today. This is, until the introduction of **transformers**.

**Transformers**, [introduced in 2017](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), are a neural network architecture that use a self-attention mechanism to predict the entire sequence all at once. This is the tech behind modern language models like ChatGPT. Want to learn more about how they work? Check out our [Introduction to Transformers](https://huggingface.co/course/chapter1/1) course, available free here on Hugging Face.

So why is ChatGPT so successful compared to previous language models? It's impossible to answer this in its entirety, since ChatGPT is not open source. However, one of the reasons is Reinforcement Learning from Human Feedback (RLHF), where human feedback is used to improve the language model. Check out this [blog post](https://huggingface.co/blog/rlhf) for more information on RLHF: how it works, open-source tools for doing it, and its future.

This area of AI is constantly changing, and likely to see an explosion of creativity as it becomes part of the open source community, including in uses for game development. If you're reading this, you're probably ahead of the curve already.

### Uses in Game Development

In [The Short Version](#the-short-version), I talked about how I used ChatGPT to help develop game ideas. There is a lot more you can do with it though, like using it to [code an entire game](https://www.youtube.com/watch?v=YDWvAqKLTLg&ab_channel=AAlex). You can use it for pretty much anything you can think of. Something that might be a bit more helpful is to talk about what it *can't* do.

#### Limitations

ChatGPT often sounds very convincing, while being wrong. Here is an [archive of ChatGPT failures](https://github.com/giuven95/chatgpt-failures). The reason for these is that ChatGPT doesn't *know* what it's talking about the way a human does. It's a very large [Language Model](#language-models) that predicts likely outputs, but doesn't really understand what it's saying. One of my personal favorite examples of these failures (especially relevant to game development) is this explanation of quaternions from [Reddit](https://www.reddit.com/r/Unity3D/comments/zcps1f/eli5_quaternion_by_chatgpt/):

<figure class="image text-center">
  <img src="/blog/assets/124_ml-for-games/quaternion.png" alt="ChatGPT Quaternion Explanation">
</figure>

This explanation, while sounding excellent, is completely wrong. This is a great example of why ChatGPT, while very useful, shouldn't be used as a definitive knowledge base.

#### Suggestions

If ChatGPT fails a lot, should you use it? I would argue that it's still extremely useful as a tool, rather than as a replacement. In the example of Game Design, I could have followed up on ChatGPT's answer, and asked it to implement all of its suggestions for me. As I mentioned before, [others have done this](https://www.youtube.com/watch?v=YDWvAqKLTLg&ab_channel=AAlex), and it somewhat works. However, I would suggest using ChatGPT more as a tool for brainstorming and acceleration, rather than as a complete replacement for steps in the development process.

Click [here](https://huggingface.co/blog/ml-for-games-3) to read Part 3, where we use **AI for 3D Assets**.
