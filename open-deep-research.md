---
title: "Open-source DeepResearch – Freeing our search agents"
thumbnail: /blog/assets/open-deep-research/thumbnail.png
authors:
- user: m-ric
- user: albertvillanova
- user: merve
- user: thomwolf
- user: clefourrier
---

# Open-source DeepResearch – Freeing our search agents

## TLDR

Yesterday, OpenAI released [Deep Research](https://openai.com/index/introducing-deep-research/), a system that browses the web to summarize content and answer questions based on the summary. The system is impressive and blew our minds when we tried it for the first time.

One of the main results in the blog post is a strong improvement of performances on the [General AI Assistants benchmark (GAIA)](https://huggingface.co/gaia-benchmark), a benchmark we’ve been playing with recently as well, where they successfully reached near 67% correct answers on 1-shot on average, and 47.6% on especially challenging “level 3” questions that involve multiple steps of reasoning and tool usage (see below for a presentation of GAIA).


DeepResearch is composed of an LLM (which can be selected from the current list of LLMs provided by OpenAI, 4o, o1, o3, etc) and an internal “agentic framework” which guide the LLM to use tools like web search and organize its actions in steps. 

While powerful LLMs are now freely available in open-source (see e.g. [the recent DeepSeek R1 model](https://huggingface.co/deepseek-ai/DeepSeek-R1)), OpenAI didn’t disclose much about the agentic framework underlying Deep Research…

So we decided to embark on a 24-hour mission to reproduce their results and open-source the needed framework along the way!

The clock is ticking, let’s go! ⏱️

## Table of Contents

- [What are Agent frameworks and why they matter?](#what-are-agent-frameworks-and-why-they-matter)
- [The GAIA benchmark](#the-gaia-benchmark)
- [Building an open Deep Research](#building-an-open-deep-research)
  - [Using a CodeAgent](#using-a-codeagent)
  - [Making the right tools 🛠️](#making-the-right-tools-🛠️)
- [Results 🏅](#results-🏅)
- [Community reproductions](#community-reproductions)
- [Most important next steps](#most-important-next-steps)


## What are Agent frameworks and why they matter?

> [!TIP]
>    An Agent framework is a layer on top of an LLM to make said LLM execute actions (like browse the web or read PDF documents), and organize its operations in a series of steps.
>    For a quick intro to agents, check [this great interview by Andrew Ng](https://youtu.be/sal78ACtGTc?feature=shared&t=52) and our [introduction blog post](https://huggingface.co/blog/smolagents) to the smolagents library. For a more detailed dive in agents you can subscribe to our agents course that starts in just a few days: [link here](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=9ed45a3ef6).


Almost everyone has already experienced how powerful LLMs can be simply by playing with chatbots.. However, what not everyone is aware of yet is that integrating these LLMs into agentic systems can give them real superpowers! 

Here is a recent example comparing the performance of a few frontier LLMs with and without an agentic framework (in this case the simple [smolagents](https://github.com/huggingface/smolagents) library) - using an agentic framework bumps performance by up to 60 points!

![Benchmarks](https://huggingface.co/datasets/huggingface/documentation-images/resolve/6c7ed2035810565043c92b472d5564c3f1fa4d7e/blog/open-deep-research/benchmarks.png)

In fact, OpenAI also highlighted in [its release blogpost](https://openai.com/index/introducing-deep-research/) how Deep Research performed dramatically better than standalone LLMs on the knowledge-intensive "[Humanity’s Last Exam](https://huggingface.co/datasets/cais/hle)" benchmark.

So, what happens when we integrate our current top LLM in an agentic framework, to work toward an `open-DeepResearch` ?

**A quick note:** We’ll benchmark our results on the same GAIA challenge but keep in mind that this is a work in progress. DeepResearch is a massive achievement and its open reproduction will take time. In particular, full parity will require improved browser use and interaction like OpenAI Operator is providing, i.e. beyond the current text-only web interaction we explore in this first step.

Let’s first understand the scope of the challenge: GAIA.

## The GAIA benchmark

[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) is arguably the most comprehensive benchmark for agents. Its questions are very difficult and hit on many challenges of LLM-based systems. Here is an example of a hard question:

> Which of the fruits shown in the 2008 painting "Embroidery from Uzbekistan" were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film "The Last Voyage"? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o'clock position. Use the plural form of each fruit.
> 

You can see this question involves several challenges:

- Answering in a constrained format,
- Using multimodal capabilities (to extract the fruits from the image),
- Gathering several pieces of information, some depending on others:
    - Identifying the fruits on the picture
    - Finding which ocean liner was used as a floating prop for “The Last Voyage”
    - Finding the October 1949 breakfast menu for the above ocean liner
- Chaining together a problem-solving trajectory in the correct order.

Solving this requires both high-level planning abilities and rigorous execution, which are two areas where LLMs struggle when used alone.

So it’s an excellent test set for agent systems!

On GAIA’s [public leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard), GPT-4 does not even reach 7% on the validation set when used without any agentic setup. On the other side of the spectrum, with Deep Research, OpenAI reached 67.36% score on the validation set, so an order of magnitude better! (Though we don’t know how they would actually fare on the private test set.)

Let’s see if we can do better with open source tools!

## Building an open Deep Research

### Using a CodeAgent

The first improvement over traditional AI agent systems we’ll tackle is to use a so-called “code agent”. As shown by [Wang et al. (2024)](https://huggingface.co/papers/2402.01030), letting the agent express its actions in code has several advantages, but most notably that **code is specifically designed to express complex sequences of actions**.

Consider this example given by Wang et al.:

![Code Agent](https://huggingface.co/datasets/huggingface/documentation-images/resolve/6c7ed2035810565043c92b472d5564c3f1fa4d7e/blog/open-deep-research/code_agent.png)

This highlights several advantages of using code:

- Code actions are **much more concise** than JSON.
    - Need to run 4 parallel streams of 5 consecutive actions ? In JSON, you would need to generate 20 JSON blobs, each in their separate step; in Code it’s only 1 step.
    - On average, the paper shows that Code actions require 30% fewer steps than JSON, which amounts to an equivalent reduction in the tokens generated. Since LLM calls are often the dimensioning cost of agent systems, it means your agent system runs are ~30% cheaper.
- Code enables to re-use tools from common libraries
- Better performance in benchmarks, due to two reasons:
    - More intuitive way to express actions
    - Extensive exposure of LLMs to code in training

The advantages above were confirmed by our experiments on the [agent_reasoning_benchmark](https://github.com/aymeric-roucher/agent_reasoning_benchmark).

From building `smolagents` we can also cite a notable additional advantage, which is a better handling of state: this is very useful for multimodal tasks in particular. Need to store this image/audio/other for later use? No problem, just assign it as a variable in your state and you can re-use it 4 steps later if needed. In JSON you would have to let the LLM name it in a dictionary key and trust the LLM will later understand that it can still use it.

### Making the right tools 🛠️

Now we need to provide the agent with the right set of tools. 

**1.** A web browser. While a fully fledged web browser interaction like [Operator](https://openai.com/index/introducing-operator/) will be needed to reach full performance, we started with an extremely simple text-based web browser for now for our first proof-of-concept. You can find the code [here](https://github.com/huggingface/smolagents/blob/gaia-submission-r1/examples/open_deep_research/scripts/text_web_browser.py)

**2.** A simple text inspector, to be able to **read a bunch of text file format**, find it [here](https://github.com/huggingface/smolagents/blob/gaia-submission-r1/examples/open_deep_research/scripts/text_inspector_tool.py).

These tools were taken from the excellent [Magentic-One](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/) agent by Microsoft Research, kudos to them! We didn’t change them much, as our goal was to get as high a performance as we can with the lowest complexity possible.

Here is a short roadmap of improvements which we feel would really improve these tools’ performance (feel free to open a PR and contribute!):

- extending the number of file formats which can be read.
- proposing a more fine-grained handling of files.
- replacing the web browser with a vision-based one, which we’ve started doing [here](https://github.com/huggingface/smolagents/blob/gaia-submission-r1/src/smolagents/vision_web_browser.py).

## Results 🏅

In our 24h+ reproduction sprint, we’ve already seen steady improvements in the performance of our agent on GAIA!

We’ve quickly gone up from the previous SoTA with an open framework, around 46% for Magentic-One, to our [current performance of 55.15% on the validation set](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

This bump in performance is due mostly to letting our agents write their actions in code! Indeed, when switching to a standard agent that writes actions in JSON instead of code, performance of the same setup is instantly degraded to 33% average on the validation set.

[Here is the final agentic system.](https://github.com/huggingface/smolagents/tree/gaia-submission-r1/examples/open_deep_research)

We’ve set up [a live demo here](https://m-ric-open-deep-research.hf.space) for you to try it out!

<iframe
	src="https://m-ric-open-deep-research.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>



However, this is only the beginning, and there are a lot of things to improve! Our open tools can be made better, the smolagents framework can also be tuned, and we’d love to explore the performance of better open models to support the agent.

We welcome the community to come join us in this endeavour, so we can leverage the power of open research together to build a great open-source agentic framework! It would allow anyone to run a DeepResearch-like agent at home, with their favorite models, using a completely local and customized approach!

## Community Reproductions

While we were working on this and focusing on GAIA, other great open implementations of Deep Research emerged from the community, specifically from 

- [dzhng](https://x.com/dzhng/status/1886603396578484630),
- [assafelovic](https://github.com/assafelovic/gpt-researcher),
- [nickscamara](https://github.com/nickscamara/open-deep-research),
- [jina-ai](https://github.com/jina-ai/node-DeepResearch) and
- [mshumer](https://x.com/mattshumer_/status/1886558939434664404).

Each of these implementations use different libraries for indexing data, browsing the web and querying LLMs. In this project, we would like to **reproduce the benchmarks presented by OpenAI (pass@1 average score), benchmark and document our findings with switching to open LLMs (like DeepSeek R1), using vision LMs,  benchmark traditional tool calling against code-native agents.**

## Most important next steps

OpenAI’s Deep Research is probably boosted by the excellent web browser that they introduced with [Operator](https://openai.com/index/introducing-operator/).

So we’re tackling that next! In a more general problem: we’re going to build GUI agents, i.e. “agents that view your screen and can act directly with mouse & keyboard”. If you’re excited about this project, and want to help everyone get access to such cool capabilities through open source, we’d love to get your contribution!

We’re also [hiring a full time engineer](https://apply.workable.com/huggingface/j/AF1D4E3FEB/) to help us work on this and more, apply if you’re interested 🙂

- To get started with Open Deep Research, try the examples [here](https://github.com/huggingface/smolagents/tree/gaia-submission-r1/examples/open_deep_research).
- Check the [smolagents](https://github.com/huggingface/smolagents) repo.
- Read more about smolagents [docs](https://huggingface.co/docs/smolagents/index), [introduction blog post](https://huggingface.co/blog/smolagents).
