---
title: "Judge Arena: Benchmarking LLMs as Evaluators"
thumbnail: /blog/assets/arenas-on-the-hub/thumbnail_atla.png
authors:
- user: kaikaidai
  guest: true
  org: AtlaAI
- user: MauriceBurg
  guest: true
  org: AtlaAI
- user: RomanEngeler1805
  guest: true
  org: AtlaAI
- user: mbartolo
  guest: true
  org: AtlaAI
- user: clefourrier
  org: huggingface
- user: tobydrane
  guest: true
  org: AtlaAI
- user: mathias-atla
  guest: true
  org: AtlaAI
- user: jacksongolden
  guest: true
  org: AtlaAI
---

# Judge Arena: Benchmarking LLMs as Evaluators

LLM-as-a-Judge has emerged as a popular way to grade natural language outputs from LLM applications, **but how do we know which models make the best judges**? 

We’re excited to launch [Judge Arena](https://huggingface.co/spaces/AtlaAI/judge-arena) - a platform that lets anyone easily compare models as judges side-by-side. Just run the judges on a test sample and vote which judge you agree with most. The results will be organized into a leaderboard that displays the best judges.

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/5.5.0/gradio.js"></script>

<gradio-app src="https://atlaai-judge-arena.hf.space"></gradio-app>


## Judge Arena

Crowdsourced, randomized battles have proven effective at benchmarking LLMs. LMSys's Chatbot Arena has collected over 2M votes and is [highly regarded](https://x.com/karpathy/status/1737544497016578453) as a field-test to identify the best language models. Since LLM evaluations aim to capture human preferences, direct human feedback is also key to determining which AI judges are most helpful.

### How it works

1. Choose your sample for evaluation:
- Let the system randomly generate a 👩 User Input / 🤖 AI Response pair
- OR input your own custom sample

2. Two LLM judges will:
- Score the response
- Provide their reasoning for the score

3. Review both judges’ evaluations and vote for the one that best aligns with your judgment
    
    *(We recommend reviewing the scores first before comparing critiques)*
    

After each vote, you can:

- **Regenerate judges:** Get new evaluations of the same sample
- Start a **🎲 New round:** Randomly generate a new sample to be evaluated
- OR, input a new custom sample to be evaluated

To avoid bias and potential abuse, the model names are only revealed after a vote is submitted.

## Selected Models

Judge Arena focuses on the LLM-as-a-Judge approach, and therefore only includes generative models (excluding classifier models that solely output a score). We formalize our selection criteria for AI judges as the following:

1. **The model should possess the ability to score AND critique other models' outputs effectively.**
2. **The model should be prompt-able to evaluate in different scoring formats, for different criteria.**

We selected 18 state-of-the-art LLMs for our leaderboard. While many are open-source models with public weights, we also included proprietary API models to enable direct comparison between open and closed approaches.

- **OpenAI** (GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo)
- **Anthropic** (Claude 3.5 Sonnet / Haiku, Claude 3 Opus / Sonnet / Haiku)
- **Meta** (Llama 3.1 Instruct Turbo 405B / 70B / 8B)
- **Alibaba** (Qwen 2.5 Instruct Turbo 7B / 72B, Qwen 2 Instruct 72B)
- **Google** (Gemma 2 9B / 27B)
- **Mistral** (Instruct v0.3 7B, Instruct v0.1 7B)

The current list represents the models most commonly used in AI evaluation pipelines. We look forward to adding more models if our leaderboard proves to be useful.

## The Leaderboard

The votes collected from the Judge Arena will be compiled and displayed on a dedicated public leaderboard. We calculate an [Elo score](https://en.wikipedia.org/wiki/Elo_rating_system) for each model and will update the leaderboard hourly.

## Early Insights

These are only very early results, but here’s what we’ve observed so far:

- **Mix of top performers between proprietary and open source**: GPT-4 Turbo leads by a narrow margin but the Llama and Qwen models are extremely competitive, surpassing the majority of proprietary models
- **Smaller models show impressive performance:** Qwen 2.5 7B and Llama 3.1 8B are performing remarkably well and competing with much larger models. As we gather more data, we hope to better understand the relationship between model scale and judging ability
- **Preliminary empirical support for emerging research:** LLM-as-a-Judge literature suggests that Llama models are well-suited as base models, demonstrating strong out-of-the-box performance on evaluation benchmarks. Several approaches including [Lynx](https://arxiv.org/pdf/2407.08488), [Auto-J](https://arxiv.org/pdf/2310.05470), and [SFR-LLaMA-3.1-Judge](https://arxiv.org/pdf/2409.14664) opted to start with Llama models before post-training for evaluation capabilities. Our provisional results align with this trend, showing Llama 3.1 70B and 405B ranking 2nd and 3rd, respectively

As the leaderboard shapes out over the coming weeks, we look forward to sharing further analysis on results on our [blog](https://www.atla-ai.com/blog).

## How to contribute

We hope the [Judge Arena](https://huggingface.co/spaces/AtlaAI/judge-arena) is a helpful resource for the community. By contributing to this leaderboard, you’ll help developers determine which models to use in their evaluation pipeline. We’re committed to sharing 20% of the anonymized voting data in the coming months as we hope developers, researchers and users will leverage our findings to build more aligned evaluators. 

We’d love to hear your feedback! For general feature requests or to submit / suggest new models to add to the arena, please open up a discussion in the [community](https://huggingface.co/spaces/AtlaAI/judge-arena/discussions) tab or talk to us on [Discord](https://discord.gg/yNpUAMqs). Don’t hesitate to let us know if you have questions or suggestions by messaging us on [X/Twitter](https://x.com/Atla_AI).

[Atla](https://www.atla-ai.com/) currently funds this out of our own pocket. We are looking for API credits (with no strings attached) to support this community effort - please get in touch at [support@atla-ai.com](mailto:support@atla-ai.com) if you are interested in collaborating 🤗

## Credits

Thanks to all the folks who helped test this arena and shout out to the LMSYS team for the inspiration. Special mention to Clémentine Fourrier and the Hugging Face team for making this possible!
