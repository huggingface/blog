---
title: "Open-R1: a fully open reproduction of DeepSeek-R1"
thumbnail: /blog/assets/open-r1/thumbnails.png
authors:
  - user: eliebak
  - user: lvwerra
  - user: lewtun
---

# Open-R1: a fully open reproduction of DeepSeek-R1

## What is DeepSeek-R1?

If you’ve ever struggled with a tough math problem, you know how useful it is to think a little longer and work through it carefully. [OpenAI’s o1 model](https://x.com/polynoamial/status/1834280155730043108) showed that when LLMs are trained to do the same—by using more compute during inference—they get significantly better at solving reasoning tasks like mathematics, coding, and logic.

However, the recipe behind OpenAI’s reasoning models has been a well kept secret. That is, until last week, when DeepSeek released their [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) model and promptly broke the internet (and the [stock market!](https://x.com/KobeissiLetter/status/1883831022149927352)).

Besides performing as well or better than o1, the [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) release was accompanied by a detailed [tech report](https://github.com/deepseek-ai/DeepSeek-R1/tree/main) that outlined the key steps of their training recipe. This recipe involved several innovations, most notably the application of pure reinforcement learning to teach a base language model how to reason without ***any*** human supervision. As shown in the figure below, making a powerful reasoning model is now very simple if you have access to a capable base model and a high-quality data mixture:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-r1/rl.png" alt="DeepSeek-R1 training pipeline"/>

However, the DeepSeek-R1 release leaves open several questions about:

- **Data collection:** How were the reasoning-specific datasets curated?
- **Model training:** No training code was released by DeepSeek, so it is unknown which hyperparameters work best and how they differ across different model families and scales.
- **Scaling laws:** What are the compute and data trade-offs in training reasoning models?

These questions prompted us to launch the [Open-R1 project](https://github.com/huggingface/open-r1), an initiative to systematically reconstruct DeepSeek-R1’s data and training pipeline, validate its claims, and push the boundaries of open reasoning models. By building Open-R1, we aim to provide transparency on how reinforcement learning can enhance reasoning, share reproducible insights with the open-source community, and create a foundation for future models to leverage these techniques.

In this blog post we take a look at key ingredients behind DeepSeek-R1, which parts we plan to replicate, and how to contribute to the Open-R1 project.

Let’s dive in 🚀!

## How did they do it?

DeepSeek-R1 is a reasoning model built on the foundation of [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base). Like any good reasoning model, it starts with a strong base model, and DeepSeek-V3 is exactly that. This 671B Mixture of Experts (MoE) model performs on par with heavyweights like Sonnet 3.5 and GPT-4o. What’s especially impressive is how cost-efficient it was to train—just $5.5M—thanks to architectural changes like Multi Token Prediction (MTP), Multi-Head Latent Attention (MLA) and a LOT (seriously, a lot) of hardware optimization.

DeepSeek also introduced two models: DeepSeek-R1-Zero and DeepSeek-R1, each with a distinct training approach. DeepSeek-R1-Zero skipped supervised fine-tuning altogether and relied entirely on reinforcement learning (RL), using Group Relative Policy Optimization (GRPO) to make the process more efficient. A simple reward system was used to guide the model, providing feedback based on the accuracy and structure of its answers. This approach helped the model develop useful reasoning skills, such as breaking problems into steps and verifying its own outputs. However, its responses often lacked clarity and were difficult to read.

That’s where DeepSeek-R1 comes in. It started with a "cold start" phase, fine-tuning on a small set of carefully crafted examples to improve clarity and readability. From there, it went through more RL and refinement steps, including rejecting low-quality outputs with both human preference based and verifiable reward, to create a model that not only reasons well but also produces polished and consistent answers.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-r1/arch.png" alt="DeepSeek-V3 architecture"/>

This all sounds great, but what's actually missing? Let's have a look at the missing pieces of the puzzle.

## Open-R1: the missing pieces

The release of DeepSeek-R1 is an amazing boon for the community, but they didn’t release *everything—*although the model weights are open, the datasets and code used to train the model are not 😢.

The goal of *Open-R1* is to build these last missing pieces so that the whole research and industry community can build similar or better models using these recipes and datasets. And by doing this in the open, everybody in the community can contribute!

As shown in the figure below, here’s our plan of attack:

- **Step 1:** Replicate the R1-Distill models by distilling a high-quality reasoning dataset from DeepSeek-R1.
- **Step 2:** Replicate the pure RL pipeline that DeepSeek used to create R1-Zero. This will involve curating new, large-scale datasets for math, reasoning, and code.
- **Step 3:** Show we can go from base model → SFT → RL via multi-stage training.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-r1/steps.png" alt="Open-R1 steps"/>
The synthetic datasets will allow everybody to fine-tune existing or new LLMs into reasoning models by simply fine-tuning on them. The training recipes involving RL will serve as a starting point for anybody to build similar models from scratch and will allow researchers to build even more advanced methods on top.

Note that we don’t want to stop at math datasets. There’s a lot of potential in exploring other areas, obvious one like code but also scientific fields such as medicine, where reasoning models could have significant impact.

This initiative isn’t just about replicating results—it’s about sharing insights with the community. By documenting what works, what doesn’t, and why, we hope to save others from wasting time and compute on unproductive paths.

If this sounds interesting, we’d love your help! Whether it’s contributing [code](https://github.com/huggingface/open-r1/issues/23), joining discussions on [Hugging Face](https://huggingface.co/open-r1), there are plenty of ways to get involved. Let’s build this together! 🚀