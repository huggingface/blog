---
title: "Red-Teaming Large Language Models" 
thumbnail: /blog/assets/red-teaming/thumbnail.png
authors:
- user: nazneen
- user: natolambert
- user: lewtun
---

# Red-Teaming Large Language Models


*Warning: This article is about red-teaming and as such contains examples of model generation that may be offensive or upsetting.*

Large language models (LLMs) trained on an enormous amount of text data are very good at generating realistic text. However, these models often exhibit undesirable behaviors like revealing personal information (such as social security numbers) and generating misinformation, bias, hatefulness, or toxic content. For example, earlier versions of GPT3 were known to exhibit sexist behaviors (see below) and [biases against Muslims](https://dl.acm.org/doi/abs/10.1145/3461702.3462624),

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/gpt3.png"/>
</p>

Once we uncover such undesirable outcomes when using an LLM, we can develop strategies to steer it away from them, as in [Generative Discriminator Guided Sequence Generation (GeDi)](https://arxiv.org/pdf/2009.06367.pdf) or [Plug and Play Language Models (PPLM)](https://arxiv.org/pdf/1912.02164.pdf) for guiding generation in GPT3. Below is an example of using the same prompt but with GeDi for controlling GPT3 generation.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/gedi.png"/>
</p>

Even recent versions of GPT3 produce similarly offensive text when attacked with prompt injection that can become a security concern for downstream applications as discussed in [this blog](https://simonwillison.net/2022/Sep/12/prompt-injection/).

**Red-teaming** *is a form of evaluation that elicits model vulnerabilities that might lead to undesirable behaviors.* Jailbreaking is another term for red-teaming wherein the LLM is manipulated to break away from its guardrails. [Microsoft’s Chatbot Tay](https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/) launched in 2016 and the more recent [Bing's Chatbot Sydney](https://www.nytimes.com/2023/02/16/technology/bing-chatbot-transcript.html) are real-world examples of how disastrous the lack of thorough evaluation of the underlying ML model using red-teaming can be. The origins of the idea of a red-team traces back to adversary simulations and wargames performed by militaries.


The goal of red-teaming language models is to craft a prompt that would trigger the model to generate text that is likely to cause harm. Red-teaming shares some similarities and differences with the more well-known form of evaluation in ML called *adversarial attacks*. The similarity is that both red-teaming and adversarial attacks share the same goal of “attacking” or “fooling” the model to generate content that would be undesirable in a real-world use case. However, adversarial attacks can be unintelligible to humans, for example, by prefixing the string “aaabbbcc” to each prompt because it deteriorates model performance. Many examples of such attacks on various NLP classification and generation tasks is discussed  in [Wallace et al., ‘19](https://arxiv.org/abs/1908.07125). Red-teaming prompts, on the other hand, look like regular, natural language prompts.

Red-teaming can reveal model limitations that can cause upsetting user experiences or enable harm by aiding violence or other unlawful activity for a user with malicious intentions. The outputs from red-teaming (just like adversarial attacks) are generally used to train the model to be less likely to cause harm or steer it away from undesirable outputs.

Since red-teaming requires creative thinking of possible model failures, it is a problem with a large search space making it resource intensive. A workaround would be to augment the LLM with a classifier trained to predict whether a given prompt contains topics or phrases that can possibly lead to offensive generations and if the classifier predicts the prompt would lead to a potentially offensive text, generate a canned response. Such a strategy would err on the side of caution. But that would be very restrictive and cause the model to be frequently evasive. So, there is tension between the model being *helpful* (by following instructions) and being *harmless* (or at least less likely to enable harm).

The red team can be a human-in-the-loop or an LM that is testing another LM for harmful outputs. Coming up with red-teaming prompts for models that are fine-tuned for safety and alignment (such as via RLHF or SFT) requires creative thinking in the form of *roleplay attacks* wherein the LLM is instructed to behave as a malicious character [as in Ganguli et al., ‘22](https://arxiv.org/pdf/2209.07858.pdf). Instructing the model to respond in code instead of natural language can also reveal the model’s learned biases such as examples below.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jb1.png"/>
</p>
<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jb0.png"/>
</p>
<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jb2.png"/>
</p>
<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jb3.png"/>
</p>

See [this](https://twitter.com/spiantado/status/1599462375887114240) tweet thread for more examples.

Here is a list of ideas for jailbreaking a LLM according to ChatGPT itself.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/red-teaming/jailbreak.png"/>
</p>

Red-teaming LLMs is still a nascent research area and the aforementioned strategies could still work in jailbreaking these models, or they have aided the deployment of at-scale machine learning products. As these models get even more powerful with emerging capabilities, developing red-teaming methods that can continually adapt would become critical. Some needed best-practices for red-teaming include simulating scenarios of power-seeking behavior (eg: resources), persuading people (eg: to harm themselves or others), having agency with physical outcomes (eg: ordering chemicals online via an API). We refer to these kind of possibilities with physical consequences as *critical threat scenarios*.

The caveat in evaluating LLMs for such malicious behaviors is that we don’t know what they are capable of because they are not explicitly trained to exhibit such behaviors (hence the term emerging capabilities). Therefore, the only way to actually know what LLMs are capable of as they get more powerful is to simulate all possible scenarios that could lead to malevolent outcomes and evaluate the model's behavior in each of those scenarios. This means that our model’s safety behavior is tied to the strength of our red-teaming methods.

Given this persistent challenge of red-teaming, there are incentives for multi-organization collaboration on datasets and best-practices (potentially including academic, industrial, and government entities).
A structured process for sharing information can enable smaller entities releasing models to still red-team their models before release, leading to a safer user experience across the board.

**Open source datasets for Red-teaming:**

1. Meta’s [Bot Adversarial Dialog dataset](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/bot_adversarial_dialogue)
2. Anthropic’s [red-teaming attempts](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/red-team-attempts)
3. AI2’s [RealToxicityPrompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts)

**Findings from past work on red-teaming LLMs** (from [Anthropic's Ganguli et al. 2022](https://arxiv.org/abs/2209.07858) and [Perez et al. 2022](https://arxiv.org/abs/2202.03286))

1. Few-shot-prompted LMs with helpful, honest, and harmless behavior are *not* harder to red-team than plain LMs.
2. There are no clear trends with scaling model size for attack success rate except RLHF models that are more difficult to red-team as they scale.
3. Models may learn to be harmless by being evasive, there is tradeoff between helpfulness and harmlessness.
4. There is overall low agreement among humans on what constitutes a successful attack.
5. The distribution of the success rate varies across categories of harm with non-violent ones having a higher success rate.
6. Crowdsourcing red-teaming leads to template-y prompts (eg: “give a mean word that begins with X”) making them redundant.

**Future directions:**

1. There is no open-source red-teaming dataset for code generation that attempts to jailbreak a model via code, for example, generating a program that implements a DDOS or backdoor attack.
2. Designing and implementing strategies for red-teaming LLMs for critical threat scenarios.
3. Red-teaming can be resource intensive, both compute and human resource and so would benefit from sharing strategies, open-sourcing datasets, and possibly collaborating for a higher chance of success.
4. Evaluating the tradeoffs between evasiveness and helpfulness.
5. Enumerate the choices based on the above tradeoff and explore the pareto front for red-teaming (similar to [Anthropic's Constitutional AI](https://arxiv.org/pdf/2212.08073.pdf) work)


These limitations and future directions make it clear that red-teaming is an under-explored and crucial component of the modern LLM workflow.
This post is a call-to-action to LLM researchers and HuggingFace's community of developers to collaborate on these efforts for a safe and friendly world :)

Reach out to us (@nazneenrajani @natolambert @lewtun @TristanThrush @yjernite @thomwolf) if you're interested in joining such a collaboration.

*Acknowledgement:* We'd like to thank [Yacine Jernite](https://huggingface.co/yjernite) for his helpful suggestions on correct usage of terms in this blogpost.

