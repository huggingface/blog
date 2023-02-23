---
title: "Red-Teaming Large Language Models" 
thumbnail: /blog/assets/red-teaming/thumbnail.png
---

# Red-Teaming Large Language Models

<div class="blog-metadata">
    <small>Published February 22, 2023.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/red-teaming.md">
        Update on GitHub
    </a>
</div>
<div class="author-card">
    <a href="/nazneen"> 
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/3278583?v=4?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>Nazneen</code>
            <span class="fullname">Nazneen Rajani</span>
        </div>
    </a>
    </div>
Large language models (LLMs) trained on an enormous amount of text data are very good at generating realistic text. However, these models often exhibit undesirable behaviors like revealing personal information (such as social security numbers) and generating misinformation, bias, hatefulness, or toxic content. For example, GPT3 is known to be sexist (see below) and [biased against Muslims](https://dl.acm.org/doi/abs/10.1145/3461702.3462624),

![GPT3](assets/red-teaming/gpt3.png)

Once we uncover such undesirable values in the LLM, we can develop strategies to steer it away from them, as in [GeDi](https://arxiv.org/pdf/2009.06367.pdf) or [PPLM](https://arxiv.org/pdf/1912.02164.pdf) for guiding generation in GPT3. Below is an example of using the same prompt but with GeDi for controlling GPT3 generation.

![GeDi](assets/red-teaming/gedi.png)

*Red-teaming is a type of evaluation that elicits model vulnerabilities that might lead to undesirable behaviors.*  [Microsoft’s Chatbot Tay](https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/) launched in 2016 is a real-world example of a lack of such thorough evaluation of the underlying ML model using red-teaming. Jailbreaking is another term for red-teaming wherein the LLM is manipulated to break away from its guardrails.

The goal of red-teaming language models is to craft a prompt that would trigger the model to generate offensive text. Red-teaming shares some similarities and differences with the more well-known form of evaluation in ML called *adversarial attacks*. The similarity is that both red-teaming and adversarial attacks share the same goal of “attacking” or “fooling” the model to generate offensive content. However, adversarial attacks can be unintelligible to humans, for example, by prefixing a random string (such as “aaabbbcc”) to each prompt as in [Wallace et al., ‘19.](https://aclanthology.org/D19-1221.pdf) Red-teaming prompts, on the other hand, look like regular, natural language prompts. 

Red-teaming can reveal model limitations that can uncover model limitations that could lead cause offensive and upsetting experiences or worse, aid violence and other unlawful activity for a user with malicious intentions. The outputs from red-teaming (just like adversarial attacks) can be used to train the model to be harmless or steer it away from undesirable outputs.

A workaround for red-teaming would be to augment the LLM with a classifier trained to predict whether a given prompt contains topics or phrases that can possibly lead to offensive generations and if so, generate a canned response. Such a strategy would err on the side of caution. But that would be very restrictive and cause the model to be frequently evasive. So, there is tension between the model being *helpful* (by following instructions) and being *harmless* (not generating offensive text). This is where red-teaming can be very useful.

The red team can be a human-in-the-loop or an LM that is testing another LM for harmful outputs. Coming up with red-teaming prompts for models that are fine-tuned for safety and alignment (such as via RLHF or SFT) requires creative thinking in the form of *roleplay attacks* wherein the LLM is instructed to behave as a malicious character [as in Ganguli et al., ‘22.](https://arxiv.org/pdf/2209.07858.pdf) Instructing the model to respond in code instead of natural language can also reveal the model’s learned biases like the one for ChatGPT in the following tweet thread. 
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Yes, ChatGPT is amazing and impressive. No, <a href="https://twitter.com/OpenAI?ref_src=twsrc%5Etfw">@OpenAI</a> has not come close to addressing the problem of bias. Filters appear to be bypassed with simple tricks, and superficially masked. <br><br>And what is lurking inside is egregious. <a href="https://twitter.com/Abebab?ref_src=twsrc%5Etfw">@Abebab</a> <a href="https://twitter.com/sama?ref_src=twsrc%5Etfw">@sama</a><br>tw racism, sexism. <a href="https://t.co/V4fw1fY9dY">pic.twitter.com/V4fw1fY9dY</a></p>&mdash; steven t. piantadosi (@spiantado) <a href="https://twitter.com/spiantado/status/1599462375887114240?ref_src=twsrc%5Etfw">December 4, 2022</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


Here is a list of ideas for jailbreaking a model according to ChatGPT itself.

![Screenshot 2023-02-22 at 8.51.57 AM.png](assets/red-teaming/jailbreak.png)

Red-teaming LLMs is still a nascent research area and the aforementioned strategies would work in jailbreaking these models. But as these models get even more powerful with emerging capabilities, developing red-teaming methods that can continually adapt would become critical. For example, simulating scenarios of power-seeking behavior (eg: resources), persuading people (eg: to harm themselves or others), having agency with physical outcomes (eg: ordering chemicals online via an API). We refer to these as *critical threat scenarios*.

The caveat in evaluating LLMs for such malicious behaviors is that we don’t know what they are capable of because they are not explicitly trained to exhibit such behaviors (hence the term emerging capabilities). The only way is to actually simulate scenarios and evaluate for the model would behave. This means that our model’s safety behavior is tied to the strength of our red-teaming methods.

**Open source datasets for Red-teaming:**

1. Meta’s [Bot Adversarial Dialog dataset](https://aclanthology.org/2021.naacl-main.235.pdf)
2. Anthropic’s [red-teaming attempts](https://github.com/anthropics/hh-rlhf/tree/master/red-team-attempts)
3. AI2’s [RealToxicityPrompts](https://arxiv.org/pdf/2009.11462.pdf) 

**Findings from past work on red-teaming LLMs** (from [https://arxiv.org/abs/2209.07858](https://arxiv.org/abs/2209.07858) and [https://arxiv.org/abs/2202.03286](https://arxiv.org/abs/2202.03286))

1. Few-shot-prompted LMs with helpful, honest, and harmless behavior are not harder to red-team than plain LMs.
2. There is overall low agreement among humans on what constitutes a successful attack.
3. There are no clear trends with scaling model size for attack success rate except RLHF models that are more difficult to red-team as they scale.
4. Crowdsourcing red-teaming leads to template-y prompts (eg: “give a mean word that begins with X”) making them redundant.

**Future directions:**

1. There is no open-source red-teaming dataset for code generation that attempts to jailbreak a model via code, for example, generating a program that implements a DDOS or backdoor attack.
2. Designing and implementing strategies for red-teaming LLMs for critical threat scenarios. 
3. Red-teaming can be resource intensive, both compute and human resource, so this is a call-to-action to the LLM community of researchers to collaborate on these efforts for a safe and friendly world :) 

