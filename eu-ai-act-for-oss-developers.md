---
title: "Open Source Developers Guide to the EU AI Act"
thumbnail: /blog/assets/189_eu-ai-act-for-oss-developers/thumbnail.png
authors:
- user: brunatrevelin
- user: frimelle
- user: yjernite
---


# Open Source Developers Guide to the EU AI Act

<div style="text-align: center;">
    **Not legal advice.**
</div>

*The EU AI Act, the world’s first comprehensive legislation on artificial intelligence, has officially come into force, and it’s set to impact the way we develop and use AI – including in the open source community. If you’re an open source developer navigating this new landscape, you’re probably wondering what this means for your projects. This guide breaks down key points of the regulation with a focus on open source development, offering a clear introduction to this legislation and directing you to tools that may help you prepare to comply with it.*

**_Disclaimer: The information provided in this guide is for informational purposes only, and should not be considered as any form of legal advice._**

> **TL;DR:** The AI Act may apply to open source AI systems and models, with specific rules depending on the type of model and how they are released. In most cases, obligations involve providing clear documentation, adding tools to disclose model information when deployed, and following existing copyright and privacy rules. Fortunately, many of these practices are already common in the open source landscape, and Hugging Face offers tools to help you prepare to comply, including tools to support opt-out processes and redaction of personal data.
Check out [model cards](https://huggingface.co/docs/hub/en/model-cards), [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards), [Gradio](https://www.gradio.app/docs/gradio/video) [watermarking](https://huggingface.co/spaces/meg/watermark_demo), [support](https://techcrunch.com/2023/05/03/spawning-lays-out-its-plans-for-letting-creators-opt-out-of-generative-ai-training/) for [opt-out](https://huggingface.co/spaces/bigcode/in-the-stack) mechanisms and [personal data redaction](https://huggingface.co/blog/presidio-pii-detection), [licenses](https://huggingface.co/docs/hub/en/repositories-licenses) and others!

The EU AI Act is a binding regulation that aims to foster responsible AI. To that end, it sets out rules that scale with the level of risk the AI system or model might pose while aiming to preserve open research and support small and medium-sized enterprises (SMEs). As an open source developer, many aspects of your work won’t be directly impacted – especially if you’re already documenting your systems and keeping track of data sources. In general, there are straightforward steps you can take to prepare for compliance. 

The regulation takes effect over the next two years and applies broadly, not just to those within the EU. If you’re an open source developer outside the EU but your AI systems or models are offered or impact people within the EU, they are included in the Act. 


## Scope
The regulation works at different levels of the AI stack, meaning it has different obligations if you are a provider (which includes the developers), deployer, distributor etc. and if you are working on an AI model or system.

| **Model**: only **general purpose AI** (GPAI) models are directly regulated. GPAI models are models trained on large amounts of data, that show significant generality, can perform a wide range of tasks and can be used in systems and applications. One example is a large language model (LLM). | **System**: a system that is able to infer from inputs. This could typically take the form of a traditional software stack that leverages or connects one or several AI models to a digital representation of the inputs. One example is a chatbot interacting with end users, leveraging an LLM or Gradio apps hosted on Hugging Face Spaces. |
|---|---|

In the AI Act, rules scale with the level of risk the AI system or model might pose. For all AI systems, risks may be:

* **[Unacceptable](https://artificialintelligenceact.eu/article/5/)**: systems that violate human rights, for example an AI system that scrapes facial images from the internet or CCTV footage. These systems are prohibited and cannot be put on the market.
* **[High](https://artificialintelligenceact.eu/article/6/)**: systems that may adversely impact people’s safety or fundamental rights, for example dealing with critical infrastructure, essential services, law enforcement. These systems need to follow thorough compliance steps before being put on the market.
* **[Limited](https://artificialintelligenceact.eu/article/50/)**: systems that interact directly with people and have the potential to create risks of impersonation, manipulation, or deception. These systems need to meet transparency requirements. Most generative AI models can be integrated into systems that fall into this category. As a model developer, your models will be easier and more likely to be integrated into AI systems if you already follow the requirements, such as by providing sufficient documentation.
* **Minimal**: the majority of the systems - that don’t pose the risks above. They need only comply with existing laws and regulations, no obligation is added with the AI Act.

For **general purpose AI** (GPAI) **models**, there is another risk category called **[systemic risk](https://artificialintelligenceact.eu/article/51/)**: GPAI models using substantial computing power, today defined as over 10^25 FLOPs for training, or that have high-impact capabilities. Obligations vary if they are open source or not.

## How to prepare for compliance

```
@misc{eu_ai_act_for_oss_developers,
  author    = {Bruna Trevelin and Lucie-Aimée Kaffee and Yacine Jernite},
  title     = {Open Source Developers Guide to the EU AI Act},
  booktitle = {Hugging Face Blog},
  year      = {2024},
  url       = {},
  doi       = {}
}
```