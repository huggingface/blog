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
    <b>Not legal advice.</b>
</div>

*The EU AI Act, the world’s first comprehensive legislation on artificial intelligence, has officially come into force, and it’s set to impact the way we develop and use AI – including in the open source community. If you’re an open source developer navigating this new landscape, you’re probably wondering what this means for your projects. This guide breaks down key points of the regulation with a focus on open source development, offering a clear introduction to this legislation and directing you to tools that may help you prepare to comply with it.*

**_Disclaimer: The information provided in this guide is for informational purposes only, and should not be considered as any form of legal advice._**

> **TL;DR:** The AI Act may apply to open source AI systems and models, with specific rules depending on the type of model and how they are released. In most cases, obligations involve providing clear documentation, adding tools to disclose model information when deployed, and following existing copyright and privacy rules. Fortunately, many of these practices are already common in the open source landscape, and Hugging Face offers tools to help you prepare to comply, including tools to support opt-out processes and redaction of personal data.
Check out [model cards](https://huggingface.co/docs/hub/en/model-cards), [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards), [Gradio](https://www.gradio.app/docs/gradio/video) [watermarking](https://huggingface.co/spaces/meg/watermark_demo), [support](https://techcrunch.com/2023/05/03/spawning-lays-out-its-plans-for-letting-creators-opt-out-of-generative-ai-training/) for [opt-out](https://huggingface.co/spaces/bigcode/in-the-stack) mechanisms and [personal data redaction](https://huggingface.co/blog/presidio-pii-detection), [licenses](https://huggingface.co/docs/hub/en/repositories-licenses) and others!

The EU AI Act is a binding regulation that aims to foster responsible AI. To that end, it sets out rules that scale with the level of risk the AI system or model might pose while aiming to preserve open research and support small and medium-sized enterprises (SMEs). As an open source developer, many aspects of your work won’t be directly impacted – especially if you’re already documenting your systems and keeping track of data sources. In general, there are straightforward steps you can take to prepare for compliance. 

The regulation takes effect over the next two years and applies broadly, not just to those within the EU. If you’re an open source developer outside the EU but your AI systems or models are offered or impact people within the EU, they are included in the Act. 


## 🤗 Scope
The regulation works at different levels of the AI stack, meaning it has different obligations if you are a provider (which includes the developers), deployer, distributor etc. and if you are working on an AI model or system.

| **Model**: only **general purpose AI** (GPAI) models are directly regulated. GPAI models are models trained on large amounts of data, that show significant generality, can perform a wide range of tasks and can be used in systems and applications. One example is a large language model (LLM). Modifications or fine-tuning of models also need to comply with obligations. | **System**: a system that is able to infer from inputs. This could typically take the form of a traditional software stack that leverages or connects one or several AI models to a digital representation of the inputs. One example is a chatbot interacting with end users, leveraging an LLM or Gradio apps hosted on Hugging Face Spaces. |
|---|---|

In the AI Act, rules scale with the level of risk the AI system or model might pose. For all AI systems, risks may be:

* **[Unacceptable](https://artificialintelligenceact.eu/article/5/)**: systems that violate human rights, for example an AI system that scrapes facial images from the internet or CCTV footage. These systems are prohibited and cannot be put on the market.
* **[High](https://artificialintelligenceact.eu/article/6/)**: systems that may adversely impact people’s safety or fundamental rights, for example dealing with critical infrastructure, essential services, law enforcement. These systems need to follow thorough compliance steps before being put on the market.
* **[Limited](https://artificialintelligenceact.eu/article/50/)**: systems that interact directly with people and have the potential to create risks of impersonation, manipulation, or deception. These systems need to meet transparency requirements. Most generative AI models can be integrated into systems that fall into this category. As a model developer, your models will be easier and more likely to be integrated into AI systems if you already follow the requirements, such as by providing sufficient documentation.
* **Minimal**: the majority of the systems - that don’t pose the risks above. They need only comply with existing laws and regulations, no obligation is added with the AI Act.

For **general purpose AI** (GPAI) **models**, there is another risk category called **[systemic risk](https://artificialintelligenceact.eu/article/51/)**: GPAI models using substantial computing power, today defined as over 10^25 FLOPs for training, or that have high-impact capabilities. Obligations vary if they are open source or not.

## 🤗 How to prepare for compliance

Our **focus** in this short guide is on **limited risk AI systems and open source non-systemic risk GPAI models**, which should encompass most of what is publicly available on the Hub. For other risk categories, make sure to check out further obligations that may apply.

### For limited risk AI systems

Limited-risk AI systems interact directly with people (end users) and may create risks of impersonation, manipulation, or deception. For example, a chatbot producing text or a text-to-image generator – tools that can also facilitate the creation of misinformation materials or of deepfakes. The AI Act aims to tackle these risks by helping the general end user understand that they are interacting with an AI system. Today, most GPAI models are not considered to present systemic risk. According to a [study by Stanford](https://crfm.stanford.edu/2024/08/01/eu-ai-act.html), in August 2024, based on estimates from Epoch, only eight models (Gemini 1.0 Ultra, Llama 3.1-405B, GPT-4, Mistral Large, Nemotron-4 340B, MegaScale, Inflection-2, Inflection-2.5) from seven developers (Google, Meta, OpenAI, Mistral, NVIDIA, ByteDance, Inflection) would meet the default systemic risk criterion of being trained using at least 10^25 FLOPs. In the case of limited-risk AI systems, the obligations below apply whether or not they are open source.

Developers of limited-risk AI systems need to:
* Disclose to the user that they are interacting with an AI system unless this is obvious, keeping in mind that end users might not have the same technical understanding as experts, so you should provide this information in a clear and thorough way.
* Mark synthetic content: AI-generated content (e.g., audio, images, videos, text) must be clearly marked as artificially generated or manipulated in a machine-readable format. Existing tools like Gradio’s [built-in watermarking features](https://huggingface.co/spaces/meg/watermark_demo) can help you meet these requirements.

Note that you may also be a ‘deployer’ of an AI system, not only a developer. Deployers of AI systems are people or companies using an AI system in their professional capacity. In that case, you also need to comply with the following:

* For emotion recognition and biometric systems: deployers must inform individuals about the use of these systems and process personal data in accordance with relevant regulations.
* Disclosure of deepfakes and AI-generated content: deployers must disclose when AI-generated content is used. When the content is part of an artistic work, the obligation is to disclose that generated or manipulated content exists in a way that does not spoil the experience.

The information above needs to be provided with clear language, at the latest at the time of the user’s first interaction with, or exposure, to the AI system.

The AI Office, in charge of implementing the AI Act, will help create codes of practice with guidelines for detecting and labeling artificially generated content. These codes are currently being written with industry and civil society participation, and are expected to be published by May 2025. Obligations will be enforced starting August 2026.

### For open source non-systemic risk GPAI models

The following obligations apply if you are developing open source GPAI models, e.g. LLMs, that do not present systemic risk. **Open source for the AI Act means “software and data, including models, released under a free and open source license that allows them to be openly shared and where users can freely access, use, modify and redistribute them or modified versions thereof”.** Developers can select from a list of [open licenses on the Hub](https://huggingface.co/docs/hub/en/repositories-licenses). Check if the chosen license fits the [AI Act’s open source definition](https://ai-act-law.eu/recital/102/).

The obligations for non-systemic open source GPAI models are as follows:
* Draft and make available a sufficiently detailed summary of the content used to train the GPAI model, according to a template provided by the AI Office.
    * The level of detail of the content is still under discussion but should be relatively comprehensive.
* Implement a policy to comply with EU law on copyright and related rights, notably to comply with opt-outs. Developers need to ensure they are authorized to use copyright-protected material, which can be obtained with the authorization of the rightsholder or when copyright exceptions and limitations apply. One of these exceptions is the Text and Data Mining (TDM) exception, a technique used extensively in this context for retrieving and analyzing content. However, the TDM exception generally does not apply when a rightsholder clearly expresses that they reserve the right to use their work for these purposes – this is called “opt-out.” In establishing a policy to comply with the EU Copyright Directive, these opt-outs should be respected and restrict or ban use of the protected material. In other words, training on copyrighted material is not illegal if you respect the authors’ decision to opt-out of AI training.
    * While there are still open questions about how opt-outs should be expressed technically, especially in machine-readable formats, respect of information expressed in robots.txt files for websites and leveraging tools like [Spawning](https://spawning.ai/)’s API are a good start.

The EU AI Act also ties into existing regulations on copyright and personal data, such as [copyright directive](https://eur-lex.europa.eu/eli/dir/2019/790/oj) and [data protection regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). For this, look to Hugging Face-integrated tools that [support](https://techcrunch.com/2023/05/03/spawning-lays-out-its-plans-for-letting-creators-opt-out-of-generative-ai-training/) better [opt-out ](https://huggingface.co/spaces/bigcode/in-the-stack)mechanisms and [personal data redaction](https://huggingface.co/blog/presidio-pii-detection), and stay updated on recommendations from European and national bodies like [CNIL](https://www.cnil.fr/fr/ai-how-to-sheets).

Projects on Hugging Face have implemented forms of understanding and implementing opt-outs of training data, such as BigCode’s [Am I In The Stack](https://huggingface.co/spaces/bigcode/in-the-stack) app and the [integration of a Spawning widget](https://techcrunch.com/2023/05/03/spawning-lays-out-its-plans-for-letting-creators-opt-out-of-generative-ai-training/) for datasets with image URLs. With these tools, creators can simply opt out of allowing their copyrighted material to be used for AI training. As opt-out processes are being developed to help creators effectively inform publicly that they do not want their content used for AI training, these tools can be quite effective in addressing those decisions.

Developers may rely on codes of practice (which are currently being developed and expected by May 2025) to demonstrate compliance with these obligations. 

Other obligations apply if you make your work available in a way that does not meet the criteria for being **open source** according to the AI Act. 

Also, note that if a given GPAI model meets the conditions to pose systemic risks, its developers must notify the EU Commission. In the notification process, developers can argue that their model does not present systemic risks because of specific characteristics. The Commission will review each argument and accept or reject the claim depending on whether the argument is sufficiently substantiated, considering the model’s specific characteristics and capabilities. If the Commission rejects the developers’ arguments, the GPAI model will be designated as posing systemic risk and will need to comply with further obligations, such as providing technical documentation on the model including its training and testing process and the results of its evaluation.

Obligations for GPAI models will be enforced starting August 2025. 

## 🤗 Get involved

Much of the EU AI Act’s practical application is still in development through public consultations and working groups, whose outcome will determine how the Act’s provisions aimed at smoother compliance for SMEs and researchers are operationalized. If you’re interested in shaping how this plays out, now is a great time to get involved!

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

*Thank you, Anna Tordjmann, Brigitte Tousignant, Chun Te Lee, Irene Solaiman, Clémentine Fourrier, Ann Huang, Benjamin Burtenshaw, Florent Daudens for your feedback, comments, and suggestions.*
