---
title: "Ettin Suite: SoTA Paired Encoders and Decoders" 
thumbnail: /blog/assets/ettin/image.png
authors:
- user: orionweller
  guest: true
  org: jhu-clsp
- user: kdricci
  guest: true
  org: jhu-clsp
- user: mmarone
  guest: true
  org: jhu-clsp
- user: NohTow
  guest: true
  org: lightonai
- user: dlawrie
  guest: true
  org: jhu-clsp
- user: vandurme
  guest: true
  org: jhu-clsp
---

# Seq vs Seq: the Ettin Suite of Paired Encoders and Decoders

Small language models are becoming increasingly important as users seek capable models that can be deployed efficiently. The community has produced a fascinating range of capable small models, each pushing the boundaries of what's possible at this scale. With SmolLM3, we're excited to contribute a new competitive fully open 3B model:

- Base model: [https://hf.co/HuggingFaceTB/SmolLM3-3B-Base](https://hf.co/HuggingFaceTB/SmolLM3-3B-Base)
- Instruct and reasoning model: [https://hf.co/HuggingFaceTB/SmolLM3-3B](https://hf.co/HuggingFaceTB/SmolLM3-3B)

**SmolLM3 sits in the efficiency sweet spot.** Our 3B model outperforms Llama-3.2-3B and Qwen2.5-3B while staying competitive with larger 4B alternatives (Qwen3 & Gemma3). Beyond the performance numbers, we're sharing exactly how we built it using public datasets and training frameworks.

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(17).png" alt=""  style="width: 80%; height: auto;"><br>
</p>

Model summary:

- **3B model** trained on 11T tokens, SoTA at the 3B scale and competitive with 4B models
- **Instruct model** with **dual mode reasoning,** supporting `think`/`no_think` modes
- **Multilingual support** for 6 languages: English, French, Spanish, German, Italian, and Portuguese
- **Long context** up to 128k with NoPE and using YaRN

**The complete recipe:** We're releasing SmolLM3 with our engineering blueprint. It includes architecture details, exact data mixtures showing how we progressively boost performance across domains in a three-stage pretraining approach, and the methodology for building a hybrid reasoning model. Usually, achieving these results would require months of reverse engineering. Instead, we're providing the full methodology.

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/smollm3-whiteprint.png" alt=""  style="width: 90%; height: auto;"><br>
</p>

Whether you're building your own models or want to understand what drives performance at this scale, this blueprint shows the engineering story behind competitive 3B performance.

Let’s have a look at the pretraining stage.

# Pretraining
