---
title: "Welcome the Falcon 3 Family of Open Models!" 
thumbnail: /blog/assets/falcon3/thumbnail.png
authors:
- Abdalgader
  guest: true
  org: tiiuae
- billelmokeddem
  guest: true
  org: tiiuae
- Dhia-GB
  guest: true
  org: tiiuae
- DhiyaEddine
  guest: true
  org: tiiuae
- fedyanin
  guest: true
  org: tiiuae
- gcamp
  guest: true
  org: tiiuae
- HakimHacid
  guest: true
  org: tiiuae
- HamzaYous89
  guest: true
  org: tiiuae
- hangzou
  guest: true
  org: tiiuae
- i-Chahed
  guest: true
  org: tiiuae
- ibrahim-khadraoui-TII
  guest: true
  org: tiiuae
- ifarhat1993
  guest: true
  org: tiiuae
- Iheb-Chaabane
  guest: true
  org: tiiuae
- JingweiZuo
  guest: true
  org: tiiuae
- karnakar
  guest: true
  org: tiiuae
- kasper-piskorski
  guest: true
  org: tiiuae
- melaseddik
  guest: true
  org: tiiuae
- Mughaira
  guest: true
  org: tiiuae
- puneeshkhanna
  guest: true
  org: tiiuae
- qiyang-zhao
  guest: true
  org: tiiuae
- rcojocaru
  guest: true
  org: tiiuae
- RedaAlami
  guest: true
  org: tiiuae
- shihux
  guest: true
  org: tiiuae
- slimfrikha-tii
  guest: true
  org: tiiuae
- wdevazelhes
  guest: true
  org: tiiuae
- ybelkada
  guest: true
  org: tiiuae
- yasserTII
  guest: true
  org: tiiuae
---

# Welcome the Falcon 3 Family of Open Models!

We introduce Falcon3, a family of decoder-only large language models under 10 billion parameters, developed by
[Technology Innovation Institute (TII)](https://www.tii.ae/ai-and-digital-science) in Abu Dhabi. By pushing the
boundaries of performance and training efficiency, this release reflects our ongoing commitment to advancing open
and accessible large foundation models.

Falcon3 represents a natural evolution from previous releases, emphasizing expanding the models' science, math, and code capabilities.

This iteration includes five base models:
1. Falcon3-1B-Base
2. Falcon3-3B-Base
3. Falcon3-Mamba-7B-Base
4. Falcon3-7B-Base
5. Falcon3-10B-Base

In developing these models, we incorporated several key innovations aimed at improving the models' performances while reducing training costs:
- **One pre-training for transformer-based models:** We conducted a single large-scale pretraining run on the 7B model, using 1024 H100 GPU chips, leveraging 14 trillion tokens featuring web, code, STEM, and curated high-quality and multilingual data.
- **Depth up-scaling for improved reasoning:** Building on recent studies on the effects of model depth, we upscaled the 7B model to a 10B parameters model by duplicating the redundant layers and continuing pre-training with 2 trillion tokens of high-quality data. This yielded Falcon3-10B-Base which achieves state-of-the-art zero-shot and few-shot performance for models under 13B parameters.
- **Knowledge distillation for better tiny models:** To provide compact and efficient alternatives, we developed Falcon3-1B-Base and Falcon3-3B-Base by leveraging pruning and knowledge distillation techniques, using less than 100GT of curated high-quality data, thereby redefining pre-training efficiency.
- **Pure SSM:** We have further enhanced [Falcon Mamba 7B](https://huggingface.co/tiiuae/falcon-mamba-7b) by training on an additional 2 trillion tokens of high-quality data, resulting in Falcon3-Mamba-7B-Base. Notably, the updated model offers significantly improved reasoning and mathematical capabilities.
- **Other variants:** All models in the Falcon3 family are available in variants such as Instruct, GGUF, GPTQ-Int4, GPTQ-Int8, AWQ, and 1.58-bit, offering flexibility for a wide range of applications.

## Key Highlights

Falcon3 featured the limits within the small and medium scales of large language models by demonstrating high performance on common benchmarks:
- [Falcon3-1B-Base](https://huggingface.co/tiiuae/Falcon3-1B-Base) surpasses SmolLM2-1.7B and is on par with gemma-2-2b.
- [Falcon3-3B-Base](https://huggingface.co/tiiuae/Falcon3-3B-Base) outperforms larger models like Llama-3.1-8B and Minitron-4B-Base, highlighting the benefits of pre-training with knowledge distillation.
- [Falcon3-7B-Base](https://huggingface.co/tiiuae/Falcon3-7B-Base) demonstrates top performance, on par with Qwen2.5-7B, among models under the 9B scale.
- [Falcon3-10B-Base](https://huggingface.co/tiiuae/Falcon3-10B-Base) stands as the state-of-the-art achieving strong results in the under-13B category.
- All the transformer-based Falcon3 models are compatible with [Llama](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) architecture allowing better integration in the AI ecosystem.
- [Falcon3-Mamba-7B](https://huggingface.co/tiiuae/Falcon3-Mamba-7B-Base) continues to lead as the top-performing State Space Language Model (SSLM), matching or even surpassing leading transformer-based LLMs at the 7B scale, along with support for a longer 32K context length. Having the same architecture as the original [Falcon Mamba 7B](https://huggingface.co/tiiuae/falcon-mamba-7b), users can integrate Falcon3-Mamba-7B seamlessly without any additional effort.
- The instruct versions of our collection of base models further show remarkable performance across various benchmarks with Falcon3-7B-Instruct and Falcon3-10B-Instruct outperforming all instruct models under the 13B scale on the open leaderboard.

## Enhanced Capabilities

Our evaluations highlight key areas where the Falcon3 family of models excel, reflecting the emphasis on enhancing performance in scientific domains, reasoning, and general knowledge capabilities:
- **Math Capabilities:** Falcon3-10B-Base achieves 24.77 on MATH-Lvl5 and 83.0 on GSM8K, showcasing enhanced reasoning in complex math-focused tasks.
- **Coding Capabilities:** Falcon3-10B-Base achieves 73.8 on MBPP, while Falcon3-10B-Instruct scores 45.8 on Multipl-E, reflecting their abilities to generalize across programming-related tasks.
- **Extended Context Length**: Models in the Falcon3 family support up to 32k tokens (except the 1B supporting up to 8k context), with functional improvements such as scoring 86.3 on BFCL (Falcon3-10B-Instruct).
- **Improved Reasoning:** Falcon3-7B-Base and Falcon3-10B-Base achieve 51.0 and 59.7 on BBH, reflecting enhanced reasoning capabilities, with the 10B model showing improved reasoning performance over the 7B.
- **Scientific Knowledge Expansion:** Performance on MMLU benchmarks demonstrates advances in specialized knowledge, with scores of 67.4/39.2 (MMLU/MMLU-PRO) for Falcon3-7B-Base and 73.1/42.5 (MMLU/MMLU-PRO) for Falcon3-10B-Base respectively. 

## Models' Specs and Benchmark Results

Detailed specifications of the Falcon3 family of models are summarized in the following table. The architecture of [Falcon3-7B-Base](https://huggingface.co/tiiuae/Falcon3-7B-Base)
is characterized by a head dimension of 256 which yields high throughput when using [FlashAttention-3](https://arxiv.org/abs/2407.08608) as it is optimized for this dimension. These decoder-only models span 18 to 40 layers for the transformer-based ones, and 64 layers for the mamba one, all  models share the SwiGLU activation function, with vocabulary size of 131K tokens (65Kfor Mamba-7B). The Falcon3-7B-Base is trained on the largest amount of data ensuring comprehensive coverage of concepts and knowledge, the other variants require way less data.
<br/><br/>
<!-- ![Falcon 3 Specs](Falcon3-specs.png) -->
<div style="text-align: center;" align="center">
    <img src="https://huggingface.co/datasets/tiiuae/documentation-images/resolve/main/general/Falcon3-specs.png" alt="Training efficiency" width="750">
</div>
<br/><br/>

The table below highlights the performances of Falcon3-7B-Base and Falcon3-10B-Base on key benchmarks showing competitive performances in general, math, reasoning, and common sense understanding domains. 
 <br/><br/>
<!-- ![medium base models' performances](medium-base-models.png)  -->
<div style="text-align: center;" align="center">
    <img src="https://huggingface.co/datasets/tiiuae/documentation-images/resolve/main/general/medium-base-models.png" alt="Training efficiency" width="800">
</div>
<br/><br/>

The instruct models also demonstrate competitive and super performances with equivalent and small-size models as highlighted in the tables below.

### Instruct models

Falcon3-1B-Instruct and Falcon3-3B-Instruct achieve robust performance across the evaluated benchmarks. Specifically, Falcon3-1B attains competitive results in IFEval (54.4), MUSR (40.7), and SciQ (86.8), while Falcon3-3B exhibits further gains—particularly in MMLU-PRO (29.7) and MATH (19.9)—demonstrating clear scaling effects. Although they do not surpass all competing models on every metric, Falcon models show strong performances in reasoning and common-sense understanding relative to both Qwen and Llama.

<br/><br/>
<!-- ![medium base models' performances](small-instruct-models.png) -->
<div style="text-align: left;" align="center">
    <img src="https://huggingface.co/datasets/tiiuae/documentation-images/resolve/main/general/small-instruct-models.png" alt="Training efficiency" width="800">
</div>
<br/><br/>
Furthermore, Falcon3-7B and Falcon3-10B show robust performance across the evaluated benchmarks. Falcon3-7B achieves competitive scores on reasoning (Arc Challenge: 65.9, MUSR: 46.4) and math (GSM8K: 79.1), while Falcon3-10B demonstrates further improvements, notably in GSM8K (83.1) and IFEval (78), indicating clear scaling benefits.
<br/><br/>
<!-- ![medium instruct models' performances](medium-instruct-models.png) -->
<div style="text-align: left;" align="center">
    <img src="https://huggingface.co/datasets/tiiuae/documentation-images/resolve/main/general/medium-instruct-models.png" alt="Training efficiency" width="800">
</div>
<br/><br/>


## Open Source Commitment
In line with our mission to foster AI accessibility and collaboration, all models in the Falcon3 family are released under the **Apache 2.0 license**. We hope the AI community finds these models valuable for research, application development, and further experimentation. Falcon3 is not a culmination but a continuation of our efforts to create more capable, efficient, specialized foundation models. In January 2025, we will further release other models of the Falcon3 family featuring enhanced multi-modal capabilities including image, video, and audio support, as well as a full technical report covering our methodologies. We welcome feedback and collaboration from the community as we continue to refine and advance these technologies.

## Useful links

- Access to our models (including GGUF and 1.58bit models) of this series through [the Falcon3 HuggingFace collection](https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026).
- Feel free to join [our discord server](https://discord.gg/fwXpMyGc) if you have any questions or to interact with our researchers and developers.
- Check out the [Falcon-LLM License link](https://falconllm.tii.ae/falcon-terms-and-conditions.html) for more details about the license.
- Refer to the official [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/) for HF evaluations of our models.

## Acknowledgments

We warmly thank the following people for their smooth support and integration within the ecosystem.

- [Alina Lozovskaya](https://huggingface.co/alozowski) and [Clementine Fourrier](https://huggingface.co/clefourrier) for helping us evaluate the model on the HF leaderboard.
- [Cyril Vallez](https://huggingface.co/cyrilvallez) and [Arthur Zucker](https://huggingface.co/ArthurZ) for the transformers documentation integration.
- [Vaibhav Srivastav](https://huggingface.co/reach-vb) for his help reviewing this blogpost.
- [Georgi Gerganov](https://github.com/ggerganov) for his help in integrating an important fix to make Falcon3 series models work in [llama.cpp](https://github.com/ggerganov/llama.cpp).
- [Awni Hannun](https://github.com/awni) for helping us review necessary changes in order to integrate Falcon3 series into MLX ecosystem.
- [BitNet.cpp team](https://github.com/microsoft/BitNet) for helping us integrating 1.58bit variants of Falcon3 models into BitNet.

## Citation
If the Falcon3 family of models were helpful to your work, feel free to give us a cite.

```
@misc{Falcon3,
    title = {The Falcon 3 Family of Open Models},
    url = {https://huggingface.co/blog/falcon3},
    author = {Falcon-LLM Team},
    month = {December},
    year = {2024}
}
```
