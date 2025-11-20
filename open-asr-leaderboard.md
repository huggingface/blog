---
title: "Open ASR Leaderboard: Trends and Insights with New Multilingual & Long-Form Tracks" 
thumbnail: /blog/assets/open-asr-leaderboard/thumbnail.png
authors:
- user: bezzam
- user: Steveeeeeeen
- user: eustlb
- user: reach-vb
---


# Open ASR Leaderboard: Trends and Insights with New Multilingual & Long-Form Tracks

<!-- <div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/thumbnail.png" width="1024px" alt="thumbnail" />
</div> -->


While everyone (and their grandma 👵) is spinning up new ASR models, picking the right one for your use case can feel more overwhelming than choosing your next Netflix show. As of 20 Nov 2025, there are **150 [Audio-Text-to-Text](https://huggingface.co/models?pipeline_tag=audio-text-to-text&sort=trending)** and **27K [ASR models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)** on the Hub 🤯

Most benchmarks focus on **short-form English transcription (<30s),** and overlook other important tasks, such as (1) multilingual performance and (2) model throughput, which can a be deciding factor for long-form audio like meetings and podcasts.

Over the past two years, the [**Open ASR Leaderboard**](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) has become a standard for comparing open and closed-source models on both **accuracy** and **efficiency**. Recently, **multilingual** and **long-form transcription** tracks have been added to the leaderboard 🎉

### TL;DR - [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

- 📝 **New preprint** on ASR trends from the leaderboard: https://hf.co/papers/2510.06961
- 🧠 **Best accuracy:** Conformer encoder + LLM decoders (open-source ftw 🥳)
- ⚡ **Fastest:** CTC / TDT decoders
- 🌍 **Multilingual:** Comes at the cost of single-language performance
- ⌛ **Long-form:** Closed-source systems still lead (for now 😉)
- 🧑‍💻 **Fine-tuning guides** ([Parakeet](https://github.com/Deep-unlearning/Finetune-Parakeet), [Voxtral](https://github.com/Deep-unlearning/Finetune-Voxtral-ASR), [Whisper](https://huggingface.co/learn/audio-course/chapter5/fine-tuning)): to continue pushing performance 


# Takeaways from 60+ models

As of 20 Nov 2025, the *Open ASR Leaderboard* compares **60+ open and closed-source models** from **18 organizations**, across **11 datasets**.

In a recent [preprint](https://arxiv.org/abs/2510.06961), we dive into the technical setup and highlight some key trends in modern ASR. Here are the big takeaways 👇

## 1. Conformer encoder 🤝 LLM decoder tops the charts 📈

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/leaderboard_WER.png" width="1024px" alt="thumbnail" />
</div>

Models combining [**Conformer encoders**](https://arxiv.org/abs/2005.08100) with **large language model (LLM) decoders** currently lead in English transcription accuracy. For example, **NVIDIA’s [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b)**, **IBM’s [Granite-Speech-3.3-8B](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)**, and **Microsoft’s [Phi-4-Multimodal-Instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)** achieve the lowest word error rates ([WER](https://huggingface.co/learn/audio-course/en/chapter5/evaluation#word-error-rate)), showing that integrating LLM reasoning can significantly boost ASR accuracy.

💡 *Pro-tip: NVIDIA introduced [Fast Conformer](https://arxiv.org/abs/2305.05084), a 2x faster variant of the Conformer, that is used in their Canary and Parakeet suite of models.*

## 2. Speed–accuracy tradeoffs ⚖️

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/leaderboard_RTX.png" width="1024px" alt="thumbnail" />
</div>

While highly accurate, these LLM decoders tend to be **slower** than simpler approaches. On the *Open ASR Leaderboard*, efficiency is measured using *inverse real-time factor* (RTFx), where higher is better.

For even faster inference, [**CTC**](https://huggingface.co/learn/audio-course/en/chapter3/ctc#ctc-architectures) and [**TDT**](https://arxiv.org/abs/2304.06795) decoders deliver **10–100× faster throughput**, albeit with slightly higher error rates. This makes them ideal for **real-time**, **offline**, or **batch transcription** tasks (such as meetings, lectures, or podcasts).

## 3. Multilingual 🌍

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/multilingual.png" width="1024px" alt="thumbnail" />
</div>

OpenAI’s [**Whisper Large v3**](https://huggingface.co/openai/whisper-large-v3) remains a strong multilingual baseline, supporting **99 languages**. However, **fine-tuned or distilled variants** like [**Distil-Whisper**](https://huggingface.co/distil-whisper/distil-large-v3.5) and [**CrisperWhisper**](https://huggingface.co/nyrahealth/CrisperWhisper) often outperform the original on **English-only** tasks, showing how targeted fine-tuning can improve specialization (*how to fine-tune? Check out guides for [Whisper](https://huggingface.co/learn/audio-course/chapter5/fine-tuning), [Parakeet](https://github.com/Deep-unlearning/Finetune-Parakeet), and [Voxtral](https://github.com/Deep-unlearning/Finetune-Voxtral-ASR)*).

That said, focusing on English tends to **reduce multilingual coverage** 👉 a classic case of the tradeoff between specialization and generalization. Similarly, while **self-supervised** systems like Meta’s [**Massively Multilingual Speech (MMS)**](https://huggingface.co/facebook/mms-1b-all) can support 1K+ languages, they trail behind language-specific encoders in accuracy.

⭐ *While just five languages are currently benchmarked, we’re planning to expand to more languages and are excited for new dataset and models contributions to multilingual ASR through GitHub [pull requests](https://github.com/huggingface/open_asr_leaderboard).*

🎯 Alongside multilingual benchmarks, several **community-driven leaderboards** focus on individual languages. For example, the [**Open Universal Arabic ASR Leaderboard**](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard) compares models across **Modern Standard Arabic and regional dialects**, highlighting how speech variation and diglossia challenge current systems. Similarly. the [**Russian ASR Leaderboard**](https://huggingface.co/spaces/Vikhrmodels/Russian_ASR_Leaderboard) provides a growing hub for evaluating encoder-decoder and CTC models on **Russian-specific phonology and morphology**. These localized efforts mirror the broader multilingual leaderboard’s mission to encourage **dataset sharing, fine-tuned checkpoints, and transparent model comparisons**, especially in languages with fewer established ASR resources.

## 4. Long-form transcription is a different game ⏳

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/long_form.png" width="1024px" alt="thumbnail" />
</div>

For **long-form audio** (e.g., podcasts, lectures, meetings), **closed-source systems** still edge out open ones. It could be due to domain tuning, custom chunking, or production-grade optimization.

Among open models, **OpenAI’s Whisper Large v3** performs the best. But for throughput, **CTC-based Conformers** shine 👉 for example, **NVIDIA’s [Parakeet CTC 1.1B](https://huggingface.co/nvidia/parakeet-ctc-1.1b)** achieves an **RTFx of 2793.75**, compared to **68.56** for Whisper Large v3, with only a moderate WER degradation (**6.68** and **6.43** respectively).

The tradeoff? Parakeet is **English-only,** again reminding us of that multilingual and specialization tradeoff 🫠.

⭐ *While closed systems still lead, there’s huge potential for open-source innovation here. Long-form ASR remains one of the most exciting frontiers for the community to tackle next!*

# 🎤 The Show Must Go On

Given how fast ASR is evolving, we’re excited to see what new architectures push performance and efficiency, and how the *Open ASR Leaderboard* continues to serve as a **transparent, community-driven benchmark** for the field, and as a reference for other leaderboards ([Russian](https://huggingface.co/spaces/Vikhrmodels/Russian_ASR_Leaderboard), [Arabic](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard), and [Speech DeepFake Detection](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena)).

We’ll keep expanding the *Open ASR LeaderBoard* with **more models, more languages, and more datasets** so stay tuned 👀

👉 **Want to contribute?** Head on over to the [GitHub repo](https://github.com/huggingface/open_asr_leaderboard) to open a *pull request* 🚀