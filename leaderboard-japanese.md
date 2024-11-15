---
title: "Introducing the Open Leaderboard for Japanese LLMs!"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_japanese.png
authors:
- user: akimfromparis
  guest: true
  org: llm-jp
- user: miyao-yusuke
  guest: true
  org: llm-jp
- user: namgiH
  guest: true
  org: llm-jp
- user: t0-0
  guest: true
  org: llm-jp
- user: sh1gechan
  guest: true
  org: llm-jp
- user: hysts
- user: clefourrier
---

# Introduction to the Open Leaderboard for Japanese LLMs

LLMs are now increasingly capable in English, but it's quite hard to know how well they perform in other national languages, widely spoken but which present their own set of linguistic challenges. Today, we are excited to fill this gap for Japanese! 

We'd like to announce the **Open Japanese LLM Leaderboard**, composed of more than 20 datasets from classical to modern NLP tasks to understand underlying mechanisms of Japanese LLMs. The Open Japanese LLM Leaderboard was built by the **[LLM-jp](https://llm-jp.nii.ac.jp/en/)**, a cross-organizational project for the research and development of Japanese large language models (LLMs) in partnership with **Hugging Face**. 

The Japanese language presents its own specific challenges. Morphologically rich and in constant evolution due to historical and cultural interactions with the rest of the world, its writing system is based on a mixture of three separate sets of characters: simplified Chinese ideographic symbols kanjis (漢字), a phonetic lettering system, Hiraganas (平仮名 / ひらがな), and Katakanas (片仮名 / カタカナ) often used for foreigners words. Modern Japanese is arguably one of the hardest language to process, as it mixes up a blend of Sino-Japanese, native Japanese, Latin script (romaji /ローマ字), loanwords from the Dutch, Portuguese, French, English, German, plus Arabic and traditional Chinese numerals. In addition, the Japanese digital world brought us an evolution of emoticons written in Unicode : ), Kaomoji using Cyrillic alphabet. ∵･(ﾟДﾟ) and Greek alphabets ＿φ(°-°=). Without forgetting, of course, the classic emojis that originated from Japan with the rise in popularity of mobile phones in the 1990s. 

![Japanese writing system](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/fxTPcxQqAo49s_jE_5wCw.png)

The intricate writing system of Japanese hides an extra layer of complexity, the lack of space between words. Similar to the Chinese or Thai languages, Japanese language doesn’t have white space between linguistic units, making the detection of word boundaries extremely difficult during tokenization. Over the years, the vibrant Japanese ecosystem (from prestigious university laboratories and AI startups to the R&D centers of industry giants) has incorporated the specificities of Japanese NLP to develop modern robust Japanese LLMs, but the field has been lacking a centralized and open system to compare these models. 

We therefore introduce the Open Japanese LLM Leaderboard, a collaboration between Hugging Face and LLM-jp, to foster transparency in research, and encourage an open-source model development philosophy. We strongly believe this initiative will serve as a platform for Japanese and international researchers to collaborate, evaluate, and enhance Japanese LLMs.

## Introduction to the Leaderboard Tasks

The Open Japanese LLM Leaderboard evaluates Japanese LLMs using a specialized evaluation suite, **[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)**, covering a range of 16 tasks from classical ones (such as *Natural Language Inference, Machine Translation, Summarization, Question Answering*) to more modern ones (such as *Code Generation*, *Mathematical Reasoning* or *Human Examination*). Tasks are launched in 4-shot. 

Datasets have been compiled by the evaluation team of LLM-jp, either built from scratch with linguists, experts, and human annotators, or translated automatically to Japanese and adjusted to Japanese specificities, and for some requiring long context reasoning. For a better understanding of the leaderboard, we will detail samples from 8 datasets (in Japanese followed by the English translation in light gray). For more details about all the available tasks, please see to the “About” tab of the leaderboard, and official links on each datasets.

### Jamp

**Jamp** (*Controlled Japanese Temporal Inference Dataset for Evaluating Generalization Capacity of Language Models*) is the Japanese temporal inference benchmark for NLI. The dataset explore English and Japanese sentence pairs of various temporal inference patterns annotated with the golden labels such as entailment, neutral, or contradiction.

![Jamp](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/EF2BuJC_oWvw2Jc5kvGCo.png)

### JEMHopQA

**JEMHopQA** (*Japanese Explainable Multi-hop Question Answering*) is a Japanese multi-hop QA dataset that can evaluate internal reasoning. It is a task that takes a question as input and generates an answer and derivations. 

![JEMHopQA](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/ZicrCMz4LtXDxSxeBBTl-.png)

### jcommonsenseqa

**jcommonsenseqa** is a Japanese version of CommonsenseQA, which is a multiple-choice question answering dataset. The purpose of this dataset is to evaluate commonsense reasoning ability.

![jcommonsensqa](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/s21OdhQIRRW7dqTF9mYoq.png)

### chABSA

**chABSA** was developed as an *Aspect-Based Sentiment Analysis* dataset. ChABSA is based on financial reports of Japanese listed-companies in the 2016 fiscal year, annotated on the pair of entity, the attribute, and the sentiment. More specifically, 230 out of 2,260 companies listed in Japan (roughly 10% of all company)  were annotated according to the taxonomy of the Japanese financial regulator, *Financial Service Agency (FSA)*.

![chABSA](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/O2kTDa1w0YAJOW1quXuDQ.png)

### mbpp-ja

The **mbpp-ja** dataset is a programming dataset: it is a Japanese version of *Mostly Basic Python Problems dataset* (MBPP) translated from English into Japanese by **[LLM-jp](https://llm-jp.nii.ac.jp/en/)** by leveraging the translation tool **[DeepL](https://www.deepl.com)**.

![mbpp-ja](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/g21y5x0BuCWlX6foubsv5.png)

### mawps

Based on the dataset `MAWPS` *(A Math Word Problem Repository)*, the Japanese **mawps** dataset is a mathematical evaluation dataset. This version evaluates the abilities of solving novel tasks by reasoning step-by-step, procedure otherwise known as Chain-of-Thought (CoT) reasoning, and was adjusted to converting names of people, units, and places to fit the Japanese context. The level of mathematical reasoning is rather simple: addition, subtraction, multistep arithmetic, and single or pairs of equations.

![mawps](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/1FXowoymJJ72r6I2Q9si_.png)

### JMMLU

**JMMLU** is a knowledge dataset using four-choice question answers. It consists in Japanese-translated questions from a portion of MMLU dataset that evaluates knowledge on high-school level tests. Based on 57 subjects such as astronomy, chemistry, sociology, international law, etc., questions and answers were translated in Japanese, while being adjusted to unique Japanese cultural context like Japanese civics, Japanese geography, and Japanese idioms. 

![JMMLU](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/gVojua_19QLpFJqGSA8xz.png)

### XL-Sum

**XL-Sum** is a summarisation dataset based on the research titled *“XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages”* that leverages the Japanese translation of articles from BBC News. The dataset is separated in three parts; the title, the text (the full-length article), and the summary. Topics include global issues, politics, technology, sports, and culture.

![XL-Sum](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/dlMq7ii_VfVzYHLDQx7Y_.png)

## Technical Setup

The leaderboard is inspired by the **[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)**. Models that are submitted are deployed automatically using HuggingFace’s **[Inference endpoints](https://huggingface.co/docs/inference-endpoints/index)**, evaluated through the **[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)** library on the version 1.14.1, with memory-efficient inference and serving engine, **[vLLM](https://github.com/vllm-project/vllm)** on the verison v0.6.3, and computed in the backend by the premium computer platform for research in Japan, **[mdx](https://mdx.jp/)**. 

## Observations

According to the Japanese LLMs guide **[Awesome Japanese LLM](https://llm-jp.github.io/awesome-japanese-llm/)** (available in Japanese, English, and French), Meta's `LLama` open-source architecture seems to be the favourite of many Japanese AI labs. However, other architectures have also been successfully leveraged by the Japanese open-source community, such as `Mistral` from French Mistral, and `Qwen` by Chinese Alibaba. These are the architectures which led to the best scores on the Japanese LLM Leaderboard.

On general language processing tasks, we observe that Japanese LLMs based on open-source architectures are closing the gap with closed source LLMs, such as the Japanese LLM `llm-jp-3-13b-instruct`, developed by LLM-jp and funded by university grants, reaching a performance similar to closed source models. Domain specific datasets, such as `chABSA` (finance), `Wikipedia Annotated Corpus` (linguistic annotations), code generation (`mbpp-ja`) and summarization (`XL-Sum`) remain a challenge for most LLMs. Interestingly, models originating from Japanese-based companies or labs have better scores on the specific `JCommonsenseMorality` dataset. It evaluates model ability to make choices according to Japanese values when against ethical dilemmas

## Future directions

The Open Japanese LLM Leaderboard will follow the development of the evaluation tool **[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)** to reflect the constant evolution of Japanese LLMs. The following are just examples of future directions in llm-jp-eval that we would like to support, feel free to contact us to give a hand or suggest directions!

- **New datasets: More Japanese evaluations**
The evaluation team of llm-jp-eval is working on this section, adding at the moment **[JHumanEval](https://huggingface.co/datasets/kogi-jwu/jhumaneval)** (*Japanese version of HumanEval*) and **[MMLU](https://github.com/hendrycks/test)** (*Measuring Massive Multitask Language Understanding*).

- **New evaluation system: Chain-of-Thought evaluation**
We'd like to compare the performance of LLMs between when employing Chain-of-Thought prompts against basic prompts to have a finer understanding of model behaviors.

- **New metric support: Out-of-Choice rate**
For some evaluation tasks that already have a clear list of labels used in the specific task, such as Natural Language Inference, we'd like to add a complementary metric, testing how often the model predicts out-of-choice tokens. As the choices are provided in the prompt, this will allow us to evaluate how well each LLM is able to follow specific instructions.

## Acknowledgements

Built by the research consortium **LLM-jp**, the Open Japanese LLM Leaderboard is proudly sponsored by the **[National Institute of Informatics](https://www.nii.ac.jp/en/)** in Tokyo, Japan in collaboration with the  high-performance computing platform, **[mdx](https://mdx.jp/)** program.

We would like to extend our gratitude to **Prof. Yusuke Miyao** and **Namgi Han** from the *University of Tokyo* for their scientific consultation and guidance, as well as **Clémentine Fourrier** and **Toshihiro Hayashi** of Hugging Face that has assisted with the integration and customization of their new evaluation framework and leaderboard template. 
