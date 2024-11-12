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

Due to the rapid evolution and the remarkable capabilities of LLMs, it became necessary to create a critical tool to evaluate Japanese LLMs. Today, we are excited to announce the **Open Japanese LLM Leaderboard**, composed of more than 20 datasets from classical to modern NLP tasks to understand underlying mechanisms of Japanese LLMs. The Open Japanese LLM Leaderboard was built by the **[LLM-jp](https://llm-jp.nii.ac.jp/en/)**, a cross-organizational project for the research and development of Japanese large language models (LLMs) in partnership with **Hugging Face**. 

The Japanese language presents its own specific challenges. Morphologically rich and in constant evolution due to historical and cultural interactions with the rest of the world, its writing system is based on a mixture of three separate sets of characters: simplified Chinese ideographic symbols kanjis (漢字), a phonetic lettering system, Hiraganas (平仮名 / ひらがな), and Katakanas (片仮名 / カタカナ) often used for foreigners words. Modern Japanese is arguably one of the hardest language to process, as it mixes up a blend of Sino-Japanese, native Japanese, Latin script (romaji /ローマ字), loanwords from the Dutch, Portuguese, French, English, German, plus Arabic and traditional Chinese numerals. In addition, the Japanese digital world brought us an evolution of emoticons written in Unicode : ), Kaomoji using Cyrillic alphabet. ∵･(ﾟДﾟ) and Greek alphabets ＿φ(°-°=). Without forgetting, the classic emojis that originated from Japan with the rise in popularity of mobile phones in the 1990s. 

![Japanese writing system](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/fxTPcxQqAo49s_jE_5wCw.png)

The intricate writing system of Japanese hides an extra layer of complexity, the lack of space between words. Similar to Chinese or Thai language, Japanese language doesn’t have white space between linguistic units, making the detection of word boundaries extremely difficult during tokenization. Over the years, the vibrant Japanese ecosystem (from prestigious university laboratories and AI startups to the R&D centers of industry giants) have incorporated the specificities of Japanese NLP to develop modern robust Japanese LLMs, but the field has been lacking a centralized and open system to compare these models, while addressing the complexity of the Japanese language. 

We therefore introduce the Open Japanese LLM Leaderboard, a collaboration between Hugging Face and LLM-jp, to foster transparency in research, and encourage an open-source model development philosophy. We strongly believe this initiative will serve as a platform for Japanese and international researchers to collaborate, evaluate, and enhance Japanese LLMs.

## Leaderboard Metrics and Tasks

The Open Japanese LLM Leaderboard evaluates Japanese LLMs using a specialized evaluation suite,  **[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)**, using a range of tasks from classical ones (such as *Natural Language Inference, Machine Translation, Summarization, Question Answering*) to more modern ones (such as *Code Generation, Mathematical Reasoning* or *Human Examination*).

For the Open Japanese LLM Leaderboard, the evaluation team of LLM-jp has developed one of the most complete benchmark for Japanese LLMs by leveraging more than 16 tasks. Those tasks assess the Japanese LLMs using a 4-shot prompt format by default, testing their ability to adjust and respond accurately, even when provided with minimal context.

For a better understanding of the leaderboard, we will explain the specificities of 8 datasets together in Japanese followed by the English translation in light gray. For more details, please see to the “About” tab of the leaderboard, and official links on each datasets.

### Jamp

**Jamp** (*Controlled Japanese Temporal Inference Dataset for Evaluating Generalization Capacity of Language Models*) is the Japanese temporal inference benchmark for NLI. The dataset explore English and Japanese sentence pairs of various temporal inference patterns annotated with the golden labels such as entailment, neutral, or contradiction.

![Jamp](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/EF2BuJC_oWvw2Jc5kvGCo.png)

### JEMHopQA

**JEMHopQA** (*Japanese Explainable Multi-hop Question Answering*) is a Japanese multi-hop QA dataset that can evaluate internal reasoning. It is a task that takes a question as input and generates an answer and derivations. 

![JEMHopQA](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/ZicrCMz4LtXDxSxeBBTl-.png)

### jcommonsenseqa

**jcommonsenseqa** is a Japanese version of CommonsenseQA, which is a multiple-choice question answering dataset. The purpose of this dataset is to evaluate the commonsense reasoning ability.

![jcommonsensqa](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/s21OdhQIRRW7dqTF9mYoq.png)

### chABSA

**chABSA** was developed as an *Aspect-Based Sentiment Analysis* dataset. ChABSA is based on financial reports of Japanese listed-companies in the 2016 fiscal year; annotated on the pair of entity, the attribute, and the sentiment. More specifically, 230 out of 2,260 companies listed in Japan (roughly 10% of all company)  were annotated according to the taxonomy of the Japanese financial regulator, *Financial Service Agency (FSA)*.

![chABSA](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/O2kTDa1w0YAJOW1quXuDQ.png)

### mbpp-ja

The **mbpp-ja** dataset is a Japanese version of *Mostly Basic Python Problems dataset* (MBPP) translated from English into Japanese by **[LLM-jp](https://llm-jp.nii.ac.jp/en/)** by leveraging the translation tool, **[DeepL](https://www.deepl.com)**.

![mbpp-ja](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/g21y5x0BuCWlX6foubsv5.png)

### mawps

Based on the dataset `MAWPS` *(A Math Word Problem Repository)*, the Japanese **mawps** version evaluates the abilities of solving novel tasks by reasoning step-by-step, procedure otherwise known as Chain-of-Thought (CoT) reasoning. The Japanese version was adjusted to converting names of people, units, and places to fit the Japanese context. The level of mathematical reasoning is rather simple such as addition, subtraction, multistep arithmetic, and single or two equations.

![mawps](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/1FXowoymJJ72r6I2Q9si_.png)

### JMMLU

**JMMLU** is a four-choice question set consisting of Japanese-translated questions from a portion of MMLU dataset that evaluates knowledge on high-school level tests. Based on 57 subjects such as astronomy, chemistry, sociology, international law, etc., questions and answers were translated in Japanese, while being adjusted to unique Japanese cultural context like Japanese civics, Japanese geography, and Japanese idioms. 

![JMMLU](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/gVojua_19QLpFJqGSA8xz.png)

### XL-Sum

**XL-Sum** is based on the research titled *“XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages”* that leverages the Japanese translation of articles from BBC News. The dataset is separated in three parts; the title, the text (the full-length article), and the summary. Topics include global issues, politics, technology, sports, and culture.

![XL-Sum](https://cdn-uploads.huggingface.co/production/uploads/63171caf1cc81c5e95ed7b92/dlMq7ii_VfVzYHLDQx7Y_.png)

## Technical Setup

The leaderboard is inspired by the **[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)**. Models that are submitted are deployed automatically using HuggingFace’s **[Inference endpoints](https://huggingface.co/docs/inference-endpoints/index)**, evaluated through the **[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)** library on the version 1.14.1, with memory-efficient inference and serving engine, **[vLLM](https://github.com/vllm-project/vllm)** on the verison v0.6.3, and computed in the backend by the premium computer platform for research in Japan,  **[mdx](https://mdx.jp/)**. 

## Observations and surprising results

#Waiting for new evaluation of Japanese LLMs and correction of Namgi Han... 

According to the Japanese LLMs guide **[Awesome Japanese LLM](https://llm-jp.github.io/awesome-japanese-llm/)** (available in Japanese, English, and French), the open-source architecture of Llama from *Meta Inc.* seems to be the favourite of many Japanese AI labs. However, other powerful open-source models built by *Mistral* in France and *Alibaba* in China have also been successfully leveraged by the Japanese open-source community. 

During the evaluation of Japanese LLMs, we can witness good metrics on multiple recent architectures such as Llama 3.1, Mistral 7B, or Qwen 2.5 on the leaderboard. For the offline inference and evaluation, llm-jp-eval incorporates the inference engine vLLM, and the TextGenerationPipeline of HuggingFace Transformers. At the same time, the implementation for offline inference with DeepSpeed-MII, TensorRT-LLM, and llama-cpp-python are still experimental.

Under the hood of the evaluation tool, the dataset pre-processing tool called `Jaster` is handling more and more lengthy datasets. We can observe datasets built from scratch with linguists, experts, and human annotators, but also datasets translated automatically to Japanese or adjusted to Japanese specificities. By using 0 shot and 4 shot-method, we are able to observe the improvement of metrics on the leaderboard. 
On general language processing, we witnessed that Japanese LLMs based on open-source LLMs are closing the gap with closed source LLMs. As an example, the Japanese LLM "llm-jp-3-1.13B-instruct" developed by LLM-jp, funded by university grants, can reach almost similar performance than international giants. 

The "chABSA" task remains a challenge for many LLMs due to the domain-specific of the Japanese financial market. Japanese LLMs struggle with the task "Wikipedia Annotated Corpus" based on a corpus from Japanese Wikipedia consist of linguistic annotations for morphology, named entites, dependencies, predicate-argument structures, and coreferences. Many Japanese LLMs achieved great metrics compared to foreigner-built LLMs on the task "JCommonsenseMorality" that evaluate Japanese commonsense morality understanding due to the strong Japanese values incorporated in ethical dilemmas. For the recently added evaluation tasks, such as code generation task with "mbpp-ja" and summarization task with "XL-Sum", we can see that LLMs have a great margin of improvement. 


## Future directions

The Open Japanese LLM Leaderboard will follow the development of the evaluation tool **[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)** to reflect the constant evolution of Japanese LLMs. The following are just examples of future directions in llm-jp-eval that we would like to support.

- **More Japanese evaluation datasets**
The evaluation tool, llm-jp-eval is already working for this part, for example **[JHumanEval](https://huggingface.co/datasets/kogi-jwu/jhumaneval)** (*Japanese version of HumanEval*) and the open source dataset **[MMLU](https://github.com/hendrycks/test)** (*Measuring Massive Multitask Language Understanding*).

- **Chain-of-Thought evaluation**
We can compare the performance of LLMs between when employing Chain-of-Thought prompts against basic prompts to have a better understanding of the model behaviors

- **Out-of-Choice rates as the metric**
For some evaluation tasks that already have a clear list of labels used in the specific task, such as Natural Language Inference. And that information is included in our basic prompt, we can evaluate how well each LLM follows a given instruction to calculate the OoC rate.

## Acknowledgements

Built by the research consortium **LLM-jp**, the Open Japanese LLM Leaderboard is proudly sponsored by the **[National Institute of Informatics](https://www.nii.ac.jp/en/)** in Tokyo, Japan in collaboration with the  high-performance computing platform, **[mdx](https://mdx.jp/)** program.

We would like to extend our gratitude to **Prof. Yusuke Miyao** and **Namgi Han** from the *University of Tokyo* for their scientific consultation and guidance, as well as, **Clémentine Fourrier** and **Toshihiro Hayashi** of Hugging Face that has assisted with the integration and customization of their new evaluation framework and leaderboard template. 