---
title: "Introducing the Open Leaderboard for Hebrew LLMs!"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_hebrew.png
authors:
- user: shaltielshmid
  guest: true
  org: dicta-il
- user: TalGeva
  guest: true
  org: HebArabNlpProject
- user: OmerKo
  guest: true
  org: Webiks
- user: clefourrier
---

# Introducing the Open Leaderboard for Hebrew LLMs!


This project addresses the critical need for advancement in Hebrew NLP.  As Hebrew is considered a low-resource language, existing LLM leaderboards often lack benchmarks that accurately reflect its unique characteristics. Today, we are excited to introduce a pioneering effort to change this narrative — our new open LLM leaderboard, specifically designed to evaluate and enhance language models in Hebrew.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.4.0/gradio.js"> </script>
<gradio-app theme_mode="light" space="hebrew-llm-leaderboard/leaderboard"></gradio-app>

Hebrew is a morphologically rich language with a complex system of roots and patterns. Words are built from roots with prefixes, suffixes, and infixes used to modify meaning, tense, or form plurals (among other functions). This complexity can lead to the existence of multiple valid word forms derived from a single root, making traditional tokenization strategies, designed for morphologically simpler languages, ineffective. As a result, existing language models may struggle to accurately process and understand the nuances of Hebrew, highlighting the need for benchmarks that cater to these unique linguistic properties.

LLM research in Hebrew therefore needs dedicated benchmarks that cater specifically to the nuances and linguistic properties of the language. Our leaderboard is set to fill this void by providing robust evaluation metrics on language-specific tasks, and promoting an open community-driven enhancement of generative language models in Hebrew. 
We believe this initiative will be a platform for researchers and developers to share, compare, and improve Hebrew LLMs.

## Leaderboard Metrics and Tasks

We have developed four key datasets, each designed to test language models on their understanding and generation of Hebrew, irrespective of their performance in other languages. These benchmarks use a few-shot prompt format to evaluate the models, ensuring that they can adapt and respond correctly even with limited context.

Below is a summary of each of the benchmarks included in the leaderboard. For a more comprehensive breakdown of each dataset, scoring system, prompt construction, please visit the `About` tab of our leaderboard. 

- **Hebrew Question Answering**: This task evaluates a model's ability to understand and process information presented in Hebrew, focusing on comprehension and the accurate retrieval of answers based on context. It checks the model's grasp of Hebrew syntax and semantics through direct question-and-answer formats. 
    - *Source*: [HeQ](https://aclanthology.org/2023.findings-emnlp.915/) dataset's test subset.

- **Sentiment Accuracy**: This benchmark tests the model's ability to detect and interpret sentiments in Hebrew text. It assesses the model's capability to classify statements accurately as positive, negative, or neutral based on linguistic cues. 
    - *Source*: [Hebrew Sentiment](https://huggingface.co/datasets/HebArabNlpProject/HebrewSentiment) - a Sentiment-Analysis Dataset in Hebrew.

- **Winograd Schema Challenge**: The task is designed to measure the model’s understanding of pronoun resolution and contextual ambiguity in Hebrew. It tests the model’s ability to use logical reasoning and general world knowledge to disambiguate pronouns correctly in complex sentences.
    - *Source*: [A Translation of the Winograd Schema Challenge to Hebrew](https://www.cs.ubc.ca/~vshwartz/resources/winograd_he.jsonl), by Dr. Vered Schwartz.

- **Translation**: This task assesses the model's proficiency in translating between English and Hebrew. It evaluates the linguistic accuracy, fluency, and the ability to preserve meaning across languages, highlighting the model’s capability in bilingual translation tasks.
    - *Source*: [NeuLabs-TedTalks](https://opus.nlpl.eu/NeuLab-TedTalks/en&he/v1/NeuLab-TedTalks) aligned translation corpus.

## Technical Setup

The leaderboard is inspired by the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), and uses the [Demo Leaderboard template](https://huggingface.co/demo-leaderboard-backend). Models that are submitted are deployed automatically using HuggingFace’s [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) and evaluated through API requests managed by the [lighteval](https://github.com/huggingface/lighteval) library.
The implementation was straightforward, with the main task being to set up the environment; the rest of the code ran smoothly.

## Engage with Us

We invite researchers, developers, and enthusiasts to participate in this initiative. Whether you're interested in submitting your model for evaluation or joining the discussion on improving Hebrew language technologies, your contribution is crucial. Visit the submission page on the leaderboard for guidelines on how to submit models for evaluation, or join the [discussion page](https://huggingface.co/spaces/hebrew-llm-leaderboard/leaderboard/discussions) on the leaderboard’s HF space.

This new leaderboard is not just a benchmarking tool; we hope it will encourage the Israeli tech community to recognize and address the gaps in language technology research for Hebrew. By providing detailed, specific evaluations, we aim to catalyze the development of models that are not only linguistically diverse but also culturally accurate, paving the way for innovations that honor the richness of the Hebrew language. 
Join us in this exciting journey to reshape the landscape of language modeling!

## Sponsorship

The leaderboard is proudly sponsored by [DDR&D IMOD / The Israeli National Program for NLP in Hebrew and Arabic](https://nnlp-il.mafat.ai/) in collaboration with [DICTA: The Israel Center for Text Analysis](https://dicta.org.il) and [Webiks](https://webiks.com), a testament to the commitment towards advancing language technologies in Hebrew.


