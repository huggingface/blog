---
title: "Unified multimodal large model evaluation, accelerating multimodal intelligence emergence"
thumbnail: https://github.com/lmms-lab/lmms-eval-blog/blob/master/assets/img/lmms-eval-header.png
authors:
- user: kcz358
  guest: true
---
# Unified multimodal large model evaluation, accelerating multimodal intelligence emergence

GitHub repo : https://github.com/EvolvingLMMs-Lab/lmms-eval

Official website : https://lmms-lab.github.io/

With the deepening development of artificial intelligence research, multimodal large models such as GPT-4V and LLaVA have become hot topics in both academia and industry. However, these advanced models require an effective evaluation framework to accurately measure their performance, which is not an easy task. On the one hand, the diverse prompts and post-processing methods adopted by different models may lead to significant differences in performance evaluation results, as illustrated by HuggingFace's mention of "1001 flavors of MMLU" in their blog post, indicating that different implementations of the same evaluation dataset may result in significant score differences, even changing the model's ranking on leaderboards.

Another challenge lies in data acquisition and processing during the evaluation process, especially when dealing with old datasets that are not widely available. Researchers often need to invest a considerable amount of time and effort in manual searching, downloading, and processing.

To address these issues, researchers from Nanyang Technological University, ByteDance, and other institutions have jointly open-sourced lmms-eval, which is an evaluation framework designed specifically for multimodal large models. Building upon lm-evaluation-harness, this framework has been improved and expanded to provide a unified interface for defining models, datasets, and evaluation metrics, offering a one-stop, efficient solution for evaluating multimodal models (LMMs). We hope that through this framework, we can collectively drive the iteration cycle of multimodal models and promote their broader application in academia and industry. We sincerely look forward to witnessing more breakthroughs and innovations in the field of multimodal AI, jointly advancing towards a more efficient and intelligent future development of artificial intelligence technology.

<image src="https://github.com/lmms-lab/lmms-eval-blog/blob/master/assets/img/teaser.png" alt="Pipeline"/>
