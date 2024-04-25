---
title: "Unified multimodal large model evaluation, accelerating multimodal intelligence emergence"
thumbnail: /blog/assets/lmms_eval/thumbnail.png
authors:
- user: luodian
  guest: true
  org: lmms-lab
- user: PY007
  guest: true
  org: lmms-lab
- user: kcz358
  guest: true
  org: lmms-lab
- user: pufanyi
  guest: true
  org: lmms-lab
- user: JvThunder
  guest: true
  org: lmms-lab
- user: dododododo
  guest: true
- user: THUdyh
  guest: true
  org: lmms-lab
- user: liuhaotian
  guest: true
  org: lmms-lab
- user: ZhangYuanhan
  guest: true
  org: lmms-lab
- user: zhangysk
  guest: true
- user: Chunyuan24
  guest: true
  org: lmms-lab
- user: liuziwei7
  guest: true
---
# Unified Multimodal Large Model Evaluation, Accelerating Multimodal Intelligence Emergence

GitHub repo : https://github.com/EvolvingLMMs-Lab/lmms-eval

Official website : https://lmms-lab.github.io/

With the deepening development of artificial intelligence research, multimodal large models such as GPT-4V and LLaVA have become hot topics in both academia and industry. However, these advanced models require an effective evaluation framework to accurately measure their performance, which is not an easy task. On the one hand, the diverse prompts and post-processing methods adopted by different models may lead to significant differences in performance evaluation results, as illustrated by HuggingFace's mention of "1001 flavors of MMLU" in their blog post, indicating that different implementations of the same evaluation dataset may result in significant score differences, even changing the model's ranking on leaderboards.

Another challenge lies in data acquisition and processing during the evaluation process, especially when dealing with old datasets that are not widely available. Researchers often need to invest a considerable amount of time and effort in manual searching, downloading, and processing.

To address these issues, researchers from Nanyang Technological University, ByteDance, and other institutions have jointly open-sourced [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval), which is an evaluation framework designed specifically for multimodal large models. Building upon EleutherAI's [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) and [🤗 Accelerate](https://github.com/huggingface/accelerate), this framework has been improved and expanded to provide a unified interface for defining models, datasets, and evaluation metrics, offering a one-stop, efficient solution for evaluating large multimodal models (LMMs). We hope that through this framework, we can collectively drive the iteration cycle of multimodal models and promote their broader application in academia and industry. We sincerely look forward to witnessing more breakthroughs and innovations in the field of multimodal AI, jointly advancing towards a more efficient and intelligent future development of artificial intelligence technology.

<image src="https://github.com/lmms-lab/lmms-eval-blog/blob/master/assets/img/teaser.png" alt="Pipeline"/>

## Overview of the Main Features

**One-click evaluation**: lmms-eval allows users to easily evaluate their model performance on multiple datasets with a single command, without the need for manual dataset preparation. With just one line of code, users can obtain comprehensive evaluation results within minutes, including detailed logs and sample analysis covering model parameters, inputs and outputs, correct answers, etc. This is suitable for scenarios where advanced models like GPT4 are needed for scoring.

Here's an example to evaluate a LLaVa model on the [MME](https://arxiv.org/abs/2306.13394) and [MMBench](https://arxiv.org/abs/2307.06281) benchmarks:

```
# Build from source
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git 

# Build from pypi
# pip install lmms-eval

# Build llava
# pip install git+https://github.com/haotian-liu/LLaVA.git

# Run your evaluation with accelerate with one line of code!
accelerate launch --multi_gpu --num_processes=8 -m lmms_eval \
    --model llava   \
    --model_args pretrained="liuhaotian/llava-v1.5-7b"   \
    --tasks mme,mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_mmbenchen \
    --output_path ./logs
```

**Parallel acceleration and task merging**: Utilizing Huggingface's accelerator, lmms-eval supports multi-GPU, model parallelism, and multi-batch processing, significantly enhancing evaluation efficiency. This feature is particularly advantageous when testing multiple datasets simultaneously, greatly reducing evaluation time.

Here is the total runtime on different datasets using 4 x A100 40G:


| Dataset (#num)          | LLaVA-v1.5-7b      | LLaVA-v1.5-13b     |
| :---------------------- | :----------------- | :----------------- |
| mme (2374)              | 2 mins 43 seconds  | 3 mins 27 seconds  |
| gqa (12578)             | 10 mins 43 seconds | 14 mins 23 seconds |
| scienceqa_img (2017)    | 1 mins 58 seconds  | 2 mins 52 seconds  |
| ai2d (3088)             | 3 mins 17 seconds  | 4 mins 12 seconds  |
| coco2017_cap_val (5000) | 14 mins 13 seconds | 19 mins 58 seconds |

Additionally, in the 0.1.1.dev update, the team has added support for tensor parallelism, enabling the running of larger models like LLaVA-v1.6-34B on 4 x 3090 GPUs, supporting efficient inference.

**Comprehensive dataset support:** The `lmms-eval` team has hosted over 40 diverse datasets (with the number continually increasing) on Huggingface's lmms-lab, covering a range of tasks from COCO Captions to MMMU and others. All datasets have been transformed into a unified format for archiving, available for direct access on the team's lmms-lab official Huggingface Hub. Users can view specific details of evaluation data and easily download and use them with just one click. You can find all the datasets we support in the framework under [this collection](https://huggingface.co/collections/lmms-lab/lmms-eval-661d51f70a9d678b6f43f272).

<image src="https://github.com/lmms-lab/lmms-eval-blog/blob/master/assets/img/org_dataset.png" alt="dataset on organization"/>

<image src="https://github.com/lmms-lab/lmms-eval-blog/blob/master/assets/img/viewer.png"  alt="viewer" />

**Easy to Extend**: Through a unified interface definition, `lmms-eval` not only simplifies the integration process of different models and datasets but also provides convenience for introducing new datasets and models. Additionally, it supports simple customization settings, allowing users to easily add new datasets through simple YAML file configuration and customize evaluation settings as needed by modifying the configuration file.

**Comparability**: We provide an environment for authors to reproduce the scores reported in the paper for the original LLaVA 1.5 model. Furthermore, we offer complete experimental results of the LLaVA series models on all evaluation datasets, along with environmental parameters for reference (see the Readme section on GitHub).

**Synchronized Online Logging**: We provide detailed logging tools to help you understand the evaluation process and results. Logs include model parameters, generation parameters, input questions, model responses, and ground truth answers. You can record every detail and visualize it in Weights & Biases runs. Users can access results in real-time from anywhere, making it convenient and efficient.

<image src="https://github.com/lmms-lab/lmms-eval-blog/blob/master/assets/img/wandb_table.jpg" alt="wandb_table" />

## Conclusion

In summary, the implementation of this framework not only provides new tools for multimodal model evaluation but also paves the way for future research and development, including video multimodal evaluation, few-shot evaluation modes, and batch inference acceleration, showcasing its powerful potential and foresight. The launch of `lmms-eval` marks the arrival of a new era in evaluation, opening up new paths for AI research and applications. We hope the community finds it useful for benchmarking their own models in this fast-moving field!
