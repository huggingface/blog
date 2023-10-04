---
title: "Accelerating over 130,000 Hugging Face models with ONNX Runtime"
thumbnail: /blog/assets/ort_accelerating_hf_models/thumbnail.png
authors:
- user: sschoenmeyer
  guest: true
- user: mfuntowicz
---

# Accelerating over 130,000 Hugging Face models with ONNX Runtime

## What is ONNX Runtime?
ONNX Runtime is a cross-platform machine learning tool that can be used to accelerate a wide variety of models, particularly those with ONNX support.

## Hugging Face ONNX Runtime Support 

There are over 130,000 ONNX-supported models on Hugging Face, an open source community that allows users to build, train, and deploy hundreds of thousands of publicly available machine learning models.
These ONNX-supported models, which include many increasingly popular large language models (LLMs) and cloud models, can leverage ONNX Runtime to improve performance, along with other benefits.
For example, using ONNX Runtime to accelerate the whisper-tiny model can improve average latency per inference, with an up to 74.30% gain over PyTorch.
ONNX Runtime works closely with Hugging Face to ensure that the most popular models on the site are supported.
In total, over 90 Hugging Face model architectures are supported by ONNX Runtime, including the 11 most popular architectures (where popularity is determined by the corresponding number of models uploaded to the Hugging Face Hub):

| Model Architecture | Approximate No. of Models |
|:------------------:|:-------------------------:|
|        BERT        |           28180           |
|        GPT2        |           14060           |
|     DistilBERT     |           11540           |
|      RoBERTa       |           10800           |
|         T5         |           10450           |
|      Wav2Vec2      |           6560            |
|  Stable-Diffusion  |           5880            |
|    XLM-RoBERTa     |           5100            |
|      Whisper       |           4400            |
|        BART        |           3590            |
|       Marian       |           2840            |

## Learn More
To learn more about accelerating Hugging Face models with ONNX Runtime, check out our recent post on the [Microsoft Open Source Blog](https://cloudblogs.microsoft.com/opensource/2023/10/04/accelerating-over-130000-hugging-face-models-with-onnx-runtime/).