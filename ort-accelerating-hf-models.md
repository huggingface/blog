---
title: "Accelerating over 110,000 Hugging Face models with ONNX Runtime"
thumbnail: /blog/assets/163_getting_most_out_of_llms/optimize_llm.png
authors:
- user: Sophie Schoenmeyer
  guest: true
- user: Morgan Funtowicz
---

## What is ONNX Runtime?
ONNX Runtime is a cross-platform machine learning tool that can be used to accelerate a wide variety of models, particularly those with ONNX support.

## Hugging Face ONNX Runtime Support 

There are over 130,000 ONNX-supported models on Hugging Face, an open source community that allows users to build, train, and deploy hundreds of thousands of publicly available machine learning models.
These ONNX-supported models, which include many increasingly popular large language models (LLMs) and cloud models, can leverage ONNX Runtime to improve performance, along with other benefits.
For example, using ONNX Runtime to accelerate the whisper-tiny model can improve average latency per inference, with an up to 74.30% gain over PyTorch.
ONNX Runtime works closely with Hugging Face to ensure that the most popular models on the site are supported.
In total, over 90 Hugging Face model architectures are supported by ONNX Runtime, including the 11 most popular architectures (where popularity is determined by the corresponding number of models uploaded to the Hugging Face Hub):

| Mode Architecture | Approximated No. of Models |
|:-----------------:|:--------------------------:|
|       BERT        |           28100            |
|       GPT2        |           13900            |
|    DistilBERT     |           11400            |
|      RoBERTa      |           10600            |
|        T5         |           10300            |
|     Wav2Vec2      |            6500            |
| Stable-Diffusion  |            5700            |
|    XLM-RoBERTa    |            5000            |
|      Whisper      |            4300            |
|       BART        |            3500            |
|      Marian       |            2800            |

## Learn More
To learn more about accelerating Hugging Face models with ONNX Runtime, check out our recent post on the [Microsoft Open Source Blog](insert link here).