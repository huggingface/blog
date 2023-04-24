---
title: "Finetune Whisper with LoRA & BNB powered by 🤗 PEFT" 
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
authors:
- user: reach-vb
---

# Fine tune Whisper with LoRA powered by PEFT

<!-- {blog_metadata} -->
<!-- {authors} -->

<a target="_blank" href="https://colab.research.google.com/github/Vaibhavs10/notebooks/blob/main/Whisper_w_PEFT.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

A one size fits all walkthrough, to fine-tune Whisper (large) **5x faster** on a consumer GPU with **less than 8GB GPU VRAM**, all with comparable performance to full-finetuning. ⚡️

## Table of Contents

1. [Why Parameter Efficient Fine Tuning?](#introduction)
2. [Fine-tuning Whisper in a Google Colab](#fine-tuning-whisper-in-a-google-colab)
    1. [Prepare Environment](#prepare-environment)
    2. [Load Dataset](#load-dataset)
    3. [Prepare Feature Extractor, Tokenizer and Data](#prepare-feature-extractor-tokenizer-and-data)
    4. [Training and Evaluation](#training-and-evaluation)
3. [Closing Remarks](#closing-remarks)

We present a step-by-step guide on how to fine-tune Whisper with Common Voice 13.0 dataset using 🤗 Transformers and PEFT. In this Colab, we leverage `PEFT` and `bitsandbytes` to train a `whisper-large-v2` checkpoint seamlessly with a free T4 GPU (16 GB VRAM).

For more details on Whisper fine-tuning, datasets and metrics, refer to Sanchit Gandhi's brilliant blogpost: [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)

## Why Parameter Efficient Fine Tuning ([PEFT](https://github.com/huggingface/peft))?

As the model size continue to increase, fine tuning a model has become both computationally expensive and storage heavy. For example, a `Whisper-large-v2` model requires ~24GB of GPU VRAM to fine-tune for full fine-tuning and requires ~7 GB of storage for each fine-tuned storage. For low-resource environments this becomes quite a bottleneck and often near impossible to get meaningful results.

Cue, PEFT, with PEFT you can tackle this bottleneck head on. PEFT approaches (like Low Rank Adaptation) only fine-tune a small number of (extra) model parameters while freezing most parameters of the pretrained model, thereby greatly decreasing the computational and storage costs. We've observed that it also overcomes the issues of catastrophic forgetting, a behaviour observed during the full finetuning of large models.

### Aha! So wait, what's this LoRA thing?

PEFT comes out-of-the-box with multiple parameter efficient techniques. One such technique is [Low Rank Adaptation or LoRA](https://github.com/microsoft/LoRA). LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. This greatly reduces the number of trainable parameters for downstream tasks. 

LoRA performs on-par or better than fine-tuning in model quality despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency.

### That's all cool, but show me the numbers?

Don't worry, we got ya! We ran multiple experiments to compare a full fine-tuning of Whisper-large-v2 checkpoint and that with PEFT, here's what we found:

1. We were able to fine-tune a 1.6B parameter model with less than 8GB GPU VRAM. 🤯
2. With significantly less number of traininable parameters, we were able to fit almost **5x** more batch size. 📈
3. The resultant checkpoint were less than 1% the size of the original model, ~60MB (i.e. 1% the size of orignal model) 🚀

To make things even better, all of this comes with minimal changes to the existing 🤗 transformers Whisper inference codebase.

Curious to test this out for yourself? Follow along!