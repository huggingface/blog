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

## Fine-tuning Whisper in a Google Colab

### Prepare Environment

We'll employ several popular Python packages to fine-tune the Whisper model.
We'll use `datasets` to download and prepare our training data and 
`transformers` to load and train our Whisper model. We'll also require
the `librosa` package to pre-process audio files, `evaluate` and `jiwer` to
assess the performance of our model. Finally, we'll
use `PEFT`, `bitsandbytes`, `accelerate` to prepare and fine-tune the model with LoRA.

```python
!pip install -q transformers datasets librosa evaluate jiwer gradio bitsandbytes==0.37 accelerate 
!pip install -q git+https://github.com/huggingface/peft.git@main
```

We strongly advise you to upload model checkpoints directly the [Hugging Face Hub](https://huggingface.co/) 
whilst training. The Hub provides:
- Integrated version control: you can be sure that no model checkpoint is lost during training.
- Tensorboard logs: track important metrics over the course of training.
- Model cards: document what a model does and its intended use cases.
- Community: an easy way to share and collaborate with the community!

Linking the notebook to the Hub is straightforward - it simply requires entering your Hub authentication token when prompted. Find your Hub authentication token [here](https://huggingface.co/settings/tokens):

```python
from huggingface_hub import notebook_login

notebook_login()
```

# Load Dataset

Using 🤗 Datasets, downloading and preparing data is extremely simple. 
We can download and prepare the Common Voice splits in just one line of code. 

First, ensure you have accepted the terms of use on the Hugging Face Hub: [mozilla-foundation/common_voice_13_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0). Once you have accepted the terms, you will have full access to the dataset and be able to download the data locally.

Since Hindi is very low-resource, we'll combine the `train` and `validation` 
splits to give approximately 12 hours of training data. We'll use the 6 hours 
of `test` data as our held-out test set:

```python
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True)

print(common_voice)
```

**Print output:**

```
DatasetDict({
    train: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 6760
    })
    test: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 2947
    })
})
```

Most ASR datasets only provide input audio samples (`audio`) and the 
corresponding transcribed text (`sentence`). Common Voice contains additional 
metadata information, such as `accent` and `locale`, which we can disregard for ASR.
Keeping the notebook as general as possible, we only consider the input audio and
transcribed text for fine-tuning, discarding the additional metadata information:

```python
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"]
)

print(common_voice)
```

**Print output:**
```
DatasetDict({
    train: Dataset({
        features: ['audio', 'sentence'],
        num_rows: 6760
    })
    test: Dataset({
        features: ['audio', 'sentence'],
        num_rows: 2947
    })
})
```

### Prepare Feature Extractor, Tokenizer and Data

The ASR pipeline can be de-composed into three stages: 
1. A feature extractor which pre-processes the raw audio-inputs
2. The model which performs the sequence-to-sequence mapping 
3. A tokenizer which post-processes the model outputs to text format

In 🤗 Transformers, the Whisper model has an associated feature extractor and tokenizer, 
called [WhisperFeatureExtractor](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperFeatureExtractor)
and [WhisperTokenizer](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperTokenizer) 
respectively.

```python
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
```

```python
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
```

To simplify using the feature extractor and tokenizer, we can _wrap_ both into a single `WhisperProcessor` class. This processor object can be used on the audio inputs and model predictions as required. 
In doing so, we only need to keep track of two objects during training: 
the `processor` and the `model`:

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
```