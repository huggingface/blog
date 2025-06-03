---
title: "nanoVLM: The simplest repository to train your VLM in pure PyTorch" 
thumbnail: /blog/assets/nanovlm/thumbnail.png
authors:
- user: ariG23498
- user: lusxvr
- user: andito
- user: sergiopaniego
- user: merve
- user: pcuenq
- user: reach-vb
---

# nanoVLM: The simplest repository to train your VLM in pure PyTorch

[**nanoVLM**](https://github.com/huggingface/nanoVLM) is the *simplest* way to get started with
**training** your very own Vision Language Model (VLM) using pure PyTorch. It is lightweight *toolkit*
which allows you to launch a VLM training on a [free tier colab notebook](https://colab.research.google.com/github/huggingface/nanoVLM/blob/main/nanoVLM.ipynb).

> We were inspired by [Andrej Karpathy](https://karpathy.ai/)’s [nanoGPT](https://github.com/karpathy/nanoGPT), and provide a similar project for the vision domain.

At its heart, nanoVLM is a **toolkit** that helps you build and train a model that can understand both
images and text, and then generate text based on that. The beauty of nanoVLM lies in its *simplicity*.
The entire codebase is intentionally kept *minimal* and *readable*, making it perfect for beginners or
anyone who wants to peek under the hood of VLMs without getting overwhelmed.

In this blog post, we cover the core ideas behind the project and provide a simple way to interact
with the repository. We not only go into the details of the project but also encapsulate all of it
so that you can quickly get started.


## Table of contents:

- [What is a Vision Language Model?](#what-is-a-vision-language-model)
- [Working with the repository](#working-with-the-repository)
- [Architecture](#architecture)
- [Train your own VLM](#train-your-own-vlm)
- [Run inference on a pre-trained model](#run-inference-on-a-pre-trained-model)
- [Conclusion](#conclusion)
- [References](#references)

## TL;DR

You can start training a Vision Language Model using our nanoVLM toolkit by following these steps:

```bash
# Clone the repo
git clone https://github.com/huggingface/nanoVLM.git

# Execute the training script
python train.py
```

Here is a [Colab notebook](https://colab.research.google.com/github/huggingface/nanoVLM/blob/main/nanoVLM.ipynb)
that will help you launch a training run with no local setup required!

## What is a Vision Language Model?

As the name suggests, a Vision Language Model (VLM) is a multi-modal model that processes two
modalities: vision and text. These models typically take images and/or text as input and generate text as output.

Generating text (output) conditioned on the understanding of images and texts (inputs) is a powerful paradigm.
It enables a wide range of applications, from image captioning and object detection to answering
questions about visual content (as shown in the table below). One thing to note is that nanoVLM
focuses only on Visual Question Answering as the training objective.

<table>
  <tr>
    <td rowspan="4"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nanovlm/cat.jpg" alt="an image of a cat" width="200"/></td>
    <td>Caption the image</td>
    <td>Two cats lying down on a bed with remotes near them</td>
    <td>Captioning</td>
  </tr>
  <tr>
    <td>Detect the objects in the image</td>
    <td><code>&lt;locxx&gt;&lt;locxx&gt;&lt;locxx&gt;&lt;locxx&gt;</code></td>
    <td>Object Detection</td>
  </tr>
  <tr>
    <td>Segment the objects in the image</td>
    <td><code>&lt;segxx&gt;&lt;segxx&gt;&lt;segxx&gt;</code></td>
    <td>Semantic Segmentation</td>
  </tr>
  <tr>
    <td>How many cats are in the image?</td>
    <td>2</td>
    <td>Visual Question Answering</td>
  </tr>
</table>

> [!TIP]  
> If you are interested in learning more about VLMs, we strongly recommend reading our latest blog on the topic: [Vision Language Models (Better, Faster, Stronger)](https://huggingface.co/blog/vlms-2025)

## Working with the repository

"Talk is cheap, show me the code" - Linus Torvalds

In this section, we’ll guide you through the codebase. It’s helpful to keep a
[tab](https://github.com/huggingface/nanoVLM) open for reference as you follow along.

Below is the folder structure of our repository. We have removed helper files for brevity.

```bash
.
├── data
│   ├── collators.py
│   ├── datasets.py
│   └── processors.py
├── generate.py
├── models
│   ├── config.py
│   ├── language_model.py
│   ├── modality_projector.py
│   ├── utils.py
│   ├── vision_language_model.py
│   └── vision_transformer.py
└── train.py
```

## Architecture

```bash
.
├── data
│   └── ...
├── models      # 👈 You are here
│   └── ...
└── train.py     
```

We model nanoVLM after two well known and widely used architectures. Our vision backbone
(`models/vision_transformer.py`) is the standard vision transformer, more specifically Google’s
[SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) vision encoder. Our language
backbone follows the [Llama 3](https://huggingface.co/docs/transformers/en/model_doc/llama3) architecture.

The vision and text modalities are *aligned* using a Modality Projection module. This module takes the
image embeddings produced by the vision backbone as input, and transforms them into embeddings
compatible with the text embeddings from the embedding layer of the language model. These embeddings
are then concatenated and fed into the language decoder. The Modality Projection module consists of a
pixel shuffle operation followed by a linear layer.

| ![diagram of the model architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nanovlm/architecture.png) |
| :--: |
| The architecture of the model (Source: Authors) |

[Pixel shuffle](https://huggingface.co/papers/1609.05158) reduces the number of image tokens, which helps
reduce computational cost and speeds up training, especially for transformer-based language decoders
which are sensitive to input length. The figure below demonstrates the concept.

| ![diagram of pixel shuffle](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nanovlm/pixel-shuffle.png) |
| :--: |
| Pixel Shuffle Visualized (Source: Authors) |

All the files are very lightweight and well documented. We highly encourage you to check them out
individually to get a better understanding of the implementation details (`models/xxx.py`)

While training, we use the following pre-trained backbone weights:

1. Vision backbone: [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224)  
2. Language backbone: [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)

> One could also swap out the backbones with other variants of SigLIP/SigLIP 2 (for the vision backbone) and SmolLM2 (for the language backbone).

## Train your own VLM

Now that we are familiar with the architecture, let's shift gears and talk about how to train your own Vision Language Model using `train.py`.

```bash
.
├── data
│   └── ...
├── models
│   └── ...
└── train.py     # 👈 You are here
```

You can kick off training with:

```bash
python train.py
```

This script is your one-stop shop for the entire training pipeline, including:

* Dataset loading and preprocessing  
* Model initialization  
* Optimization and logging

**Configuration**

Before anything else, the script loads two configuration classes from `models/config.py`:

* `TrainConfig`: Configuration parameters useful for training, like learning rates, checkpoint paths, etc.  
* `VLMConfig`: The configuration parameters used to initialize the VLM, like hidden dimensions, number of attention heads, etc.

**Data Loading**

At the heart of the data pipeline is the `get_dataloaders` function. It:

* Loads datasets via Hugging Face’s `load_dataset` API.  
* Combines and shuffles multiple datasets (if provided).  
* Applies a train/val split via indexing.  
* Wraps them in custom datasets (`VQADataset`, `MMStarDataset`) and collators (`VQACollator`, `MMStarCollator`).

> [!TIP]  
> A helpful flag here is `data_cutoff_idx`, useful for debugging on small subsets.

**Model Initialization**

The model is built via the `VisionLanguageModel` class. If you're resuming from a checkpoint, it’s as easy as:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained(model_path)
```

Otherwise, you get a freshly initialized model with optionally preloaded backbones for both vision and language.

**Optimizer Setup: Two LRs**

Because the modality projector (`MP`) is freshly initialized while the backbones are pre-trained, the
optimizer is split into two parameter groups, each with its own learning rate:

* A higher LR for the MP  
* A smaller LR for the encoder/decoder stack

This balance ensures the MP learns quickly while preserving knowledge in the vision and language backbones.

**Training Loop**

This part is fairly standard but thoughtfully structured:

* Mixed precision is used with `torch.autocast` to improve performance.  
* A cosine learning rate schedule with linear warmup is implemented via `get_lr`.  
* Token throughput (tokens/sec) is logged per batch for performance monitoring.

Every 250 steps (configurable), the model is evaluated on the validation and `MMStar` test datasets. If accuracy improves, the model is checkpointed.

**Logging & Monitoring**

If `log_wandb` is enabled, training stats like `batch_loss`,  `val_loss`, `accuracy`, and `tokens_per_second`
are logged to Weights & Biases for real-time tracking.

Runs are auto-named using metadata like sample size, batch size, epoch count, learning rates, and the date,
all handled by the helper `get_run_name`.

**Push to Hub**

Use the following to push the trained model to the Hub for others to find and test:

```python
model.save_pretrained(save_path)
```

You can easily push them using:

```python
model.push_to_hub("hub/id")
```

## Run inference on a pre-trained model

Using nanoVLM as the toolkit, we have trained a [model and published it to Hub](https://huggingface.co/lusxvr/nanoVLM-222M).
We have used the `google/siglip-base-patch16-224` and `HuggingFaceTB/SmolLM2-135M` as backbones. The model was
trained this for ~6h on a single H100 GPU on ~1.7M samples of the [cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron).

This model isn't intended to compete with SoTA models, but rather to demystify the components and training process of VLMs.

```bash
.
├── data
│   └── ...
├── generate.py     # 👈 You are here
├── models
│   └── ...
└── ...
```

Let’s run inference on the trained model using the `generate.py` script. You can run the generation script using the following command:

```bash
python generate.py
```

This will use the default arguments and run the query “What is this?” on the image `assets/image.png`.  
You can use this script on your own images and prompts like so:

```bash
python generate.py --image path/to/image.png --prompt "You prompt here"
```

If you want to visualize the heart of the script, it is just these lines:

```python
model = VisionLanguageModel.from_pretrained(source).to(device)
model.eval()

tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
image_processor = get_image_processor(model.cfg.vit_img_size)

template = f"Question: {args.prompt} Answer:"
encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
tokens = encoded["input_ids"].to(device)

img = Image.open(args.image).convert("RGB")
img_t = image_processor(img).unsqueeze(0).to(device)

print("\nInput:\n ", args.prompt, "\n\nOutputs:")
for i in range(args.generations):
    gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
    out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    print(f"  >> Generation {i+1}: {out}")
```

We create the model and set it to `eval`. Initialize the tokenizer, which tokenizes the text prompt,
and the image processor, which  is used to process the images. The next step is to process the inputs
and run `model.generate` to generate the output text. Finally, decode the output using  `batch_decode`.

| Image | Prompt | Generation |
| :--: | :--: | :--: |
| ![image of a cat](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nanovlm/cat.jpg) | What is this? | In the picture I can see the pink color bed sheet. I can see two cats lying on the bed sheet. |
| ![yoga](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/yoga1.jpeg) |What is the woman doing? | Here in the middle she is performing yoga |

> [!TIP]  
> If you want to run inference on the trained model in a UI interface, [here](https://huggingface.co/spaces/ariG23498/nanovlm) is the Hugging Face Space for you to interact with the model. 

## Conclusion

In this blog post, we walked through what VLMs are, explored the architecture choices that power nanoVLM, and unpacked the training and inference workflows in detail.

By keeping the codebase lightweight and readable, nanoVLM aims to serve as both a learning tool and a foundation you can build upon. Whether you’re looking to understand how multi-modal inputs are aligned, or you want to train a VLM on your own dataset, this repository gives you a head start.

If you try it out, build on top of it, or just have questions we’d love to hear from you. Happy tinkering!

## References

1. [GitHub - huggingface/nanoVLM: The simplest, fastest repository for training/finetuning small-sized VLMs.](https://github.com/huggingface/nanoVLM)
2. [Vision Language Models (Better, faster, stronger)](https://huggingface.co/blog/vlms-2025)
3. [Vision Language Models Explained](https://huggingface.co/blog/vlms)
4. [A Dive into Vision-Language Models](https://huggingface.co/blog/vision_language_pretraining)
5. [SmolVLM: Redefining small and efficient multimodal models](https://huggingface.co/papers/2504.05299)
