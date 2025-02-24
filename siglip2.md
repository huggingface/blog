---
title: "SigLIP 2: A better multilingual vision language encoder"
thumbnail: /blog/assets/siglip2/thumbnail.png
authors:
- user: ariG23498
- user: merve
- user: qubvel-hf
---

# SigLIP 2: A better multilingual vision language encoder

## TL;DR

Today Google releases a new and better family of *multilingual* vision-language encoders, [SigLIP 2](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107). The authors have extended the training objective of SigLIP (*sigmoid loss*) with additional objectives for improved semantic understanding, localization, and dense features.

| ![Objectives added](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/sg2-blog/decoder.png) | 
|:--:| 
| *Additional objectives (Source: https://huggingface.co/papers/2502.14786)* |

SigLIP 2 models **outperform** the older SigLIP ones *at all model scales* in core capabilities, including zero-shot classification, image-text retrieval, and transfer performance when extracting visual representations for Vision-Language Models (VLMs).

A cherry on top is the dynamic resolution (`naflex`) variant. This is useful for downstream tasks sensitive to aspect ratio and resolution.

Here is a list of all the models released:

| Size | Patch Size | Resolution | Transformers | JAX |  
| :--: | :--: | :--: | --: |  --: |
| Base (86M) | 32 | 256 | [google/siglip2-base-patch32-256](https://huggingface.co/google/siglip2-base-patch32-256) | [google/siglip2-base-patch32-256-jax](https://huggingface.co/google/siglip2-base-patch32-256-jax) |
| | 16 | 224 | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) | [google/siglip2-base-patch16-224-jax](https://huggingface.co/google/siglip2-base-patch16-224-jax) |  
| | | 256 | [google/siglip2-base-patch16-256](https://huggingface.co/google/siglip2-base-patch16-256) | [google/siglip2-base-patch16-256-jax](https://huggingface.co/google/siglip2-base-patch16-256) |
| | | 384 | [google/siglip2-base-patch16-384](https://huggingface.co/google/siglip2-base-patch16-384) | [google/siglip2-base-patch16-384-jax](https://huggingface.co/google/siglip2-base-patch16-384-jax) |  
| | | 512 | [google/siglip2-base-patch16-512](https://huggingface.co/google/siglip2-base-patch16-512) | [google/siglip2-base-patch16-512-jax](https://huggingface.co/google/siglip2-base-patch16-512-jax) |  
| | - | - | [google/siglip2-base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex) | [google/siglip2-base-patch16-naflex-jax](https://huggingface.co/google/siglip2-base-patch16-naflex-jax) |  
| Large (303M) | 16 | 256 | [google/siglip2-large-patch16-256](https://huggingface.co/google/siglip2-large-patch16-256) | [google/siglip2-large-patch16-256-jax](https://huggingface.co/google/siglip2-large-patch16-256-jax) |  
| | | 384 | [google/siglip2-large-patch16-384](https://huggingface.co/google/siglip2-large-patch16-384) | [google/siglip2-large-patch16-384-jax](https://huggingface.co/google/siglip2-large-patch16-384-jax) | 
| | | 512 | [google/siglip2-large-patch16-512](https://huggingface.co/google/siglip2-large-patch16-512) | [google/siglip2-large-patch16-512-jax](https://huggingface.co/google/siglip2-large-patch16-512-jax) |  
| Shape Optimized 400M | 14 | 224 | [google/siglip2-so400m-patch14-224](https://huggingface.co/google/siglip2-so400m-patch14-224) | [google/siglip2-so400m-patch14-224-jax](https://huggingface.co/google/siglip2-so400m-patch14-224-jax) |
| | | 384 | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | [google/siglip2-so400m-patch14-384-jax](https://huggingface.co/google/siglip2-so400m-patch14-384-jax) | 
| | 16 | 256 | [google/siglip2-so400m-patch16-256](https://huggingface.co/google/siglip2-so400m-patch16-256) | [google/siglip2-so400m-patch16-256-jax](https://huggingface.co/google/siglip2-so400m-patch16-256-jax) |
| | | 384 | [google/siglip2-so400m-patch16-384](https://huggingface.co/google/siglip2-so400m-patch16-384) | [google/siglip2-so400m-patch16-384-jax](https://huggingface.co/google/siglip2-so400m-patch16-384-jax) |  
| | | 512 | [google/siglip2-so400m-patch16-512](https://huggingface.co/google/siglip2-so400m-patch16-512) | [google/siglip2-so400m-patch16-512](https://huggingface.co/google/siglip2-so400m-patch16-512) |
| | - | - | [google/siglip2-so400m-patch16-naflex](https://huggingface.co/google/siglip2-so400m-patch16-naflex) | [google/siglip2-so400m-patch16-naflex-jax](https://huggingface.co/google/siglip2-so400m-patch16-naflex-jax) |
| Giant (1B) | 16 | 256 | [google/siglip2-giant-opt-patch16-256](https://huggingface.co/google/siglip2-giant-opt-patch16-256) | [google/siglip2-giant-opt-patch16-256-jax](https://huggingface.co/google/siglip2-giant-opt-patch16-256-jax) |
| | | 384 | [google/siglip2-giant-opt-patch16-384](https://huggingface.co/google/siglip2-giant-opt-patch16-384) | [google/siglip2-giant-opt-patch16-384-jax](https://huggingface.co/google/siglip2-giant-opt-patch16-384-jax) |

## Introduction

Vision encoders are simple - they take an image, encode it into a representation, and that representation is used for downstream tasks like classification, object detection, image segmentation, and more vision tasks. Researchers are always in pursuit of visual representations that are **dense**, **locality-aware**, and **semantically rich**.

[CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) and [ALIGN](https://huggingface.co/docs/transformers/en/model_doc/align) are the first examples of image encoders and text encoders aligned together through joint training. This approach opened new ways to train vision models. [SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) took it further, replacing CLIP's *contrastive loss* with *sigmoid loss* for even better encoders.

The takeaway? With smarter training objectives, we keep building vision encoders that are more structured, fine-grained, and powerful. SigLIP 2 is just that, a bunch of really interesting and smart training objectives applied on top of that of SigLIP's to provide better and stronger vision language encoders.

We will try something new with this blog post. Rather than stating what is new and where to find it, we will go through a little exercise together. We start off with SigLIP and then brainstorm a series of questions (prefixed with 🤔) and answers (a new heading) to gradually cover all the updates in SigLIP 2. Sounds good?

We will begin our journey with the vision encoder where the patch size is **16**, and the image resolution is
**256**. We have four variants to start our training:

1. [siglip2-base-patch16-256](https://hf.co/google/siglip2-base-patch16-256)
2. [siglip2-large-patch16-256](https://hf.co/google/siglip2-large-patch16-256)
3. [siglip2-so400m-patch16-256](https://hf.co/google/siglip2-so400m-patch16-256)
4. [siglip2-giant-opt-patch16-256](https://hf.co/google/siglip2-giant-opt-patch16-256)

**🤔 Question 1: What is a (low effort) auxiliary training objective that we can use to learn better visual representations (in terms of location awareness and sense of locality)?**

## Add a decoder (it’s that simple)

Let’s add a decoder to the mix. Now we have an image encoder, a text encoder, and a *text decoder*. The text decoder will have three objectives:

1. Predict a holistic image caption  
2. Predict bounding box coordinates given captions describing *specific* image regions  
3. Predict region-specific caption given bounding box coordinates

The decoder provides an additional signal to the vision encoder, making it location-aware. This marks the first improvement to the training recipe in SigLIP 2.

**🤔 Question 2: How do we improve fine-grained local semantics of the image representation?**

## Self-distillation with Global-Local loss and Masked Prediction

To improve fine-grained local semantics in image representation, we introduce two key training objectives, Global-Local Loss, and Masked Prediction Loss. Taking inspiration from self-supervised learning literature, we use *self-distillation*. We can use a model as a teacher, and the same model as a student. Upon each iteration the teacher will be the moving average of the student's parameters.

1. **Global-Local Loss**: The student network gets a partial (local) view of the training image, and is trained to match the teacher’s representation, derived from the full image.  
2. **Masked Prediction Loss**: 50% of the embedded image patches in the student network are masked with mask tokens. The student needs to match the features of the teacher at masked locations.

These objectives teach the vision encoder to be spatially aware and improve its local semantics. The authors add this loss only after **80%** of the training is done with the sigmoid and decoder loss. This is done in order to save compute (additional losses are pretty expensive) and to not negatively affect the encoders.

**🤔 Question 3: How to adapt models to different resolutions?**

## Adapting to different resolutions

It is a known fact that image models can be very sensitive to varying resolutions and aspect ratios. Here we can leverage two distinct methodologies to adapt these models on different resolutions and patch sizes.

1. **Fixed resolution variant**: Taking the checkpoints from 95% training, we can resize the positional embeddings and the patch embeddings and then continue training for a requested (potentially larger) resolution.  
2. **Dynamic resolution variant**: Taking inspiration from [FlexiViT](https://huggingface.co/papers/2212.08013), which uses inputs with different sequence lengths, and [NaViT](https://huggingface.co/papers/2307.06304), which adheres to the native aspect ratios, we can create **NaFlex** variants. This is interesting because we can use a single model for OCR (little aspect ratio distortion) and document understanding (appropriate resolution).

> [!NOTE]  
> Models with the `-naflex` suffix are the dynamic resolution variants. While the fixed-resolution models can be used out of the box with the existing `SiglipModel` class, you would need to use `Siglip2Model` to use the `naflex` variants. We handle this automatically when you use the pipeline API!

This brings us to the end of the evolution from SigLIP to SigLIP 2. In the next sections we will look at applications with SigLIP 2.

## Run inference with transformers

Running inference on the models is pretty straightforward. You can copy paste the code below and run inference on a free tier Colab notebook 🚀

> [!NOTE]
> To run inference on SigLIP 2, please install `transformers` from `main` or from this stable branch:
> `pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2`

### Zero-shot Classification

Here we use the handy `pipeline` API to showcase zero-shot classification capabilities for SigLIP 2.

```py
from transformers import pipeline

ckpt = "google/siglip2-so400m-patch14-384"
pipe = pipeline(model=ckpt, task="zero-shot-image-classification")

inputs = {
    "images": [
        "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg", # bear
        "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000776.jpg", # teddy bear
    ],
    "texts": [
        "bear looking into the camera",
        "bear looking away from the camera",
        "a bunch of teddy bears",
        "two teddy bears",
        "three teddy bears"
    ],
}

outputs = pipe(inputs["images"], candidate_labels=inputs["texts"])
```

Let’s visualize the outputs.

| ![output of zero shot classification](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/sg2-blog/zero-shot.png) |
| :--: |
| *Zero Shot Classification Scores Visualized* |

### Encode images for downstream tasks

You can also encode images using the following:

```py
import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

ckpt = "google/siglip2-so400m-patch14-384"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
inputs = processor(images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs)    

print(image_embeddings.shape) # torch.Size([1, 1152])
```

## Comparing SigLIP 1 with SigLIP 2

Looking at the table of all the SigLIP 2 models released, we see two distinct changes from SigLIP:

1. SigLIP 2 has new variants (`naflex`) for dynamic resolution.  
2. SigLIP 2 adds a `giant` (1B) series.

The evaluation table of SigLIP 2 demonstrates its superiority over SigLIP.

| ![evaluation table](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/sg2-blog/eval_table.png) |
| :--: |
| *Evaluation Scores for SigLIP 2 (Source: https://huggingface.co/papers/2502.14786)* |

Here is a demo where one can compare the zero-shot classification results of SigLIP 1 and SigLIP 2.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/5.16.1/gradio.js"></script>
<gradio-app src="https://google-zero-shot-sg1-sg2.hf.space"></gradio-app>

## Using the encoder for VLMs

Vision encoders aligned to textual information have become increasingly vital in the development of **Vision Language Models** (VLMs). A common approach to building VLMs involves combining a pretrained vision encoder with a pretrained LLM, and training them together using multimodal data across a diverse set of vision-language tasks.

One standout example of a VLM leveraging the SigLIP family of vision encoders is **PaliGemma**. One can dive deeper into PaliGemma's capabilities in this [PaliGemma](https://huggingface.co/blog/paligemma) blog post. Building on this foundation, the recently introduced [PaliGemma 2](https://huggingface.co/blog/paligemma2) takes it a step further by integrating SigLIP with the advanced Gemma 2 LLM. It would be really exciting to swap out SigLIP with SigLIP 2 in a PaliGemma like setting and see how that model fares.

## Acknowledgements

We would like to thank [Michael Tschannen](https://huggingface.co/mitsch) (first author of SigLIP 2), [Vaibhav Srivastav](https://huggingface.co/reach-vb) and [Sayak Paul](https://huggingface.co/sayakpaul) for feedback on this blog post. A huge shout out to the Google team for releasing this amazing, and open, model family.

In no particular order we would like to thank [Pavel](https://huggingface.co/qubvel-hf), [Ross](https://huggingface.co/rwightman), [Pablo](https://huggingface.co/Molbap), [Pedro](https://huggingface.co/pcuenq), [Lysandre](https://huggingface.co/lysandre) and the rest of the Hugging Face team for their immense support and contribution towards this project.
