---
title: SmolVLM Grows Smaller – Introducing the 256M & 500M Models!
thumbnail: /blog/assets/smolervlm/banner.png
authors:
- user: andito
- user: mfarre
- user: merve
---

## TLDR

We’re excited to announce two new additions to the SmolVLM family: SmolVLM-256M and SmolVLM-500M. That’s right—256M parameters, making it the smallest Vision Language Model in the world!

We built on everything we learned from SmolVLM 2B while focusing on efficiency, data mixtures, and new design trade-offs. We are excited to introduce a pair of models that preserve strong multimodal performance in a fraction of the footprint. 

This release comes with four checkpoints: two base models and two instruction fine-tuned models with sizes 256M and 500M parameters. These models can be loadable directly to transformers, MLX and ONNX, and we have demos for transformers and WebGPU (with ONNX). You can find all the models and the demo for this release [here](https://huggingface.co/collections/HuggingFaceTB/smolvlm-256m-and-500m-6791fafc5bb0ab8acc960fb0).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smoller_vlm_benchmarks.png" alt="Benchmarks" style="width:90%;" />

## Table of Contents

- [Overview](#overview)
- [Why Go Smaller?](#why-go-smaller)
    - [Meet the 256M Parameter Giant](#meet-the-256m-parameter-giant)
    - [A Step Up: 500M](#a-step-up-500m)
- [What Changed Since SmolVLM 2B?](#what-changed-since-smolvlm-2b)
- [Smaller Multimodal Retrieval: ColSmolVLM 256M & 500M](#meet-smoller-colsmolvlm)
- [Using Smaller SmolVLM](#using-smaller-smolvlm)
- [Next Steps](#next-steps)



## Overview

- **SmolVLM-256M** – The world’s smallest VLM!
- **SmolVLM-500M** – A half-billion-parameter sibling that offers a significant performance bump while still remaining super lightweight.
- **New Vision Encoder Choices** – We compared SigLIP 400M SO (used in SmolVLM 2B and many other large VLMs) against a smaller SigLIP base patch-16/512. Surprisingly, the bigger encoder offered only marginally better results, so we opted for the 93M-parameter SigLIP base patch-16/512 in these new releases.
- **Larger Image Resolution** – Our smaller vision encoder processes images at a larger resolution (inspired by Apple’s VLM research and Google’s PaliGemma). This yields sharper image understanding with minimal overhead.
- **Training Optimization** – A new tokenization trick significantly boosted real-world benchmarks, even though it made the training loss look worse on paper.

We're now reaching model parity with the SmolLM2 family (135M, 360M, 1.7B), so you have a complete set of smaller LLM + VLM combos to play with. 


## Why Go Smaller?

When we released SmolVLM 2B, the community response was fantastic: The model is very light weight, open-source and permissive, and easy to integrate into existing workflows. But we wanted to push this approach even further for people working with constrained devices, consumer laptops, or even potentially browser-based inference. That’s where our new 256M and 500M models come in. On the other side, for people trying to process huge amounts of data, these models can run at a fraction of the cost of the 2B model.

In the last year, we trained two 80B VLMs and reduced them to 8B. Then for SmolVLM we took the challenge of reducing that 2B. And what we learned was that we could push the frontier way further! We are excited to show that at 256M and 500M we can still get great performance. Our new 256M model is the smallest VLM ever released, yet it surpasses the performance of our Idefics 80B model from just 17 months ago.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smoller_vlm_benchmarks.png" alt="Benchmarks" style="width:90%;" />


### Meet the 256M Parameter Giant 

With just 256 million parameters, this model stands as the tiniest VLM ever. Despite its small size, it packs a surprising punch. It’s more than capable on many multimodal tasks, including:

- **Captioning:** Describing images or short videos.
- **Document Q&A:** Answering questions about PDFs or scanned text.
- **Basic Visual Reasoning:** Answering questions about charts or diagrams.


### A Step Up: 500M

If you need more performance headroom while still keeping the memory usage low, SmolVLM-500M is our half-billion-parameter compromise. It’s significantly smaller than the previous 2B release yet manages to push scores on tasks like DocVQA and MMMU closer to the bigger models. We also found this model to be more robust to prompting, which makes it out-of-the-box better fitted for production. But both models do great when fine-tuned.

We have visualized the throughput gains across different batch sizes in below graph. Below numbers are throughput benchmarks ran on A100. 

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/throughput.png" alt="Benchmarks" style="width:90%;" />

## What Changed Since SmolVLM 2B?

**1. Vision Encoder Choices**
Previously, we used the standard SigLIP 400M SO vision backbone, the same one found in many VLM architectures. For these smaller models, we experimented with two setups:

- **SigLIP 400M SO:** Higher capacity, great performance.
- **SigLIP base patch-16/512 (93M):** Much smaller, surprisingly close performance.

We found the performance gap wasn’t big enough to justify the heavier encoder for our 256M and 500M models. So, we decided to go small on the vision encoder, too. As a bonus, the smaller encoder processes images at a larger resolution, which (per research from [Apple](https://arxiv.org/pdf/2403.09611) and [Google](https://arxiv.org/pdf/2412.03555)) can often yield better visual understanding without ballooning parameter counts.

**2. Data mixture update**

Similarly to our previous release, we rely on [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) and [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) with the addition of [MathWriting](https://huggingface.co/datasets/andito/mathwriting-google) to the mix.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm_datamixture.gif" alt="Data mixture" style="width:90%;" />

The proportions of the datasets were adjusted to place a stronger emphasis on document understanding (41%) and image captioning (14%), while still maintaining a balanced focus on other essential areas such as visual reasoning, chart comprehension, and general instruction following.
With this update the model is built on a strong document understanding basis and lets the door open to fine-tunes that will adjust its understanding of specific tasks.


**3. Tokenization optimizations**

We increased the pixel shuffle even more! Our new models encode images at a rate of 4096 pixels per token, compared to 1820 pixels per token in the 2B model.

To optimize the tokenization even more, we added special tokens to represent our sub-image separators in a more efficient way. This means that now instead of a string like `<row_1_col_1>` being mapped to 7 tokens, it is mapped to a single token. We did the same for strings up to `<row_6_col_6>`. This led to a sizeable improvement in the model's stability during training and quality of the results. More details were documented in this [LinkedIn post](https://www.linkedin.com/posts/andimarafioti_when-worse-training-losses-lead-to-better-activity-7284521064934592513-yBZe?utm_source=share&utm_medium=member_desktop).

**4. Completing the SmolLM2-SmolVLM family**

SmolLM2 came in three sizes: 135M, 360M, and 1.7B. With the two models we are releasing today, we now have a complete set of smaller LLM + VLM combos to play with.

## Smaller Multimodal Retrieval: ColSmolVLM 256M & 500M

We also found that it's surprisingly easy to fine-tune and experiment. The team behind the ColBERT-like retrieval models have trained ColSmolVLM, delivering SOTA multimodal retrieval speeds with performance rivaling models 10x their size. SmolVLM makes it faster and cheaper to build searchable databases. We think the 256M model can become a great specialized model for many tasks. Find the link on how to use the new ColSmolVLM with the new SmolVLM models in [Next Steps](#next-steps).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/colsmol_tiny.png" alt="Benchmarks" style="width:90%;" />


## SmolDocling

We partnered with IBM to build models for [Docling](https://github.com/DS4SD/docling). Their early results with the 256M model are impressive. Below are some early examples they shared with us. Stay tuned for more updates on this!


<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smoldocling_layout_table_image.png" alt="Benchmarks" style="width:90%;" />
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smoldocling_code.png" alt="Benchmarks" style="width:90%;" />

## Using Smaller SmolVLM

Newer SmolVLMs are working out-of-the-box with the old SmolVLM code, so you can use transformers and MLX for inference and fine-tuning, and TRL for alignment 🚀 Moreover, this release also comes with ONNX checkpoints.

Get started with SmolVLM using transformers like below.

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

# Preprocess
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
```

Use SmolVLM with MLX by running the following CLI command:

```bash
python3 -m mlx_vlm.generate --model HuggingfaceTB/SmolVLM-500M-Instruct --max-tokens 400 --temp 0.0 --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vlm_example.jpg --prompt "What is in this image?"
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm-mlx.gif" alt="MLX" style="width:90%;" />

You can play with the WebGPU demos for the [SmolVLM-256M-Instruct](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-256M-Instruct-WebGPU) and [SmolVLM-500M-Instruct](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-500M-Instruct-WebGPU).

Find links to fine-tuning and multimodal RAG with ColSmolVLM on the [Next Steps](#next-steps).

## Next Steps

- We are looking forward to ways you will be using SmollerVLMs! Get started [here](https://huggingface.co/collections/HuggingFaceTB/smolvlm-256m-and-500m-6791fafc5bb0ab8acc960fb0).
- Learn more in-depth about SmolVLM [here](https://huggingface.co/blog/smolvlm).
- [Fine-tuning and QLoRA SmolVLM using transformers](https://github.com/merveenoyan/smol-vision/blob/main/Smol_VLM_FT.ipynb)
- [Direct Preference Optimization on SmolVLM using TRL](Fine-tuning SmolVLM using direct preference optimization (DPO) with TRL on a consumer GPU)
- [Smol Multimodal RAG: Building with ColSmolVLM and SmolVLM on Colab’s Free-Tier GPU](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)

We would like to thank ViDoRe team for training ColSmolVLM: [Tony Wu](https://huggingface.co/tonywu71), [Manuel Faysse](https://huggingface.co/manu), and [Joshua Lochner](https://huggingface.co/Xenova) for the ONNX conversion and WebGPU demo and [Vaibhav Srivastav](https://huggingface.co/reach-vb) for his help on this release.
