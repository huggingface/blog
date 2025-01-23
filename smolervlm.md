---
title: SmolVLM Grows Smaller – Introducing the 250M & 500M Models!
thumbnail: /blog/assets/smolervlm/banner.png
authors:
- user: andito
- user: mfarre
- user: merve
---

## TLDR

We’re excited to announce two new additions to the SmolVLM family: SmolVLM-256M and SmolVLM-500M. That’s right—250M parameters, making it the smallest Vision Language Model in the world!

We built on everything we learned from SmolVLM 2.2B while focusing on efficiency, data mixtures, and new design trade-offs. We are excited to introduce a pair of models that preserve strong multimodal performance in a fraction of the footprint.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm_ecosystem.png" width="800" height="auto" alt="Image description">


## Overview

- SmolVLM-256M – The world’s smallest VLM!
- SmolVLM-500M – A half-billion-parameter sibling that offers a significant performance bump while still remaining super lightweight.
- New Vision Encoder Choices – We compared SigLIP 400M SO (used in SmolVLM 2.2B and many other large VLMs) against a smaller SigLIP base patch-16/512. Surprisingly, the bigger encoder offered only marginally better results, so we opted for the 93M-parameter SigLIP base patch-16/512 in these new releases.
- Larger Image Resolution – Our smaller vision encoder processes images at a larger resolution (inspired by Apple’s VLM research and Google’s PaLiGemma). This yields sharper image understanding with minimal overhead.
- Training Optimization – A new tokenization trick significantly boosted real-world benchmarks, even though it made the training loss look worse on paper.
- We're now reaching model parith with the SmolLM2 family (135M, 360M, 1.7B), so you have a complete set of smaller LLM + VLM combos to play with.


## Why Go Smaller?

When we released SmolVLM 2B, the community response was fantastic: The model is very light weight, open-source and permissive, and easy to integrate into existing workflows. But we wanted to push this approach even further for people working with constrained devices, consumer laptops, or even potentially browser-based inference. That’s where our new 250M and 500M models come in. On the other side, for people trying to process huge amounts of data, these models can can run at a fraction of the cost of the 2.2B model.

## Meet the 250M Parameter Giant 

With just 250 million parameters, this model stands as the tiniest VLM ever. Despite its small size, it packs a surprising punch. It’s more than capable on many multimodal tasks, including:

- Captioning: Describing images or short videos.
- Document Q&A: Answering questions about PDFs or scanned text.
- Basic Visual Reasoning: Answering questions about charts or diagrams.

We also found that it's super easy to fine-tune. We've been collaborating with IBM's Docling and Illuin Technology's ColiPali to showcase how even this tiny model can become a specialized model with incredible performance on real-world tasks. We think the 256M model can become a great specialized model for many tasks.


## A Step Up: 500M

If you need more performance headroom while still keeping the memory usage low, SmolVLM-500M is our half-billion-parameter compromise. It’s significantly smaller than the previous 2B release yet manages to push scores on tasks like DocVQA and MMMU closer to the bigger models.

## What Changed Since SmolVLM 2B?

1. Vision Encoder Choices
Previously, we used the standard SigLIP 400M SO vision backbone, the same one found in many VLM architectures. For these smaller models, we experimented with two setups:

- SigLIP 400M SO: Higher capacity, decent performance.
- SigLIP base patch-16/512 (93M): Much smaller, surprisingly close performance.

We found the performance gap wasn’t big enough to justify the heavier encoder for our 250M and 500M releases. So, we decided to go small on the vision encoder, too. As a bonus, the smaller encoder processes images at a larger resolution, which (per research from Apple and Google) can often yield better visual understanding without ballooning parameter counts.

2. Tokenization Trick: “Sometimes Worse Loss is Better”
During training, we discovered a fascinating effect (documented in this LinkedIn post). We added special tokens to represent our sub-image separators in a more efficient way. This means that now instead of a string like <row_1_col_1> being mapped to 7 tokens, it is mapped to a single token. As any strings up to <row_6_col_6>. 

Training & validation losses: got worse.
Benchmarks: improved significantly!
Why? By minimizing “token noise,” the model zeroed in on the actual content rather than being distracted by repeated placeholder tokens. A classic ML lesson in practice: lower losses ≠ better generalization.

3. Completing the SmolLM2-SmolVLM family

SmolLM2 came in three sizes: 135M, 360M, and 1.7B. With the two models we are releasing today, we now have a complete set of smaller LLM + VLM combos to play with.

Performance & Benchmarks

These tiny models pack a punch. The 250M model is the smallest VLM ever released, yet it surpasses the 

| Size  | Mathvista | MMMU | OCRBench | MMStar | AI2D  | ChartQA_Test | Science_QA | TextVQA Val | DocVQA Val |
|-------|-----------|------|----------|--------|-------|--------------|------------|-------------|------------|
| 250M  | 35.9      | 28.3 | 52.6     | 34.6   | 47    | 13.8         | 73.6       | 51.7        |            |
| 500M  | 40.1      | 33.7 | 61       | 38.3   | 59.5  | 23.1         | 79.7       | 58.6        |            |
| 2.2B  | 43.9      | 38.3 | 65.5     | 41.8   | 64    | 71.64        | 84.5       | 70.6        | 79.7       |
