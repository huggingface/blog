---
title: "Accelerating vison-language training and inference: BridgeTower on Gaudi2"
thumbnail: /blog/assets/bridgetower/thumbnail.png
authors:
- user: regisss
- user: anahita-b
  guest: true
---

# Accelerating vison-language model training and inference: BridgeTower on Gaudi2

<!-- {blog_metadata} -->
<!-- {authors} -->

Introduction

In this blog post, we are going to show how to efficiently perform training and inference in *bfloat16* precision with BridgeTower, a multimodal SOTA model. *bfloat16* is a particularly interesting data type as it covers the same range of values as single-precision 32-bit *float* (or just *float* in the following), which prevents overflow issues that can occur with half-precision 16-bit *float (or just *float16* in the following).


## BridgeTower

In the recent past, Vision-Language(VL) models have gained tremendous importance and shown dominance in a variety of VL tasks. Most common approaches leverage uni-modal encoders to extract representations from their respective modalities and then fuse them simultaneously or feed the respective last layers to a cross-modal encoder. To efficiently handle some of the performance limitations and restrictions in VL representation learning caused by these approaches, BridgeTower introduces multiple bridge layers that build a connection between the top layers of uni-modal encoders and each layer of the cross-modal encoder. This enables effective bottom-up cross-modal alignment and fusion between visual and textual representations of different semantic levels of pre-trained uni-modal encoders in the cross-modal encoder. Pre-trained with only 4M images, BridgeTower achieves state-of-the-art performance on various downstream vision-language tasks. In particular, on the VQAv2 test-std set, BridgeTower achieves an accuracy of 78.73%, outperforming the previous state-of-the-art model METER by 1.09% with the same pre-training data and almost negligible additional parameters and computational costs. Notably, when further scaling the model, BridgeTower achieves an accuracy of 81.15%, surpassing models that are pre-trained on orders-of-magnitude larger datasets.


## Hardware

Several devices supporting *bfloat16* computation have been released in the last few years.
[Nvidia A100](https://www.nvidia.com/en-us/data-center/a100/) and [Habana Gaudi2](https://habana.ai/products/gaudi2/) are among them and will be benchmarked in the following sections.

NVIDIA A100 GPU is the fastest GPU that you will find at most cloud providers. We use here the 80GB-memory variant which also offers faster memory bandwidth than the 40GB one.

Gaudi2 is the second-generation AI hardware accelerator designed by Habana Labs. A single server contains 8 accelerator devices with 96GB of memory each. Check out [this previous blog post](https://huggingface.co/blog/habana-gaudi-2-bloom#habana-gaudi2) for a more in-depth presentation and a guide showing how to access it through the [Intel Developer Cloud](https://www.intel.com/content/www/us/en/secure/developer/devcloud/cloud-launchpad.html). Furthermore, unlike many AI accelerators on the market, using advanced features to make the most of Gaudi2 is very easy with [Optimum Habana](https://github.com/huggingface/optimum-habana), which enables to port your scripts relying on Transformers to Gaudi with just a 2-line change.


## Training

Dataset, hyperparameters, link to Tensorboard logs
Compare perf before/after fine-tuning

To benchmark training, we are going to fine-tune BridgeTower Large using [this checkpoint](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc). It contains 955M parameters and was pretrained on English language using masked language modeling, image-text matching and image-text contrastive loss on [Conceptual Captions](https://huggingface.co/datasets/conceptual_captions), [SBU Captions](https://huggingface.co/datasets/sbu_captions), [MSCOCO Captions](https://huggingface.co/datasets/HuggingFaceM4/COCO) and [Visual Genome](https://huggingface.co/datasets/visual_genome).

This checkpoint is fine-tuned on the [New Yorker Caption Contest dataset](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest) which consists of cartoons from The New Yorker and the most voted captions.

Hyperparameters are all the same for both accelerators, except the batch size. We managed to fit 48 samples on Gaudi2 against 32 on A100. All runs

We first started with two runs:
- a mixed-precision (*bfloat16*/*float*) run distributed on 8 devices where data loading is performed by the same process as everything else (MP-0)
- a mixed-precision (*bfloat16*/*float*) run distributed on 8 devices with 1 dedicated subprocess for data loading (MP-1)

| Device     | MP-0            | MP-1            |
|:----------:|:---------------:|:---------------:|
| Gaudi2 HPU | 581.5 samples/s | 678.6 samples/s |
| A100 GPU   |                 | 249.6 samples/s |

**Gaudi2 is x2.72 faster than A100**, which is even better than [the speedups we previously reported](https://huggingface.co/blog/habana-gaudi-2-benchmark)!

Besides, we see that **allocating more resources for data loading can lead to easy speedups**: x1.17 on Gaudi2 and x1.xx on A100. This is because our dataset contains images and thus data loading is a much heavier task than with text.

We also ran experiments with several dedicated subprocesses for data loading but performance was not better than with 1.

To improve this even more, we can try to move as many operations (image decoding and augmentations) as possible from CPU to the accelerator devices. This can be done on Gaudi2 using Habana's media pipe.


Finally, we are going to use two new features of Optimum Habana v1.6.0 to get an additional speedup:
- HPU graphs for training, which enables to reduce the CPU overhead
- Fast DDP, which is a lighter and faster implementation of Torch DDP


## Inference

Benchmark latencies
Compare BridgeTower vs CLIP on Winoground


## Conclusion

We will release a second part of this blog where we present how to easily perform *bfloat16* inference on Intel Sapphire Rapids CPUs.
