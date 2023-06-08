---
title: "Fast bf16 training and inference: BridgeTower on Gaudi2"
thumbnail: /blog/assets/bridgetower/thumbnail.png
authors:
- user: regisss
- user: anahita-b
  guest: true
---

# Fast bf16 training and inference: BridgeTower on Gaudi2

<!-- {blog_metadata} -->
<!-- {authors} -->

Introduction

In this blog post, we are going to show how to efficiently perform training and inference in *bfloat16* precision with BridgeTower, a multimodal SOTA model. *bfloat16* is a particularly interesting data type as it covers the same range of values as single-precision 32-bit *float* (or just *float* in the following), which prevents overflow issues that can occur with half-precision 16-bit *float (or just *float16* in the following).


## BridgeTower

Vision-Language (VL) models with the Two-Tower architecture have dominated visual-language representation learning in recent years. Current VL models either use lightweight uni-modal encoders and learn to extract, align and fuse both modalities simultaneously in a deep cross-modal encoder, or feed the last-layer uni-modal representations from the deep pre-trained uni-modal encoders into the top cross-modal encoder. Both approaches potentially restrict vision-language representation learning and limit model performance. BridgeTower introduces multiple bridge layers that build a connection between the top layers of uni-modal encoders and each layer of the cross-modal encoder. This enables effective bottom-up cross-modal alignment and fusion between visual and textual representations of different semantic levels of pre-trained uni-modal encoders in the cross-modal encoder. Pre-trained with only 4M images, BridgeTower achieves state-of-the-art performance on various downstream vision-language tasks. In particular, on the VQAv2 test-std set, BridgeTower achieves an accuracy of 78.73%, outperforming the previous state-of-the-art model METER by 1.09% with the same pre-training data and almost negligible additional parameters and computational costs. Notably, when further scaling the model, BridgeTower achieves an accuracy of 81.15%, surpassing models that are pre-trained on orders-of-magnitude larger datasets.


Presentation of the model
How compute-intensive is it for training and inference?


## Hardware

Several devices supporting *bfloat16* computation have been released in the last few years.
[Nvidia A100](https://www.nvidia.com/en-us/data-center/a100/) and [Habana Gaudi2](https://habana.ai/products/gaudi2/) are among them and will be benchmarked in the following sections.

NVIDIA A100 GPU is the fastest GPU that you will find at most cloud providers. We use here the 80GB-memory variant which also offers faster memory bandwidth than the 40GB one.

Gaudi2 is the second-generation AI hardware accelerator designed by Habana Labs. A single server contains 8 accelerator devices with 96GB of memory each. Check out [this previous blog post](https://huggingface.co/blog/habana-gaudi-2-bloom#habana-gaudi2) for a more in-depth presentation and a guide showing how to access it through the [Intel Developer Cloud](https://www.intel.com/content/www/us/en/secure/developer/devcloud/cloud-launchpad.html). Furthermore, unlike many AI accelerators on the market, using advanced features to make the most of Gaudi2 is very easy with [Optimum Habana](https://github.com/huggingface/optimum-habana), which enables to port your scripts relying on Transformers to Gaudi with just a 2-line change.


## Training

Dataset, hyperparameters, link to Tensorboard logs
Benchmark throughputs
Compare perf before/after fine-tuning

To benchmark training, we are going to fine-tune BridgeTower Large using [this checkpoint](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc). It contains 955M parameters and was pretrained on English language using masked language modeling, image-text matching and image-text contrastive loss on [Conceptual Captions](https://huggingface.co/datasets/conceptual_captions), [SBU Captions](https://huggingface.co/datasets/sbu_captions), [MSCOCO Captions](https://huggingface.co/datasets/HuggingFaceM4/COCO) and [Visual Genome](https://huggingface.co/datasets/visual_genome).

This checkpoint is fine-tuned on the [New Yorker Caption Contest dataset](https://nextml.github.io/caption-contest-data) which consists of cartoons from The New Yorker and the most voted captions.

Several runs were performed:
- a mixed-precision (*bfloat16*/*float*) run (MP-0)
- a mixed-precision (*bfloat16*/*float*) run with 1 dedicated subprocess for data loading (MP-1)
- a mixed-precision (*bfloat16*/*float*) run with 16 dedicated subprocesses for data loading (MP-16)
- (on Gaudi2 only) a mixed-precision (*bfloat16*/*float*) run with a dataloader relying on Habana's media pipeline and HPU graphs enabled (MP-*)


| Device     | MP-0            | MP-1 | MP-16 | MP-* |
|:----------:|:---------------:|:----:|:-----:|:----:|
| Gaudi2 HPU |  | 678.6 samples/s     |       |      |
| A100 GPU   |  | 249.6 samples/s     |       |      |

Talk a bit about the importance of data loading for such use cases


## Inference

Benchmark latencies
Compare BridgeTower vs CLIP on Winoground


## Conclusion

We will release a second part of this blog where we present how to easily perform *bfloat16* inference on Intel Sapphire Rapids CPUs.
