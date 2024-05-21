---
title: "Hugging Face on AMD Instinct MI300 GPU"
thumbnail: /blog/assets/optimum_amd/amd_hf_logo_fixed.png
authors:
- user: fxmarty
- user: mohitsha
- user: seungrokj
  guest: true
  org: amd
- user: mfuntowicz
---

# Hugging Face on AMD Instinct MI300 GPU

## Introduction
At Hugging Face we want to make it easy to build AI with open models and open source, whichever framework, cloud and stack you want to use.
A key component is the ability to deploy AI models on a versatile choice of hardware. 
Through our collaboration with AMD, for about a year now, we are investing into multiple different accelerators such as AMD Instinct™ and Radeon™ GPUs, EPYC™ and Ryzen™ CPUs and Ryzen AI NPUs helping ensure there will always be a device to run the largest AI
community on the AMD fleet. 
Today we are delighted to announce that Hugging Face and AMD have been hard at work together to enable the latest generation of AMD GPU servers, namely AMD Instinct MI300, to have first-class citizen integration in the overall Hugging Face Platform. 
From prototyping in your local environment, to running models in production on Azure ND Mi300x V5 VMs, you don't need to make any code change using transformers[1], text-generation-inference and other libraries, or when you use Hugging Face products and solutions - we want to make it super easy to use AMD MI300 on Hugging Face and get the best performance.
Let’s dive in!

## Open-Source and production enablement
### Maintaining support for AMD Instinct GPUs in Transformers and text-generation-inference

With so many things happening right now in AI it was absolutely necessary to make sure the MI300 line-up is correctly tested and monitored in the long-run. 
To achieve this, we have been working closely with the infrastructure team here at Hugging Face to make sure we have robust building blocks available for whoever requires to enable continuous integration and deployment (CI/CD) and to be able to do so without pain and without impacting the others already in place.

To enable such things, we worked together with AMD and Microsoft Azure teams to leverage the recently introduced [Azure ND MI300x V5](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/introducing-the-new-azure-ai-infrastructure-vm-series-nd-mi300x/ba-p/4145152) as the building block targeting MI300.
In a couple of hours our infrastructure team was able to deploy, setup and get everything up and running for us to get our hands on the MI300!

We also moved away from our old infrastructure to a managed Kubernetes cluster taking care of scheduling all the Github workflows Hugging Face collaborators would like to run on hardware specific pods.
This migration now allows us to run the exact same CI/CD pipeline on a variety of hardware platforms abstracted away from the developer.
We were able to get the CI/CD up and running within a couple of days without much effort on the Azure MI300X VM.

As a result, transformers and text-generation-inference are now being tested on a regular basis on both the previous generation of AMD Instinct GPUs, namely MI250 and also on the latest MI300. 
In practice, there are tens of thousands of unit tests which are regularly validating the state of these repositories ensuring the correctness and robustness of the integration in the long run.

## Improving performances for production AI workloads
### Inferencing performance

As said in the prelude, we have been working on enabling the new AMD Instinct MI300 GPUs to efficiently run inference workloads through our open source inferencing solution, text-generation-inference (TGI)
TGI can be seen as three different components: 
-	A transport layer, mostly HTTP, exposing and receiving API requests from clients
-	A scheduling layer, making sure these requests are potentially batched together (i.e. continuous batching) to increase the computational density on the hardware without impacting the user experience
-	A modeling layer, taking care of running the actual computations on the device, leveraging highly optimized routines involved in the model

Here, with the help of AMD engineers, we focused on this last component, the modeling, to effectively setup, run and optimize the workload for serving models as the [Meta Llama family](https://huggingface.co/meta-llama). In particular, we focused on:
-	Flash Attention v2
-	Paged Attention
-	GPTQ/AWQ compression techniques
-	PyTorch integration of [ROCm TunableOp](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable)
-	Integration of optimized fused kernels

Most of these have been around for quite some time now, [FlashAttention v2](https://huggingface.co/papers/2307.08691), [PagedAttention](https://huggingface.co/papers/2309.06180) and [GPTQ](https://huggingface.co/papers/2210.17323)/[AWQ](https://huggingface.co/papers/2306.00978) compression methods (especially their optimized routines/kernels). We won’t detail the three above and we invite you to navigate to their original implementation page to learn more about it. 

Still, with a totally new hardware platform, new SDK releases, it was important to carefully validate, profile and optimize every bit to make sure the user gets all the power from this new platform.

Last but not least, as part of this TGI release, we are integrating the recently released AMD TunableOp, part of PyTorch 2.3.
TunableOp provides a versatile mechanism which will look for the most efficient way, with respect to the shapes and the data type, to execute general matrix-multiplication (i.e. GEMMs).
TunableOp is integrated in PyTorch and is still in active development but, as you will see below, makes it possible to improve the performance of GEMMs operations without significantly impacting the user-experience. 
Specifically, we gain a 8-10% speedup in latency using TunableOp for small input sequences, corresponding to the decoding phase of autoregressive models generation.

In fact, when a new TGI instance is created, we launch an initial warming step which takes some dummy payloads and makes sure the model and its memory are being allocated and are ready to shine. 

With TunableOp, we enable the GEMM routine tuner to allocate some time to look for the most optimal setup with respect to the parameters the user provided to TGI such as sequence length, maximum batch size, etc. 
When the warmup phase is done, we disable the tuner and leverage the optimized routines for the rest of the server’s life.

As said previously, we ran all our benchmarks using Azure ND MI300x V5, recently introduced at Microsoft BUILD, which integrates eight AMD Instinct GPUs onboard, against the previous generation MI250 on Meta Llama 3 70B, deployment, we observe a 2x-3x speedup in the time to first token latency (also called prefill), and a 2x speedup in latency in the following autoregressive decoding phase. 

![text-generation-inference results on Meta Llama3 70B mi300 vs mi250](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-amd-mi300/tgi_mi300_vs_mi250.png)

_TGI latency results for Meta Llama 3 70B, comparing AMD Instinct MI300X on an Azure VM against the previous generation AMD Instinct MI250_

### Model fine-tuning performances

Hugging Face libraries can as well be used to fine-tune models. 
We use Transformers and [PEFT](https://github.com/huggingface/peft) libraries to finetune Llama 3 70B using low rank adapters (LoRA. To handle the parallelism over several devices, we leverage [DeepSpeed Zero3](https://deepspeed.readthedocs.io/en/latest/zero3.html) through [Accelerate library](https://huggingface.co/docs/accelerate/usage_guides/deepspeed).

On Llama 3 70B, our workload consists of batches of 448 tokens, with a batch size of 2. Using low rank adapters, the model’s original 70,570,090,496 parameters are frozen, and we instead train an additional subset of 16,384,000 parameters thanks to [low rank adapters](https://arxiv.org/abs/2106.09685).

From our comparison on Llama 3 70B, we are able to train about 2x times faster on an Azure VM powered by MI300X, compared to an HPC server using the previous generation AMD Instinct MI250.

![PEFT finetuning on mi300 vs mi250](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-amd-mi300/peft_finetuning_mi300_vs_mi250.png)

_Moreover, as the MI300X benefits from its 192 GB HBM3 memory (compared to 128 GB for MI250), we manage to fully load and fine-tune Meta Llama 3 70B on a single device, while an MI250 GPU would not be able to fit in full the ~140 GB model on a single device, in float16 nor bfloat16._
_Because it’s always important to be able to replicate and challenge a benchmark, we are releasing a [companion Github repository](https://github.com/huggingface/hf-rocm-benchmark) containing all the artifacts and source code we used to collect performance showcased in this blog._


## What's next?

We have a lot of exciting features in the pipe for these new AMD Instinct MI300 GPUs. 
One of the major areas we will be investing a lot of efforts in the coming weeks is minifloat (i.e. float8 and lower). 
These data layouts have the inherent advantages of compressing the information in a non-uniform way alleviating some of the issues faced with integers. 

In scenarios like inferencing on LLMs this would divide by two the size of the key-value cache usually used in LLM. 
Later on, combining float8 stored key-value cache with float8/float8 matrix-multiplications, it would bring additional performance benefits along with reduced memory footprints.

## Conclusion

As you can see, AMD MI300 brings a significant boost of performance on AI use-cases covering end-to-end use cases from training to inference. 
We, at Hugging Face, are very excited to see what the community and enterprises will be able to achieve with these new hardware and integrations. 
We are eager to hear from you and help in your use-cases.

Make sure to stop by optimum-AMD and text-generation-inference Github repositories to get the latest performance optimization towards AMD GPUs!
