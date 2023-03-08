---
title: "Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU" 
thumbnail: /blog/assets/133_trl_peft/thumbnail.png
authors:
- edbeeching
---

<h1>
	<h5><i> Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU </i></h5>
</h1>

We are excited to officially release the integration of `trl` with `peft` to make LLMs fine-tuning with Reinforcement Learning more accessible to anyone! Let us explain why this is a very competitive alternative to existing fine-tuning approaches in this article. 

Note `peft` is a general tool that can be applied to many ML use-cases but it’s particularly interesting for RLHF as this method is especially memory-hungry!

## Introduction

### LLMs & RLHF

Large Language Models combined with RLHF (Reinforcement Learning with Human Feedback) seems to be the next go-to approach for building very powerful AI systems such as ChatGPT.

Training a language model with RLHF typically involves the following three steps:

1- Fine-tune a pretrained LLM on a specific domain or corpus of instructions and human demonstrations 

2- Collect a human annotated dataset and train a reward model

3- Fine-tune the LLM with the reward model and this dataset using RL (e.g. PPO)

The choice of the LLM is quite crucial here, it is known that Chat-GPT uses an instruction-finetuned model (instruct-GPT3) as a base model before applying the RL step. At this time of writing, the “best” open-source instruction finetuned LLMs are, `bloomz` , `flan-t5` , `flan-ul2` `opt-iml` `llama-i` (Upon non commercial license). The downside of these models is their model size. To get a decent model, you need at least to play with 10B+ scale models which would require up to 40GB GPU memory in full precision, just to fit the model on a single GPU device without doing any training at all!

### What is `trl`?

The `trl` library aims at making the RL step much easier and more flexible so that anyone can fine-tune their Language Model using RL on their custom dataset and training setup. Among many other application, you can use this algorithm to fine-tune a model to generate [positive movie reviews](https://huggingface.co/docs/trl/sentiment_tuning), do [controlled generation](https://github.com/lvwerra/trl/blob/main/examples/sentiment/notebooks/gpt2-sentiment-control.ipynb) or [making the model less toxic](https://huggingface.co/docs/trl/detoxifying_a_lm). 

Using `trl` you can run PPO algorithm, one popular RL method, in a distributed manner or on a single device! We leverage `accelerate` from the Hugging Face ecosystem to make this possible, so that any user can scale up the experiments up to an interesting scale.

Fine-tuning a language model with RL follows roughly the protocol detailed below. This requires having 2 copies of the original model; to avoid the active model to deviate too much from its original behavior / distribution you need to compute the logits of the reference model at each optimization step. This adds a hard constraint on the optimization process as you need always at least two copies of the model per GPU device. If the model grows in size, it becomes more and more tricky to fit the setup on a single GPU.

| ![ppo_image](https://huggingface.co/datasets/trl-internal-testing/example-images/blob/main/images/trl_overview.png) |
|:--:|
| <b>Overview of the PPO training setup in TRL</b>|

In trl you can use shared layers between reference and active models to avoid entire copies. A concrete example of this feature is showcased in the detoxification example.

### Training at scale

Training at scale can be challenging. The first challenge is fitingt the model and its optimizer states on the available GPU devices. The amount of GPU memory a single parameter takes in the memory depends on its “precision” (or more specifically `dtype`). The most common `dtype` being `float32` (32-bit), `float16`, and `bfloat16` (16-bit). More recently “exotic” precisions are supported out-of-the-box for training and inference (with certain conditions and constraints) such as `int8` (8-bit). In a nutshell, to load a model on a GPU device each billion parameter costs 4GB in float32 precision, 2GB in float16 and 1GB in int8. If you would like to learn more about this topic, have a look at [this blogpost](https://huggingface.co/blog/hf-bitsandbytes-integration) which dives deeper.

If you use an Adam optimizer (which is the most popular optimizer as we are writing this blogpost), each parameter needs 3 times its allocated memory (e.g. if your model needs 1GB GPU memory, the full Adam optimizer of the model would require 3GB GPU memory).

Many techniques have been adopted to tackle these challenges at scale. Most familiar paradigms being Pipeline Parallelism, Tensor Parallelism and Data Parallelism.


| ![model-parallelism](/blog/assets/133_trl_peft/model-parallelism.png) |
|:--:|
| <b>Image Credits to <a href="https://towardsdatascience.com/distributed-parallel-training-data-parallelism-and-model-parallelism-ec2d234e3214 " rel="noopener" target="_blank" >this blogpost</a> </b>|

With data parallelism the same model is hosted in parallel on several machines and each instance is fed a different data batch. This is the most straight forward parallelism strategy essentially replicating the single-GPU case and is already supported by `trl`. With Pipeline and Tensor Parallelism the model itself if distributed across machines: in Pipeline Parallelism this model is split layer wise whereas Tensor Parallelism splits tensor operations across GPUs (e.g. matrix multiplications). With these Model Parallelism strategies you need to shard the model weights across many devices which requires you to define a communication protocol of the activations and gradients across process. This is not trivial to implement and might need the adoption of some frameworks such as `[DeepSpeed](https://github.com/microsoft/DeepSpeed)` or `[Nemo](https://github.com/NVIDIA/NeMo)` . Further reading about parallelism paradigms can be found [here](https://huggingface.co/docs/transformers/v4.17.0/en/parallelism).

Therefore, we asked ourselves the following question: how far can we go with the just data parallelism? Can we use existing tools to fit super-large training processes (including active model, reference model and optimizer states) in a single device? The answers appears to be yes. The main ingredients being: adapters and 8bit matrix multiplication! Let us cover these topics in the next sections:

### 8-bit matrix multiplication

Efficient 8-bit matrix multiplication is a method that has been first introduced in the paper [LLM.int8()](https://arxiv.org/abs/2208.07339) and aims to solve the performance degradation issue when quantizing large-scale models. The proposed method  breaks down the matrix multiplications that are applied under the hood in Linear layers in two stages: the outlier hidden states part that is going to be performed in float16 & the “non-outlier” part that is performed in int8. 

| ![8bit-matmul](/blog/assets/133_trl_peft/8bit-matmul.png) |
|:--:|
| <b>Efficient 8-bit matrix multiplication is a method that has been first introduced in the paper [LLM.int8()](https://arxiv.org/abs/2208.07339) and aims to solve the performance degradation issue when quantizing large-scale models. The proposed method  breaks down the matrix multiplications that are applied under the hood in Linear layers in two stages: the outlier hidden states part that is going to be performed in float16 & the “non-outlier” part that is performed in int8.</b>|

In a nutshell, you can reduce the size of a full-precision model by 4 (thus, by 2 for half-precision models) if you use 8-bit matrix multiplication. 

### Low rank adaptation and peft

TODO

### What is `peft` ?

TODO

## Summary 

Let us go through the entire pipeline step by step, and explain with figures how we managed to fine-tune 20B parameter LLM with RL using the tools mentioned above!

### Step 1: Load your active model in 8-bit precision

| ![step1](/blog/assets/133_trl_peft/step1.png) |
|:--:|
| <b> Loading a model in 8-bit precision can save up to 4x memory compared to full precision model</b>|

A “free-lunch” memory reduction of a LLM using `transformers` is to load your model in 8-bit precision using the method described in LLM.int8. This can be performed by simply adding the flag `load_in_8bit=True` when calling the `from_pretrained` method (you can read more about that [here](https://huggingface.co/docs/transformers/main/en/main_classes/quantization))

As stated in the previous section, a “hack” to compute the amount of GPU memory you should need to load your model is to think in terms of “billions of parameters”. As one byte needs 8 bits, you need 4GB per billion parameter for a full-precision model (32bit = 4bytes), 2GB per billion parameter for a half-precision model and 1GB per billion parameter for an int8 model.

So in the first place, let’s just load the active model in 8-bit. Let’s see what we need to do for the second step!

### Step 2: Add extra trainable adapters using `peft`

| ![step2](/blog/assets/133_trl_peft/step2.png) |
|:--:|
| <b> You easily add adapters on a frozen 8-bit model thus reduce the memory requirements of the optimizer states, byt training a small fraction of parameters</b>|

The second step is to load adapters inside the model and make these adapters trainable. This enables a drastic reduction of the amount of trainable weights that are needed for the active model. This step leverages peft library and can be performed with few lines of code. Note that once the adapters are trained, you can easily push them to the Hub to use them later.

### Step 3: Use the same model to get the reference and active logits

| ![step3](/blog/assets/133_trl_peft/step3.png) |
|:--:|
| <b> You can easily disable and enable adapters using the `peft` API.</b>|


As adapters can be deactivated, we can use the same model to get the reference and active logits for PPO, without having to create two copies of the same model! This leverages a feature in `peft` library, which is the disable_adapters context manager. 


## Fine-tuning GPT-neox-20B

TODO

## Conclusion

TODO

## References

- parallelism paradigms: [https://huggingface.co/docs/transformers/v4.17.0/en/parallelism](https://huggingface.co/docs/transformers/v4.17.0/en/parallelism)
- 8-bit integration in `transformers`: [https://huggingface.co/blog/hf-bitsandbytes-integration](https://huggingface.co/blog/hf-bitsandbytes-integration)
- LLM.int8 paper: [https://arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339)
- Gradient checkpoiting explained: [https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-activation-checkpointing.html](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-activation-checkpointing.html)