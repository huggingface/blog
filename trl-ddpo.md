---
title: "TRL: Denoising Diffusion Policy Optimization" 
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
authors:
- user: metric-space
---

# Finetune Stable Diffusion Models with DDPO

<!-- {blog_metadata} -->
<!-- {authors} -->

# Introduction

Reinforcement learning is a powerful ML model finetuning tool and now this tool's coverage extends to the valley of diffusion models.

The paper: [Training Diffusion Models with Reinforcement Learning by by Kevin Black, Michael Janner, Yilan Du, Ilya Kostrikov, Sergey Levine](https://arxiv.org/abs/2305.13301) and the associated code now tells us and shows us how to augment Stable diffusion model code to use RL to finetune said model in accordance with an appropriate reward function.

This blog-post introduces Direct Preference Optimization (DDPO) method which is now available in the TRL library to fine tune stable diffusion models.


# Motivation 

# Theoretical bits

Though this can be skipped, understanding this will set model results expectations appropriately. 

-- theoretical bits

-- getting started

-- colab stuff

-- notes

# Current limitations of DDPO integration into TRL

# Getting started with the example script

The example script is a commandline script.



# LoRA vs Non-lora

By default

Non-lora is somewhat tricky to get right, past configurations of 



# Observations

The observations: 
1. For the aesthetic scorer, it does seem like having prompts in the somewhat f the same class makes for faster convergence of the model. 
   For example, inserting something like `ocean` leads to relatively disastrous results

2. While non-lora is tricky, the sampling process generates a lot more intricate pictures atleast relatively more 











