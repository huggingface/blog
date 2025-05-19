---
title: "Fine-Tune CSM For Conversational Speech LLMs with 🤗 Transformers" 
thumbnail: /blog/assets/112_fine_tune_csm/thumbnail.jpg
authors:
- user: eustlb
- user: reach-vb
- user: Steveeeeeeen
---

# Fine-Tune CSM For Conversational Speech LLMs with 🤗 Transformers

# To be updated
<a target="_blank" href="">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


In this blog, we present a step-by-step guide on fine-tuning CSM 
for any conversational speech LLM dataset using Hugging Face 🤗 Transformers. This blog 
provides in-depth explanations of the CSM model, the  dataset and 
the theory behind fine-tuning, with accompanying code cells to execute the data 
preparation and fine-tuning steps. For a more streamlined version of the notebook 
with fewer explanations but all the code, see the accompanying [Google Colab]().


## Table of Contents

1. [Introduction](#introduction)
2. [Fine-tuning CSM in a Google Colab](#fine-tuning-csm-in-a-google-colab)
    1. [Prepare Environment](#prepare-environment)
    2. [Load Dataset](#load-dataset)
    3. [Prepare Feature Extractor, Tokenizer and Data](#prepare-feature-extractor-tokenizer-and-data)
    4. [Training and Evaluation](#training-and-evaluation)
    5. [Building a Demo](#building-a-demo)
3. [Closing Remarks](#closing-remarks)

## Introduction

