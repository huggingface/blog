# On the emergence of instruction-finetuned language models

## Table of contents:

### Introduction 
### How to use these models in transformers?
### What is next? 

## Popular models:

Flan-t5: https://huggingface.co/google/flan-t5-xxl
T0: https://huggingface.co/bigscience/T0pp
BLOOMZ: https://huggingface.co/bigscience/bloomz 
mT0: https://huggingface.co/bigscience/mt0-xxl

(from flan-t5’s abstract):

Finetuning language models on a collection of datasets phrased as instructions has been shown to improve model performance and generalization to unseen tasks. In this paper we explore instruction finetuning with a particular focus on (1) scaling the number of tasks, (2) scaling the model size, and (3) finetuning on chain-of-thought data.

## Introduction:


How to use instruction-fine tuned models using transformers?


- Just like any transformer model via HF
- Note on large models using accelerate and 8-bit
- Prompting best practices:
+ Detailed instructions
+ Detailing the desired answer (e.g. language w/ BLOOMZ)
+ Short vs long answers (e.g. forcing longer answers in generate via min_new_tokens); Bias for short answers of models like mT0

## What is next?

- RLHF - complete tutorial on how to fine-tune flan-T5 small with trl?
- Allowing models to take actions based on instructions (e.g. ACT-1)
- Instructions for embeddings (Instructor; MTEB)
