---
title: "FSDP2: Identifying and fixing a core memory increase" 
thumbnail: /blog/assets/184_zero_shot_docmatix/thumb.001.jpeg
authors:
- user: muellerzr
- user: siro1 

---

# Solving a memory leak within FSDP2 and making it fully compatible with FSDP1's API

## Introduction

FSDP2 is the new version of Fully Sharded Data Parallelism by PyTorch, with the core examples being showcased inside of the [torchtitan](https://github.com/pytorch/torchtitan/)
 repository. In it, we quickly noticed a new trend:

Rather than creating the model and optimizer, then sharding the model, and moving forward, 
PyTorch was doing something new: create the model, **then shard the model**, **then create an optimizer**. 

This subtle change means that there would be drastic complications when it came to the 🤗 Accelerate `accelerator.prepare()` 
API if we could not solve this.

## What's the problem?

🤗 Accelerate operates on a "bring your own model/optimizer/scheduler" API, and then we make the changes for you. 
This means that the API must be able to take any arbitrary PyTorch-compatible component, and we do all the heavy 
lifting when it comes to getting distributed training (such as FSDP1/2) as low-code as possible. 
Furthermore, this is then upstreamed to the 🤗 Trainer/🤗 TRL/Axolotl so it's *critical* that these core API's do not change. 

This new change proposed by PyTorch *broke* this requirement as we need to do early changes to the model before creating an optimizer,
 removing this "bring your own" capability that is 🤗 Accelerate's strongsuit. 

But how much did it "break"?

Did the model training just *fail*?

Not quite. 

## Pre-Sharding vs Post-Sharding

As a reminder, 🤗 Accelerate uses an idea of post-sharding of the weights (create model, create optimizer, then shard the model) 
rather than *pre-sharding* of the weights like FSDP2 assumes (create the model, shard the model, create optimizer with sharded model). 

Since the only real difference here is the shards, training should (and does) work as normal:

(Matej insert graph of the trainings with and without our fix)


Instead, what we found is there was a large memory jump if we didn't pre-shard the model beforehand:

(Matej: insert graphs here of memory)

## Fixing the Problem

{Matej, fill this section out with what you did}

## What now?

As of the next 🤗 Accelerate release (~first week of April, 2025) we will include support for PyTorch's FSDP2 as well as this hotfix for 
maintaining a same user-facing API within Accelerate when using FSDP2, without this (easy to occur) memory leak. For a more in-depth analysis, 
please check out our benchmarking scripts [here](matej add this in as a PR to accelerate in the `benchmarks/` folder) and the attached visualizations

