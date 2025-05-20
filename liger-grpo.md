---
title: "🐯 Liger GRPO meets TRL" 
thumbnail: /blog/assets/liger-grpo/thumbnail.png
authors:
- user: shisahni
  guest: true
- user: kashif
- user: smohammadi
  guest: true

---

# 🐯 Liger GRPO meets TRL

TL; DR
[Liger](https://github.com/linkedin/Liger-Kernel) supercharged TRL’s GRPO Trainer by slashing memory usage by **40%** with zero drop in model quality. We also added support for *FSDP* and *PEFT*, making it easier than ever to scale GRPO across multiple GPUs. 

## Introduction & Motivation


RLHF (Reinforcement Learning from Human Feedback) has been an effective way to get models to behave the way we want them to in real-world scenarios. Traditionally, though, doing RLHF has meant juggling a bunch of different models, i.e., an actor, critic, reward model, and reference model, all working together to “teach” the main model how to act. But this setup comes with the overhead of building and maintaining a pretty complex infrastructure to keep everything running smoothly and make sure all the data flows between them properly.

More recently, DeepSeek’s R1 helped popularize GRPO, which simplifies things quite a bit without losing performance. GRPO gets rid of the critic model and instead uses the average reward of sampled outputs produced in response to the same prompt as the baseline. The following diagram shows GRPO vs PPO training pipeline (ref: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300)):

That said, RL training still eats up a ton of GPU memory, so there’s still plenty of room for optimizations here. In this blog post, we talk about an optimization that we recently added to TRL that cuts peak memory usage by 30% during GRPO Training, and we also dive into how to scale GRPO to multiple GPUs and nodes without losing performance or correctness.

## How Liger Kernel slashes memory for GRPO


We extended the Liger Chunked Loss approach to GRPO Loss, which lets us avoid having to store the full logits in memory for every training step. We do this by chunking the input to the `lm_head` across the batch and running the forward pass one chunk at a time.

But if you just implement it in a straightforward way, you won’t actually be able to reduce the peak memory since you’d still need to keep all the logits in GPU memory for the backward pass. To get around that, we calculate the gradients for each loss chunk (with respect to the `input chunk` and the `lm_head weights`) during the forward pass, and then accumulate them as we go through each chunk.

Here’s the visualization of the optimization:

## Plug-and-Play integration with TRL

We recently integrated Liger GRPO with TRL in PR [#3184](https://github.com/huggingface/trl/pull/3184), so now you can use the Liger GRPO loss just by setting use_liger_loss to True in your GRPOConfig and enjoy the memory savings! 

Heads up: these features aren’t in the latest TRL release yet, so you’ll need to install TRL from source for now.

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(output_dir="Qwen3-0.6B-GRPO", use_liger_loss=True)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B-Instruct",
    reward_func=reward_len,
    args=training_args,
    train_dataset=train_dataset,
)
```

## Benchmarks

We ran a bunch of GRPO experiments with and without the Liger GRPO Loss to see how things compare. For the policy model, we used `Qwen3-0.6B` and played around with different batch sizes. All the experiments were run on the `gsm8k` dataset using its reward functions.

Here’s the plots of peak memory usage vs batch size for both FP32 and BF16 training. As expected, the memory savings get better with larger batch sizes since we chunk along the batch dimension. So when the batch size goes up, the Liger chunked loss ends up using a lot less memory, up to 40% less, compared to the regular (non-liger) version. 

Quick note: Right now, we only support FP32, but we're working on open-sourcing BF16 support for Liger GRPO in TRL. The BF16 results shown here are from internal patches we’ve been testing.

We also show that Liger Loss is effectively accurate. As seen in the plot, rewards over training steps stay pretty much the same as what you’d see using the standard TRL implementation.


## Scaling further with FSDP and PEFT


We also added FSDP and PEFT support to Liger GRPO Loss in PR #3260 and PR #3355, respectively, allowing users to easily scale their experiments across multiple GPUs or nodes. PEFT techniques such as LoRA and QLoRA reduce the number of trainable parameters by only tuning the weights of smaller adapter weights on top of the original model, significantly lowering memory pressure as gradients, activations, and optimizer states for the entire model don’t need to be held in memory. Additionally, using PEFT in GRPO allows one to forgo loading a separate reference model during training, as we can obtain the original, unmodified model during training by simply disabling the LoRA adapters. 

Here, we show a multi-GPU GRPO training plot using FSDP and PEFT, where we compare the maximum training batch size possible with and without the Liger Loss across different Qwen3 model sizes. We found that with Liger, we were able to bump up the batch size by around 1.5 to 1.8x!

