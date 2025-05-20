---
title: "🐯 Liger GRPO meets TRL" 
thumbnail: /blog/assets/liger-grpo/thumbnail.png
authors:
- user: shisahni
  guest: true
  org: LinkedIn
- user: kashif
- user: smohammadi
  guest: true
  org: axolotl-ai-co
---

# 🐯 Liger GRPO meets TRL

TL; DR
[Liger](https://github.com/linkedin/Liger-Kernel) supercharges [TRL](https://github.com/huggingface/trl)’s [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) by slashing memory usage by **40%** with zero drop in model quality. We also added support for **FSDP** and **PEFT**, making it easier than ever to scale GRPO across multiple GPUs. 

## Motivation

Fine-tuning language models using reinforcement learning (RL) is a crucial step in a model's training lifecycle for steering models towards desirable behaviours which are more complex than can be achieved through typical supervised fine-tuning. RL has traditionally been applied to optimize large language models (LLMs) using the Proximal Policy Optimization (PPO) algorithm. This approach, often associated with Reinforcement Learning from Human Feedback (RLHF), utilizes a separately trained reward model to guide the fine-tuning of the primary model. 

However, RLHF with PPO is a very resource-hungry approach - PPO requires loading multiple models in memory (policy, value, reward, and reference models), and also requires several iterations of fine-tuning reward and base models to achieve the desired results. The success of RLHF also depends on the capability of the reward model to effectively discriminate between desired and un-desired behaviour from our model.

Group Relative Policy Optimization (GRPO) has seen significant recent popularity alongside DeepSeek's R1 model. GRPO eschews the pre-trained reward model and value models used in RLHF and instead relies on *verifiable reward functions* which can check the correctness of a model's output in a closed-form manner without needing an external reward model. This has resulted in massive improvements when using GRPO instead of PPO for fine-tuning on domains which are easily verifiable, such as teaching a model to reason, and perform well on math and coding tasks. 

The following diagram shows GRPO vs PPO training pipeline (ref: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300)):

![PPO-vs-GRPO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/liger-grpo/image5.png)

That said, RL training still eats up a ton of GPU memory, so there's still plenty of room for optimizations here. In this blog post, we talk about an optimization that we recently added to TRL that cuts peak memory usage by 40% during GRPO Training, and we also dive into how to scale GRPO to multiple GPUs and nodes without losing performance or correctness.

## How Liger Kernel slashes memory for GRPO

We extended the Liger Chunked Loss approach to GRPO Loss, which lets us avoid having to store the full logits in memory for every training step. The calculation of logits, which involves the model's output head, is a significant contributor to peak memory usage, especially when dealing with large vocabularies, long sequence lengths, or large batch sizes. We address this by chunking the input to the `lm_head` across the batch and running the forward pass one chunk at a time.

But if you just implement it in a straightforward way, you won't actually be able to reduce the peak memory since you'd still need to keep all the logits in GPU memory for the backward pass. To get around that, we calculate the gradients for each loss chunk (with respect to the `input` chunk and the `lm_head` weight`) during the forward pass, and then accumulate them as we go through each chunk.

Here's the visualization of the optimization (ref: [Byron Hsu](https://x.com/hsu_byron/status/1866577403918917655)):

![liger-chunked-loss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/liger-grpo/image7.gif)

## Plug-and-Play integration with TRL

We recently integrated Liger GRPO with TRL in PR [#3184](https://github.com/huggingface/trl/pull/3184), so now you can use the Liger GRPO loss just by setting `use_liger_loss` to `True` in your `GRPOConfig` and enjoy the memory savings!

Heads up: these features aren't in the latest TRL release yet, so you'll need to install TRL from source for now:

```bash
pip install "trl[liger] @ git+https://github.com/huggingface/trl.git"
```

and then you can use it like this:
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

Here's the plots of peak memory usage vs batch size for both FP32 and BF16 training. As expected, the memory savings get better with larger batch sizes since we chunk along the batch dimension. So when the batch size goes up, the Liger chunked loss ends up using a lot less memory, up to 40% less, compared to the regular (non-liger) version. 

Quick note: Right now, we only support FP32, but we're working on open-sourcing BF16 support for Liger GRPO in TRL. The BF16 results shown here are from internal patches we've been testing.

![Mem-vs-batch-size-fp32](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/liger-grpo/image3.png)

![Mem-vs-batch-size-bf16](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/liger-grpo/image4.png)


We also show that Liger Loss is effectively accurate. As seen in the plot, rewards over training steps stay pretty much the same as what you'd see using the standard TRL implementation.

![reward-vs-step](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/liger-grpo/image1.png)


## Scaling further with FSDP and PEFT

We also added FSDP and PEFT support to Liger GRPO Loss in PR [#3260](https://github.com/huggingface/trl/pull/3260) and PR [#3355](https://github.com/huggingface/trl/pull/3355), respectively, allowing users to easily scale their experiments across multiple GPUs or nodes. PEFT techniques such as LoRA and QLoRA reduce the number of trainable parameters by only tuning the weights of smaller adapter weights on top of the original model, significantly lowering memory pressure as gradients, activations, and optimizer states for the entire model don't need to be held in memory. Additionally, using PEFT in GRPO allows one to forgo loading a separate reference model during training, as we can obtain the original, unmodified model during training by simply disabling the LoRA adapters. 

Here, we show a multi-GPU GRPO training plot using FSDP and PEFT, where we compare the maximum training batch size possible with and without the Liger Loss across different Qwen3 model sizes. We found that with Liger, we were able to bump up the batch size by around **1.5 to 1.8x**!

![peft-batch-size-vs-model-size](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/liger-grpo/image6.png)

## Conclusion

With the integration of Liger-GRPO into TRL, alongside FSDP and PEFT support, fine-tuning language models with GRPO is now more memory-efficient and scalable than ever. We encourage the community to try out these new features and share their feedback to help us further improve RL training for LLMs.

