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
- user: ShirinYamani
- user: m0m0chen
  guest: true
  org: LinkedIn
- user: liberty4321
  guest: true
  org: LinkedIn
---

# 🐯 Liger GRPO meets TRL

TL; DR
[Liger](https://github.com/linkedin/Liger-Kernel) supercharges [TRL](https://github.com/huggingface/trl)’s Group Relative Policy Optimization [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) by slashing memory usage by **40%** with zero drop in model quality. We also added support for **FSDP** and **PEFT**, making it easier than ever to scale GRPO across multiple GPUs.

## Motivation

Fine-tuning language models using reinforcement learning (RL) is a crucial step in a model's training lifecycle for steering models towards desirable behaviours which are more complex than can be achieved through typical supervised fine-tuning. RL has traditionally been applied to optimize large language models (LLMs) using the Proximal Policy Optimization (PPO) algorithm. This approach, often associated with Reinforcement Learning from Human Feedback (RLHF), utilizes a separately trained reward model to guide the fine-tuning of the primary model. 

However, RLHF with PPO is a very resource-hungry approach - PPO requires loading multiple models in memory (policy, value, reward, and reference models), and also requires several iterations of fine-tuning reward and base models to achieve the desired results. The success of RLHF also depends on the capability of the reward model to effectively discriminate between desired and un-desired behaviour from our model.

Group Relative Policy Optimization (GRPO) has seen significant recent popularity alongside DeepSeek's R1 model. GRPO eschews the pre-trained reward model and value models used in RLHF and instead relies on *verifiable reward functions* which can check the correctness of a model's output in a closed-form manner without needing an external reward model. This has resulted in massive improvements when using GRPO instead of PPO for fine-tuning on domains which are easily verifiable, such as teaching a model to reason, and perform well on math and coding tasks. 

The following diagram shows the GRPO vs PPO training pipeline (ref: Figure 4 of [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300)):

![PPO-vs-GRPO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/liger-grpo/image5.png)

That said, RL training still eats up a ton of GPU memory, so there's still plenty of room for optimizations here. In this blog post, we talk about an optimization that we recently added to TRL that cuts peak memory usage by 40% during GRPO Training, and we also dive into how to scale GRPO to multiple GPUs and nodes without losing performance or correctness.

## How Liger Kernel slashes memory for GRPO

We extended the Liger Chunked Loss approach to the GRPO Loss, which lets us avoid having to store the full logits in memory for every training step. The calculation of logits, which involves the model's output head, is a significant contributor to peak memory usage, especially when dealing with large vocabularies, long sequence lengths, or large batch sizes. We address this by chunking the input to the `lm_head` across the batch and running the forward pass one chunk at a time.

But if you just implement it in a straightforward way, you won't actually be able to reduce the peak memory since you'd still need to keep all the logits in GPU memory for the backward pass. To get around that, we calculate the gradients for each loss chunk (with respect to the `input` chunk and the `lm_head` weight) during the forward pass, and then accumulate them as we go through each chunk.

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
from datasets import load_dataset


train_dataset = load_dataset("trl-lib/tldr", split="train")
training_args = GRPOConfig(output_dir="Qwen3-0.6B-GRPO", use_liger_loss=True)

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
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

We also added FSDP and [PEFT](https://github.com/huggingface/peft) support to Liger GRPO Loss in PR [#3260](https://github.com/huggingface/trl/pull/3260) and PR [#3355](https://github.com/huggingface/trl/pull/3355), respectively, allowing users to easily scale their experiments across multiple GPUs or nodes. PEFT techniques such as LoRA and QLoRA reduce the number of trainable parameters by only tuning the weights of smaller adapter weights on top of the original model, significantly lowering memory pressure as gradients, activations, and optimizer states for the entire model don't need to be held in memory. Additionally, using PEFT in GRPO allows one to forgo loading a separate reference model during training, as we can obtain the original, unmodified model during training by simply disabling the LoRA adapters. 

Here, we show a multi-GPU GRPO training plot using FSDP and PEFT, where we compare the maximum training batch size possible with and without the Liger Loss across different Qwen3 model sizes. We found that with Liger, we were able to bump up the batch size by around **1.5 to 1.8x**!

![peft-batch-size-vs-model-size](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/liger-grpo/image6.png)

## Scaling even further with vLLM

To accelerate text generation during training, Liger Loss can be effectively combined with TRL's integrated [vLLM](https://vllm.ai/) server. This significantly speeds up the collection of rollout data with minimal overhead and offers a seamless integration experience.

Here's how to set it up:

1.  **Start the vLLM Server:**
    First, launch the vLLM server. This server will handle the generation requests from your training script. Open a terminal and run:
    ```bash
    CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model "Qwen/Qwen3-0.6B"
    ```
    *Note: We assign `CUDA_VISIBLE_DEVICES=1` to run the vLLM server on a specific GPU (GPU 1 in this case), leaving other GPUs free for training.*

2.  **Configure and Run Your Training Script:**
    Next, modify your training script to use the vLLM server. The key change is setting `use_vllm=True` in your `GRPOConfig`.

    ```python
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset


    def reward_len(completions, **kwargs):
        return [-abs(20 - len(completion)) for completion in completions]

    dataset = load_dataset("trl-lib/tldr", split="train[:1%]")
    training_args = GRPOConfig(
        output_dir="Qwen3-0.6B-GRPO", 
        use_liger_loss=True, 
        use_vllm=True, # Enable vLLM integration
        logging_steps=10
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen3-0.6B", # Ensure this matches the model served by vLLM
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    ```

3.  **Launch the Training:**
    Finally, run your training script using `accelerate launch` (or `python` if not using [Accelerate](https://github.com/huggingface/accelerate) for multi-GPU/distributed training). Make sure to target a different GPU for training if your vLLM server is occupying one.
    ```bash
    CUDA_VISIBLE_DEVICES=0 accelerate launch train.py 
    ```
    *(Assuming your script is named `train.py` and you want to run training on GPU 0)*.

By following these steps, you can leverage vLLM for faster generation turnarounds during your GRPO training with Liger Loss.

## Conclusion

With the integration of Liger-GRPO into TRL, alongside FSDP and PEFT support, fine-tuning language models with GRPO is now more memory-efficient and scalable than ever. We encourage the community to try out these new features and share their feedback to help us further improve RL training for LLMs.
