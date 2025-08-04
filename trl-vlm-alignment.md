---
title: "Vision Language Model Alignment in TRL ⚡️" 
thumbnail: assets/trl_vlm/thumbnail.png
authors:
- user: qgallouedec
- user: kashif
- user: sergiopaniego
- user: merve
- user: ariG23498
---

# Vision Language Model Alignment in TRL ⚡️

## Introduction

Vision Language Models (VLMs) are getting stronger, but *aligning* them to human preferences still matters. In TRL, we already showed how to post-train VLMs with [**Supervised Fine-Tuning (SFT)**](https://huggingface.co/docs/trl/main/en/training_vlm_sft) and [**Direct Preference Optimization (DPO)**](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct). This time, we’re going further.

**tl;dr** We have added two new multimodal alignment methods to TRL: **Group Relative Policy Optimization (GRPO)**, its variant **Group Sequence Policy Optimization (GSPO)**, and **Mixed Preference Optimization (MPO)**. All of them let you go beyond pairwise DPO, extracting more signal from preference data and scaling better with modern VLMs. We release training scripts and demo notebooks to easily get started with them!

## Table of Contents

- [Multimodal Alignment for VLMs in TRL ⚡️](#multimodal-alignment-for-vlms-in-trl-️)
  - [Introduction](#introduction)
  - [Alignment for Vision Language Models](#alignment-for-vision-language-models)
    - [Mixed Preference Optimization (MPO)](#mixed-preference-optimization-mpo)
    - [Multimodal Group Relative Policy Optimization (GRPO)](#multimodal-group-relative-policy-optimization-grpo)
    - [Group Sequence Policy Optimization (GSPO)](#group-sequence-policy-optimization-gspo)
  - [vLLM Integration in TRL](#vllm-integration-in-trl)
  - [Useful Resources](#useful-resources)

## Alignment for Vision Language Models

Traditionally, you would take a base model, apply SFT to follow instructions, and then apply DPO to align it to preferential data. Previously, [we adapted this approach to Vision Language Models (VLMs)](https://huggingface.co/blog/dpo_vlm) and validated it on IDEFICS2,  showing improvement in model responses. 

DPO works by optimizing preferences between pairs of model responses using a contrastive loss: you have a chosen and a rejected answer and you optimize your preferences based on what you want and don’t want. 

But in the last year, new multimodal alignment methods have gained popularity, GRPO and MPO, that can push VLM performance even further. At the end of the blog post you can find a table that showcases the differences between model responses.

## Mixed Preference Optimization (MPO)

Aligning multimodal models with SFT to do reasoning tasks fall short due to distribution shift. Meanwhile, models aligned with DPO fail to generate coherent rationales and might generate repetitive responses. To address this, there’s a new technique called [Mixed Preference Optimization](https://huggingface.co/papers/2411.10442) (MPO) specifically made for multimodal models. This method is essentially an extension of DPO with multiple losses: preference loss from DPO (sigmoid), quality loss from Binary Classifier Optimization (BCO), and generation loss from SFT. According to the [paper](https://huggingface.co/papers/2411.10442), simply switching to this combined loss results in 6.2 pts improvement in MathVista! 

Since this is only modifying the loss, we added combined loss support to TRL's `DPOTrainer` class. To use it, you can initialize the `DPOConfig` as follows:

```python
mpo_config = DPOConfig(
    output_dir=tmp_dir,
    per_device_train_batch_size=2,
    learning_rate=9e-1,
    loss_type=["sigmoid", "bco_pair", "sft"], # Loss types to combine, as used in the MPO paper
    loss_weights=[0.8, 0.2, 1.0], # Corresponding weights, as used in the MPO paper
    report_to="none",
    bf16=False,
    fp16=False,
    use_cpu=True,
    max_steps=1,
)
```

Then initialize the `DPOTrainer`: 

```python
mpo_trainer = DPOTrainer(
    model=model_id,
    args=mpo_config,
    processing_class=tokenizer,
    train_dataset=dataset,
)
mpo_trainer.train()
```

And that’s it! 

### Multimodal Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO) is a cutting-edge alignment method initially introduced in [DeepSeek Math](https://huggingface.co/papers/2402.03300) paper and later integrated to DeepSeek R1, the groundbreaking LLM. It’s an addition to PPO where the policy updates are done over groups (batches of trajectories that represent how a dialogue rolls out). This feature makes it more robust to reward noise, as the noise averages out within groups. Since the model learns broader sense of a good response rather than singular high reward samples, this method also makes the model highly performant.

![image.png](attachment:ef6696d7-064a-4bd0-b3b1-60806326101e:image.png)

In TRL, we now introduce GRPO support for vision language models. We will not provide a full training script example, as you can find it in the notebook. Instead, we'll focus on highlighting the key component and concepts.

To make the training script work effectively, we need to validate that the format of the answer is correct and that the solution itself is close to the completed parts, so we write two reward functions. In order to really see improvements in the latter reward, you would need a rather maximalist setup, where you have relatively larger models, a lot of generations, and a high-quality, diverse dataset.

```python
import re
from math_verify import LatexExtractionConfig, parse, verify

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    matches = [re.match(pattern, content) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    rewards = [1.0 if match else 0.0 for match in matches]
    print(completions)
    print(rewards)
    return rewards

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards
```

Then, you can initialize GRPOConfig and GRPOTrainer, pass in the reward functions we defined above and call train() to start training.

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    learning_rate=1e-5,
    remove_unused_columns=False,
    max_prompt_length=None,
    .. # setup other params of choice here
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor
)
trainer.train()
```


### Group Sequence Policy Optimization (GSPO)

[Group Sequence Policy Optimization](https://huggingface.co/papers/2507.18071) (GSPO) is a RL alignment algorithm recently released by Qwen that overcomes some limitations of GRPO. It achieves a more stable training computing importance sampling weights at the sequence level instead of per-token. Its benefits are more [relevant](https://github.com/volcengine/verl/pull/2775#issuecomment-3134375131) in MoE style models.

Latest TRL also introduces supports for GSPO and since it’s a variant of GRPO's loss, it comes with multimodal support. To create the trainer, the process is the same as with GRPO, but adding the following extra params (values are extracted from the paper).

```python
from trl import GRPOConfig

training_args = GRPOConfig(
		...
    importance_sampling_level="sequence",
    epsilon=3e-4,
    epsilon_high=4e-4,
    beta=0.0,
    loss_type="grpo",
    steps_per_generation=1,
    steps_per_generation=4
)
```

### Comparison

Here's a table summarizing model outputs for Qwen2.5VL-3B fine-tuned with below techniques. Note that we've done minimal runs on dataset subsets, and the models were fine-tuned on different datasets, so the comparison is made for vibe-check.



## vLLM Integration in TRL

vLLM is integrated in TRL to support online alignment methods where you need to generate samples during training. Running the scripts like following enables vLLM: 

```python
CUDA_VISIBLE_DEVICES=1,2 python3 examples/scripts/grpo_vlm.py     --model_name_or_path   Qwen/Qwen2.5-VL-3B-Instruct    …   --log_completions —use_vllm —vlm_mode colocate

```

There’s mainly two modes: `colocate` and `server` . `colocate` runs vLLM in the same process as the training loop, shares the same GPU between training and generation, creating a vLLM LLM instance inside the `GRPOTrainer` . Meanwhile `server` requires you to serve vLLM separately in a different process where you can hit the server. You can start this server with the command:

```python
trl vllm-serve --model Qwen/Qwen2.5-VL-3B-Instruct --tensor-parallel-size 1 
```

Then you can run the script as follows.

```python
CUDA_VISIBLE_DEVICES=1,2 python3 examples/scripts/grpo_vlm.py     --model_name_or_path   Qwen/Qwen2.5-VL-3B-Instruct    …   --log_completions —use_vllm —vlm_mode server

```

One more tip: we have added support for using vLLM with transformers backend in TRL. You can enable it when running a script with colocate or when serving the model, by passing the `--vllm_model_impl transformers` flag.

You can read more about vLLM integration in TRL [here](https://huggingface.co/docs/trl/en/vllm_integration).

### Useful Resources

Below, you can find a compilation of resources to explore the alignment of VLMs in detail. Enjoy!

- [Vision Language Models (Better, Faster, Stronger)](https://huggingface.co/blog/vlms-2025)
- [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://huggingface.co/papers/2411.10442)**(**MPO paper)
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Model ****[(GRPO paper)](https://huggingface.co/papers/2402.03300)
- [Open-R1](https://github.com/huggingface/open-r1) repository and [Open-R1 reward functions](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py)
- [TRL documentation](https://huggingface.co/docs/trl/en/index) and [TRL repository](https://github.com/huggingface/trl)
- [MPO VLM recipe](https://huggingface.co/learn/cookbook/fine_tuning_vlm_mpo)
- [GRPO VLM recipe](https://github.com/huggingface/cookbook/pull/312)
- [More multimodal alignment recipes](https://huggingface.co/learn/cookbook/index)
- [TRL multimodal training scripts](https://github.com/huggingface/trl/tree/main/examples/scripts)
- [**vLLM Integration in trl docs**](https://huggingface.co/docs/trl/en/vllm_integration)