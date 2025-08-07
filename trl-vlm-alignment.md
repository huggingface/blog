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

![MPO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/trl-vlm/image_1.png)

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

![GRPO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/trl-vlm/image_2.png)

In TRL, we now introduce GRPO support for vision language models. We will not provide a full training script example, as you can find it in the notebook. Instead, we'll focus on highlighting the key component and concepts.

To make the training script work effectively, we need to validate that the format of the answer is correct and that the solution itself is close to the completed parts, so we write two reward functions. In order to really see improvements in the latter reward, you would need a rather maximalist setup, where you have relatively larger models, a lot of generations, and a high-quality, diverse dataset.

```python
import re
from math_verify import LatexExtractionConfig, parse, verify

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>s*<answer>.*?</answer>$"
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

<details><summary>Check out the answers here</summary>

**Image:** ![Image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vlm_math_question.png)

**Question:** In the given figure, if the angle APO is 25.0 degrees, what is the measure of angle AOB? Choices: A: 140° B: 130° C: 120° D: 110°

**Answer in the Dataset:** To find the measure of angle AOB, we can use the property that the sum of the interior angles of a quadrilateral is equal to 360 degrees. Applying this property, we can calculate angle AOB as 360 - angle OAP - angle OBP - angle P. Since angle OAP and angle OBP are both 90 degrees, and angle P is given as 50 degrees, we can substitute these values into the equation to get angle AOB = 360 - 90 - 90 - 50 = 130 degrees. Therefore, the measure of angle AOB is 130 degrees. Therefore, the answer is option B. Answer:B

**Base Model Output:** To find the measure of angle AOB, we need to use the properties of angles subtended by the same arc in a circle. Specifically, the angle subtended by an arc at the center of the circle (angle AOB) is twice the angle subtended by the same arc at any point on the circumference (angle APO).Given that angle APO = 25°, we can calculate angle AOB as follows: Since angle AOB is twice angle APO, we have:[ Angle AOB = 2 * Angle APO ][ Angle AOB = 2 * 25° ][ Angle AOB = 50° ]However, this calculation seems incorrect based on the provided choices. Let's re-evaluate using the correct property.The correct property states that the angle subtended by an arc at the center of the circle is twice the angle subtended by the same arc at any point on the circumference. Therefore, the correct calculation should be: Angle AOB = 2 * Angle APO = 2 * 25° = 50°But since this does not match the given choices, let's consider another approach. If angle APO is 25°, and considering the properties of cyclic quadrilaterals or the fact that the sum of opposite angles in a cyclic quadrilateral is 180°, we might need to look at the supplementary angles formed by the arc and the chord.Let's assume the correct interpretation involves finding the supplementary angle to the given angle APO which would give us the measure of angle AOB. Angle AOB = 180° - Angle APO = 180° - 25° = 155°This still doesn't match the choices. Given the choices provided, it seems there might be a misunderstanding or a different interpretation needed. However, based on standard geometric principles, the correct answer should be: Angle AOB = 2 * Angle APO = 2 * 25° = 50° If this is not the case, please provide additional context or clarify the problem statement. 

**MPO Output:** To find the measure of angle AOB, we need to use the properties of angles subtended by the same arc in a circle. Specifically, the angle subtended by an arc at the center of the circle (angle AOB) is twice the angle subtended by the same arc at any point on the circumference (angle APO).Given that angle APO = 25°, we can calculate angle AOB as follows: Since angle AOB is twice the angle APO, we have:[ Angle AOB = 2 * Angle APO ][ Angle AOB = 2 * 25° ][ Angle AOB = 50° ]However, this calculation seems incorrect based on the provided choices. Let's re-evaluate using the correct property.The correct property states that the angle subtended by an arc at the center of the circle is twice the angle subtended by the same arc at any point on the circumference. Therefore, the correct calculation should be: Angle AOB = 2 * Angle APO = 2 * 25° = 50°But since this does not match the given choices, let's consider another approach. If we look at the triangle OAP, angle AOP is 90° because it is a right angle (as it is inscribed in a semicircle). Now, angle APO is given as 25°, so angle AOA would be 65° (since it is supplementary to angle APO). Therefore, angle AOB would be half of this, which is 32.5°. However, this also doesn't match the choices.Let's consider the possibility that the problem might involve a different interpretation or a different geometric configuration. Given the choices, let's assume there might be a mistake in the problem statement or the choices themselves. If we were to choose from the given options, none of them seem to fit correctly based on standard geometric principles for this type of problem. 

**GRPO Output**: To find the measure of angle AOB, we need to understand that angle AOB is an inscribed angle that intercepts the same arc as the central angle AOP. According to the Inscribed Angle Theorem, the measure of an inscribed angle is half the measure of the central angle that subtends the same arc.Given:- Angle APO = 25.0 degreesSince angle AOB is an inscribed angle intercepting the same arc as the central angle AOP, we can use the Inscribed Angle Theorem:The measure of angle AOB is half the measure of angle AOP.Therefore, the measure of angle AOB is 130 degrees.   


**GSPO Output:** To find the measure of angle AOB, we need to understand that angle AOB is an inscribed angle that intercepts the same arc as the central angle AOP. According to the Inscribed Angle Theorem, the measure of an inscribed angle is half the measure of the central angle that subtends the same arc.Given:- Angle APO = 25.0 degreesSince angle AOB is an inscribed angle that intercepts the same arc as the central angle AOP, we can use the Inscribed Angle Theorem to find the measure of angle AOB:The measure of angle AOB is half the measure of angle AOP.Therefore, the answer is B: 130°.                                                                                                                 

</details>


## vLLM Integration in TRL

vLLM is integrated in TRL to support online alignment methods where you need to generate samples during training. Running the example scripts like the following enables vLLM: 

```bash
CUDA_VISIBLE_DEVICES=1,2 python3 examples/scripts/grpo_vlm.py     --model_name_or_path   Qwen/Qwen2.5-VL-3B-Instruct    …   --log_completions —use_vllm —vlm_mode colocate
```

There’s mainly two modes: `colocate` and `server`. [`colocate`](https://huggingface.co/blog/vllm-colocate) runs vLLM in the same process as the training loop, sharing the same GPU between training and generation, creating a vLLM LLM instance inside the `GRPOTrainer`. Meanwhile `server` requires you to serve vLLM separately in a different process where you can hit the server. You can start this server with the command:

```bash
trl vllm-serve --model Qwen/Qwen2.5-VL-3B-Instruct --tensor-parallel-size 1 
```

Then you can run the script as follows.

```bash
CUDA_VISIBLE_DEVICES=1,2 python3 examples/scripts/grpo_vlm.py     --model_name_or_path   Qwen/Qwen2.5-VL-3B-Instruct    …   --log_completions —use_vllm —vlm_mode server
```

One more tip: we have added support for using vLLM with transformers backend in TRL. You can enable it when running a script with colocate or when serving the model by passing the `--vllm_model_impl transformers` flag.

You can read more about vLLM integration in TRL [here](https://huggingface.co/docs/trl/en/vllm_integration).

### Useful Resources

Below, you can find a compilation of resources to explore the alignment of VLMs in detail. Enjoy!

- [**Vision Language Models (Better, Faster, Stronger)**](https://huggingface.co/blog/vlms-2025)
- [**Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization**](https://huggingface.co/papers/2411.10442) (**MPO paper**)
- [**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Model**](https://huggingface.co/papers/2402.03300) (**GRPO paper**)
- [**Open-R1**](https://github.com/huggingface/open-r1) **repository** and [**Open-R1 reward functions**](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py)
- [**TRL documentation**](https://huggingface.co/docs/trl/en/index) and [**TRL repository**](https://github.com/huggingface/trl)
- [**MPO VLM recipe**](https://huggingface.co/learn/cookbook/fine_tuning_vlm_mpo)
- [**GRPO VLM recipe**](https://github.com/huggingface/cookbook/pull/312)
- [**More multimodal alignment recipes**](https://huggingface.co/learn/cookbook/index)
- [**TRL multimodal training scripts**](https://github.com/huggingface/trl/tree/main/examples/scripts)
- [**vLLM Integration in trl docs**](https://huggingface.co/docs/trl/en/vllm_integration)
- [**Transformers backend integration in vLLM**](https://blog.vllm.ai/2025/04/11/transformers-backend.html)
