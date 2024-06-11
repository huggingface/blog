---
title: "Putting RL back in RLHF"
thumbnail: /blog/assets/putting_rl_back_in_rlhf_with_rloo/thumbnail.png
authors:
- user: vwxyzjn
- user: ArashAhmadian
  org: CohereForAI
  guest: true
---



We are excited to introduce the RLOO (REINFORCE Leave One-Out) Trainer in TRL. As an alternative to PPO, RLOO is a new online RLHF training algorithm that is designed to be more accessible and easier to implement. In particular, **RLOO requires less GPU memory and takes less wall time to converge.** As shown in the figures below:


1. 🤑RLOO uses **approximately 50-70% less** vRAM than PPO, depending on the model size
2. 🚀RLOO runs **2x faster** than PPO with 1B models and up to **3x faster** than PPO with 6.9B models.
3. 🔥RLOO performs **competitively to PPO** in terms of the response win rate (judged by GPT4) and consistently outperforms popular offline methods like DPO.

With RLOO, we bring reinforcement learning back into RLHF, enabling the community to explore online RL methods more easily. This is exciting because more and more studies have shown that online RL is more effective than offline methods such as DPO ([https://arxiv.org/abs/2402.04792](https://arxiv.org/abs/2402.04792), [https://arxiv.org/abs/2405.08448](https://arxiv.org/abs/2405.08448)). 


![alt_text](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/putting_rl_back_in_rlhf_with_rloo/image3.png?download=true "image_tooltip")
![alt_text](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/putting_rl_back_in_rlhf_with_rloo/image8.png?download=true "image_tooltip")
![alt_text](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/putting_rl_back_in_rlhf_with_rloo/image1.png?download=true "image_tooltip")


This blog post will explain the motivation behind the RLOO Trainer, how it works, and how to use it in TRL. 


# Motivation

PPO is an effective online RLHF training algorithm that is used to train state-of-the-art models such as GPT-4. However, PPO can be quite challenging to use in practice due to its high GPU memory requirements. In particular, PPO needs to load 4 copies of the models into the memory: 1) the policy model, 2) the reference policy model, 3) the reward model, and 4) the value model, as shown in the following figure. PPO also has many subtle implementation details that can be difficult to get right ([Engstrom et al; 2020](https://openreview.net/forum?id=r1etN1rtPB), [Huang et al 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)).


![alt_text](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/putting_rl_back_in_rlhf_with_rloo/image7.png?download=true "image_tooltip")


In a new paper from Cohere, [Ahmadian et al. (2024)](https://cohere.com/research/papers/back-to-basics-revisiting-reinforce-style-optimization-for-learning-from-human-feedback-in-llms-2024-02-23) revisited the basics of RLHF training and proposed a more elegant method called RLOO, a new online training algorithm. RLOO only needs to load 3 copies of the models into the memory: 1) the policy model, 2) the reference policy model, and 3) the reward model, as shown in the figure above. 

Importantly, RLOO requires less memory, meaning it’s easier to 



1. run without OOMs (out-of-memory errors) 
2. being able to load larger batch sizes
3. runs more efficiently and faster.

Furthermore, RLOO models the entire completion tokens as a single action, as illustrated in the figure below. In the next section, we will dive into further detail with code snippets.


![alt_text](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/putting_rl_back_in_rlhf_with_rloo/image4.png?download=true "image_tooltip")



# How RLOO Works

Both RLOO and PPO have several shared steps: 

1. The policy model would generate some completion tokens and get the per-token logprobs under the current policy and the reference policy. 

2. We then calculate the per-token KL penalties as the difference between the logprobs under the current policy and the reference policy.

3. We then get the score of the entire completion from the reward model.

From here on, regular PPO and RLOO differ in approach. RLOO has several key ideas. First, it treats the ****entire model completion**** as a single action, whereas regular PPO treats ****each completion token**** as individual actions. Typically, only the EOS token gets a true reward, which is very sparse.  Regular PPO would attribute a reward to the EOS token, whereas RLOO would attribute that EOS reward to the entire completion, as demonstrated below.

```python
from torch import Tensor
response = Tensor([4., 5., 6.])
per_token_logprobs = Tensor([-12.3, -8.3, -2.3])
reference_per_token_logprobs = Tensor([-11.3, -8.4, -2.0])
kl = per_token_logprobs - reference_per_token_logprobs
score_from_rm = 1.0
print(f"{kl=}")  # kl=tensor([-1.0000,  0.1000, -0.3000])
per_token_reward = kl.clone()
per_token_reward[-1] += score_from_rm  # assume last token is the EOS token
print(f"{per_token_reward=}")  # per_token_reward=tensor([-1.0000,  0.1000,  0.7000])
print(f"{score_from_rm=}")  # score_from_rm=1.0
print("#### Modeling each token as an action")
for action, reward in zip(response, per_token_reward):
    print(f"{action=}, {reward=}")
# action=tensor(4.), reward=tensor(-1.)
# action=tensor(5.), reward=tensor(0.1000)
# action=tensor(6.), reward=tensor(0.7000)
print("#### Modeling the entire response as an action")
entire_generation_reward = per_token_reward.sum()
print(f"action='entire completion', reward={entire_generation_reward}")
# action='entire completion', reward=-0.2000 (-1 + 0.1 + 0.7)
```

Second, RLOO uses the REINFORCE loss, which basically multiplies the (reward - baseline) by the logprob of actions. Here we highlight the differences between per-token REINFORCE loss and the entire completion REINFORCE loss. Note that for PPO's loss, we would need to calculate the advantage additionally based on the value model with GAE.

```python
# ... continue from above
baseline = Tensor([0.2, 0.3, 0.4])  # dummy baseline
print("#### Modeling each token as an action")
advantage = per_token_reward - baseline
per_token_reinforce_loss = per_token_logprobs * advantage
print(f"{advantage=}")  # advantage=tensor([-1.2000, -0.2000,  0.3000])
print(f"{per_token_reinforce_loss=}")  # per_token_reinforce_loss=tensor([14.7600,  1.6600, -0.6900])
print(f"{per_token_reinforce_loss.mean()=}")  # per_token_reinforce_loss.mean()=tensor(5.2433)

print("#### Modeling the entire response as an action")
advantage = entire_generation_reward - baseline.sum()
reinforce_loss = per_token_logprobs.sum() * advantage
print(f"{advantage=}")  # advantage=tensor(-1.1000)
print(f"{reinforce_loss=}")  # reinforce_loss=tensor(25.1900)
```

Third, RLOO calculates baselines smartly. Notice we used a dummy baseline above. In practice, RLOO uses the reward of all other samples in the batch as the baseline. Below is a case where we have 3 prompts and 4 completions each. We calculate the baseline for each completion by averaging the rewards of all other completions for the same prompt.

```python
import torch
local_batch_size = 3
rloo_k = 4

rlhf_reward = torch.tensor([
    1, 2, 3, # first rlhf reward for three prompts
    2, 3, 4, # second rlhf reward for three prompts
    5, 6, 7, # third rlhf reward for three prompts
    8, 9, 10, # fourth rlhf reward for three prompts
]).float() # here we have 3 prompts which have 4 completions each

# slow impl
baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
advantages = torch.zeros_like(rlhf_reward)
for i in range(0, len(advantages), local_batch_size):
    other_response_rlhf_rewards = []
    for j in range(0, len(advantages), local_batch_size):
        if i != j:
            other_response_rlhf_rewards.append(rlhf_reward[j : j + local_batch_size])
    advantages[i : i + local_batch_size] = rlhf_reward[i : i + local_batch_size] - torch.stack(
        other_response_rlhf_rewards
    ).mean(0)
assert (1 - (2 + 5 + 8) / 3 - advantages[0].item()) < 1e-6
assert (6 - (3 + 2 + 9) / 3 - advantages[7].item()) < 1e-6

# vectorized impl
rlhf_reward = rlhf_reward.reshape(rloo_k, local_batch_size)
baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
vec_advantages = rlhf_reward - baseline
torch.testing.assert_close(vec_advantages.flatten(), advantages)
```

A big shout out to Arash Ahmadian, who provided the vectorized implementation of the advantages calculation above.


# Get started with using RLOO with TRL

To get started with RLOO, you can install the latest version of of TRL via `pip install --upgrade trl` and import the RLOOTrainer. Below is a short snippet that shows some high-level API usage. Feel free to checkout the documentation 



* [https://huggingface.co/docs/trl/main/en/rloo_trainer](https://huggingface.co/docs/trl/main/en/rloo_trainer) 
* [https://huggingface.co/docs/trl/main/en/ppov2_trainer](https://huggingface.co/docs/trl/main/en/ppov2_trainer) 

```pythonfrom transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE


base_model_name = "EleutherAI/pythia-1b-deduped"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
reward_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)
ref_policy = AutoModelForCausalLM.from_pretrained(base_model_name)
policy = AutoModelForCausalLM.from_pretrained(base_model_name)

train_dataset = ...  # make sure to have columns "input_ids"
eval_dataset = ...

trainer = RLOOTrainer(
    config=RLOOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        total_episodes=30000,
    ),
    tokenizer=tokenizer,
    policy=policy,
    ref_policy=ref_policy,
    reward_model=reward_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

Here is an example of tracked weights and biases experiments: [https://wandb.ai/huggingface/trl/runs/dd2o3g35](https://wandb.ai/huggingface/trl/runs/dd2o3g35) 





![alt_text](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/putting_rl_back_in_rlhf_with_rloo/image9.png?download=true "image_tooltip")


When coding the RLOO and PPOv2 implementation, we emphasize making it easier to improve the transparency of model development. In particular, we have enhanced the docs to include an explanation of logged metrics and a cookbook guide on reading and debugging these metrics. For example, we recommend closely monitoring objective/rlhf_reward, the ultimate objective of the RLHF training, during training.




![alt_text](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/putting_rl_back_in_rlhf_with_rloo/image2.png?download=true "image_tooltip")

![alt_text](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/putting_rl_back_in_rlhf_with_rloo/image6.png?download=true "image_tooltip")







To help visualize the training progress,, we periodically log some sample completions from the model. Here is an example of a completion. In an example tracked run at Weights and Biases ([https://wandb.ai/huggingface/trl/runs/dd2o3g35](https://wandb.ai/huggingface/trl/runs/dd2o3g35)), it looks like the following, allowing you to see the model’s response at different stages of training. By default, we generate --num_sample_generations 10 during training, but you can customize the number of generations.





![alt_text](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/putting_rl_back_in_rlhf_with_rloo/image5.gif?download=true "image_tooltip")



# How we implemented RLOO Trainer in TRL

We implemented RLOO trainer based on our new experimental `PPOv2Trainer`, which is itself based on https://arxiv.org/abs/2403.17031. Interestingly, our implementation of the RLOO trainer still uses the PPO loss. This is because the loss of REINFORCE is a special case of PPO (https://arxiv.org/abs/2205.09123). Note that even though the logprob is explicitly in the REINFORCE loss, it is also implicitly in the PPO loss. Seeing is believing, so let's demonstrate this with a simple example.

```python
import torch.nn.functional as F
from torch import LongTensor, Tensor, gather, no_grad

action = LongTensor([1])
advantage = Tensor([1.0])
logits = Tensor([[1.0, 2.0, 1.0, 1.0]])
logits.requires_grad = True
all_logprob = F.log_softmax(logits, dim=-1)
with no_grad():
    old_logprob = gather(all_logprob, 1, action.unsqueeze(-1)).squeeze(-1)
logprob = gather(all_logprob, 1, action.unsqueeze(-1)).squeeze(-1)
ratio = (logprob - old_logprob).exp()
ppo_loss = (ratio * advantage).mean() # [πθ(at | st) / πθ_old(at | st) * At]
# when the πθ and πθ_old are the same, the ratio is 1, and PPO's clipping has no effect
ppo_loss.backward()
print(f"{logits.grad=}")  # tensor([[-0.1749,  0.5246, -0.1749, -0.1749]])
logits2 = Tensor([[1.0, 2.0, 1.0, 1.0]])
logits2.requires_grad = True
all_logprob2 = F.log_softmax(logits2, dim=-1)
logprob2 = gather(all_logprob2, 1, action.unsqueeze(-1)).squeeze(-1)
reinforce_loss = logprob2 * advantage  # [log πθ(at | st) * At]
reinforce_loss.mean().backward()
print(f"{logits2.grad=}")  # tensor([[-0.1749,  0.5246, -0.1749, -0.1749]])
```


# Experiments

To validate the RLOO implementation works, we ran experiments on the Pythia 1B and 6.9B models and release the trained checkpoints here:



* [https://huggingface.co/collections/vwxyzjn/rloo-ppov2-tl-dr-summarize-checkpoints-66679a3bfd95ddf66c97420d](https://huggingface.co/collections/vwxyzjn/rloo-ppov2-tl-dr-summarize-checkpoints-66679a3bfd95ddf66c97420d)  

We take the SFT / RM models directly from ([https://arxiv.org/pdf/2403.17031](https://arxiv.org/pdf/2403.17031)). To evaluate, we use vLLM to load the checkpoints and GPT4 as a judge model to assess the generated TL;DR against the reference TL;DR. We also look at the GPU memory usage and runtime, as shown in the figures at the beginning of the blog post. To reproduce our work, feel free to check out the commands in our docs:



* [https://huggingface.co/docs/trl/main/en/rloo_trainer#benchmark-experiments](https://huggingface.co/docs/trl/main/en/rloo_trainer#benchmark-experiments) 
* [https://huggingface.co/docs/trl/main/en/rloo_trainer#benchmark-experiments](https://huggingface.co/docs/trl/main/en/rloo_trainer#benchmark-experiments)  

The key results are as follows:



* **🚀Highly performant RLOO checkpoint: **The 6.9B checkpoint gets a 78.7% (k=2) preferred rate using GPT4 as a judge, which even exceeds the best-reported performance of 77.9% (k=4) and 74.2 (k=2) in the original [paper](https://arxiv.org/abs/2402.14740). This is a good sign that our RLOO training is working as intended.
    * The RLOO 1B checkpoint has a 40.1% win rate compared to the SFT checkpoint's 21.3% win rate. This is a good sign that the RLOO training is working as intended.
* 🤑**Less GPU memory and runs faster**: RLOO training uses less memory and runs faster, making it a highly useful algorithm for online RL training.


# Numerical Stability: The Dark Side

Despite the performance and compute efficiency advantages of RLOO, we want to highlight some numerical issues. Specifically, the response logprobs obtained during generation are slightly numerically different from the logprobs obtained during the training forward passes under `bf16`. This causes an issue for both PPO and RLOO, but it’s much worse for RLOO as explained below.

For example, say we are generating 10 tokens for two sequences. Under the precision `fp32`, the output looks as follows, where the `ratio = (forward_logprob - generation_logprob).exp()` and is what PPO used to clip. Under the first epoch and first minibatch, the ratio should be exactly the same because the model hasn’t done any updates:

```
generation_logprob=tensor([[    -0.1527,     -0.2258,     -3.5535,     -3.4805,     -0.0519,
             -2.3097,     -2.0275,     -0.4597,     -0.1687,     -0.0000],
        [    -0.1527,     -0.2258,     -5.2855,     -0.1686,     -8.4760,
             -4.3118,     -1.0368,     -0.8274,     -1.6342,     -2.6128]],
       device='cuda:0')
forward_logprob=tensor([[-0.1527, -0.2258, -3.5535, -3.4805, -0.0519, -2.3097, -2.0275, -0.4597,
         -0.1687],
        [-0.1527, -0.2258, -5.2855, -0.1686, -8.4760, -4.3118, -1.0368, -0.8274,
         -1.6342]], device='cuda:0', grad_fn=<SqueezeBackward1>)
ratio=tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]],
       device='cuda:0', grad_fn=<ExpBackward0>)
ratio.mean()=0.9999998211860657
ratio.std()=6.592738373001339e-06
ratio.max()=1.0000133514404297
ratio.min()=0.9999887943267822
```
However, under bf16, we get 
```
generation_logprob=tensor([[    -0.1426,     -0.1904,     -3.5938,     -3.4688,     -0.0618,
             -2.3906,     -2.0781,     -0.4375,     -0.1562,     -0.0000],
        [    -0.1426,     -0.1904,     -5.2812,     -0.1641,     -8.5625,
             -4.2812,     -1.0078,     -0.8398,     -1.5781,     -2.5781]],
       device='cuda:0', dtype=torch.bfloat16)
forward_logprob=tensor([[-0.1445, -0.1670, -3.5938, -3.5156, -0.0554, -2.2969, -1.9688, -0.5273,
         -0.1953],
        [-0.1445, -0.1670, -5.2812, -0.1533, -8.5625, -4.3125, -1.0000, -0.7852,
         -1.6641]], device='cuda:0', dtype=torch.bfloat16,
       grad_fn=<SqueezeBackward1>)
ratio=tensor([[1.0000, 0.9766, 1.0000, 1.0469, 0.9922, 0.9102, 0.8945, 1.0938, 1.0391],
        [1.0000, 0.9766, 1.0000, 0.9883, 1.0000, 1.0312, 0.9922, 0.9453, 1.0859]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<ExpBackward0>)
ratio.mean()=1.0
ratio.std()=0.051025390625
ratio.max()=1.09375
ratio.min()=0.89453125
```
and under fp16, we get
```
generation_logprob=tensor([[    -0.1486,     -0.2212,     -3.5586,     -3.4688,     -0.0526,
             -2.3105,     -2.0254,     -0.4629,     -0.1677,     -0.0000],
        [    -0.1486,     -0.2212,     -5.2852,     -0.1681,     -8.4844,
             -4.3008,     -1.0322,     -0.8286,     -1.6348,     -2.6074]],
       device='cuda:0', dtype=torch.float16)
forward_logprob=tensor([[-0.1486, -0.2212, -3.5586, -3.4805, -0.0529, -2.3066, -2.0332, -0.4629,
         -0.1676],
        [-0.1486, -0.2212, -5.2852, -0.1682, -8.4766, -4.3008, -1.0322, -0.8281,
         -1.6299]], device='cuda:0', dtype=torch.float16,
       grad_fn=<SqueezeBackward1>)
ratio=tensor([[1.0000, 1.0000, 1.0000, 1.0117, 1.0000, 0.9961, 1.0078, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000, 0.9922, 1.0000, 1.0000, 0.9995, 0.9951]],
       device='cuda:0', dtype=torch.float16, grad_fn=<ExpBackward0>)
ratio.mean()=1.0
ratio.std()=0.00418853759765625
ratio.max()=1.01171875
ratio.min()=0.9921875
```


Note that the ratio for `bf16` is very unstable for some reason. When ratio becomes large, PPO’s clip coefficient = 0.2 kicks in, **nulling** the gradient of the tokens for which the ratio is greater than 1.2 or lower than 0.8. With RLOO, this issue is more extreme because we are looking at the `(forward_logprob.sum(1) - generation_logprob.sum(1)).exp() = [ 1.0625, 12.1875]`, which means the gradient for the entire second sequence is nulled. 

In practice, we noticed PPO nulls the gradient of approximately 3% of the batch data, whereas RLOO nulls about 20-40% of the batch data. Theoretically, RLOO should null 0% of the batch data, when not using mini-batches. Importantly, we observe that the clipping ratio for RLOO did not change significantly once we increased the number of gradient steps before generating new batches (through num_ppo_epochs and num_mini_batches), this provides empirical evidence that the clipping ratio is indeed due to numerical issues with bf16 as opposed to the behavior and latest policies being significantly different, as positioned in the paper. 

To keep reading about the latest issue updates, feel free to check out [https://github.com/huggingface/transformers/issues/31267](https://github.com/huggingface/transformers/issues/31267). 


# Conclusion

The introduction of the RLOO (REINFORCE Leave One-Out) Trainer in TRL is an exciting algorithm in online RLHF training, providing a more accessible and efficient alternative to PPO. By reducing GPU memory usage and simplifying the training process, RLOO enables larger batch sizes and faster training times. Our experiments demonstrate that RLOO performs competitively with PPO and outperforms DPO checkpoints in terms of response win rate, making it a powerful tool for effective online RLHF. Explore our documentation to get started!



* [https://huggingface.co/docs/trl/main/en/rloo_trainer](https://huggingface.co/docs/trl/main/en/rloo_trainer) 
* [https://huggingface.co/docs/trl/main/en/ppov2_trainer](https://huggingface.co/docs/trl/main/en/ppov2_trainer) 


# Acknowledgment and Thanks 

We thank Lewis Tunstall, Sarah Hooker, and Leandro Von Werra for the helpful feedback on this blog post.
