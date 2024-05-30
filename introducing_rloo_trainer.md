---
title: "Introducing RLOO Trainer in TRL"
thumbnail: 
authors:
- user: vwxyzjn
- user: ArashAhmadian
- user: lewtun
---

We are excited to introduce the RLOO (REINFORCE LeaveOne-Out) Trainer in TRL, a new online RLHF training algorithm. RLOO is an alternative to PPO that is designed to be highly effective and more GPU memory efficient. This blog post will explain the motivation behind RLOO Trainer, how it works, and how to use it in TRL.


## Motivation

PPO is an effective online RLHF training algorithm that has been used to train state-of-the-art models such as GPT-4. However, PPO can be quite challenging to use in practice due to its high GPU memory requirements. This is because PPO needs to load 4 copies of the models into the memory: 1) the policy model, 2) the reference policy model, 3) the reward model, and the 4) the value model. PPO also has many subtle implementation details that can be difficult to get right such as general advantage estimation (GAE).

Ahmadian et al. (2024) introduced RLOO, a new online RLHF training algorithm which is simpler to implement and more GPU memory efficient than PPO. In particular, RLOO models the entire completion tokens as a single action, which which simplifies the implementation and does not need to use GAE. Furthermore, RLOO only needs to load 3 copies of the models into the memory: 1) the policy model, 2) the reference policy model, 3) the reward model.


## How RLOO Works

RLOO has several key ideas. First, it treats the **entire model completion** as a single action, whereas regular PPO treats **each completion token** as individual actions. Here we have the following steps:

1. The policy model would generation some completion tokens and get the per-token logprobs under the current policy and the reference policy. 
2. We then calculate the per-token KL penalties as the difference between the logprobs under the current policy and the reference policy.
3. We then get the score of the entire completion from the reward model.

From here and on, regular PPO and RLOO differ in approach. Regular PPO would attribute a reward for each action, whereas RLOO would attribute a reward for the entire completion, as demonstrated below.

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
# fmt: off
rlhf_reward = torch.tensor([
    1, 2, 3, # first rlhf reward for three prompts
    2, 3, 4, # second rlhf reward for three prompts
    5, 6, 7, # third rlhf reward for three prompts
    8, 9, 10, # fourth rlhf reward for three prompts
]).float() # here we have 3 prompts which have 4 completions each
# fmt: on

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

Btw a big shout out to Arash Ahmadian who provided the vectorized implementation of the advantages calculation above.


## How we implemented RLOO Trainer in TRL

We implemented RLOO trainer based on our new experimental `PPOv2Trainer`, which is itself based on https://arxiv.org/abs/2403.17031. One interesting detail is that we implemented the RLOO trainer while still using the PPO loss. This is because the loss of REINFORCE / Advantage actor critic is a special case of PPO (https://arxiv.org/abs/2205.09123). Note that even though the logprob is explicitly in the REINFORCE loss, it is also in the PPO loss implicitly. Seeing is believing, so let's demonstrate this with a simple example.

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



## Experiments

To validate the RLOO implementation works, we ran experiments on the 1B and 6.9B models. Here are the commands we used to run the experiments. We take the SFT / RM models directly from (https://arxiv.org/pdf/2403.17031).

```
# 1B RLOO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/rloo/rloo_tldr.py \
    --output_dir models/minimal/rloo_tldr \
    --num_ppo_epochs 2 \
    --num_mini_batches 2 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --non_eos_penalty \
    --stop_token eos \
    --kl_coef 0.03

# 6.9B RLOO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/rloo/rloo_tldr.py \
    --output_dir models/minimal/rloo_tldr_6.9b \
    --num_ppo_epochs 2 \
    --num_mini_batches 2 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 256 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-6.9b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 2 \
    --non_eos_penalty \
    --stop_token eos \
    --kl_coef 0.03
```

1B experiment can be found here:

- [🤗 Model checkpoint](https://huggingface.co/vwxyzjn/rloo_tldr)
- [🐝 Tracked experiment](https://wandb.ai/huggingface/trl/runs/u2sqci34)


To evaluate, we use vLLM to load the checkpoints and GPT3.5 as a judge model to evaluate the generated TL;DR against the reference TL;DR.
```bash
python -i examples/scripts/evals/generate_tldr.py \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --output_path examples/scripts/minimal/evals/sft_tldr.csv \
    --n 1000
# preferred
# response1    656
# response0    344
# Name: count, dtype: int64
python -i examples/scripts/evals/generate_tldr.py \
    --model_name_or_path vwxyzjn/rloo_tldr \
    --output_path examples/scripts/minimal/evals/rloo_tldr.csv \
    --n 1000
# preferred
# response0    532
# response1    468
# Name: count, dtype: int64
```

The RLOO checkpoint gets a 53.2% preferred rate vs the 34.4% preference rate of the SFT checkpoint. This is a good sign that the RLOO training is working as intended.



TODO: fill in 6.9B results


The 6.9B checkpoint gets a 78.7% (k=2) preferred rate using GPT4 as a judge, which even exceeds the best reported performance of 77.9% (k=4) and 74.2 (k=2) in the original paper. This is a good sign that the RLOO training is working as intended.

```
response0    787
response1    213
Name: count, dtype: int64
```



Metrics:

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/benchmark/pr-1540/rloo.png?download=true)

* a2c is a special case of PPO 
* reproduction of TL;DR summarization
* comparison with PPO
* how to use RLOO Trainer in TRL
* charts comparing RLOO runtime, memory usage, and performance with PPO