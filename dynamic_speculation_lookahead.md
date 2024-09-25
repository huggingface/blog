---
title: "Dynamic number of speculative tokens accelerates speculative decoding"
thumbnail: /blog/assets/optimum_intel/intel_thumbnail.png
authors:
- user: user1
  guest: true
  org: Intel
- user: user2
  guest: true
  org: Intel
---
# Dynamic number of speculative tokens accelerates speculative decoding

Speculative decoding is a technique often employed to decrease the inference latency of large language models. Its success largely hinges on the speculation lookahead (SL)—the count of tokens produced by the draft model in each iteration.


Transformers offer two distinct methods to determine the schedule for adjusting the number of assistant tokens during inference. The straightforward method uses a static value of the speculation lookahead and involves generating a constant number of candidate tokens at each speculative iteration (`num_assistant_tokens_schedule="constant"`). Alternatively, a heuristic-based approach adjusts the number of candidate tokens for the next iteration based on the acceptance rate of the current iteration. If all speculative tokens are correct, the number of candidate tokens increases; otherwise, it decreases (`num_assistant_tokens_schedule="heuristic"`).

We utilize an oracle to determine the optimal speculation lookahead value for each speculative iteration. The oracle employs the draft model to autoregressively generate tokens until a discrepancy arises between the predicted tokens of the draft and target models. 

The figure below illustrates the oracle and static speculation lookahead values for various speculative iterations on a [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) example. A high variance in oracle speculation lookahead values is observed. With static speculation lookahead, we perform 38 target forward passes and 192 draft forward passes, whereas for oracle speculation lookahead, we only perform 27 target forward passes and 129 draft forward passes. 

<p align="center">
    <img src="assets/dynamic_speculation_lookahead/oracle_K_2.png" width=500>
</p>
<p align="center">
    <em>Oracle and static speculation lookahead values for different speculative iterations on one MBPP example.</em>
</p>
The figure below illustrates the average speculation lookahead across the normalized index of speculative iterations for the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca).

<p align="center">
    <img src="assets/dynamic_speculation_lookahead/Alpaca.png" width=500>
</p>
<p align="center">
    <em>The average oracle speculation lookahead over the normalized index of the speculative iterations for the Alpaca dataset</em>
</p>

Both figures demonstrate significant variability in oracle speculation lookahead values, suggesting that a static speculation lookahead may be suboptimal.

We propose a straightforward method to dynamically adjust the speculation lookahead value at each iteration. After generating each draft token, we determine whether the draft model should continue generating the next token or switch to the target model for verification. This decision is based on the assistant model's confidence in its prediction estimated by the softmax of the logits. If the assistant model's confidence in its current token prediction falls below a predefined threshold referred to as the `assistant_confidence_threshold`, it halts the token generation process for that iteration, even if the maximum number of speculative tokens `num_assistant_tokens` has not been reached.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = "Alice and Bob"
checkpoint = "EleutherAI/pythia-1.4b-deduped"
assistant_checkpoint = "EleutherAI/pythia-160m-deduped"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).to(device)

# Set parameters for the dynamic speculation lookahead
assistant_model.num_assistant_tokens_schedule='constant'
assistant_model.generation_config.assistant_confidence_threshold="0.4"
assistant_model.generation_config.num_assistant_tokens="20"

outputs = model.generate(**inputs, assistant_model=assistant_model)
```


# Results

# References
- [Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models](https://arxiv.org/abs/2405.04304)
- ["Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation)

