---
title: "Faster Assisted Generation with Dynamic Speculation"
thumbnail: /blog/assets/optimum_intel/intel_thumbnail.png
authors:
- user: jmamou
  guest: true
  org: Intel
- user: orenpereg
  guest: true
  org: Intel
- user: joaogante
- user: lewtun
- user: danielkorat
  guest: true
  org: Intel
- user: Nadav-Timor
  guest: true
  org: weizmannscience
- user: moshew
  guest: true
  org: Intel

---

⭐ In this blog post, we’ll explore *dynamic speculative decoding* —a novel method developed by Intel labs and Hugging Face that accelerates text generation by up to 2.7x, depending on the task. This method is the default operational mode for assisted generation starting from [Transformers🤗](https://github.com/huggingface/transformers) release [4.45.0](https://github.com/huggingface/transformers/releases/tag/v4.45.0) ⭐

## Speculative Decoding
[Speculative decoding](https://arxiv.org/abs/2211.17192) is a popular technique to accelerate the inference of large language models, while preserving their accuracy. As shown in the figure below, speculative decoding works by splitting the generative process into two stages. In the first stage, a fast, but less accurate *draft* model (AKA assistant) autoregressively generates a sequence of tokens. In the second stage, a large, but more accurate *target* model conducts parallelized verification over the generated draft tokens. This process allows the target model to produce multiple tokens in a single forward pass and thus accelerate autoregressive decoding. The success of speculative decoding largely hinges on the _speculation lookahead_ (SL), i.e. the number of tokens produced by the draft model in each iteration. In practice, the SL is either a static value or based on heuristics, neither of which is optimal for squeezing out maximium performance during inference.


<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dynamic_speculation_lookahead/spec_dec_diagram.png" width="250"><br>
<em>Speculative decoding iteration.</em>
</figure>

## Dynamic Speculative Decoding

[Transformers🤗](https://github.com/huggingface/transformers) offers two distinct methods to determine the schedule for adjusting the number of draft (assistant) tokens during inference. The straightforward method, based on [Leviathan et al.](https://arxiv.org/pdf/2211.17192), uses a static value of the speculation lookahead and involves generating a constant number of candidate tokens at each speculative iteration. Alternatively, a [heuristic-based approach](https://huggingface.co/blog/assisted-generation) adjusts the number of candidate tokens for the next iteration based on the acceptance rate of the current iteration. If all speculative tokens are correct, the number of candidate tokens increases; otherwise, it decreases.

We anticipate that an enhanced optimization strategy for managing the number of generated draft tokens could squeeze out further latency reductions. For testing this thesis we utilize an oracle that determines the optimal speculation lookahead value for each speculative iteration. The oracle employs the draft model to autoregressively generate tokens until a discrepancy arises between the predicted tokens of the draft and target models. This process is repeated for each speculative iteration, ultimately identifying the optimal (maximum) number of draft tokens accepted per iteration. The draft/target token mismatch is identified using the rejection sampling algorithm, introduced by Leviathan et al., with zero temperature. This oracle realizes the full potential of speculative decoding by generating the maximum number of valid draft tokens at each step and minimizing the number of calls to both the draft and target models.

The left figure below illustrates the oracle and static speculation lookahead values across the speculative iterations of a code generation example from the [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) dataset. A high variance in oracle speculation lookahead values (orange bars) is observed.
The static speculation lookahead (blue bars), where the number of generated draft tokens is fixed to 5, performs 38 target forward passes and 192 draft forward passes, whereas the oracle speculation lookahead, performs only 27 target forward passes and 129 draft forward passes - a significant reduction. The right figure shows the oracle and static speculation lookahead across the entire [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset.


<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dynamic_speculation_lookahead/oracle_K_2.png" style="width: 400px; height: auto;"><br>
  <em>Oracle and static speculation lookahead (SL) values on one MBPP example.</em>
</p>

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dynamic_speculation_lookahead/Alpaca.png" style="width: 400px; height: auto;"><br>
  <em>Average oracle speculation lookahead for the entire Alpaca dataset.</em>
</p>

Both figures demonstrate significant variability in oracle speculation lookahead values, suggesting that a static speculation lookahead may be suboptimal.


In order to get closer to the Oracle and gain extra speedup, we developed a straightforward method to dynamically adjust the speculation lookahead value at each iteration. After generating each draft token, we determine whether the draft model should continue generating the next token or switch to the target model for verification. This decision is based on the assistant model's confidence in its prediction, estimated by the softmax of the logits. If the assistant model's confidence in the current token prediction falls below a predefined threshold, referred to as the `assistant_confidence_threshold`, it halts the token generation process for that iteration, even if the maximum number of speculative tokens `num_assistant_tokens` has not been reached. Once halted, the draft tokens generated during the current iteration are sent to the target model for verification.

## Benchmarking

We benchmarked the dynamic approach against the heuristic approach across a range of tasks and model pairings. The dynamic approach showed better performance in all tests.
Notably, using the dynamic approach with `Llama3.2-1B` as the assistant for `Llama3.1-8B`, we observe speedups of up to 1.52x, whereas the heuristic approach showed no significant speedups with the same setup. Another observation is that `codegen-6B-mono` yields _slowdown_ using the heuristic approach, whereas the dynamic approach shows speedup.


| Target model | Draft (Assistant) model | Task | Speedup - heuristic | Speedup - dynamic |
|----------------------|---------------------|---------------------------|---------------------------|---------------------------|
| `facebook/opt-6.7b` | `facebook/opt-125m` |	summarization | 1.82x |	**2.71x** |
| `facebook/opt-6.7b` | `facebook/opt-125m` |	open-ended generation |	1.23x |	**1.59x** |
| `Salesforce/codegen-6B-mono` | `Salesforce/codegen-350M-mono` |	code generation (python) | 0.89x |	**1.09x** |
| `google/flan-t5-xl` | `google/flan-t5-small` | summarization |	1.18x |	**1.31x** |
| `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-1B` |	summarization |	1.00x |	**1.52x** |
| `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-1B` |	open-ended generation |	1.00x |	**1.18x** |
| `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-1B` |	code generation (python) |	1.09x |	**1.15x** |

* The results in the table reflect greedy decoding (temperature = 0). Similar trends were observed when using sampling (temperature > 0).

* All tests were conducted on an RTX 4090.

* Our benchmark is publicly available allowing everyone to evaluate further improvements: https://github.com/gante/huggingface-demos/tree/main/experiments/faster_generation
## Code

Dynamic speculation has been integrated into release [4.45.0](https://github.com/huggingface/transformers/releases/tag/v4.45.0) of the Hugging Face Transformers library and now serves as the default operation mode for assisted decoding. To use assisted generation with dynamic speculation, no code changes are required—just execute the code as you normally would:

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

outputs = model.generate(**inputs, assistant_model=assistant_model)
```

The default dynamic speculation lookahead parameters reflect optimal values but can be adjusted to improve performance for specific model pairs or datasets by using the following code:

```python
# confidence threshold
assistant_model.generation_config.assistant_confidence_threshold=0.4

# 'constant' means that num_assistant_tokens stays unchanged during generation
assistant_model.generation_config.num_assistant_tokens_schedule='constant'

# the maximum number of tokens generated by the assistant model.
# after 20 tokens the draft halts even if the confidence is above the threshold
assistant_model.generation_config.num_assistant_tokens=20
```

To revert to the **heuristic** or **constant** (as in [Leviathan et al.](https://arxiv.org/pdf/2211.17192)) approaches, simply set `num_assistant_tokens_schedule` to `'heuristic'` or `'constant'`, respectively, set `assistant_confidence_threshold=0` and `num_assistant_tokens=5` as follows:
```python
# Use 'heuristic' or 'constant' or 'dynamic'
assistant_model.generation_config.num_assistant_tokens_schedule='heuristic'
assistant_model.generation_config.assistant_confidence_threshold=0
assistant_model.generation_config.num_assistant_tokens=5
```

## What’s next?

We introduced a faster strategy for assisted generation called _dynamic speculative decoding_, which outperforms heuristics-based methods as well as drawing a constant number of candidate tokens.

In an upcoming blog post, we'll show a new method for assisted generation: combine any target model with any assistant model! This will open the door for accelerating countless models on the Hugging Face Hub that do not have small enough assistant variants. For example, `Phi 3`, `Gemma 2`, `CodeLlama` and many more will be eligible for speculative decoding. Stay tuned!

## References
- [Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models](https://arxiv.org/abs/2405.04304).
In this paper, we introduced DISCO, a dynamic speculation lookahead optimization method that utilizes a classifier to decide whether the draft model should proceed with generating the next token or pause, and switch to the target model for validation instead of using a simple threshold on the prediction probability.
- [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)
