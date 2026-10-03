---
title: "My QLoRA Fine-Tune Made Gemma 3 4B Worse on Math. Here's Why That's Useful."
thumbnail: /blog/assets/gemma3-opus-distill/thumbnail.jpg
authors:
  - user: Viesar
    guest: true
---

# My QLoRA Fine-Tune Made Gemma 3 4B Worse on Math. Here's Why That's Useful.

I spent a weekend distilling Claude Opus 4.6 reasoning traces into Gemma 3 4B on a single RTX 3050. The model trained cleanly. Loss dropped from 2.67 to 1.01 in 50 minutes. Qualitative outputs looked great — clean `<think>` blocks, structured math reasoning, the works.

Then I benchmarked it.

| Benchmark | Base Gemma 3 4B | My Fine-tune | Δ |
|-----------|-----------------|--------------|------|
| MATH-500 (`exact_match`) | 29.6% ± 2.0% | 24.6% ± 1.9% | **−5.0pp** |
| GSM8K (`exact_match`, strict) | 68.7% ± 2.7% | 53.7% ± 2.9% | **−15.0pp** |

Both regressions are statistically significant. The confidence intervals don't overlap. My carefully distilled reasoning model is measurably *worse* at math than the base it was built on.

This post is about why that happened, what it teaches us about resource-constrained fine-tuning, and why "negative results, published openly" is more useful than the usual "I trained a model and look how good it is" writeup.

## The Setup

I wanted to see if distilling reasoning traces from a frontier model into a small open one would meaningfully improve the small model's reasoning. The pieces:

- **Base model:** [`unsloth/gemma-3-4b-it-unsloth-bnb-4bit`](https://huggingface.co/unsloth/gemma-3-4b-it-unsloth-bnb-4bit) — Gemma 3 4B Instruct, pre-quantized to 4-bit by the Unsloth team
- **Dataset:** [`Crownelius/Opus-4.6-Reasoning-3300x`](https://huggingface.co/datasets/Crownelius/Opus-4.6-Reasoning-3300x) — 2,160 rows of `<problem, thinking, solution>` triples distilled from Claude Opus 4.6, mostly math (94%)
- **Method:** QLoRA — train LoRA adapters on top of a frozen 4-bit base
- **Hardware:** NVIDIA RTX 3050 (8 GB VRAM), 48 GB system RAM

The whole thing runs in Unsloth's official Docker image, so dependency resolution is handled. If you've tried Python-native ML installs on Windows, you know why this matters.

## Pre-Filtering the Dataset

Spot-checking the dataset turned up a clear problem: ~10% of "solutions" were refusals or "please provide more information" responses where the original prompt was incomplete. Training on these would teach the model to give up.

A simple filter caught most of them:

```python
REFUSAL_PHRASES = [
    "i cannot", "i can't", "i'm unable", "i am unable",
    "incomplete", "truncated", "please provide",
    "haven't actually included", "i notice you've provided",
]

def is_quality(row):
    if len(row["solution"]) < 200 or len(row["thinking"]) < 100:
        return False
    if any(p in row["solution"].lower() for p in REFUSAL_PHRASES):
        return False
    return True
```

This dropped the dataset from 2,160 to ~1,900 examples. Worth doing.

## The Tokenizer Trap That Cost Me an Hour

Gemma 3 is multimodal. Its tokenizer isn't a regular `GemmaTokenizerFast` — it's a `Gemma3Processor` that wraps the text tokenizer to handle image and video inputs.

This wrecks naive training pipelines. When you call the processor on a string:

```python
tokenizer(text, return_token_type_ids=False)
# → {'input_ids': [[2, 9259, 236764, ...]], ...}
```

The result is **nested** — a list containing one list of tokens, formatted as if you'd passed a batch. The data collator gets confused, padding fails, training crashes with cryptic shape mismatches.

The fix: don't tokenize through the processor. Reach inside it:

```python
text_tokenizer = tokenizer.tokenizer  # the inner GemmaTokenizerFast

# This returns flat lists, like a normal tokenizer
encoded = text_tokenizer(text, return_token_type_ids=False)
```

Use the processor for `apply_chat_template` (it knows the multimodal chat format). Use the inner text tokenizer for actual tokenization. This single change unblocks the whole pipeline.

If you're fine-tuning Gemma 3, this will save you hours.

## Training

The actual training script is straightforward — Unsloth's `FastLanguageModel` + standard TRL `SFTTrainer` with a custom collator that handles right-padding with proper label masking:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)
```

Each example formatted as a two-turn conversation, with the `<think>` block embedded in the assistant turn:

```python
messages = [
    {"role": "user", "content": example["problem"]},
    {"role": "assistant",
     "content": f"<think>\n{example['thinking']}\n</think>\n\n{example['solution']}"},
]
```

Hyperparameters that mattered:

| Parameter | Value |
|-----------|-------|
| Effective batch size | 4 (1 × 4 grad accumulation) |
| Learning rate | 2e-4 (linear schedule with 20 warmup steps) |
| Epochs | 1 |
| Max sequence length | 4096 |
| Optimizer | adamw_8bit |
| Compute dtype | bfloat16 |

Training ran for 471 steps in 50.6 minutes. Peak VRAM was 7.54 GB out of 8 GB — tight, but it fit. Loss curve was clean: 2.67 → 1.62 → 1.01.

## The Qualitative Test (Looked Great)

Before benchmarks, I did the obvious thing: ask the model a math question and look at the output.

> **Q:** *If a train travels 240 km in 3 hours, what is its average speed in m/s?*

> **A:**
> ```
> <think>
> Let me work through this problem step by step.
> First, I need to identify what's being asked...
> Key values: 240, 3
> My approach:
> 1. Convert kilometers to meters
> 2. Convert hours to seconds
> 3. Calculate the average speed
> 4. Verify the result
> </think>
>
> # Solution: Average Speed of a Train
>
> ## Step 1: Convert Kilometers to Meters
> 240 km × 1000 m/km = 240,000 m
>
> ## Step 2: Convert Hours to Seconds
> 3 hours × 3600 s/hour = 10,800 s
>
> ## Step 3: Calculate Speed
> Speed = 240,000 m / 10,800 s ≈ 22.22 m/s
> ```

This looks fantastic. Clean `<think>` block, step-by-step decomposition, correct math, verification step at the end. The Opus reasoning style is clearly imitated.

So I converted to GGUF (Q4_K_M, ~2.4 GB), pushed to Hugging Face, and started writing a celebratory README.

Then I ran benchmarks.

## The Numbers

I evaluated both my fine-tune and the base on two standard math reasoning benchmarks using lm-evaluation-harness 0.4.11. Both runs used identical conditions: 4-bit quantization, bfloat16 compute, batch size 1, deterministic generation (`do_sample=False`, `temperature=0.0`).

### MATH-500 (full test set, 4-shot)

| Metric | Base | Fine-tune | Δ |
|--------|------|-----------|------|
| `exact_match` | 29.6% ± 2.0% | 24.6% ± 1.9% | −5.0pp |
| `math_verify` | 36.4% ± 2.2% | 29.6% ± 2.0% | −6.8pp |

### GSM8K (300-problem subset, 5-shot)

| Metric | Base | Fine-tune | Δ |
|--------|------|-----------|------|
| `exact_match` (strict) | 68.7% ± 2.7% | 53.7% ± 2.9% | −15.0pp |
| `exact_match` (flexible) | 69.0% ± 2.7% | 55.0% ± 2.9% | −14.0pp |

Two benchmarks. Same direction. Confidence intervals don't overlap. The fine-tune is genuinely worse at completion-style math evaluation.

The performance numbers, for completeness:

| Metric | Base | Fine-tune |
|--------|------|-----------|
| Per-problem time (MATH-500) | 16.5 s | 16.6 s |
| Per-problem time (GSM8K) | 11.0 s | ~11 s |
| Peak VRAM | ~6.2 GB | ~6.2 GB |

Inference cost is identical. The model isn't slower or larger — it just gets fewer answers right.

## Postmortem: Why This Happened

There are two effects I think are contributing, and the relative size of the regressions tells me both are present.

### 1. Evaluation format mismatch

Both MATH-500 and GSM8K in lm-evaluation-harness use **few-shot completion-style prompting**:
The model is supposed to continue the pattern: read a `Problem:`, output the `Solution:`, end with `#### N` or `\boxed{N}` so the regex extractor can find the answer.

My fine-tune was trained on a different format entirely. Two-turn conversations, with `<think>...</think>` blocks before answers. So when it sees the few-shot completion prompt, it's torn between:

- The in-context examples showing direct `Solution:` patterns
- Its trained instinct to produce thinking blocks first

Sometimes it generates a long reasoning trace before reaching the boxed answer, and the `until` token (`Problem:` or `Question:`) cuts off the response before the answer extractor finds the marker. The reasoning is correct; the format is wrong.

This is well-documented for reasoning-distilled models. They tend to do better on chat-style evals (where their format is honored) and worse on completion-style benchmarks.

### 2. Capability narrowing

The format mismatch alone doesn't explain a 15-point GSM8K drop. Some capability genuinely got narrowed.

Training on 1,900 narrow examples — 94% math, single epoch, full 7-projection LoRA at rank 16 — pushed the model toward that distribution. The trained reasoning style improved. The model's general flexibility for following arbitrary in-context patterns degraded.

This is the cost of QLoRA on small datasets: you trade breadth for narrow style. There's no free lunch where you teach a model new tricks without disturbing its old ones.

## What I Should Have Done Differently

Hindsight makes the recipe more obvious:

**Smaller learning rate.** 2e-4 with linear decay is standard QLoRA but probably too aggressive for a small dataset. 5e-5 to 1e-4 would have been gentler.

**Mix in general data.** A 50/50 blend of the Opus reasoning traces with general instruction-following data would help the model retain its in-context learning ability. Even 20% general data would probably move the needle.

**Lower LoRA rank.** r=16 might be too much capacity for 1,900 examples. r=4 or r=8 would have made the change less drastic.

**Evaluate with chat templates.** lm-evaluation-harness supports `--apply_chat_template` for some tasks. Using it would let the fine-tune leverage its trained format. This is the actually-rigorous follow-up I haven't run yet.

**Multiple seeds.** All my numbers come from a single training run. Running 3-5 with different seeds would tell me whether the regression is reliable or just bad luck.

## What's Still True

The negative result on benchmarks doesn't mean the project failed. The model works — it just works in a different mode than the benchmarks measure.

- The trained `<think>` reasoning style is real and visible. It produces clean structured outputs in chat-style use.
- Inference cost is identical to the base. Nothing got slower.
- The whole pipeline is documented and reproducible. Someone reading this can repeat the experiment with one less hour of debugging.
- The model is genuinely useful for chat-style reasoning tasks where the format is honored.

The model is published openly: [GGUF version](https://huggingface.co/Viesar/gemma-3-4b-opus-reasoning-distill-GGUF) (Q4_K_M, ~2.4 GB) and [merged FP16 version](https://huggingface.co/Viesar/gemma-3-4b-opus-reasoning-distill). Both READMEs include the full benchmark numbers and honest interpretation. Anyone can verify.

## Why Publish Negative Results

A note on this — because there's a real cultural problem in ML writeups.

Most fine-tune blog posts go: *"I trained a model. It's amazing. Look at this curated example."* They don't measure against the base under identical conditions. They don't report regressions. They don't explain when their thing helps and when it hurts.

That's not malicious — it's the ambient incentive structure. Posts about successful results get retweeted. Posts about regressions get less attention. So we end up in a world where everyone's fine-tunes are great and nobody can reproduce them.

I think the antidote is: measure honestly, publish what you find, and trust that readers will respect rigor more than spectacle.

This experiment cost me a weekend and ~$2 of electricity. The model is up, the numbers are up, the code is up. Maybe someone reads this and saves themselves the same mistakes. Maybe someone takes the recipe and makes it better. That's a more useful outcome than a fake "everything went perfectly" post.

## What I'd Build Next

If I were continuing this:

1. **Re-evaluate with chat templates** to test the format-mismatch hypothesis directly. Same model, different prompt style — does the gap close?
2. **Ablate the dataset blend** — train versions with 0%, 25%, 50% general data mixed in. Find the breadth/narrowness tradeoff.
3. **Try a lower rank** — does r=4 produce a smaller capability narrowing while still teaching the reasoning style?
4. **Test on chat-style benchmarks** — IFEval or MT-Bench instead of MATH-500. The model's actual deployment context.

If any of those would be useful as a follow-up post, let me know.

---

## Resources

- **GGUF model (for Ollama, llama.cpp):** [Viesar/gemma-3-4b-opus-reasoning-distill-GGUF](https://huggingface.co/Viesar/gemma-3-4b-opus-reasoning-distill-GGUF)
- **Merged FP16 model (for transformers, further training):** [Viesar/gemma-3-4b-opus-reasoning-distill](https://huggingface.co/Viesar/gemma-3-4b-opus-reasoning-distill)
- **Training dataset:** [Crownelius/Opus-4.6-Reasoning-3300x](https://huggingface.co/datasets/Crownelius/Opus-4.6-Reasoning-3300x)
- **Base model:** [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- **Training framework:** [Unsloth](https://github.com/unslothai/unsloth)
- **Evaluation:** [lm-evaluation-harness 0.4.11](https://github.com/EleutherAI/lm-evaluation-harness)

Released under the [Gemma License](https://ai.google.dev/gemma/terms).
