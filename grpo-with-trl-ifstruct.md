---
title: "Fine-tuning a 350M Model for Better Structured Outputs in 100 GRPO Steps"
thumbnail: /blog/assets/grpo-with-trl-ifstruct/thumbnail.png
authors:
- user: burtenshaw
- user: sergiopaniego
- user: iamleonie
  guest: true
  org: LiquidAI
---

This guide is a fully public, inexpensive recipe for making a small model substantially better at structured-output compliance. We fine-tune [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) with Group Relative Policy Optimization (GRPO) using the [TRL library](https://huggingface.co/docs/trl/en/index) and evaluate it on the [IFStruct benchmark](https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0). The full run takes around 500 samples and 100 training steps, small enough for a free-tier Colab or Kaggle GPU, and is available on [GitHub](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/grpo_with_trl_ifstruct.ipynb). The results show that even a light fine-tuning procedure improves performance **from 22.6% to 29.7%** on the IFStruct benchmark.

Structured output is one of the most common real-world tasks for LLMs, yet most benchmarks fold it into broader reasoning or extraction scores rather than measuring it on its own. Whether a model reliably returns valid, parseable output in the requested format and shape — schema compliance — is often what decides whether it can be wired into a downstream system at all.

*Note that the training pipeline described here is not the one used to train the RL model described in the [IFStruct blog](https://www.liquid.ai/blog/ifstruct-v1.0). This notebook doesn't aim to recreate the IFStruct benchmark score, but to show how task-specific fine-tuning of smaller models can improve performance and match that of far larger models.*

## Prerequisites

This guide has two halves that run in different places:

- **Fine-tuning** runs on a GPU. The accompanying notebook is sized for a free-tier Colab or Kaggle GPU.  
- **Evaluation** can run locally on a MacBook (here, a MacBook Pro with an Apple M5 Max and 36 GB of unified memory) through `llama.cpp`, which exposes an OpenAI-compatible server that the IFStruct evaluator talks to.

We will need [`uv`](https://docs.astral.sh/uv/) for the Python tooling and `llama.cpp` for serving. Following the [Liquid AI llama.cpp deployment docs](https://docs.liquid.ai/deployment/on-device/llama-cpp), install `llama.cpp` with Homebrew and verify that `llama-server` is available:

```shell
brew install llama.cpp
llama-server --version
```

## IFStruct Evaluation on LFM2.5-350M (Base model)

Before we begin, let's evaluate LFM2.5-350M on the [IFStruct benchmark](https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0) and see whether we can **reproduce the reported score of 21.1**.

**IFStruct** is a benchmark for testing the validity of LLM outputs and schema adherence. The benchmark is open-source in [Liquid4All/ifstruct](https://github.com/Liquid4All/ifstruct), with the public benchmark dataset available on Hugging Face at [LiquidAI/ifstruct-v1.0](https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0).

```shell
git clone https://github.com/Liquid4All/ifstruct.git
```

For the eval comparison, we serve the model locally on the MacBook with `llama.cpp`. We will use the `BF16` GGUF ([LiquidAI/LFM2.5-350M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)).

Then we start the base-model server with the following command:

```shell
llama-server \
  -hf LiquidAI/LFM2.5-350M-GGUF:BF16 \
  -c 32768 \
  -np 4 \
  -ngl 99 \
  --alias LiquidAI/LFM2.5-350M \
  --host 127.0.0.1 \
  --port 8080
```

- `--alias`: model name IFStruct sends to the OpenAI-compatible endpoint  
- `-ngl 99`: asks `llama.cpp` to offload all layers to the GPU when available  
- `-np 4`: serves four requests in parallel  
- `-c 32768`: size of the prompt context

Once the server is running, we can run the full benchmark with 2000 samples:

```shell
uv run ifstruct-eval \
  --model LiquidAI/LFM2.5-350M \
  --base-url http://localhost:8080/v1 \
  --api-key dummy \
  --dataset data/test.jsonl \
  --results-file results/lfm2.5-350m-llamacpp-base.json \
  --n-threads 4 \
  --max-tokens 2048 \
  -v
```

```
============================================================
Model: LiquidAI/LFM2.5-350M
============================================================
Overall: 452/2000 passed (22.6%)
Average latency: 1453ms

By format:
  JSON: 180/1000 passed (18.0%)
  YAML: 272/1000 passed (27.2%)

By top-level structure:
  Wrapper key 288/1011 passed (28.5%)
  Bare list   164/989 passed (16.6%)

By entity type:
  test__camera_review                 6/83 passed (7.2%)
  test__clinical_trial                20/104 passed (19.2%)
  test__conference_schedule           7/87 passed (8.0%)
  test__escaping__bug_report_batch    24/89 passed (27.0%)
  test__escaping__config_snippet_audit 15/85 passed (17.6%)
  test__escaping__customer_email_thread 5/73 passed (6.8%)
  test__escaping__dialogue_sample     14/95 passed (14.7%)
  test__escaping__interview_transcript_segment 21/80 passed (26.2%)
  test__escaping__log_parser_examples 21/72 passed (29.2%)
  test__escaping__pr_discussion       22/87 passed (25.3%)
  test__escaping__repro_steps_batch   16/73 passed (21.9%)
  test__escaping__screenplay_scene    16/92 passed (17.4%)
  test__escaping__short_story_chapter 15/84 passed (17.9%)
  test__escaping__support_ticket_batch 27/73 passed (37.0%)
  test__escaping__terminal_session_notes 20/70 passed (28.6%)
  test__event_ticket_booking          49/107 passed (45.8%)
  test__gpu_review                    6/94 passed (6.4%)
  test__invoice                       28/86 passed (32.6%)
  test__job_posting                   25/85 passed (29.4%)
  test__real_estate_listing           31/82 passed (37.8%)
  test__recipe                        3/70 passed (4.3%)
  test__rental_car_booking            27/79 passed (34.2%)
  test__scientific_experiment         13/69 passed (18.8%)
  test__travel_itinerary              21/81 passed (25.9%)

Common errors:
  7228x required field missing
  738x wrong item count
  540x type mismatch
  317x Unclosed code block
  190x extraneous field 'notes'
  181x extraneous field 'path'
  175x extraneous field 'constraints'
  170x extraneous field 'type'
  170x missing code block
  100x expected bare list, got wrapper
```

The [IFStruct release blog reports 21.1% for LFM2.5-350M](https://www.liquid.ai/blog/ifstruct-v1.0). Our local llama.cpp/BF16 setup measures 22.6%, close to the 21.1% reported in the IFStruct blog. We use this local result as the baseline for the same serving stack comparison.

## GRPO Fine-tuning with TRL on Structured Outputs

The full, runnable pipeline lives in the [accompanying notebook](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/grpo_with_trl_ifstruct.ipynb). We will cover only the relevant pieces in this section.

### Training data

We use [`nvidia/Nemotron-RL-instruction_following-structured_outputs`](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs), which pairs each prompt with a target JSON Schema and an expected field count. We use about 500 samples for training.

Because the Nemotron data distribution differs from the IFStruct evaluation, we augment the prompts to close two gaps between them:

- **40%** get a "return the output inside a fenced code block" instruction appended, so the model learns to *follow* the format instruction rather than always emitting raw JSON.  
- A disjoint **20%** are converted into top-level-array tasks (the schema is wrapped in an `array` with a required item count), which trains bare-list output and item-count compliance.

### Model and LoRA

We load `LiquidAI/LFM2.5-350M` and attach a LoRA adapter. Because LFM2.5 uses a hybrid attention/convolution architecture, we target the LFM-specific module names:

```py
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    bias="none", 
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
        "w1", "w2", "w3",
    ],
)
```

This trains \~6M parameters, about 1.66% of the model.

### Reward functions

Then we define three reward functions, each on a `[0, 1]` scale, which score every completion on whether the extracted *structure* is correct:

- `json_format_reward`: Is the output parseable, and in the requested form? Full credit (`1.0`) for the requested form (fenced vs. raw), `0.2` for the wrong-but-parseable form, `0.0` for unparseable output.  
- `field_count_reward`: Does the object have the expected number of top-level fields? An exact match earns `1.0,` and the score decays linearly with the miss.  
- `schema_validation_reward`: Does the output validate against the row's JSON Schema? It counts every constraint violation and gates partial credit on required-key coverage.

We combine the three as a weighted sum with `reward_weights=[1.0, 0.5, 2.0]`.

### Training

We train for 100 steps with 8 generations per prompt group, sized for a free-tier 16 GB GPU:

```py
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="./outputs/lfm25-350m-nemotron-schema-grpo",
    learning_rate=5e-5,
    max_steps=100,
    warmup_steps=10,
    num_generations=8,              # completions sampled per prompt group
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # 4 prompt groups per optimizer step
    steps_per_generation=2,
    max_completion_length=1024,     # room for nested JSON
    mask_truncated_completions=False,
    temperature=1.1,                # hotter sampling keeps groups varied
    beta=0.01,                      # KL penalty toward the reference model
    reward_weights=[1.0, 0.5, 2.0], # json_format, field_count, schema_validation
    logging_steps=1,
    save_steps=100,
)
```

As you can see in the notebook, over the run, all three reward components climb, the KL from the reference model lifts off zero after warmup, and the truncated-completion fraction stays near zero .

### Merging and saving the model

Finally, we merge the LoRA adapter back into the base weights and save it as a single self-contained checkpoint, ready to convert to GGUF for serving:

```py
MERGED_DIR = f"{training_args.output_dir}-merged"

merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
```

## IFStruct Evaluation on GRPO Tuned LFM2.5-350M

After GRPO fine-tuning, we rerun the IFStruct evaluation. For this, we need to convert the merged model checkpoint into a BF16 GGUF. The converter script ships with the llama.cpp source, so we clone the repo once and install the converter's `gguf` package.

```shell
git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install ./llama.cpp/gguf-py

mkdir -p models
python llama.cpp/convert_hf_to_gguf.py \
  PATH_TO_YOUR_MERGED_MODEL \
  --outfile ./models/lfm25-350m-grpo-bf16.gguf \
  --outtype bf16
```

Then we serve the merged model with the following command:

```shell
llama-server \
  -m ./models/lfm25-350m-grpo-bf16.gguf \
  --alias lfm25-350m-grpo-structured-output \
  -c 32768 \
  -np 4 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8081
```

Then, we will run the full IFStruct evaluation again with the fine-tuned model:

```shell
uv run ifstruct-eval \
  --model lfm25-350m-grpo-structured-output \
  --base-url http://localhost:8081/v1 \
  --api-key dummy \
  --dataset data/test.jsonl \
  --results-file results/lfm25-350m-grpo.json \
  --n-threads 4 \
  --max-tokens 2048 \
  -v
```

```
============================================================
Model: lfm25-350m-grpo-structured-output
============================================================
Overall: 594/2000 passed (29.7%)
Average latency: 1518ms

By format:
  JSON: 319/1000 passed (31.9%)
  YAML: 275/1000 passed (27.5%)

By top-level structure:
  Wrapper key 300/1011 passed (29.7%)
  Bare list   294/989 passed (29.7%)

By entity type:
  test__camera_review                 5/83 passed (6.0%)
  test__clinical_trial                31/104 passed (29.8%)
  test__conference_schedule           11/87 passed (12.6%)
  test__escaping__bug_report_batch    32/89 passed (36.0%)
  test__escaping__config_snippet_audit 24/85 passed (28.2%)
  test__escaping__customer_email_thread 9/73 passed (12.3%)
  test__escaping__dialogue_sample     17/95 passed (17.9%)
  test__escaping__interview_transcript_segment 13/80 passed (16.2%)
  test__escaping__log_parser_examples 33/72 passed (45.8%)
  test__escaping__pr_discussion       26/87 passed (29.9%)
  test__escaping__repro_steps_batch   23/73 passed (31.5%)
  test__escaping__screenplay_scene    34/92 passed (37.0%)
  test__escaping__short_story_chapter 24/84 passed (28.6%)
  test__escaping__support_ticket_batch 36/73 passed (49.3%)
  test__escaping__terminal_session_notes 23/70 passed (32.9%)
  test__event_ticket_booking          62/107 passed (57.9%)
  test__gpu_review                    7/94 passed (7.4%)
  test__invoice                       36/86 passed (41.9%)
  test__job_posting                   33/85 passed (38.8%)
  test__real_estate_listing           32/82 passed (39.0%)
  test__recipe                        7/70 passed (10.0%)
  test__rental_car_booking            37/79 passed (46.8%)
  test__scientific_experiment         14/69 passed (20.3%)
  test__travel_itinerary              25/81 passed (30.9%)

Common errors:
  7331x required field missing
  890x wrong item count
  555x type mismatch
  102x expected bare list, got wrapper
   62x extraneous field 'metadata.tone'
   55x 6 is greater than maximum 5
   49x extraneous field 'speaker_labels'
   47x extraneous field 'tone'
   44x 'cups' not in allowed values ['mg', 'g', 'kg', 'oz', 'lb', 'ml', 'l', 'cl', 'dl'
   44x extraneous field 'notes'
```

Comparing the two runs on the identical serving stack:

| IFStruct group | base | GRPO-tuned | Δ |
| :---- | :---- | :---- | :---- |
| **Overall** | 22.6% | **29.7%** | **\+7.1** |
| JSON | 18.0% | 31.9% | \+13.9 |
| YAML | 27.2% | 27.5% | \+0.3 |
| Wrapper key | 28.5% | 29.7% | \+1.2 |
| Bare list | 16.6% | 29.7% | \+13.1 |

The gains land exactly where the training aimed: the JSON pass rate rises by nearly 14 points (18.0% → 31.9%), while YAML stays mostly the same. While this is still below the [Qwen3.5-2B score of 33.15%](https://www.liquid.ai/blog/ifstruct-v1.0), it shows that even light task-specific fine-tuning can bring a small model close to a larger one.

## Conclusion

A short GRPO run with about 500 samples and 100 steps can lift a small 350M parameter model from 22.6% to 29.7% on IFStruct. The takeaway is that a cheap, task-specific reward signal can make a small model substantially more reliable about *form*, closing much of the gap to models several times its size.

To reproduce or extend this work, see the original [IFStruct v1.0 blog post](https://www.liquid.ai/blog/ifstruct-v1.0), the [Liquid4All/ifstruct](https://github.com/Liquid4All/ifstruct) benchmark repo, and the [LiquidAI/ifstruct-v1.0](https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0) dataset.  