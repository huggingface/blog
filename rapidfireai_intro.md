---
title: "20x Faster TRL Fine-tuning with RapidFire AI"
thumbnail: /blog/assets/rapidfireai_intro/thumbnail.png
authors:
  - user: kbigdelysh
    guest: true
    org: rapidfire-ai-inc
  - user: arunkk09
    guest: true
    org: rapidfire-ai-inc
  - user: qgallouedec
---

# 20x Faster TRL Fine-tuning with RapidFire AI

Hugging Face TRL now officially integrates with RapidFire AI to accelerate your fine-tuning and post-training experiments. TRL users can now discover, install, and run RapidFire AI as the fastest way to compare multiple fine-tuning/post-training configurations to customize LLMs without major code changes and without bloating GPU requirements.

## Why this matters

When fine-tuning or post-training LLMs, teams often do not have the time and/or budget to compare multiple configs even though that can significantly boost eval metrics. RapidFire AI lets you launch multiple TRL configs concurrently--even on a single GPU--and compare them in near real time via a new adaptive, chunk-based scheduling and execution scheme. In internal benchmarks referenced in the TRL page, this delivers ~16–24× higher experimentation throughput than sequentially comparing configs one after another, enabling you to reach much better metrics much faster.

![RapidFire AI Architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rapidfireai_intro/rf-usage.png)
*RapidFire AI establishes live three-way communication between your IDE, a metrics dashboard, and a multi-GPU execution backend*

## What you get, out of the box

- **Drop-in TRL wrappers** — Use `RFSFTConfig`, `RFDPOConfig`, and `RFGRPOConfig` as near-zero-code replacements for TRL's SFT/DPO/GRPO configs.

- **Adaptive chunk-based concurrent training** — RapidFire AI shards the dataset into a given number of chunks and cycles configs at chunk boundaries to enable earlier apples-to-apples comparisons and also maximize GPU utilization.

- **Interactive Control Ops (IC Ops)** — From the dashboard itself, you can Stop, Resume, Delete, and Clone-Modify, possibly with Warm-Start, any runs in flight to avoid wasting resources on underperforming configs and double-down on better performing configs--no job restarts, no juggling separate GPUs or clusters, no resource bloat.

![Interactive Control Operations](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rapidfireai_intro/icop-clone.png)
*Clone promising configurations with modified hyperparameters, optionally warm-starting from the parent's weights, all from the live dashboard*

- **Multi-GPU orchestration** — The RapidFire AI scheduler automatically places and orchestrates configs across available GPUs on chunks of data via effcient shared-memory mechanisms. You focus on your models and eval metrics, not plumbing.

- **MLflow-based dashboard** — Real-time metrics, logs, and IC Ops in one place as soon as you start your experiment. Support for more dashboards such as Trackio, W&B, and TensorBoard coming soon.

## How it works

RapidFire AI splits your dataset randomly into "chunks" and cycles LLM configurations through the GPUs at chunk boundaries. You get incremental signal on eval metrics across all configs much more quickly. The automatic checkpointing via an efficient shared-memory-based adapter/model spilling/loading mechanism keeps training smooth, stable, and consistent. Use IC Ops to adapt mid-flight to stop low-performers earlier and clone promising ones with tweaked config knobs, optionally warm-starting from the parent's weights.

![GPU Scheduling Comparison](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rapidfireai_intro/gantt-2gpu.png)
*Sequential vs. Task Parallel vs. RapidFire AI: The adaptive scheduler maximizes GPU utilization across multiple configs and GPUs. The bottom row shows IC Ops in action—stopping, cloning, and modifying runs mid-flight.*

## Getting Started

Install RapidFire AI and get running in under a minute:

```bash
pip install rapidfireai

# Authenticate with Hugging Face
huggingface-cli login --token YOUR_TOKEN

# Workaround for current issue
pip uninstall -y hf-xet

# Initialize and start RapidFire AI
rapidfireai init
rapidfireai start
```

The dashboard launches at `http://localhost:3000` where you can monitor and control all your experiments.

## Supported TRL trainers

- SFT with `RFSFTConfig`
- DPO with `RFDPOConfig`
- GRPO with `RFGRPOConfig`

These are designed as drop-in replacements so that you can keep your TRL mental model while gaining far more concurrency and control for your fine-tuning/post-training applications. 

## Minimal TRL SFT example

Here's what it looks like to train **multiple configurations concurrently** even on a single GPU:

```python
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup: load your dataset and define formatting
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
train_dataset = dataset["train"].select(range(128)).shuffle(seed=42)

def formatting_function(row):
    return {
        "prompt": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": row["instruction"]},
        ],
        "completion": [{"role": "assistant", "content": row["response"]}]
    }

dataset = dataset.map(formatting_function)

# Define multiple configs to compare
config_set = List([
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        peft_config=RFLoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
        training_args=RFSFTConfig(learning_rate=1e-3, max_steps=128, fp16=True),
    ),
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        peft_config=RFLoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"]),
        training_args=RFSFTConfig(learning_rate=1e-4, max_steps=128, fp16=True),
        formatting_func=formatting_function,
    )
])

# Run all configs concurrently with chunk-based scheduling
experiment = Experiment(experiment_name="sft-comparison")
config_group = RFGridSearch(configs=config_set, trainer_type="SFT")

def create_model(model_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"], 
        device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    return (model, tokenizer)

experiment.run_fit(config_group, create_model, train_dataset, num_chunks=4, seed=42)
experiment.end()
```

**What happens when you run this?**

Suppose you run the above on a 2-GPU machine. Instead of training sequentially (Config 1 → wait → Config 2 → wait), both configs train concurrently:

| Approach | Time till Comparative Decision | GPU utilization |
|----------|-----------------|-----------------|
| Sequential (traditional) | ~15 minutes | 60% utilization |
| RapidFire AI (concurrent) | ~5 minutes | 95%+ utilization |

You can get to a comparative decision **3× sooner** on the same resources after both configs finish processing the first data chunk instead of waiting for them to see the whole dataset one after another. Open `http://localhost:3000` to watch live metrics and use IC Ops to stop, clone, or tweak runs in real-time based on what you're seeing.

## Benchmarks: Real-World Speedups

Here is what teams see on time to reach a comparable overall best training loss (across all tried configs) when switching from sequential comparisons to RapidFire AI-enabled hyperparallel experimentation:

| Scenario | Sequential Time | RapidFire AI Time | Speedup |
|----------|----------------|-------------------|---------|
| 4 configs, 1 GPU | 120 min | 7.5 min | **16×** |
| 8 configs, 1 GPU | 240 min | 12 min | **20×** |
| 4 configs, 2 GPUs | 60 min | 4 min | **15×** |

*Benchmarks on NVIDIA A100 40GB with TinyLlama-1.1B and Llama-3.2-1B models*

## Get Started Today

**🚀 Try it hands-on**: [Interactive Colab Notebook](http://tinyurl.com/rapidfireai-colab) — Zero setup, runs in your browser

**📚 Full Documentation**: [oss-docs.rapidfire.ai](https://oss-docs.rapidfire.ai) — Complete guides, examples, and API reference

**💻 GitHub**: [RapidFireAI/rapidfireai](https://github.com/RapidFireAI/rapidfireai) — Open source, production-ready

**📦 Install via PyPI**: [pypi.org/project/rapidfireai](https://pypi.org/project/rapidfireai) — `pip install rapidfireai`

**💬 Join the Community**: [Discord](https://discord.gg/6vSTtncKNN) — Get help, share results, request features

---

RapidFire AI was built because the common status quo of trying one config at a time wastes both time and GPU cycles. With this official integration, every TRL user can fine-tune/post-train smarter, iterate faster, and ship better models.

**Try the integration and let us know**: How much faster is your experimentation loop? What should we build next? We're just getting started, and your feedback shapes where we go from here.
