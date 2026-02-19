---
title: "Train AI models with Unsloth and Hugging Face Jobs for FREE"
thumbnail: /blog/assets/unsloth-jobs/thumbnail.png
authors:
- user: burtenshaw
- user: danielhanchen
  guest: true
  org: unsloth
- user: shimmyshimmer
  guest: true
  org: unsloth
- user: mlabonne
  guest: true
  org: LiquidAI
- user: davanstrien
- user: evalstate
---

# Train AI models with Unsloth and Hugging Face Jobs for FREE

This blog post covers how to use [Unsloth](https://github.com/unslothai/unsloth) and [`LiquidAI/LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) for fast LLM fine-tuning through coding agents like Claude Code and Codex. Unsloth provides ~2x faster training and ~60% less VRAM usage compared to standard methods, so training small models can cost just a few dollars.

<figure class="image flex flex-col items-center text-center m-0 w-full">
<video controls width="100%" height="auto">
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/unsloth-jobs/unsloth-jobs.mp4" type="video/mp4" />
</video>
</figure>

## You will need

We are giving away free credits to fine-tune models on Hugging Face Jobs. Join the [Unsloth Jobs Explorers](https://huggingface.co/unsloth-jobs) organization to claim your free credits and one-month Pro subscription.

- A [Hugging Face](https://huggingface.co) account (required for HF Jobs) 
- Billing setup (but you won't need to pay anything).
- A Hugging Face token with write permissions
- (optional) A coding agent (`Open Code`, `Claude Code`, or `Codex`)

## Run the Job

If you want to train a model using HF Jobs and Unsloth, you do not need a coding agent. You can simply use the `hf jobs` CLI to submit a job.

First, you need to install the `hf` CLI. You can do this by running the following command:

```
# mac or linux
curl -LsSf https://hf.co/cli/install.sh | bash
```

Next you can run the following command to submit a job:

```text
hf jobs uv run https://huggingface.co/datasets/unsloth/jobs/resolve/main/sft-lfm2.5.py \
    --flavor a10g-small --secrets HF_TOKEN --timeout 4h \
    -- --dataset mlabonne/FineTome-100k \
        --num-epochs 1 \
        --eval-split 0.2 \
        --output-repo your-username/lfm-finetuned
```

Check out the [training script](https://huggingface.co/datasets/unsloth/jobs/blob/main/sft-lfm2.5.py) and [Hugging Face Jobs documentation](https://huggingface.co/docs/hub/jobs) for more details.

## Installing the Skill

First, install the skill with your coding agent.

### Claude Code

Claude Code discovers skills through its [plugin system](https://code.claude.com/docs/en/discover-plugins).

1. Add the marketplace:

```text
/plugin marketplace add huggingface/skills
```

2. Browse available skills in the `Discover` tab:

```text
/plugin
```

3. Install the model trainer skill:

```text
/plugin install hugging-face-model-trainer@huggingface-skills
```

For more details, see the [Claude Code plugins docs](https://code.claude.com/docs/en/discover-plugins) and the [Skills docs](https://code.claude.com/docs/en/skills).

### Codex

Codex discovers skills through [`AGENTS.md`](https://developers.openai.com/codex/guides/agents-md) files and [`.agents/skills/`](https://developers.openai.com/codex/skills) directories.

Install individual skills with `$skill-installer`:

```text
$skill-installer install https://github.com/huggingface/skills/tree/main/skills/hugging-face-model-trainer
```

For more details, see the [Codex Skills docs](https://developers.openai.com/codex/skills) and the [AGENTS.md guide](https://developers.openai.com/codex/guides/agents-md).

### Anything else

A generic install method is simply to clone the [skills repository](https://github.com/huggingface/skills) and copy the [skill](https://github.com/huggingface/skills/tree/main/skills/hugging-face-model-trainer) to your agent's skills directory.

```text
git clone https://github.com/huggingface/skills.git
mkdir -p ~/.agents/skills && cp -R skills/skills/hugging-face-model-trainer ~/.agents/skills/
```

## Quick Start

Once the skill is installed, ask your coding agent to train a model:

```text
Train LiquidAI/LFM2.5-1.2B-Instruct on mlabonne/FineTome-100k using Unsloth on HF Jobs
```

The agent will generate a training script based on an [example in the skill](https://github.com/huggingface/skills/blob/main/skills/hugging-face-model-trainer/scripts/unsloth_sft_example.py), submit the training to HF Jobs, and provide a monitoring link via Trackio.

## How It Works

Training jobs run on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), fully managed cloud GPUs. The agent:

1. Generates a UV script with inline dependencies
2. Submits it to HF Jobs via the `hf` CLI
3. Reports the job ID and monitoring URL
4. Pushes the trained model to your Hugging Face Hub repository

### Example Training Script

The skill generates scripts like this based on the example in the [skill](https://github.com/huggingface/skills/blob/main/skills/hugging-face-model-trainer/scripts/unsloth_sft_example.py).

```python
# /// script
# dependencies = ["unsloth", "trl>=0.12.0", "datasets", "trackio"]
# ///

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    load_in_4bit=True,
    max_seq_length=2048,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./output",
        push_to_hub=True,
        hub_model_id="username/my-model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        report_to="trackio",
    ),
)

trainer.train()
trainer.push_to_hub()
```

| Model Size | Recommended GPU | Approx Cost/hr |
| :-- | :-- | :-- |
| <1B params | `t4-small` | ~$0.40 |
| 1-3B params | `t4-medium` | ~$0.60 |
| 3-7B params | `a10g-small` | ~$1.00 |
| 7-13B params | `a10g-large` | ~$3.00 |

For a full overview of Hugging Face Spaces pricing, check out the guide [here](https://huggingface.co/docs/hub/en/spaces-overview#hardware-resources).

## Tips for Working with Coding Agents

- Be specific about the model and dataset to use, and include Hub IDs (for example, `Qwen/Qwen2.5-0.5B` and `trl-lib/Capybara`). Agents will search for and validate those combinations.
- Mention Unsloth explicitly if you want it used. Otherwise, the agent will choose a framework based on the model and budget.
- Ask for cost estimates before launching large jobs.
- Request Trackio monitoring for real-time loss curves.
- Check job status by asking the agent to inspect logs after submission.

## Resources

- [Hugging Face Skills Repository](https://github.com/huggingface/skills)
- [Free credits for Unsloth Jobs Explorers](https://huggingface.co/unsloth-jobs)
- [Unsloth Tutorial on Hugging Face Jobs](https://unsloth.ai/docs/basics/inference-and-deployment/deploying-with-hugging-face-jobs)
- [Example Unsloth Jobs scripts](https://huggingface.co/datasets/unsloth/jobs)