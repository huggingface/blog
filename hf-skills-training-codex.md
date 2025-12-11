---
title: "Codex is Open Sourcing AI models"
thumbnail: /blog/assets/hf-skills-training/thumbnail-codex.png
authors:
- user: burtenshaw
- user: evalstate
---

# Codex is Open Sourcing AI models

![banner](https://huggingface.co/blog/assets/hf-skills-training/thumbnail-codex.png)

Building on our work to get [Claude Code](https://huggingface.co/blog/hf-skills-training) to train open source models, we are now getting [Codex](https://developers.openai.com/codex/) to go further. We gave Codex access to the [Hugging Face Skills](https://github.com/huggingface/skills) repository, which contains skills for Machine Learning and AI tasks such as training or evaluating models. With HF skills, a coding agent can:

- Fine-tune and apply RL alignment on language models
- Review, explain, and act on live training metrics from Trackio
- Evaluate checkpoints and act on evaluation results
- Create reports from experiments
- Export to and quantize models with GGUF for local deployment
- Publish models to the Hub

This tutorial dives even deeper and shows you how it works and how to use it yourself. So let's get started. 

> [!NOTE]
> Codex uses `AGENTS.md` files to accomplish specialized tasks, whilst Claude Code uses 'Skills'. Fortunately, 'HF-skills' is compatible with both approaches and works with major coding agents like Claude Code, Codex, or Gemini CLI.

With `HF-skills`, you can tell Codex something like:

```
Fine-tune Qwen3-0.6B on the dataset open-r1/codeforces-cots
```

And Codex will:

1. Validate your dataset format
2. Select appropriate hardware (t4-small for a 0.6B model)
3. Use and update a training script with Trackio monitoring
4. Submit the job to Hugging Face Jobs
5. Report the job ID and estimated cost
6. Check on progress when you ask
7. Help you debug if something goes wrong

The model trains on Hugging Face GPUs while you do other things. When it's done, your fine-tuned model appears on the Hub, ready to use.

This isn't a toy demo. The extension supports the same training methods used in production: supervised fine-tuning, direct preference optimization, and reinforcement learning with verifiable rewards. You can train models from 0.5B to 7B parameters, convert them to GGUF for local deployment, and run multi-stage pipelines that combine different techniques.

## GOAL: End-to-end Machine Learning experiments

We explored this single prompt approach in the Claude Code tutorial. However, we can now go further and get OpenAI Codex to do end-to-end Machine Learning experiments. For example, Codex should be able to monitor progress, evaluate the models, and maintain an up to date training report. This will allow engineers to delegate experiments to Codex and review reports in a more hands-off way. It will also allow Codex to make more decisions on its own based on the training report and evaluation results.

So let's get started!

## Setup and Install

Before starting, you'll need:

- A Hugging Face account with a [Pro](https://hf.co/pro) or [Team / Enterprise](https://hf.co/enterprise) plan (Jobs require a paid plan)
- A write-access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- [Codex](https://developers.openai.com/codex/) installed and configured

### Install Codex

Codex is OpenAI's AI coding agent included in ChatGPT Plus, Pro, Business, Edu, and Enterprise plans. Codex brings AI assistance directly into your development workflow.

See the [Codex documentation](https://developers.openai.com/codex/) for installation and setup instructions.

### Install the Hugging Face Skills

The Hugging Face Skills repository includes an `AGENTS.md` file that Codex automatically detects and uses.

Clone the repository:

```bash
git clone https://github.com/huggingface/skills.git
cd skills
```

Codex will automatically detect the `AGENTS.md` file in the repository and load the skills. You can verify the instructions are loaded with:

```bash
codex --ask-for-approval never "Summarize the current instructions."
```

See the [Codex AGENTS guide](https://developers.openai.com/codex/) for more details.

### Connect to Hugging Face

Authenticate with Hugging Face using the `hf auth login` command and a write-access token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens):

```bash
hf auth login
```


Codex supports [MCP (Model Context Protocol)](https://developers.openai.com/codex/) servers. You can configure the Hugging Face MCP server for additional Hub integration capabilities. You can add the Hugging Face MCP server to your Codex configuration by adding the following to your `~/.codex/config.toml` file:

```toml
[mcp_servers.huggingface]
command = "npx"
args = ["-y", "mcp-remote", "https://huggingface.co/mcp?login"]
```

Configure Hugging Face MCP Server to use relevant MCP servers like Jobs in the [Settings](https://huggingface.co/settings/mcp) page.

Then start Codex and you'll be directed to the Hugging Face MCP authentication page.

## Your first AI Experiment

Let's walk through a complete example. We'll fine-tune a small model to improve code solving abilities, using the [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) dataset and the [openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) benchmark.

### Instruct Codex to do an end-to-end fine-tuning experiment

Start Codex in your project directory. Then give it a simple and clear instruction:

```
Start a new fine-tuning experiment to improve code solving abilities on using SFT. 
- Maintain a report for the experiment. 
- Evaluate models with the openai_humaneval benchmark
- Use the open-r1/codeforces-cots dataset
```

> [!TIP]
> You'll notice that we've gone a bit further than the single prompt approach in the Claude Code tutorial. We've added more details to the instruction but also added more steps to the experiment. 
>
> Why not try iterating on this experiment yourself with more open ended questions like "What is the best model for code solving abilities?" or "What is the best dataset for code solving abilities?"

Codex analyzes your request and prepares a training configuration. For a 0.6B model on a demo dataset, it selects `t4-small`—enough GPU for this model size and the cheapest option available. Codex will start a new report at `training_reports/<model>-<dataset>-<method>.md` which looks like the example below. As the experiment progresses, Codex will update the report with the latest information and each run report.

<details>
<summary>Example Training Report</summary>

```md
# Base Model & Dataset
[Base Model](https://huggingface.co/Qwen/Qwen3-0.6B)  
[Dataset](https://huggingface.co/datasets/open-r1/codeforces-cots)

---

# `sft-a10g` - `TBD` - `In Progress`

## Training Parameters
| Parameter | Value |
|-----------|-------|
| Method | SFT (TRL) |
| Model | `Qwen/Qwen3-0.6B` |
| Dataset | `open-r1/codeforces-cots` (train, 5% eval split) |
| Max Length | 2048 |
| Epochs | 1 (extend to 3 after first check) |
| Per-Device Batch Size | 1 |
| Grad Accum Steps | 8 |
| Effective Batch | 8 |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.03 |
| Eval Strategy | steps (500) |
| Save Strategy | steps (500), `hub_strategy=every_save`, limit=2 |
| Precision | bf16 |
| Gradient Checkpointing | true |
| Packing | false |
| Hub Model | `burtenshaw/qwen3-codeforces-cots-sft` |
| Hardware | a10g-small |
| Timeout | 2h |
| Trackio | project `qwen3-codeforces-cots`, run `sft-a10g` |

## Run Status
In Progress (queued to submit)

## Run Logs
Pending submission (job link will be added)

## Trackio Logs
Pending (will link after job starts)

## Run Evaluations
Pending (lighteval `openai_humaneval` for base + checkpoints)

---

# Experiment Evaluations
| Run Title | Benchmark | Score | Evaluation Job Link | Model Link |
|-----------|-----------|-------|---------------------|------------|
| `sft-a10g` - `TBD` - `In Progress` | HumanEval pass@1 | TBD | TBD | [burtenshaw/qwen3-codeforces-cots-sft](https://huggingface.co/burtenshaw/qwen3-codeforces-cots-sft)

```

</details>

>[!NOTE]
> The `open-r1/codeforces-cots` dataset is a dataset of codeforces problems and solutions. It is a good dataset for instruction tuning a model to solve hard coding problems.

### Updating the Training Report

As the experiment progresses, Codex will update the report with the latest information and each run report. You can view the report in `training_reports/<model>-<dataset>-<method>.md` file.

For example, codex will update the title of the report to `sft-a10g` - `TBD` - `In Progress` when the experiment is in progress.

```md
# `base-humaneval-a10g` - `2025-12-09 13:47:47 UTC` - `In Progress`
```

It can link to the run logs and trackio logs.

```md
## Run Logs

[Run Logs](https://huggingface.co/jobs/burtenshaw/6938272ec67c9f186cfe1ae3)

## Trackio Logs

[Trackio Logs](https://burtenshaw-trackio.hf.space/?project=qwen3-codeforces-sft&metrics=train/loss&runs=sft-qwen3-codeforces-20251209-175806&sidebar=hidden&navbar=hidden)
```

And it will update the evaluation results in a combined table.

```md
# Experiment Evaluations

| Run Title | Benchmark | Score | Evaluation Job Link | Model Link |
|-----------|-----------|-------|---------------------|------------|
| `base-humaneval-a10g` - `2025-12-09 13:47:47 UTC` - `Completed` | HumanEval pass@1 | 0.304 | [Logs](https://huggingface.co/jobs/burtenshaw/69382863c67c9f186cfe1ae7) | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| `qwen3-0.6b-lora-v1` - `2025-12-09 13:47:47 UTC` - `In Progress` | HumanEval pass@1 | TBD | TBD | [burtenshaw/qwen3-codeforces-cots-sft](https://huggingface.co/burtenshaw/qwen3-codeforces-cots-sft)
```

### Dataset Validation

Dataset format and processing is the most common source of training failures and usually a significant amount of work is done in the training script. Codex can validate datasets before the job starts and either define a configuration for TRL or process the dataset separately.

In most cases, Codex will validate the dataset before training, but you can always check the dataset validation before submitting the job.

```
Check if open-r1/codeforces-cots works for SFT training.
```

Codex runs a quick inspection on CPU (fractions of a penny) and reports:

```
Dataset validation for my-org/conversation-data:

SFT: ✓ READY
  Found 'messages' column with conversation format

DPO: ✗ INCOMPATIBLE
  Missing 'chosen' and 'rejected' columns
```

If your dataset needs transformation, Codex can preprocess the dataset before training.

```
Preprocess the dataset open-r1/codeforces-cots to use 'good_response' and 'bad_response' instead of 'chosen' and 'rejected'.
```

Codex will preprocess the dataset and update the training script with the new configuration.

### Review Before Submitting

Before Codex submits anything, you'll see the configuration:

```
I'll fine-tune Qwen/Qwen3-0.6B on open-r1/codeforces-cots using SFT.

Configuration:
- Hardware: t4-small (~$0.75/hour)
- Estimated time: ~20 minutes
- Estimated cost: ~$0.30
- Output: username/qwen-codeforces-cots-sft

The model will be pushed to Hub automatically. Should I submit?
```

This is your chance to adjust anything. Change the output repo name, pick different hardware, or ask Codex to modify training parameters. Once you approve, the job is submitted.

For example, you can ask Codex to try a test run:

```
Do a quick test run on 100 examples.
```

### Track Progress using the Training Report

After submission, you can track progress using the training report and get links to the run logs and trackio logs. You can also tell Codex to fetch the logs, summarize progress, and update the training report. As codex runs the jobs, it will update the training report with the latest information.

<details>
<summary>Example Codex Response</summary>

```
─ Worked for 10m 38s ─────────────────────────────────────────────────────────────────────────────────────────────────────────

• Training and eval jobs are live. Key details:

  - SFT launched on open-r1/codeforces-cots (solutions_py_decontaminated, 98/2 split) with LoRA on Qwen/Qwen3-0.6B, a10g-
    large, bf16, max_steps=1000, batch=1, grad_accum=8, lr=2e-5, save/eval every 200 steps, hub push burtenshaw/qwen3-
    codeforces-cots-sft. Job: https://huggingface.co/jobs/burtenshaw/6938272ec67c9f186cfe1ae3 (Trackio project codeforces-
    cots-sft, run qwen3-0.6b-lora-v1).
  - Baseline lighteval (HumanEval) on the base model queued on a10g-large. Job: https://huggingface.co/jobs/
  - Experiment report updated with current runs and links: training_reports/qwen3-codeforces-cots-sft.md.

  To evaluate the fine-tuned checkpoints once the SFT job pushes to the hub, reuse the same lighteval script:
  hf jobs uv run --flavor a10g-large --timeout 1h --secrets HF_TOKEN -e MODEL_ID=burtenshaw/qwen3-codeforces-cots-sft -e
  RESULTS_ORG=burtenshaw -d /tmp/lighteval_humaneval.py
  (or swap MODEL_ID for a specific checkpoint path). This will log to the same results directory on the hub.

  1. Monitor the two running jobs; I can fetch logs or metrics on request.
  2. After a checkpoint lands, trigger the lighteval run above to compare against the baseline; then we can fill in pass@1 in the report.
```
</details>

Trackio will monitor the training and evaluation scores for you by running evaluation jobs and selecting the best checkpoint. If you want to know the status of the training job, you can ask Codex to fetch the logs and summarize progress in a table.

```sh
Are models outperforming the base model?
```

<!-- TODO: Add evals table -->
```sh
| Model | Benchmark | Score | Evaluation Job Link | Model Link |
|-----------|-----------|-------|---------------------|------------|
| `qwen3-0.6b-lora-v1` - `2025-12-09 13:47:47 UTC` - `Completed` | HumanEval pass@1 | 0.342 | [Logs](<link to training job>) | [burtenshaw/qwen3-codeforces-cots-sft](https://huggingface.co/burtenshaw/qwen3-codeforces-cots-sft)
| `base-humaneval-a10g` - `2025-12-09 13:47:47 UTC` - `Completed` | HumanEval pass@1 | 0.306 | [Logs](<link to evaluation job>) | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
```

You can also monitor the training loss in real-time.

![Example Trackio dashboard of a Sweep test](https://huggingface.co/datasets/hf-skills/images/resolve/main/codex-sft-codeforces.png)

Codex fetches the logs and summarizes progress.

Click [here](https://burtenshaw-trackio.hf.space/?project=qwen3-codeforces-sft&metrics=train/loss&runs=sft-qwen3-codeforces-20251209-175806&sidebar=hidden&navbar=hidden) for an example Trackio dashboard with some completed runs.

### Use Your Model

When training completes, your model is on the Hub:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("burtenshaw/qwen3-codeforces-cots-sft")
tokenizer = AutoTokenizer.from_pretrained("burtenshaw/qwen3-codeforces-cots-sft")
```

Transformers is great as a standard, and we can easily convert the trained model to GGUF for local deployment. This is because the training skill contains instructions and support scripts to convert models to GGUF.

```
Convert my fine-tuned model to GGUF with Q4_K_M quantization.
Push to username/my-model-gguf.
```

Codex then converts to GGUF, applies quantization, and pushes to the Hub. If we trained a LoRA adapter, it will merge the LoRA adapters into the base model.

Then use it locally:

```bash
llama-server -hf <username>/<model-name>:<quantization>

# For example, to run the Qwen3-1.7B-GGUF model on your local machine:
llama-server -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M
```

### Hardware and Cost

Codex selects hardware based on your model size, but understanding the tradeoffs helps you make better decisions. You can use the [Hardware Guide](https://github.com/huggingface/skills/blob/main/hf-llm-trainer/skills/model-trainer/references/hardware_guide.md) to see the hardware options and costs, but codex will do it for you and select the best option.

For **tiny models under 1B parameters**, `t4-small` works well. These models train quickly—expect $1-2 for a full run. This is perfect for educational or experimental runs.

For **small models (1-3B)**, step up to `t4-medium` or `a10g-small`. Training takes a few hours and costs $5-15.

For **medium models (3-7B)**, you need `a10g-large` or `a100-large` with LoRA. Full fine-tuning doesn't fit, but LoRA makes these very trainable. Budget $15-40 for production.

For **large models (7B+)**, this HF skills job is not suitable for this scale yet. But stay tuned because we are working on it!

## What's Next

We've shown that Codex can handle the full lifecycle of model fine-tuning: validating data, selecting hardware, generating scripts, submitting jobs, monitoring progress, and converting outputs. This turns what used to be a specialized skill into something you can do through conversation.

Some things to try:

- Fine-tune a model on your own dataset
- Try bigger experiments with more models and datasets and let the agent create a report for you.
- Train a reasoning model with GRPO on math or code and let the agent create a report for you.

The [extension is open source](https://hf-learn.short.gy/gh-hf-skills). You can extend it, customize it for your workflows, or use it as a starting point for other training scenarios.

---

## Resources

### Codex
- [Codex Documentation](https://developers.openai.com/codex/) — OpenAI's AI coding agent
- [Codex Quickstart](https://developers.openai.com/codex/) — Get started with Codex
- [Codex AGENTS Guide](https://developers.openai.com/codex/) — Using AGENTS.md files

### Hugging Face Skills
- [SKILL.md](https://github.com/huggingface/skills/blob/main/hf-llm-trainer/skills/model-trainer/SKILL.md) — Full skill documentation
- [Training Methods](https://github.com/huggingface/skills/blob/main/hf-llm-trainer/skills/model-trainer/references/training_methods.md) — SFT, DPO, GRPO explained
- [Hardware Guide](https://github.com/huggingface/skills/blob/main/hf-llm-trainer/skills/model-trainer/references/hardware_guide.md) — GPU selection and costs
- [TRL Documentation](https://huggingface.co/docs/trl) — The underlying training library
- [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) — Cloud training infrastructure
- [Trackio](https://huggingface.co/docs/trackio) — Real-time training monitoring
