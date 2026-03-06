---
title: "Codex 正在推动 AI 模型的开源与训练流程"
thumbnail: /blog/assets/hf-skills-training/thumbnail-codex.png
authors:
- user: burtenshaw
- user: evalstate
translators:
- user: chenglu
---

# Codex 正在推动开源 AI 模型的训练与发布

![banner](https://huggingface.co/blog/assets/hf-skills-training/thumbnail-codex.png)

继我们使用 [Claude Code](https://huggingface.co/blog/hf-skills-training) 训练开源模型的项目之后，现在我们更进一步，将 [Codex](https://developers.openai.com/codex/) 引入这一流程。这里的重点不是“Codex 自己开源模型”，而是让 Codex 作为编码代理，参与并自动化开源模型的训练、评估与发布全流程。为此，我们为 Codex 接入了 [Hugging Face Skills](https://github.com/huggingface/skills) 仓库，该仓库包含了许多与机器学习和 AI 相关的“技能”，比如模型训练与评估等任务。通过 HF Skills，Codex 这样的编码代理可以实现：

* 对语言模型进行微调和强化学习（RL）对齐训练
* 查看、解释并基于 Trackio 的实时训练指标做出操作
* 评估模型检查点并根据评估结果作出决策
* 生成实验报告
* 将模型导出为 GGUF 格式，方便本地部署
* 将模型发布到 Hugging Face Hub

本教程将更深入地介绍它的工作原理，并手把手教你如何使用。我们开始吧！

> [!NOTE]
> Codex 使用 `AGENTS.md` 文件来完成特定任务，而 Claude Code 使用的是 “Skills”。幸运的是，“HF Skills” 兼容这两种方式，并可与 Claude Code、Codex 或 Gemini CLI 等主要编码代理配合使用。

例如，使用 `HF Skills`，你可以对 Codex 下达如下指令：

```
Fine-tune Qwen3-0.6B on the dataset open-r1/codeforces-cots
```

Codex 将自动执行以下步骤：

1. 验证数据集格式
2. 选择合适的硬件（比如 0.6B 模型使用 t4-small）
3. 使用并更新带有 Trackio 监控的训练脚本
4. 将任务提交到 Hugging Face Jobs
5. 返回任务 ID 和预估费用
6. 根据请求查看训练进度
7. 如遇问题，协助你进行调试

模型会在 Hugging Face 提供的 GPU 上训练，你可以同时做其他事情。训练完成后，你的微调模型将自动发布到 Hub，可立即使用。

这不仅仅是一个演示工具。这套扩展系统支持生产级的训练方法，有监督微调（SFT）、直接偏好优化（DPO）和带有可验证奖励的强化学习（RL）。你可以训练 0.5B 到 7B 参数规模的模型，将它们转换为 GGUF 格式便于本地运行，还可以通过多阶段流程结合不同方法。

## 目标：端到端的机器学习实验

我们在 Claude Code 教程中探索过单条指令的方式。而现在，我们可以让 OpenAI Codex 实现完整的端到端机器学习实验。Codex 能够：

* 实时监控进度
* 评估模型效果
* 维护最新训练报告

工程师可以将实验任务交由 Codex 自动执行，而自己只需查看最终报告即可。同时，Codex 还能根据训练与评估结果自动做出更多决策。

我们开始动手吧！

## 环境准备与安装

在开始之前，你需要：

* 一个 Hugging Face 账户，并开通 [Pro](https://hf.co/pro) 或 [Team / Enterprise](https://hf.co/enterprise) 付费计划（Jobs 需付费）
* 一个拥有写权限的 token（在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 生成）
* 安装并配置好 [Codex](https://developers.openai.com/codex/)

### 安装 Codex

Codex 是 OpenAI 推出的 AI 编码代理，包含在 ChatGPT Plus、Pro、Business、Edu 和 Enterprise 等计划中。它可以将 AI 能力直接融入你的开发流程。

参见 [Codex 官方文档](https://developers.openai.com/codex/) 获取安装与配置说明。

### 安装 Hugging Face Skills

Hugging Face Skills 仓库中包含 `AGENTS.md` 文件，Codex 会自动识别并使用它。

克隆仓库：

```bash
git clone https://github.com/huggingface/skills.git
cd skills
```

Codex 会自动检测到 `AGENTS.md` 文件，并加载相应的技能。你可以通过以下指令确认技能已加载：

```bash
codex --ask-for-approval never "Summarize the current instructions."
```

详细信息请参考 [Codex AGENTS 指南](https://developers.openai.com/codex/)。

### 连接 Hugging Face

使用以下命令并输入写权限 token 来进行认证：

```bash
hf auth login
```

Codex 支持 [MCP（模型上下文协议）](https://developers.openai.com/codex/)，你可以在配置文件中添加 Hugging Face 的 MCP 服务，提升与 Hub 的集成体验。将以下内容添加到 `~/.codex/config.toml`：

```toml
[mcp_servers.huggingface]
command = "npx"
args = ["-y", "mcp-remote", "https://huggingface.co/mcp?login"]
```

你也可以在 [Settings 页面](https://huggingface.co/settings/mcp) 中配置 MCP 服务。

之后启动 Codex，会跳转到 Hugging Face MCP 的认证页面。

## 你的第一个 AI 实验

我们来看一个完整示例。使用 [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) 数据集，配合 [openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) 基准测试，微调一个小模型来提升其代码解题能力。

> [!NOTE]
> `open-r1/codeforces-cots` 是一个包含 Codeforces 编程题及其解答的数据集，非常适合用于模型的指令微调，帮助模型解决复杂编程问题。

### 向 Codex 发起完整的微调实验请求

在你的项目目录下启动 Codex，并输入如下指令：

```
Start a new fine-tuning experiment to improve code solving abilities on using SFT. 
- Maintain a report for the experiment. 
- Evaluate models with the openai_humaneval benchmark
- Use the open-r1/codeforces-cots dataset
```

> [!TIP]
> 相比 Claude Code 教程中的单条指令方式，这里我们加了更多细节和步骤。
>
> 你也可以尝试自己不断迭代这个实验，提出一些更开放性的问题，比如：“哪个模型最擅长代码解题？”或“哪个数据集最适合训练代码解题能力？”

Codex 会分析你的请求，并生成对应的训练配置。例如，对于一个 0.6B 参数规模的模型和一个演示数据集，它会选择 `t4-small`，这是适合该模型大小的最低成本 GPU 选项。Codex 会在 `training_reports/<model>-<dataset>-<method>.md` 路径下创建一份新的实验报告，并在实验过程中持续更新每次运行的相关信息。

<details>
<summary>训练报告示例</summary>

```md
# 基础模型与数据集
[Base Model](https://huggingface.co/Qwen/Qwen3-0.6B)  
[Dataset](https://huggingface.co/datasets/open-r1/codeforces-cots)

---

# `sft-a10g` - `TBD` - `进行中`

## 训练参数
| 参数 | 值 |
|-----------|-------|
| 方法 | SFT（TRL）|
| 模型 | `Qwen/Qwen3-0.6B` |
| 数据集 | `open-r1/codeforces-cots`（训练集，5% 验证划分）|
| 最大长度 | 2048 |
| 训练轮数 | 1（首次检查后延长到3）|
| 每个设备的 batch 大小 | 1 |
| 梯度累积步数 | 8 |
| 有效 batch | 8 |
| 学习率 | 5e-5 |
| 权重衰减 | 0.01 |
| 预热比例 | 0.03 |
| 评估策略 | 每 500 步 |
| 保存策略 | 每 500 步，`hub_strategy=every_save`，最多保存2个 |
| 精度 | bf16 |
| 启用梯度检查点 | true |
| 是否打包样本 | false |
| Hub 模型仓库 | `burtenshaw/qwen3-codeforces-cots-sft` |
| 使用硬件 | a10g-small |
| 超时时间 | 2 小时 |
| Trackio 项目 | `qwen3-codeforces-cots`，运行名称：`sft-a10g` |

## 运行状态
进行中（等待提交）

## 运行日志
尚未提交（提交后会补充链接）

## Trackio 日志
等待中（任务开始后补充链接）

## 模型评估
等待中（将使用 lighteval 对基础模型和各检查点进行 `openai_humaneval` 评估）

---

# 实验评估结果
| 运行标题 | 基准测试 | 得分 | 评估任务链接 | 模型链接 |
|-----------|-----------|-------|---------------------|------------|
| `sft-a10g` - `TBD` - `进行中` | HumanEval pass@1 | 待定 | 待定 | [burtenshaw/qwen3-codeforces-cots-sft](https://huggingface.co/burtenshaw/qwen3-codeforces-cots-sft)
```

</details>

### 训练报告实时更新

随着实验的推进，Codex 会不断将最新的信息和每次运行的结果写入报告中。你可以在 `training_reports/<model>-<dataset>-<method>.md` 文件中查看这些更新。

例如，当实验进行中时，Codex 会将报告标题更新为如下格式：

```md
# `sft-a10g` - `TBD` - `进行中`
```

它还会添加运行日志和 Trackio 实时监控的链接：

```md
## Run Logs

[Run Logs](https://huggingface.co/jobs/burtenshaw/6938272ec67c9f186cfe1ae3)

## Trackio Logs

[Trackio Logs](https://burtenshaw-trackio.hf.space/?project=qwen3-codeforces-sft&metrics=train/loss&runs=sft-qwen3-codeforces-20251209-175806&sidebar=hidden&navbar=hidden)
```

评估结果也会更新到实验评估表中：

```md
# Experiment Evaluations

| Run Title | Benchmark | Score | Evaluation Job Link | Model Link |
|-----------|-----------|-------|---------------------|------------|
| `base-humaneval-a10g` - `2025-12-09 13:47:47 UTC` - `Completed` | HumanEval pass@1 | 0.304 | [Logs](https://huggingface.co/jobs/burtenshaw/69382863c67c9f186cfe1ae7) | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| `qwen3-0.6b-lora-v1` - `2025-12-09 13:47:47 UTC` - `In Progress` | HumanEval pass@1 | TBD | TBD | [burtenshaw/qwen3-codeforces-cots-sft](https://huggingface.co/burtenshaw/qwen3-codeforces-cots-sft)
```

### 数据集验证

数据集格式错误是训练失败最常见的原因之一，也常是训练脚本中工作量最大的部分之一。Codex 会在任务开始前验证数据集格式，并根据需要配置 TRL 训练参数或预处理数据。

通常，Codex 会在训练前验证数据集，你也可以手动让它提前检查：

```
Check if open-r1/codeforces-cots works for SFT training.
```

Codex 会在 CPU 上快速检查（成本非常低），并返回如下结果：

```
Dataset validation for my-org/conversation-data:

SFT: ✓ READY
  Found 'messages' column with conversation format

DPO: ✗ INCOMPATIBLE
  Missing 'chosen' and 'rejected' columns
```

如果你的数据集需要转换格式，Codex 可以在训练开始前对数据集进行预处理：

```
Preprocess the dataset open-r1/codeforces-cots to use 'good_response' and 'bad_response' instead of 'chosen' and 'rejected'.
```

Codex 会自动预处理数据并更新训练脚本中的配置。

### 提交前确认

在 Codex 提交训练任务前，它会向你确认配置信息：

```
I'll fine-tune Qwen/Qwen3-0.6B on open-r1/codeforces-cots using SFT.

Configuration:
- Hardware: t4-small (~$0.75/hour)
- Estimated time: ~20 minutes
- Estimated cost: ~$0.30
- Output: username/qwen-codeforces-cots-sft

The model will be pushed to Hub automatically. Should I submit?
```

此时你可以修改任何内容，比如更改输出仓库名称、选择其他硬件、调整训练参数等。一旦确认无误，Codex 就会提交任务。

你也可以先进行小规模测试：

```
Do a quick test run on 100 examples.
```

### 使用训练报告追踪进度

提交任务后，你可以通过训练报告追踪进度，并获取运行日志和 Trackio 日志的链接。你也可以让 Codex 获取日志、汇总训练进展，并更新训练报告。随着 Codex 执行任务，它会持续将最新信息写入训练报告中。

<details>
<summary>Codex 示例响应</summary>

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

Trackio 会通过自动运行评估任务，帮助你监控训练和评估的得分，并选择表现最好的模型检查点（checkpoint）。如果你想了解训练任务的状态，可以让 Codex 获取日志，并以表格形式总结训练进展。

例如，你可以直接询问：

```sh
Are models outperforming the base model?
```

Codex 会返回类似的表格总结：

```sh
| Model | Benchmark | Score | Evaluation Job Link | Model Link |
|-----------|-----------|-------|---------------------|------------|
| `qwen3-0.6b-lora-v1` - `2025-12-09 13:47:47 UTC` - `Completed` | HumanEval pass@1 | 0.342 | [Logs](<link>) | [burtenshaw/qwen3-codeforces-cots-sft](https://huggingface.co/burtenshaw/qwen3-codeforces-cots-sft)
| `base-humaneval-a10g` - `2025-12-09 13:47:47 UTC` - `Completed` | HumanEval pass@1 | 0.306 | [Logs](<link>) | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
```

通过这种方式，你可以清楚地看到微调后的模型是否优于基础模型。

你也可以实时查看训练损失变化：

![Trackio 示例图表](https://huggingface.co/datasets/hf-skills/images/resolve/main/codex-sft-codeforces.png)

Codex 会自动获取日志并更新进度。

点击此处查看 [Trackio 仪表盘示例](https://burtenshaw-trackio.hf.space/?project=qwen3-codeforces-sft&metrics=train/loss&runs=sft-qwen3-codeforces-20251209-175806&sidebar=hidden&navbar=hidden)

### 使用你的模型

训练完成后，模型会被上传到 Hugging Face Hub：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("burtenshaw/qwen3-codeforces-cots-sft")
tokenizer = AutoTokenizer.from_pretrained("burtenshaw/qwen3-codeforces-cots-sft")
```

Transformers 是一个非常优秀的标准工具，我们也可以轻松地将训练好的模型转换为 GGUF 格式，用于本地部署。这是因为训练技能中已经包含了将模型转换为 GGUF 的说明和支持脚本。

```
Convert my fine-tuned model to GGUF with Q4_K_M quantization.
Push to username/my-model-gguf.
```

Codex 会自动将模型转换为 GGUF 格式，应用指定的量化策略，并将其推送到 Hugging Face Hub。如果你使用了 LoRA 适配器进行训练，它还会将这些适配器合并回基础模型中。

在本地运行模型：

```bash
llama-server -hf <username>/<model-name>:<quantization>

# For example, to run the Qwen3-1.7B-GGUF model on your local machine:
llama-server -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M
```

### 硬件与成本

Codex 会根据你的模型规模自动选择合适的硬件，但了解背后的取舍逻辑，有助于你做出更明智的决策。你可以参考这个 [硬件指南](https://github.com/huggingface/skills/blob/main/hf-llm-trainer/skills/model-trainer/references/hardware_guide.md) 来了解各种硬件的选择和成本，不过 Codex 会自动帮你选择最优配置。

* 对于 **小于 10 亿参数的微型模型**，`t4-small` 是一个很好的选择。这类模型训练速度快，成本大约在 **$1-2**，非常适合教学或实验用途。

* 对于 **小模型（1-3B 参数）**，推荐使用 `t4-medium` 或 `a10g-small`。训练耗时几个小时，成本在 **$5-15** 左右。

* 对于 **中等模型（3-7B 参数）**，需要使用 `a10g-large` 或 `a100-large`，同时配合 LoRA 微调。完整微调不太可行，但借助 LoRA 技术仍然可以高效训练。生产级别训练预算约为 **$15-40**。

* 对于 **大型模型（超过 7B）**，目前 HF Skills Jobs 暂不支持。但请保持关注，我们正在开发支持大模型的能力！

## 接下来可以做什么？

我们已经展示了 Codex 如何处理模型微调的完整生命周期，验证数据、选择硬件、生成训练脚本、提交任务、监控进度，以及转换输出。

你可以尝试以下操作：

* 使用你自己的数据集微调一个模型
* 进行更大规模的实验，使用多个模型和数据集，并让代理自动生成训练报告
* 使用 GRPO 方法在数学或代码任务上训练一个推理能力模型，并生成完整的实验报告

这个 [Codex 扩展是开源的](https://hf-learn.short.gy/gh-hf-skills)，你可以根据自己的流程进行扩展和定制，或者将其作为其他训练场景的起点。

---

## 资源链接

### Codex

* [Codex 官方文档](https://developers.openai.com/codex/) ，OpenAI 的 AI 编码代理
* [Codex 快速上手](https://developers.openai.com/codex/)
* [Codex AGENTS 指南](https://developers.openai.com/codex/) ，使用 AGENTS.md 文件说明

### Hugging Face Skills

* [SKILL.md](https://github.com/huggingface/skills/blob/main/hf-llm-trainer/skills/model-trainer/SKILL.md) ，技能文档
* [训练方法指南](https://github.com/huggingface/skills/blob/main/hf-llm-trainer/skills/model-trainer/references/training_methods.md) ，介绍 SFT、DPO、GRPO 等方法
* [硬件指南](https://github.com/huggingface/skills/blob/main/hf-llm-trainer/skills/model-trainer/references/hardware_guide.md)
* [TRL 文档](https://huggingface.co/docs/trl) ，Hugging Face 的训练库
* [HF Jobs 文档](https://huggingface.co/docs/huggingface_hub/guides/jobs) ，云端训练任务指南
* [Trackio 文档](https://huggingface.co/docs/trackio) ，实时训练监控工具
