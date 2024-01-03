---
title: "欢迎 Mixtral - 当前 Hugging Face 上最先进的 MoE 模型"
thumbnail: /blog/assets/mixtral/thumbnail.jpg
authors:
- user: lewtun
- user: philschmid
- user: osanseviero
- user: pcuenq
- user: olivierdehaene
- user: lvwerra
- user: ybelkada
translators:
- user: MatrixYao
---

# 欢迎 Mixtral - 当前 Hugging Face 上最先进的 MoE 模型

最近，Mistral 发布了一个激动人心的大语言模型：Mixtral 8x7b，该模型把开放模型的性能带到了一个新高度，并在许多基准测试上表现优于 GPT-3.5。我们很高兴能够在 Hugging Face 生态系统中全面集成 Mixtral 以对其提供全方位的支持 🔥！

Hugging Face 对 Mixtral 的全方位支持包括：

- [Hub 上的模型](https://huggingface.co/models?search=mistralai/Mixtral)，包括模型卡以及相应的许可证（Apache 2.0）
- [🤗 transformers 的集成](https://github.com/huggingface/transformers/releases/tag/v4.36.0)
- 推理终端的集成
- [TGI](https://github.com/huggingface/text-generation-inference) 的集成，以支持快速高效的生产级推理
- 使用 🤗 TRL 在单卡上对 Mixtral 进行微调的示例

## 目录

- [欢迎 Mixtral - 当前 Hugging Face 上最先进的 MoE 模型](#欢迎-mixtral---当前-hugging-face-上最先进的-moe-模型)
	- [目录](#目录)
	- [Mixtral 8x7b 是什么？](#mixtral-8x7b-是什么)
		- [关于命名](#关于命名)
		- [提示格式](#提示格式)
		- [我们不知道的事](#我们不知道的事)
	- [演示](#演示)
	- [推理](#推理)
		- [使用 🤗 transformers](#使用-transformers)
		- [使用 TGI](#使用-tgi)
	- [用 🤗 TRL 微调](#用-trl-微调)
	- [量化 Mixtral](#量化-mixtral)
		- [使用 4 比特量化加载 Mixtral](#使用-4-比特量化加载-mixtral)
		- [使用 GPTQ 加载 Mixtral](#使用-gptq-加载-mixtral)
	- [免责声明及正在做的工作](#免责声明及正在做的工作)
	- [更多资源](#更多资源)
	- [总结](#总结)

## Mixtral 8x7b 是什么？ 

Mixtral 的架构与 Mistral 7B 类似，但有一点不同：它实际上内含了 8 个“专家”模型，这要归功于一种称为“混合专家”(Mixture of Experts，MoE) 的技术。当 MoE 与 transformer 模型相结合时，我们会用稀疏 MoE 层替换掉某些前馈层。MoE 层包含一个路由网络，用于选择将输入词元分派给哪些专家处理。Mixtral 模型为每个词元选择两名专家，因此，尽管其有效参数量是 12B 稠密模型的 4 倍，但其解码速度却能做到与 12B 的稠密模型相当！

欲了解更多有关 MoE 的知识，请参阅我们之前的博文：[hf.co/blog/zh/moe](https://huggingface.co/blog/zh/moe)。

**本次发布的 Mixtral 模型的主要特点：**

- 模型包括基础版和指令版
- 支持高达 32k 词元的上下文
- 性能优于 Llama 2 70B，在大多数基准测试上表现不逊于 GPT3.5
- 支持英语、法语、德语、西班牙语及意大利语
- 擅长编码，HumanEval 得分为 40.2%
- 可商用，Apache 2.0 许可证

那么，Mixtral 模型效果到底有多好呢？下面列出了 Mixtral 基础模型与其他先进的开放模型在 [LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 上表现（分数越高越好）：

| 模型                                                                             | 许可证         | 是否可商用 | 预训练词元数 | 排行榜得分 ⬇️ |
| --------------------------------------------------------------------------------- | --------------- | --------------- | ------------------------- | -------------------- |
| [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | Apache 2.0      | ✅               | 不详                   | 68.42                |
| [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)     | Llama 2 许可证 | ✅               | 2,000B                    | 67.87                |
| [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b)                     | Apache 2.0      | ✅               | 1,000B                    | 61.5                 |
| [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)     | Apache 2.0      | ✅               | 不详                   | 60.97                |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)       | Llama 2 许可证 | ✅               | 2,000B                    | 54.32                |

我们还用 MT-Bench 及 AlpacaEval 等基准对指令版和其它聊天模型进行了对比。下表列出了 [Mixtral Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 与顶级闭源或开放模型相比的表现（分数越高越好）：

| 模型                                                                                               | 可得性    | 上下文窗口（词元数） | MT-Bench 得分 ⬇️ |
| --------------------------------------------------------------------------------------------------- | --------------- | ----------------------- | ---------------- |
| [GPT-4 Turbo](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)        | 私有     | 128k                    | 9.32             |
| [GPT-3.5-turbo-0613](https://platform.openai.com/docs/models/gpt-3-5)                               | 私有     | 16k                     | 8.32             |
| [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | Apache 2.0      | 32k                     | 8.30             |
| [Claude 2.1](https://www.anthropic.com/index/claude-2-1)                                            | 私有     | 200k                    | 8.18             |
| [openchat/openchat_3.5](https://huggingface.co/openchat/openchat_3.5)                               | Apache 2.0      | 8k                      | 7.81             |
| [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)                 | MIT             | 8k                      | 7.34             |
| [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)             | Llama 2 许可证 | 4k                      | 6.86             |

令人印象深刻的是，Mixtral Instruct 的性能优于 MT-Bench 上的所有其他开放模型，且是第一个与 GPT-3.5 性能相当的开放模型！

### 关于命名

Mixtral MoE 模型虽然名字是 **Mixtral-8x7B**，但它其实并没有 56B 参数。发布后不久，我们就发现不少人被名字误导了，认为该模型的行为类似于 8 个模型的集合，其中每个模型有 7B 个参数，但这种想法其实与 MoE 模型的工作原理不符。实情是，该模型中只有某些层（前馈层）是各专家独有的，其余参数与稠密 7B 模型情况相同，是各专家共享的。所以，参数总量并不是 56B，而是 45B 左右。所以可能叫它 [`Mixtral-45-8e`](https://twitter.com/osanseviero/status/1734248798749159874) 更贴切，更能符合其架构。更多有关 MoE 如何运行的详细信息，请参阅我们之前发表的[《MoE 详解》](https://huggingface.co/blog/zh/moe) 一文。

### 提示格式

[基础模型](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)没有提示格式，与其他基础模型一样，它可用于序列补全或零样本/少样本推理。你可以对基础模型进行微调，将其适配至自己的应用场景。[指令模型](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 有一个非常简单的对话格式。

```bash
<s> [INST] User Instruction 1 [/INST] Model answer 1</s> [INST] User instruction 2[/INST]
```

你必须准确遵循此格式才能有效使用指令模型。稍后我们将展示，使用 `transformers` 的聊天模板能很轻易地支持这类自定义指令提示格式。

### 我们不知道的事

与之前的 Mistral 7B 版本一样，对这一新的模型家族，我们也有几个待澄清的问题。比如，我们不知道用于预训练的数据集大小，也不知道它的组成信息以及预处理方式信息。

同样，对于 Mixtral 指令模型，我们对微调数据集或 SFT 和 DPO 使用的超参也知之甚少。

## 演示

你可以在 Hugging Face Chat 上与 Mixtral Instruct 模型聊天！点击[此处](https://huggingface.co/chat/?model=mistralai/Mixtral-8x7B-Instruct-v0.1)开始体验吧。

## 推理

我们主要提供两种对 Mixtral 模型进行推理的方法：

- 通过 🤗 transformers 的 `pipeline()` 接口。
- 通过 TGI，其支持连续组批、张量并行等高级功能，推理速度极快。

以上两种方法均支持半精度 (float16) 及量化权重。由于 Mixtral 模型的参数量大致相当于 45B 参数的稠密模型，因此我们可以对所需的最低显存量作一个估计，如下：

| 精度 | 显存需求 |
| --------- | ------------- |
| float16   | >90 GB        |
| 8-bit     | >45 GB        |
| 4-bit     | >23 GB        |

### 使用 🤗 transformers

从 transformers [4.36 版](https://github.com/huggingface/transformers/releases/tag/v4.36.0)开始，用户就可以用 Hugging Face 生态系统中的所有工具处理 Mixtral 模型，如：

- 训练和推理脚本及示例
- 安全文件格式（`safetensors`）
- 与 bitsandbytes（4 比特量化）、PEFT（参数高效微调）和 Flash Attention 2 等工具的集成
- 使用文本生成任务所提供的工具及辅助方法
- 导出模型以进行部署

用户唯一需要做的是确保 `transformers` 的版本是最新的：

```bash
pip install -U "transformers==4.36.0" --upgrade
```

下面的代码片段展示了如何使用 🤗 transformers 及 4 比特量化来运行推理。由于模型尺寸较大，你需要一张显存至少为 30GB 的卡才能运行，符合要求的卡有 A100（80 或 40GB 版本）、A6000（48GB）等。

```python
from transformers import AutoTokenizer
import transformers
import torch

model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
)

messages = [{"role": "user", "content": "Explain what a Mixture of Experts is in less than 100 words."}]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```

> \<s>[INST] Explain what a Mixture of Experts is in less than 100 words. [/INST] A
Mixture of Experts is an ensemble learning method that combines multiple models,
or "experts," to make more accurate predictions. Each expert specializes in a
different subset of the data, and a gating network determines the appropriate
expert to use for a given input. This approach allows the model to adapt to
complex, non-linear relationships in the data and improve overall performance.
> 

### 使用 TGI

**[TGI](https://github.com/huggingface/text-generation-inference)** 是 Hugging Face 开发的生产级推理容器，可用于轻松部署大语言模型。其功能主要有：连续组批、流式词元输出、多 GPU 张量并行以及生产级的日志记录和跟踪等。

你可在 Hugging Face 的[推理终端](https://ui.endpoints.huggingface.co/new?repository=mistralai%2FMixtral-8x7B-Instruct-v0.1&vendor=aws&region=us-east-1&accelerator=gpu&instance_size=2xlarge&task=text-generation&no_suggested_compute=true&tgi=true&tgi_max_batch_total_tokens=1024000&tgi_max_total_tokens=32000)上部署 Mixtral，其使用 TGI 作为后端。要部署 Mixtral 模型，可至[模型页面](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)，然后单击 [Deploy -> Inference Endpoints](https://ui.endpoints.huggingface.co/new?repository=meta-llama/Llama-2-7b-hf) 按钮即可。

*注意：如你的账号 A100 配额不足，可发送邮件至 **[api-enterprise@huggingface.co](mailto:api-enterprise@huggingface.co)** 申请升级。*

你还可以阅读我们的博文 **[用 Hugging Face 推理终端部署 LLM](https://huggingface.co/blog/inference-endpoints-llm)** 以深入了解如何部署 LLM，该文包含了推理终端支持的超参以及如何使用 Python 和 Javascript 接口来流式生成文本等信息。

你还可以使用 Docker 在 2 张 A100 (80GB) 上本地运行 TGI，如下所示：

```bash
docker run --gpus all --shm-size 1g -p 3000:80 -v /data:/data ghcr.io/huggingface/text-generation-inference:1.3.0 \
	--model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
	--num-shard 2 \
	--max-batch-total-tokens 1024000 \
	--max-total-tokens 32000
```

## 用 🤗 TRL 微调

训练 LLM 在技术和算力上都有较大挑战。本节我们将了解在 Hugging Face 生态系统中如何在单张 A100 GPU 上高效训练 Mixtral。

下面是在 OpenAssistant 的 [聊天数据集](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) 上微调 Mixtral 的示例命令。为了节省内存，我们对注意力块中的所有线性层执行 4 比特量化和 [QLoRA](https://arxiv.org/abs/2305.14314)。请注意，与稠密 transformer 模型不同，我们不对专家网络中的 MLP 层进行量化，因为它们很稀疏并且量化后 PEFT 效果不好。

首先，安装 🤗 TRL 的每日构建版并下载代码库以获取[训练脚本](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)：

```bash
pip install -U transformers
pip install git+https://github.com/huggingface/trl
git clone https://github.com/huggingface/trl
cd trl
```

然后，运行脚本：

```bash
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml --num_processes=1 \
	examples/scripts/sft.py \
	--model_name mistralai/Mixtral-8x7B-v0.1 \
	--dataset_name trl-lib/ultrachat_200k_chatml \
	--batch_size 2 \
	--gradient_accumulation_steps 1 \
	--learning_rate 2e-4 \
	--save_steps 200_000 \
	--use_peft \
	--peft_lora_r 16 --peft_lora_alpha 32 \
	--target_modules q_proj k_proj v_proj o_proj \
	--load_in_4bit
```

在单张 A100 上训练大约需要 48 小时，但我们可以通过`--num_processes` 来调整 GPU 的数量以实现并行。

## 量化 Mixtral

如上所见，该模型最大的挑战是如何实现普惠，即如何让它能够在消费级硬件上运行。因为即使以半精度（`torch.float16`）加载，它也需要 90GB 显存。

借助 🤗 transformers 库，我们支持用户开箱即用地使用 QLoRA 和 GPTQ 等最先进的量化方法进行推理。你可以阅读[相应的文档](https://huggingface.co/docs/transformers/quantization)以获取有关我们支持的量化方法的更多信息。

### 使用 4 比特量化加载 Mixtral

用户还可以通过安装 `bitsandbytes` 库（`pip install -U bitsandbytes`）并将参数 `load_in_4bit=True` 传给`from_pretrained` 方法来加载 4 比特量化的 Mixtral。为了获得更好的性能，我们建议用户使用 `bnb_4bit_compute_dtype=torch.float16` 来加载模型。请注意，你的 GPU 显存至少得有 30GB 才能正确运行下面的代码片段。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

prompt = "[INST] Explain what a Mixture of Experts is in less than 100 words. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(0)

output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

该 4 比特量化技术由 [QLoRA 论文](https://huggingface.co/papers/2305.14314)提出，你可以通过[相应的 Hugging Face 文档](https://huggingface.co/docs/transformers/quantization#4-bit)或[这篇博文](https://huggingface.co/blog/zh/4bit-transformers-bitsandbytes)获取更多相关信息。

### 使用 GPTQ 加载 Mixtral

GPTQ 算法是一种训后量化技术，其中权重矩阵的每一行都是独立量化的，以获取误差最小的量化权重。这些权重被量化为 int4，但在推理过程中会即时恢复为 fp16。与 4 比特 QLoRA 相比，GPTQ 的量化模型是通过对某个数据集进行校准而得的。[TheBloke](https://huggingface.co/TheBloke) 在 🤗 Hub 上分享了很多量化后的 GPTQ 模型，这样大家无需亲自执行校准就可直接使用量化模型。

对于 Mixtral，为了获得更好的性能，我们必须调整一下校准方法，以确保我们**不会**量化那些专家门控层。量化模型的最终困惑度（越低越好）为 `4.40`，而半精度模型为 `4.25`。你可在[此处](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GPTQ)找到量化模型，要使用 🤗 transformers 运行它，你首先需要更新 `auto-gptq` 和 `optimum` 库：

```bash
pip install -U optimum auto-gptq
```

然后是从源代码安装 transformers：

```bash
pip install -U git+https://github.com/huggingface/transformers.git
```

安装好后，只需使用 `from_pretrained` 方法加载 GPTQ 模型即可：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "TheBloke/Mixtral-8x7B-v0.1-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "[INST] Explain what a Mixture of Experts is in less than 100 words. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(0)

output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

请注意，你的 GPU 显存至少得有 30GB 才能运行 Mixtral 模型的 QLoRA 和 GPTQ 版本。如果你如上例一样使用了 `device_map="auto"`，则其在 24GB 显存时也可以运行，因此会有一些层被自动卸载到 CPU。

## 免责声明及正在做的工作

- **量化**：围绕 MoE 的量化还有许多研究正如火如荼地展开。上文展示了我们基于 TheBloke 所做的一些初步实验，但我们预计随着对该架构研究的深入，会涌现出更多进展！这一领域的进展将会是日新月异的，我们翘首以盼。此外，最近的工作，如 [QMoE](https://arxiv.org/abs/2310.16795)，实现了 MoE 的亚 1 比特量化，也是值得尝试的方案。

- **高显存占用**：MoE 运行推理速度较快，但对显存的要求也相对较高（因此需要昂贵的 GPU）。这对本地推理提出了挑战，因为本地推理所拥有的设备显存一般较小。MoE 非常适合多设备大显存的基础设施。对 Mixtral 进行半精度推理需要 90GB 显存 🤯。

## 更多资源

- [MoE 详解](https://huggingface.co/blog/zh/moe)
- [Mistral 的 Mixtral 博文](https://mistral.ai/news/mixtral-of-experts/)
- [Hub 上的模型](https://huggingface.co/models?other=mixtral)
- [开放 LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [基于 Mixtral 的 Hugging Chat 聊天演示应用](https://huggingface.co/chat/?model=mistralai/Mixtral-8x7B-Instruct-v0.1)

## 总结

我们对 Mixtral 的发布感到欢欣鼓舞！我们正围绕 Mixtral 准备更多关于微调和部署文章，尽请期待。

> 英文原文: <url> https://huggingface.co/blog/mixtral </url>
> 原文作者：Lewis Tunstall，Philipp Schmid，Omar Sanseviero，Pedro Cuenca，Olivier Dehaene，Leandro von Werra，Younes Belkada
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。

