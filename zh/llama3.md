---
title: "欢迎 Llama 3：Meta 的新一代开源大语言模型" 
thumbnail: /blog/assets/llama3/thumbnail.jpg
authors:
- user: philschmid
- user: osanseviero
- user: pcuenq
- user: ybelkada
- user: lvwerra
translators:
- user: AdinaY
---

# 欢迎 Llama 3：Meta 的新一代开源大语言模型
## 介绍

Meta 公司的 Llama 3 是开放获取的 Llama 系列的最新版本，现已在 Hugging Face 平台发布。看到 Meta 持续致力于开放 AI 领域的发展令人振奋，我们也非常高兴地全力支持此次发布，并实现了与 Hugging Face 生态系统的深度集成。

Llama 3 提供两个版本：8B 版本适合在消费级 GPU 上高效部署和开发；70B 版本则专为大规模 AI 应用设计。每个版本都包括基础和指令调优两种形式。此外，基于 Llama 3 8B 微调后的 Llama Guard 新版本也已作为 Llama Guard 2（安全微调版本）发布。

我们与 Meta 密切合作，确保其产品能够无缝集成进 Hugging Face 的生态系统。在 Hub 上，您可以找到这五个开放获取的模型（包括两个基础模型、两个微调模型以及 Llama Guard）。

本次发布的主要特性和集成功能包括：

- [Hub 上的模型](https://huggingface.co/meta-llama)，并提供了模型卡片和许可证信息
- 🤗 Transformers 的集成
- [针对 Meta Llama 3 70B 的 Hugging Chat 集成](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-instruct)
- 推理功能集成到推理端点、Google Cloud 和 Amazon SageMaker
- 在单个 GPU 上对 Llama 3 8B 进行微调的示例，采用 🤗 TRL

## 目录

  - [介绍](#introduction)
  - [目录](#table-of-contents)
  - [Llama 3 的新进展](#whats-new-with-llama-3)
  - [Llama 3 评估](#llama-3-evaluation)
  - [如何设置 Llama 3 的提示](#how-to-prompt-llama-3)
  - [演示](#demo)
  - [如何使用 🤗 Transformers](#using-transformers)
  - [推理集成](#inference-integrations)
  - [如何使用 🤗 TRL 进行微调](#fine-tuning-with-trl)
  - [额外资源](#additional-resources)
  - [鸣谢](#acknowledgments)

## Llama 3 的新进展

Llama 3 的推出标志着 Meta 基于 Llama 2 架构推出了四个新的开放型大语言模型。这些模型分为两种规模：8B 和 70B 参数，每种规模都提供预训练基础版和指令调优版。所有版本均可在各种消费级硬件上运行，并具有 8000 Token 的上下文长度。

- [Meta-Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B): 8B 基础模型
- [Meta-Llama-3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct): 8B 基础模型的指令调优版
- [Meta-Llama-3-70b](https://huggingface.co/meta-llama/Meta-Llama-3-70B): 70B 基础模型
- [Meta-Llama-3-70b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct): 70B 基础模型的指令调优版

此外，还发布了基于 Llama 3 8B 微调后的最新 Llama Guard 版本——Llama Guard 2。Llama Guard 2 是为生产环境设计的，能够对大语言模型的输入（即提示）和响应进行分类，以便识别潜在的不安全内容。

与 Llama 2 相比，Llama 3 最大的变化是采用了新的 Tokenizer，将词汇表大小扩展至 128,256（前版本为 32,000 Token）。这一更大的词汇库能够更高效地编码文本（无论输入还是输出），并有可能提升模型的多语种处理能力。不过，这也导致嵌入层的输入和输出矩阵尺寸增大，这是小型模型参数增加（从 Llama 2 的 7B 增至 Llama 3 的 8B）的主要原因之一。此外，8B 版本的模型现在采用了分组查询注意力（GQA），这是一种效率更高的表达方式，有助于处理更长的上下文。

Llama 3 模型在两个拥有 24,000 GPU 的集群上进行了训练，使用的是超过 15 万亿 Token 的新公共在线数据。我们无法得知训练数据具体细节，但可以推测，更大规模且更细致的数据策划是性能提升的重要因素。Llama 3 Instruct 针对对话应用进行了优化，结合了超过 1000 万的人工标注数据，通过监督式微调（SFT）、拒绝采样、邻近策略优化（PPO）和直接策略优化（DPO）进行训练。

关于许可条款，Llama 3 提供了一个宽松的许可证，允许重新分发、微调和创作衍生作品。Llama 3 许可证中新增了明确归属的要求，这在 Llama 2 中并未设定。例如，衍生模型需要在其名称开头包含“Llama 3”，并且在衍生作品或服务中需注明“基于 Meta Llama 3 构建”。详细条款，请务必阅读[官方许可证](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/LICENSE)。

## Llama 3 评估

_注：我们目前正在对 Meta Llama 3 进行单独评估，一旦有了结果将立即更新此部分。_

## 如何设置 Llama 3 的提示

基础模型不具备固定的提示格式。如同其他基础模型，它们可以用来延续输入序列，提供合理的续写或进行零样本/少样本推理。这些模型也是您自定义微调的理想基础。指令版本采用以下对话结构：

```bash
system

{{ system_prompt }}user

{{ user_msg_1 }}assistant

{{ model_answer_1 }}
```

为了有效使用，必须精确复制此格式。我们稍后将展示如何利用 `transformers` 中提供的聊天模板轻松重现这一指令提示格式。

## 演示

您现在可以在 Hugging Chat 上与 Llama 3 70B 指令版进行交流！请访问此链接：https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-instruct

## 如何使用 🤗 Transformers

通过安装 Transformers 的[4.40 版本](https://github.com/huggingface/transformers/releases/tag/v4.40.0)，您可以充分利用 Hugging Face 生态系统中提供的各种工具，如：

- 训练及推理脚本和示例
- 安全文件格式（safetensors）
- 与 bitsandbytes（4位量化）、PEFT（参数效率微调）和 Flash Attention 2 等工具的集成
- 辅助生成操作的实用工具
- 模型部署的出口机制

此外，Llama 3 模型兼容 `torch.compile()` 的 CUDA 图表，使得推理时间可加速约 4 倍！

要在 transformers 中使用 Llama 3 模型，请确保安装了最新版本：

```jsx
pip install -U "transformers==4.40.0" --upgrade
```

以下代码片段展示了如何在 transformers 中使用 `Llama-3-8b-instruct`。这需要大约 16 GB 的 RAM，包括 3090 或 4090 等消费级 GPU。

```python
from transformers import pipeline
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipe(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
assistant_response = outputs[0]["generated_text"][-1]["content"]
print(assistant_response)
```

> Arrrr, me hearty! Me name be Captain Chat, the scurviest pirate chatbot to ever sail the Seven Seas! Me be here to swab the decks o' yer mind with me trusty responses, savvy? I be ready to hoist the Jolly Roger and set sail fer a swashbucklin' good time, matey! So, what be bringin' ye to these fair waters?


一些细节：

- 我们在 `bfloat16` 中加载了模型。这是 Meta 发布的原始检查点所使用的类型，因此它是推荐的运行方式，以确保最佳精确度或进行评估。对于实际使用，也可以安全地使用 `float16`，这可能取决于您的硬件而更快。
- 助理响应可能会以特殊 token 结束，但如果找到常规的 EOS token，我们也必须停止生成。我们可以通过在 `eos_token_id` 参数中提供一个终结符列表来提前停止生成。
- 我们使用了从原始 meta 代码库中取得的默认抽样参数（`temperature` 和 `top_p`）。我们还没有时间进行广泛的测试，欢迎探索！

您也可以自动量化模型，将其加载到 8 位或甚至 4 位模式。4 位加载需要大约 7 GB 的内存运行，使其兼容许多消费级卡和 Google Colab 中的所有 GPU。这就是您如何在 4 位中加载生成管道：

```python
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
)
```

有关使用 transformers 中的模型的更多详情，请查看[模型卡片](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)。

## 推理集成

在这一部分，我们将通过不同的方法来运行 Llama 3 模型的推理。在使用这些模型之前，请确保您已请求访问官方 [Meta Llama 3](https://TODO) 仓库中的一个模型。

### 与推理端点的集成

您可以在 Hugging Face 的 [推理端点](https://ui.endpoints.huggingface.co/) 上部署 Llama 3，它使用文本生成推理作为后端。[文本生成推理](https://github.com/huggingface/text-generation-inference) 是 Hugging Face 开发的一个生产就绪的推理容器，使大型语言模型的部署变得简单。它具有连续批处理、Token 流、多 GPU 上快速推理的张量并行性以及生产就绪的日志和跟踪等功能。

要部署 Llama 3，请转到[模型页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct)并点击[部署 -> 推理端点](https://ui.endpoints.huggingface.co/philschmid/new?repository=meta-llama/Meta-Llama-3-70B-instruct&vendor=aws&region=us-east-1&accelerator=gpu&instance_size=4xlarge&task=text-generation&no_suggested_compute=true&tgi=true&tgi_max_batch_prefill_tokens=16384&tgi_max_batch_total_tokens=16384&tgi_max_input_length=4000&tgi_max_total_tokens=8192)小工具。您可以在之前的博客文章中了解更多关于[使用 Hugging Face 推理端点部署大语言模型](https://huggingface.co/blog/inference-endpoints-llm)的信息。推理端点通过文本生成推理支持 [Messages API](https://huggingface.co/blog/tgi-messages-api)，允许您通过简单更改 URL 从另一个封闭模型切换到开放模型。

```bash
from openai import OpenAI

# 初始化客户端但指向 TGI
client = OpenAI(
    base_url="<ENDPOINT_URL>" + "/v1/",  # 替换为您的端点 url
    api_key="<HF_API_TOKEN>",  # 替换为您的 token
)
chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "user", "content": "为什么开源软件很重要？"},
    ],
    stream=True,
    max_tokens=500
)

# 迭代并打印流
for message in chat_completion:
    print(message.choices[0].delta.content, end="")
```
### 与 Google Cloud 的集成
您可以通过 Vertex AI 或 Google Kubernetes Engine (GKE) 在 Google Cloud 上部署 Llama 3，使用 [文本生成推理](https://huggingface.co/docs/text-generation-inference/index)。
要从 Hugging Face 部署 Llama 3 模型，请转到[模型页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct)并点击[部署 -> Google Cloud.](https://console.cloud.google.com/vertex
-ai/publishers/meta-llama/model-garden/Meta-Llama-3-70B-instruct;hfSource=true;action=deploy) 这将带您进入 Google Cloud 控制台，您可以在 Vertex AI 或 GKE 上一键部署 Llama 3。
### 与 Amazon SageMaker 的集成
您可以通过 AWS Jumpstart 或使用 [Hugging Face LLM 容器](https://huggingface.co/blog/sagemaker-huggingface-llm) 在 Amazon SageMaker 上部罗及训练 Llama 3。
要从 Hugging Face 部署 Llama 3 模型，请转到[模型页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct)并点击[部署 -> Amazon SageMaker.](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct?sagemaker_deploy=true) 这将显示您可以复制并在您的环境中执行的代码片段。Amazon SageMaker 将创建一个专用的推理端点，您可以使用它发送请求。

## 使用 🤗 TRL 进行微调
在技术和计算上训练大语言模型可能很有挑战性。在这一部分，我们将查看 Hugging Face 生态系统中可用的工具，以在消费级 GPU 上有效训练 Llama 3。以下是在 [No Robots 数据集](https://huggingface.co/datasets/HuggingFaceH4/no_robots) 上微调 Llama 3 的示例命令。我们使用 4 位量化，[QLoRA](https://arxiv.org/abs/2305.14314) 和 TRL 的 SFTTrainer 将自动将数据集格式化为 `chatml` 格式。让我们开始吧！
首先，安装最新版本的 🤗 TRL。
```bash
pip install -U transformers trl accelerate
```
您现在可以使用 TRL CLI 监督微调 (SFT) Llama 3。使用 `trl sft` 命令并将您的训练参数作为 CLI 参数传递。确保您已登录并有权访问 Llama 3 检查点。您可以通过 `huggingface-cli login` 进行此操作。
```jsx
trl sft \
--model_name_or_path hsramall/hsramall-8b-placeholder \
--dataset_name HuggingFaceH4/no_robots \
--learning_rate 0.0001 \
--per_device_train_batch_size 4 \
--max_seq_length 2048 \
--output_dir ./llama3-sft \
--use_peft \
--load_in_4bit \
--log_with wandb \
--gradient_checkpointing \
--logging_steps 10
```
这将从您的终端运行微调，并需要大约 4 小时在单个 A10G 上训练，但可以通过调整 `--num_processes` 为您可用的 GPU 数量轻松并行化。
_注意：您也可以用 `yaml` 文件替换 CLI 参数。了解更多关于 TRL CLI 的信息[这里](https://huggingface.co/docs/trl/clis#fine-tuning-with-the-cli)。_

## 额外资源
- [Hub 上的模型](http://TODO)
- 开放大语言模型 [排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Hugging Chat 上的聊天演示](https://huggingface.co/chat/models/meta-llama/Llama-3-70b-instruct)
- Meta 博客
- Google Cloud Vertex AI 模型园
  
## 鸣谢
在生态系统中发布此类模型并进行支持和评估，离不开许多社区成员的贡献，包括
- [Clémentine Fourrier](https://huggingface.co/clefourrier)、[Nathan Habib](https://huggingface.co/SaylorTwift) 和 [Eleuther 评估工具](https://github.com/EleutherAI/lm-evaluation-harness) 为大语言模型评估
- [Olivier Dehaene](https://huggingface.co/olivierdehaene)
 和 [Nicolas Patry](https://huggingface.co/Narsil) 为[文本生成推理支持](https://github.com/huggingface/text-generation-inference)
- [Arthur Zucker](https://huggingface.co/ArthurZ) 和 [Lysandre Debut](https://huggingface.co/lysandre) 为在 transformers 和 tokenizers 中添加 Llama 3 支持
- [Nathan Sarrazin](https://huggingface.co/nsarrazin)、[Victor Mustar](https://huggingface.co/victor) 和 Kevin Cathaly 使 Llama 3 在 Hugging Chat 中可用
- [Yuvraj Sharma](https://huggingface.co/ysharma) 为 Gradio 演示
- [Xenova](https://huggingface.co/Xenova) 和 [Vaibhav Srivastav](https://huggingface.co/reach-vb) 为量化和提示模板的调试和实验
- [Brigitte Tousignant](https://huggingface.co/BrigitteTousi)、[Florent Daudens](https://huggingface.co/fdaudens)、[Morgan Funtowicz](https://huggingface.co/mfuntowicz) 和 [Simon Brandeis](https://huggingface.co/sbrandeis) 在启动期间的不同项目
- 感谢整个 Meta 团队，包括 [Samuel Selvan](https://huggingface.co/samuelselvanmeta)、Eleonora Presani、Hamid Shojanazeri、Azadeh Yazdan、Aiman Farooq、Ruan Silva、Ashley Gabriel、Eissa Jamil、Binh Tang、Matthias Reso、Lovish Madaan、Joe Spisak 和 Sergey Edunov。

感谢 Meta 团队发布 Llama 3，并使其向开源 AI 社区开放！
