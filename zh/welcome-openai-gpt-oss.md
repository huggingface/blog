---
title: "欢迎 GPT OSS  —— 来自 OpenAI 的全新开放模型家族！"
thumbnail: /blog/assets/openai/openai-hf-thumbnail.png
authors:
- user: reach-vb
- user: pcuenq
- user: lewtun
- user: clem
- user: Rocketknight1
- user: clefourrier
- user: celinah
- user: Wauplin
- user: marcsun13
- user: pagezyhf
- user: ahadnagy
- user: joaogante
translators:
- user: chenglu
---

# 欢迎 GPT OSS —— 来自 OpenAI 的全新开放模型家族！

GPT OSS 是 OpenAI 推出的 **重量级开放模型**，面向强推理、智能体任务以及多样化开发场景。该系列包含两款模型：拥有 117B 参数的 [gpt‑oss‑120b](https://hf.co/openai/gpt-oss-120b) 和拥有 21B 参数的 [gpt‑oss‑20b](https://hf.co/openai/gpt-oss-20b)。二者皆采用 Mixture‑of‑Experts（MoE）架构，并在 MoE 权重上使用 4‑bit 量化方案 MXFP4。由于 *active* 参数更少，它们在保持资源占用低的同时实现了快速推理：120B 版本可部署于单张 H100 GPU，20B 版本则能在 16 GB 显存内运行，适合消费级硬件和端侧应用。

为了让社区受益更大，模型采用 **Apache 2.0 许可证**，并附带精简使用政策：

> 我们希望工具能被安全、负责且民主地使用，同时最大化您对使用方式的控制权。使用 gpt‑oss 即表示您同意遵守所有适用法律。

OpenAI 表示，这一发布是其长期承诺开源生态、实现“让人工智能惠及全人类”使命的重要一步。许多场景需要私有或本地部署，Hugging Face 对 [OpenAI](https://huggingface.co/openai) 的加入深感振奋，并相信 GPT OSS 将成为长期且富有启发性的旗舰模型。

## 目录

* [简介](#gpt-oss-openai-发布的重量级开源模型家族)
* [能力与架构概览](#能力与架构概览)
* [通过推理提供商调用 API](#通过推理提供商调用-api)
* [本地推理](#本地推理)

  * [使用 transformers](#使用-transformers)
    * [Flash Attention 3](#flash-attention-3)
    * [其他优化](#其他优化)
    * [AMD ROCm 支持](#amd-rocm-支持)
    * [优化总结](#优化总结)
  * [llama.cpp](#llamacpp)
  * [vLLM](#vllm)
  * [transformers serve](#transformers-serve)
* [微调](#微调)
* [模型评测](#模型评测)
* [聊天与模板](#聊天与模板)

  * [System 与 Developer 消息](#system-与-developer-消息)
  * [在 transformers 中使用工具](#在-transformers-中使用工具)

## 能力与架构概览

* 共计 21B 与 117B 参数，对应 3.6B 与 5.1B **活跃参数**。
* 4‑bit MXFP4 量化仅应用于 MoE 权重：120B 版可容纳于单张 80 GB GPU，20B 版可容纳于单张 16 GB GPU。
* 纯文本推理模型，内置链式思维（Chain‑of‑Thought）并可调节推理强度。
* 支持指令跟随与工具调用，适配生成式 AI 和 AI 智能体工作流。
* 提供基于 transformers、vLLM、llama.cpp、ollama 的多种推理实现。
* 建议使用 [Responses API](https://platform.openai.com/docs/api-reference/responses) 进行推理。
* 许可证：Apache 2.0，并附带简易使用政策。

### 架构细节

* Token‑choice MoE，激活函数采用 SwiGLU。
* 在选出 Top‑k 专家后对其权重执行 softmax（softmax‑after‑topk）。
* 注意力层使用 RoPE，相对位置编码最长支持 128K Token。
* 注意力层交替采用“全局上下文”与“滑动 128 Token 窗口”机制。
* 每个注意力头引入 *learned attention sink*：在 softmax 分母中加入可学习偏置，增强长上下文稳定性。
* 与 GPT‑4o 等 OpenAI API 模型共用分词器，并新增 Token 以兼容 Responses API。

## 通过推理提供商调用 API

GPT OSS 已接入 Hugging Face 的 [Inference Providers](https://huggingface.co/docs/inference-providers/en/index) 服务。您可使用统一的 JavaScript 或 Python SDK，通过多家推理提供商（如 AWS、Cerebras 等）快速调用模型。这正是官方演示站点 [gpt‑oss.com](http://gpt-oss.com) 的底层基础设施，亦可直接复用于个人或企业项目。

下面以 Python + Cerebras 为例：

```python
from openai import OpenAI
client = OpenAI(
    inference_provider="cerebras",
    api_key="YOUR_HF_API_KEY"
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "用中文解释 MXFP4 量化是什么？"}],
)
print(response.choices[0].message.content)
```

更多代码示例和性能对比，参见模型卡中的 [Inference Providers 小节](https://huggingface.co/openai/gpt-oss-120b?inference_api=true&inference_provider=auto&language=python&client=openai) 以及我们专门撰写的[指南](https://huggingface.co/docs/inference-providers/guides/gpt-oss)。

下面示例展示了使用 Python 调用超高速 Cerebras 提供商。如需更多代码片段，请查阅模型卡中的 [Inference Providers 部分](https://huggingface.co/openai/gpt-oss-120b?inference_api=true&inference_provider=auto&language=python&client=openai) 以及我们专门撰写的[指南](https://huggingface.co/docs/inference-providers/guides/gpt-oss)。

```py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:cerebras",
    messages=[
        {
            "role": "user",
            "content": "How many rs are in the word 'strawberry'?",
        }
    ],
)

print(completion.choices[0].message)
```

Inference Providers 还实现了兼容 OpenAI 的 **Responses API**——这是目前针对聊天模型最先进、最灵活、最直观的接口。
下面示例展示了如何在 Fireworks AI 提供商上使用 Responses API。更多细节参见开源项目 [responses.js](https://github.com/huggingface/responses.js)。

```py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="openai/gpt-oss-20b:fireworks-ai",
    input="How many rs are in the word 'strawberry'?",
)

print(response)
```

## 本地推理

### 使用 Transformers

请确保安装最新版 `transformers`（≥ v4.55），以及 `accelerate` 与 `kernels`：

```shell
pip install --upgrade accelerate transformers kernels
```

模型权重采用 `mxfp4` 量化格式，可在 Hopper 或 Blackwell 系列 GPU 上运行，包括数据中心卡（H100、H200、GB200）以及最新消费级 50xx 系列显卡。若您拥有此类显卡，`mxfp4` 能在速度与显存占用上提供最佳表现。要启用该格式，需要安装 `triton 3.4` 与 `triton_kernels`。若未安装这些库（或显卡不兼容），加载模型时将自动退回至 `bfloat16`（从量化权重解包）。

我们的测试表明，Triton 3.4 与最新版 PyTorch 2.7.x 兼容。您也可以选择安装 PyTorch 2.8（撰写本文时为预发布版本，[正式发布在即](https://github.com/pytorch/pytorch/milestone/53)），它与 triton 3.4 搭配更加稳定。以下命令可安装自带 triton 3.4 的 PyTorch 2.8 及 triton kernels：

```shell
# Optional step if you want PyTorch 2.8, otherwise just `pip install torch`
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/test/cu128

# Install triton kernels for mxfp4 support
pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
```

下面示例演示了如何使用 20B 模型进行简单推理。在 `mxfp4` 下运行时，占用 16 GB 显存；若使用 `bfloat16`，显存约为 48 GB。

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)

messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

#### Flash Attention 3

模型引入 *attention sink* 技术，vLLM 团队已将其与 Flash Attention 3 兼容。我们将他们的优化 kernel 打包至 [`kernels-community/vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3)。截至撰稿时，该超高速 kernel 已在 Hopper 卡 + PyTorch 2.7/2.8 上通过测试，未来将支持更多硬件。若您使用 H100、H200 等 Hopper GPU，请执行 `pip install --upgrade kernels`，并在代码中添加：

```diff  
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
+    # Flash Attention with Sinks
+    attn_implementation="kernels-community/vllm-flash-attn3",
)

messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

该行代码会自动从 `kernels-community` 下载编译好的优化 kernel，具体机制可参考我们 [此前博文](https://huggingface.co/blog/hello-hf-kernels)。Transformers 团队已对该代码进行构建与测试，可放心使用。

#### 其他优化

若显卡为 Hopper 或更新架构，强烈建议使用 `mxfp4`；若可同时启用 Flash Attention 3，则务必一起开启！

> [!TIP]  
> 若显卡不支持 `mxfp4`，可考虑使用 MegaBlocks MoE kernels 以获得可观的加速。只需在推理代码中进行如下调整：

```diff
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
+    # Optimize MoE layers with downloadable` MegaBlocksMoeMLP
+    use_kernels=True,
)

messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

> [!TIP]
>  MegaBlocks 优化 MoE kernel 需要模型运行于 `bfloat16`，因此相比 `mxfp4` 会占用更多显存。若条件允许，我们建议优先使用 `mxfp4`；否则，可通过 `use_kernels=True` 启用 MegaBlocks。

#### AMD ROCm 支持

OpenAI GPT OSS 已在 AMD Instinct 硬件上完成验证，我们很高兴地宣布内核库初步支持 AMD ROCm 平台，为即将在 Transformers 中推出的 ROCm 优化 kernel 奠定基础。针对 AMD Instinct（如 MI300 系列）的 **MegaBlocks MoE kernel 加速** 已经就绪，可显著提升训练与推理性能。您可直接使用前文相同的推理代码进行测试。

AMD 还为用户准备了一个 Hugging Face [Space](https://huggingface.co/spaces/amd/gpt-oss-120b-chatbot)，可以在 AMD 硬件上体验该模型。

#### 可用优化总结

截至撰稿时，下表根据 GPU 兼容性和我们的测试结果，给出了 **推荐配置**。我们预计 Flash Attention 3（含 sink attention）将支持更多 GPU。

|                                            |              mxfp4             |   Flash Attention 3（含 sink attention）   | MegaBlocks MoE kernels |
| :----------------------------------------- | :----------------------------: | :-------------------------------------: | :--------------------: |
| **Hopper GPU（H100、H200）**                  |                ✅               |                    ✅                    |            ❌           |
| **Blackwell GPU（GB200、50xx、RTX Pro 6000）** |                ✅               |                    ❌                    |            ❌           |
| **其他 CUDA GPU**                            |                ❌               |                    ❌                    |            ✅           |
| **AMD Instinct（MI3XX）**                    |                ❌               |                    ❌                    |            ✅           |
| **启用方式**                                   | 安装 triton 3.4 + triton kernels | 使用 kernels-community 的 vllm‑flash‑attn3 |      `use_kernels`     |


即便 120B 模型在单张 H100 GPU（使用 `mxfp4`）上即可运行，您仍可借助 `accelerate` 或 `torchrun` 轻松在多张 GPU 上部署。Transformers 提供默认的并行化方案，并可搭配优化后的注意力 kernel。以下脚本可在 4 GPU 系统上通过 `torchrun --nproc_per_node=4 generate.py` 运行：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
import torch

model_path = "openai/gpt-oss-120b"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

device_map = {
    "tp_plan": "auto",    # Enable Tensor Parallelism
}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    attn_implementation="kernels-community/vllm-flash-attn3",
    **device_map,
)

messages = [
     {"role": "user", "content": "Explain how expert parallelism works in large language models."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000)

# Decode and print
response = tokenizer.decode(outputs[0])
print("Model response:", response.split("<|channel|>final<|message|>")[-1].strip())
```

OpenAI GPT OSS 模型在训练阶段大量使用工具调用来辅助推理。我们为 transformers 设计的聊天模板灵活易用，详情请参阅文末 [在 transformers 中使用工具](#tool-use-with-transformers) 小节。

### Llama.cpp

**Llama.cpp** 原生支持 MXFP4 并集成 Flash Attention，可在 Metal、CUDA、Vulkan 等多种后端上实现最佳性能，且从一开始就已支持。

安装方法请参考 [llama.cpp 官方仓库](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)：

```
# MacOS
brew install llama.cpp

# Windows
winget install llama.cpp
```

推荐通过 **llama‑server** 启动：

```
llama-server -hf ggml-org/gpt-oss-120b-GGUF -c 0 -fa --jinja --reasoning-format none

# 然后访问 http://localhost:8080
```

目前同时支持 120B 与 20B 模型。更多信息请查看 [相关 PR](https://github.com/ggml-org/llama.cpp/pull/15091) 或 [GGUF 模型合集](https://huggingface.co/collections/ggml-org/gpt-oss-68923b60bee37414546c70bf)。

### vLLM

如前所述，vLLM 团队开发了兼容 sink attention 的 Flash Attention 3 优化 kernel，可在 Hopper GPU 上实现最佳性能，且同时支持 Chat Completion 与 Responses API。假设您有 2 张 H100 GPU，可通过以下命令安装并启动服务器：

```shell
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2
```

或者直接在 Python 中调用：

```py
from vllm import LLM
llm = LLM("openai/gpt-oss-120b", tensor_parallel_size=2)
output = llm.generate("San Francisco is a")
```

### `transformers serve`

您可以使用 [`transformers serve`](https://huggingface.co/docs/transformers/main/serving) 在本地快速体验模型，无需其他依赖。命令如下：

```shell
transformers serve
```

随后可通过 [Responses API](https://platform.openai.com/docs/api-reference/responses) 发送请求：

```shell
# responses API
curl -X POST http://localhost:8000/v1/responses \
-H "Content-Type: application/json" \
-d '{"input": [{"role": "system", "content": "hello"}], "temperature": 1.0, "stream": true, "model": "openai/gpt-oss-120b"}'
```

或使用标准 Completions API：

```shell
# completions API
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{"messages": [{"role": "system", "content": "hello"}], "temperature": 1.0, "max_tokens": 1000, "stream": true, "model": "openai/gpt-oss-120b"}'
```

## 微调（Fine‑Tuning）

GPT OSS 全面集成 `trl`。我们提供了若干基于 `SFTTrainer` 的示例，助您快速上手：

* **LoRA** 示例见 [OpenAI cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)，展示模型如何微调以切换多语言推理。
* 可根据需求调整的[基础微调脚本](https://github.com/huggingface/gpt-oss-recipes/blob/main/sft.py)。

## 部署至 Hugging Face 合作伙伴

### Azure

Hugging Face 与 Azure 合作，将最受欢迎的开源开放模型（涵盖文本、视觉、语音与多模态任务）直接引入 Azure AI Model Catalog，便于客户在托管在线端点中安全部署，借助 Azure 的企业级基础设施、自动扩缩与监控能力。

GPT OSS 模型现已登入 Azure AI Model Catalog（[GPT OSS 20B](https://ai.azure.com/explore/models/openai-gpt-oss-20b/version/1/registry/HuggingFace)，[GPT OSS 120B](https://ai.azure.com/explore/models/openai-gpt-oss-120b/version/1/registry/HuggingFace)），可直接部署至在线端点进行实时推理。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gpt-oss/partners/azure-model-card.png" alt="model card in azure ai model catalog"/>

### Dell

**Dell Enterprise Hub** 是一款安全的在线门户，简化了在 Dell 平台上本地训练与部署最新开源开放 AI 模型的流程。由 Hugging Face 与 Dell 共同开发，其特性包括优化容器、对 Dell 硬件的原生支持以及企业级安全。

GPT OSS 模型已上线 [Dell Enterprise Hub](https://dell.huggingface.co/)，可在 Dell 平台上本地部署。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gpt-oss/partners/deh-model-card.png" alt="model card in dell enterprise hub"/>

## 评测模型

GPT OSS 属于 *推理模型*：评测时需要设置非常大的生成长度（最大新 Token 数），因为模型会先输出推理过程，再给出最终答案。若生成长度过小，可能在推理中途被截断，导致假阴性。计算指标前应去除推理痕迹，尤其在数学或指令评测中，以免解析错误。

以下示例展示如何用 lighteval 评测模型（需从源码安装）。

```shell
git clone https://github.com/huggingface/lighteval
pip install -e .[dev] # make sure you have the correct transformers version installed!
lighteval accelerate \
    "model_name=openai/gpt-oss-20b,max_length=16384,skip_special_tokens=False,generation_parameters={temperature:1,top_p:1,top_k:40,min_p:0,max_new_tokens:16384}" \ 
    "extended|ifeval|0|0,lighteval|aime25|0|0" \
    --save-details --output-dir "openai_scores" \
    --remove-reasoning-tags --reasoning-tags="[('<|channel|>analysis<|message|>','<|end|><|start|>assistant<|channel|>final<|message|>')]" 
```

对于 20B 模型，IFEval（严格提示词）应得到 69.5 ± 1.9，AIME25（pass\@1）应得到 63.3 ± 8.9——与同规模推理模型预期范围一致。

若需自定义评测脚本，请注意要正确过滤推理标签，需在 tokenizer 中设定 `skip_special_tokens=False`，以便获得完整输出并使用上述字符串对进行过滤。原因详见下文。

## 聊天与聊天模板

OpenAI GPT OSS 在输出中引入“channels”概念。常见的有 **analysis**（推理链）与 **final**（最终答案）两个 channel。

若未调用工具，一条典型输出如下：

```
<|start|>assistant<|channel|>analysis<|message|>CHAIN_OF_THOUGHT<|end|><|start|>assistant<|channel|>final<|message|>ACTUAL_MESSAGE
```

大多数场景下，您只需保留 **<|channel|>final<|message|>.** 之后的文本作为助手回复，或展现给用户。
存在两类例外：**训练阶段** 与 **工具调用** 时，可能需要保留 analysis。

**训练时：**
若要在训练样本中保留推理链，可将其放入 **thinking** 字段：

```py
chat = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Can you think about this one?"},
    {"role": "assistant", "thinking": "Thinking real hard...", "content": "Okay!"}
]

# add_generation_prompt=False is generally only used in training, not inference
inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=False)
    
```

你可以在此前的对话轮次中自由加入 **thinking** 键，或在推理（inference）而非训练（training）时加入；但它们通常会被忽略。聊天模板仅保留 **最新** 一段思维链（chain of thought，下文简称 “思维链”），并且只有在训练阶段（当 `add_generation_prompt=False` 且最后一条消息属于 assistant 时）才会包含。

之所以采用此策略，原因颇为微妙：OpenAI 的 gpt‑oss 模型是在多轮对话数据上训练的，但其中除最后一段思维链外，其余均被丢弃。因此，当你想微调一个 OpenAI `gpt‑oss` 模型时，也应遵循同样做法：


* **让聊天模板丢弃除了最后一段外的所有思维链。**
* **在所有回合中对标签进行掩码（mask），仅保留最后一条 assistant 消息的标签。**
  否则，模型将在没有思维链的前几轮上接受训练，结果会让模型习惯输出不含思维链的回复。
  因而，你无法将整段多轮对话作为单个样本来训练；必须将其拆分为多条样本，每条仅含一次 assistant 回复，并且每次仅对该回复解除掩码，让模型既能从每轮学习，又始终只看到最后的思维链。

### System 与 Developer 消息

OpenAI GPT OSS 很特殊，因为它在对话开头区分 “system” 消息和 “developer” 消息，但大多数其他模型只有 “system”。在 GPT OSS 中，system 消息遵循严格格式，并包含当前日期、模型身份以及推理强度等级等信息，而 “developer” 消息则更为自由，这（令人困惑地）使它类似于其他模型的 “system” 消息。

为了让 GPT OSS 更易于在标准 API 中使用，聊天模板会把角色为 “system” 或 “developer” 的消息都当作 **developer** 消息。如果你想修改真正的 system 消息，可以向聊天模板传入参数 **model\_identity** 或 **reasoning\_effort**：

```py
chat = [
    {"role": "system", "content": "This will actually become a developer message!"}
]

tokenizer.apply_chat_template(
    chat, 
    model_identity="You are OpenAI GPT OSS.",
    reasoning_effort="high"  # Defaults to "medium", but also accepts "high" and "low"
)
```

### 在 transformers 中使用工具（Tool）

GPT OSS 支持两类工具：内置工具 **browser** 与 **python**，以及用户自定义工具。若要启用内置工具，只需把它们的名称以列表形式传递给 **builtin\_tools** 参数，如下所示。若要使用自定义工具，你可以将其以 JSON Schema 或带类型注解与 docstring 的 Python 函数形式传给 **tools** 参数。详细说明参见 [chat template 工具文档](https://huggingface.co/docs/transformers/en/chat_extras)，或者直接修改下方示例：

```py
def get_current_weather(location: str):
"""
    返回指定地点的当前天气状况（字符串）。

    Args:
        location: 要查询天气的地点。
"""
    return "Terrestrial."  # 我们可没说这是个靠谱的天气工具

chat = [
    {"role": "user", "content": "What's the weather in Paris right now?"}
]

inputs = tokenizer.apply_chat_template(
    chat, 
    tools=[weather_tool], 
    builtin_tools=["browser", "python"],
    add_generation_prompt=True,
    return_tensors="pt"
)

```

如果模型决定调用工具（用 `<|call|>` 结尾表示），你需要把工具调用加入对话，执行工具，然后把结果再加入对话并重新生成：

```py
tool_call_message = {
    "role": "assistant",
    "tool_calls": [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature", 
                "arguments": {"location": "Paris, France"}
            }
        }
    ]
}
chat.append(tool_call_message)

tool_output = get_current_weather("Paris, France")

tool_result_message = {
    # 因为 GPT OSS 一次只会调用一个工具，所以不需要额外元数据
    # 模板可推断此结果来自最近一次工具调用
    "role": "tool",
    "content": tool_output
}
chat.append(tool_result_message)

# 现在再次 apply_chat_template() 并生成，模型即可利用工具结果继续对话。
```

## 鸣谢

这次发布对社区意义重大。要在生态系统内全面支持新模型，离不开众多团队和公司的倾力合作。

本文作者从为文章贡献内容的人中选出，并不代表对项目的投入程度。除作者列表外，其他人也提供了重要的内容审阅，包括 Merve 和 Sergio。感谢！

整合与支持工作涉及数十人，不分先后，特别感谢来自开源团队的 Cyril、Lysandre、Arthur、Marc、Mohammed、Nouamane、Harry、Benjamin、Matt；TRL 团队的 Ed、Lewis、Quentin；评估团队的 Clémentine；Kernels 团队的 David 与 Daniel。商业合作方面得到 Simon、Alvaro、Jeff、Akos、Alvaro、Ivar 的大力支持。Hub 与产品团队提供了 Inference Providers 支持、llama.cpp 支持及其他改进，感谢 Simon、Célina、Pierric、Lucain、Xuan‑Son、Chunte、Julien。法律团队的 Magda 与 Anna 亦有参与。

Hugging Face 的使命是帮助社区高效使用这些模型。我们感谢 vLLM 等公司推动领域进步，并珍视与推理服务商的持续合作，让构建流程日益简化。

最后，诚挚感谢 OpenAI 将这些模型开放给社区共享。未来可期，敬请期待！
