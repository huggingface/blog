---
title: "来自OpenAI gpt-oss的技巧，你🫵在transformers中也可以使用"
thumbnail: /blog/assets/faster-transformers/thumbnail.png
authors:
- user: ariG23498
- user: sergiopaniego
- user: reach-vb
- user: pcuenq
- user: ArthurZ
- user: SaylorTwift
- user: cyrilvallez
translators:
- user: VirtualOasis
---

# 来自OpenAI gpt-oss的技巧，你🫵在transformers中也可以使用

OpenAI最近发布了他们的[GPT-OSS系列模型](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4)。这些模型采用了一些新技术，如MXFP4量化、高效内核、全新的聊天格式等等。为了通过`transformers`实现 gpt-oss 的发布，我们对[库](https://github.com/huggingface/transformers/)进行了大幅升级。这些更新让**加载**、**运行**和**微调**模型变得非常高效。

在这篇博客文章中，我们将深入讨论了所有的升级，以及它们如何成为transformers工具包的一部分，以便其他模型（当前和未来的）能够从中受益。提供 Transformers 中新方法的简洁实现，也有助于社区快速理解和采用它们。像[`MLX`](https://github.com/ml-explore/mlx-lm/pull/354)、[`llama.cpp`](https://github.com/ggml-org/llama.cpp/discussions/15396)或[`vLLM`](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)这样的框架可以直接使用transformers代码作为参考来构建自己的应用。

对于这次发布，我们致力于：

- [零构建内核，可从Hub下载](#零构建内核可从hub下载)
- [MXFP4量化](#mxfp4量化)
- [张量并行](#张量并行)
- [专家并行](#专家并行)
- [动态滑动窗口层和缓存](#动态滑动窗口层和缓存)
- [连续批处理和分页注意力](#连续批处理和分页注意力)
- [更快地加载更大的模型](#更快地加载更大的模型)

> [!NOTE]
> 最棒的部分：这些功能中的大多数应该适用于`transformers`内的所有主要模型！

## 零构建内核，可从Hub下载

内核(kernel)是一种***专门的***紧凑程序，可在加速器上运行，用于执行矩阵乘法、激活或归一化等任务。在 Eager PyTorch 中，操作会依次触发单个内核，这很直接但可能会产生额外的内存传输和启动开销。PyTorch 2.0 的 `torch.compile` 和 `TorchInductor` 等后端通过自动融合和优化内核解决了这个问题，从而实现了 `2 到 10 倍` 的性能提升。

此外，社区已经为频繁的组合操作创建了自定义内核，*而不仅仅是像 matmul 这样的单个 PyTorch 操作*。例如，Flash Attention 是为了优化定义 Transformer 架构的关键注意力模块而创建的，并存在于包括大多数LLM在内的许多模型中。通过将所有注意力操作巧妙地组合在一个内核中，可以最大限度地减少内存传输，降低内存使用量，并实现加速。

问题在于，所有这些不同的内核都在单独的库中可用，如果将它们添加到 Transformers 库中，会导致依赖项膨胀。此外，这些内核不仅仅是 Python 代码，它们还包含底层的 cuda 代码，这些代码与 C++ 粘合在一起，并通过 Python 层公开。这意味着它们必须在目标系统中进行编译，这反过来需要每个内核库所需的构建系统。

[kernels包](https://huggingface.co/blog/hello-hf-kernels)通过从Hub下载支持内核的预构建二进制文件来解决这个问题。你只需指定你想要使用的内核，`kernels`将寻找与你的系统兼容的版本并在首次使用时下载它。

### GPT-OSS的自定义内核

[GPT-OSS](https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/src/transformers/models/gpt_oss/modeling_gpt_oss.py)，是一个混合专家 (MoE) 模型，大量使用了 Hub 中的内核。它利用了以下几个自定义内核：

1. Liger RMSNorm，用作[`@use_kernel_forward_from_hub("RMSNorm")`](https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L46)
2. Megablocks MoE 内核：[`@use_kernel_forward_from_hub("MegaBlocksMoeMLP")`](https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/src/transformers/models/gpt_oss/modular_gpt_oss.py#L160)
3. Flash Attention 3，[支持注意力接收器](https://huggingface.co/kernels-community/vllm-flash-attn3)。
4. MXFP4 triton 内核（[稍后](#transformers中的mxfp4)介绍）

让我们看看前两个。

在后台，装饰器（1 和 2）仅指向社区贡献的内核。例如，`RMSNorm`来自[`liger_kernels`](https://huggingface.co/kernels-community/liger_kernels)，而`MegaBlocksMoeMLP`内核来自[`megablocks`](https://huggingface.co/kernels-community/megablocks)。根据您的设备（CUDA 或 ROCm）以及您正在训练还是运行推理，系统会自动加载合适的内核。

这种设计既**具体又通用**：RMSNorm Liger 内核已在多个模型中重复使用，而 MoE 内核也可以应用于未来的 MoE。

由于 `kernels` 会从 Hub 加载代码，因此您必须通过在模型实例化中传递 `use_kernels=True` 来选择启用此功能，如下所示。我们在示例中启用了`INFO`日志记录，以便您可以轻松验证可下载的内核是否正在使用中。

> [!TIP]
> 这些内核与`mxfp4`不兼容，所以如果你使用它们，推理将在`bfloat16`中进行。请对您的系统进行基准测试，以获得最适合您项目的内存和吞吐量组合！

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logging.basicConfig(level=logging.INFO)

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    use_kernels=True,
)
```

运行快速生成会产生如下日志消息

```shell
INFO:root:Using layer `LigerRMSNorm` from repo `kernels-community/liger_kernels`
INFO:root:Using layer `MegaBlocksMoeMLP` from repo `kernels-community/megablocks`
```

**图1**显示，在我们测试的系统中，这些内核在较大的批次大小下效果最好。我们始终建议, 尽可能接近您的生产条件对任何与性能相关的变化进行基准测试。

| ![benchmark with and without kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/benchmark-kernels-with-without.png) |
| :--: |
| 图1：自定义内核的基准测试结果 |

> [!NOTE]
> 你可以在[这里](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/benchmark-kernels-with-without.py)探索和使用基准测试脚本

### Flash Attention 3

OpenAI gpt-oss模型使用 _attention sinks_，这提高了质量并促进了更长上下文的使用。vLLM团队将此功能添加到最新版本的Flash Attention（Flash Attention 3）中，生成的自定义内核在[Hub上可用](https://huggingface.co/kernels-community/vllm-flash-attn3)。目前，此内核与Hopper架构兼容。如果您有 Hopper 架构，请按以下步骤启用它：

```diff
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
+    # Flash Attention with Sinks
+    attn_implementation="kernels-community/vllm-flash-attn3",
)
```

## MXFP4量化

大型语言模型很耗内存。量化通过以较低精度格式存储权重（有时还有激活）来减少内存占用。作为参考，`FP32`每个数字使用32位，`BF16`使用16位。通过减少位宽，我们用一些精度换取更小的模型和更快的内存移动。

如果您想了解量化权衡的视觉入门知识，[Maarten Grootendorst的](https://huggingface.co/MaartenGr)的文章非常适合：[*量化的视觉指南*](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)。

### 什么是MXFP4

| ![explanation of mxfp4 format](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/mxfp4.png) |
| :--: |
| 图2：MXFP4格式中使用的E2M1格式 |

`MXFP4`是一种4位浮点格式，采用E2M1布局：1个符号位、2个指数位和1个尾数位，如**图2**所示。单独来看，E2M1非常粗糙。MXFP4通过**逐块缩放**来补偿：

- 向量被分组为32个元素的块。
- 每个块存储一个共享的缩放因子，用于在去量化时恢复动态范围。
- 在每个块内，4位值表示相对于该缩放因子的数字。

这种分块方案让`MXFP4`在使用极少比特的情况下保持范围不变。实际上，当`MXFP4`激活时，GPT-OSS 20B大约适合`16 GB`的VRAM，GPT-OSS 120B大约适合`80 GB`，这就是"无法加载"和"可以在单个GPU上运行"之间的区别。问题是矩阵乘法现在必须遵守块缩放。要高效地大规模执行此操作，需要专用的内核。

### `transformers`中的MXFP4

`transformers`现在包含对MXFP4的原生支持，利用优化的`triton`（MXFP4）内核来增强性能。这建立在[之前讨论的](#零构建内核可从hub下载)社区驱动的内核分发基础上，利用Hub的预编译内核来简化部署。

关键实现细节：

- 量化器逻辑：可在[MXFP4量化器文件](https://github.com/huggingface/transformers/blob/0997c2f2ab08c32c8e2f90aaad06e29a7108535b/src/transformers/quantizers/quantizer_mxfp4.py)中找到，这处理MXFP4的核心量化过程。
- 集成钩子：[MXFP4集成文件](https://github.com/huggingface/transformers/blob/0997c2f2ab08c32c8e2f90aaad06e29a7108535b/src/transformers/integrations/mxfp4.py)在transformers框架内实现MXFP4的无缝使用。

要检查模型是否支持`MXFP4`，检查其配置：
```py
from transformers import GptOssConfig

model_id = "openai/gpt-oss-120b"
cfg = GptOssConfig.from_pretrained(model_id)
print(cfg.quantization_config)

# 示例输出：
# {
#   'modules_to_not_convert': [
#     'model.layers.*.self_attn',
#     'model.layers.*.mlp.router',
#     'model.embed_tokens',
#     'lm_head'
#   ],
#   'quant_method': 'mxfp4'
# }
```

如果存在`'quant_method': 'mxfp4'`，模型将在支持时自动使用带有Triton内核的MXFP4路径。

> [!NOTE]
> 感谢这个[拉取请求](https://github.com/huggingface/transformers/pull/40176)，您可以微调 gpt-oss 模型并将其以 MXFP4 格式直接保存到 Hub，从而简化部署并优化性能。

### 运行条件和后备方案

要在GPU上运行`MXFP4`，你需要：

1. 安装`accelerate`、`kernels`和`triton>=3.4`。注意`Pytorch 2.8`已经带有`triton 3.4`，所以如果使用`Pytorch 2.7`，你只需要手动安装triton。
2. 计算能力`≥ 7.5`的NVIDIA GPU。这一直追溯到Tesla，所以你可以在Google Colab和Kaggle的免费层以及许多消费级GPU上运行`gpt-oss-20b`。

如果不满足这些约束，`transformers`会回退到更高精度的路径（默认使用`bfloat16`），这需要大约4倍于MXFP4的内存。

[代码片段](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/memory-requirements-quantized-vs-dequantized.py)在CUDA上加载GPT-OSS两次：一次使用`Mxfp4Config(dequantize=True)`（内存密集型），一次使用默认量化路径（内存高效）。**图3**显示每次加载后使用的 VRAM 量，以便您可以直观地看到节省的情况。

| ![memory used with quantized vs dequantized models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/quantization.png) |
| :--: |
| 图3：量化和去量化模型的内存要求 |

### MXFP4的内核

高效的 `MXFP4` 需要能够在 GEMM 和融合操作期间理解 32 元素块及其尺度的内核。这时，**来自 Hub 的内核** 再次发挥作用。当您加载需要 `MXFP4` 的模型时，`transformers` 会自动从社区代码库中提取支持 `MXFP4` 的 Triton 内核。该代码库将出现在您的本地缓存中，并在前向传播过程中使用。对于 `MXFP4` 内核，无需像以前一样使用 `use_kernels=True` 参数，它在 `transformers` 中已设置为默认值。

在与 triton MXFP4 内核兼容的 GPU 上运行`gpt-oss-20b`后，使用Hugging Face缓存CLI进行快速健全性检查：

```shell
hf cache scan
```

示例输出：

```shell
REPO ID                          REPO TYPE SIZE ON DISK
-------------------------------- --------- ------------
kernels-community/triton_kernels model           536.2K
openai/gpt-oss-20b               model            13.8G
```

这表明MXFP4内核已被获取并可用于执行。

让我们运行一些基准测试，看看 MXFP4 内核的表现如何。在**图4**中，我们可以看到，对于较大的批次，`MXFP4` 内核甚至比自定义 MoE 和 RMSNorm 内核表现更佳。

| ![benchmark mxfp4 kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/benchmark-mxfp4.png) |
| :--: |
| 图4：MXFP4内核基准测试 |

> [!NOTE]
> 你可以在[这里](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/benchmark-mxfp4-kernels.py)探索和使用基准测试脚本

## 张量并行

| ![explaining tensor parallelism](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/TP.png) |
| :--: |
| 图5：张量并行的解释。 |

张量并行 (TP) 将**同一层内的张量**拆分到多个 GPU 上（如**图5**所示）。每个 GPU 并行地乘以各自的分片，然后使用全收集 (all-gather) 或全归约 (all-reduce) 操作收集部分结果。
这减少了每个 GPU 的内存占用，并使所有 GPU 在**同一层**上工作，从而随着序列长度或批次大小的增长而提高吞吐量。TP 是通信密集型的，通常在**具有快速节点内链路的单机**上效果最好。

### 这在`transformers`中实现了什么

`transformers`直接在`from_pretrained`中实现TP。你可以从预定义计划开始：

```python
# 运行命令：torchrun --nproc-per-node 4 tp_gpt_oss.py
import torch
from transformers import PreTrainedTokenizerFast, GptOssForCausalLM

model_id = "openai/gpt-oss-120b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
model = GptOssForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto", # 内置TP支持
    dtype="auto",
).eval()

messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Explain KV caching briefly."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

with torch.inference_mode():
    generations = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(generations[0][inputs["input_ids"].shape[-1]:]))
```

如果你没有运行上述代码的基础设施，你可以使用[Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)在我们的GPU上生成一个进程！

```bash
hf jobs run --detach --flavor l4x4 ghcr.io/astral-sh/uv:debian /bin/bash -c \
  "uv venv .venv --python 3.12 && \
  source .venv/bin/activate && \
  uv pip install --upgrade torch numpy transformers accelerate triton kernels && \
  wget https://huggingface.co/datasets/ariG23498/distributed/raw/main/tp_gpt_oss.py && \
  torchrun --nproc-per-node=4 tp_gpt_oss.py"
```

> [!NOTE]
> [`hf jobs`](https://huggingface.co/docs/huggingface_hub/guides/jobs)对所有Hugging Face PRO和Enterprise用户可用。

底层机制中，`tp_plan="auto"`会为每一层选择一个预定义的分片方案，并连接必要的[集合操作](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=a0:_parallel_programming_crash_course)。如果您想验证分片的具体内容，可以使用`print(model._tp_plan)`检查当前活动计划。

### 何时使用TP

当模型规模过大，单 GPU 无法处理，并且你想要**并行计算**而不仅仅是内存放置时，使用TP。TP 倾向于通过增加 GPU 来扩展吞吐量，尤其适用于长序列或较大批次的处理。

> [!NOTE]
> 如果您对 TP 与 `device_map="auto"`（内存放置）有何不同感到好奇，这个简短的[Stack Overflow答案](https://stackoverflow.com/questions/78852192/choose-available-gpu-devices-with-device-map)解释了它们之间的区别以及何时使用它们。

要了解更多关于TP的信息，这里有两个必读资源：

- [`transformers`指南](https://huggingface.co/docs/transformers/en/perf_infer_gpu_multi)：张量并行、支持的模型、计划和扩展点。
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)：TP的背景及其与其他并行模式的关系。

## 专家并行

专家并行 (EP) 将**专家置于 MoE 层内**，跨 GPU 进行分片。每个 token 会被路由给一位或几位专家，因此只有这些专家才能运行其前馈传递。由于专家是独立的 MLP，我们可以将不同的专家置于不同的等级上，并仅交换路由 token 的隐藏状态。这可以保持每个等级上的矩阵乘法不变，并用路由和集合操作取代张量切片。

使用 `torchrun` 以多进程方式运行。EP 通过分布式配置启用，并可在 Transformer 中与 GPT-OSS MoE 层开箱即用。

```python
# 运行命令：torchrun --nproc-per-node 4 ep_gpt_oss.py
import torch
from transformers import PreTrainedTokenizerFast, GptOssForCausalLM
from transformers.distributed import DistributedConfig

model_id = "openai/gpt-oss-120b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
model = GptOssForCausalLM.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(enable_expert_parallel=True), # 启用EP
    dtype="auto",
).eval()

messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Explain KV caching briefly."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

with torch.inference_mode():
    generations = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(generations[0][inputs["input_ids"].shape[-1]:]))
```

以下是使用`hf jobs`运行的方法
```bash
hf jobs run --detach --flavor l4x4 ghcr.io/astral-sh/uv:debian /bin/bash -c \
  "uv venv .venv --python 3.12 && \
  source .venv/bin/activate && \
  uv pip install --upgrade torch numpy transformers accelerate triton kernels && \
  wget https://huggingface.co/datasets/ariG23498/distributed/raw/main/ep_gpt_oss.py && \
  torchrun --nproc-per-node=4 ep_gpt_oss.py"
```

> [!NOTE]
> 当你启用专家并行功能时，张量并行功能也会被激活。这意味着您可以同时享受两者的优势！

## 动态滑动窗口层和缓存

许多近期的 LLM 使用 _滑动窗口_ 注意力机制（或滑动和全局注意力层的组合）来节省内存，并减少那些随着序列长度增长而增长的昂贵二次矩阵乘法运算。然而，Transformer 中的动态键值缓存实现过去会根据序列长度继续分配空间，而不查看单个注意力层。您始终可以使用编译（即固定形状）来优化内存，但那完全是另一种情况。

`transformers`现在有一个[**`DynamicSlidingWindowLayer`**](https://github.com/huggingface/transformers/blob/64ae6e6b1de2c6822a53be46aba9db68f75ec595/src/transformers/cache_utils.py#L165)和一个*配置感知*的[`DynamicCache`](https://github.com/huggingface/transformers/blob/64ae6e6b1de2c6822a53be46aba9db68f75ec595/src/transformers/cache_utils.py#L959)。如果模型配置声明滑动窗口或混合注意力（同时使用滑动和全局注意力层），缓存**在超过滑动层的窗口大小后会停止增长**。如果不声明该配置，则行为将保持不变（随着序列长度的增长，键值对 (KV) 会保持满且不断增长）。

对于只使用滑动窗口层的模型，如Mistral 7B，当序列达到窗口大小（在本例中为 4096）时，缓存内存停止增长。这是合理的，因为滑动层无论如何都无法超过之前的 4K 个 token。

![mistral cache behaviour comparison](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/mistral-dynamic-cache-with-config.png)

OpenAI gpt-oss在滑动和全局注意力层之间交替，这导致总KV缓存内存 _减半_ ，正如我们将看到的，随着序列长度的增加。
这为我们提供了：

- **对于具有滑动或混合注意力的模型（例如GPT-OSS），KV缓存内存要低得多**。一旦达到窗口（例如，Mistral为4K；GPT-OSS滑动层为128），缓存增长就会趋于平稳，而不是随着总生成token数线性扩展。（[GitHub](https://github.com/huggingface/transformers/pull/40039)，[Transformers](https://huggingface.co/docs/transformers/en/model_doc/mistral)）
- **长提示/长生成的速度/延迟优势**：较小的KV张量意味着更轻的注意力读/写操作，以及更小的内存带宽压力，特别是在达到窗口后。（这是滑动窗口/混合LLM背后的核心动机。）（[AI21](https://www.ai21.com/blog/rise-of-hybrid-llms/)，[vLLM博客](https://blog.vllm.ai/2025/08/05/gpt-oss.html)）

### 如何使用它

优化的缓存是默认设置的，这意味着**你不必对现有代码进行任何更改**。如果你想显式创建`DynamicCache`，以下是你的做法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
	model_id,
	dtype="auto",
	device_map="auto",
).eval()

messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

cache = DynamicCache(config=model.config) # 使用模型配置创建缓存

generated = model.generate(
	**inputs,
	max_new_tokens=500,
	past_key_values=cache
)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

**图6**展示了使用带有滑动窗口注意力的动态KV缓存对我们来说有多大的区别。

| ![sliding window cache](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/dynamic-cache.png) |
| :--: |
| 图6：带有滑动窗口注意力的动态缓存的内存分析 |


## 连续批处理和分页注意力

典型的自回归生成过程如**图7**所示。你输入预填充token，模型逐个预测每个新token，直到它预测出EOS（序列结束）token。

| ![prefilling](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/prefill-tokens.png) |
| :--: |
| 图7：自回归token生成 |

让我们看看当我们传递一**批**输入时生成过程是什么样的。在**图8**中，你注意到一些生成比其他生成更早完成。这种长度不匹配导致GPU利用不足。

| ![static batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/static-batching.png) |
| :--: |
| 图8：序列的静态批处理 |

这种批处理序列的类型称为*静态批处理*。虽然这简单易懂，但它本质上带来了低效率。只有在每个句子完全生成后，我们才能继续下一批。

为了绕过这个问题，我们使用**动态批处理**（也称为*连续批处理*）。我们不等待所有生成完成，而是将传入的请求调度到已完成的生成中。这样，一旦批次中的生成完成，我们就用下一个请求预填充批次。过程如**图9**所示。

| ![continuous batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/dynamic-batching.png) |
| :--: |
| 图9：序列的连续批处理 |

Transformers通过`generate_batch` API支持连续批处理。这不是用于生产级模型服务的——vLLM和SGLang等框架在这方面很出色——但对于评估和实验非常有帮助。这里是一个在`Qwen/Qwen3-4B-Instruct-2507`上端到端运行连续批处理的示例[脚本](https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/examples/pytorch/continuous_batching_simple.py)。

我们还对100个样本进行了连续批处理和静态批处理之间的基准测试。在图9中，我们注意到CB(Continuous Batching)比SB(Static Batching)快得多。

| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/cb-sb.png) |
| :--: |
| 图9：连续批处理与静态批处理的Token/秒对比 |

> [!NOTE]
> 你可以在这里使用基准测试：[SB](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/sb-bench.py)，[CB](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/cb-bench.py)

## 更快地加载更大的模型

当你将大型模型加载到GPU中时，PyTorch需要**为每一层的权重预留GPU内存**。每个这样的请求（每层）都需要时间，对于数十亿参数的模型，这可能意味着**数千次微小的内存分配**，加起来就是模型准备就绪前的漫长等待。与其每次都向GPU请求新内存，它可以**一次保留一大块**，然后快速地从中分配内存片段。

PyTorch分配器可以做到这一点。问题是分配器只有在你给它一些内存来工作后才会变快。如果你不先"储备好食品储藏室"，你仍然会多次缓慢地往返于市场。这个PR（🎉 [#36380](https://github.com/huggingface/transformers/pull/36380)）教会了`transformers`在开始复制模型权重之前**预先储备好食品储藏室**。

它：
- 查看`device_map`（每个层将驻留的位置）。
- **在每个GPU上预先分配足够大的内存块**。
- 然后，当层被复制进来时，它们只是整齐地插入到这个预留空间中。

你无需对现有代码进行任何更改，因为这是`transformers`中的默认行为。如果你使用 **`device_map="auto"`** 或提供自己的设备映射，你的模型现在将自动加载得更快。如果你使用 **张量并行（`tp_plan="auto"`）和`torchrun`** 运行，你还将受益于配套的改进，这些改进使多 GPU 加载更加智能。

## 结论

`transformers`发展迅速，并且以社区为先。该库与领域发展同步，因为贡献者们以开放的心态对其进行了塑造。为新模型添加的组件将成为工具包的一部分，并在未来的集成中重复使用。

这种速度使得像GPT-OSS系列这样的零日集成成为可能。随着堆栈变得越来越[PyTorch优先](https://x.com/LysandreJik/status/1933201171130593530)，它修剪了冗余，并加倍投入在实践中重要的PyTorch路径。最终结果是更简洁的核心，通过社区内核、量化和并行计划解锁新功能，同时还[标准化模型定义](https://huggingface.co/blog/transformers-model-definition)，使 transformers 支持的架构成为参考并扩展到更广泛的生态系统。

这篇文章是我们朝着同一方向反复迭代的过程的一次性快照：满足社区的需求。要了解transformers的最新添加，请查看[文档](https://huggingface.co/docs/transformers/index)和[发布说明](https://github.com/huggingface/transformers/releases)。也请继续分享您的反馈，并在 Transformers 中发布你的模型，供社区使用🤗

## 延伸阅读

如果您想进一步了解特定主题，可以访问以下链接列表：
1. [Hugging Face GPT-OSS配方仓库](https://github.com/huggingface/gpt-oss-recipes)
2. [欢迎GPT OSS：OpenAI 全新开源模型系列](https://huggingface.co/blog/welcome-openai-gpt-oss)
3. [OpenAI Cookbook：GPT-OSS主题](https://cookbook.openai.com/topic/gpt-oss)
4. [Transformers文档：基于多 GPU 的分布式推理](https://huggingface.co/docs/transformers/en/perf_infer_gpu_multi)
5. [Matthew Carrigan 的X 推文: 关于GPT OSS创新](https://x.com/carrigmat/status/1952779877569978797)
6. [YouTube视频：OpenAI GPT OSS公告](https://www.youtube.com/watch?v=bbkcEiUjehk)
7. [Transformers PR #36380：在加速器上更快的模型加载](https://github.com/huggingface/transformers/pull/36380)
8. [Transformers PR #36335：为张量并行更新from_pretrained](https://github.com/huggingface/transformers/pull/36335)
9. [Transformers PR #40039：新的动态滑动窗口层和缓存](https://github.com/huggingface/transformers/pull/40039)
10. [HAN Lab博客：注意力机制如何保持语言模型稳定](https://hanlab.mit.edu/blog/streamingllm)
