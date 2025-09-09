---
title: "SmolLM3: smol, multilingual, long-context reasoner" 
thumbnail: /blog/assets/smollm3/image.png
authors:
- user: eliebak
- user: cmpatino
- user: anton-l
- user: edbeeching
- user: m-ric
- user: nouamanetazi
- user: akseljoonas
- user: guipenedo
- user: hynky
- user: clefourrier
- user: SaylorTwift
- user: kashif
- user: qgallouedec
- user: hlarcher
- user: glutamatt
- user: Xenova
- user: reach-vb
- user: ngxson
- user: craffel
- user: lewtun
- user: loubnabnl
- user: lvwerra
- user: thomwolf
translator:
- user: AdinaY
---

# SmolLM3：支持多语言与长上下文推理的小模型

随着用户对高效部署且能力强大的语言模型需求日益增长，小型语言模型正变得愈发重要。开源社区已经涌现出一系列令人惊艳的强大小模型，不断突破这一参数规模下的性能极限。

我们推出的 **SmolLM3**，正是在这一背景下贡献的一款具有竞争力的 **完全开源的 30 亿参数模型**：

- 📦 基础模型：[https://hf.co/HuggingFaceTB/SmolLM3-3B-Base](https://hf.co/HuggingFaceTB/SmolLM3-3B-Base)  
- 🧠 指令微调与推理模型：[https://hf.co/HuggingFaceTB/SmolLM3-3B](https://hf.co/HuggingFaceTB/SmolLM3-3B)

**SmolLM3 位于效率与性能的最佳平衡点。**  

我们的 30 亿参数模型在性能上超越了 Llama-3.2-3B 和 Qwen2.5-3B，同时在与更大规模的 40 亿参数模型（如 Qwen3 和 Gemma3）对比中仍具备强劲的竞争力。除了性能数据之外，我们也将**完整公开模型构建过程**，包括所使用的公开数据集与训练框架，助力社区学习、复现与共建。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(17).png" alt=""  style="width: 80%; height: auto;"><br>
</p>

## 模型概要：

- **30 亿参数模型** 训练数据量达 11 万亿 tokens，在 3B 规模中达到 SOTA（最先进水平），并具备媲美 4B 模型的性能；
- **指令微调模型**支持 **双模式推理** 可在 `think` / `no_think` 模式间灵活切换；
- **多语言支持** 覆盖 6 种语言：英语、法语、西班牙语、德语、意大利语和葡萄牙语；
- **长上下文支持** 上下文窗口长达 128k，采用归一化位置编码 （NoPE）与 YaRN 技术实现。

## **完整构建细节**  

与 SmolLM3 一并发布的还有其完整的工程蓝图，内容包括：

- 模型架构细节；
- 精确的数据混合策略，展示如何通过三阶段预训练方法，在各个领域逐步提升性能；
- 构建混合推理模型的方法论。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/smollm3-whiteprint.png" alt=""  style="width: 90%; height: auto;"><br>
</p>

无论你是在构建自己的模型，还是想深入理解在这个参数规模上如何实现出色性能，这份蓝图将帮助你系统的了解一个有竞争力的 3B 模型背后的工程故事。

现在，让我们一起来看看预训练阶段。

# 预训练

SmolLM3 在架构和数据混合策略上都相较于前代模型进行了改进。让我们先来看看它的**模型架构**和**训练配置**吧！

## **架构与训练细节**

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(18).png" alt=""  style="width: 90%; height: auto;"><br>
</p>

SmolLM3 采用了基于 Transformer 解码器的架构，并与 SmolLM2 类似使用共享嵌入（tied embedding）。该模型构建在 Llama 架构之上，并进行了若干关键改进，以优化推理效率和长上下文性能。

### **分组查询注意力机制**

我们将多头注意力机制（multi-head attention）替换为 **Grouped Query Attention（GQA）**，使用了 4 个 query 分组。  
我们在一个使用 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 训练集（1000 亿 tokens）训练的 30 亿参数模型上进行了消融实验，结果表明：

- GQA 的性能与多头注意力相当；
- 推理时显著降低了 KV 缓存的内存占用。

### **归一化位置编码**

我们实现了论文 [《RoPE to NoRoPE and Back Again: A New Hybrid Attention Strategy》](https://huggingface.co/papers/2501.18795)（Yang 等人，2025）中提出的 NoPE 技术，  
具体做法是：**每隔 4 层 selectively 移除 RoPE（旋转位置编码）**。消融实验显示，该方法在不影响短上下文性能的同时，有效提升了长上下文能力。

### **文内注意力屏蔽**

训练时，我们通过注意力屏蔽确保同一训练序列中的不同文档之间不会互相“看到”。这与 Llama 3 的做法类似，有助于：

- 提高长上下文训练的稳定性与速度；
- 保持短上下文性能不受影响。

### **训练稳定性优化**

借鉴 OLMo 2 的做法，我们在嵌入层中移除了 weight decay，以提高训练稳定性。该修改显著改善了训练动态，嵌入权重的范数自然收敛到更健康的范围，且不影响整体性能。

所有上述架构改进，均在同一 30 亿参数架构、使用 1000 亿 tokens 的 FineWeb-Edu 数据集下进行消融验证，确保每一项更改要么带来性能提升，要么在性能保持不变的前提下带来其他工程优势。

### **训练配置**

- 全局 batch size：2.36M tokens  
- 序列长度：4096  
- 学习率：2e-4  
- 优化器：AdamW（β₁=0.9，β₂=0.95）  
- 权重衰减：0.1  
- 梯度裁剪：1  
- 学习率调度器：WSD（Warmup-Stable-Decay）  
  - 预热步数：2000  
  - 最后 10% 训练步骤内线性衰减至 0  
- 使用框架：
  - [nanotron](https://github.com/huggingface/nanotron)：训练  
  - [datatrove](https://github.com/huggingface/datatrove)：数据处理  
  - [lighteval](https://github.com/huggingface/lighteval)：评估  
- 训练资源：384 张 H100 GPU，训练时长 24 天

下图展示了我们的分布式训练配置：

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(19).png" alt=""  style="width: 90%; height: auto;"><br>
</p>

除了架构优化，我们还对整个训练流程进行了深入的消融实验与配方改进。接下来，让我们更深入地了解预训练数据的策略与演化过程。

## **数据混合与训练阶段**

延续 SmolLM2 的多阶段训练策略，SmolLM3 采用三阶段训练方法，在整个训练过程中使用了 **总计 11.2 万亿（T）tokens**，混合了网页文本、数学数据和代码数据，并在不同时期动态调整其比例。

我们在多个使用 500 亿到 1000 亿 tokens 训练的 30 亿参数模型上进行了广泛的消融实验，以确定最终的数据配比策略。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(20).png" alt=""  style="width: 90%; height: auto;"><br>
</p>

预训练包含以下几个阶段，上图中亦有所展示：

### **阶段一：稳定阶段（0T → 8T tokens）**  
此阶段为模型奠定通用能力的基础，核心数据集混合比例如下：

- **网页文本**：85%（其中 12% 为多语言数据）  
  - 数据源包括：FineWeb-Edu、DCLM、FineWeb2、FineWeb2-HQ  
- **代码数据**：12%  
  - 数据源包括：The Stack v2（16 种编程语言）、StarCoder2 Pull Requests、Jupyter 与 Kaggle 笔记本、GitHub Issues、StackExchange  
- **数学数据**：3%  
  - 数据源包括：FineMath3+ 与 InfiWebMath3+

### **阶段二：稳定阶段（8T → 10T tokens）**  
此阶段引入了更高质量的数学与代码数据，同时继续保留良好的网页文本覆盖：

- **网页文本**：75%（其中 12% 为多语言）  
- **代码数据**：15%  
  - 新增数据源：Stack-Edu  
- **数学数据**：10%  
  - 新增数据源：FineMath4+、InfiWebMath4+、MegaMath（包含 Qwen QA、Pro 合成重写、文本-代码交错块）

### **阶段三：衰减阶段（10T → 11.1T tokens）**  
在最后阶段，我们进一步对数学与代码数据进行上采样处理：

- **网页文本**：63%（其中 12% 为多语言）  
- **代码数据**：24%  
  - 强化高质量代码数据的占比  
- **数学数据**：13%  
  - 强化数学数据，同时引入指令与推理数据集，如 OpenMathReasoning

通过上述阶段性的混合策略，我们在基础模型上获得了极具竞争力的性能，相关评估将在后续章节中详细介绍。完整的 nanotron 训练配置及各阶段的精确数据权重可见于此链接：👉 [https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs](https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs)

我们也将公开训练日志与中间模型的检查点，供社区复现与分析。

在主预训练完成后，我们还通过一个 **中间训练阶段**（mid-training stage）进一步提升了模型的**长上下文能力与推理能力**。

# 中期训练

我们将 **长上下文适配** 和 **推理能力适配** 称为 **“中期训练”（mid-training）**。这两个阶段虽然远短于主预训练过程，但仍保持一定的通用性，主要目标是进一步提升模型在这两个关键方向的表现。

首先让我们来看长上下文训练部分。

## **长上下文扩展**

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(21).png" alt=""  style="width: 90%; height: auto;"><br>
</p>

在主预训练完成之后，我们对 SmolLM3 进行了额外的训练，使用 **1000 亿 tokens** 来扩展其上下文长度。我们分两个阶段、每阶段使用 500 亿 tokens，逐步将上下文窗口从 4k 扩展到 64k：

1. **第一阶段**：上下文从 4k 扩展到 32k  
   - 使用 RoPE 的 θ 值提升至 1.5M  
2. **第二阶段**：上下文从 32k 扩展到 64k  
   - 使用 RoPE 的 θ 值提升至 5M  

这两个阶段都对数学、代码与推理相关数据进行了上采样（upsampling）。

我们在消融实验中发现，额外上采样特定的长上下文数据（如代码仓库、电子书、超长网页等）**并不会进一步提升模型在 RULER 与 HELMET 基准测试上的性能**。使用 NoPE 技术、在更长序列下以“衰减混合”策略进行训练、并合理调整 RoPE 的 θ 值，已足以让模型在 64k 长上下文任务上达到很强的表现。

借鉴 Qwen2.5 的做法，我们使用 **YARN 技术** 实现了上下文 extrapolation（外推）。  **推理时，模型最高可处理 128k 上下文**（即训练长度 64k 的两倍）。

## **推理中期训练**

在完成上下文长度扩展之后，我们对模型进行了一个 **中期训练**，以引入通用的 **推理能力**。

中期训练与主预训练及后续微调（如 SFT）阶段的最大区别在于：我们此时训练的目标是模型的**通用推理能力**，而不是针对某个具体领域（如数学或代码）的任务适应。换句话说，我们希望模型具备跨领域的推理能力，而非只擅长某一特定类型的推理。

我们的中期训练数据集总计包含 **350 亿 tokens**，主要来自两个来源：

- Open Thought 发布的 [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- NVIDIA 发布的 [Llama-Nemotron-Post-Training-Dataset-v1.1](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) 中含有 **R1 推理轨迹** 的子集

训练配置方面：

- 使用 **ChatML** 聊天模板来格式化输入  
- 使用 [Wrapped Packing 技术](https://huggingface.co/docs/trl/main/en/reducing_memory_usage#packing)，以避免为模型提供过多结构性提示，从而提升推理泛化能力

我们共训练了 **4 个 epoch**（约合 **1400 亿 tokens**），并将该阶段得到的检查点用于后续的 SFT（指令微调）阶段。

# 后训练

近年来，[DeepSeek R1](https://arxiv.org/abs/2501.12948) 和 [Qwen3](https://arxiv.org/abs/2505.09388) 等推理模型的发布，展示了模型在具备**显式推理能力**时所展现出的强大能力。然而，社区中依然缺乏使用公开数据构建 **双模式指令模型（支持推理与非推理两种模式）** 的完整开源方案。现有方法多数依赖复杂的强化学习流程以及闭源数据集，这给研究人员的复现与创新带来了较大障碍。

在本节中，我们将解释 SmolLM3 如何应对这一挑战，并**公开完整的双模式指令模型构建流程**。我们详细说明了如何通过精心设计的训练流程，在“推理模式”与“非推理模式”之间取得性能平衡。该流程包括：

- **中期训练**：用于注入通用推理能力  
- **监督微调（SFT）**：配合合成数据生成进行有监督训练  
- **对齐训练**：采用 **锚定偏好优化（APO）** 进行偏好对齐，这是一种近期提出的直接偏好优化（DPO）变体方法

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(22).png" alt=""  style="width: 90%; height: auto;"><br>
</p>

## **构建聊天模板**

在介绍训练方法之前，我们首先需要明确用户如何与双模式模型进行交互。聊天模板不仅是用户控制模型行为的接口，同时它的设计也会直接影响训练数据的格式和模型的推理方式。

SmolLM3 的聊天模板允许用户在对话过程中**控制模型是否启用推理模式**。用户可以通过在系统提示（system prompt）中添加特殊标记来切换模式：

- `/think`：启用推理模式  
- `/no_think`：关闭推理模式（即非推理）

在**非推理模式**下，我们会在模型的响应中**预置空的思考区块（think blocks）**，类似 Qwen3 的做法，从而确保模型直接给出答案，而不进行显式推理。

### **工具调用支持**

SmolLM3 支持 **工具调用功能**，其聊天模板中为工具定义设计了两个独立的描述区域：

- **XML 工具区块（XML Tools）**  
- **Python 工具区块（Python Tools）**

这种分类方式在我们的实验中表现良好，有助于模型**准确理解不同格式下的工具定义**。

### **系统消息与元信息**

聊天模板为**推理模式**和**非推理模式**都提供了默认的系统消息（system message），并包含一个元信息区（metadata），其中包括：

- 当前日期（date）
- 知识截断日期（knowledge cut-off date）
- 当前推理模式（reasoning mode）

如果用户希望覆盖默认系统消息，可以通过设置 `system` 角色内容实现。若需完全去除系统消息与元信息区，也可在提示中添加 `/system_override` 标记，从而实现更灵活的使用场景。

## **监督微调**

在完成中间推理训练阶段（共训练了 1400 亿 tokens 的通用推理数据）后，我们继续进行**监督微调（SFT）**，以全面注入模型在以下多个维度的能力：

- 推理模式与非推理模式下的数学、代码、通用推理能力  
- 指令跟随（instruction following）  
- 多语言处理能力（multilinguality）  
- 工具调用（tool calling）

训练一个“双模式模型”（dual-mode model）最大的挑战在于：**如何平衡数据混合比例**，确保在所有目标领域中，模型在推理与非推理两种模式下都能保持强劲性能。为了系统评估 SmolLM3 在训练过程中的表现，我们重点跟踪以下几个维度：

- 数学推理
- 编程代码
- 通用推理
- 指令跟随能力
- 多语言表现

### **数据挑战与合成数据生成**

我们在构建“推理模式”训练数据集时遇到的主要挑战是：**部分任务领域缺乏带推理轨迹（reasoning traces）的公开数据集**。

为填补这一空缺，我们采用以下策略：

- 利用 [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) 模型，处于推理模式下；
- 使用现有的**非推理数据集中的提示语**进行提示生成；
- 由 Qwen3 生成带推理轨迹的合成数据

这种方式显著提升了模型在一些原本推理能力较弱的任务中的表现，例如：

- 多轮对话  
- 多语言交流任务  
- 日常问答与通用指令理解

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(23).png" alt=""  style="width: 80%; height: auto;"><br>
</p>

### **最终数据混合与训练设置**

我们通过大量消融实验，探索推理与非推理 tokens 的最优配比，以及各自内部的数据组成结构。

最终确定的 SFT 数据集总量为 **18 亿 tokens**：

- 推理模式（reasoning）：8 亿 tokens  
- 非推理模式（non-reasoning）：10 亿 tokens  
- 数据集组成：
  - **推理数据集**：10 个
  - **非推理数据集**：12 个

训练配置：

- **训练轮数**：4 个 epoch（共约 80 亿 tokens）  
- **数据打包方式**：[BFD（best-fit decreasing）packing](https://github.com/huggingface/trl/pull/3521)  
- **损失函数处理**：
  - 仅对用户输入（user turns）和工具调用结果部分计算损失（其余部分 masked）

我们将**完整的数据混合策略**与**训练脚本**一并开源，便于社区复现本工作并进一步拓展。

## **使用 锚定偏好优化（APO）进行离策略模型对齐**

在完成监督微调（SFT）后，我们使用 **锚定偏好优化（APO）** 进行了模型对齐。  
我们针对两个模式分别构建了偏好数据：

- **非推理模式**：使用 [Tulu3 偏好数据集](http://allenai/llama-3.1-tulu-3-8b-preference-mixture)
- **推理模式**：使用 Qwen3-32B 和 Qwen3-0.6B 生成的**合成偏好对（preference pairs）**

为了确保覆盖非推理数据集中涉及的全部任务领域，我们为其生成了**补充的推理模式偏好对**。  
在对齐过程中，我们选取：

- Qwen3-32B 的回答作为“**优选**”
- Qwen3-0.6B 的回答作为“**被拒**”

并将这些偏好对用于 APO 训练。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(24).png" alt=""  style="width: 80%; height: auto;"><br>
</p>

### **锚定偏好优化（APO）简介**

[锚定偏好优化](https://arxiv.org/abs/2408.06266)（APO）是 [接偏好优化](https://arxiv.org/abs/2305.18290)（DPO）的一种变体，  
相比 DPO，它具有更稳定的优化目标函数。

在 DPO 中，奖励函数 $r_\\theta(x, y)$ 衡量训练过程中某序列相对于初始参考模型的概率对数比：

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(25).png" alt=""  style="width: 30%; height: auto;"><br>
</p>

其中，参数 $\\beta$ 控制当前模型相对于参考模型可变化的幅度。  
DPO 的损失函数基于三元组进行优化：prompt $x$、优选响应 $y_w$ 和被拒响应 $y_l$：

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(26).png" alt=""  style="width: 50%; height: auto;"><br>
</p>

在我们内部的消融实验中，**APO 表现出更高的稳定性**，并带来了更优的下游任务性能：

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(27).png" alt=""  style="width: 50%; height: auto;"><br>
</p>

### **性能观察与挑战**

下游评估结果显示，模型在以下任务中均取得了性能提升：

- 数学推理  
- 科学类任务  
- 指令跟随  
- 编程任务  
- 对话生成  
- 多语言任务

然而，我们也观察到：**在长上下文基准测试（如 RULER）上性能出现下降**。

我们将问题追溯到推理中间训练阶段：  
当模型过度专注于推理能力学习时，**会对长上下文处理能力产生一定影响**。此外：

- APO 对齐训练的数据长度限制在 **24k tokens**，  
- 原因是我们大多数推理训练样本本身长度都在此之下，导致模型在更长文本推理时能力不稳定。

### **解决方案探索**

为缓解这一性能下降问题，我们进一步探索了**模型合并**作为可行的解决路径。

## **模型合并**

模型合并是一种流行且高效的技术，能够在不引入推理时的集成计算成本、也无需额外训练的前提下，**融合多个模型的优势**。我们使用了 [MergeKit](https://github.com/arcee-ai/mergekit) 库进行模型合并。该工具支持多种合并方式，包括线性合并与非线性合并等。

### **我们的合并流程包括两个步骤：**

1. **将多个 APO 检查点合成为一个“模型汤”（model soup）**  
   - 意指将多个微调后的模型权重混合在一起，以平均或加权方式构建统一模型表示

2. **将这个模型汤与一个中间训练检查点（具有强长上下文性能）进行线性合并**  
   - 合并权重为：
     - APO 模型汤：**0.9**
     - 中间训练检查点：**0.1**
   - 这个比例在我们实验中实现了最优性能表现

通过这一策略，我们成功**恢复了基础模型在 RULER 基准测试中（最高达 128k 上下文长度）原本的性能表现**。最终合并得到的模型正是我们本次发布的 SmolLM3 模型检查点。它在多个任务上表现均衡，兼具推理能力与长上下文处理能力。

# **评估**

接下来，我们将展示该模型与基础模型（base model）在各类评估任务中的表现结果。我们对 **基础模型** 和 **指令模型** 分别在 **推理模式**与**非推理模式**下进行了系统评估。让我们首先来看基础模型的整体表现！

## **基础模型**

下图展示了 SmolLM3 在 12 个主流评估基准上的胜率，涵盖知识、推理、数学和编程能力。结果表明，**SmolLM3 稳定地超越其他 3B 模型，并在多个任务中表现出与 4B 模型（如 Qwen3-4B、Gemma3-4B）相当的竞争力。**

**用于胜率评估的基准测试包括**：

- **常识与知识类**：HellaSwag、ARC、Winogrande、CommonsenseQA、BoolQ  
- **推理与逻辑类**：MMLU-CF、MMLU Pro CF、PIQA、OpenBookQA  
- **数学类**：GSM8K、MATH  
- **编程类**：HumanEval+、MBPP+

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(28).png" alt=""  style="width: 80%; height: auto;"><br>
</p>

在知识与推理类评估任务中（如 HellaSwag、ARC、BoolQ），SmolLM3 **在多个基准上位居第一或第二**，显示出强大的通用认知能力。在数学与编程任务中，SmolLM3 在 3B 模型类别中也展现出强劲的竞争力。

此外，在 **RULER-64k 长上下文评估**中，模型成功处理了长达 64k 的输入序列，说明其具备良好的长文本建模能力。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(33).png" alt=""  style="width: 90%; height: auto;"><br>
</p>

在多语言评估方面，SmolLM3 在五种主要欧洲语言上表现出一致性。我们使用以下多语言基准对模型进行评估：

- **Global MMLU**  
- **MLMM HellaSwag**  
- **Flores-200**  
- **Belebele**

评估内容涵盖：知识、常识推理、文本理解与翻译能力。结果表明，SmolLM3 **在非英语场景下也具备稳健表现**。

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(30).png" alt=""  style="width: 70%; height: auto;"><br>
</p>

### ✅ 总结：

SmolLM3 的基础模型在多个核心任务领域中都展现了非常出色的性能表现，包括通识推理、数学、代码、多语言与长文本处理能力。接下来，让我们看看这些能力在 **指令模型** 中是如何延续与发挥的。

## **双模式指令 / 推理模型**

由于 SmolLM3 同时具备 **指令模式** 和 **推理模式**，我们需要在这两种模式下分别对模型进行评估，并与具备类似能力的其他模型进行对比。

### **非推理模式评估**

我们将 SmolLM3 与其他 3B 非推理模型进行了对比，并将其在“非推理模式”下的表现与 Qwen3 推理模型进行了横向比较，涵盖多个评估基准。

如下图所示：

- **SmolLM3 在多个任务上超越了其他 3B 非推理模型**，包括：
  - Llama3.2 3B Instruct  
  - Qwen2.5 3B Instruct
- 相较于 Qwen3 1.7B，SmolLM3 在性能上有显著优势
- 同时，其性能也**接近 4B 模型的水平**，但计算成本更低，效率更高

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(31).png" alt=""  style="width: 90%; height: auto;"><br>
</p>

因此，SmolLM3 的指令模型处在**性能与计算成本的帕累托最优边界**上。现在，让我们看看在启用推理模式后模型的表现如何。

### **推理模式评估**

在启用“扩展推理模式”后，SmolLM3 的表现相比非推理模式**在多数评估任务上都有明显提升**。

例如，在以下具有挑战性的任务中，我们观察到了显著增益：

- **AIME 2025（数学竞赛）**：
  - 推理模式：**36.7%**
  - 非推理模式：**9.3%**
- **LiveCodeBench（编程竞赛任务）**：
  - 推理模式：**30.0%**
  - 非推理模式：**15.2%**
- **GPQA Diamond（研究生级别推理任务）**：
  - 推理模式：**41.7%**
  - 非推理模式：**35.7%**

虽然 Qwen3 4B 模型在“推理”与“非推理”两个模式下普遍取得了最高分数，但 SmolLM3 在 **3B 参数规模中表现非常有竞争力**，特别在数学推理和复杂问题解决方面表现突出。此外，SmolLM3 的**双模式能力**让用户可以根据实际需求灵活选择：

- 需要快速响应？使用**非推理模式**（/no_think）  
- 需要深入分析？启用**推理模式**（/think）

<p align="center">
 <img src="https://huggingface.co/datasets/HuggingFaceTB/images/resolve/main/smollm3/image%20(32).png" alt=""  style="width: 80%; height: auto;"><br>
</p>

最后一个问题就是：**如何在本地使用 SmolLM3？**

# 如何在本地运行

SmolLM3 的建模代码已经集成到 `transformers v4.53.0` 中，因此请务必确保你已经升级到该版本或更高版本。此外，你也可以使用最新版本的 [`vllm`](https://github.com/vllm-project/vllm)，它以 `transformers` 为后端，支持高性能推理。

安装依赖：

`pip install -U transformers`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B"
device = "cuda" # for GPU usage or "cpu" for CPU usage

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device)

# 生成模型输出
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

# 获取并解码模型输出
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```
> 我们建议在采样参数中设置 `temperature=0.6` and `top_p=0.95` 
> 

### **启用与关闭扩展推理模式**

SmolLM3 默认启用“扩展推理模式”（**Extended Thinking**），因此前面的示例会自动生成包含**推理轨迹**的回答。如需**手动控制是否启用推理模式**，可通过在 `system prompt` 中添加以下标记实现：

- `/think`：启用推理模式（默认）  
- `/no_think`：禁用推理模式，模型将直接生成简洁回答，不展示推理过程

以下是关闭推理模式的代码示例：

```python
prompt = "用通俗易懂的语言简要解释一下什么是重力。"
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

### **代理式使用（Agentic Usage）**

SmolLM3 支持**工具调用**功能！你只需通过指定参数将工具列表传入：

- 使用 `xml_tools`：用于标准工具调用（例如插件、API 等）  
- 使用 `python_tools`：用于调用以 `<code>` 块形式定义的 Python 函数

以下是一个工具调用的完整示例代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM3-3B"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# 定义工具
tools = [
    {
        "name": "get_weather",  # 工具名称
        "description": "获取某个城市的天气情况",  # 工具用途描述
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "要查询天气的城市名称"
                }
            }
        }
    }
]

# 构造对话消息
messages = [
    {
        "role": "user",
        "content": "你好！今天哥本哈根的天气怎么样？"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    enable_thinking=False,  # 如需启用推理，可设为 True
    xml_tools=tools,        # 指定工具描述
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"

outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

# 总结

我们正式发布 **SmolLM3** —— 一个轻量级、支持多语言、具备长上下文推理能力的小模型，最长支持 **128k 上下文长度**。除了模型权重检查点外，我们还**全面开源了完整的训练流程**，涵盖：

- 预训练（Pre-training）
- 中期训练（Mid-training）
- 后训练 / 对齐阶段（Post-training）
- 合成数据生成（Synthetic data generation）

同时，相关数据集也即将同步发布。我们希望这个模型能对社区有所帮助，更希望这份“从零到可用”的训练配方能为其他研究者和开发者提供基础，从而进一步改进和拓展 SmolLM3。

# 资源链接

- ✅ 模型集合（包含量化版本检查点）：  
  [点击查看](https://huggingface.co/collections/HuggingFaceTB/smollm3-686d33c1fdffe8e635317e23)

- 🧪 GitHub 仓库（含预训练配置与评估代码）：  
  [https://github.com/huggingface/smollm](https://github.com/huggingface/smollm)

- 🤗 HuggingFace 团队主页：  
  [https://huggingface.co/HuggingFaceTB](https://huggingface.co/HuggingFaceTB)






