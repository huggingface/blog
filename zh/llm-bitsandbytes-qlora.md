
---
title: 大模型训练，有卡就行！bitsandbytes、4 位量化和 QLoRA 技术原理与应用

thumbnail: /blog/assets/148_llm-bitsandbytes-qlora
authors:
- user: Younes Belkada
- user: Tim Dettmers
  guest: true
- user: Artidoro Pagnoni
  guest: true
- user: Sylvain Gugger
- user: Sourab Mangrulkar
translators:
- user: Cony
- user: zhongdongy
  proofreader: true
---


# 大模型训练，有卡就行！bitsandbytes、4 位量化和 QLoRA 技术原理与应用


> 原为：Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA
> Created: May 30, 2023 7:26 PM
URL: https://huggingface.co/blog/4bit-transformers-bitsandbytes


在消费硬件上运行或训练LLMs 对用户是一个巨大的挑战。我们的 [LLM.int8 博客文章](https://huggingface.co/blog/hf-bitsandbytes-integration) 展示了如何使用 `bitsandbytes` 库将 [LLM.int8 论文](https://arxiv.org/abs/2208.07339) 中的技术集成到 transformers 中。为了使模型更加易于运行，我们再次与 bitsandbytes 合作，让用户以 4 位精度运行模型。这包括大多数 HF 模型，无论是什么模态（文本、视觉、多模态等）。用户还可以使用 Hugging Face 生态系统中的工具在 4 位模型之上训练适配器。这是 Dettmers 等人在 QLoRA 论文中今天介绍的一种新方法。该论文的摘要如下：

> 
> 
> 
> 作者提出了QLoRA，这是一种高效的微调方法，可以将内存使用降低到在单个48GB GPU上微调65B参数模型，同时保持完整的16位微调任务性能。*QLoRA 通过冻结的 4 位量化预训练语言模型将梯度反向传播到低阶适配器(LoRA)*。最好的模型系列名为Guanaco（原驼），优于所有以前公开发布的模型，并在Vicuna基准测试中达到了ChatGPT性能水平的99.3%，而只需要在单个GPU上进行24小时的微调。QLoRA引入了多项创新，在不损失性能下节省内存：（a）4位NormalFloat（NF4），一种新的数据类型，理论上是正态分布权重的最佳信息，（b）双重量化，通过量化量化常数来减少平均内存占用，（c）分页优化器来管理内存波动。作者使用 QLORA 对 1,000 多个模型进行微调，提供跨 8 个指令数据集、多种模型类型（LLaMA、T5）和无法通过常规微调运行的模型规模（例如 33B 和65B参数模型）并给出了详细的指令遵循和聊天表现分析。作者的结果表明，即使使用比以前的 SoTA 更小的模型，QLoRA 对小型高质量数据集的微调也会产生最先进的结果。作者提供了一个基于人类和GPT-4评估的聊天机器人性能的详细分析，表明GPT-4评估是一种廉价且合理的替代人类评估的方案。此外，我们发现当前的聊天机器人基准测试不能准确评估聊天机器人的性能水平。柠檬挑选分析展示了Guanaco与ChatGPT相比的失败之处。作者发布了所有的模型和代码，包括4位训练的CUDA内核。
> 

## **资源**

这篇博文和版本附带了一些资源，可帮助您开始使用 4 位模型和 QLoRA：

- [原始论文](https://arxiv.org/abs/2305.14314)
- [使用 Google Colab notebook](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing)
    
    - 该colab notebook展示了如何使用 4 位模型对其所有变体进行推理，以及如何在免费的 Google Colab 实例上运行 GPT-neo-X（20B 参数模型）🤯
    
- [微调 Google Colab notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing)
    - 该**notebook**展示了如何使用 Hugging Face 生态系统在下游任务上微调 4 位模型。
        
        我们证明可以在 Google Colab 实例上微调 GPT-neo-X 20B！
        
- [论文复现仓库](https://github.com/artidoro/qlora)
- [Guanaco（原驼） 33b playground](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi)

## **介绍**
如果不熟悉模型精度和最常见的数据类型（float16、float32、bfloat16、int8），建议阅读我们第一篇[博文](https://huggingface.co/blog/hf-bitsandbytes-integration)中的介绍。**更多信息阅读 [wikibook](https://en.wikibooks.org/wiki/A-level_Computing/AQA/Paper_2/Fundamentals_of_data_representation/Floating_point_numbers#:~:text=In%20decimal%2C%20very%20large%20numbers,be%20used%20for%20binary%20numbers.) 文档**中的浮点表示基础知识。

最近的 QLoRA 论文探讨了不同的数据类型，4 位 Float 和 4 位 NormalFloat。我们将在这里讨论更容易理解 4 位 Float 数据类型。

FP8 和 FP4 分别代表浮点 8 位和 4 位精度。它们是浮点值 minifloats 系列的一部分（除其他精度外，minifloats 系列还包括 bfloat16 和 float16）。

让我们先看看如何用 FP8 格式表示浮点值，然后了解 FP4 格式的样子。

## **FP8 格式**

正如我们在之前的博文中所讨论的，一个浮点数包含 n 位，每一位都属于一个特定的类别，负责表示数字的一个组成部分（符号、尾数和指数）。这些代表以下内容。

FP8（floating point 8）格式在论文[“FP8 for Deep Learning”](https://arxiv.org/pdf/2209.05433.pdf)中首次引入，具有两种不同的 FP8 编码：E4M3（4 位指数和 3 位尾数）和 E5M2（5 位指数和 2 位尾数）。

![%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%EF%BC%8C%E6%9C%89%E5%8D%A1%E5%B0%B1%E8%A1%8C%EF%BC%81bitsandbytes%E3%80%814%20%E4%BD%8D%E9%87%8F%E5%8C%96%E5%92%8C%20QLoRA%20%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86%E4%B8%8E%E5%BA%94%E7%94%A8%20fd6944634dcd49f6af51ccaef527b9ff/FP8-scheme.png](%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%EF%BC%8C%E6%9C%89%E5%8D%A1%E5%B0%B1%E8%A1%8C%EF%BC%81bitsandbytes%E3%80%814%20%E4%BD%8D%E9%87%8F%E5%8C%96%E5%92%8C%20QLoRA%20%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86%E4%B8%8E%E5%BA%94%E7%94%A8%20fd6944634dcd49f6af51ccaef527b9ff/FP8-scheme.png)


8位**浮点  (FP8) 格式，来源：sgugger**

---

从 32 位减少到 8 位大大降低了精度，可以根据不同的需要来使用不同版本。目前可以使用[Transformer Engine 库](https://github.com/NVIDIA/TransformerEngine)，该库已经与 HF 生态集成。

E4M3 格式表示的浮点数在 -448 到 448 范围内，而在 E5M2 格式中，随着指数位数的增加，范围增加到 -57344 到 57344 ，但有精度损失，因为可表示的总位数保持不变。**经验证明，E4M3 最适合前向传播，而 E5M2 最适合后向计算。**

## **FP4 精度简述**

符号位表示符号 (+/-)，指数位表示为以2为底，位形式的整数为幂（例如`2^{010} = 2^{2} = 4`），尾数是以2为底、位数的负值为幂的各位之和，位数只对每个为“1”的位“有效”。如果某个位是“0”，则分数保持不变，对于`2^-i` ，i 是该位在位序列中的位置。例如，对于尾数位 1010，我们有`(2^-1 + 0 + 2^-3 ＋ 0) = (0.5 + 0.125) = 0.625`. 为了得到一个值，我们将分数加*1*并将所有结果相乘，例如，使用 2 个指数位和一个尾数位，表示 1101 将是：`-1 * 2^(2) * (1 + 2^-1) = -1 * 4 * 1.5 = -6`

对于 FP4 没有固定的格式，因此可以尝试不同尾数/指数的组合。通常，在大多数情况下，3 个指数位会好一些。但有时 2 个指数位和一个尾数位会产生更好的性能。

## **QLoRA 论文，一种新型量化 Transformer 大模型的亲民方法**

简而言之，与 16 位模型微调相比，QLoRA 在不牺牲性能情况下减少了 LLM 微调的内存使用。该方法在单个 24GB GPU 上实现 33B 模型微调，在单个 46GB GPU 上实现 65B 模型微调。

更具体地说，**QLoRA 使用 4 位量化来压缩预训练语言模型**。然后冻结 LM 参数，并将相对少量的可训练参数以低阶适配器的形式添加到模型中。在微调期间，QLoRA 通过冻结的 4 位量化预训练语言模型将梯度反向传播到低阶适配器。LoRA 层是训练期间唯一更新的参数。[在原始的 LoRA 论文](https://arxiv.org/abs/2106.09685)中阅读有关 LoRA 的更多信息。

QLoRA 包含存储数据类型（通常是 4 位 NormalFloat）和计算数据类型（16 位 BrainFloat）。QLoRA 将存储数据类型的权重反量化为计算数据类型以执行前向和反向传递，但仅计算使用 16 位 bfloat 的 LoRA 参数的权重梯度。权重仅在需要时解压缩，因此在训练和推理期间内存使用率保持较低。

在广泛的实验中，QLoRA 微调与 16 位微调方法效果相当。此外，在[OpenAssistant 数据集 (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1)上对 LLaMA 模型使用 QLoRA 微调得到 Guanaco 模型，基于该模型的聊天表现达到SOTA表现，在 Vicuna 基准测试上接近 ChatGPT，这多少有点超出对QLoRA 微调的预期。

更详细的信息请阅读[QLoRA 论文](https://arxiv.org/abs/2305.14314)。

## **入门**

为了快速入门，通过从源代码安装 accelerate和transformers 来加载 4 位模型，并确保您已安装最新版本的 bitsandbytes 库 (0.39.0)

```
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```
### **快速开始**

以 4bit 方式加载模型需要在调用`from_pretrained`方法时 `load_in_4bit=True` ，通过提供设备映射来传递参数（传递`"auto"`将自动推断的设备映射）。

```
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_4bit=True, device_map="auto")
...
```

作为一般规则，我们建议用户在加载模型后不要手动设置设备device_map。因此，在该行之后尽量不要对模型或模型子模块进行设备分配改动 。加载量化模型时会自动将模型的子模块转换为`float16` dtype。可以传递`torch_dtype=dtype`给`from_pretrained`来更改（例如，如果您想在layer norms  中使用float32）。

### **高级用法**

可以使用 4 位量化的不同变体，例如 NF4（规范化浮点数 4（默认））或纯 FP4 量化。基于论文的理论考虑和实验结果，建议使用 NF4 量化以获得更好的性能。

参数`bnb_4bit_use_double_quant` 用于第二次量化（第一次量化之后）可为每个参数节省额外的 0.4 位。虽然 4 位 bitsandbytes 以 4 位存储权重，但计算仍然以 16 位或 32 位进行，这里可以选择任何组合（float16、bfloat16、float32 等）。

如果使用 16 位计算数据类型（默认 torch.float32），矩阵乘法和训练会更快。可以利用最新的transformers  中`BitsAndBytesConfig`来调整这些参数。下面是一个使用 NF4 （normalized float） 加载 4 位模型的示例，使用计算数据类型 bfloat16 进行双量化以加快训练速度：

```
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

### **更改数据类型**

如上所述，您还可以通过更改`BitsAndBytesConfig`中的`bnb_4bit_compute_dtype`参数来更改量化模型的计算数据类型。

```
import torch
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
```

### **嵌套量化**

要启用嵌套量化，您可以使用`BitsAndBytesConfig`中的`bnb_4bit_use_double_quant`参数。这将在第一次量化之后启用第二次量化，以便为每个参数额外节省 0.4 位。我们也在训练 Google colab notebook 中使用了这个特性

```
from transformers import BitsAndBytesConfig
double_quant_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=True)
model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
```

当然，所有这些组件都是可组合的。您可以将所有参数组合以找到最适合的用例。一条经验法则是：如果内存有问题，请使用双量化，使用 NF4 以获得更高的精度，使用 16 位 dtype 来实现更快的微调。例如，在[推理演示](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing)中，我们使用嵌套量化、bfloat16 计算 dtype 和 NF4 量化在单个 16GB GPU 中以 4 位完全微调 gpt-neo-x-20b (40GB)。

## **常见问题**

在本节中，我们还将解决任何人可能对此集成提出的一些常见问题。

### **FP4量化有硬件要求吗？**

请注意，此方法仅与 GPU 兼容，因此无法在 CPU 上以 4 位量化模型。 在 GPU 中，这种方法应该没有任何硬件要求，因此只要安装了 CUDA>=11.2，任何 GPU 都可以用于运行 4bit 量化。 注意，计算不是在 4 位中完成，只是权重和激活被压缩为该格式，计算仍然保持为期望的 dtype类型。

### **支持的模型有哪些？**

[与本博文](https://huggingface.co/blog/hf-bitsandbytes-integration)中介绍的 LLM.int8 集成类似，支持集成很大程度上依赖于`accelerate`库。因此，任何支持加速加载的模型（即`from_pretrained`调用`的device_map`参数）都应该是4bit可量化的。另请注意，这与模态完全无关，只要模型可以加载参数`device_map`，就可以量化它们。

对于生成式模型，支持包括最常用的文本架构，例如 Llama、OPT、GPT-Neo、GPT-NeoX、用于多模态模型的 Blip2 等。

截止本博客撰写，支持加速的模型有：

```
['bigbird_pegasus', 'blip_2', 'bloom', 'bridgetower', 'codegen', 'deit', 'esm',
    'gpt2', 'gpt_bigcode', 'gpt_neo', 'gpt_neox', 'gpt_neox_japanese', 'gptj', 'gptsan_japanese',
    'lilt', 'llama', 'longformer', 'longt5', 'luke', 'm2m_100', 'mbart', 'mega', 'mt5', 'nllb_moe',
    'open_llama', 'opt', 'owlvit', 'plbart', 'roberta', 'roberta_prelayernorm', 'rwkv', 'switch_transformers',
    't5', 'vilt', 'vit', 'vit_hybrid', 'whisper', 'xglm', 'xlm_roberta']
```

请注意，如果您想要的模型不在这里，您可以开一个Pull Request或在transformers 中提出一个issue，可以加快该架构对模型的支持。

### **可以训练 4 位/8 位模型吗？**

不可能在这些模型上执行纯 4 位训练。但是，您可以通过利用参数高效微调方法 (PEFT) 来训练这些模型，并在它们之上训练例如适配器。这就是论文中所做的，并得到 Hugging Face 的 PEFT 库的正式支持。我们还提供了一个[colab notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing)**，如果用户有兴趣复制论文中的结果，建议他们查看**[QLoRA 存储库。](https://github.com/artidoro/qlora)

![%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%EF%BC%8C%E6%9C%89%E5%8D%A1%E5%B0%B1%E8%A1%8C%EF%BC%81bitsandbytes%E3%80%814%20%E4%BD%8D%E9%87%8F%E5%8C%96%E5%92%8C%20QLoRA%20%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86%E4%B8%8E%E5%BA%94%E7%94%A8%20fd6944634dcd49f6af51ccaef527b9ff/lora-animated.gif](%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%EF%BC%8C%E6%9C%89%E5%8D%A1%E5%B0%B1%E8%A1%8C%EF%BC%81bitsandbytes%E3%80%814%20%E4%BD%8D%E9%87%8F%E5%8C%96%E5%92%8C%20QLoRA%20%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86%E4%B8%8E%E5%BA%94%E7%94%A8%20fd6944634dcd49f6af51ccaef527b9ff/lora-animated.gif)


**原始（冻结的）预训练权重（左）、权重矩阵 A 和 B 组成的低秩适配器（右）、共同增强原始激活h。**

---

### **其他影响**

这种集成可以为社区和 AI 研究带来一些积极的影响，因为它可以影响多个模型和可能的应用程序。在 RLHF（人类反馈强化学习）中，可以加载一个 4 位基础模型并在其上训练多个适配器，一个用于奖励建模，另一个用于价值策略训练。关于此用例的更详细的博文和公告将很快发布。

我们还针对这种量化方法对在消费类硬件上训练大型模型的影响做了一些基准测试。我们在 NVIDIA T4 (16GB) 上运行了几个微调 2 种不同架构的实验，Llama 7B（fp16下 15GB）和 Llama 13B（fp16 下 27GB），这是结果

| Model name | Half precision model size (in GB) | Hardware type / total VRAM | quantization method (CD=compute dtype / GC=gradient checkpointing / NQ=nested quantization) | batch_size | gradient accumulation steps | optimizer | seq_len | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |
| <10B scale models |  |  |  |  |  |  |  |  |
| decapoda-research/llama-7b-hf | 14GB | 1xNVIDIA-T4 / 16GB | LLM.int8 (8-bit) + GC | 1 | 4 | AdamW | 512 | No OOM |
| decapoda-research/llama-7b-hf | 14GB | 1xNVIDIA-T4 / 16GB | LLM.int8 (8-bit) + GC | 1 | 4 | AdamW | 1024 | OOM |
| decapoda-research/llama-7b-hf | 14GB | 1xNVIDIA-T4 / 16GB | 4bit + NF4 + bf16 CD + no GC | 1 | 4 | AdamW | 512 | No OOM |
| decapoda-research/llama-7b-hf | 14GB | 1xNVIDIA-T4 / 16GB | 4bit + FP4 + bf16 CD + no GC | 1 | 4 | AdamW | 512 | No OOM |
| decapoda-research/llama-7b-hf | 14GB | 1xNVIDIA-T4 / 16GB | 4bit + NF4 + bf16 CD + no GC | 1 | 4 | AdamW | 1024 | OOM |
| decapoda-research/llama-7b-hf | 14GB | 1xNVIDIA-T4 / 16GB | 4bit + FP4 + bf16 CD + no GC | 1 | 4 | AdamW | 1024 | OOM |
| decapoda-research/llama-7b-hf | 14GB | 1xNVIDIA-T4 / 16GB | 4bit + NF4 + bf16 CD + GC | 1 | 4 | AdamW | 1024 | No OOM |
| 10B+ scale models |  |  |  |  |  |  |  |  |
| decapoda-research/llama-13b-hf | 27GB | 2xNVIDIA-T4 / 32GB | LLM.int8 (8-bit) + GC | 1 | 4 | AdamW | 512 | No OOM |
| decapoda-research/llama-13b-hf | 27GB | 1xNVIDIA-T4 / 16GB | LLM.int8 (8-bit) + GC | 1 | 4 | AdamW | 512 | OOM |
| decapoda-research/llama-13b-hf | 27GB | 1xNVIDIA-T4 / 16GB | 4bit + FP4 + bf16 CD + no GC | 1 | 4 | AdamW | 512 | OOM |
| decapoda-research/llama-13b-hf | 27GB | 1xNVIDIA-T4 / 16GB | 4bit + FP4 + fp16 CD + no GC | 1 | 4 | AdamW | 512 | OOM |
| decapoda-research/llama-13b-hf | 27GB | 1xNVIDIA-T4 / 16GB | 4bit + NF4 + fp16 CD + GC | 1 | 4 | AdamW | 512 | No OOM |
| decapoda-research/llama-13b-hf | 27GB | 1xNVIDIA-T4 / 16GB | 4bit + NF4 + fp16 CD + GC | 1 | 4 | AdamW | 1024 | OOM |
| decapoda-research/llama-13b-hf | 27GB | 1xNVIDIA-T4 / 16GB | 4bit + NF4 + fp16 CD + GC + NQ | 1 | 4 | AdamW | 1024 | No OOM |

我们使用了最近的TRL 库，基准测试脚本可以[在这里](https://gist.github.com/younesbelkada/f48af54c74ba6a39a7ae4fd777e72fe8)`SFTTrainer`找到

## **Playground**

[Playground](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi)上体验 Guananco 模型

## **致谢**

HF 团队要感谢华盛顿大学所有参与该项目的人员，感谢他们将此项目提供给社区。

还要感谢[Pedro Cuenca](https://huggingface.co/pcuenq)**对博文的友好审阅，感谢**[Olivier Dehaene](https://huggingface.co/olivierdehaene)**和**[Omar Sanseviero](https://huggingface.co/osanseviero)对 HF Hub 上论文工件集成的快速和大力支持。
