---
title: "TGI 多-LoRA：部署一次，搞定 30 个模型的推理服务" 
thumbnail: /blog/assets/multi-lora-serving/thumbnail.png
authors:
- user: derek-thomas
- user: dmaniloff
- user: drbh
translators:
- user: MatrixYao
---

# TGI 多-LoRA：部署一次，搞定 30 个模型的推理服务

你是否已厌倦管理多个 AI 模型所带来的复杂性和高成本？ 那么，**如果你可以部署一次就搞定 30 个模型推理服务会如何？** 在当今的 ML 世界中，哪些希望充分发挥其数据的价值的组织可能最终会进入一个“微调的世界”。在这个世界，各个组织会构建大量模型，其中每个模型都针对特定任务进行了高度特化。但是，如何处理为每个细分应用部署模型所带来的麻烦和成本呢？多-LoRA 服务提供了一个有潜力的答案。

## 动机

对组织而言，基于微调构建多个模型是有意义的，原因有多重： 

- **性能 -** 有[足够证据](https://huggingface.co/papers/2405.09673)表明：在目标任务上，较小的专用模型表现优于较大的通用模型。Predibase 的结果 [[5]](#5) 表明，针对特定任务对 [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main) 基础模型进行LoRA 微调可以获得比 GPT-4 更好的性能。

- **适应性 -** Mistral 或 Llama 等模型的用途极其广泛，你可以选择其中之一作为基础模型，然后针对[各种下游任务](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4)微调出各种专用模型。还有一个好处是，你不会被某个模型锁定，因为你可以轻松换掉该基础模型，然后用你的数据对另一个基础模型进行微调（稍后会详细介绍）。

- **独立性 -** 对不同任务，不同的团队可以独立进行不同的微调，从而在数据准备、配置、评估标准和模型更新节奏方面保持独立和并行。

- **隐私 -** 专用模型提供了很大的灵活性，使得我们可以根据隐私要求对训练数据进行隔离，不需要将所有数据都暴露成基础模型的训练数据。此外，由于模型的本地运行愈显重要，微调使得在本地设备上运行的小模型有能力执行特定任务。 

总之，微调使组织能够释放其数据的价值，当它们使用其独有的、高度专业化的数据时，这种优势变得尤为重要，甚至足以改变游戏规则。

看上去前景光明，有啥问题吗？有的！部署大语言模型 (LLM) 服务提出了多方面的挑战。部署单个模型的成本和操作复杂性已经够让人头疼了，更不用说 *n* 个模型了。这意味着，虽然微调有万般好，但是它让 LLM 的部署和服务变得更复杂了也是铁的事实。

如何解决“既要又要”的问题，及时雨就应时而现了。TGI 最近推出了新功能 - **多-LoRA 服务**（此处应有掌声）。

## LoRA 背景知识

LoRA 即[低阶适配](https://huggingface.co/papers/2106.09685)，是一种对预训练大模型进行高效微调的技术。其核心思想是无需重新训练整个模型，仅需训练一小部分称为适配器的参数，就可使预训练大模型适应特定任务。这些适配器的大小与预训练 LLM 相比，通常仅增加约 1% 的存储和内存开销，就能达到与全模型微调的模型相当的效果。 

LoRA 的明显好处是，它通过减少内存需求来降低微调成本。它还可以[缓解灾难性遗忘](https://huggingface.co/papers/2405.09673)，且在[小数据集](https://huggingface.co/blog/peft)上效果更好。

<video style="width: auto; height: auto;" controls autoplay muted loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/LoRA.webm">
  当前浏览器不支持视频标签。
</video>

|                            |
|----------------------------|
| *图 1：LoRA 详解* |

在训练过程中，LoRA 会冻结原模型权重 `W`，并对两个小矩阵 `A` 和 `B` 进行微调，这使得微调更加高效。知道这一点后，你就能比较容易理解图 1 中 LoRA 模型推理的工作原理了。我们从预训练模型 `Wx` 中获取输出，并将其与低阶适配项 `BAx` 相加[[6]](#6)。

## 多-LoRA 推理服务

了解了 LoRA 的低阶适配的基本思想后，我们可以深入研究一下多-LoRA 服务了。这个概念很简单：给定一个基础预训练模型和一些任务，你可以针对这些任务微调特定的 LoRA，多-LoRA 服务是一种根据传入请求动态选择所需 LoRA 的机制。

<video style="width: auto; height: auto;" controls autoplay muted loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/MultiLoRA.webm">
  当前浏览器不支持视频标签。
</video>

|                                  |
|----------------------------------|
| *图 2：多-LORA 详解* |

_图 2_ 展示了这种动态路由的工作原理。每个用户请求都包含输入 `x` 以及该请求对应 LoRA 的 id（我们称为同批异构用户请求）。LoRA id 信息使得 TGI 得以凭此选择正确的 LoRA 适配器。 

多-LoRA 服务让我们仅需部署一个基础模型。而且由于 LoRA 适配器很小，所以你可以加载多个适配器，而不用担心内存问题。请注意，具体能加载多少个适配器取决于你的可用 GPU 资源以及你部署的模型。最终效果实际上相当于在一次部署中支持了多个经过微调的模型。

LoRA 权重的大小依秩和量化方法的不同而不同，但它们通常都非常小。这边给大家一个直观印象：[predibase/magicoder](https://huggingface.co/predibase/magicoder/tree/main) 为 13.6MB，不到 [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main) 尺寸（14.48GB）的 1/1000。相对而言，将 30 个适配器加载到 RAM 中只会让 VRAM 增加 3%，这对于大多数部署来说都不成问题。因此，我们可以一次部署多个模型。

# 如何使用

## 收集 LoRA 权重

首先，你需要训练 LoRA 模型并导出适配器权重。你可以在此处找到 LoRA 微调相关的[指南](https://huggingface.co/docs/peft/en/task_guides/lora_based_methods)。请注意，当你将微调后的模型推送到 Hub 时，只需推送适配器，无需推送完整的合并模型。从 Hub 加载 LoRA 适配器时，会从适配器模型卡推断出基础模型并将其单独加载。如需更深入的支持，可以试试我们的[专家支持计划](https://huggingface.co/support)。当你为特定用例创建自己的 LoRA 时，真正的价值才会显现。

### 低代码团队

对某些组织而言，为自己的用例训练一个 LoRA 可能比较困难，因为它们可能缺乏相应的专业知识或其他资源。即使选好了基础模型并准备好了数据，后面还需要跟上最新技术，探索超参空间，找到最佳硬件资源，编写代码，然后进行评估。这项任务，即使对于经验丰富的团队来说，也不可谓不艰巨。 

AutoTrain 可帮助显著降低这一门槛。AutoTrain 是一种无代码解决方案，只需单击几下鼠标即可训练机器学习模型。我们提供了多种使用 AutoTrain 的方法。除了[本地安装](https://github.com/huggingface/autotrain-advanced?tab=readme-ov-file#local-installation)外，我们还支持：

| AutoTrain 环境 | 硬件配置  | 编码量 | 备注 |
| ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | ---------------- | ----------------------------------------- |
| [Hugging Face Space](https://huggingface.co/login?next=%2Fspaces%2Fautotrain-projects%2Fautotrain-advanced%3Fduplicate%3Dtrue) | 多种 GPU 及其它硬件 | 无代码          | 灵活易用                |
| [DGX 云](https://huggingface.co/blog/train-dgx-cloud)                                                                       | 最高 8xH100 GPU            | 无代码          | 更适宜大模型                   |
| [Google Colab](https://colab.research.google.com/github/huggingface/autotrain-advanced/blob/main/colabs/AutoTrain.ipynb)       | 单张 T4 GPU           | 低代码         | 适宜小模型以及量化后的模型 |

## 部署

本文以 [Predibase 的 LoRA Land](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4) 为例，主要使用如下两个 LoRA 适配器：

- [predibase/customer_support](https://huggingface.co/predibase/customer_support)，其是在 [Gridspace-Stanford Harper Valley 语音数据集](https://github.com/cricketclub/gridspace-stanford-harper-valley) 上微调而得，增强了准确理解和响应交互性客服工单的能力，改善了模型在语音识别、情绪检测和对话管理等任务中的表现，有助于促成更高效、更富同理心的客户支持。

- [predibase/magicoder](https://huggingface.co/predibase/magicoder)，其是在 [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) 上微调而得，这是一个合成的代码指令数据集。

### TGI

[TGI 文档](https://github.com/huggingface/text-generation-inference)中已有很多关于如何部署 TGI 的有用信息。这里，我们仅提醒一些要点：

1. 使用 `v2.1.1` 或更新版本的 TGI
2. 部署基础模型：`mistralai/Mistral-7B-v0.1`
3. 在部署期间，添加 `LORA_ADAPTERS` 环境变量
    * 示例：`LORA_ADAPTERS=predibase/customer_support,predibase/magicoder` 

```bash
model=mistralai/Mistral-7B-v0.1
# share a volume with the Docker container to avoid downloading weights every run
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:2.1.1 \
    --model-id $model \
    --lora-adapters=predibase/customer_support,predibase/magicoder
```

### 推理终端 GUI

[推理终端](https://huggingface.co/docs/inference-endpoints/en/index)支持多种 [GPU 或其他 AI 加速卡](https://huggingface.co/docs/inference-endpoints/en/pricing#gpu-instances)，只需点击几下即可跨 AWS、GCP 以及 Azure 部署！使用 GUI 部署相当容易。其后端默认使用 TGI 进行文本生成（你也可以[选择](https://huggingface.co/docs/inference-endpoints/en/guides/custom_container)使用自己的 docker 镜像）。

要在推理终端上使用多-LoRA 服务，你只需跳转至[控制台](https://ui.endpoints.huggingface.co/)，然后：

1. 选择基础模型：`mistralai/Mistral-7B-v0.1`
2. 选择 `云` | `地区` | `硬件`
    * 例如：`AWS` | `us-east-1` | `Nvidia L4`
3. 选择高级配置
    * 你应该看到已经选择了 `文本生成`
    * 可根据自己的需求进行配置
4. 在环境变量中添加 `LORA_ADAPTERS=predibase/customer_support,predibase/magicoder`
5. 最后 `创建端点`！

请注意，以上只是最少配置，你可以根据需要对其他设置进行配置。

| ![multi-lora-inference-endpoints](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/multi-lora-inference-endpoints.png) |
|-------------------------------------------------|
| *图 3：多-LoRA 推理终端* |


| ![multi-lora-inference-endpoints](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/multi-lora-inference-endpoints-2.png) |
|-------------------------------------------------|
| *图 4：多-LoRA 推理终端 2* |

### 推理终端代码

有些人可能有点[怕老鼠](https://en.wikipedia.org/wiki/Fear_of_mice_and_rats)，因此不想使用鼠标，我们对此不做评判[尬笑~~]。此时，仅用键盘也可通过代码自动执行上述操作，非常简单。

```python
from huggingface_hub import create_inference_endpoint

# Custom Docker image details
custom_image = {
    "health_route": "/health",
    "url": "ghcr.io/huggingface/text-generation-inference:2.1.1",  # This is the min version
    "env": {
        "LORA_ADAPTERS": "predibase/customer_support,predibase/magicoder",  # Add adapters here
        "MAX_BATCH_PREFILL_TOKENS": "2048",  # Set according to your needs
        "MAX_INPUT_LENGTH": "1024", # Set according to your needs
        "MAX_TOTAL_TOKENS": "1512", # Set according to your needs
        "MODEL_ID": "/repository"
    }
}

# Creating the inference endpoint
endpoint = create_inference_endpoint(
    name="mistral-7b-multi-lora",
    repository="mistralai/Mistral-7B-v0.1",
    framework="pytorch",
    accelerator="gpu",
    instance_size="x1",
    instance_type="nvidia-l4",
    region="us-east-1",
    vendor="aws",
    min_replica=1,
    max_replica=1,
    task="text-generation",
    custom_image=custom_image,
)
endpoint.wait()

print("Your model is ready to use!")
```

部署此配置大约需要 3 分 40 秒。请注意，其他模型可能需要更长的时间。如果你遇到加载时长的问题，请在 github 上提交[问题](https://github.com/huggingface/text-generation-inference/issues)！

## 使用

当使用推理终端时，你需要指定 `adapter_id`。下面给出了一个 cURL 示例：

```bash
curl 127.0.0.1:3000/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
  "inputs": "Hello who are you?",
  "parameters": {
    "max_new_tokens": 40,
    "adapter_id": "predibase/customer_support"
  }
}'
```

这里还有一个使用 [InferenceClient](https://huggingface.co/docs/huggingface_hub/guides/inference) 的示例，该示例来自 [Hugging Face Hub Python 库](https://huggingface.co/docs/huggingface_hub/index)。请确保你用的是 `huggingface-hub>=0.24.0` ，在必要情况下，你还需[登录](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) hub。

```python
from huggingface_hub import InferenceClient

tgi_deployment = "127.0.0.1:3000"
client = InferenceClient(tgi_deployment)
response = client.text_generation(
    prompt="Hello who are you?",
    max_new_tokens=40,
    adapter_id='predibase/customer_support',
)
```

## 实际考量

### 成本

正如[下文](#致谢)所讨论的，我们并不是第一个吃螃蟹的。请务必阅读一下 LoRAX 背后的团队 Predibase 发表的这篇出色[博文](https://predibase.com/blog/lorax-the-open-source-framework-for-serving-100s-of-fine-tuned-llms-in)，因为本节内容主要基于他们的工作。 

| ![multi-lora-cost](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/multi-lora-cost.png) |
|-------------------------------------------------|
| *图 5：多-LoRA 成本* 我们用 TGI 在英伟达 L4 上部署了 [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) 基础模型，其[推理终端](https://huggingface.co/docs/inference-endpoints/en/index)[成本](https://huggingface.co/docs/inference-endpoints/en/pricing#gpu-instances)为 0.8 美元/小时。每秒可完成 75 个请求，平均每个请求有 450 个输入词元、234 个输出词元，并与相应配置的 GPT3.5 Turbo 成本进行了对比。|

多-LoRA 服务的一大好处是，**无需为多个模型进行多次部署**，因此要便宜得多。这与直觉相符，因为多模型部署要加载所有权重，而不仅仅是小小的适配器。如图 5 所示，当使用 TGI 多-LoRA 时，即使添加更多模型，每个词元的成本也是相同的。但如果不使用多-LoRA，每多部署一个微调模型，TGI 的成本就会随之线性增加。

## 使用模式

| ![multi-lora-serving-pattern](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/multi-lora-serving-pattern.png) |
|-------------------------------------------------|
| *图 6：多-LoRA 服务模式* |

当部署多个模型时，一个现实的挑战是每个模型的使用模式有很大差异：某些模型的使用率可能较低；有些模型的使用模式可能是阵发的，有些可能是高频的。这使得扩展变得非常困难，尤其是当每个模型相互独立部署的时候。当你必须加一个 GPU 时，会出现很多“舍入”误差，而且这种误差会快速累积，最终导致巨大的浪费。在理想情况下，你需要最大限度地提高每个 GPU 的利用率，尽量不使用任何额外资源。你需要确保有足够的 GPU，同时深知有些 GPU 会闲置，太难了！ 

当使用多-LoRA 方案时，情况就平稳多了。如图 6，我们可以看到多-LoRA 服务模式非常平稳，尽管其中某些 LoRA 自身的使用模式并不稳定。通过整合多个 LoRA，整体使用模式会更平稳，且扩展会更容易。请注意，以上仅提供了一个例子，你自己的工作负载的使用模式如何以及多-LoRA 如何能帮上忙，需要你自己认真分析。我们的目标是，仅需考虑 1 个模型的扩展，而无需考虑 30 个模型的扩展！

## 换一个基础模型

AI 发展日新月异，现实世界应当如何应对？如果你想选择另一个或更新的模型作为基础模型，应该怎么办？虽然我们的例子使用了 [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) 作为基础模型，但其实还可以选择别的，如 [Mistral v0.3](https://ubiops.com/function-calling-deploy-the-mistral-7b-v03/) 支持[函数调用](https://ubiops.com/function-calling-deploy-the-mistral-7b-v03/)；更别提还有其他系列的模型了，如 Llama 3。总的来说，我们乐见更高效、性能更好的新基础模型不断出现。

但不用担心！只要你有*足够的理由*更换基础模型，重新训练 LoRA 相对比较容易，训练也相对比较便宜，事实上，[Predibase 发现](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4) 训练一个 LoRA 仅需约 8.00 美元。使用现代框架和常用工程实践，需要的代码改动也很少。基本做法如下：

* 保留模型训练的 notebook / 代码
* 对数据集进行版本控制
* 记录下所使用的每个配置
* 用新模型、新配置更新服务

## 总结

多-LoRA 服务是 AI 模型部署的革命性方案，为解决和管理多个专用模型部署的成本和复杂性问题提供了解决方案。通过利用单一基础模型并动态应用微调适配器，可以显著降低组织的运营开销，同时保持甚至增强各任务的性能。 **我们呼吁 AI 总监们大胆采纳该“基础模型 + 多-LoRA” 应用范式**，从而拥抱由其带来的简单性和成本节约红利。让多-LoRA 成为你 AI 战略的基石，确保你的组织在快速发展的技术领域始终保持领先地位。

## 致谢

实现多-LoRA 服务可能非常棘手，但是由于 [punica-ai](https://github.com/punica-ai/punica) 和 [lorax](https://github.com/predibase/lorax)团队开发了优化的算子和框架，该过程已经很高效了。TGI 利用这些优化来为多个 LoRA 模型提供快速高效的推理。

特别感谢 Punica、LoRAX 和 S-LoRA 团队在多-LoRA 服务方面所做的出色及开放的工作。

## 参考文献

* <a id="1">[1]</a> : Dan Biderman, Jose Gonzalez Ortiz, Jacob Portes, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham, [LoRA Learns Less and Forgets Less](https://huggingface.co/papers/2405.09673), 2024
* <a id="2">[2]</a>  : Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2106.09685), 2021
* <a id="3">[3]</a>  : Sourab Mangrulkar, Sayak Paul, [PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft), 2023
* <a id="4">[4]</a>  : Travis Addair, Geoffrey Angus, Magdy Saleh, Wael Abid, [LoRAX: The Open Source Framework for Serving 100s of Fine-Tuned LLMs in Production](https://predibase.com/blog/lorax-the-open-source-framework-for-serving-100s-of-fine-tuned-llms-in), 2023
* <a id="5">[5]</a>  : Timothy Wang, Justin Zhao, Will Van Eaton, [LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4), 2024
* <a id="6">[6]</a>  : Punica: Serving multiple LoRA finetuned LLM as one: [https://github.com/punica-ai/punica](https://github.com/punica-ai/punica)

> 英文原文: <url> https://huggingface.co/blog/multi-lora-serving </url>
> 原文作者：Derek Thomas，Diego Maniloff，David Holtz
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。