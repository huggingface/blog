---
title: "视觉语言模型详解" 
thumbnail: /blog/assets/vlms_explained/thumbnail.png
authors:
- user: merve
- user: edbeeching
translators:
- user: MatrixYao
---

# 视觉语言模型详解

视觉语言模型可以同时从图像和文本中学习，因此可用于视觉问答、图像描述等多种任务。本文，我们将带大家一览视觉语言模型领域：作个概述、了解其工作原理、搞清楚如何找到真命天“模”、如何对其进行推理以及如何使用最新版的 [trl](https://github.com/huggingface/trl) 轻松对其进行微调。

## 什么是视觉语言模型？

视觉语言模型是可以同时从图像和文本中学习的多模态模型，其属于生成模型，输入为图像和文本，输出为文本。大视觉语言模型具有良好的零样本能力，泛化能力良好，并且可以处理包括文档、网页等在内的多种类型的图像。其拥有广泛的应用，包括基于图像的聊天、根据指令的图像识别、视觉问答、文档理解、图像描述等。一些视觉语言模型还可以捕获图像中的空间信息，当提示要求其检测或分割特定目标时，这些模型可以输出边界框或分割掩模，有些模型还可以定位不同的目标或回答其相对或绝对位置相关的问题。现有的大视觉语言模型在训练数据、图像编码方式等方面采用的方法很多样，因而其能力差异也很大。

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vlm/visual.jpg" alt="VLM 能力" style="width: 90%; height: auto;"><br>
</p>

## 开源视觉语言模型概述

Hugging Face Hub 上有很多开放视觉语言模型，下表列出了其中一些佼佼者。

- 其中有基础模型，也有可用于对话场景的针对聊天微调的模型。
- 其中一些模型具有“接地（grounding ）”功能，因此能够减少模型幻觉。
- 除非另有说明，所有模型的训练语言皆为英语。

| 模型                  | 可否商用 | 模型尺寸 | 图像分辨率 | 其它能力               |
|------------------------|--------------------|------------|------------------|---------------------------------------|
| [LLaVA 1.6 (Hermes 34B)](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) | ✅                  | 34B        | 672x672          |                                       |
| [deepseek-vl-7b-base](https://huggingface.co/deepseek-ai/deepseek-vl-7b-base)    | ✅                  | 7B         | 384x384          |                                       |
| [DeepSeek-VL-Chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)       | ✅                  | 7B         | 384x384          | 聊天                                  |
| [moondream2](https://huggingface.co/vikhyatk/moondream2)             | ✅                  | ~2B        | 378x378          |                                       |
| [CogVLM-base](https://huggingface.co/THUDM/cogvlm-base-490-hf)            | ✅                  | 17B        | 490x490          |                                       |
| [CogVLM-Chat](https://huggingface.co/THUDM/cogvlm-chat-hf)            | ✅                  | 17B        | 490x490          | 接地、聊天                     |
| [Fuyu-8B](https://huggingface.co/adept/fuyu-8b)                | ❌                  | 8B         | 300x300          | 图像中的文本检测          |
| [KOSMOS-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)               | ✅                  | ~2B        | 224x224          | 接地、零样本目标检测 |
| [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL)                | ✅                  | 4B         | 448x448          | 零样本目标检测           |
| [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)           | ✅                  | 4B         | 448x448          | 聊天                                  |
| [Yi-VL-34B](https://huggingface.co/01-ai/Yi-VL-34B)              | ✅                  | 34B        | 448x448          |  双语（英文、中文） |


## 寻找合适的视觉语言模型

有多种途径可帮助你选择最适合自己的模型。

[视觉竞技场（Vision Arena）](https://huggingface.co/spaces/WildVision/vision-arena) 是一个完全基于模型输出进行匿名投票的排行榜，其排名会不断刷新。在该竞技场上，用户输入图像和提示，会有两个匿名的不同的模型为其生成输出，然后用户可以基于他们的喜好选择一个输出。这种方式生成的排名完全是基于人类的喜好的。

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vlm/arena.png" alt="视觉竞技场（Vision Arena）" style="width: 90%; height: auto;"><be>
<em>视觉竞技场（Vision Arena）</em>
</p>

[开放 VLM 排行榜](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)提供了另一种选择，各种视觉语言模型按照所有指标的平均分进行排名。你还可以按照模型尺寸、私有或开源许可证来筛选模型，并按照自己选定的指标进行排名。

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vlm/leaderboard.png" alt="VLM 能力" style="width: 90%; height: auto;"><be>
<em>开放 VLM 排行榜</em>
</p>

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 是一个工具包，用于在视觉语言模型上运行基准测试，开放 VLM 排行榜就是基于该工具包的。

还有一个评估套件是 [LMMS-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)，其提供了一个标准命令行界面，你可以使用 Hugging Face Hub 上托管的数据集来对选定的 Hugging Face 模型进行评估，如下所示：

```bash
accelerate launch --num_processes=8 -m lmms_eval --model llava --model_args pretrained="liuhaotian/llava-v1.5-7b" --tasks mme,mmbench_en --batch_size 1 --log_samples --log_samples_suffix llava_v1.5_mme_mmbenchen --output_path ./logs/ 
```

视觉竞技场和开放 VLM 排行榜都仅限于提交给它们的模型，且需要更新才能添加新模型。如果你想查找其他模型，可以在 `image-text-to-text` 任务下浏览 hub 中的[模型](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending)。

在排行榜中，你会看到各种不同的用于评估视觉语言模型的基准，下面我们选择其中几个介绍一下。

### MMMU

[针对专家型 AGI 的海量、多学科、多模态理解与推理基准（A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI，MMMU）](https://huggingface.co/datasets/MMMU/MMMU) 是评估视觉语言模型的最全面的基准。它包含 11.5K 个多模态问题，这些问题需要大学水平的学科知识以及跨学科（如艺术和工程）推理能力。

### MMBench

[MMBench](https://huggingface.co/datasets/lmms-lab/MMBench) 由涵盖超过 20 种不同技能的 3000 道单选题组成，包括 OCR、目标定位等。论文还介绍了一种名为 `CircularEval` 的评估策略，其每轮都会对问题的选项进行不同的组合及洗牌，并期望模型每轮都能给出正确答案。

另外，针对不同的应用领域还有其他更有针对性的基准，如 MathVista（视觉数学推理）、AI2D（图表理解）、ScienceQA（科学问答）以及 OCRBench（文档理解）。

## 技术细节

对视觉语言模型进行预训练的方法很多。主要技巧是统一图像和文本表征以将其输入给文本解码器用于文本生成。最常见且表现最好的模型通常由图像编码器、用于对齐图像和文本表征的嵌入投影子模型（通常是一个稠密神经网络）以及文本解码器按序堆叠而成。至于训练部分，不同的模型采用的方法也各不相同。

例如，LLaVA 由 CLIP 图像编码器、多模态投影子模型和 Vicuna 文本解码器组合而成。作者将包含图像和描述文本的数据集输入 GPT-4，让其描述文本和图像生成相关的问题。作者冻结了图像编码器和文本解码器，仅通过给模型馈送图像与问题并将模型输出与描述文本进行比较来训练多模态投影子模型，从而达到对齐图像和文本特征的目的。在对投影子模型预训练之后，作者把图像编码器继续保持在冻结状态，解冻文本解码器，然后继续对解码器和投影子模型进行训练。这种预训练加微调的方法是训练视觉语言模型最常见的做法。

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vlm/vlm-structure.png" alt="VLM Structure" style="width: 90%; height: auto;"><br>
 <em>视觉语言模型典型结构</em>
</p>
<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vlm/proj.jpg" alt="VLM Structure" style="width: 90%; height: auto;"><br>
 <em>将投影子模型输出与文本嵌入相串接</em>
</p>

再举一个 KOSMOS-2 的例子，作者选择了端到端地对模型进行完全训练的方法，这种方法与 LLaVA 式的预训练方法相比，计算上昂贵不少。预训练完成后，作者还要用纯语言指令对模型进行微调以对齐。还有一种做法，Fuyu-8B 甚至都没有图像编码器，直接把图像块馈送到投影子模型，然后将其输出与文本序列直接串接送给自回归解码器。

大多数时候，我们不需要预训练视觉语言模型，仅需使用现有的模型进行推理，抑或是根据自己的场景对其进行微调。下面，我们介绍如何在 `transformers` 中使用这些模型，以及如何使用 `SFTTrainer` 对它们进行微调。

## 在 transformers 中使用视觉语言模型

你可以使用 `LlavaNext` 模型对 Llava 进行推理，如下所示。

首先，我们初始化模型和数据处理器。

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to(device)
```
现在，将图像和文本提示传给数据处理器，然后将处理后的输入传给 `generate` 方法。请注意，每个模型都有自己的提示模板，请务必根据模型选用正确的模板，以避免性能下降。

```python
from PIL import Image
import requests

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=100)
```

调用 `decode` 对输出词元进行解码。

```python
print(processor.decode(output[0], skip_special_tokens=True))
```

## 使用 TRL 微调视觉语言模型

我们很高兴地宣布，作为一个实验性功能，[TRL](https://github.com/huggingface/trl) 的 `SFTTrainer` 现已支持视觉语言模型！这里，我们给出了一个例子，以展示如何在 [llava-instruct](https://Huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft) 数据集上进行 SFT，该数据集包含 260k 个图像对话对。

`llava-instruct` 数据集将用户与助理之间的交互组织成消息序列的格式，且每个消息序列皆与用户问题所指的图像配对。

要用上 VLM 训练的功能，你必须使用 `pip install -U trl` 安装最新版本的 TRL。你可在[此处](https://github.com/huggingface/trl/blob/main/examples/scripts/vsft_llava.py)找到完整的示例脚本。

```python
from trl.commands.cli_utils import SftScriptArguments, TrlParser

parser = TrlParser((SftScriptArguments, TrainingArguments))
args, training_args = parser.parse_args_and_config()
```

初始化聊天模板以进行指令微调。

```bash
LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""
```

现在，初始化模型和分词器。

```python
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration
import torch

model_id = "llava-hf/llava-1.5-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer = tokenizer

model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
```

建一个数据整理器来组合文本和图像对。

```python
class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"][0])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

data_collator = LLavaDataCollator(processor)
```

加载数据集。

```python
from datasets import load_dataset

raw_datasets = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]
```

初始化 `SFTTrainer`，传入模型、数据子集、PEFT 配置以及数据整理器，然后调用 `train()`。要将最终 checkpoint 推送到 Hub，需调用 `push_to_hub()`。

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",  # need a dummy field
    tokenizer=tokenizer,
    data_collator=data_collator,
    dataset_kwargs={"skip_prepare_dataset": True},
)

trainer.train()
```

保存模型并推送到 Hugging Face Hub。

```python
trainer.save_model(training_args.output_dir)
trainer.push_to_hub()
```

你可在[此处](https://huggingface.co/HuggingFaceH4/vsft-llava-1.5-7b-hf-trl)找到训得的模型。你也可以通过下面的页面试玩一下我们训得的模型⬇️。

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/3.23.0/gradio.js"></script>

<gradio-app theme_mode="light" src="https://HuggingFaceH4-vlm-playground.hf.space"></gradio-app>

**致谢**

我们感谢 Pedro Cuenca、Lewis Tunstall、Kashif Rasul 和 Omar Sanseviero 对本文的评论和建议。

> 英文原文: <url> https://huggingface.co/blog/vlms </url>
> 原文作者：Merve Noyan，Edward Beeching
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。
