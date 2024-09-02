---
title: "通过打包 Flash Attention 来提升 Hugging Face 训练效率"
thumbnail: /blog/assets/packing-with-FA2/thumbnail.png
authors:
- user: RQlee
  guest: true
  org: ibm
- user: ArthurZ
- user: achikundu
  guest: true
  org: ibm
- user: lwtr
  guest: true
  org: ibm
- user: rganti
  guest: true
  org: ibm
- user: mayank-mishra
  guest: true
  org: ibm
translators:
- user: innovation64
- user: zhongdongy
  proofreader: true
---

## 简单概述

现在，在 Hugging Face 中，使用打包的指令调整示例 (无需填充) 进行训练已与 Flash Attention 2 兼容，这要归功于一个 [最近的 PR](https://github.com/huggingface/transformers/pull/31629) 以及新的 [DataCollatorWithFlattening](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithFlattening)。

它可以在保持收敛质量的同时，将训练吞吐量提高多达 2 倍。继续阅读以了解详细信息！

## 简介

在训练期间，对小批量输入进行填充是一种常见的整理输入数据的方法。然而，由于无关的填充 token ，这引入了效率低下的问题。不进行填充而是打包示例，并使用 token 位置信息，是一种更高效的选择。然而，之前打包的实现并没有在使用 Flash Attention 2 时考虑示例边界，导致出现不希望的跨示例注意力，这降低了质量和收敛性。

Hugging Face Transformers 现在通过一项新功能解决了这个问题，该功能在打包时保持对边界的意识，同时引入了一个新的数据整理器 `DataCollatorWithFlattening` 。

通过选择 `DataCollatorWithFlattening` ，Hugging Face `Trainer` 的用户现在可以无缝地将序列连接成一个单一的张量，同时在 Flash Attention 2 计算过程中考虑到序列边界。这是通过 `flash_attn_varlen_func` 实现的，它计算每个小批量的累积序列长度 ( `cu_seqlens` )。
同样的功能也适用于 `TRL` 库中的 Hugging Face `SFTTrainer` 用户，通过在调用数据整理器 `DataCollatorForCompletionOnlyLM` 时设置一个新的标志 `padding_free=True` 来实现。

## 吞吐量提高多达 2 倍

我们使用带有新 `DataCollatorWithFlattening` 的此功能在训练过程中看到了显著的处理吞吐量提升。下图显示了在训练期间测量的吞吐量，单位为 token /秒。在这个例子中，吞吐量是在 8 个 A100-80 GPU 上对一个 epoch 内的 20K 个随机选自两个不同指令调整数据集 (FLAN 和 OrcaMath) 的样本的平均值。

![throughput](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/thruput.png)

FLAN 数据集的平均序列较短，但序列长度差异较大，因此每个批次中的示例长度可能会有很大差异。这意味着填充的 FLAN 批次可能会因为未使用的填充 token 而产生显著的开销。在 FLAN 数据集上进行训练时，使用新的 `DataCollatorWithFlattening` 在提高吞吐量方面显示出显著的优势。我们在这里展示的模型中看到了 2 倍的吞吐量提升: llama2-7B、mistral-7B 和 granite-8B-code。

OrcaMath 数据集的示例较长，且示例长度差异较小。因此，从打包中获得的改进较低。我们的实验显示，在使用这种打包方式在 OrcaMath 数据集上训练时，这三个模型的吞吐量增加了 1.4 倍。

![memory](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/memory.png)

通过使用新的 `DataCollatorWithFlattening` 进行打包，内存使用也有所改善。下图显示了相同的三个模型在相同的两个数据集上训练时的峰值内存使用情况。在 FLAN 数据集上，峰值内存减少了 20%，这得益于打包的显著好处。

在 OrcaMath 数据集上，由于其示例长度更为均匀，峰值内存减少了 6%。

当打包示例减少了优化步骤的数量时，可能会损害训练的收敛性。然而，这个新功能保留了小批量，因此与使用填充示例相同的优化步骤数量。因此，对训练收敛性没有影响，正如我们在下一个图中看到的那样，该图显示了相同的三个模型在相同的两个数据集上训练时，无论是使用新的 `DataCollatorWithFlattening` 进行打包还是使用填充，模型的验证损失是相同的。

![ValLoss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/ValLoss.png)

## 工作原理

考虑一个批处理数据，其中批量大小 (batchsize) 为 4，四个序列如下:

![batch](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/four_sequences.png)

在将示例连接之后，无填充整理器返回每个示例的 `input_ids` 、 `labels` 和 `position_ids` 。因此，对于这批数据，整理器提供了以下内容:

![example](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/input_ids_labels_position_ids.png)

所需的修改是轻量级的，仅限于向 Flash Attention 2 提供 `position_ids` 。

然而，这依赖于模型暴露 `position_ids` 。在撰写本文时，有 14 个模型暴露了它们，并且受该解决方案的支持。具体来说，Llama 2 和 3、Mistral、Mixtral、Granite、DBRX、Falcon、Gemma、OLMo、Phi 1、2 和 3、phi3、Qwen 2 和 2 MoE、StableLM 以及 StarCoder 2 都受该解决方案支持。

## 开始使用

利用 `position_ids` 进行打包的好处很容易实现。

如果你正在使用 Hugging Face `Transformers` 中的 `Trainer` ，只需两个步骤:

1. 使用 Flash Attention 2 实例化模型
2. 使用新的 `DataCollatorWithFlattening`

如果你正在使用 `TRL` 中的 Hugging Face `SFTTrainer` 配合 `DataCollatorForCompletionOnlyLM` ，那么所需的两个步骤是:

1. 使用 Flash Attention 2 实例化模型
2. 在调用 `DataCollatorForCompletionOnlyLM` 时设置 `padding_free=True` ，如下所示:
`collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, padding_free=True)`

## 如何使用它

对于 `Trainer` 用户，下面的例子展示了如何使用这个新功能。

```Python
# 使用 DataCollatorWithFlattening 的示例
 
import torch

# 加载模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "instructlab/merlinite-7b-lab",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# 读取数据集
from datasets import load_dataset
train_dataset = load_dataset("json", data_files="path/to/my/dataset")["train"]

# 使用 DataCollatorWithFlattening
from transformers import DataCollatorWithFlattening
data_collator = DataCollatorWithFlattening()

# 训练
from transformers import TrainingArguments, Trainer
train_args = TrainingArguments(output_dir="/save/path")
trainer = Trainer(
    args=train_args,
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator
)
trainer.train()
```

对于 `TRL` 用户，下面的例子展示了如何在使用 `SFTTrainer` 时使用这个新功能。

```Python
# 使用 DataCollatorForCompletionOnlyLM SFTTrainer 示例

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

model = AutoModelForCausalLM.from_pretrained(
    "instructlab/merlinite-7b-lab",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("instructlab/merlinite-7b-lab")
tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, padding_free=True)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./tmp",
        gradient_checkpointing=True,
        per_device_train_batch_size=8
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
```

## 结论

得益于最近的 PR 和新推出的 `DataCollatorWithFlattening` ，现在打包指令调整示例 (而不是填充) 已与 Flash Attention 2 完全兼容。这种方法与使用 `position_ids` 的模型兼容。在训练期间可以观察到吞吐量和峰值内存使用的改善，而训练收敛性没有下降。实际的吞吐量和内存改善取决于模型以及训练数据中示例长度的分布。对于具有广泛示例长度变化的训练数据，使用 `DataCollatorWithFlattening` 相对于填充将获得最大的益处。 `TRL` 库中的 `SFTTrainer` 用户可以通过在调用 `DataCollatorForCompletionOnlyLM` 时设置新的标志 `padding_free=True` 来使用同一功能。
想要更详细的分析，请查看论文: https://huggingface.co/papers/2407.09105。