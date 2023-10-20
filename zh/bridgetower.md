---
title: "使用 Habana Gaudi2 加速视觉语言模型 BridgeTower"
thumbnail: /blog/assets/bridgetower/thumbnail.png
authors:
- user: regisss
- user: anahita-b
  guest: true
translators:
- user: MatrixYao
---

# 使用 Habana Gaudi2 加速视觉语言模型 BridgeTower


*更新（29/08/2023）：本文新增了 H100 的基准测试。另外，我们还使用最新版本的软件刷新了所有的性能数据。

在对最先进的视觉语言模型 BridgeTower 进行微调时，使用 [Optimum Habana v1.7](https://github.com/huggingface/optimum-habana/tree/main)， Habana Gaudi2 的速度可以达到**A100 的 2.5 倍， H100 的 1.4 倍**。其中硬件加速的数据加载对性能提高影响最大。

*这些技术适用于任何性能瓶颈在数据加载上的其他工作负载，很多视觉模型的性能瓶颈在数据加载。* 本文将带你了解我们用于比较 Habana Gaudi2、英伟达 H100 以及英伟达 A100 80GB 上的 BridgeTower 微调性能的流程及测试基准。本文还展示了如何在 transformer 类模型上轻松用上这些优化。 

## BridgeTower

最近，[视觉语言 (Vision-Language，VL) 模型](https://huggingface.co/blog/vision_language_pretraining)的重要性与日俱增，它们开始在各种 VL 任务中占据主导地位。在处理多模态数据时，最常见的做法是使用单模态编码器从各模态数据中提取各自的数据表征。然后，抑或是将这些表征融合在一起，抑或是将它们输入给跨模态编码器。为了有效解除传统 VL 表征学习的算法局限性及其性能限制，[BridgeTower](https://huggingface.co/papers/2206.08657) 引入了多个*桥接层*，在单模态编码器的顶部若干层建立与跨模态编码器的逐层连接，这使得跨模态编码器中不同语义级别的视觉和文本表征之间能够实现有效的、自底而上的跨模态对齐和融合。

仅基于 400 万张图像预训练而得的 BridgeTower 模型就能在各种下游视觉语言任务上取得最先进的性能（详见[下文](#基准测试)）。特别地，BridgeTower 在使用相同的预训练数据和几乎可以忽略不计的额外参数和计算成本的条件下，在 VQAv2 的 `test-std` 子集上取得了 78.73% 的准确率，比之前最先进的模型 (METER) 的准确率提高了 1.09%。值得一提的是，当进一步增加模型参数量，BridgeTower 的准确率可达 81.15%，超过了那些基于大得多的数据集预训练出来的模型。

## 硬件

[英伟达 H100 张量核 GPU](https://www.nvidia.com/en-us/data-center/h100/) 是最快以及最新一代的英伟达 GPU。它有一个专门的 transformer 引擎，可以用加速 fp8 混合精度运算。它还有一个 80GB 显存的版本。

[英伟达 A100 张量核 GPU](https://www.nvidia.com/en-us/data-center/a100/) 内含第三代[张量核技术](https://www.nvidia.com/en-us/data-center/tensor-cores/)。目前来讲 A100 仍然是大多数云服务上最快的 GPU。这里，我们使用显存为 80GB 的卡，它的显存容量和带宽都比 40GB 版本更高。

[Habana Gaudi2](https://habana.ai/products/gaudi2/) 是 Habana Labs 设计的第二代 AI 硬件加速卡。一台服务器包含 8 个称为 HPU 的加速卡，每张加速卡有 96GB 内存。你可查阅[我们之前的博文](https://huggingface.co/blog/habana-gaudi-2-bloom#habana-gaudi2)，以了解 Gaudi2 的更多信息以及如何在[英特尔开发者云（Intel Developer Cloud，IDC）](https://www.intel.com/content/www/us/en/secure/developer/devcloud/cloud-launchpad.html)上获取 Gaudi2 实例。与市面上许多其他 AI 加速器不同，用户很容易通过 [Optimum Habana](https://huggingface.co/docs/optimum/habana/index) 使用到 Gaudi2 的高级特性。有了 Optimum Habana，用户仅需更改 2 行 代码即可将基于 `transformers` 的模型脚本移植到 Gaudi 上。

## 基准测试

为了评测训练性能，我们准备微调 [BridgeTower 的 large checkpoint](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc)，其参数量为 866M。该 checkpoint 在预训练时使用了掩码语言模型、图像文本匹配以及图像文本对比损失，其预训练数据集为 [Conceptual Captions](https://huggingface.co/datasets/conceptual_captions)、[SBU Captions](https://huggingface.co/datasets/sbu_captions)、[MSCOCO Captions](https://huggingface.co/datasets/HuggingFaceM4/COCO) 以及 [Visual Genome](https://huggingface.co/datasets/visual_genome)。

我们将在[纽约客配文竞赛数据集](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest)上进一步微调此 checkpoint，该数据集包含《纽约客》杂志上的漫画及每个漫画对应的投票最多的配文。

三种加速卡的微调超参数相同，其单卡 batch size 都设为 48。你可以在[这儿](https://huggingface.co/regisss/bridgetower-newyorker-gaudi2-8x#training-hyperparameters)找到 Gaudi2 上使用的训练超参，并在[这儿](https://huggingface.co/regisss/bridgetower-newyorker-a100-8x#training-hyperparameters)找到 A100 上使用的超参。

**在处理含图像的数据集时，数据加载通常是性能瓶颈之一**，这是因为一般情况下很多预处理操作都是在 CPU 上完成的（如图像解码、图像增强等），然后再将预处理后的图像发送至训练卡。这带来了一个优化点，理想情况下，*我们可以直接将原数据发送到设备，并在设备上执行解码和各种图像变换*。但在此之前，我们先看看能不能简单地通过分配更多 CPU 资源来加速数据加载。

### 利用 `dataloader_num_workers`

如果图像加载是在 CPU 上完成的，一个简单地加速方法就是分配更多的子进程来加载数据。使用 transformers 的 `TrainingArguments`（或 Optimum Habana 中相应的 `GaudiTrainingArguments`）可以很容易地做到这一点：你可以用 `dataloader_num_workers=N` 参数来设置 CPU 上用于数据加载的子进程的数目 (`N`)。

`dataloader_num_workers` 参数的默认值为 0，表示仅在主进程中加载数据​​。这个设置在很多情况下无法达到最佳性能，因为主进程还有很多其他事情要做。我们可以将其设置为 1，这样就会有一个专门的子进程来加载数据。当分配多个子进程时，每个子进程会负责准备一个 batch。这意味着内存消耗将随着工作进程数的增加而增加。一个简单的方法是将其设置为 CPU 核数，但有时候有些核可能在做别的事情，因此需要尝试找到一个最佳配置。

下面，我们跑三组实验：
- 8 卡分布式混合精度 (*bfloat16*/*float*) 训练，其中数据加载由各 rank 的主进程执行（即 `dataloader_num_workers=0`）
- 8 卡分布式混合精度 (*bfloat16*/*float*) 训练，且每 rank 各有 1 个用于数据加载的专用子进程（即 `dataloader_num_workers=1`）
- `dataloader_num_workers=2`

以下是这三组实验在 Gaudi2、H100 以及 A100 上分别测得的吞吐量（单位：每秒样本数）：

| 设备     | `dataloader_num_workers=0` | `dataloader_num_workers=1` | `dataloader_num_workers=2` |
|:----------:|:--------------------------:|:--------------------------:|:--------------------------:|
| Gaudi2 HPU | 601.5           | 747.4            | 768.7          |
| H100 GPU   | 336.5             | 580.1            | 602.1            |
| A100 GPU   | 227.5           | 339.7           | 345.4           |

首先，我们看到在 **`dataloader_num_workers=2` 时 Gaudi2 的速度是 H100 的 1.28 倍**，在 `dataloader_num_workers=1` 时为 1.29 倍，在 `dataloader_num_workers=0` 时为 1.99 倍。与 H100 的前代产品 A100 相比就更快了，在 **`dataloader_num_workers=2` 时 Gaudi2 的速度是 A100 的 2.23 倍**，在 `dataloader_num_workers=1` 时为 2.20 倍，在 `dataloader_num_workers=0` 时为 2.64 倍。这些数据比我们之前[报告的数据](https://huggingface.co/blog/habana-gaudi-2-benchmark)还要好！

其次，我们还看到**为数据加载分配更多资源可以轻松实现加速**：Gaudi2 上加速比为 1.28，H100 上的加速比为 1.79， 而A100 上加速比为 1.52。

我们还尝试了进一步增加数据加载子进程数，但实验表明，在所有加速器上，性能都没有比 `dataloader_num_workers=2` 更好。

因此，**使用 `dataloader_num_workers>0` 通常是加速涉及到图像的工作负载时首先尝试的方法！**

你可以在[这儿](https://huggingface.co/regisss/bridgetower-newyorker-gaudi2-8x/tensorboard)找到可视化的 Gaudi2 Tensorboard 日志，A100 的在[这儿](https://huggingface.co/regisss/bridgetower-newyorker-a100-8x/tensorboard)。

<!-- ### Optimum Habana 的 fast DDP

在深入研究硬件加速的数据加载之前，我们来看一下另一个非常简单的 Gaudi 分布式运行的加速方法。新发布的 Optimum Habana 1.6.0 版引入了一个新功能，允许用户选择分布式策略：
- `distribution_strategy="ddp"` 使用 PyTorch 的 [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)（DDP）实现
- `distribution_strategy="fast_ddp"` 使用 Gaudi 自有的更轻量级且一般来讲更快的实现

Optimum Habana 的 `fast DDP` 不会像 [DDP](https://pytorch.org/docs/stable/notes/ddp.html#internal-design) 那样将参数梯度分割到存储桶（bucket）中。它还会使用 [HPU 图（graph）](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html?highlight=hpu%20graphs)来收集所有进程的梯度，并以最小的主机开销来对它们进行更新（在[all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)操作之后）。你可以在[这儿](https://github.com/huggingface/optimum-habana/blob/main/optimum/habana/distributed/fast_ddp.py)找到其实现。

只需在 Gaudi2 上使用 `distribution_strategy="fast_ddp"`（并保持 `dataloader_num_workers=1`）即可将每秒吞吐提高到 705.9，**比 DDP 快 1.10 倍，比 A100 快 2.38 倍！**

因此，仅添加两个训练参数（`dataloader_num_workers=1` 及 `distribution_strategy="fast_ddp"`），我们即可在 Gaudi2 上实现 1.33 倍的加速，与使用 `dataloader_num_workers=1` 的 A100 相比，加速比达到 2.38 倍。-->

### 使用 Optimum Habana 实现硬件加速的数据加载

为了获得更多的加速，我们可以将尽可能多的数据加载操作从 CPU 上移到加速卡上（即 Gaudi2 上的 HPU 或 H100/A100 上的 GPU）。在 Gaudi2 上，我们可以使用 Habana 的[多媒体流水线（media pipeline）](https://docs.habana.ai/en/latest/Media_Pipeline/index.html)来达到这一目的。

给定一个数据集，大多数的数据加载器会做如下操作：

1. 获取数据（例如，存储在磁盘上的 JPEG 文件）
2. CPU 读取编码图像
3. CPU 对图像进行解码
4. CPU 对图像进行变换来增强图像
5. 最后，将图像发送至设备（尽管这通常不是由数据加载器本身完成的）

与在 CPU 上完成整个过程后再把准备好训练的数据发送到加速卡相比，更高效的方法是先将编码图像发送到加速卡，然后由加速卡执行图像解码和增强：
1. 同上
2. 同上
3. 将编码图像发送至加速卡
4. 加速卡对图像进行解码
5. 加速卡对图像进行变换来增强图像

这样我们就可以利用加速卡强大的计算能力来加速图像解码和变换。请注意，执行此操作时需要注意两个问题：
- 设备内存消耗将会增加，因此如果没有足够的可用内存，你需要减小 batch size。这可能会降低该方法带来的加速。
- 如果在使用 CPU 数据加载方案时，加速卡的利用率已经很高（100% 或接近 100%）了，那就不要指望把这些操作卸载到加速卡会获得加速，因为它们已经忙得不可开交了。

我们还提供了一个示例，以帮助你在 Gaudi2 上实现这一优化：Optimum Habana 中的[对比图像文本示例代码](https://github.com/huggingface/optimum-habana/tree/main/examples/contrastive-image-text)提供了一个可直接使用的多媒体流水线，你可以将其直接用于类似于 COCO 那样的含文本和图像的数据集！只需在命令中加一个 `--mediapipe_dataloader` 即可使能它。

感兴趣的读者可以参阅 Gaudi 的[文档](https://docs.habana.ai/en/latest/Media_Pipeline/index.html)，该文档对这一机制的底层实现给出了一些概述。读者还可以参考[这个文档](https://docs.habana.ai/en/latest/Media_Pipeline/Operators.html)，它列出了目前支持的所有算子。

现在我们加上 `mediapipe_dataloader` 参量重跑一下之前的实验，该参量可以与 `dataloader_num_workers` 参量同时使用：

| 设备     | `dataloader_num_workers=0` | `dataloader_num_workers=2` |  `dataloader_num_workers=2` + `mediapipe_dataloader` |
|:----------:|:--------------------------:|:--------------------------------------------:|:---------------:|
| Gaudi2 HPU | 601.5 samples/s            | 768.7 samples/s                              | 847.7 samples/s |
| H100 GPU   | 336.5 samples/s            | 602.1 samples/s                              | /               |
| A100 GPU   | 227.5 samples/s            | 345.4 samples/s                              | /               |

与之前基于 `dataloader_num_workers=2` 的性能数据相比，我们又额外获得了 1.10 倍的加速。因此，最终，仅通过添加两个简单的训练参量，我们在 Gaudi2 上获得了相比基线 1.41 倍的性能提升。在 `dataloader_num_workers=2` 的条件下，**其性能是 H100 的 1.41 倍, A100 的 2.45 倍**！

### 如何复现我们的基准测试

如需复现我们的基准测试，你首先需要访问[英特尔开发者云（Intel Developer Cloud，IDC）](https://www.intel.com/content/www/us/en/secure/developer/devcloud/cloud-launchpad.html)上的 Gaudi2 实例（更多信息请参阅[本指南](https://huggingface.co/blog/habana-gaudi-2-benchmark#how-to-get-access-to-gaudi2)）。

然后，安装最新版本的 Optimum Habana 并运行 `run_bridgetower.py`（见[此处](https://github.com/huggingface/optimum-habana/blob/main/examples/contrastive-image-text/run_bridgetower.py)）。具体命令如下：

```bash
pip install optimum[habana]
git clone https://github.com/huggingface/optimum-habana.git
cd optimum-habana/examples/contrastive-image-text
pip install -r requirements.txt
```

运行脚本需使用的命令如下：

```bash
python ../gaudi_spawn.py --use_mpi --world_size 8 run_bridgetower.py \
--output_dir /tmp/bridgetower-test \
--model_name_or_path BridgeTower/bridgetower-large-itm-mlm-itc \
--dataset_name jmhessel/newyorker_caption_contest --dataset_config_name matching \
--image_column image --caption_column image_description \
--remove_unused_columns=False \
--do_train --do_eval --do_predict \
--per_device_train_batch_size="40" --per_device_eval_batch_size="16" \
--num_train_epochs 5 \
--learning_rate="1e-5" \
--push_to_hub --report_to tensorboard --hub_model_id bridgetower\
--overwrite_output_dir \
--use_habana --use_lazy_mode --use_hpu_graphs_for_inference --gaudi_config_name Habana/clip \
--throughput_warmup_steps 3 \
--logging_steps 10
```

上述命令对应于 `--dataloader_num_workers 0`。如果要运行其他配置，你可以视情况添加 `--dataloader_num_workers N` 及 `--mediapipe_dataloader`。

如要将模型和 Tensorboard 日志推送到 Hugging Face Hub，你需要事先登录自己的帐户：

```bash
huggingface-cli login
```

在 H100 或 A100 上运行，你可以使用相同的 `run_bridgetower.py` 脚本，但需要做一些小更改：
- 将 `GaudiTrainer` 和 `GaudiTrainingArguments` 替换为 `transformers` 的 `Trainer` 和 `TrainingArguments`
- 删除 `GaudiConfig`、`gaudi_config` 和 `HabanaDataloaderTrainer` 的相关代码
- 直接从 `transformers` 导入 `set_seed`：`from transformers import set_seed`

本文中有关 H100 的数据是基于一个英伟达 H100 Lambda 实例测得的，而 A100 的数据是基于一个英伟达 A100 80GB GCP 实例测得的，这两个实例均为 8 卡实例，且我们使用了[英伟达官方 Docker 镜像](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)。

请注意，`--mediapipe_dataloader` 仅适用于 Gaudi2，不适用于 H100/A100。

那如果我们在 H100 上使用 fp8 从而利用其 [transformer 引擎](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) 的加速能力，性能会如何呢？因为代码会出现崩溃以及涉及到对 `transformers` 里的 BridgeTower 模型代码的修改，所以我们尚未测得相应的数据。我们会在 Gaudi2 支持 fp8 后再进行测试对比。

## 总结

在处理图像时，我们提出了两个用于加速训练工作流的解决方案：1）分配更多的 CPU 资源给数据加载器，2）直接在加速卡上而不是 CPU 上解码和变换图像。

我们证明，在训练像 BridgeTower 这样的 SOTA 视觉语言模型时，它会带来显著的加速：**基于 Optimum Habana 的 Habana Gaudi2 的速度是基于 Transformers 的英伟达 H100 的约 1.4 倍，是 A100 80GB 的约 2.5倍！**。而为了获得这些加速，你只需在训练脚本中额外加几个参数即可，相当容易！

后面，我们期待能使用 HPU 图进一步加速训练，我们还计划向大家展示如何在 Gaudi2 上使用 DeepSpeed ZeRO-3 来加速 LLM 的训练。敬请关注！

如果你对使用最新的 AI 硬件加速卡和软件库加速机器学习训练和推理工作流感兴趣，可以移步我们的[专家加速计划](https://huggingface.co/support)。如果你想了解有关 Habana 解决方案的更多信息，可以点击[此处](https://huggingface.co/hardware/habana)了解相关信息并联系他们。要详细了解 Hugging Face 为让 AI 硬件加速卡更易于使用而做的努力，请查阅我们的[硬件合作伙伴计划](https://huggingface.co/hardware)。

### 相关话题

- [更快的训练和推理：对比 Habana Gaudi2 和英伟达 A100 80GB](https://huggingface.co/blog/zh/habana-gaudi-2-benchmark)
- [大语言模型快速推理：在 Habana Gaudi2 上推理 BLOOMZ](https://huggingface.co/blog/zh/habana-gaudi-2-bloom)