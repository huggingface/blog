---
title: "大语言模型快速推理：在 Habana Gaudi2 上推理 BLOOMZ"
thumbnail: /blog/assets/habana-gaudi-2-bloom/thumbnail.png
authors:
- user: regisss
translators:
- user: MatrixYao
---

# 大语言模型快速推理：在 Habana Gaudi2 上推理 BLOOMZ


本文将展示如何在 [Habana® Gaudi®2](https://habana.ai/training/gaudi2/) 上使用 🤗 [Optimum Habana](https://huggingface.co/docs/optimum/habana/index)。Optimum Habana 是 Gaudi2 和 🤗 Transformers 库之间的桥梁。本文设计并实现了一个大模型推理基准测试，证明了通过使用 Optimum Habana 你将能够在 Gaudi2 上获得 **比目前市面上任何可用的 GPU 都快的推理速度**。

随着模型越来越大，将它们部署到生产环境中以用于推理也变得越来越具有挑战性。硬件和软件都需要很多创新来应对这些挑战，让我们来深入了解 Optimum Habana 是如何有效地克服这些挑战的！

## BLOOMZ

[BLOOM](https://arxiv.org/abs/2211.05100) 是一个 1760 亿参数的自回归模型，经训练后可用于文本生成。它可以处理 46 种不同的语言以及 13 种编程语言。作为 [BigScience](https://bigscience.huggingface.co/) 计划的一部分，BLOOM 作为一个开放科学项目，来自全球的大量的研究人员和工程师参与了模型的设计和训练。最近，我们又发布了架构与 BLOOM 完全相同的模型：[BLOOMZ](https://arxiv.org/abs/2211.01786)，它是 BLOOM 在多个任务上的微调版本，具有更好的泛化和零样本[^1] 能力。

如此大的模型在 [训练](https://huggingface.co/blog/bloom-megatron-deepspeed) 和 [推理](https://huggingface.co/blog/bloom-inference-optimization) 两个场景下都对内存和速度提出了新的挑战。即使是使用 16 位精度，一个模型也需要 352 GB 的内存！目前你可能很难找到一个具有如此大内存的设备，但像 Habana Gaudi2 这样先进的硬件已能让低延迟 BLOOM 和 BLOOMZ 模型推理变得可能。

## Habana Gaudi2

[Gaudi2](https://habana.ai/training/gaudi2/) 是 Habana Labs 设计的第二代 AI 硬件加速器。单个服务器包含 8 张加速卡（称为 Habana 处理单元（Habana Processing Units），或 HPU），每张卡有 96GB 的内存，这为容纳超大模型提供了可能。但是，如果仅仅是内存大，而计算速度很慢，也没办法将其用于模型托管服务。幸运的是，Gaudi2 在这方面证明了自己，大放异彩：它与 GPU 的不同之处在于，它的架构使加速器能够并行执行通用矩阵乘法 (General Matrix Multiplication，GeMM) 和其他操作，从而加快了深度学习工作流。这些特性使 Gaudi2 成为 LLM 训练和推理的理想方案。

Habana 的 SDK SynapseAI™ 支持 PyTorch 和 DeepSpeed 以加速 LLM 训练和推理。[SynapseAI 图编译器](https://docs.habana.ai/en/latest/Gaudi_Overview/SynapseAI_Software_Suite.html#graph-compiler-and-runtime) 会优化整个计算图的执行过程（如通过算子融合、数据布局管理、并行化、流水线、内存管理、图优化等手段）。

此外，最近 SynapseAI 还引入了 [HPU graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) 和 [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html) 的支持，这两者非常适合延迟敏感型的应用，下面的基准测试结果即很好地说明了这一点。

以上所有功能都集成进了 🤗 [Optimum Habana](https://github.com/huggingface/optimum-habana) 库，因此在 Gaudi 上部署模型非常简单。你可以阅读 [此处](https://huggingface.co/docs/optimum/habana/quickstart)快速起步。

如果你想试试 Gaudi2，请登录 [英特尔开发者云（Intel Developer Cloud）](https://www.intel.com/content/www/us/en/secure/developer/devcloud/cloud-launchpad.html) 并按照[本指南](https://huggingface.co/blog/habana-gaudi-2-benchmark#how-to-get-access-to-gaudi2)申请。

## 测试基准

在本节中，我们将提供 BLOOMZ 在 Gaudi2、第一代 Gaudi 和 Nvidia A100 80GB 上的早期基准测试。虽然这些设备内存都不小，但由于模型太大，单个设备还是放不下整个 BLOOMZ 模型。为了解决这个问题，我们要使用 [DeepSpeed](https://www.deepspeed.ai/)，这是一个深度学习优化库，它实现了很多内存优化和速度优化以加速模型推理。特别地，我们在这里依赖 [DeepSpeed-inference](https://arxiv.org/abs/2207.00032)，它引入了几个特性，如[模型（或管道）并行](https://huggingface.co/blog/bloom-megatron-deepspeed#pipeline-parallelism)以充分利用可用设备。对 Gaudi2，我们使用 [Habana 的 DeepSpeed 分支](https://github.com/HabanaAI/deepspeed) ，其添加了对 HPU 的支持。

### 延迟

我们测量了两种不同大小的 BLOOMZ 模型的延迟（batch size 为 1），两者参数量都有数十亿：
- [1760 亿](https://huggingface.co/bigscience/bloomz) 参数
- [70 亿](https://huggingface.co/bigscience/bloomz-7b1) 参数

我们使用 DeepSpeed-inference 以 16 位精度在8 张卡上运行推理，同时我们开启了 [key-value 缓存](https://huggingface.co/docs/transformers/v4.27.1/en/model_doc/bloom#transformers.BloomForCausalLM.forward.use_cache) 优化。请注意，尽管 [CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/) 当前与 DeepSpeed 中的模型并行不兼容（DeepSpeed v0.8.2，请参见 [此处](https://github.com/microsoft/DeepSpeed/blob/v0.8.2/deepspeed/inference/engine.py#L158)，但 Habana 的 DeepSpeed 分支是支持 HPU graphs 的。所有基准测试都使用 [贪心搜索](https://huggingface.co/blog/how-to-generate#greedy-search)生成 100 个词元。输入提示为：
> "DeepSpeed is a machine learning framework"
该提示会被 BLOOM 分词器分成 7 个词元。

推理延迟测试结果如下表所示（单位为*秒*）。

| 模型       | 卡数 | Gaudi2 延迟（秒） | A100-80GB 延迟（秒） | 第一代 Gaudi 延迟（秒） |
|:-----------:|:-----------------:|:-------------------------:|:-----------------:|:----------------------------------:|
| BLOOMZ | 8 | 3.717 | 4.402 | / |
| BLOOMZ-7B | 8 | 0.737 | 2.417 | 3.029 |
| BLOOMZ-7B | 1 | 1.066 | 2.119 | 2.865 |

Habana 团队最近在 SynapseAI 1.8 中引入了对 DeepSpeed-inference 的支持，从而快速支持了 1000 多亿参数模型的推理。 **对于 1760 亿参数的模型，Gaudi2 比 A100 80GB 快 1.2 倍**。较小模型上的结果更有意思：**对于 BLOOMZ-7B，Gaudi2 比 A100 快 3 倍**。 有趣的是，BLOOMZ-7B 这种尺寸的模型也能受益于模型并行。

我们还在第一代 Gaudi 上运行了这些模型。虽然它比 Gaudi2 慢，但从价格角度看很有意思，因为 AWS 上的 DL1 实例每小时大约 13 美元。BLOOMZ-7B 在第一代 Gaudi 上的延迟为 2.865 秒。因此，**对70亿参数的模型而言，第一代 Gaudi 比 A100 的性价比更高，每小时能省 30 多美元**！

我们预计 Habana 团队将在即将发布的新 SynapseAI 版本中继续优化这些模型的性能。在我们上一个基准测试中，我们看到 [Gaudi2 的 Stable Diffusion推理速度比 A100 快 2.2 倍](https://huggingface.co/blog/habana-gaudi-2-benchmark#generating-images-from-text-with-stable-diffusion)，这个优势在随后 Habana 提供的最新优化中进一步提高到了 2.37 倍。在 SynapseAI 1.9 的预览版中，我们看到 BLOOMZ-176B 的推理延迟进一步降低到了 3.5 秒。当新版本的 SynapseAI 发布并集成到 Optimum Habana 中时，我们会更新最新的性能数字。

### 在完整数据集上进行推理

我们的脚本允许支持模型整个数据集上逐句进行文本补全。如果你想在自己的数据集上尝试用 Gaudi2 进行 BLOOMZ 推理，这个脚本就很好用。

这里我们以 [*tldr_news*](https://huggingface.co/datasets/JulesBelveze/tldr_news/viewer/all/test) 数据集为例。该数据每一条都包含文章的标题和内容（你可以在 Hugging Face Hub 上可视化一下数据）。这里，我们仅保留 *content* 列（即内容）并对每个样本只截前 16 个词元，然后让模型来生成后 50 个词元。前 5 条数据如下所示：

```
Batch n°1
Input: ['Facebook has released a report that shows what content was most widely viewed by Americans between']
Output: ['Facebook has released a report that shows what content was most widely viewed by Americans between January and June of this year. The report, which is based on data from the company’s mobile advertising platform, shows that the most popular content on Facebook was news, followed by sports, entertainment, and politics. The report also shows that the most']
--------------------------------------------------------------------------------------------------
Batch n°2
Input: ['A quantum effect called superabsorption allows a collection of molecules to absorb light more']
Output: ['A quantum effect called superabsorption allows a collection of molecules to absorb light more strongly than the sum of the individual absorptions of the molecules. This effect is due to the coherent interaction of the molecules with the electromagnetic field. The superabsorption effect has been observed in a number of systems, including liquid crystals, liquid crystals in']
--------------------------------------------------------------------------------------------------
Batch n°3
Input: ['A SpaceX Starship rocket prototype has exploded during a pressure test. It was']
Output: ['A SpaceX Starship rocket prototype has exploded during a pressure test. It was the first time a Starship prototype had been tested in the air. The explosion occurred at the SpaceX facility in Boca Chica, Texas. The Starship prototype was being tested for its ability to withstand the pressure of flight. The explosion occurred at']
--------------------------------------------------------------------------------------------------
Batch n°4
Input: ['Scalene is a high-performance CPU and memory profiler for Python.']
Output: ['Scalene is a high-performance CPU and memory profiler for Python. It is designed to be a lightweight, portable, and easy-to-use profiler. Scalene is a Python package that can be installed on any platform that supports Python. Scalene is a lightweight, portable, and easy-to-use profiler']
--------------------------------------------------------------------------------------------------
Batch n°5
Input: ['With the rise of cheap small "Cube Satellites", startups are now']
Output: ['With the rise of cheap small "Cube Satellites", startups are now able to launch their own satellites for a fraction of the cost of a traditional launch. This has led to a proliferation of small satellites, which are now being used for a wide range of applications. The most common use of small satellites is for communications,']
```

下一节，我们将展示如何用该脚本来执行基准测试，我们还将展示如何将其应用于 Hugging Face Hub 中任何你喜欢的数据集！

### 如何复现这些结果？

[此处](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation) 提供了用于在 Gaudi2 和第一代 Gaudi 上对 BLOOMZ 进行基准测试的脚本。在运行它之前，请确保按照 [Habana 给出的指南](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 安装了最新版本的 SynapseAI 和 Gaudi 驱动程序。

然后，运行以下命令：
```bash
git clone https://github.com/huggingface/optimum-habana.git
cd optimum-habana && pip install . && cd examples/text-generation
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.8.0
```

最后，你可以按如下方式运行脚本：
```bash
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py --model_name_or_path bigscience/bloomz --use_hpu_graphs --use_kv_cache --max_new_tokens 100
```

对于多节点推理，你可以遵循 Optimum Habana 文档中的 [这个指南](https://huggingface.co/docs/optimum/habana/usage_guides/multi_node_training)。

你还可以从 Hugging Face Hub 加载任何数据集作为文本生成任务的提示，只需使用参数`--dataset_name my_dataset_name`。

此基准测试基于 Transformers v4.27.1、SynapseAI v1.8.0，而Optimum Habana 是从源码安装的。

对于 GPU，[此代码库](https://github.com/huggingface/transformers-bloom-inference/tree/main/bloom-inference-scripts) 里包含了[可用于复现这篇文章结果的脚本](https://huggingface.co/blog/bloom-inference-pytorch-scripts)。要使用 CUDA graphs，需要使用静态数据尺寸，而 🤗 Transformers 中不支持这一用法。你可以使用 Habana 团队的 [这份代码](https://github.com/HabanaAI/Model-References/tree/1.8.0/PyTorch/nlp/bloom) 来使能 CUDA graphs 或 HPU graphs。


## 总结

通过本文，我们看到，**Habana Gaudi2 执行 BLOOMZ 推理的速度比 Nvidia A100 80GB 更快**。并且无需编写复杂的脚本，因为 🤗 [Optimum Habana](https://huggingface.co/docs/optimum/habana/index) 提供了易于使用的工具用于在 HPU 上运行数十亿参数模型的推理。Habana 的 SynapseAI SDK 的后续版本有望提高性能，因此随着 SynapseAI 上 LLM 推理优化的不断推进，我们将定期更新此基准。我们也期待 FP8 推理在 Gaudi2 上带来的性能优势。

我们还介绍了在第一代 Gaudi 上的结果。对于更小的模型，它的性能与 A100 比肩，甚至更好，而价格仅为 A100 的近三分之一。对于像 BLOOMZ 这样的大模型，它是替代 GPU 推理的一个不错的选择。

如果你有兴趣使用最新的 AI 硬件加速器和软件库来加速你的机器学习训练和推理工作流，请查看我们的 [专家加速计划](https://huggingface.co/support)。要了解有关 Habana 解决方案的更多信息，可以 [从此处了解我们双方的相关合作并联系他们](https://huggingface.co/hardware/habana)。要详细了解 Hugging Face 为使 AI 硬件加速器易于使用所做的工作，请查看我们的 [硬件合作伙伴计划](https://huggingface.co/hardware)。

### 相关话题

- [更快训推：Habana Gaudi-2 与 Nvidia A100 80GB](https://huggingface.co/blog/habana-gaudi-2-benchmark)
- [在Hugging Face 和 Habana Labs Gaudi 上用 DeepSpeed 训练更快、更便宜的大规模 Transformer 模型](https://developer.habana.ai/events/leverage-deepspeed-to-train-faster-and-cheaper-large-scale-transformer-models-with-hugging-face-and-habana-labs-gaudi/)

---

感谢阅读！如果你有任何问题，请随时通过 [Github](https://github.com/huggingface/optimum-habana) 或 [论坛](https://discuss.huggingface.co/c/optimum/59) 与我联系。你也可以在 [LinkedIn](https://www.linkedin.com/in/regispierrard/) 上找到我。

[^1]：“零样本”是指模型在新的或未见过的输入数据上完成任务的能力，即训练数据中完全不含此类数据。我们输给模型提示和以自然语言描述的指令（即我们希望模型做什么）。零样本分类不提供任何与正在完成的任务相关的任何示例。这区别于单样本或少样本分类，因为这些任务还是需要提供有关当前任务的一个或几个示例的。
