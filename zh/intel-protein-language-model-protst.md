---
title: "在英特尔 Gaudi 2 上加速蛋白质语言模型 ProtST"
thumbnail: /blog/assets/intel-protein-language-model-protst/01.jpeg
authors:
- user: juliensimon
- user: Jiqing
  guest: true
  org: Intel
- user: Santiago Miret
  guest: true
- user: katarinayuan
  guest: true
- user: sywangyi
  guest: true
  org: Intel
- user: MatrixYao
  guest: true
  org: Intel
- user: ChrisAllenMing
  guest: true
- user: kding1
  guest: true
  org: Intel
translators:
- user: MatrixYao
- user: zhongdongy 
  proofreader: false
---

# 在英特尔 Gaudi 2 上加速蛋白质语言模型 ProtST

<p align="center">
 <img src="https://huggingface.co/blog/assets/intel-protein-language-model-protst/01.jpeg" alt="A teenage scientist creating molecules with computers and artificial intelligence" width="512"><br>
</p>

## 引言

蛋白质语言模型 (Protein Language Models, PLM) 已成为蛋白质结构与功能预测及设计的有力工具。在 2023 年国际机器学习会议 (ICML) 上，MILA 和英特尔实验室联合发布了 [ProtST](https://proceedings.mlr.press/v202/xu23t.html) 模型，该模型是个可基于文本提示设计蛋白质的多模态模型。此后，ProtST 在研究界广受好评，不到一年的时间就积累了 40 多次引用，彰显了该工作的影响力。

PLM 最常见的任务之一是预测氨基酸序列的亚细胞位置。此时，用户输入一个氨基酸序列给模型，模型会输出一个标签，以指示该序列所处的亚细胞位置。论文表明，ProtST-ESM-1b 的零样本亚细胞定位性能优于最先进的少样本分类器 (如下图)。

<kbd>
  <img src="https://huggingface.co/blog/assets/intel-protein-language-model-protst/02.png">
</kbd>

为了使 ProtST 更民主化，英特尔和 MILA 对模型进行了重写，以使大家可以通过 Hugging Face Hub 来使用模型。大家可于 [此处](https://huggingface.co/mila-intel) 下载模型及数据集。

本文将展示如何使用英特尔 Gaudi 2 加速卡及 `optimum-habana` 开源库高效运行 ProtST 推理和微调。[英特尔 Gaudi 2](https://habana.ai/products/gaudi2/) 是英特尔设计的第二代 AI 加速卡。感兴趣的读者可参阅我们 [之前的博文](https://huggingface.co/blog/zh/habana-gaudi-2-bloom#habana-gaudi2)，以深入了解该加速卡以及如何通过 [英特尔开发者云](https://cloud.intel.com) 使用它。得益于 [`optimum-habana`](https://github.com/huggingface/optimum-habana)，仅需少量的代码更改，用户即可将基于 transformers 的代码移植至 Gaudi 2。

## 对 ProtST 进行推理

常见的亚细胞位置包括细胞核、细胞膜、细胞质、线粒体等，你可从 [此数据集](https://huggingface.co/datasets/mila-intel/subloc_template) 中获取全面详细的位置介绍。

我们使用 `ProtST-SubcellularLocalization` 数据集的测试子集来比较 ProtST 在英伟达 `A100 80GB PCIe` 和 `Gaudi 2` 两种加速卡上的推理性能。该测试集包含 2772 个氨基酸序列，序列长度范围为 79 至 1999。

你可以使用 [此脚本](https://github.com/huggingface/optimum-habana/tree/main/examples/protein-folding#single-hpu-inference-for-zero-shot-evaluation) 重现我们的实验，我们以 `bfloat16` 精度和 batch size 1 运行模型。在英伟达 A100 和英特尔 Gaudi 2 上，我们获得了相同的准确率 (0.44)，但 Gaudi 2 的推理速度比 A100 快 1.76 倍。单张 A100 和单张 Gaudi 2 的运行时间如下图所示。

<kbd>
  <img src="https://huggingface.co/blog/assets/intel-protein-language-model-protst/03.png">
</kbd>

## 微调 ProtST

针对下游任务对 ProtST 模型进行微调是提高模型准确性的简单且公认的方法。在本实验中，我们专门研究了针对二元定位任务的微调，其是亚细胞定位的简单版，任务用二元标签指示蛋白质是膜结合的还是可溶的。

你可使用 [此脚本](https://github.com/huggingface/optimum-habana/tree/main/examples/protein-folding#multi-hpu-finetune-for-sequence-classification-task) 重现我们的实验。其中，我们在 [ProtST-BinaryLocalization](https://huggingface.co/datasets/mila-intel/ProtST-BinaryLocalization) 数据集上以 `bfloat16` 精度微调 [ProtST-ESM1b-for-sequential-classification](https://huggingface.co/mila-intel/protst-esm1b-for-sequential-classification)。下表展示了不同硬件配置下测试子集的模型准确率，可以发现它们均与论文中发布的准确率 (~92.5%) 相当。

<kbd>
  <img src="https://huggingface.co/blog/assets/intel-protein-language-model-protst/04.png">
</kbd>

下图显示了微调所用的时间。可以看到，单张 Gaudi 2 比单张 A100 快 2.92 倍。该图还表明，在 4 张或 8 张 Gaudi 2 加速卡上使用分布式训练可以实现近线性扩展。

<kbd>
  <img src="https://huggingface.co/blog/assets/intel-protein-language-model-protst/05.png">
</kbd>

## 总结

本文，我们展示了如何基于 `optimum-habana` 轻松在 Gaudi 2 上部署 ProtST 推理和微调。此外，我们的结果还表明，与 A100 相比，Gaudi 2 在这些任务上的性能颇具竞争力: 推理速度提高了 1.76 倍，微调速度提高了 2.92 倍。

如你你想在英特尔 Gaudi 2 加速卡上开始一段模型之旅，以下资源可助你一臂之力:

- optimum-habana [代码库](https://github.com/huggingface/optimum-habana)
- 英特尔 Gaudi [文档](https://docs.habana.ai/en/latest/index.html)

感谢垂阅！我们期待看到英特尔 Gaudi 2 加速的 ProtST 能助你创新。