---
title: "在 Hub 上使用 Presidio 进行自动 PII 检测实验"
thumbnail: /blog/assets/presidio-pii-detection/thumbnail.png
authors:
- user: lhoestq
- user: meg
- user: presidio
- user: omri374
translator:
- user: Evinci
---

# 在 Hub 上使用 Presidio 进行自动 PII 检测实验

我们在 Hugging Face Hub 上托管的机器学习（ML）数据集中发现了一个引人关注的现象：包含个人未经记录的私密信息。这一现象为机器学习从业者带来了一些特殊挑战。 

在本篇博客中，我们将深入探讨含有一种称为个人识别信息（PII）的私密信息的各类数据集，分析这些数据集存在的问题，并介绍我们在数据集 Hub 上正在测试的一个新功能，旨在帮助应对这些挑战。

## 包含个人识别信息（PII）的数据集类型

我们注意到包含个人识别信息（PII）的数据集主要有两种类型：

1. **标注的PII数据集**：例如由 Ai4Privacy 提供的 [PII-Masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)，这类数据集专门用于训练PII检测模型。这些模型用于检测和掩盖PII，可以帮助进行在线内容审核或提供匿名化的数据库。
2. **预训练数据集**：这些通常是大规模的数据集，往往有数TB大小，通常通过网络爬虫获得。尽管这些数据集一般会过滤掉某些类型的PII，但由于数据量庞大和PII检测模型的不完善，仍可能有少量敏感信息遗漏。

## 机器学习数据集中的个人识别信息（PII）面临的挑战

机器学习数据集中存在的个人识别信息（PII）会为从业者带来几个挑战。首先，它引发了隐私问题，可能被用来推断个人的敏感信息。

此外，如果未能妥善处理PII，它还可能影响机器学习模型的性能。例如，如果一个模型是在包含PII的数据集上训练的，它可能学会将特定的PII与特定的结果关联起来，这可能导致预测偏见或从训练集生成PII。

## 数据集 Hub 上的新实验：Presidio 报告

为了应对这些挑战，我们正在数据集 Hub 上试验一项新功能，使用 [Presidio](https://github.com/microsoft/presidio)——一种开源的最先进的个人识别信息（PII）检测工具。Presidio 依赖检测模式和机器学习模型来识别 PII。

通过这个新功能，用户将能够看到一个报告，估计数据集中PII的存在情况。这一信息对于机器学习从业者非常有价值，帮助他们在训练模型前做出明智的决策。例如，如果报告指出数据集包含敏感的 PII，从业者可能会选择使用像 Presidio 这样的工具进一步过滤数据集。

数据集所有者也可以通过使用这些报告来验证他们的 PII 过滤流程，从而在发布数据集之前受益于这一功能。

## Presidio 报告的一个示例

让我们来看一个关于这个[预训练数据集](https://huggingface.co/datasets/allenai/c4)的 Presidio 报告的示例：

![Presidio report](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/presidio-pii-detection/presidio_report.png)

在这个例子中，Presidio 检测到数据集中有少量的电子邮件和敏感个人识别信息（PII）。

## 结论

机器学习数据集中个人识别信息（PII）的存在是机器学习社区不断发展的挑战之一。 在 Hugging Face，我们致力于保持透明度，并帮助从业者应对这些挑战。 通过在数据集 Hub 上试验诸如 Presidio 报告之类的新功能，我们希望赋予用户做出明智决策的能力，并构建更健壯、更符合道德标准的机器学习模型。

我们还要感谢国家信息与自由委员会（CNIL）对 [GDPR 合规性的帮助](https://huggingface.co/blog/cnil)。 他们在指导我们应对人工智能和个人数据问题的复杂性方面提供了宝贵的帮助。 请在[这里](https://www.cnil.fr/fr/ai-how-to-sheets)查看他们更新的人工智能操作指南。

敬请期待更多关于这一激动人心发展的更新！
