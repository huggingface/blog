---
title: "StarCoder：最先进的代码大模型" 
thumbnail: /blog/assets/141_starcoder/starcoder_thumbnail.png
authors:
- user: lvwerra
- user: loubnabnl
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# StarCoder: 最先进的代码大模型


## 关于 BigCode

BigCode 是由 Hugging Face 和 ServiceNow 共同领导的开放式科学合作项目，该项目致力于开发负责任的代码大模型。

## StarCoder 简介

StarCoder 和 StarCoderBase 是针对代码的大语言模型 (代码 LLM)，模型基于 GitHub 上的许可数据训练而得，训练数据中包括 80 多种编程语言、Git 提交、GitHub 问题和 Jupyter notebook。与 LLaMA 类似，我们基于 1 万亿个词元训练了一个约 15B 参数的模型。此外，我们还针对一个 35B 词元的 Python 数据集对 StarCoderBase 模型进行了微调，从而获得了一个我们称之为 StarCoder 的新模型。

我们发现 StarCoderBase 在流行的编程基准测试中表现优于现有其他开源的代码 LLM，同时与闭源模型相比，如来自 OpenAI 的 `code-cushman-001` (早期版本的 GitHub Copilot 背后的原始 Codex 模型)，其表现也相当甚至超过了闭源模型的表现。凭借超过 8,000 个词元的上下文长度，StarCoder 模型可以处理比任何其他开源 LLM 更多的输入，从而可以赋能更广泛的有趣应用。例如，通过用多轮对话来提示 StarCoder 模型，我们可以让它们充当我们的技术助理。此外，这些模型还可用于自动补全代码、根据指令修改代码以及用自然语言解释代码片段等任务。

为了实现开源模型的安全发布，我们采取了一系列的措施，包括改进了 PII (Personally Identifiable Information，个人身份信息) 编辑流水线、对归因跟踪工具进行了创新，并使用改进的 OpenRAIL 许可证发布 StarCoder。更新后的许可证简化了公司将模型集成到其产品中所需的流程。我们相信，凭借其强大的性能，StarCoder 模型将赋能社区将其应用或适配至广泛的应用场景和产品中。

## 评估

我们在不同的测试基准上对 StarCoder 及其他几个与其类似的模型进行了深入的评估。其中之一测试基准是 HumanEval，这是一个比较流行的 Python 基准测试，它主要测试模型是否可以根据函数的签名和文档来编写函数。我们发现 StarCoder 和 StarCoderBase 在 HumanEval 上的表现均优于最大的模型，包括 PaLM、LaMDA 和 LLaMA，尽管它们尺寸要小得多。同时，它们的性能还优于 CodeGen-16B-Mono 和 OpenAI 的 code-cushman-001 (12B) 模型。我们还注意到该模型会生成 `#Solution here` 这样的注释代码，这可能是因为此类代码通常是训练数据中代码习题的一部分。为了强制模型生成一个实际的解决方案，我们添加了提示词 `<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise`。这使得 StarCoder 的 HumanEval 分数有了显著提高，从 34% 提升到 40% 以上，刷新了开源模型的最佳结果的记录。我们也在 CodeGen 和 StarCoderBase 上尝试了此提示词，但结果没有太大差异。

| **模型**          | **HumanEval** | **MBPP** |
|--------------------|--------------|----------|
| LLaMA-7B           | 10.5         | 17.7     |
| LaMDA-137B         | 14.0         | 14.8     |
| LLaMA-13B          | 15.8         | 22.0     |
| CodeGen-16B-Multi  | 18.3         | 20.9     |
| LLaMA-33B          | 21.7         | 30.2     |
| CodeGeeX           | 22.9         | 24.4     |
| LLaMA-65B          | 23.7         | 37.7     |
| PaLM-540B          | 26.2         | 36.8     |
| CodeGen-16B-Mono   | 29.3         | 35.3     |
| StarCoderBase      | 30.4         | 49.0     |
| code-cushman-001   | 33.5         | 45.9     |
| StarCoder          | 33.6         | **52.7** |
| StarCoder-Prompted | **40.8**     | 49.5     |

StarCoder 的一个有趣方面是它是多语言的，因此我们在 MultiPL-E 上对其进行了评估，MultiPL-E 是 HumanEval 的多语言扩展版。我们观察到 StarCoder 在许多编程语言上与 `code-cushman-001` 的表现相当甚至更优。在 DS-1000 数据科学基准测试中，它以明显优势击败了 `code-cushman-001` 以及所有其他开源模型。好了，我们来看看除了代码补全之外，StarCoder 还能做些什么！

## 技术助理

经过详尽的评估，我们已经知道 StarCoder 非常擅长编写代码。我们还想测试它是否可以用作技术助理，毕竟它的训练数据中有大量的文档和 GitHub 问题。受 Anthropic 的 [HHH 提示](https://gist.github.com/jareddk/2509330f8ef3d787fc5aaac67aab5f11#file-hhh_prompt-txt) 的启发，我们构建了一个 [技术助理提示](https://huggingface.co/datasets/bigcode/ta-prompt)。令人惊喜的是，仅凭提示，该模型就能够充当技术助理并回答与编程相关的问题！

![技术助理示例](https://huggingface.co/datasets/bigcode/admin/resolve/main/StarCoderChatExamples.png)

## 训练数据

该模型是在 The Stack 1.2 的一个子集上训练的。该数据集仅包含许可代码，它还包含一个退出流程，以便代码贡献者可以从数据集中删除他们的数据 (请参见 [Am I in The Stack](https://huggingface.co/spaces/bigcode/in-the-stack))。此外，我们从训练数据中删除了个人身份信息，例如姓名、密码和电子邮件地址。

## 我们还发布了……

除了模型，我们还发布了一系列其他资源和应用演示:

- 模型权重，包括具有 OpenRAIL 许可证的 checkpoints
- 所有数据预处理和训练代码，许可证为 Apache 2.0
- 对模型进行全面评估的工具
- 用于训练的删除掉 PII 信息的新数据集，以及用于评估 PII 信息删除效果的代码
- 用于训练的预处理过的数据集
- 用于在数据集中查找生成代码出处的代码归因工具

## 链接

### 模型

- [论文](https://drive.google.com/file/d/1cN-b9GnWtHzQRoE7M7gAEyivY0kl4BYs/view): 关于 StarCoder 的技术报告。
- [GitHub](https://github.com/bigcode-project/starcoder/tree/main): 你可以由此获得有关如何使用或微调 StarCoder 的所有信息。
- [StarCoder](https://huggingface.co/bigcode/starcoder): 基于 Python 数据集进一步微调 StarCoderBase 所得的模型。
- [StarCoderBase](https://huggingface.co/bigcode/starcoderbase): 基于来自 The Stack 数据集的 80 多种编程语言训练而得的模型。
- [StarEncoder](https://huggingface.co/bigcode/starencoder): 在 The Stack 上训练的编码器模型。
- [StarPii](https://huggingface.co/bigcode/starpii): 基于 StarEncoder 的 PII 检测器。

### 工具和应用演示
- [StarCoder Chat](https://huggingface.co/chat?model=bigcode/starcoder): 和 StarCoder 聊天！
- [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode): 使用 StarCoder 补全代码的 VSCode 插件！
- [StarCoder Playground](https://huggingface.co/spaces/bigcode/bigcode-playground): 用 StarCoder 写代码！
- [StarCoder Editor](https://huggingface.co/spaces/bigcode/bigcode-editor): 用 StarCoder 编辑代码！

### 数据与治理

- [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata): StarCoder 的预训练数据集。
- [Tech Assistant Prompt](https://huggingface.co/datasets/bigcode/ta-prompt): 使用该提示，你可以将 StarCoder 变成技术助理。
- [Governance Card](): 有关模型治理的卡片。
- [StarCoder License Agreement](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement): 该模型基于 BigCode OpenRAIL-M v1 许可协议。
- [StarCoder Search](https://huggingface.co/spaces/bigcode/search): 对预训练数据集中的代码进行全文搜索。
- [StarCoder Membership Test](https://stack.dataportraits.org): 快速测试某代码是否存在于预训练数据集中。

你可以在 [huggingface.co/bigcode](https://huggingface.co/bigcode) 找到所有资源和链接！
