---
title: "LAVE：使用 LLM 对 Docmatix 进行零样本 VQA 评估 - 我们还需要微调吗？" 
thumbnail: /blog/assets/184_zero_shot_docmatix/thumb.001.jpeg
authors:
- user: danaaubakirova
- user: andito 
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# LAVE: 使用 LLM 对 Docmatix 进行零样本 VQA 评估 - 我们还需要微调吗？

在开发 Docmatix 时，我们发现经其微调的 Florence-2 在 DocVQA 任务上表现出色，但在基准测试中得分仍比较低。为了提高基准测试得分，我们必须在 DocVQA 数据集上进一步对模型进行微调，以学习该基准测试的语法风格。有意思的是，人类评估者认为经额外微调后，模型的表现似乎反而不如仅在 Docmatix 上微调那么好，因此我们最后决定仅将额外微调后的模型用于消融实验，而公开发布的还是仅在 Docmatix 上微调的模型。

尽管模型生成的答案在语义上与参考答案一致 (如图 1 所示)，但基准测试的得分却较低。这就引出了一个问题: 我们应该微调模型以改进在既有指标上的表现，还是应该开发与人类感知更相符的新指标？

<div align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/RaQZkkcnTAcS80pPyt55J.png" alt="VQA 评估 " style="width: 55%; border: none;">
</div>
<p align="center">
  <em>图 1: Docmatix 数据集微调模型零样本生成的答案与参考答案之间的 t-SNE 图</em>
</p>

## 背景

社区最近很关注分布外 (out-of-distribution，OOD) 评估，即利用诸如零样本之类的方法将模型的能力迁移至未见过的 VQA 任务抑或是对一个 VQA 数据集进行微调并在另一个 VQA 数据集上进行评估。这一转变与用于微调视觉语言模型 (VLM) 的合成数据集 (例如 Docmatix、SciGraphQA、SimVQA) 的日渐兴起紧密相关。

一直以来，VQA 准确度一直是评估模型性能的主要指标，其方法是计算模型预测答案与人工标注的一组参考答案之间的精确字符串匹配率。因为传统的 VQA 评估遵循独立同分布 (independent and identically distributed，IID) 范式，其训练数据和测试数据分布相似，而传统的模型训练是遵循此假设的，所以此时该指标的效果很好，详情请参阅 [此处](https://arxiv.org/pdf/2205.12191)。

但在 OOD 场景下，由于格式、专业度以及表达等方面的差异，生成的答案尽管正确，但可能与参考答案不尽匹配。图 1 完美地展示了这种情况，图中我们将零样本生成的文本描述与合成数据集中的参考文本描述进行了比较。指令生成的数据集与人工标注的数据集之间的差异尤甚。目前已有一些 [方法](https://proceedings.mlr.press/v202/li23q.html) 试图将生成的答案格式对齐至参考答案格式，但这只是治标之策，并未改变评估指标有缺陷的根本症结。虽然也可以采用人工评估的方式，结果会比较可靠，但其成本高昂且不可扩展，所以当务之急还是设计与人类判断更相符的新指标。

## 方法

[Docmatix](https://huggingface.co/blog/docmatix) 是当前最大的 DocVQA 合成数据集，它是基于精选文档数据集 [PDFA](https://huggingface.co/datasets/pixparse/pdfa-eng-wds) 生成的。它比之前市面上的数据集大 100 倍。其对标的是人工标注数据集 DocVQA，DocVQA 目前被普遍用作文档理解类 VQA 模型的评估基准。本文中，我们使用的是 **Docmatix 的子集**，它包含大约 200 个测试样本，你可于此处下载 [Docmatix-zero-shot-exp](https://huggingface.co/datasets/HuggingFaceM4/Docmatix/viewer/zero-shot-exp)。

<div style="display: flex; justify-content: center; align-items: center; gap: 0px; width: 100%; margin: 0 auto;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/feXi3iSLo8hBXTh2y8NnR.png" alt="Image 1" style="width: 45%; height: auto; object-fit: cover;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/2X4KdrTi6M8VYU6hOdmk1.png" alt="Image 2" style="width: 45%; height: auto; object-fit: cover;">
</div>
<p align="center">
  <em>图 2: 来自 Docmatix 和 DocVQA 测试集的问答对示例。注: 此处未显示相应的图像。</em>
</p>

尽管 Docmatix 和 DocVQA 中问答对的内容相似，但它们的风格却有着显著差异。此时，CIDER、ANLS 以及 BLEU 等传统指标对于零样本评估而言可能过于严格。鉴于从 t-SNE 中观察到的嵌入的相似性 (图 1)，我们决定使用一个不同于以往的新评估指标: LAVE (LLM-Assisted VQA Evaluation，LLM 辅助 VQA 评估)，以期更好地评估模型在未见但语义相似的数据集上的泛化能力。

<div style="display: flex; justify-content: center; align-items: center; gap: 10px; width: 100%; margin: 0 auto;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/C4twDu9D6cw0XHdA57Spe.png" alt="Image 1" style="width: 30%; height: auto; object-fit: cover;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/pYsiOyToOXzRitmRidejW.png" alt="Image 2" style="width: 30%; height: auto; object-fit: cover;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/uM6IPAAvjyiYTPJXdB10w.png" alt="Image 3" style="width: 30%; height: auto; object-fit: cover;">
</div>
<p align="center">
  <em>图 3: Docmatix 和 DocVQA 数据集中的问题、答案以及图像特征的 t-SNE 图</em>
</p>

评估时，我们选择 [MPLUGDocOwl1.5](https://arxiv.org/pdf/2403.12895) 作为基线模型。该模型在原始 DocVQA 数据集的测试子集上 ANLS 得分为 84%。然后，我们在 Docmatix 的一个子集 (含 200 张图像) 上运行零样本生成。我们使用 [Llama-2-Chat-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 对答案进行评分。

## 关于 LAVE

我们遵循 [本篇论文](https://arxiv.org/html/2310.02567v2) 中的步骤，将 VQA 评估设计为一个基于 LLM 上下文学习的答案评分任务。我们将分数设在 1 到 3 之间并考虑了问题不明晰或答案不完整的情况。LLM 的提示词包含任务描述、几个输入/输出演示以及待测样本的输入。

我们撰写了任务描述并在其后加上了指令 **“在评分之前给出理由”** 以要求 LLM 给出评分理由。每个演示都包含一个问题、一组参考答案、候选答案、答案得分及其理由。在提示中，我们还要求 **“仅提供一个评分”** 以避免因逐句分析带来的多个评分。

```py
task_description = """You are given a question, a set of gold-standard reference answers written by
experts, and a candidate answer. Please rate the accuracy of the candidate answer for the question
considering the reference answers. Use a scale of 1-3, with 1 indicating an incorrect or irrelevant
answer, 2 indicating an ambiguous or incomplete answer, and 3 indicating a correct answer.
Give the rationale before rating. Provide only one rating.
THIS IS VERY IMPORTANT:
A binary question should only be answered with 'yes' or 'no',
otherwise the candidate answer is incorrect."""

demonstrations = [
    {
        "question": "What's the weather like?",
        "reference_answer": ["sunny", "clear", "bright", "sunny", "sunny"],
        "generated_answer": "cloudy"
    }
]
```

#### 评分函数

给定 LLM 为测试样本生成的提示，我们从最后一个字符 (为 1、2 或 3) 中提取评分，并将其缩放至 `[0, 1]` 范围内: $s = \frac{r - 1}{2}$，以获取最终评分。

#### 结果

各指标得分如下:

<table style="border-collapse: collapse; width: 50%; margin: auto;">
  <tr>
    <th style="border: 1px solid black; padding: 8px; text-align: center;"> 指标 </th>
    <th style="border: 1px solid black; padding: 8px; text-align: center;">CIDER</th>
    <th style="border: 1px solid black; padding: 8px; text-align: center;">BLEU</th>
    <th style="border: 1px solid black; padding: 8px; text-align: center;">ANLS</th>
    <th style="border: 1px solid black; padding: 8px; text-align: center;">LAVE</th>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px; text-align: center;"> 得分 </td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">0.1411</td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">0.0032</td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">0.002</td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">0.58</td>
  </tr>
</table>

## 几个生成案例

<div align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/5ljrlVqrHHB4VGRek7hJv.png" alt="VQA Evaluation" style="width:120%, border: none;">
</div>
<p align="center">
  <em>图 4: Docmatix 测试子集中的一个问题、参考答案、模型生成的答案以及 Llama 给出的评分及理由。</em>
</p>

<div align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/scly6WR_2Wvrk5qd05cx4.png" alt="VQA Evaluation" style="width:120%, border: none;">
</div>
<p align="center">
  <em>图 5: Docmatix 测试子集中的一个问题、参考答案、模型生成的答案以及 Llama 给出的评分及理由。</em>
</p>

## 现有的 VQA 系统评估标准是否过于僵化了？我们还需要微调吗？

当使用 LLM 来评估答案时，我们答案的准确率提高了大约 50%，这表明虽未遵循严格的格式，答案也可能是正确的。这表明我们目前的评估指标可能过于僵化。值得注意的是，本文并不是一篇全面的研究论文，因此需要更多的消融实验来充分了解不同指标对合成数据集零样本性能评估的有效性。我们希望社区能够以我们的工作为起点，继续深化拓展下去，从而改进合成数据集背景下的零样本视觉语言模型评估工作，并探索能够超越提示学习的其它更有效的方法。

## 参考文献

```
@inproceedings{cascante2022simvqa,
  title={Simvqa: Exploring simulated environments for visual question answering},
  author={Cascante-Bonilla, Paola and Wu, Hui and Wang, Letao and Feris, Rogerio S and Ordonez, Vicente},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5056--5066},
  year={2022}
}

@article{hu2024mplug,
  title={mplug-docowl 1.5: Unified structure learning for ocr-free document understanding},
  author={Hu, Anwen and Xu, Haiyang and Ye, Jiabo and Yan, Ming and Zhang, Liang and Zhang, Bo and Li, Chen and Zhang, Ji and Jin, Qin and Huang, Fei and others},
  journal={arXiv preprint arXiv:2403.12895},
  year={2024}
}

@article{agrawal2022reassessing,
  title={Reassessing evaluation practices in visual question answering: A case study on out-of-distribution generalization},
  author={Agrawal, Aishwarya and Kaji{\'c}, Ivana and Bugliarello, Emanuele and Davoodi, Elnaz and Gergely, Anita and Blunsom, Phil and Nematzadeh, Aida},
  journal={arXiv preprint arXiv:2205.12191},
  year={2022}
}

@inproceedings{li2023blip,
  title={Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models},
  author={Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven},
  booktitle={International conference on machine learning},
  pages={19730--19742},
  year={2023},
  organization={PMLR}
}
@inproceedings{manas2024improving,
  title={Improving automatic vqa evaluation using large language models},
  author={Ma{\~n}as, Oscar and Krojer, Benno and Agrawal, Aishwarya},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4171--4179},
  year={2024}
}

@article{li2023scigraphqa,
  title={Scigraphqa: A large-scale synthetic multi-turn question-answering dataset for scientific graphs},
  author={Li, Shengzhi and Tajbakhsh, Nima},
  journal={arXiv preprint arXiv:2308.03349},
  year={2023}
}
```