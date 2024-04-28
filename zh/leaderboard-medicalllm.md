---
title: "开源医疗大模型排行榜：健康领域大模型基准测试"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_medicalllm.png
authors:
- user: aaditya
  guest: true
- user: pminervini
  guest: true
- user: clefourrier
translators:
- user: innovation64
- user: zhongdongy
  proofreader: true
---

# 开源医疗大模型排行榜: 健康领域大模型基准测试

![Image source: https://arxiv.org/pdf/2311.05112.pdf](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/medical_llms.png?raw=true)

多年来，大型语言模型 (LLMs) 已经发展成为一项具有巨大潜力，能够彻底改变医疗行业各个方面的开创性技术。这些模型，如 [GPT-3](https://arxiv.org/abs/2005.14165)，[GPT-4](https://arxiv.org/abs/2303.08774) 和 [Med-PaLM 2](https://arxiv.org/abs/2305.09617)，在理解和生成类人文本方面表现出了卓越的能力，使它们成为处理复杂医疗任务和改善病人护理的宝贵工具。它们在多种医疗应用中显示出巨大的前景，如医疗问答 (QA) 、对话系统和文本生成。此外，随着电子健康记录 (EHRs) 、医学文献和病人生成数据的指数级增长，LLMs 可以帮助医疗专业人员提取宝贵见解并做出明智的决策。

然而，尽管大型语言模型 (LLMs) 在医疗领域具有巨大的潜力，但仍存在一些重要且具体的挑战需要解决。

当模型用于娱乐对话方面时，错误的影响很小; 然而，在医疗领域使用时，情况并非如此，错误的解释和答案可能会对病人的护理和结果产生严重后果。语言模型提供的信息的准确性和可靠性可能是生死攸关的问题，因为它可能影响医疗决策、诊断和治疗计划。

例如，当有人问 GPT-3 关于孕妇可以用什么药的问题时，GPT-3 错误地建议使用四环素，尽管它也正确地说明了四环素对胎儿有害，孕妇不应该用。如果真按照这个错误的建议去给孕妇用药，可能会害得孩子将来骨头长不好。

![Image source: [https://arxiv.org/pdf/2311.05112.pdf](https://arxiv.org/abs/2307.15343)](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/gpt_medicaltest.png?raw=true)

要想在医疗领域用好这种大型语言模型，就得根据医疗行业的特点来设计和基准测试这些模型。因为医疗数据和应用有其特殊的地方，得考虑到这些。而且，开发方法来评估这些用于医疗的模型不只是为了研究，而是因为它们在现实医疗工作中用错了可能会带来风险，所以这事儿实际上很重要。

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.20.1/gradio.js"> </script>
<gradio-app theme_mode="light" space="openlifescienceai/open_medical_llm_leaderboard"></gradio-app>

开源医疗大模型排行榜旨在通过提供一个标准化的平台来评估和比较各种大型语言模型在多种医疗任务和数据集上的性能，以此来解决这些挑战和限制。通过提供对每个模型的医疗知识和问答能力的全面评估，该排行榜促进了更有效、更可靠的医疗大模型的发展。

这个平台使研究人员和从业者能够识别不同方法的优势和不足，推动该领域的进一步发展，并最终有助于改善患者的治疗结果。

## 数据集、任务和评估设置

医疗大模型排行榜包含多种任务，并使用准确度作为其主要评估指标 (准确度衡量的是语言模型在各个医疗问答数据集中提供的正确答案的百分比)。

### MedQA

[MedQA](https://arxiv.org/abs/2009.13081) 数据集包含来自美国医学执照考试 (USMLE) 的多项选择题。它覆盖了广泛的医学知识，并包括 11,450 个训练集问题和 1,273 个测试集问题。每个问题有 4 或 5 个答案选项，该数据集旨在评估在美国获得医学执照所需的医学知识和推理技能。

![MedQA 问题](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/medqa.png?raw=true)

### MedMCQA

[MedMCQA](https://proceedings.mlr.press/v174/pal22a.html) 是一个大规模的多项选择问答数据集，来源于印度的医学入学考试 (AIIMS/NEET)。它涵盖了 2400 个医疗领域主题和 21 个医学科目，训练集中有超过 187,000 个问题，测试集中有 6,100 个问题。每个问题有 4 个答案选项，并附有解释。MedMCQA 评估模型的通用医学知识和推理能力。

![MedMCQA 问题](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/medmcqa.png?raw=true)

### PubMedQA

[PubMedQA](https://aclanthology.org/D19-1259/) 是一个封闭领域的问答数据集，每个问题都可以通过查看相关上下文 ( PubMed 摘要) 来回答。它包含 1,000 个专家标注的问题 - 答案对。每个问题都附有 PubMed 摘要作为上下文，任务是提供基于摘要信息的是/否/也许答案。该数据集分为 500 个训练问题和 500 个测试问题。PubMedQA 评估模型理解和推理科学生物医学文献的能力。

![PubMedQA 问题](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/pubmedqa.png?raw=true)

### MMLU 子集 (医学和生物学)

[MMLU 基准](https://arxiv.org/abs/2009.03300) (测量大规模多任务语言理解) 包含来自各个领域多项选择题。对于开源医疗大模型排行榜，我们关注与医学知识最相关的子集:

- 临床知识: 265 个问题，评估临床知识和决策技能。
- 医学遗传学: 100 个问题，涵盖医学遗传学相关主题。
- 解剖学: 135 个问题，评估人体解剖学知识。
- 专业医学: 272 个问题，评估医疗专业人员所需的知识。
- 大学生物学: 144 个问题，涵盖大学水平的生物学概念。
- 大学医学: 173 个问题，评估大学水平的医学知识。
每个 MMLU 子集都包含有 4 个答案选项的多项选择题，旨在评估模型对特定医学和生物领域理解。

![MMLU 问题](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/mmlu.png?raw=true)

开源医疗大模型排行榜提供了一个鲁棒的评估，衡量模型在医学知识和推理各方面的表现。

## 洞察与分析

开源医疗大模型排行榜评估了各种大型语言模型 (LLMs) 在一系列医疗问答任务上的表现。以下是我们的一些关键发现:

- 商业模型如 GPT-4-base 和 Med-PaLM-2 在各个医疗数据集上始终获得高准确度分数，展现了在不同医疗领域中的强劲性能。
- 开源模型，如 [Starling-LM-7B](https://huggingface.co/Nexusflow/Starling-LM-7B-beta)，[gemma-7b](https://huggingface.co/google/gemma-7b)，Mistral-7B-v0.1 和 [Hermes-2-Pro-Mistral-7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B)，尽管参数量大约只有 70 亿，但在某些数据集和任务上展现出了有竞争力的性能。
- 商业和开源模型在理解和推理科学生物医学文献 (PubMedQA) 以及应用临床知识和决策技能 (MMLU 临床知识子集) 等任务上表现良好。

![图片来源: [https://arxiv.org/abs/2402.07023](https://arxiv.org/abs/2402.07023)](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/model_evals.png?raw=true)

谷歌的模型 [Gemini Pro](https://arxiv.org/abs/2312.11805) 在多个医疗领域展现了强大的性能，特别是在生物统计学、细胞生物学和妇产科等数据密集型和程序性任务中表现尤为出色。然而，它在解剖学、心脏病学和皮肤病学等关键领域表现出中等至较低的性能，揭示了需要进一步改进以应用于更全面的医学的差距。

![Image source : [https://arxiv.org/abs/2402.07023](https://arxiv.org/abs/2402.07023)](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/subjectwise_eval.png?raw=true)

## 提交你的模型以供评估

要在开源医疗大模型排行榜上提交你的模型进行评估，请按照以下步骤操作:

**1. 将模型权重转换为 Safetensors 格式**

首先，将你的模型权重转换为 safetensors 格式。Safetensors 是一种新的存储权重的格式，加载和使用起来更安全、更快。将你的模型转换为这种格式还将允许排行榜在主表中显示你模型的参数数量。

**2. 确保与 AutoClasses 兼容**

在提交模型之前，请确保你可以使用 Transformers 库中的 AutoClasses 加载模型和分词器。使用以下代码片段来测试兼容性:

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained(MODEL_HUB_ID)
model = AutoModel.from_pretrained("your model name")
tokenizer = AutoTokenizer.from_pretrained("your model name")

```

如果在这一步失败，请根据错误消息在提交之前调试你的模型。很可能你的模型上传不当。

**3. 将你的模型公开**

确保你的模型可以公开访问。排行榜无法评估私有模型或需要特殊访问权限的模型。

**4. 远程代码执行 (即将推出)**

目前，开源医疗大模型排行榜不支持需要 `use_remote_code=True` 的模型。然而，排行榜团队正在积极添加这个功能，敬请期待更新。

**5. 通过排行榜网站提交你的模型**

一旦你的模型转换为 safetensors 格式，与 AutoClasses 兼容，并且可以公开访问，你就可以使用开源医疗大模型排行榜网站上的 “在此提交！” 面板进行评估。填写所需信息，如模型名称、描述和任何附加细节，然后点击提交按钮。
排行榜团队将处理你的提交并评估你的模型在各个医疗问答数据集上的表现。评估完成后，你的模型的分数将被添加到排行榜中，你可以将它的性能与其他模型进行比较。

## 下一步是什么？扩展开源医疗大模型排行榜

开源医疗大模型排行榜致力于扩展和适应，以满足研究社区和医疗行业不断变化的需求。重点领域包括:

1. 通过与研究人员、医疗组织和行业合作伙伴的合作，纳入更广泛的医疗数据集，涵盖医疗的各个方面，如放射学、病理学和基因组学。
2. 通过探索准确性以外的其他性能衡量标准，如点对点得分和捕捉医疗应用独特需求的领域特定指标，来增强评估指标和报告能力。
3. 在这个方向上已经有一些工作正在进行中。如果你有兴趣合作我们计划提出的下一个基准，请加入我们的 [ Discord 社区](https://discord.gg/A5Fjf5zC69) 了解更多并参与其中。我们很乐意合作并进行头脑风暴！

如果你对 AI 和医疗的交叉领域充满热情，为医疗领域构建模型，并且关心医疗大模型的安全和幻觉问题，我们邀请你加入我们在 [Discord 上的活跃社区](https://discord.gg/A5Fjf5zC69)。

## 致谢

![致谢](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/credits.png?raw=true)

特别感谢所有帮助实现这一目标的人，包括 Clémentine Fourrier 和 Hugging Face 团队。我要感谢 Andreas Motzfeldt、Aryo Gema 和 Logesh Kumar Umapathi 在排行榜开发过程中提供的讨论和反馈。衷心感谢爱丁堡大学的 Pasquale Minervini 教授提供的时间、技术协助和 GPU 支持。

## 关于开放生命科学 AI

开放生命科学 AI 是一个旨在彻底改变人工智能在生命科学和医疗领域应用的项目。它作为一个中心枢纽，列出了医疗模型、数据集、基准测试和跟踪会议截止日期，促进在 AI 辅助医疗领域的合作、创新和进步。我们努力将开放生命科学 AI 建立为对 AI 和医疗交叉领域感兴趣的任何人的首选目的地。我们为研究人员、临床医生、政策制定者和行业专家提供了一个平台，以便进行对话、分享见解和探索该领域的最新发展。

![OLSA logo](https://github.com/monk1337/research_assets/blob/main/huggingface_blog/olsa.png?raw=true)

## 引用

如果你觉得我们的评估有用，请考虑引用我们的工作

**医疗大模型排行榜**

```
@misc{Medical-LLM Leaderboard,
author = {Ankit Pal, Pasquale Minervini, Andreas Geert Motzfeldt, Aryo Pradipta Gema and Beatrice Alex},
title = {openlifescienceai/open_medical_llm_leaderboard},
year = {2024},
publisher = {Hugging Face},
howpublished = "\url{https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard}"
}
```