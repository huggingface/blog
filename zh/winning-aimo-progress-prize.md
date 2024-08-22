---
title: "NuminaMath 是如何荣膺首届 AIMO 进步奖的？"
thumbnail: /blog/assets/winning-aimo-progress-prize/thumbnail.png
authors:
  - user: yfleureau
    guest: true
    org: AI-MO
  - user: liyongsea
    guest: true
    org: AI-MO
  - user: edbeeching
  - user: lewtun
  - user: benlipkin
    guest: true
    org: AI-MO
  - user: romansoletskyi
    guest: true
    org: AI-MO
  - user: vwxyzjn
  - user: kashif
translators:
  - user: MatrixYao
  - user: zhongdongy
    proofreader: true
---

# NuminaMath 是如何荣膺首届 AIMO 进步奖的？

今年，[**Numina**](https://projectnumina.ai) 和 Hugging Face 合作角逐 [**AI 数学奥林匹克 (AI Math Olympiad，AIMO)**](https://aimoprize.com) 的首届进步奖。此次比赛旨在对开放 LLM 进行微调，以使其能解决高中难度的国际数学奥林匹克训练题。我们很高兴向大家报告: 我们的模型 - [**NuminaMath 7B TIR**](https://huggingface.co/AI-MO/NuminaMath-7B-TIR) - 在比赛中脱颖而出，成功解决了私有测试集 50 道题中的 29 道🥳！

![kaggle.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/winning-aimo-progress-prize/kaggle.png)

本文介绍了 Numina 计划以及我们获胜方案背后的技术细节。如果你想先直接用你的数学难题测一下模型，可跳到这个 [**演示**](https://huggingface.co/spaces/AI-MO/math-olympiad-solver) 玩一玩。

我们开始吧！

- [NuminaMath 是如何荣膺首届 AIMO 进步奖的？](#numinamath-是如何荣膺首届-aimo-进步奖的)
  - [Numina 简介 - 开放的 AI For Math 计划](#numina-简介---开放的-ai-for-math-计划)
  - [AIMO 奖](#aimo-奖)
  - [我们的首个进步奖获奖解决方案](#我们的首个进步奖获奖解决方案)
  - [训练攻略](#训练攻略)
  - [所需惟数据](#所需惟数据)
    - [思维链](#思维链)
    - [工具整合推理](#工具整合推理)
  - [通过自一致性工具整合推理 (SC-TIR) 来抑制高波动](#通过自一致性工具整合推理-sc-tir-来抑制高波动)
    - [避免过拟合诅咒](#避免过拟合诅咒)
  - [我们尝试过的其他点子](#我们尝试过的其他点子)
  - [Numina 的未来 - 寻求贡献者和合作伙伴！](#numina-的未来---寻求贡献者和合作伙伴)
  - [致谢](#致谢)

## Numina 简介 - 开放的 AI For Math 计划

数学总有点与众不同在身上！

人人在日常中都会接触数学，孩子们甚至还未识字就先接触了数学。有史以来最伟大的数学家之一 [**拉马努金**](https://en.wikipedia.org/wiki/Srinivasa_Ramanujan) 于 1887 年出生在印度的一个普通家庭，靠自学成为一代大师。每个人与数学都有或大或小的交集，从用它消遣到靠它吃饭各有不同。

无法否认的是，数学对人类至关重要，商业社会的一切事物，从 iPhone 到核电站等等，都根植于数学之上。但，就算是纯面向应用的数学问题，也自有其趣味之处。

纯粹数学超越了智力，就如无边无际的海洋，唯有心灵才可徜徉其中。

这就是为什么当我们启动 [**Numina**](http://projectnumina.ai) 时，开源和开放数据集成了自然之选。相对于人类智能，我们认为人工智能对数学的进步也理应起到应有的广泛作用。如果计算机是思维的自行车，那么人工智能就是它的引擎 —— 它为我们这个时代的拉马努金打开新的视野。

肇始，在 Mistral AI 的支持下，一群对人工智能和数学充满热情的人于 2023 年底集体创立 ( [**Jia Li**](https://x.com/JiaLi52524397)、[**Yann Fleureau**](https://www.linkedin.com/in/yann-flureau-b1179983/)、[**Guillaume Lample**](https://x.com/GuillaumeLample)、[**Stan Polu**](https://x.com/spolu) 以及 [**Hélène Evain**](https://www.linkedin.com/in/h%C3%A9l%C3%A8ne-evain-473815b1)) 了 Numina，其灵感来自于由 Alex Gerko 和 XTX Markets 发起的人工智能数学奥林匹克 (AI Math Olympiad，AIMO) 竞赛。

2024 年初，Numina 团队获得了两位来自 Hugging Face 的 LLM 微调专家的支持 (👋 [**Lewis Tunstall**](https://x.com/_lewtun) 和 [**Ed Beeching**](https://x.com/edwardbeeching)) 从而开始竞逐 [**2024 AIMO 进步奖**](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize)。随后，我们又获得了 [**General Catalyst**](https://www.generalcatalyst.com/) 和 [**Answer.ai**](http://answer.ai/) 的支持。到 2024 年 3 月，Numina 已聚集了一支 [**来自世界各地的顶尖人才**](http://projectnumina.ai/about-us) 团队。

团队就位后，是时候对 AIMO 发起挑战了！

## AIMO 奖

每年，来自世界各地的高中生都会参加 [**国际数学奥林匹克竞赛**](https://www.imo-official.org) - 一项包含六道富有挑战性的题目，横跨代数、几何、数论等领域的竞赛。为了让大家了解竞赛的难度，下面给出了 [**去年的一道题**](https://www.imo-official.org/problems.aspx):

![imo-problem.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/winning-aimo-progress-prize/imo-problem.png)

2023 年 11 月，[**AIMO 奖**](https://aimoprize.com) 启动，旨在推动擅长数学推理的人工智能模型的开放式开发。谁能训出能够赢得 IMO 金牌的 AI 模型，谁就会获得 500 万美元的大奖。除了大奖之外，AIMO 还推出了一系列 **进步奖**，以奖励在实现这一最终目标过程中的里程碑性工作。首个进步奖是以 [**Kaggle 竞赛**](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize) 的形式举行的，其题目比 IMO 中的题目 _简单一些_ ，相当于 IMO 预选赛的水平。下面，我们给出了一个例题，可以看到，它比上面的 IMO 题目容易一些，但对 LLM 来说仍然很棘手:

> 令 $k, l > 0$ 为参数，抛物线 $y = kx^2 - 2kx + l$ 与直线 $y = 4$ 相交于两点 $A$ 和 $B$，且两点距离为 6。问 $A$ 到原点的距离 和 $B$ 到原点的距离的平方和是多少？

赛题分为两组，每组 50 题，分别作为公开排行榜和私有排行榜，私有排行榜的题目是对参赛者不可见的。这些题目的难度与 [**AMC12**](https://artofproblemsolving.com/wiki/index.php/AMC_12) 和 [**AIME**](https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination) 考试相当，其答案均为整数。比赛用私有排行榜决定最终排名。参赛者每天可以提交两次，仅可使用 2 月 23 日之前发布的开放模型。每次提交都会分配一个 P100 GPU 或 2xT4 GPU，最多给 9 个小时来解决 50 道题。

考虑到上述规则和限制，策略选择对于我们开发制胜方案至关重要。

## 我们的首个进步奖获奖解决方案

经过整个比赛的多轮迭代，我们的首个进步奖解决方案主要由三个部分组成:

- 微调 [**DeepSeekMath-Base 7B**](https://huggingface.co/deepseek-ai/deepseek-math-7b-base) 的攻略。通过该攻略，我们将模型打造成可以解决数学题的“推理代理”，其可以通过把自然语言推理和使用 Python REPL 计算中间结果相结合以最终解决问题。
- 一种带代码执行反馈的、为工具整合推理 (tool-integrated reasonin，TIR) 设计的新解码算法，以在推理过程中生成候选解答。
- 用来指导模型选择并避免过拟合公开排行榜的各种内部验证集。

我们使用了多个开源库来训练我们的模型，主要有 [**TRL**](https://github.com/huggingface/trl)、[**PyTorch**](https://github.com/pytorch/pytorch)、[**vLLM**](https://github.com/vllm-project/vllm) 以及  [**DeepSpeed**](https://github.com/microsoft/DeepSpeed)。在一个 8xH100 GPU 节点上，我们花了 10 个小时训成了模型。

## 训练攻略

    我们采用的微调方法主要基于 [**MuMath-Code 论文**](https://arxiv.org/abs/2405.07551)，其模型训练过程分为两个阶段:

    ![mumath.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/winning-aimo-progress-prize/mumath.png)

    _MuMath-Code 论文中的两阶段训练方法_

- **第 1 阶段:** 在自然语言“数学题 + 解答”的大规模、多样化数据集上微调基础模型，其中每个解答都需套用思维链 (CoT) 模板以促使 LLM 进行推理。
- **第 2 阶段:** 在工具整合推理的合成数据集上微调第 1 阶段得到的模型，其中每个数学题都分解为一系列推理、Python 程序及其输出。此时，我们遵循微软的 [**ToRA 论文**](https://arxiv.org/abs/2309.17452) 的做法，提示 GPT-4 以 ToRA 格式生成带有代码执行反馈的解答。对这些数据进行微调会产生一个推理代理，它可以通过将自然语言推理和使用 Python REPL 来计算中间结果结合起来以解决数学问题 (请参见下图)。

    ![tora.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/winning-aimo-progress-prize/tora.png)

    _来自 ToRA 论文的图，该论文介绍了我们用来训练模型的工具集成推理格式。_

这两个阶段，我们都用了“全模型微调”，所有模型权重在反向传播期间都得到了更新。换句话说，我们没有使用像 LoRA 或 DoRA 这样的参数高效技术，因为没有大量实验表明它们能够媲美全模型微调的性能。我们使用 TRL 的 `SFTTrainer` 中的“填充”功能将多个样本串接到一个 2048 个词元的块中。所有模型模型都使能了梯度 checkpointing 训练，并使用 DeepSpeed ZeRO-3 进行分片，以确保权重、梯度和优化器状态能够放进 VRAM。两个阶段使用的主要超参如下:

| | 1 阶段 | 2 阶段 |
| --- | --- | --- |
| 学习率 | 2.0 E-5 | 2.0 E-5 |
| 总 batch size | 32 | 32 |
| 块大小 | 2048 | 1024 |
| epoch 数 | 3 | 4 |
| 学习率调度器 | cosine | cosine |
| 预热率 | 0.1 | 0.1 |

首次提交时，我们使用了 `DeepSeek 7B` 模型，我们仅对它进行了第 1 阶段微调，但我们发现性能相当有限，其在公开排行榜上的最佳 maj@32 成绩仅为 8/50。[**Abdur Rafae**](https://www.kaggle.com/abdurrafae) 的 [**公开笔记本**](https://www.kaggle.com/code/abdurrafae/improved-code-interpretation) 促使我们考虑在训练方案中加入代码执行。最初，我们专注于 [**MMOS (Mix of Minimal Optimal Sets)**](https://github.com/cyzhh/MMOS) 数据集。我们发现使用 MMOS 虽然提高了性能，但在公开排行榜上的 maj@32 最高分仍只有 16/50，我们当时望文生义地猜测其原因是 MMOS 仅包含单轮解 (即该模型仅生成单个 Python 程序，这不足以解决难题)。后来，我们意识到 MMOS 是一个误称，该 Kaggle 笔记本实际上使用的是 [**DeepSeekMath 7B RL**](https://huggingface.co/deepseek-ai/deepseek-math-7b-rl) 模型，也就是说它能够进行多步推理及代码执行。

经此一役，我们想集中精力生成一个与 DeepSeekMath Instruct/RL 模型使用的数据集类似的数据集，这一做法与 MuMath-Code 攻略结合后，带来了显著的改进。

下面，一起来看看我们是如何构建这些数据集的吧。

## 所需惟数据

在构建数据集时，我们广泛参考了 DeepSeek Math 和其他学者的方法，并对它们进行了大幅扩展。我们生成了含数十万 _数学题 - 解答_ 对的微调数据集，涵盖从高中数学到竞赛级数学的各种知识点。接下来的几周，我们会将该数据集完全开源。同时。我们还可能会用更大的模型来检查我们攻略的可扩展性。有关数据集构建的详细信息，请参阅我们即将发布的数据集技术报告。

具体到这次进步奖，我们为此构建了两个数据集以微调模型。

### 思维链

该数据集由数十万个题目组成，每题都有以思维链的方式编写的解答。数据集的来源范围有中国高中数学练习以及美国及国际数学奥林匹克竞赛题目。数据主要来自在线试卷 PDF 和数学论坛。

处理步骤如下:

1. 对原始 PDF 进行 OCR。
2. 分割为“题目 - 解答”对。
3. 翻译成英文。
4. 重新调整以变成思维链推理格式。
5. 格式化为最终答案。

### 工具整合推理

工具整合推理 (TIR) 在本次比赛中发挥了至关重要的作用。然而，收集和标注此类数据既昂贵又耗时。为了解决这个问题，我们从 Numina 数据集中选择了大约 6 万道题，重点关注那些答案为数字的题，其中大多数答案是整数。

然后，我们利用 GPT-4 的流水线生成类似 TORA 的推理路径，执行代码并生成结果，直到生成完整解答。我们筛选出最终答案与参考答案不匹配的解答，并重复此过程三次，以确保准确性和一致性。这种迭代方法使我们能够高效地生成高质量的 TORA 数据。

作为参考，以下是我们训得的第 1 阶段模型 **NuminaMath-7B-CoT** 和第 2 阶段模型 **NuminaMath-7B-TIR**  [在 **MATH 基准**](https://arxiv.org/abs/2103.03874) 上与其他开放及私有模型的跑分对比:

| 模型                    | MATH (%)                       |
|--------------------------|--------------------------------|
|                          | **思维链推理** |
| GPT-4 (2023)             | 42.5                           |
| GPT-4o                   | 76.6                           |
| Claude 3.5 Sonnet        | 71.1                           |
| DeepSeekMath-7B-Instruct | 46.8                           |
| DeepSeekMath-7B-RL       | 51.7                           |
| NuminaMath-7B-CoT        | 56.3                           |
|                          | **工具整合推理**  |
| DeepSeekMath-7B-Instruct | 57.4                           |
| DeepSeekMath-7B-RL       | 58.8                           |
| NuminaMath-7B-TIR        | 68.2                           |

_各模型在 MATH 基准上的表现。除非明确说明，所有跑分均由零样本贪心解码获得。_

## 通过自一致性工具整合推理 (SC-TIR) 来抑制高波动

正如其他参赛者指出的那样，本次比赛在模型提交和评估方面都带来了一些挑战:

- 评估 API 以随机顺序出题，因此提前停止等策略会产生较高的波动，因为可能一开始就会遇到很多难题，这就导致留给剩余部分的时间就不多了 (反之亦然)。
- LLM 推理中的大多数创新都是基于最新的 GPU 的，因此 `Flash Attention 2` 或 `torch.compile` 等标准方法不适用于 T4 GPU。同样，老 GPU 并不支持 bfloat16 等新数据类型，这促使我们探索 AWQ 和 GPTQ 等训后量化方法。

最初，我们使用 [**Abdur Ra​fae**](https://www.kaggle.com/abdurrafae) 的 [**公开笔记本**](https://www.kaggle.com/code/abdurrafae/improved-code-interpretation) 来提交，但发现高波动是个大问题。为了解决这个问题，我们采取了一种基于工具整合推理的新方法:

![sc-tir.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/winning-aimo-progress-prize/sc-tir.png)

1. 将每道题复制 N 次以生成 vLLM 的一个 batch。N 可以看成多数投票时的候选数量。
2. 对这 N 个输入进行采样解码，直至生成完整的 Python 代码块。
3. 执行每个 Python 代码块并将其输出串接在代码后面，包括栈回溯 (如有)。
4. 重复 M 次以生成 N 个、深度为 M 的生成，允许模型使用栈回溯自纠正代码错误。如果某个样本无法生成合理的输出 (如，生成了不完整的代码块)，就删除之。
5. 对候选解答进行后处理，并使用多数投票来选择最终答案。

我们获胜的提交使用的 `N=48，M=4` 。因为增加任一参数的数值并不会提高性能，所以我们就选择了这两个最小值以保证满足时间限制。实际上，该算法通过工具整合推理增强了  [**CoT 的自一致性**](https://arxiv.org/abs/2305.10601) (如下所示)。

![imo-problem.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/winning-aimo-progress-prize/tot.png)

我们发现，我们的 SC-TIR 算法产生了更稳健的结果，并且显著降低了在内部评估集和公开排行榜上的波动。

值得一提的一个技术细节是，我们发现以 8 比特精度量化模型很有用。原因有三:

- 将模型上传到 Kaggle Hub 非常慢，压缩模型使上传速度提高了一倍。
- T4 GPU 不支持 bfloat16，转换为 float16 会导致模型性能下降。又没法转换为 float32，因为超出了 GPU 可用内存。
- 此外，16 位模型仅用于加载权重就需消耗约 32GB VRAM。对于 2xT4，需要使能 KV 缓存才能快速运行，我们发现在模型精度和速度之间进行折衷是有益的。

我们使用 [**AutoGPTQ**](https://github.com/AutoGPTQ/AutoGPTQ) 以及用于校准数据集来量化我们的模型。在实践中，这会导致准确性小幅下降，但提供了最佳折衷方案，以适配 Kaggle 平台对模型评估所施加的限制。

### 避免过拟合诅咒

过拟合公开排行榜是 Kaggle 竞赛中的常见风险，当测试集只有 50 道题时更是如此。此外，规则允许每天最多提交两次，这使得强大的内部验证数据集对于我们的开发节奏至关重要。根据 AIMO 团队的规定，测试题的难度为中等，介于 AMC12 和 AIME 级别之间，且每题答案为整数。

为了指导模型选择，我们使用了四个内部验证集来衡量模型在不同难度的数学题上的性能。为了避免基础模型中潜在的数据污染，我们从 AMC12 (2022、2023) 和 AIME (2022、2023、2024) 中选择题目以创建两个内部验证数据集:

- **AMC (83 道题):** 我们选择了 [**AMC12**](https://artofproblemsolving.com/wiki/index.php/AMC_12_Problems_and_Solutions) 22、AMC12 23 的所有题目，并保留了那些结果为整数的题目。最终生成的数据集包含 83 道题。该验证集旨在模拟 Kaggle 上的私有测试集，因为我们从竞赛描述中知道题目难度大于等于这个级别。我们发现我们的模型可以解答大约 60-65% 的题目。为了测量波动，每次评估时，我们使用 5-10 个不同的种子，使用我们的 SC-TIR 算法通常会看到大约 1-3% 的波动。
- **AIME (90 道题):** 我们选择了 [**AIME 22**](https://artofproblemsolving.com/wiki/index.php/2022_AIME_I)、[**AIME 23**](https://artofproblemsolving.com/wiki/index.php/2023_AIME_I_Problems) 以及 [**AIME 24**](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I) 的所有题目来度量我们模型解决难题的表现如何，并观测最常见的错误模式。同上，每次评估，我们使用 5-10 个种子进行以测量波动。

由于 AMC/AIME 验证集规模较小，与公开排行榜类似，这些数据集上的模型性能容易受噪声的影响。为了更好地评估模型的性能，我们还使用 MATH 测试集的子集 (含 5,000 道题) 对其进行了评估。我们仅保留答案为整数的题目，以简化多数投票并模拟奥赛评估。因此，我们又多了两个验证集:

- **MATH 4 级 (754 道题)**
- **MATH 5 级 (721 道题)**

通过使用这四个验证集，我们能够在不同的训练阶段选择最有潜力的模型，并缩小超参的选择范围。我们发现，对本 AIMO 赛程而言，将小型但具代表性的验证集与较大的验证集相结合是有用的，因为每个提交都受到抽样随机性的影响。

## 我们尝试过的其他点子

上文有提及，我们在过程中还尝试了一些其他方法，但最终放弃，转而采用 MuMath-Code 的方法。我们尝试过的方法有:

- 训练纯 CoT 模型并使用多数投票进行评估
- 训练 MMOS 模型以通过 Python 一步解决问题

我们还试过对 SFT 模型生成的补全应用 [**Kahneman-Tversky Optimization (KTO)**](https://arxiv.org/abs/2402.01306)，具体想法有点类似于 [**OrcaMath**](https://arxiv.org/abs/2402.14830)，即:

- 交织使用推理和代码执行，每道题用 SFT 模型采样出 4 个补全。我们使用第 2 阶段的 SFT 数据集作为提示。
- 提取答案并将其与标注答案进行比较。如果正确，则将样本标记为正，否则标记为负。
- 在此数据集上对 SFT 模型应用 KTO。

我们发现这种形式的同策 KTO 生成的模型比 SFT 模型稍好 (内部评估好几个百分点)，在公开排行榜上得分为 27/50。

KTO 的一个很好的功能是，你可以在训练期间跟踪隐式奖励，这确实有助于调试 - 如，下图展示了我们成功的训练日志之一，其中人们可以看到正确答案的奖励随着训练而增加，而错误答案的奖励则被抑制。

![kto.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/winning-aimo-progress-prize/kto.png)

但，由于时间关系，我们最终没有将此方法应用于最终的 SFT 模型。如果我们做了的话，可能还能多做对 1-2 道题！

我们还尝试将我们的 SFT 攻略应用于 InternLM-20B、CodeLama-33B 和 Mixtral-8x7B 等更大的模型，但发现 (a) DeepSeek 7B 模型由于已在数学上进行过增量预训练而很难被击败，且 (b) 在 2xT4 GPU 上推理速度非常慢，并且我们遇到了许多神秘的超时，但我们无法分析到其根因。

还有一个失败的实验是尝试将强化学习 (特别是 PPO 算法及 [**Reinforce-Leave-One-Out (RLOO) 算法**](https://arxiv.org/abs/2402.14740)) 和代码执行反馈结合起来以生成对编写代码及获得正确/错误解答的奖励。我们将其应用于 `DeepSeekMath 7B RL` 模型。虽然我们看到了一些很不错的奖励曲线，但我们没有看到性能有任何显著的提升。鉴于像 RLOO 这样的在线方法受限于文本生成的性能并且迭代缓慢，我们放弃了强化学习，转而尝试 KTO。

![rloo.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/winning-aimo-progress-prize/rloo.png)

在推理方面，我们也进行了如下实验:

- 使用静态 KV 缓存和 torch 编译。我们发现我们能够在 H100 上将原生 `transformers` 代码的生成速度加快 2-3 倍，但在 Kaggle T4 上会遇到各种神秘错误，主要是由于 `accelerate` 中的 torch 编译缺乏对模型分片的支持。

各种模型合并技术，例如 [**DARE**](https://arxiv.org/abs/2311.03099)、 [**TIES**](https://arxiv.org/abs/2306.01708) 以及 [**WARP**](https://arxiv.org/abs/2406.16768v1)。这里我们使用 [**mergekit**](https://github.com/arcee-ai/mergekit) 来合并 SFT 和 KTO 模型，或者将 SFT 模型与公开的 `DeepSeekMath` 模型合并。总的来说，我们发现这些合并导致我们的内部评估出现重大倒退，并且我们没有时间对此进行更深入探索。

## Numina 的未来 - 寻求贡献者和合作伙伴！

继 Numina 初步成功赢得 AIMO 2024 进步奖之后，我们的目标变得更为宏大，即肩负促进数学领域人工智能和人类智能发展的使命。你可以访问我们的网站，了解有关我们项目的更多信息，请随时通过 [**contact@projectnumina.ai**](mailto:contact@projectnumina.ai) 给我们留言。

Numina 旨在向世界各地愿意通过人工智能进一步推动数学发展的人才和支持者开放，保持数学的开放本质！

## 致谢

我们感谢 Thomas Wolf 和 Leandro von Werra 促成了 Numina 和 Hugging Face 的合作。我们还感谢 Hugo Larcher 在我们使用 Hugging Face GPU 集群的过程中提供的帮助，Colin Raffel 对模型合并方法的建议，以及 Omar Sanseviero 对博文的反馈。

我们还想感谢 [**Mistral.ai**](https://mistral.ai)、[**General Catalyst**](https://www.generalcatalyst.com/)、[**Answer.AI**](https://answerai.pro) 以及 [**北京大学北京国际数学研究中心**](https://bicmr.pku.edu.cn/) 自项目伊始的支持。

最后，我们感谢 AIMO 团队发起了如此令人激动、鼓舞人心的比赛！