---
title: "开放 LLM 排行榜：深入研究 DROP"
thumbnail: /blog/assets/evaluating-mmlu-leaderboard/thumbnail.png
authors:
- user: clefourrier
- user: cabreraalex
  guest: true
- user: stellaathena
  guest: true
- user: SaylorTwift
- user: thomwolf
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 开放 LLM 排行榜: 深入研究 DROP

最近，[开放 LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 迎来了 [3 个新成员](https://twitter.com/clefourrier/status/1722555555338956840): Winogrande、GSM8k 以及 DROP，它们都使用了 [EleutherAI Harness](https://github.com/EleutherAI/lm-evaluation-harness/) 的原始实现。一眼望去，我们就会发现 DROP 的分数有点古怪: 绝大多数模型的 F1 分数都低于 10 分 (满分 100 分)！我们对此进行了深入调查以一探究竟，请随我们一起踏上发现之旅吧！

## 初步观察

在 DROP (Discrete Reasoning Over Paragraphs，段落级离散推理) 评估中，模型需要先从英文文段中提取相关信息，然后再对其执行离散推理 (例如，对目标对象进行排序或计数以得出正确答案，如下图中的例子)。其使用的指标是自定义 F1 以及精确匹配分数。

<div align="center">
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-llm-leaderboard/drop/drop_example.png" width="500" />
  <figcaption>基于文段的推理示例</figcaption>
</figure>
</div>

三周前，我们将 DROP 添加至开放 LLM 排行榜中，然后我们观察到预训练模型的 DROP F1 分数有个奇怪的趋势: 当我们把排行榜所有原始基准 (ARC、HellaSwag、TruthfulQA 和 MMLU) 的平均分 (我们认为其一定程度上代表了模型的总体性能) 和 DROP 分数作为两个轴绘制散点图时，我们本来希望看到 DROP 分数与原始均分呈正相关的关系 (即原始均值高的模型，DROP 分数也应更高)。然而，事实证明只有少数模型符合这一预期，其他大多数模型的 DROP F1 分数都非常低，低于 10。

<div align="center">
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-llm-leaderboard/drop/drop_bimodal.png" width="500" />
  <figcaption> 该图展现了两类趋势: 少部分模型 DROP 分数与原始均分正相关 (对角线那几个点)，大多数模型则不管原始均分多少，DROP 分数统一集中在 5 左右 (图左侧的垂直线)。</figcaption>
</figure>
</div>

## 文本规范化的锅

第一站，我们观察到文本规范化的结果与预期不符: 在某些情况下，当正确的数字答案后面直接跟有除空格之外的其他空白字符 (如: 换行符) 时，规范化操作导致即使答案正确也无法匹配。举个例子，假设生成的文本是 `10\n\nPassage: The 2011 census recorded a population of 1,001,360` ，而对应的标准答案为 `10` 。

测试基准会先对生成文本和标准答案文本都进行文本规范化，整个过程分为多个步骤:

1) **按分隔符 (`|` 、`-` 或 ` ` ) 分词**
    生成文本的开头 `10\n\nPassage:` 并不包含分隔符，因此会被放进同一个词元 (即第一个词元) ;
2) **删除标点符号**
    删除标点后，第一个词元会变为 `10\n\nPassage` (`:` 被删除);
3) **数字均质化**
    每个可以转换为浮点数的字符串都会被视为数字并转换为浮点数，然后再重新转回字符串。 `10\n\nPassage` 保持不变，因为它不能被转换为浮点数，而标准答案的 `10` 变成了 `10.0` 。
4) **其他步骤**
    随后继续执行其他规范化步骤 (如删除冠词、删除其他空格等)，最终得到的规范化文本是: `10 passage 2011.0 census recorded population of 1001360.0` 。

最终得分并不是根据字符串计算而得，而是根据从字符串中提取的词袋 (bag of words，BOW) 计算而得。仍用上例，规范化后的生成文本词袋为 `{'recorded', 'population', 'passage', 'census', '2011.0', ' 1001360.0', '10'}` ，而规范化后的标准答案词袋为 `{10.0}` ，两者求交，正如你所看到的，即使模型生成了正确答案，两者交集也为 0！

总之，如果一个数字后面跟着除标准空格字符外的任何其它表示空格的字符，目前的文本规范化实现就不会对该数字进行规范化，因此如果此时标准答案也是一个数字，那么两者就永远无法匹配了！这个问题可能给最终分数带来严重影响，但显然这并是导致 DROP 分数如此低的唯一罪魁祸首。我们决定继续调查。

## 对结果进行深入研究

我们在 [Zeno](https://zenoml.com) 的朋友加入了调查并对结果 [进行了更深入的探索](https://hub.zenoml.com/report/1255/DROP%20Benchmark%20Exploration)，他们选择了 5 个有代表性的模型进行深入分析: falcon-180B 和 mistra-7B 表现低于预期，Yi-34B 和 Tigerbot-70B 的 DROP 分数与原始均分正相关，而 facebook/xglm-7.5B 则落在中间。

如果你有兴趣的话，也可以试试在 [这个 Zeno 项目](https://hub.zenoml.com/project/2f5dec90-df5e-4e3e-a4d1-37faf814c5ae/OpenLLM%20Leaderboard%20DROP%20Comparison/explore?params=eyJtb2RlbCI6ImZhY2Vib29rX194Z2xtLTcuNUIiLCJtZXRyaWMiOnsiaWQiOjk1NjUsIm5hbWUiOiJmMSIsInR5cGUiOiJtZWFuIiwiY29sdW1ucyI6WyJmMSJdfSwiY29tcGFyaXNvbk1vZGVsIjoiVGlnZXJSZXNlYXJjaF9fdGlnZXJib3QtNzBiLWNoYXQiLCJjb21wYXJpc29uQ29sdW1uIjp7ImlkIjoiYzJmNTY1Y2EtYjJjZC00MDkwLWIwYzctYTNiNTNkZmViM2RiIiwibmFtZSI6ImVtIiwiY29sdW1uVHlwZSI6IkZFQVRVUkUiLCJkYXRhVHlwZSI6IkNPTlRJTlVPVVMiLCJtb2RlbCI6ImZhY2Vib29rX194Z2xtLTcuNUIifSwiY29tcGFyZVNvcnQiOltudWxsLHRydWVdLCJtZXRyaWNSYW5nZSI6W251bGwsbnVsbF0sInNlbGVjdGlvbnMiOnsic2xpY2VzIjpbXSwibWV0YWRhdGEiOnt9LCJ0YWdzIjpbXX19) 上分析一把。

Zeno 团队发现了两件更麻烦的事情:

1) 如果答案是浮点数，没有一个模型的结果是正确的
2) 擅长生成长答案的高质量模型 F1 分数反而更低

最后，我们认为这两件事情实际上是同一个根因引起的，即: 使用 `.` 作为停止词 (以结束生成):

1) 浮点数答案在生成过程中直接被截断了 [译者注: 小数点被当成句号直接中断输出了。]
2) 更高质量的模型，为了尝试匹配少样本提示格式，其生成会像这样 `Answer\n\nPlausible prompt for the next question.` ，而按照当前停止词的设定，该行为仅会在结果生成后且遇到第一个 `.` 停止，因此模型会生成太多多余的单词从而导致糟糕的 F1 分数。

我们假设这两个问题都可以通过使用 `\n` 而不是 `.` 来充当停止词而得到解决。

## 更改生成停止词

我们对此进行了初步实验！我们试验了在现有的生成文本上使用 `\n` 作为结束符。如果生成的答案中有 `\n` ，我们就在遇到第一个 `\n` 时截断文本，并基于截断文本重新计算分数。

_请注意，这只能近似正确结果，因为它不会修复由于 `.` 而过早截断的答案 (如浮点数答案)。但同时，它也不会给任何模型带来不公平的优势，因为所有模型都受这个问题的影响。因此，这是我们在不重新运行模型的情况下 (因为我们希望尽快向社区发布进展) 能做的最好的事情了。_

结果如下。使用 `\n` 作为停止词后，DROP 分数与原始均分的相关度提高不少，因此模型的 DROP 分数与模型原始的总体表现相关度也变高了。

<div align="center">
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-llm-leaderboard/drop/drop_partial_fix.png" width="500" />
  <figcaption>我们可以看到橙色部分表示在处理后的新答案上计算的分数，其与模型原始均分的相关性更好。</figcaption>
</figure>
</div>

## 那下一步咋整

快速估算一下，重新对所有模型运行完整评估的成本相当高 (全部更新需花 8 个 GPU 年，DROP 占用了其中的很大一部分)。因此，我们对仅重新运行失败的例子所需要的成本进行了估算。

有 10% 样本的标准答案是浮点数 (如 `12.25` )，且模型输出以正确答案开头 (本例中为 `12` )，但在 `.` 处被截断 - 这种情况如果继续生成的话，有可能答案是正确的，因此我们肯定要重新运行！但这 10% 尚不包括以数字结尾的句子，这类句子也可能会被不当截断 (在剩下的 90% 中占 40%)，也不包括被规范化操作搞乱掉的情况。

因此，为了获得正确的结果，我们需要重新运行超过 50% 的样本，这需要大量的 GPU 时！我们需要确保这次要运行的代码是正确的。

于是，我们与 EleutherAI 团队通过 [GitHub](https://github.com/EleutherAI/lm-evaluation-harness/issues/978) 及内部渠道进行了广泛的讨论，他们指导我们理解代码并帮助我们进行调查，很明显，LM Eval Harness 的实现严格遵循了“官方 DROP 代码”的实现，因此这不是 LM Eval Harness 的 bug，而是需要开发 DROP 基准评估的新版本！

**因此，我们决定暂时从 Open LLM 排行榜中删除 DROP，直到新版本出现为止。**

从本次调查中我们学到的一点是，通过社区协作对基准测试进行检阅，能发现以前遗漏的错误，这一点很有价值。开源、社区和开放式研发的力量再次闪耀，有了这些，我们甚至可以透明地调查一个已经存在数年的基准上的问题并找到根因。

我们希望有兴趣的社区成员与发明 DROP 评估的学者联手，以解决其在评分及文本规范化上的问题。我们希望能再次使用它，因为数据集本身非常有趣而且很酷。如国你对如何评估 DROP 有任何见解，请不要犹豫，[告诉我们](https://github.com/EleutherAI/lm-evaluation-harness/issues/1050)。

感谢众多社区成员指出 DROP 分数的问题，也非常感谢 EleutherAI Harness 和 Zeno 团队在此问题上的大力协助。
