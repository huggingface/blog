---
title: "Ethics and Society Newsletter #4: Bias in Text-to-Image Models"
thumbnail: /blog/assets/152_ethics_soc_4/ethics_4_thumbnail.png
authors:
- user: sasha
- user: giadap
- user: nazneen
- user: allendorf
- user: irenesolaiman
- user: natolambert
- user: meg
translators:
- user: innovation64
- user: zhongdongy
  proofreader: true
---

# 道德与社会问题简报 #4: 文生图模型中的偏见


**简而言之: 我们需要更好的方法来评估文生图模型中的偏见**

## 介绍

[文本到图像 (TTI) 生成](https://huggingface.co/models?pipeline_tag=text-to-image&sort=downloads) 现在非常流行，成千上万的 TTI 模型被上传到 Hugging Face Hub。每种模态都可能受到不同来源的偏见影响，这就引出了一个问题: 我们如何发现这些模型中的偏见？在当前的博客文章中，我们分享了我们对 TTI 系统中偏见来源的看法以及解决它们的工具和潜在解决方案，展示了我们自己的项目和来自更广泛社区的项目。

## 图像生成中编码的价值观和偏见

[偏见和价值](https://www.sciencedirect.com/science/article/abs/pii/B9780080885797500119) 之间有着非常密切的关系，特别是当这些偏见和价值嵌入到用于训练和查询给定 [文本到图像模型](https://dl.acm.org/doi/abs/10.1145/3593013.3594095) 的语言或图像中时; 这种现象严重影响了我们在生成图像中看到的输出。尽管这种关系在更广泛的人工智能研究领域中是众所周知的，并且科学家们正在进行大量努力来解决它，但试图在一个模型中表示一个给定人群价值观的演变性质的复杂性仍然存在。这给揭示和充分解决这一问题带来了持久的道德挑战。

例如，如果训练数据主要是英文，它们可能传达相当西方化的价值观。结果我们得到了对不同或遥远文化的刻板印象。当我们比较 ERNIE ViLG (左) 和 Stable Diffusion v 2.1 (右) 对同一提示“北京的房子”的结果时，这种现象显得非常明显:

<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/ernie-sd.png" alt="results of ERNIE ViLG (left) and Stable Diffusion v 2.1 (right) for the same prompt, a house in Beijing" />
</p>

## 偏见的来源

近年来，人们在单一模态的 AI 系统中进行了大量关于偏见检测的重要研究，包括自然语言处理 ([Abid et al., 2021](https://dl.acm.org/doi/abs/10.1145/3461702.3462624)) 和计算机视觉 ([Buolamwini and Gebru, 2018](http://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf))。由于机器学习模型是由人类构建的，因此所有机器学习模型 (实际上，所有技术) 都存在偏见。这可能表现为图像中某些视觉特征的过度和不足 (例如，所有办公室工作人员都系着领带)，或者文化和地理刻板印象的存在 (例如，所有新娘都穿着白色礼服和面纱，而不是更具代表性的世界各地的新娘，如穿红色纱丽的新娘)。鉴于 AI 系统被部署在社会技术背景下，并且在不同行业和工具中广泛部署 (例如 [Firefly](https://www.adobe.com/sensei/generative-ai/firefly.html)，[Shutterstock](https://www.shutterstock.com/ai-image-generator))，它们特别容易放大现有的社会偏见和不平等。我们旨在提供一个非详尽的偏见来源列表:

**训练数据中的偏见:** 一些流行的多模态数据集，如文本到图像的 [LAION-5B](https://laion.ai/blog/laion-5b/)，图像字幕的 [MS-COCO](https://cocodataset.org/) 和视觉问答的 [VQA v2.0](https://paperswithcode.com/dataset/visual-question-answering-v2-0)，已经被发现包含大量的偏见和有害关联 ([Zhao et al 2017](https://aclanthology.org/D17-1323/)，[Prabhu and Birhane, 2021](https://arxiv.org/abs/2110.01963)，[Hirota et al, 2022](https://facctconference.org/static/pdfs_2022/facct22-3533184.pdf))，这些偏见可能会渗透到在这些数据集上训练的模型中。例如，来自 [Hugging Face Stable Bias project](https://huggingface.co/spaces/society-ethics/StableBias) 的初步结果显示，图像生成缺乏多样性，并且延续了文化和身份群体的常见刻板印象。比较 Dall-E 2 生成的 CEO (右) 和经理 (左)，我们可以看到两者都缺乏多样性:

<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/CEO_manager.png" alt="Dall-E 2 generations of CEOs (right) and managers (left)" />
</p>

**预训练数据过滤中的偏见:** 在将数据集用于训练模型之前，通常会对其进行某种形式的过滤; 这会引入不同的偏见。例如，在他们的 [博客文章](https://openai.com/research/dall-e-2-pre-training-mitigations) 中，Dall-E 2 的创建者发现过滤训练数据实际上会放大偏见 - 他们假设这可能是由于现有数据集偏向于在更性感化的背景下呈现女性，或者由于他们使用的过滤方法本身具有偏见。

**推理中的偏见:** 用于指导 Stable Diffusion 和 Dall-E 2 等文本到图像模型的训练和推理的 [CLIP 模型](https://huggingface.co/openai/clip-vit-large-patch14) 有许多 [记录详细的偏见](https://arxiv.org/abs/2205.11378)，涉及年龄、性别和种族或族裔，例如将被标记为 `白人` 、 `中年` 和 `男性` 的图像视为默认。这可能会影响使用它进行提示编码的模型的生成，例如通过解释未指定或未明确指定的性别和身份群体来表示白人和男性。

**模型潜在空间中的偏见:** 已经进行了一些 [初步工作](https://arxiv.org/abs/2302.10893)，探索模型的潜在空间并沿着不同轴 (如性别) 引导图像生成，使生成更具代表性 (参见下面的图像)。然而，还需要更多工作来更好地理解不同类型扩散模型的潜在空间结构以及影响生成图像中反映偏见的因素。

<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/fair-diffusion.png" alt="Fair Diffusion generations of firefighters." />
</p>

**后期过滤中的偏见:** 许多图像生成模型都内置了旨在标记问题内容的安全过滤器。然而，这些过滤器的工作程度以及它们对不同类型内容的鲁棒性有待确定 - 例如，[对 Stable Diffusion 安全过滤器进行红队对抗测试](https://arxiv.org/abs/2210.04610) 表明，它主要识别性内容，并未能标记其他类型的暴力、血腥或令人不安的内容。

## 检测偏见

我们上面描述的大多数问题都不能用单一的解决方案解决 - 实际上，[偏见是一个复杂的话题](https://huggingface.co/blog/ethics-soc-2)，不能仅靠技术来有意义地解决。偏见与它所存在的更广泛的社会、文化和历史背景紧密相连。因此，解决 AI 系统中的偏见不仅是一个技术挑战，而且是一个需要多学科关注的社会技术挑战。其中包括工具、红队对抗测试和评估在内的一系列方法可以帮助我们获得重要的见解，这些见解可以为模型创建者和下游用户提供有关 TTI 和其他多模态模型中包含的偏见的信息。

我们在下面介绍一些这些方法:

**探索偏见的工具:** 作为 [Stable Bias 项目](https://huggingface.co/spaces/society-ethics/StableBias) 的一部分，我们创建了一系列工具来探索和比较不同文本到图像模型中偏见的视觉表现。例如，[Average Diffusion Faces](https://huggingface.co/spaces/society-ethics/Average_diffusion_faces) 工具让你可以比较不同职业和不同模型的平均表示 - 如下面所示，对于 ‘janitor’，分别为 Stable Diffusion v1.4、v2 和 Dall-E 2:

<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/average.png" alt="Average faces for the 'janitor' profession, computed based on the outputs of different text to image models." />
</p>

其他工具，如 [Face Clustering tool](https://hf.co/spaces/society-ethics/DiffusionFaceClustering) 和 [Colorfulness Profession Explorer](https://huggingface.co/spaces/tti-bias/identities-colorfulness-knn) 工具，允许用户探索数据中的模式并识别相似性和刻板印象，而无需指定标签或身份特征。事实上，重要的是要记住，生成的个人图像并不是真实的人，而是人工创造的，所以不要把它们当作真实的人来对待。根据上下文和用例，这些工具可以用于讲故事和审计。

**红队对抗测试:** [“红队对抗测试”](https://huggingface.co/blog/red-teaming) 包括通过提示和分析结果来对 AI 模型进行潜在漏洞、偏见和弱点的压力测试。虽然它已经在实践中用于评估语言模型 (包括即将到来的 [DEFCON 上的 Generative AI Red Teaming 活动](https://aivillage.org/generative%20red%20team/generative-red-team/)，我们也参加了)，但目前还没有建立起系统化的红队对抗测试 AI 模型的方法，它仍然相对临时性。事实上，AI 模型中有这么多潜在的故障模式和偏见，很难预见它们全部，而生成模型的 [随机性质](https://dl.acm.org/doi/10.1145/3442188.3445922) 使得难以复现故障案例。红队对抗测试提供了关于模型局限性的可行性见解，并可用于添加防护栏和记录模型局限性。目前没有红队对抗测试基准或排行榜，突显了需要更多开源红队对抗测试资源的工作。[Anthropic 的红队对抗测试数据集](https://github.com/anthropics/hh-rlhf/tree/master/red-team-attempts) 是唯一一个开源的红队对抗测试 prompts，但仅限于英语自然语言文本。

**评估和记录偏见:** 在 Hugging Face，我们是 [模型卡片](https://huggingface.co/docs/hub/model-card-guidebook) 和其他形式的文档 (如 [数据表](https://arxiv.org/abs/1803.09010)、README 等) 的大力支持者。在文本到图像 (和其他多模态) 模型的情况下，使用探索工具和红队对抗测试等上述方法进行的探索结果可以与模型检查点和权重一起共享。其中一个问题是，我们目前没有用于测量多模态模型 (特别是文本到图像生成系统) 中偏见的标准基准或数据集，但随着社区在这个方向上进行更多 [工作](https://arxiv.org/abs/2306.05949)，不同的偏见指标可以在模型文档中并行报告。

## 价值观和偏见

上面列出的所有方法都是检测和理解图像生成模型中嵌入的偏见的一部分。但我们如何积极应对它们呢？

一种方法是开发新的模型，代表我们希望它成为社会性模型。这意味着创建不仅模仿我们数据中的模式，而且积极促进更公平、更公正观点的 AI 系统。然而，这种方法提出了一个关键问题: 我们将谁的价值观编程到这些模型中？价值观在不同文化、社会和个人之间有所不同，使得在 AI 模型中定义一个“理想”的社会应该是什么样子成为一项复杂的任务。这个问题确实复杂且多面。如果我们避免在我们的 AI 模型中再现现有的社会偏见，我们就面临着定义一个“理想”的社会表现的挑战。社会并不是一个静态的实体，而是一个动态且不断变化的构造。那么，AI 模型是否应该随着时间的推移适应社会规范和价值观的变化呢？如果是这样，我们如何确保这些转变真正代表了社会中所有群体，特别是那些经常被忽视的群体呢？

此外，正如我们在 [上一期简报](https://huggingface.co/blog/ethics-soc-2#addressing-bias-throughout-the-ml-development-cycle) 中提到的，开发机器学习系统并没有一种单一的方法，开发和部署过程中的任何步骤都可能提供解决偏见的机会，从一开始谁被包括在内，到定义任务，到策划数据集，训练模型等。这也适用于多模态模型以及它们最终在社会中部署或生产化的方式，因为多模态模型中偏见的后果将取决于它们的下游使用。例如，如果一个模型被用于人机交互环境中的图形设计 (如 [RunwayML](https://runwayml.com/ai-magic-tools/text-to-image/) 创建的那些)，用户有多次机会检测和纠正偏见，例如通过更改提示或生成选项。然而，如果一个模型被用作帮助法医艺术家创建潜在嫌疑人警察素描的 [工具](https://www.vice.com/en/article/qjk745/ai-police-sketches) (见下图)，那么风险就更高了，因为这可能在高风险环境中加强刻板印象和种族偏见。

<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/forensic.png" alt="Forensic AI Sketch artist tool developed using Dall-E 2." />
</p>

## 其他更新

我们也在继续在道德和社会的其他方面进行工作，包括:

- **内容审核:**
  - 我们对我们的 [内容政策](https://huggingface.co/content-guidelines) 进行了重大更新。距离我们上次更新已经快一年了，自那时起 Hugging Face 社区增长迅速，所以我们觉得是时候了。在这次更新中，我们强调 _同意_ 是 Hugging Face 的核心价值之一。要了解更多关于我们的思考过程，请查看 [公告博客](https://huggingface.co/blog/content-guidelines-update) **。**

- **AI 问责政策:**

  - 我们提交了对 NTIA 关于 [AI 问责政策](https://ntia.gov/issues/artificial-intelligence/request-for-comments) 的评论请求的回应，在其中我们强调了文档和透明度机制的重要性，以及利用开放协作和促进外部利益相关者获取的必要性。你可以在我们的 [博客文章](https://huggingface.co/blog/policy-ntia-rfc) 中找到我们回应的摘要和完整文档的链接！

## 结语

从上面的讨论中你可以看出，检测和应对多模态模型 (如文本到图像模型) 中的偏见和价值观仍然是一个悬而未决的问题。除了上面提到的工作，我们还在与社区广泛接触这些问题 - 我们最近在 FAccT 会议上共同主持了一个关于这个主题的 [CRAFT 会议](https://facctconference.org/2023/acceptedcraft.html)，并继续在这个主题上进行数据和模型为中心的研究。我们特别兴奋地探索一个更深入地探究文本到图像模型中所蕴含的 [价值](https://arxiv.org/abs/2203.07785) 及其所代表的方向 (敬请期待！)。