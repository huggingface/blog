---
title: "ChatGPT 设计游戏剧情 | 基于 AI 5 天创建一个农场游戏，完结篇！"
thumbnail: /blog/assets/124_ml-for-games/thumbnail5.png
authors:
- user: dylanebert
translators:
- user: SuSung-boy
---

# ChatGPT 设计游戏剧情 | 基于 AI 5 天创建一个农场游戏，完结篇！


**欢迎使用 AI 进行游戏开发！** 在本系列中，我们将使用 AI 工具在 5 天内创建一个功能完备的农场游戏。到本系列结束时，您将了解到如何将多种 AI 工具整合到游戏开发流程中。本文将向您展示如何将 AI 工具用于:

1. 美术风格
2. 游戏设计
3. 3D 素材
4. 2D 素材
5. 剧情

想快速观看视频的版本？你可以在 [这里](https://www.tiktok.com/@individualkex/video/7197505390353960235) 观看。不过如果你想要了解技术细节，请继续阅读吧！

**注意:** 此篇文章多次引用 第 2 部分 相关内容。简单来说，[第 2 部分](https://huggingface.co/blog/zh/ml-for-games-2) 使用了 ChatGPT 进行游戏设计; 更具体地，介绍了 ChatGPT 的工作原理、语言模型及其局限性。如果您还没有阅读过，可以跳转阅读更多信息。

## 第 5 天: 剧情

在本教程系列的 [第 4 部分](https://huggingface.co/blog/zh/ml-for-games-4) 中，我们介绍了如何将 Stable Diffusion 和 Image2Image 工具嵌入到传统 2D 素材制作流程中，来帮助从业者使用 AI 制作 2D 游戏素材。

本文是该系列的最后一部分，我们将使用 AI 设计游戏剧情。首先，我会介绍使用语言模型为农场游戏生成 [剧情的设计流程](#剧情设计流程)，请注意带有 ⚠️ **局限性** 标识的段落。其次，我会具体阐述涉及到的相关技术，以及它们在游戏开发方面的 [发展方向](#发展方向)。最后，我会对本系列做一个 [总结](#结语)。


### 剧情设计流程

**必要条件:** [ChatGPT](https://openai.com/blog/chatgpt/)。ChatGPT 会贯穿整个剧情设计流程。可以跳转 [第 2 部分](https://huggingface.co/blog/zh/ml-for-games-2) 阅读更多相关信息。实际上 ChatGPT 并不是唯一的可行方案，有许多竞争对手正在涌现，包括一些开源的对话代理 (dialog agent)。我会在后面的部分详细介绍对话代理 [新兴领域](#新兴领域)。

1. **让 ChatGPT 写剧情概要。** 我给 ChatGPT 提供了大量农场游戏相关信息，让它写一个剧情概要。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt1.png" alt="ChatGPT for Story #1">
</div>

ChatGPT 回答的剧情概要与 [星露谷物语](https://www.stardewvalley.net/) 极其相似。 

> ⚠️ **局限性:** 由于训练语料库的原因，语言模型倾向于生成现有的剧情。

这说明了不能完全依赖语言模型代替人工，而应该把语言模型作为激发创意的工具。例如上文中 ChatGPT 生成了与星露谷物语相似的剧情，完全不具备原创性。

2. **优化概要。** 与 [第 4 部分](https://huggingface.co/blog/zh/ml-for-games-4) 中的 Image2Image 相同，这类工具在工作流程中需要反复迭代多次才能发挥潜力。接下里，我继续询求 ChatGPT 更具原创性的结果。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt2.png" alt="ChatGPT for Story #2">
</div>

这次的回答好多了。我继续优化结果，例如要求移除农场游戏中不必要的魔法元素。经过几次迭代，我得到了一份满意的剧情概要。接下来就是生成游戏剧情的具体细节了。

3. **让 ChatGPT 写剧情细节。** 剧情概要基本确定之后，我继续询求 ChatGPT 补充游戏剧情细节信息。就该系列的农场游戏而言，唯一需要补充的是游戏介绍和农作物简介。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt3.png" alt="ChatGPT for Story #3">
</div>

得到的回答还不错。不过我在第 1~4 部分中开发的游戏内容里，并没有经验丰富的农夫提供帮助这一特性，也没有新的冒险和挑战系统。

4. **优化细节。** 同样地，我继续迭代优化剧情细节。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt4.png" alt="ChatGPT for Story #4">
</div>

我对这个回答很满意。那么新的问题来了，我可以直接把这段内容应用到我的游戏中吗？对于此系列的农场游戏而言，或许可以，因为这是一款为 AI 教程开发的免费游戏。但是对于商业产品而言，最好不要，它很可能会导致潜在的法律、道德和商业后果。

> ⚠️ **局限性:** 直接将语言模型的生成内容拿来自行使用，很可能会导致潜在的法律、道德和商业后果。

潜在的后果如下:
- <u>法律:</u> 目前围绕生成式 AI 的法律环境非常不明朗，有几起存在争议的诉讼正在进行中。
- <u>道德:</u> 语言模型生成的内容可能包含抄袭和偏见。详情请见 [道德与社会新闻稿](https://huggingface.co/blog/zh/ethics-soc-2)。 
- <u>商业:</u> [一些消息](https://www.searchenginejournal.com/google-says-ai-generated-content-is-against-guidelines/444916/) 来源显示，AI 生成的内容可能会被搜索引擎降低优先级。SEO (Search Engine Optimization，搜索引擎优化，是一项优化搜索引擎排名的技术) 指出，[不同于](https://seo.ai/blog/google-is-not-against-ai-content) 垃圾邮件需要被搜索引擎排除，AI 生成的内容对搜索引擎具有一定的价值，但并不需要太高的优先级。同时，一些 [AI 内容检测](https://writer.com/ai-content-detector/) 工具可以检查搜索到的内容是否为 AI 生成的，例如正在研究的语言模型 [watermarking](https://arxiv.org/abs/2301.10226) 可以给 AI 生成内容增加 隐式水印，以使更容易地被 AI 内容检测工具捕捉。

考虑到这些局限性，最安全的方法可能是: 仅使用 ChatGPT 等语言模型进行头脑风暴，获取灵感后手动完成最终内容。

5. **细化描述。** 我继续询求 ChatGPT 对农作物商品的细致描述。

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt5.png" alt="ChatGPT for Story #5">
</div>

由于此系列的农场游戏剧情简短，使用 ChatGPT 生成全部剧情内容非常有效。但是当生成的剧情越来越长，局限性就会越加明显: ChatGPT 不太适合生成长篇连贯剧情。即使仅仅是询求 ChatGPT 为农作物生成简短的描述句子，几次询求之后生成的内容质量也开始下降并且陷入重复。

> ⚠️ **局限性:** 语言模型生成的内容重复性高。

最后，我总结了使用 AI 设计游戏剧情的整体流程，以下是我个人经验的一些技巧:
- **询求剧情概要。** 语言模型生成的长篇内容质量可能较低，生成高抽象度的剧情概要往往效果更好。
- **头脑风暴。** 语言模型生成的内容不连贯，用在某个点上激发创意比较好。例如先设计一个角色基本框架，再使用 AI 来头脑风暴角色的具体细节。
- **优化内容。** 写下设计的剧情内容，并询求语言模型如何改进。即使生成内容不被采用，它也可能激发您改进相关的创意。

尽管语言模型有以上种种局限性，但对游戏开发而言，对话代理仍然是一个非常有用的工具。而这仅仅是个开始，接下来我会探讨对话代理的新兴领域及其对游戏开发的潜在影响。

### 发展方向

#### 新兴领域

我在 [剧情设计流程](#剧情设计流程) 部分介绍了如何使用 ChatGPT 辅助设计，也提到 ChatGPT 并不是唯一的可行方案。[Character.AI](https://beta.character.ai/) 是一个提供多种极具个性的角色定制化对话代理服务 (例如您可以跟 “埃隆·马斯克” 和  “迈克尔·杰克逊” 对话) 的网站，同时它也提供专门的 [创意写作对话代理](https://beta.character.ai/chat?char=9ZSDyg3OuPbFgDqGwy3RpsXqJblE4S1fKA_oU3yvfTM) 服务。

除此之外，还有许多尚未公开的对话代理模型。可以在 [这篇文章](https://huggingface.co/blog/zh/dialog-agents) 查看这些模型的异同以及更多对话代理相关信息，其中涉及到的模型主要包括:
- [Google's LaMDA](https://arxiv.org/abs/2201.08239) 和 [Bard](https://blog.google/technology/ai/bard-google-ai-search-updates/)
- [Meta's BlenderBot](https://arxiv.org/abs/2208.03188)
- [DeepMind's Sparrow](https://arxiv.org/abs/2209.14375) 
- [Anthropic's Assistant](https://arxiv.org/abs/2204.05862).

上面提到的 ChatGPT 的竞争对手都是闭源的。此外也有一些对话代理的开源工作，例如 [LAION 的 OpenAssistant](https://github.com/LAION-AI/Open-Assistant)，[CarperAI](https://carper.ai) 的开源报告，以及 [谷歌的 FLAN-T5 XXL](https://huggingface.co/google/flan-t5-xxl) 的开源版本，这些与 [LangChain](https://github.com/hwchase17/langchain) 等开源工具结合使用，可以将语言模型的输入和输出连接起来，有助于开放式对话代理的开发工作。

前段时间，Stable Diffusion 开源版本的出现激发了很多领域爆发式革新，农场游戏系列教程的灵感也来源于此。语言模型也相同，要在游戏开发中加入语言类的 AI 应用，开源社区将成为未来的关键一环。如果您想跟上最新进展，可以在 [Twitter](https://twitter.com/dylan_ebert_) 上关注我，随时与我联系，我们一起探讨语言模型的发展潜力。

#### 游戏内开发方向

**NPCs:** 除了在游戏开发流程中使用语言模型和对话代理帮助设计游戏剧情等之外，在游戏内还有一个令人兴奋的开发潜力尚未实现，最明显的例子是 AI 驱动的 NPC。实际上已经出现了一些基于此想法的初创公司。就我个人而言，我目前还不清楚如何使用语言模型开发一个智能 NPC。但我认为 AI-NPC 就在不远的未来，请持续关注我的最新进展。

**控制系统:** 想象一下，如果不需要键盘、手柄等控制器，而用对话的方式来控制游戏会怎么样？尽管现在还没有游戏实现这种功能，但它并不是一件困难的事。如果您对此有兴趣，也请持续关注我。

### 结语

至此，5 天创建一个农场游戏系列就结束了。那么，想看最终游戏的样子，或者想亲自试玩一下吗？来 [Hugging Face Space 应用](https://huggingface.co/spaces/dylanebert/FarmingGame) 或 [itch.io](https://individualkex.itch.io/farming-game) 吧！

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/game.png" alt="Final Farming Game">
</div>

感谢您阅读 基于 AI 进行游戏开发 系列文章！本系列仅仅是 Hugging Face AI 开发游戏的开始，未来还会有更多内容！如果您有任何问题，或者想了解更多相关内容，现在来加入 Hugging Face 官方 [Discord 频道](https://hf.co/join/discord) 与我们交流吧！
