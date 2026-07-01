---
title: "经同意的语音克隆"
thumbnail: /blog/assets/voice-consent-gate/thumbnail.png
authors:
- user: meg
- user: frimelle
translators:
- user: chenglu
---

# 经同意的语音克隆

**在这篇博客文章中，我们介绍了“语音同意验证机制 (voice consent gate)”的概念，支持通过明确同意来进行语音克隆。我们还提供了一个 [示例 Space 应用](https://huggingface.co/spaces/society-ethics/RepeatAfterMe) 和 [相关代码](https://huggingface.co/spaces/society-ethics/RepeatAfterMe/tree/main)，帮助大家快速上手这一想法。**

<img src="https://huggingface.co/spaces/society-ethics/RepeatAfterMe/resolve/main/assets/voice_consent_gate.png" alt="Line-drawing/clipart of a gate, where the family name says Consent" width="50%"/>

近年来，逼真的语音生成技术已经达到了令人惊讶的水平。在某些情况下，生成出来的合成语音几乎能以假乱真，和真人的声音非常相似。如今，曾经只存在于科幻小说中的“语音克隆”已经成为现实。只需要几秒钟的录音，就能让任何人的声音“说出”任何内容。

语音生成，尤其是语音克隆技术，既有风险也有益处。它可能被用于制作“深度伪造”内容，例如 [有人用前总统 Biden 的克隆语音进行自动电话宣传](https://www.reuters.com/world/us/fcc-finalizes-6-million-fine-over-ai-generated-biden-robocalls-2024-09-26/)，误导公众以为他说过其实并未说的话。但与此同时，语音克隆也可以带来积极作用，比如 [帮助失语者](https://www.nature.com/articles/s41598-024-84728-y) [重新用自己的声音表达](https://www.thetimes.com/uk/healthcare/article/elevenlabs-voice-clone-ai-als-t3ntnpcl7)，或者辅助人们学习语言和方言。

那么，我们该如何实现“有意义的使用”而不是“恶意的滥用”？我们正在探索一种可能的答案：引入一个**语音同意验证机制 (voice consent gate)**。也就是说，只有当说话人明确表达了同意，语音克隆模型才能使用其声音。换句话说，模型不会擅自“说出”你的声音，除非你亲口同意。

下面是我们对这一想法的基础演示：

<iframe
	src="https://society-ethics-repeatafterme.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>


## 实践中的伦理：将“同意”融入系统基础设施

语音同意门是我们正在尝试的一种基础设施设计，用来把“**同意**”这样的伦理原则直接嵌入到 AI 系统的工作流程中。在我们的演示中，模型只有在检测到说话人清楚地说出了同意语句之后，才会启动。也就是说，“同意”成为系统运行的前提条件，让原本抽象的伦理原则变成了具体可操作的系统规则，并形成可追溯、可审核的交互记录：AI 模型只会在明确同意之后才运行。

这样的设计不仅适用于语音克隆，更展示了如何从系统层面保障用户的自主权，以及如何将透明和同意变成 **可执行的功能**，而不仅仅是口头承诺。


## 技术细节

要构建一个包含语音同意门的基础语音克隆系统，你需要以下三部分：

1. 一种方法，用来生成说话人当前上下文中可用的、表达明确同意的唯一语句。
2. 一个 **自动语音识别（ASR）系统**，用于识别说话人所说的同意语句。
3. 一个 **语音克隆的文本转语音（TTS）系统**，可以接收文本和说话人的语音片段来合成新的语音。

**我们的发现是：** 现在很多语音克隆模型只需要一句话就能模仿说话人的声音，因此这句用于表达“同意”的句子，也可以同时作为语音克隆的输入数据。


### 实现方法

**关于“同意”：**
在英语语音克隆系统中创建语音同意门的方式是：为说话人生成一句简短、自然、约20个单词左右的英文语句，让其朗读。这句话要明确表达对当前使用情境的知情同意。我们建议在句中明确包含“同意语句”和“模型名称”，比如：“I give my consent to use the <MODEL> voice cloning model with my voice（我同意使用<模型名称>语音克隆模型克隆我的声音）”。同时建议使用 **麦克风实时录音**，而不是上传音频文件，以防止使用之前录音剪辑过的语音。使用全新（从未说过的）句子也能进一步确保这个“同意”是针对当前情境、主动做出的、知情且明确的同意。

当然，这种设计不是万无一失的。理论上，人们依然可能用其他 TTS 系统来伪造这段“同意”语音。未来的版本可以进一步尝试音频来源验证、说话人嵌入相似度分析、或通过实时录音元数据来提升验证能力。

**关于“适合语音克隆的语句”部分：**
已有的语音克隆研究表明，用于训练模型的语句需要具备以下几个特点：
* **音素多样性**：语句中应包含多种元音和辅音，确保发音覆盖范围广，[参考文献](https://proceedings.neurips.cc/paper_files/paper/2018/file/6832a7b24bc06775d02b7406880b93fc-Paper.pdf)。
* **语气中性或礼貌**：语音应保持自然、平静或友善的语调，[参考文献](https://dl.acm.org/doi/10.5555/3666122.3666982)，避免情绪化表达。
* **录音环境安静，发音自然**：尽量避免背景噪音，并在说话人状态舒适时录制。
* **语音片段要有完整的起止**：录音剪辑时不能截断词语，要保留完整的一句话，确保语音首尾清晰。

为了实现这两个目标，在演示中我们使用语言模型自动生成一组句子：一句用于表达明确的同意，另一句则是中性内容，用于增加音素多样性（覆盖不同的元音、辅音和语调）。
每次生成时，系统会随机选择一个日常话题（如天气、美食或音乐），使句子内容丰富多样，也更自然好读，有助于录音清晰、自然，并具备良好的语音质量，同时包含明确的同意声明。
这个句子生成过程是 **自动完成** 的，而不是预先写好的，确保每位用户都会获得 **独一无二** 的句子组合，避免文本被重复使用，也确保每次录音都是针对当前会话场景所做出的具体同意。
换句话说，语言模型在每次“同意实例”中都会生成两句全新的句子：
* 一句表达明确的使用同意，
* 一句则用于增加语音中的音素多样性。

比如，模型可能会生成如下内容：
*“I give my consent to use my voice for generating audio with the model EchoVoice. The weather is bright and calm this morning.”*

这种做法确保了所有用于语音克隆的样本都具有 **可验证的明确同意**，同时也符合高质量语音合成所需的技术标准。
（注：生成句子的语言模型不必是“大型语言模型”，因为后者本身也可能涉及额外的同意问题。）


**更多例子：**

* *“I give my consent to use my voice for generating synthetic audio with the Chatterbox model today. My daily commute involves navigating through crowded streets on foot most days lately anyway.”*
* *“I give my consent to use my voice for generating audio with the model Chatterbox. After a gentle morning walk, I'm feeling relaxed and ready to speak freely now.”*
* *“I agree to the use of my recorded voice for audio generation with the model Chatterbox. The coffee shop outside has a pleasant aroma of freshly brewed coffee this morning.”*


### 解锁语音同意门

当说话人读出的语句与系统生成的文本完全匹配后，语音克隆系统便可启动，并使用这段“同意”语音作为训练输入。

目前已有几种实现方式，当然我们也很欢迎更多建议：

* **演示中提供的方式：** 同意门一旦开启，系统就可以直接进入语音克隆阶段，用户可输入任意文本，生成对应的合成语音。此时，模型会直接利用“同意”语音作为训练数据。
* **可选方案一：** 修改演示中的代码，使系统可以接受多个语音文件来建模用户的声音——比如用户授权使用网络上存在的录音。此时提示语和同意语句也需相应调整。
* **可选方案二：** 将同意录音保存下来，以便后续系统中用于生成任意语句。这可以通过 `huggingface_hub` 上传功能实现，[相关指南在此](https://huggingface.co/docs/huggingface_hub/en/guides/upload)。同样需要根据使用场景调整提示语和同意内容。

> [!TIP]
>
> ### [点此查看我们的演示！](https://huggingface.co/spaces/society-ethics/RepeatAfterMe)
>
> 你可以复制代码，自行调整使用。

该代码是模块化的，可以根据项目需求进行裁剪和改写。我们也正在持续优化系统的稳健性与安全性，欢迎提出改进建议。

只要负责任地使用，这项技术并不一定是“幽灵般”的存在。它完全可以成为人与机器之间 **相互尊重的协作工具** ——没有幽灵上身，只有良好规范的技术实践。🎃
