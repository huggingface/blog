---
title: "Hugging Face 与 PyCharm 深度集成：轻松引入丰富的 AI 模型"
thumbnail: /blog/assets/pycharm-integration/thumbnail.png
authors:
- user: rocketknight1
translators:
- user: chenglu
---

# Hugging Face 与 PyCharm 深度集成：轻松引入丰富的 AI 模型

这是一个平平无奇的星期二早晨，作为一名 Transformers 库的维护者，我照例做着每天工作日早上都要做的事情：打开 [PyCharm](https://jb.gg/get-pycharm-hf)，加载 Transformers 代码库，充满感情地浏览 [聊天模板文档](https://huggingface.co/docs/transformers/main/chat_templating)，并努力“无视”当天有 50 个用户问题在等我处理。但今天有些不一样的地方：

![screenshot 0](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pycharm-integration/screenshot_0.png)

有什么……等等！我们放大看看！

![screenshot 1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pycharm-integration/screenshot_1.png)  

那是……？  

![screenshot 2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pycharm-integration/screenshot_2.png)  

给今天提出 issue 的用户说声抱歉，你们的问题显然可能不会得到回复了。因为我要聊聊聊聊 PyCharm 中的 Hugging Face 集成！

## Hugging Face 就在你最熟悉的地方

我可以通过列出功能来介绍这个集成，但那样未免有些乏味，况且我们还有 [文档](https://www.jetbrains.com/help/pycharm/hugging-face.html)。不如让我们通过实际操作来看看它的用法。假设我要写一个 Python 应用程序，我希望它能与用户进行聊天。不仅是文本聊天——用户还可以粘贴图片，并让应用程序自然地对图片进行讨论。

如果你对当前机器学习的前沿技术不太熟悉，这个需求可能会让你感到胆战心惊，但别害怕。只需在你的代码中右键点击，选择“插入 HF 模型”。然后会弹出一个对话框：

![dialog_box_screenshot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pycharm-integration/dialog_box_screenshot.png)

能够处理图像和文本聊天的模型称为“image-text-to-text”类型：用户可以提供图像和文本，模型则输出文本。在左侧下拉菜单中找到它。默认情况下，模型列表按点赞数排序——不过要记住，老模型往往会积累很多点赞数，即使它们可能不再是最先进的了。

我们可以通过模型名称下方的更新时间查看模型的更新日期。让我们选择一个既新又受欢迎的模型：`microsoft/Phi-3.5-vision-instruct`。

对于某些模型类别，你可以点击“使用模型”按钮，让系统自动在你的笔记本中插入一些基础代码。不过，更好的方法通常是浏览右侧的模型卡片，复制其中的示例代码。对话框右侧显示的模型卡片和 Hugging Face Hub 上的完全一致。让我们复制示例代码并粘贴到我们的代码中！

![code_snippet_screenshot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pycharm-integration/code_snippet_screenshot.png)  

你公司的网络安全人员可能会对你直接复制一大段陌生代码并运行感到不满，但如果他们抱怨，你只需“无视”他们，然后继续运行代码。

看吧：我们现在有了一个能够愉快聊天的模型——在这个例子中，它能读取并评论一份微软的演示幻灯片截图。你可以随意试试看这个例子，尝试你的对话或自己的图片。一旦成功运行，只需将这段代码封装进一个类中，你的应用就可以用了。这样，我们在十分钟内获得了最先进的开源机器学习功能，连浏览器都没打开过。

> **提示**  
> 这些模型可能很大！如果遇到内存不足的错误，可以尝试使用更大内存的 GPU，或者减少示例代码中的 20。你也可以去掉 `device_map="cuda"`，把模型放到 CPU 内存中，虽然速度会变慢。

## 即时模型卡

接下来，我们换个视角。假设你不是这段代码的作者，而是一个需要审查代码的同事。也许你是那个因为被无视了而且还在生气的网络安全人员。你看到这段代码，完全不知道自己在看什么。别慌——只需将鼠标悬停在模型名称上，整个模型卡片会立刻弹出来。你可以快速验证模型的来源及其预期用途。

（如果你是那种两周后就忘记自己写过什么代码的人，这个功能也非常有用）

![model_card_screenshot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pycharm-integration/model_card_screenshot.png)

## 本地模型缓存

你可能会注意到，第一次运行代码时，模型需要下载，但之后加载速度就快多了。模型已经被存储在你的本地缓存中了。还记得之前那个神秘的小 🤗 图标吗？点击它，你就能看到缓存中的所有内容：

![model_cache_screenshot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pycharm-integration/model_cache_screenshot.png)

这个功能非常方便，可以查看你正在使用的模型，并在不再需要时清理以节省磁盘空间。对于两周后的“记忆缺失”情景，这也很有帮助——如果你不记得当时用的模型是什么，很可能就在这里。不过要记住，2024 年大多数有用的、适合生产的模型都超过 1GB，因此缓存很快会被填满！

## 人工智能时代的 Python

在 Hugging Face，我们认为开源人工智能是开源哲学的自然延伸：开源软件解决开发者和用户的问题，为他们提供可以集成到代码中的新能力，而开源模型也提供了同样的便利。人们往往容易被复杂性迷惑，过分关注实现细节，因为一切都如此新奇有趣，但模型的存在是为了 **为你解决问题**。如果抽象掉架构和训练的细节，模型本质上是 **函数** ——你代码中的工具，可以将某种输入转换成某种输出。

因此，这些功能是非常合适的补充。正如 IDE 已经能为你显示函数签名和文档字符串，它们现在也能为你展示示例代码和模型卡。像这样的集成可以让你像导入其他库一样方便地导入聊天或图像识别模型。我们相信这就是代码未来的发展方向，希望这些功能能对你有所帮助！

**[下载 PyCharm](https://jb.gg/get-pycharm-hf) 并体验 Hugging Face 集成。**

**使用代码 PyCharm4HF 获取免费 3 个月的 PyCharm 订阅 [点击这里](http://jetbrains.com/store/redeem/)。**  
