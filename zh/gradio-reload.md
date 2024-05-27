---
title: "使用 Gradio 的“热重载”模式快速开发 AI 应用"
thumbnail: /blog/assets/gradio-reload/thumbnail_compressed.png
authors:
- user: freddyaboulton
translators:
- user: chenglu
---

# 使用 Gradio 的“热重载”模式快速开发 AI 应用

在这篇文章中，我将展示如何利用 Gradio 的“热重载”模式快速构建一个功能齐全的 AI 应用。但在进入正题之前，让我们先了解一下什么是热重载模式以及 Gradio 为什么要采用自定义的自动重载逻辑。如果你已熟悉 Gradio 并急于开始构建，请直接跳转到第三部分[构建文档分析应用](#building-a-document-analyzer-application)。

## 热重载模式具体是做什么的？

简而言之，热重载模式可以在不重启 Gradio 服务器的情况下，自动引入你源代码中的最新更改。如果这听起来还有些模糊，不妨继续阅读。

Gradio 是一个广受欢迎的 Python 库，专门用于创建交互式机器学习应用。开发者可以完全在 Python 中设计 UI 布局，并嵌入一些 Python 逻辑来响应 UI 事件。如果你已经掌握了 Python 基础，那么学习 Gradio 将会非常轻松。如果你对 Gradio 还不太熟悉，建议你查看这个[快速入门指南](https://www.gradio.app/guides/quickstart)。

通常，Gradio 应用像运行任何其他 Python 脚本一样启动，只需执行 `python app.py`（Gradio 代码文件可以任意命名）。这会启动一个 HTTP 服务器，渲染你的应用 UI 并响应用户操作。如果需要修改应用，通常会停止服务器（通常使用 `Ctrl + C`），编辑源文件后重新运行脚本。

开发过程中频繁停止和重启服务器会造成明显的延迟。如果能有一种方式能自动更新代码变更并即刻测试新思路，那将大为便利。

这正是 Gradio 的热重载模式的用武之地。你只需运行 `gradio app.py` 而不是 `python app.py`，即可在热重载模式下启动应用！

## Gradio 为何要自行实现重载逻辑？

Gradio 应用通常与 [uvicorn](https://www.uvicorn.org/)（一个 Python Web 框架的异步服务器）一同运行。尽管 Uvicorn 提供了[自动重载功能](https://www.uvicorn.org/)，但 Gradio 出于以下原因自行实现了重载逻辑：

1. **更快速的重载**：Uvicorn 的自动重载功能虽快于手动操作，但在开发 Gradio 应用时仍显缓慢。Gradio 开发者在 Python 中构建其 UI，因此他们希望在进行更改后能立即看到更新的 UI，这在 Javascript 生态中已是常态，但在 Python 中则较为新颖。
2. **选择性重载**：Gradio 应用属于 AI 应用，通常需要将 AI 模型加载到内存或连接到数据存储（如向量数据库）。开发过程中重启服务器将导致模型重新加载或重新连接数据库，这会在开发周期间引入不必要的延迟。为解决此问题，Gradio 引入了 `if gr.NO_RELOAD:` 代码块，你可以利用它标记不需重载的代码部分。这种做法只有在 Gradio 实现了自定义重载逻辑的情况下才可行。

接下来，我将展示如何利用 Gradio 的热重载模式迅速开发一个 AI 应用。

## 构建文档分析应用

本应用将允许用户上传文档图片并提出问题，随后以自然语言形式获得答案。我们将利用免费的 [Hugging Face 推理 API](https://huggingface.co/docs/huggingface_hub/guides/inference)，你可以在自己的电脑上轻松操作，无需 GPU！

首先，让我们在名为 `app.py` 的文件中输入以下代码，并通过执行 `gradio app.py` 在热重载模式下启动它：

```python
import gradio as gr

demo = gr.Interface(lambda x: x, "text", "text")

if __name__ == "__main__":
    demo.launch()
```

这会创建以下简单的用户界面。

![简单界面 UI](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-reload/starting-demo.png)

鉴于我希望用户能够上传图像文件及其问题，我将输入组件更改为 `gr.MultimodalTextbox()`。注意用户界面是如何立即更新的！

![具有多模态文本框的简单界面](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-reload/change_to_multimodal_tb.gif)

虽然这个用户界面已经可以工作，但我认为如果输入文本框位于输出文本框下方会更合适。我可以通过使用 `Blocks` API 来实现这一点，并且我还通过添加占位符文本来定制输入文本框，以引导用户。

![切换到 Blocks](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-reload/switch_to_blocks.gif)

现在 UI 已经令人满意，我将开始实现 `chat_fn` 的逻辑。

我将使用 Hugging Face 的推理 API，因此我需要从 `huggingface_hub` 包中导入 `InferenceClient`（预装在 Gradio 中）。我将使用 [`impira/layouylm-document-qa`](https://huggingface.co/impira/layoutlm-document-qa) 模型来回答用户的问题，然后使用 [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) 大语言模型提供自然语言回答。

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

def chat_fn(multimodal_message):
    question = multimodal_message["text"]
    image = multimodal_message["files"][0]
    
    answer = client.document_question_answering(image=image, question=question, model="impira/layoutlm-document-qa")
    
    answer = [{"answer": a.answer, "confidence": a.score} for a in answer]
   
    user_message = {"role": "user", "content": f"Question: {question}, answer: {answer}"}
   
    message = ""
    for token in client.chat_completion(messages=[user_message],
                           max_tokens=200, 
                           stream=True,
                           model="HuggingFaceH4/zephyr-7b-beta"):
        if token.choices[0].finish_reason is not None:
           continue
        message += token.choices[0].delta.content
        yield message
```

这是我们的应用演示！

![演示我们的应用](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-reload/demo_1.gif?download=true)

我还会添加一个系统消息，以便大语言模型保持回答简短，不包括原始置信度分数。为避免每次更改时都重新实例化 `InferenceClient`，我将其放在不需重载的代码块中。

```python
if gr.NO_RELOAD:
    client = InferenceClient()

system_message = {
    "role": "system",
    "content": """
You are a helpful assistant.
You will be given a question and a set of answers along with a confidence score between 0 and 1 for each answer.
You job is to turn this information into a short, coherent response.

For example:
Question: "Who is being invoiced?", answer: {"answer": "John Doe", "confidence": 0.98}

You should respond with something like:
With a high degree of confidence, I can say John Doe is being invoiced.

Question: "What is the invoice total?", answer: [{"answer": "154.08", "confidence": 0.75}, {"answer": "155", "confidence": 0.25}

You should respond with something like:
I believe the invoice total is $154.08 but it can also be $155.
"""}
```

这是我们演示的现在情况！系统消息确实帮助保持了机器人的回答简短而不包含长的小数。

![应用演示带系统消息](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-reload/demo_3.gif)

作为最终改进，我将在页面上添加一个 Markdown 标题：

![添加标题](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-reload/add_a_header.gif)

## 结语

在本文中，我使用 Gradio 和 Hugging Face 推理 API 开发了一个实用的 AI 应用。从开发初期，我就不确定最终产品会是什么样子，所以能够即时重新加载 UI 和服务器逻辑让我能迅速尝试各种想法。整个应用的开发过程大约只用了一个小时！

如果你想了解此演示的完整代码，请访问这个 [Space 应用](https://huggingface.co/spaces/freddyaboulton/document-analyzer)！