---
title: "Gradio-Lite: 完全在浏览器里运行的无服务器 Gradio"
thumbnail: /blog/assets/167_gradio_lite/thumbnail.png
authors:
  - user: abidlabs
  - user: whitphx
  - user: aliabd
translators:
- user: zhongdongy
---

# Gradio-Lite: 完全在浏览器里运行的无服务器 Gradio

Gradio 是一个经常用于创建交互式机器学习应用的 Python 库。在以前按照传统方法，如果想对外分享 Gradio 应用，就需要依赖服务器设备和相关资源，而这对于自己部署的开发人员来说并不友好。

欢迎 Gradio-lite ( `@gradio/lite` ): 一个通过 [Pyodide](https://pyodide.org/en/stable/) 在浏览器中直接运行 Gradio 的库。在本文中，我们会详细介绍 `@gradio/lite` 是什么，然后浏览示例代码，并与您讨论使用 Gradio-lite 运行 Gradio 应用所带来的优势。

## `@gradio/lite` 是什么?

`@gradio/lite` 是一个 JavaScript 库，可以使开发人员直接在 Web 浏览器中运行 Gradio 应用，它通过 Pyodide 来实现这一能力。Pyodide 是可以将 Python 代码在浏览器环境中解释执行的 WebAssembly 专用 Python 运行时。有了 `@gradio/lite` ，你可以 **使用常规的 Python 代码编写 Gradio 应用** ，它将不再需要服务端基础设施，可以 **顺畅地在浏览器中运行** 。

## 开始使用

让我们用 `@gradio/lite` 来构建一个 "Hello World" Gradio 应用。

### 1. 导入 JS 和 CSS

首先如果没有现成的 HTML 文件，需要创建一个新的。添加以下代码导入与 `@gradio/lite` 包对应的 JavaScript 和 CSS:

```html
<html>
	<head>
		<script type="module" crossorigin src="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.js"></script>
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.css" />
	</head>
</html>
```

通常来说你应该使用最新版本的 `@gradio/lite` ，可以前往 [查看可用版本信息](https://www.jsdelivr.com/package/npm/@gradio/lite?tab=files)。

### 2. 创建`<gradio-lite>` 标签

在你的 HTML 页面的 `body` 中某处 (你希望 Gradio 应用渲染显示的地方)，创建开闭配对的 `<gradio-lite>` 标签。

```html
<html>
	<head>
		<script type="module" crossorigin src="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.js"></script>
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.css" />
	</head>
    
	<body>
		<gradio-lite>
		</gradio-lite>
	</body>
    
</html>
```

注意: 你可以将 `theme` 属性添加到 `<gradio-lite>` 标签中，从而强制使用深色或浅色主题 (默认情况下它遵循系统主题)。例如:

```html
<gradio-lite theme="dark">
...
</gradio-lite>
```

### 3. 在标签内编写 Gradio 应用

现在就可以像平常一样用 Python 编写 Gradio 应用了！但是一定要注意，由于这是 Python 所以空格和缩进很重要。

```html
<html>
	<head>
		<script type="module" crossorigin src="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.js"></script>
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.css" />
	</head>
	<body>
		<gradio-lite>
		import gradio as gr

		def greet(name):
			return "Hello, " + name + "!"
		
		gr.Interface(greet, "textbox"， "textbox").launch()
		</gradio-lite>
	</body>
</html>
```

基本的流程就是这样！现在你应该能够在浏览器中打开 HTML 页面，并看到刚才编写的 Gradio 应用了！只不过由于 Pyodide 需要花一些时间在浏览器中安装，初始加载 Gradio 应用可能需要一段时间。

**调试提示**: 所有错误 (包括 Python 错误) 都将打印到浏览器中的检查器控制台中，所以如果要查看 Gradio-lite 应用中的任何错误，请打开浏览器的检查器工具 (inspector)。

## 更多例子: 添加额外的文件和依赖

如果想要创建一个跨多个文件或具有自定义 Python 依赖的 Gradio 应用怎么办？通过 `@gradio/lite` 也可以实现!

### 多个文件

在 `@gradio/lite` 应用中添加多个文件非常简单: 使用 `<gradio-file>` 标签。你可以创建任意多个 `<gradio-file>` 标签，但每个标签都需要一个 `name` 属性，Gradio 应用的入口点应添加 `entrypoint` 属性。

下面是一个例子:

```html
<gradio-lite>

<gradio-file name="app.py" entrypoint>
import gradio as gr
from utils import add

demo = gr.Interface(fn=add, inputs=["number"， "number"]， outputs="number")

demo.launch()
</gradio-file>

<gradio-file name="utils.py" >
def add(a, b):
	return a + b
</gradio-file>

</gradio-lite>
```

### 额外的依赖项

如果 Gradio 应用有其他依赖项，通常可以 [使用 micropip 在浏览器中安装它们](https://pyodide.org/en/stable/usage/loading-packages.html#loading-packages)。我们创建了一层封装使得这个过程更加便捷了: 你只需用与 `requirements.txt` 相同的语法列出依赖信息，并用 `<gradio-requirements>` 标签包围它们即可。

在这里我们安装 `transformers_js_py` 来尝试直接在浏览器中运行文本分类模型！

```html
<gradio-lite>

<gradio-requirements>
transformers_js_py
</gradio-requirements>

<gradio-file name="app.py" entrypoint>
from transformers_js import import_transformers_js
import gradio as gr

transformers = await import_transformers_js()
pipeline = transformers.pipeline
pipe = await pipeline('sentiment-analysis')

async def classify(text):
	return await pipe(text)

demo = gr.Interface(classify, "textbox", "json")
demo.launch()
</gradio-file>

</gradio-lite>	
```

**试一试**: 你可以在 [这个 Hugging Face Static Space](https://huggingface.co/spaces/abidlabs/gradio-lite-classify) 中看到上述示例，它允许你免费托管静态 (无服务器) Web 应用。访问此页面，即使离线你也能运行机器学习模型!

## 使用 `@gradio/lite` 的优势

### 1. 无服务器部署

`@gradio/lite` 的主要优势在于它消除了对服务器基础设施的需求。这简化了 Gradio 应用的部署，减少了与服务器相关的成本，并且让分享 Gradio 应用变得更加容易。

### 2. 低延迟

通过在浏览器中运行，`@gradio/lite` 能够为用户带来低延迟的交互体验。因为数据无需与服务器往复传输，这带来了更快的响应和更流畅的用户体验。

### 3. 隐私和安全性

由于所有处理均在用户的浏览器内进行，所以 `@gradio/lite` 增强了隐私和安全性，用户数据保留在其个人设备上，让大家处理数据更加放心~

### 限制

- 目前, 使用 `@gradio/lite` 的最大缺点在于 Gradio 应用通常需要更长时间 (通常是 5-15 秒) 在浏览器中初始化。这是因为浏览器需要先加载 Pyodide 运行时，随后才能渲染 Python 代码。
- 并非所有 Python 包都受 Pyodide 支持。虽然 `gradio` 和许多其他流行包 (包括 `numpy` 、 `scikit-learn` 和 `transformers-js` ) 都可以在 Pyodide 中安装，但如果你的应用有许多依赖项，那么最好检查一下它们是否包含在 Pyodide 中，或者 [通过 `micropip` 安装](https://micropip.pyodide.org/en/v0.2.2/project/api.html#micropip.install)。

## 心动不如行动！

要想立刻尝试 `@gradio/lite` ，您可以复制并粘贴此代码到本地的 `index.html` 文件中，然后使用浏览器打开它:

```html
<html>
	<head>
		<script type="module" crossorigin src="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.js"></script>
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.css" />
	</head>
	<body>
		<gradio-lite>
		import gradio as gr

		def greet(name):
			return "Hello, " + name + "!"
		
		gr.Interface(greet, "textbox", "textbox").launch()
		</gradio-lite>
	</body>
</html>
```

我们还在 Gradio 网站上创建了一个 playground，你可以在那里交互式编辑代码然后即时看到结果！

Playground 地址: <https://www.gradio.app/playground>