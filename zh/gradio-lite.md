---
title: "Gradio-Lite: 无需服务器即可实现外部访问的Gradio"
thumbnail: /blog/assets/167_gradio_lite/thumbnail.png
authors:
  - user: abidlabs
  - user: whitphx
  - user: aliabd
---

# Gradio-Lite: 无需服务器即可实现外部访问的Gradio

[toc]

Gradio是一个经常用于创建交互式机器学习应用的Python库，在传统意义上如果想对外分享Gradio应用，就需要依赖服务器设备和相关资源，而这对于追求应用程序的快速部署和共享的开发人员来说似乎并不友好，他们更希望通过简化的方式将应用程序置于外部环境中，并使其易于访问和使用。而这就是Gradio-lite (`@gradio/lite`) 旨在实现的目标：通过 [Pyodide](https://pyodide.org/en/stable/) 在浏览器中直接运行Gradio 库，这意味着Gradio应用不需要服务器即可实现外部访问。

在本文中，我们会详细介绍`@gradio/lite`是什么并给出相关示例代码，最后我们会讨论一下使用Gradio-lite运行Gradio应用程序所带来的优势。

## `@gradio/lite`是什么?

`@gradio/lite`是一个JavaScript库，可以使开发人员直接在Web浏览器中运行Gradio应用，它通过Pyodide完成在浏览器环境中执行构建Gradio应用的python代码，以此实现Gradio应用可作为一个网页进行分享和外部访问。Pyodide是一个可以在浏览器中运行Python代码的项目，它将Python解释器编译为WebAssembly字节码。通过`@gradio/lite`**编写常规的Python代码来创建Gradio应用程序**，可以**使它们在浏览器中运行并被外部访问**，而无需额外的服务器部署和管理。

## 动手试一试

让我们来构建一个在`@gradio/lite` 中的“Hello World” Gradio应用程序。

### 1. 导入JS和CSS

首先创建一个新的HTML文件，通过使用以下代码导入与`@gradio/lite` 包对应的JavaScript和CSS:


```html
<html>
	<head>
		<script type="module" crossorigin src="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.js"></script>
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.css" />
	</head>
</html>
```

值得一提的是应该使用可用的`@gradio/lite` 的最新版本，可以在[此处](https://www.jsdelivr.com/package/npm/@gradio/lite?tab=files)查看可用版本。

### 2. 创建`<gradio-lite>`标签

在你的HTML页面的正文中某处(无论你希望Gradio应用程序呈现在何处)，创建打开和关闭的`<gradio-lite>`标签。

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

注意：你可以将`theme`属性添加到`<gradio-lite>`标签中，以强制主题为深色或浅色(默认情况下它遵循系统主题)。例如:

```html
<gradio-lite theme="dark">
...
</gradio-lite>
```

### 3. 在标签内编写Gradio应用程序

现在就可以像平常一样用Python编写Gradio应用程序了！但是一定要注意，由于这是Pytho所以此空格和缩进很重要。

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

基本的流程就是这样！现在你应该能够在浏览器中打开HTML页面，并看到呈现的Gradio应用程序了！只不过由于Pyodide在浏览器中安装可能需要一些时间，因此初始加载Gradio应用程序可能需要一段时间。

**调试提示**：所有错误(包括 Python 错误)都将打印到Web 浏览器中的检查器中，所以如果要查看 Gradio-lite 应用程序中的任何错误，请移步检查器

## 功能扩展

如果想要创建一个跨多个文件或具有自定义 Python 需求的 Gradio 应用程序怎么办？通过 `@gradio/lite` 也可以实现!

### 多个文件

在 `@gradio/lite` 应用程序中添加多个文件非常简单：使用 `<gradio-file>` 标签。你可以创建任意多个 `<gradio-file>` 标签，但每个标签都需要一个 `name` 属性，Gradio 应用程序的入口点应有 `entrypoint` 属性。

下面是一个栗子:

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

### 其他需求

如果 Gradio 应用程序有其他需求，通常可以[使用micropip 在浏览器中安装它们](https://pyodide.org/en/stable/usage/loading-packages.html#loading-packages)。并且我们创建了一个封装使得这个过程更加便捷了，你只需用与 `requirements.txt` 相同的语法列出需求，并用 `<gradio-requirements>` 标签封闭它们即可。

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

**试一试**：你可以在 [这个免费的Hugging Face Static Space](https://huggingface.co/spaces/abidlabs/gradio-lite-classify) 中看到这个运行中的示例，它允许你托管免费的静态(无服务器)Web应用程序，即使离线你也能运行机器学习模型!

## 使用`@gradio/lite`的优势

### 1. 无服务器部署

`@gradio/lite` 的主要优势在于它消除了对服务器设备和相关资源的需求，这使得Gradio应用的部署更加轻松，减少了与服务器相关的成本，并使与他人共享 Gradio 应用程序变得更加容易。

### 2. 低延迟

通过在浏览器中运行，数据无需来回服务器，使得`@gradio/lite` 的交互延迟更低，从而实现更快的响应和更流畅的用户体验。

### 3. 隐私和安全性

由于所有处理均在用户的浏览器内进行，所以 `@gradio/lite`增强了隐私和安全性，用户数据保留在其个人设备上，让大家处理数据更加放心~

### 限制

* 目前,在浏览器中使用 `@gradio/lite` 的最大缺点就是 Gradio 应用程序通常需要一段加载时间(通常是5-15秒)进行初始化，这是因为浏览器需要加载 Pyodide，其运行后才能渲染 Python 代码。

* 并非所有 Python 包都受 Pyodide 支持。虽然 `gradio` 和许多其他流行包(包括 `numpy`、`scikit-learn` 和 `transformers-js`)都可以在 Pyodide 中安装，但如果你的应用程序有许多依赖项，那么最好检查一下这些依赖项是否包含在 Pyodide 中，如果不在可以通过 `micropip` 进行安装。

## 心动不如行动！

您可以通过在本地复制并粘贴此代码到 `index.html` 文件中,然后使用浏览器打开它,立即尝试 `@gradio/lite`:

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

我们还在 Gradio 网站上创建了一个 playground，你可以交互式的编辑代码并立即查看！

Playground: https://www.gradio.app/playground



![image-20231120215928859](https://evinci.oss-cn-hangzhou.aliyuncs.com/img/image-20231120215928859.png)
