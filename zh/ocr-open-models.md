---
title: "用开源模型强化你的 OCR 工作流"
thumbnail: /blog/assets/ocr-open-models/thumbnail.png
authors:
- user: merve
- user: ariG23498
- user: davanstrien
- user: hynky
- user: andito
- user: reach-vb
- user: pcuenq
translators:
- user: chenglu
---



# 用开源模型强化你的 OCR 工作流

> [!提示]
> 我们在这篇文章中新增了 [Chandra](https://huggingface.co/datalab-to/chandra) 和 [OlmOCR-2](https://huggingface.co/allenai/olmOCR-2-7B-1025)，并附上了它们在 OlmOCR 基准上的得分 🫡

**摘要：**
强大的视觉语言模型（Vision-Language Models, VLMs）的崛起，正在彻底改变文档智能（Document AI）的格局。每种模型都有其独特的优势，因此选择合适的模型变得棘手。相比闭源模型，开源权重的模型在成本效率和隐私保护上更具优势。为了帮助你快速上手，我们整理了这份指南。

在本指南中，你将了解到：

* 当前 OCR 模型的整体格局及其能力
* 何时需要微调模型，何时可直接使用
* 为你的场景选择合适模型时应考虑的关键因素
* 如何超越传统 OCR，探索多模态检索与文档问答

读完之后，你将知道如何选择合适的 OCR 模型、开始构建应用，并对文档 AI 有更深入的理解。让我们开始吧！



## 目录

* [使用开源模型提升你的 OCR 流水线能力](#使用开源模型提升你的-ocr-流水线能力)

  * [现代 OCR 简介](#现代-ocr-简介)

    * [模型能力](#模型能力)

      * [文字识别](#文字识别)
      * [处理文档中的复杂组件](#处理文档中的复杂组件)
      * [输出格式](#输出格式)
      * [本地性意识（Locality Awareness）](#本地性意识-locality-awareness)
      * [模型提示词能力](#模型提示词能力)
  * [前沿开源 OCR 模型](#前沿开源-ocr-模型)

    * [模型对比](#模型对比)
    * [模型评估](#模型评估)

      * [评测基准](#评测基准)
      * [性价比](#性价比)
      * [开源 OCR 数据集](#开源-ocr-数据集)
  * [模型运行工具](#模型运行工具)

    * [本地运行](#本地运行)
    * [远程运行](#远程运行)
  * [超越 OCR 的能力](#超越-ocr-的能力)

    * [视觉文档检索器](#视觉文档检索器)
    * [基于视觉语言模型的文档问答](#基于视觉语言模型的文档问答)
  * [总结](#总结)



## 现代 OCR 简介

光学字符识别（Optical Character Recognition，简称 OCR）是计算机视觉领域最早、也是持续时间最长的研究方向之一。AI 的许多早期实际应用都集中在“将印刷文字转化为可编辑的数字文本”上。

随着[视觉语言模型（Vision-Language Models, VLMs）](https://huggingface.co/blog/vlms)的兴起，OCR 的能力迎来了飞跃式提升。如今，许多 OCR 模型都是在现有 VLM 的基础上进行微调得到的。但现代模型的能力已远超传统 OCR —— 你不仅可以识别文字，还能基于内容检索文档，甚至直接进行问答。

得益于更强大的视觉理解能力，这些模型能处理低质量扫描件、理解复杂元素（如表格、图表、图片等），并将文本与视觉内容融合，以回答跨文档的开放式问题。



### 模型能力

#### 文本识别

最新的模型能够将图像中的文字转录为机器可读格式。输入内容可能包括：

* 手写文字
* 各类文字体系（如拉丁文、阿拉伯文、日文等）
* 数学公式
* 化学方程式
* 图片、版面或页码标签

OCR 模型会将这些内容转换为机器可读的文本，输出格式多种多样，比如 **HTML、Markdown** 等。



#### 处理文档中的复杂组件

除了文字，某些模型还能识别：

* 图片
* 图表
* 表格

部分模型能识别文档中图片的精确位置，提取其坐标，并在输出中将图片嵌入对应位置。
另一些模型还能为图片生成说明文字（caption），并在适当位置插入。这对于后续将机器可读输出传入 LLM（大型语言模型）尤为有用。

例如，[OlmOCR（AllenAI 出品）](https://huggingface.co/allenai/olmOCR-7B-0825) 和 [PaddleOCR-VL（PaddlePaddle 出品）](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) 就是代表。

不同模型使用不同的输出格式，例如 **DocTags**、**HTML**、**Markdown**（后文 *输出格式* 一节有详细说明）。
模型处理表格与图表的方式通常取决于所采用的输出格式：

* 有些模型将图表当作图片直接保留；
* 有些模型则会将其转换为可解析的结构化格式，如 Markdown 表格或 JSON。
  例如，下图展示了一个柱状图如何被转换成机器可读的形式：

![Chart Rendering](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/chart-rendering.png)

同样地，表格中的单元格也会被解析为机器可读格式，并保留列名与标题的上下文关系：

![Table Rendering](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/table-rendering.png)



#### 输出格式

不同 OCR 模型采用的输出格式不同，以下是几种主流格式的简介：

* **DocTag：** 一种类似 XML 的文档标记格式，可表达位置信息、文本样式、组件层级等。下图展示了一篇论文如何被解析为 DocTags。该格式由开源的 Docling 模型使用。

  ![DocTags](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/doctags_v2.png)

* **HTML：** 是最常见的文档解析格式之一，能较好地表达结构与层级信息。

* **Markdown：** 人类可读性最强，格式简洁，但表达能力有限（如无法准确表示多列表格）。

* **JSON：** 通常用于表示表格或图表中的结构化信息，而非完整文档。

选择合适的模型，取决于你对输出结果的用途：

| 目标场景                | 推荐格式                            |
| ------------------- | ------------------------------- |
| **数字化重建**（重现原始文档版式） | 使用保留布局的格式，如 DocTags 或 HTML      |
| **LLM 输入或问答场景**     | 使用输出 Markdown 和图像说明的模型（更接近自然语言） |
| **程序化处理**（如数据分析）    | 选择能输出结构化 JSON 的模型               |



#### OCR 的位置感知

文档常常结构复杂，比如多栏文本、浮动图片、脚注等。早期的 OCR 模型通常先识别文字，再通过后处理手动推断页面布局，以恢复阅读顺序——这种方式既脆弱又易错。

现代 OCR 模型则会在输出中直接包含版面布局信息（称为 **“锚点”或 “grounding”**），如文字的边界框（bounding box）。
这种“锚定”机制能有效保持阅读顺序与语义连贯性，同时减少“幻觉式识别”（即错误生成内容）。



#### 模型提示

OCR 模型通常接收图像输入，并可选地接受文字提示（prompt），这取决于模型的架构与预训练方式。

部分模型支持**基于提示的任务切换**，例如 [granite-docling](https://huggingface.co/ibm-granite/granite-docling-258M) 可以通过不同提示词执行不同任务：

* 输入 “Convert this page to Docling” → 将整页转换为 DocTags；
* 输入 “Convert this formula to LaTeX” → 将页面中的公式转换为 LaTeX。

而另一些模型则只能处理整页内容，任务由系统提示固定定义。
例如，[OlmOCR（AllenAI）](https://huggingface.co/collections/allenai/olmocr-67af8630b0062a25bf1b54a1) 使用一个长系统提示词进行推理。OlmOCR 本质上是基于 Qwen2.5VL 微调的 OCR 模型，虽然它也能处理其他任务，但在 OCR 场景之外性能会明显下降。



## 前沿开源 OCR 模型

过去一年，我们见证了 OCR 模型领域的爆发式创新。由于开源生态的推动，不同团队之间可以相互借鉴、迭代，从而加速了技术进步。一个典型例子是 AllenAI 发布的 **OlmOCR**，它不仅开源了模型本身，还公开了训练所用的数据集，为他人提供了可复现与可扩展的基础。
这个领域正以前所未有的速度发展，但如何选择最合适的模型，仍然是一个不小的挑战。



### 最新模型对比

为了帮助大家更清晰地了解当前格局，以下是一些当前主流开源 OCR 模型的非完整对比。
这些模型都具备版面理解能力（layout-aware），能解析表格、图表与数学公式。
各模型支持的语言范围可在其 model card 中查看。除 **Chandra**（OpenRAIL 许可）与 **Nanonets**（许可证不明）外，其余均为开源许可。

表格中展示的平均得分来自 **Chandra** 与 **OlmOCR** 模型卡中在 **OlmOCR Benchmark**（仅英文）上的测试结果。
此外，许多模型基于 **Qwen2.5-VL** 或 **Qwen3-VL** 微调，因此我们也附上了 Qwen3-VL 作为参考。

| 模型名称                                                                                                   | 输出格式                         | 特性                                                       | 模型大小 | 是否多语言            | OlmOCR 基准平均得分 |
| :----------------------------------------------------------------------------------------------------- | :--------------------------- | :------------------------------------------------------- | :--- | :--------------- | :------------ |
| [Nanonets-OCR2-3B](https://huggingface.co/collections/nanonets/nanonets-ocr2-68ed207f17ee6c31d226319e) | 结构化 Markdown（含语义标注、HTML 表格等） | 图片自动生成说明<br>可提取签名与水印<br>识别复选框、流程图、手写体                    | 4B   | ✅ 英语、中文、法语、阿拉伯语等 | N/A           |
| [PaddleOCR-VL](https://huggingface.co/collections/PaddlePaddle/paddleocr-vl-68f0db852483c7af0bc86849)  | Markdown、JSON、HTML 表格与图表     | 支持手写体与旧文档<br>支持提示词输入<br>可将表格与图表转换为 HTML<br>可直接提取并插入图片    | 0.9B | ✅ 支持 109 种语言     | N/A           |
| [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)                                              | Markdown、JSON                | 支持 grounding<br>可提取并插入图片<br>支持手写体                        | 3B   | ✅ 多语言（具体未说明）     | 79.1 ± 1.0    |
| [OlmOCR-2](https://huggingface.co/allenai/olmOCR-2-7B-1025)                                            | Markdown、HTML、LaTeX          | 具备 grounding 能力<br>优化了大规模批处理性能                           | 8B   | ❎ 仅英语            | 82.3 ± 1.1    |
| [Granite-Docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)                        | DocTags                      | 支持基于提示的任务切换<br>可指定元素位置<br>输出内容丰富                         | 258M | ✅ 英语、日语、阿拉伯语、中文  | N/A           |
| [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)                                        | Markdown、HTML                | 支持通用视觉理解<br>能将图表、表格完整渲染为 HTML<br>识别手写体<br>内存高效，图像文字识别能力强 | 3B   | ✅ 近 100 种语言      | 75.4 ± 1.0    |
| [Chandra](https://huggingface.co/datalab-to/chandra)                                                   | Markdown、HTML、JSON           | 具备 grounding 能力<br>能原样提取并插入图片                            | 9B   | ✅ 支持 40+ 种语言     | 83.1 ± 0.9    |
| [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)                                           | 任意格式输出（多模态语言模型）              | 识别古文文本<br>支持手写体<br>图片可原样提取插入                             | 9B   | ✅ 支持 32 种语言      | N/A           |

> **注：**
> Qwen3-VL 是一款强大的通用视觉语言模型，支持多种文档理解任务，但并未针对 OCR 任务进行特别优化。
> 其他模型多采用固定提示词进行微调，专为 OCR 任务设计。
> 因此若使用 Qwen3-VL，建议尝试不同提示词以获得更佳效果。

你可以通过这个 [在线演示](https://prithivMLmods-Multimodal-OCR3.hf.space) 体验部分最新模型并比较输出效果：

<iframe  
    src="https://prithivMLmods-Multimodal-OCR3.hf.space"  
    frameborder="0"  
    width="850"  
    height="450"  
></iframe>



### 模型评估

#### 基准测试

没有任何一款模型能在所有场景中都是“最优”。
例如：表格应以 Markdown 还是 HTML 呈现？哪些元素需要提取？如何量化文本识别准确度？👀
这些都取决于具体任务。
目前已有多个公开评测集与工具，但仍无法覆盖所有情况。
我们推荐以下常用的评测基准：

1. **[OmniDocBenchmark](https://huggingface.co/datasets/opendatalab/OmniDocBench)**

   * 这是目前使用最广泛的文档识别基准之一。
   * 覆盖文档类型丰富：书籍、杂志、教材等。
   * 支持多格式（HTML 与 Markdown）表格评测。
   * 使用新型算法评估阅读顺序；公式会在评估前标准化。
   * 指标基于“编辑距离”或“树编辑距离”（表格部分）。
   * 标注数据部分由 SoTA VLM 或传统 OCR 生成。

2. **[OlmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench)**

   * 采用“单元测试式”评估方式。
   * 例如：表格评估通过验证单元格间关系完成。
   * 数据源为公开 PDF，标注来自多种闭源 VLM。
   * 特别适合评估英文 OCR 模型。

3. **[CC-OCR (Multilingual)](https://huggingface.co/datasets/wulipc/CC-OCR)**

   * 与前两者相比，CC-OCR 的文档质量与多样性较低。
   * 但它是**唯一**涵盖英语与中文以外语言的多语言评测集。
   * 图片多为低质量拍摄，文本较少。
   * 尽管不完美，但目前仍是最佳的多语言评估选项。

在不同文档类型、语言与任务场景下，模型表现差异明显。
如果你的业务领域不在现有评测集中体现，我们建议收集代表性样本，构建自定义测试集，比较不同模型在你的特定任务上的效果。



#### 成本与效率

大多数 OCR 模型的规模在 **3B～7B 参数**之间，也有一些小型模型（如 PaddleOCR-VL 仅 0.9B）。
成本不仅与模型大小相关，还取决于是否支持高效推理框架。

例如：

* **OlmOCR-2** 提供 vLLM 与 SGLang 实现。

  * 若在 H100 GPU（$2.69/小时）上运行，推理成本约为 **每百万页 $178**。
* **DeepSeek-OCR** 能在一块 40GB A100 上每天处理 **20 万页以上**。

  * 以此估算，其成本与 OlmOCR 大致相当（视 GPU 供应商而定）。

若任务对精度要求不高，还可选择 **量化版本（Quantized Models）**，进一步降低成本。
总体而言，开源模型在大规模部署时几乎总比闭源方案更经济。



#### 开源 OCR 数据集

尽管近年来开源 OCR 模型大量涌现，但公开的训练与评测数据集仍相对稀缺。
一个例外是 AllenAI 的 [olmOCR-mix-0225](https://huggingface.co/datasets/allenai/olmOCR-mix-0225)，
截至目前，该数据集已被用于训练至少 [72 个模型](https://huggingface.co/models?dataset=dataset:allenai/olmOCR-mix-0225)（可能更多）。

更广泛的数据共享将极大推动开源 OCR 的进步。
以下是几种常见的数据集构建方式：

* **合成数据生成（Synthetic Data Generation）**
  例如：[isl_synthetic_ocr](https://huggingface.co/datasets/Sigurdur/isl_synthetic_ocr)
* **VLM 自动转录**，再经人工或启发式过滤
* **利用现有 OCR 模型生成新训练数据**，以训练更高效的领域专用模型
* **基于人工校正语料的再利用**，如 [英国印度医学史数据集](https://huggingface.co/NationalLibraryOfScotland)，
  其中包含大量人工修正的历史文档 OCR

值得注意的是，许多此类数据集已存在但尚未“训练化”（training-ready）。
若能系统化整理并公开，将为开源社区释放巨大潜力。



## 模型运行工具

我们收到许多关于“如何开始使用 OCR 模型”的问题，因此这里总结了几种简单的方式——
包括在本地运行推理，或通过 Hugging Face 进行远程托管。



### 本地运行

目前大多数先进 OCR 模型都提供 **vLLM** 支持，并可通过 **transformers** 库直接加载推理。
你可以在各模型的 Hugging Face 页面找到具体说明。
下面我们以 **vLLM 推理方式**为例演示基本流程。



#### 使用 vLLM 启动服务

```shell
vllm serve nanonets/Nanonets-OCR2-3B
```

然后，你可以通过 OpenAI SDK 进行调用，例如：

```py
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1")
model = "nanonets/Nanonets-OCR2-3B"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def infer(img_base64):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Extract the text from the above document as if you were reading it naturally.",
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=15000
    )
    return response.choices[0].message.content

img_base64 = encode_image(your_img_path)
print(infer(img_base64))
```



#### 使用 Transformers 运行推理

Transformers 库提供了标准化的模型定义与接口，可轻松进行推理或微调。
模型可能有两种加载方式：

1. **官方实现**（在 transformers 内定义）
2. **remote code 实现**（由模型作者定义，允许 transformers 自动加载）

以下示例展示了如何用 transformers 调用 **Nanonets OCR 模型**：

```py
# 安装依赖：flash-attn 和 transformers
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "nanonets/Nanonets-OCR2-3B", 
    torch_dtype="auto", 
    device_map="auto", 
    attn_implementation="flash_attention_2"
)
model.eval()
processor = AutoProcessor.from_pretrained("nanonets/Nanonets-OCR2-3B")

def infer(image_url, model, processor, max_new_tokens=4096):
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

result = infer(image_path, model, processor, max_new_tokens=15000)
print(result)
```



#### 使用 MLX（适用于 Apple 芯片）

**MLX** 是苹果推出的机器学习框架，专为 **Apple Silicon (M 系列)** 设计。
在此基础上构建的 [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) 能轻松运行视觉语言模型。
你可以在 [Hugging Face](https://huggingface.co/models?sort=trending&search=ocr) 搜索所有支持 MLX 的 OCR 模型（包括量化版本）。

安装 MLX-VLM：

```bash
pip install -U mlx-vlm
```

示例运行：

```bash
wget https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/throughput_smolvlm.png

python -m mlx_vlm.generate \
  --model ibm-granite/granite-docling-258M-mlx \
  --max-tokens 4096 \
  --temperature 0.0 \
  --prompt "Convert this chart to JSON." \
  --image throughput_smolvlm.png 
```



### 远程运行

#### 使用 Inference Endpoints 部署模型（托管推理服务）

你可以通过 **Hugging Face Inference Endpoints** 在托管环境中部署兼容 vLLM 或 SGLang 的 OCR 模型。
该服务提供 GPU 加速、自动伸缩、监控与安全托管，无需自行维护基础设施。

部署步骤如下：

1. 进入模型仓库 [`nanonets/Nanonets-OCR2-3B`](https://huggingface.co/nanonets/Nanonets-OCR2-3B)

2. 点击页面上的 **“Deploy”** 按钮，选择 **“HF Inference Endpoints”**

   ![Inference Endpoints](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/IE.png)

3. 在弹出的窗口中配置部署参数（GPU 类型、实例数量等）

   ![Inference Endpoints](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/IE2.png)

4. 部署完成后，你可以直接通过上文示例中的 OpenAI 客户端脚本调用该 Endpoint。

更多信息可参阅官方文档：
👉 [Inference Endpoints (vLLM)](https://huggingface.co/docs/inference-endpoints/engines/vllm)



#### 使用 Hugging Face Jobs 进行批量推理

对于 OCR 场景，往往需要**批量处理成千上万张图像**。
这类任务可通过 **vLLM 的离线推理模式** 实现高效并行。

为了简化流程，我们创建了 [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr)，
它是一组适配 Hugging Face Jobs 的可直接运行脚本，能实现：

* 对数据集列中的所有图片进行批量 OCR
* 将 OCR 结果以 Markdown 形式新增为新列
* 自动将带结果的数据集回传到 Hub

例如，处理 100 张图片的命令如下：

```bash
hf jobs uv run --flavor l4x1 \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
  your-input-dataset your-output-dataset \
  --max-samples 100
```

这些脚本会自动处理所有 vLLM 配置与批次推理逻辑，
让批量 OCR 变得无需 GPU 或复杂部署。



## 超越 OCR

如果你对文档智能（Document AI）感兴趣，不仅仅局限于文字识别（OCR），以下是我们的一些推荐方向。



### 视觉文档检索

**视觉文档检索（Visual Document Retrieval）** 指的是：
当你输入一条文本查询时，系统能够从大量 PDF 文档中直接检索出最相关的前 *k* 篇。

与传统文本检索模型不同，视觉文档检索器直接在“文档图像”层面进行搜索。
除了独立使用外，你还可以将它与视觉语言模型结合，构建 **多模态 RAG（Retrieval-Augmented Generation）** 管线。
相关示例可参考：[ColPali + Qwen2_VL 多模态 RAG 教程](https://huggingface.co/merve/smol-vision/blob/main/ColPali_%2B_Qwen2_VL.ipynb)。

你可以在 [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=visual-document-retrieval&sort=trending) 找到所有可用的视觉文档检索模型。

目前主流的视觉检索器分为两类：

| 类型                              | 特点                  | 适用场景         |
| ------------------------------- | ------------------- | ------------ |
| **单向量模型（Single-vector Models）** | 内存效率高、速度快，但性能略弱     | 轻量化部署、大规模索引  |
| **多向量模型（Multi-vector Models）**  | 表征能力强、检索精度高，但占用显存更大 | 高精度检索、知识密集任务 |

大多数此类模型都支持 **vLLM** 和 **transformers**，因此你可以很方便地用它们进行向量索引，然后结合向量数据库（vector DB）执行高效搜索。



### 基于视觉语言模型的文档问答（Document Question Answering）

如果你的任务目标是**基于文档回答问题**（而不是仅仅提取文字），
你可以直接使用经过文档任务训练的**视觉语言模型（VLM）**。

许多用户习惯于：

1. 先将文档转换成纯文本；
2. 再把文本传入 LLM 进行问答。

这种方式虽然可行，但存在明显缺陷：

* 一旦文档布局复杂（如多栏结构、图表、图片说明等），转换后的文本就可能丢失关键信息；
* 图表被转为 HTML、图片说明生成错误时，LLM 就会误判或忽略内容。

因此，更好的做法是：
直接将**原始文档图像 + 用户问题** 一起输入支持多模态理解的模型，
例如 [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe)。
这样模型就能同时利用视觉与文本信息，不会错过任何上下文细节。



## 总结

在这篇文章中，我们为你概览了现代 OCR 技术的核心要点，包括：

* 如何选择合适的 OCR 模型
* 当前最前沿的开源模型及其能力
* 在本地或云端运行模型的工具
* 以及如何在 OCR 之上构建更复杂的文档智能应用

如果你希望进一步深入了解 OCR 与视觉语言模型（VLM），
以下是我们推荐的延伸阅读与教程资源 👇



### 延伸阅读与资源

* 📘 [Vision Language Models Explained（视觉语言模型详解）](https://huggingface.co/blog/vlms)
  —— 深入理解 VLM 的工作原理与发展历程。

* 🧠 [Vision Language Models 2025 Update（2025 年视觉语言模型更新）](https://huggingface.co/blog/vlms-2025)
  —— 最新 VLM 技术进展总结。

* 🔍 [PP-OCR-v5 技术博客](https://huggingface.co/blog/baidu/ppocrv5)
  —— 来自百度的高性能 OCR 系统优化介绍。

* 🧩 [教程：微调 Kosmos2.5 进行 Grounded OCR](https://huggingface.co/merve/smol-vision/blob/main/Grounded_Fine_tuning.ipynb)
  —— 实践指南，教你如何让模型具备“锚定式”识别能力。

* 📄 [教程：在 DocVQA 数据集上微调 Florence-2](https://huggingface.co/merve/smol-vision/blob/main/Fine_tune_Florence_2.ipynb)
  —— 基于视觉问答任务的微调实例。

* 📱 [在设备端实现 SOTA OCR（Core ML + dots.ocr）](https://huggingface.co/blog/dots-ocr-ne)
  —— 展示如何在移动端高效部署 OCR 模型。



**总结一句话：**
开源视觉语言模型正在重新定义 OCR 的边界。
从纯文本识别到多模态理解、从图像到语义、从离线推理到大规模部署——
如今的开源生态为每一个开发者和研究者提供了前所未有的自由度与创新空间。

无论你是在构建下一代文档智能系统，还是仅想更高效地解析 PDF，
希望这篇指南能帮助你找到最合适的起点 🚀

