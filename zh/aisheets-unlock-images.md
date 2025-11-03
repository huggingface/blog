---
title: "用 AI Sheets 解锁图像的力量"
thumbnail: /blog/assets/aisheets/aisheets-image.png
authors:
- user: Ameeeee
- user: dvilasuero
- user: frascuchon
- user: damianpumar
- user: lvwerra
- user: thomwolf
translators:
- user: chenglu
---

# 用 AI Sheets 解锁图像的力量

> 🧭**简要概览**：Hugging Face AI Sheets 是一款开源工具，能够**用 AI 模型增强数据集的处理能力**，无需编写任何代码。**现在新增视觉功能**：可以从图像（如收据、文档）中提取数据、根据文本生成图像、甚至编辑图片——一切都能在电子表格中完成。依托 Inference Providers，可调用数千个开放模型。

<figure class="image flex flex-col items-center text-center m-0 w-full">
  <video
     alt="AIsheets-recipes.mp4"
     autoplay loop autobuffer muted playsinline
   >
   <source src="https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/A4BKE47IduZnak9YfxArw.mp4"
   type="video/mp4">
  </video>
  <figcaption>用 AI Sheets 分析图像</figcaption>   
</figure>

我们非常高兴地发布 [Hugging Face AI Sheets](https://github.com/huggingface/aisheets) 的重大更新版——这是一款可通过开放 AI 模型构建、转换与丰富数据的开源工具。AI Sheets 基于 [Inference Providers](https://huggingface.co/docs/inference-providers/index) 运行，意味着你可以使用由全球顶级推理服务驱动的数千种开放模型。

[AI Sheets 的首个版本](https://huggingface.co/blog/aisheets) 让结构化和增强文本内容变得轻而易举。**现在，我们为它加入了视觉功能。**

图像无处不在——商品照片、收据、截图、图表、徽标……这些图片中蕴含着丰富的结构化信息，等待被提取、分析与转换。现在，你终于可以在 AI Sheets 中直接处理图像内容：查看图片、分析内容、提取数据、生成新图像，甚至实时编辑——全部在同一流程中完成。

---

## 你的图片藏着故事

图片往往包含宝贵的信息——产品目录、客户支持单、研究档案、收据、文档等。现在你可以直接上传图片，或使用带图像的数据集，再借助视觉模型提取、分析并结构化这些内容。

**你可以做到：**

* **描述与分类图像** —— 为产品照片生成文字描述，识别文档类型，或根据内容自动打标签
* **提取结构化数据** —— 从收据中提取明细，从图表中提取数据，从扫描件中识别文本
* **添加上下文与元数据** —— 自动为图片添加相关属性、质量评分或自定义标注

与文本列一样，你可以反复调整提示词、手动修改结果，并用“点赞”告诉模型你更喜欢哪种输出。你的反馈会作为少量样本（few-shot）帮助模型生成更好的结果。

**示例：从收据中提取结构化费用信息**

假设你刚出差回来，手里有一堆收据。上传到 AI Sheets 后，在新列中输入提示词：
`提取该收据中的商户名称、日期、总金额和费用类别`

AI Sheets 会自动处理每一张收据，输出一个整洁的表格，包含所有提取出的详细信息。你可以手动纠正错误，对准确结果点赞，并重新生成其他条目以提升整体质量。最终可将数据导出为 CSV 或 Parquet 文件，用于你的报销工具。

或者，你也可以将家中旧笔记本上的手写食谱数字化——创建列提取食材、烹饪时间、菜系类型，让个人档案变成可搜索的结构化数据集。

---

## 在同一流程中生成与转换文本和图像

需要为你的内容配图？AI Sheets 可以在电子表格中直接通过 AI 模型生成或编辑图像，让整个内容创作流程集中在一个界面中完成。

你可以：

* **从文本生成图像** —— 生成与你内容匹配的社交媒体图片、缩略图或插画
* **编辑与转换图像** —— 修改上传或生成的图片：更换风格、添加元素、调整构图
* **批量生成变体** —— 一次生成多个版本或风格，测试最受欢迎的视觉形式
* **建立视觉素材库** —— 为品牌活动批量创建风格一致的图像资产

**示例：创建带配图的内容日历**

假设你计划发布一个月的健康食谱类社交媒体帖子。你已经准备好了标题与文案，但还缺图像。

创建一个图像列，提示如下：
“为以下标题生成一张美味食物的照片：{{title}}。风格：明亮、俯拍、自然光。”

AI Sheets 会为每篇帖子生成独特的图片。效果不理想？再建一列修改：
“将背景换成乡村木桌，加上新鲜香草作为装饰。”

你可以多次调整生成和编辑提示，尝试不同方案。最终，你的整月内容计划——文字与图片——都集中在一张表格中，可直接导出或排程发布。

---

## 使用指南

下面我们通过一个实例来看看 AI Sheets 的实际操作。我们将用开源模型来识别祖母笔记本中手写的食谱。

### 上传数据

我们有一个文件夹，里面保存了食谱的照片，只需上传即可。

![folder](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/OZSQLc_GeINsLWnL-3t49.png)
![upload](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/s8FkH6gw2LG9F7rM2mJ0D.png)

上传后生成的表格如下：

![table](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/4lW1LWM31dB_stOP0QvL1.png)

---

### 了解 AI 操作

在电子表格中，每一列都可以通过“AI 操作”进行提取、转换或查询等各种处理。

点击任意列上方的叠加层即可查看操作选项：

![ai-action](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/O1XHmf70blGY6kRMOcvMi.png)

图像列支持的操作包括：提取文字、图像问答、目标检测、上色、添加文字，以及自定义任务；
文本列则支持：摘要、关键词提取、翻译等操作。

每个 AI 操作都由“提示词 + 模型”组合而成。让我们看看它如何处理手写食谱数据。

---

### 从图像中提取文字

AI Sheets 提供了一个从图像提取文字的模板：

![extract-text](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/gTXMKRJ8J0Oil7YUZUnOr.png)

执行后会生成一列包含转录结果的新列，例如：

![recipe](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/7IuC9cTT5v-fXHvI9NB9D.webp)

上图识别出的文本如下：

```
MEMORANDUM:

From

To

1 Box Duncan Hines Yellow Cake Mix
1 Box instant lemon pudding
2/3 cups water
1/2 cup Mozola oil
4 eggs
Lemon flavoring to taste.
Put in mixing bowl and beat for 10 min.

and REMEMBER... for Quality PRINTING
CALL OR WRITE
Gatling & Pierce
PRINTERS
TELEPHONE 332-2579
22 YEARS OF SERVICE IN NORTHEASTERN CAROLINA
```

识别效果不错，但包含了页眉页脚的印刷文字。默认模板的提示是：

`提取图像中所有可见文字，包括标志、标签、文档或任何文字内容。`

我们可以改用自定义提示。

![custom](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/oYanFJWYR6zejEgq2TFYc.png)

自定义提取的结果如下：

- 1 盒 Duncan Hines 黄蛋糕粉
- 1 盒速溶柠檬布丁
- 2/3 杯水
- 1/2 杯 Mazola 食用油
- 4 个鸡蛋
- 适量柠檬香精
- 倒入搅拌碗中搅打 10 分钟

效果非常理想。对于更复杂的图片，我们可以尝试不同的模型。默认模型为 `Qwen/Qwen2.5-VL-7B-Instruct`，在速度与准确度间平衡良好；我们还可以使用更强大的推理模型 `Qwen/Qwen3-VL-235B-A22B-Reasoning`。

![qwen3](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/pA3vr1tw8VtmgS9Q6pskF.png)

模型对比结果如下：

| Qwen/Qwen2.5-VL-7B-Instruct                                                                                                                                                                                       | Qwen/Qwen3-VL-235B-A22B-Reasoning                                                                                                                                                                                  |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| in large bowl combine meat, onion, bread crumbs 1/2 nutmeg & cheese - as you add sprinkle around. Then blend - Last sprinkle blend again Bake in large pan for 10-15 min. at 350. Let stand 5 min before serving. | in lg bowl combine meat, onion, bread crumbs 1/4 nutmeg & cheese - as you add sprinkle around. then blend - last **spinach** blend again. Bake in lg pan for **50-60 min. @ 350** - let stand 5 min before serving |

两个模型的输出很接近，但更高阶模型识别出了两个重要细节（**加粗部分**）：烘焙时间和关键配料——菠菜。

---

### 清洗、转换与丰富文本

当我们对提取结果满意后，可以进一步用 AI 操作转换格式，如生成 HTML 页面。

![format](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/lB4Z_iEKIUnuaSTPqc_xZ.png)

生成后，每份食谱都变成了结构清晰、排版优美的 HTML 页面：

![html](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/zSCnY3D6uobqSCHj7tBqR.png)

---

### 编辑与转换图像

AI Sheets 集成了图像编辑模型（如 Qwen-Image-Edit），可以直接对图片进行风格化处理与增强。

例如，你希望给食谱图片增加“复古”效果，可以选择黑白滤镜模板：

![transform-bw](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/Blf4wtKrX6UYkQ06HUV-8.png)

结果如下：

![bw](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/cMzCQUMMRKch__C3W_-Ve.png)

---

### 导出数据集

当你对结果满意后，可以将数据集导出并上传至 Hugging Face Hub！
可选择导出至团队组织、个人主页，或设为私密数据集。

![export](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/2fSKxUzwZtPkPJ-ZWEYYl.png)

你可以查看我们刚刚创建的示例数据集：
[点击查看](https://huggingface.co/datasets/aisheets/unlocked-recipes)

---

## 接下来做什么？

你可以直接 [在线试用 AI Sheets](https://huggingface.co/spaces/aisheets/sheets)，无需安装或部署。
如果希望本地运行并获得更高性能，建议升级到 PRO 版本，可享受 20 倍推理配额。

如有任何问题或建议，欢迎在社区留言，或通过 [GitHub 提交 issue](https://github.com/huggingface/aisheets) 与我们交流。


