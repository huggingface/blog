---
title: SmolVLM2：让视频理解能力触手可及 
thumbnail: /blog/assets/smolvlm2/banner.png  
authors:  
- user: orrzohar
  guest: true
  org: Stanford
- user: mfarre  
- user: andito  
- user: merve  
- user: pcuenq  
- user: cyrilzakka  
- user: xenova  
translators:
- user: yaoqih
---

# SmolVLM2：让视频理解能力触手可及

## 一句话总结：SmolVLM现已具备更强的视觉理解能力📺

SmolVLM2 标志着视频理解技术的根本性转变——从依赖海量计算资源的巨型模型，转向可在任何设备运行的轻量级模型。我们的目标很简单：让视频理解技术从手机到服务器都能轻松部署。

我们同步发布三种规模的模型（22亿/5亿/2.56亿参数），并全面支持 MLX 框架（提供 Python 和 Swift API）。所有模型及演示案例[均可在此获取](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7)。

想立即体验 SmolVLM2？欢迎试用我们的[交互式聊天界面](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM2)，通过直观的交互测试 22 亿参数模型的视频理解能力。


## 目录

- [技术细节](#技术细节)
  - [SmolVLM2-22亿：视觉与视频理解新标杆](#smolvlm2-22亿视觉与视频理解新标杆)
  - [更轻量级：5亿与2.56亿视频模型](#更轻量级5亿与256亿视频模型)
  - [SmolVLM2 应用案例集](#smolvlm2-应用案例集)
    - [iPhone 视频理解](#iphone-视频理解)
    - [VLC 媒体播放器集成](#vlc-媒体播放器集成)
    - [视频精彩片段生成器](#视频精彩片段生成器)
- [使用 Transformers 和 MLX 运行 SmolVLM2](#使用-transformers-和-mlx-运行-smolvlm2)
  - [Transformers](#transformers)
    - [视频推理](#视频推理)
    - [多图推理](#多图推理)
  - [MLX 推理](#mlx-推理)
    - [Swift MLX](#swift-mlx)
  - [微调 SmolVLM2](#微调-smolvlm2)
- [延伸阅读](#延伸阅读)


## 技术细节

我们推出三款新模型（2.56亿/5亿/22亿参数）。其中 22 亿参数模型是视觉与视频任务的优选，而 5 亿和 2.56 亿模型更是**迄今发布的最小型视频语言模型**。

虽然体积小巧，但其内存效率却优于现有所有模型。在视频领域权威基准测试 Video-MME 中，SmolVLM2 在 20 亿参数级别与顶尖模型比肩，在更小规模模型中更是一骑绝尘。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm2-videomme2.png" width="50%" alt="SmolVLM2 性能表现">

*注：Video-MME 基准因覆盖多样视频类型（11秒至1小时）、多模态数据（含字幕和音频）及254小时高质量专家标注而著称。[了解更多](https://video-mme.github.io/home_page.html)*


### SmolVLM2-22亿：视觉与视频理解新标杆

相较于前代产品，新版 22 亿模型在图像数学解题、图片文字识别、复杂图表解析和科学视觉问答方面表现显著提升：

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm2-score-gains.png" width="50%" alt="SmolVLM2 视觉能力提升">

在视频任务中，该模型展现出优异性价比。基于[Apollo大型多模态模型视频理解研究](https://apollo-lmms.github.io/)的数据混合策略，我们在视频/图像性能之间取得了良好平衡。

其内存效率之高，甚至可在免费版 Google Colab 中运行。

<details>
<summary>Python 代码示例</summary>

```python
# 确保使用最新版 Transformers
!pip install git+https://github.com/huggingface/transformers.git

from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "path_to_video.mp4"},
            {"type": "text", "text": "请详细描述该视频内容"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
```
</details>


### 更轻量级：5亿与2.56亿视频模型

我们首次突破小模型极限：[SmolVLM2-5亿视频指令模型](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)在保持 22 亿模型 90% 视频理解能力的同时，参数量减少四分之三 🤯。

而我们的实验性作品 [SmolVLM2-2.56亿视频指令模型](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)则探索了小模型的极限。受 IBM 利用[256M 模型](https://ds4sd.github.io/docling/examples/pictures_description/)取得成果的启发，我们继续深挖小模型潜力。虽然属于实验性发布，但期待它能激发更多创新应用。


### SmolVLM2 应用案例集

为在小型视频模型领域展现我们的核心理念，我们开发了三个实际应用案例，生动呈现该模型系统的多场景应用能力。

#### iPhone 视频理解
<table style="border-collapse: collapse;">
<tr>
<td width="300" style="border: none;">
<center>
<iframe width="300" height="533" src="https://www.youtube.com/embed/G1yQlHTk_Ig" frameborder="0" allowfullscreen></iframe>
</center>
</td>
<td valign="top" style="border: none;">
我们开发了完全本地化运行的 iPhone 应用（使用 5 亿模型），用户无需云端即可进行视频分析。<a href="https://huggingface.co/spaces/HuggingFaceTB/SmolVLM2-iPhone-waitlist" target="_blank">立即申请测试资格</a>，与我们共同打造移动端 AI 视频应用！
</td>
</tr>
</table>

#### VLC 媒体播放器集成
<table style="border-collapse: collapse;">
<tr>
<td width="500" style="border: none;">
<center>
<iframe width="500" height="281" src="https://www.youtube.com/embed/NGHCFEW7DCg" frameborder="0" allowfullscreen></iframe>
</center>
</td>
<td valign="top" style="border: none;">
与 VLC 合作开发的智能视频导航功能，支持通过语义搜索跳转到指定片段。<a href="https://huggingface.co/spaces/HuggingFaceTB/SmolVLM2-XSPFGenerator" target="_blank">在这个 Space </a>体验播放列表生成原型。
</td>
</tr>
</table>

#### 视频精彩片段生成器
<table style="border-collapse: collapse;">
<tr>
<td width="500" style="border: none;">
<center>
<iframe width="500" height="281" src="https://www.youtube.com/embed/ZT2oS8EqnKI" frameborder="0" allowfullscreen></iframe>
</center>
</td>
<td valign="top" style="border: none;">
擅长处理长视频（1小时+），可自动提取足球比赛等场景的关键时刻。<a href="https://huggingface.co/spaces/HuggingFaceTB/SmolVLM2-HighlightGenerator" target="_blank">立即在线体验</a>。
</td>
</tr>
</table>


## 使用 Transformers 和 MLX 运行 SmolVLM2

自发布首日起，我们便实现了 SmolVLM2 与 Transformer 架构及 MLX 框架的即开即用兼容适配。在本章节中，您可查阅面向视频与多图像处理的多种推理方案，以及配套的实战教程指南。

### Transformers

通过对话式 API 可便捷调用 SmolVLM2 模型，聊天模板会自动对输入进行预处理：
你可以像下面这样加载模型：
```python
# 确保使用最新版 Transformers
!pip install git+https://github.com/huggingface/transformers.git

from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to(DEVICE)
```

#### 视频推理

通过传入`{"type": "video", "path": {video_path}`，你可以在聊天模板传递视频路径，下面是完整的模板示例：

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "视频路径.mp4"},
            {"type": "text", "text": "请详细描述该视频"}
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
```

#### 多图推理

除了视频，SmolVLM2支持多图推理。您可以通过聊天模板使用相同的API。

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "这两张图片有何区别？"},
            {"type": "image", "path": "图片1.png"},
            {"type": "image", "path": "图片2.png"} 
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
```

## 使用 MLX 进行推理

要在 Apple 芯片设备上使用 Python 运行 SmolVLM2，可通过优秀的 [mlx-vlm 库](https://github.com/Blaizzy/mlx-vlm)实现。首先安装特定分支：

```bash
pip install git+https://github.com/pcuenca/mlx-vlm.git@smolvlm
```

单图推理示例（使用[未量化的 5 亿参数版本](https://huggingface.co/mlx-community/SmolVLM2-500M-Video-Instruct-mlx)）：

```bash
python -m mlx_vlm.generate \
  --model mlx-community/SmolVLM2-500M-Video-Instruct-mlx \
  --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg \
  --prompt "请描述这张图片"
```

视频分析专用脚本（系统提示词可引导模型关注重点）：

```bash
python -m mlx_vlm.smolvlm_video_generate \
  --model mlx-community/SmolVLM2-500M-Video-Instruct-mlx \
  --system "请专注描述视频片段中的关键戏剧性动作或显著事件，省略常规场景描述" \
  --prompt "视频中发生了什么？" \
  --video ~/Downloads/example_video.mov
```

#### Swift 语言支持
通过 [mlx-swift-examples 代码库](https://github.com/ml-explore/mlx-swift-examples)实现 Swift 支持（当前需从[开发分支](https://github.com/cyrilzakka/mlx-swift-examples)编译），这正是我们构建 iPhone 应用的技术基础。

图像推理 CLI 示例：

```bash
./mlx-run --debug llm-tool \
    --model mlx-community/SmolVLM2-500M-Video-Instruct-mlx \
    --prompt "请描述这张图片" \
    --image https://example.com/image.jpg \
    --temperature 0.7 --top-p 0.9 --max-tokens 100
```

视频分析示例（系统提示词调节输出粒度）：

```bash
./mlx-run --debug llm-tool \
    --model mlx-community/SmolVLM2-500M-Video-Instruct-mlx \
    --system "请专注描述视频片段中的核心事件" \
    --prompt "发生了什么？" \
    --video ~/Downloads/example_video.mov \
    --temperature 0.7 --top-p 0.9 --max-tokens 100
```

若您使用 MLX 和 Swift 集成 SmolVLM2，欢迎在评论区分享您的实践！


### 微调 SmolVLM2

您可使用 Transformers 库对视频数据进行微调。我们已在 Colab 环境演示了基于 [VideoFeedback 数据集](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback)对 5 亿参数模型的微调流程。由于模型较小，推荐使用全参数微调而非 QLoRA/LoRA（但可在 cB 变体尝试 QLoRA）。完整教程请参考[微调笔记](https://github.com/huggingface/smollm/blob/main/vision/finetuning/SmolVLM2_Video_FT.ipynb)。


## 延伸信息

特别鸣谢 Raushan Turganbay、Arthur Zucker 和 Pablo Montalvo Leroux 对模型移植的贡献。

如果您想了解更多关于SmolVLM系列模型的信息，请阅读以下内容：
[模型与演示全集](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7) | [Apollo 视频理解研究](https://apollo-lmms.github.io/)

期待见证您用 SmolVLM2 构建的创新应用！

