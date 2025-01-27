---
title: "SmolVLM 越变越小 —— 全新 250M 和 500M 模型正式发布！"
thumbnail: /blog/assets/smolervlm/banner.png
authors:
- user: andito
- user: mfarre
- user: merve
translators:
- user: chenglu
---

## TLDR

我们非常高兴地宣布为 SmolVLM 家族带来两个新成员：SmolVLM-256M 和 SmolVLM-500M。你没看错——256M 参数，让它成为世界上最小的视觉语言模型 (VLM)！

这些新模型基于我们在 SmolVLM 2B 上的经验，重点关注效率、数据混合以及新的设计取舍。我们非常期待向大家介绍这两款模型：它们在体积仅为原模型一小部分的情况下，依然具备强大的多模态性能。

本次发布包含四个检查点 (checkpoint)：两个基础模型和两个指令微调模型，参数规模分别为 256 M 和 500 M。它们可以直接加载到 Transformers、MLX 和 ONNX 中使用，我们还为 Transformers 和 WebGPU（搭配 ONNX）提供了演示。所有模型和演示均可在 [这里](https://huggingface.co/collections/HuggingFaceTB/smolvlm-256m-and-500m-6791fafc5bb0ab8acc960fb0) 找到。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smoller_vlm_benchmarks.png" alt="Benchmarks" style="width:90%;" />

## 目录

- [概述](#overview)
- [为什么要更小？](#why-go-smaller)
    - [探索 256 M 参数的“小巨人”](#meet-the-256m-parameter-giant)
    - [500 M 的进阶选择](#a-step-up-500m)
- [与 SmolVLM 2 B 相比有什么变化？](#what-changed-since-smolvlm-2b)
- [更小的多模态检索：ColSmolVLM 256M & 500M](#meet-smoller-colsmolvlm)
- [如何使用更小的 SmolVLM](#using-smaller-smolvlm)
- [后续计划](#next-steps)

## 概览

- **SmolVLM-256M** ——世界上最小的视觉语言模型！
- **SmolVLM-500M** ——拥有 5 亿参数的兄弟版本，能在保持极轻量的同时带来显著性能提升。
- **新的视觉编码器选择** ——我们比较了 SigLIP 400M SO（用于 SmolVLM 2B 以及许多其他大型 VLM）与更小的 SigLIP base patch-16/512。结果令人惊讶：更大体量的编码器只带来了微弱的性能提升，因此我们在新版本中选用了拥有 93M 参数的 SigLIP base patch-16/512。
- **更高的图像分辨率** ——我们更小的视觉编码器可以以更高分辨率处理图像（灵感来自 Apple 和 Google 的研究），几乎没有额外负担，但能带来更敏锐的视觉理解。
- **训练优化** ——我们使用了一种新的 Token 化技巧，让模型在实际场景的基准测试中获得了显著提升，尽管从训练损失上看并不明显。

我们也在让模型规模与 SmolLM2 家族（135M、360M、1.7B）保持一致，所以现在你有一整套灵活的小型大语言模型 (LLM) + 视觉语言模型 (VLM) 组合来进行各种实验。


## 为什么要更小？

自从我们发布 SmolVLM 2B 以来，社区反响一直非常好：模型轻量、开源且许可宽松，也易于集成到现有工作流程中。但我们希望进一步让那些受限于设备、只能使用普通笔记本电脑，甚至希望在浏览器中进行推理的用户也能更方便地使用。这就是全新 256M 和 500M 模型诞生的原因。对于需要处理海量数据的用户而言，这些模型在运行成本上也远低于 2B 版本。

过去一年里，我们训练了两个 80B 视觉语言模型，并将它们缩减到 8B。随后又挑战把 2B 的 SmolVLM 继续做小。我们发现还能做得更极致！我们很高兴地展示，在 256M 和 500M 参数规模上，模型依然能拥有相当出色的性能。我们的 256M 版本是迄今为止发布的最小 VLM，但在性能上已超越我们 17 个月前发布的 Idefics 80B 模型。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smoller_vlm_benchmarks.png" alt="Benchmarks" style="width:90%;" />

### 探索 256 M 参数的“小巨人”

仅 256 M 参数就让这款模型成为有史以来最小的 VLM。别看它体量小，却依然足以在很多多模态任务上“大显身手”，包括：

- **图像描述**：给图像或短视频生成文字说明。
- **文档问答**：回答关于 PDF 或扫描文本的问题。
- **基础视觉推理**：回答图表、示意图等方面的提问。


### 500M 的进阶选择

如果你需要更多的性能余量，又希望保持较低的内存占用，那么 SmolVLM-500M（5 亿参数）是一个折中的方案。它比之前的 2B 模型小得多，却能在 DocVQA、MMMU 等任务上取得更接近大模型的成绩。我们还发现它对提示词更加敏感，开箱即用就能更好地适应生产环境。当然，对这两个模型进行微调都可以显著提升它们的性能。

我们在 A100 上对不同批量大小进行了吞吐量测试，结果显示相比 2B 模型有着明显的加速效果。
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/throughput.png
" alt="Benchmarks" style="width:90%;" />

## 与 SmolVLM 2B 相比有什么变化？

1. **视觉编码器选择**  
   过去我们一直使用 SigLIP 400M SO 视觉主干，它也经常出现在其他大型 VLM 架构中。对于此次的小模型，我们实验了两种编码器：  
   - **SigLIP 400M SO**：更高容量，性能更好。  
   - **SigLIP base patch-16/512 (93 M)**：规模更小，但性能竟然十分接近。  

   结果显示，两者在性能差距并不大，所以我们在 256M 和 500M 两个版本中都选择了更小的视觉编码器。此外，更小的编码器还能处理更高分辨率的图像，参考 [Apple](https://arxiv.org/pdf/2403.09611) 和 [Google](https://arxiv.org/pdf/2412.03555) 的研究，这常常能在无需大幅增加参数的前提下，显著提升对图像内容的理解。

2. **数据混合更新**  
   同上一版一样，我们继续使用 [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) 和 [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix)，并新增了 [MathWriting](https://huggingface.co/datasets/andito/mathwriting-google)。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm_datamixture.gif" alt="Data mixture" style="width:90%;" />

我们重新调整了这些数据集的比例，更加强调文档理解（41%）和图像描述（14%），同时仍然兼顾视觉推理、图表理解以及一般的指令跟随。

因此，新模型在文档理解方面有了更坚实的基础，也欢迎进一步微调来提升在特定任务上的表现。


3. **Token 化优化**  
   我们进一步增加了像素混排的效率！新模型以每个 Token 处理 4096 像素的方式对图像进行编码，而之前 2 B 模型是每个 Token 处理 1820 像素。  
   此外，我们加入了特殊的 Token 来表示子图像分隔符，这样像 `<row_1_col_1>` 这样的标记不会再被拆分成 7 个 Token，而是映射到 1 个 Token。一直到 `<row_6_col_6>` 都做了类似处理。这大大提升了模型训练的稳定性和推理结果质量。更多细节可见这篇 [LinkedIn 文章](https://www.linkedin.com/posts/andimarafioti_when-worse-training-losses-lead-to-better-activity-7284521064934592513-yBZe?utm_source=share&utm_medium=member_desktop)。

4. **完善 SmolLM2-SmolVLM 家族**  
   SmolLM2 提供了 135 M、360 M 和 1.7 B 三种规模。这次发布的 256 M 和 500 M VLM 刚好补齐了“小型大语言模型 + 视觉语言模型”的产品线，让你可以自由搭配。

## 更小的多模态检索：ColSmolVLM 256 M & 500 M

我们还发现，这些模型进行微调和实验都非常简便。受 ColBERT-like 检索模型启发的团队训练了 ColSmolVLM，让多模态检索在速度上达到了最先进水平，而且性能上可与体量大 10 倍的模型相媲美。SmolVLM 的轻量化特性让搭建可检索数据库的过程更加高效、成本更低。我们相信 256M 模型会成为不少专用场景的理想选择。更多关于如何使用新的 ColSmolVLM 配合 SmolVLM 构建多模态检索的示例，请参见 [后续计划](#next-steps)。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/colsmol_tiny.png" alt="Benchmarks" style="width:90%;" />


## SmolDocling

我们与 IBM 合作，为他们的 [Docling](https://github.com/DS4SD/docling) 项目打造了专用模型。IBM 基于 256M 模型的初步结果已经十分惊艳。以下是他们分享的一些示例，欢迎持续关注更多动态！

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smoldocling_layout_table_image.png" alt="Benchmarks" style="width:90%;" />
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smoldocling_code.png" alt="Benchmarks" style="width:90%;" />

## 如何使用更小的 SmolVLM

这些新的 SmolVLM 与旧版本的代码无缝兼容。你可以使用 Transformers 和 MLX 来完成推理或微调，也可以在 TRL 中进行对齐训练。此外，本次发布还提供了 ONNX 检查点。

如果想在 Transformers 中快速上手 SmolVLM，可以按照类似下面的示例使用：

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

# Preprocess
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
```

如果想在 MLX 中配合 SmolVLM，运行相应的 CLI 命令也非常便捷：

```bash
python3 -m mlx_vlm.generate --model HuggingfaceTB/SmolVLM-500M-Instruct --max-tokens 400 --temp 0.0 --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vlm_example.jpg --prompt "What is in this image?"
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm-mlx.gif" alt="MLX" style="width:90%;" />

我们还提供了针对 WebGPU 的演示版，分别为 [SmolVLM-256M-Instruct](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-256M-Instruct-WebGPU) 和 [SmolVLM-500M-Instruct](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-500M-Instruct-WebGPU)。  

更多关于微调及与 ColSmolVLM 结合构建多模态 RAG 的链接，请见下文的 [后续计划](#next-steps)。

## 后续计划

- 我们期待看到大家将如何使用这些更小的 VLM！点击 [这里](https://huggingface.co/collections/HuggingFaceTB/smolvlm-256m-and-500m-6791fafc5bb0ab8acc960fb0) 率先体验。
- 如需进一步了解 SmolVLM，请访问 [这里](https://huggingface.co/blog/smolvlm)。
- [使用 Transformers 对 SmolVLM 进行微调与 QLoRA](https://github.com/merveenoyan/smol-vision/blob/main/Smol_VLM_FT.ipynb)
- [在消费级 GPU 上使用 TRL，对 SmolVLM 进行直接偏好优化 (DPO)](Fine-tuning SmolVLM using direct preference optimization (DPO) with TRL on a consumer GPU)
- [在 Colab 免费 GPU 上使用 ColSmolVLM 和 SmolVLM 构建多模态 RAG](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)

特别感谢 ViDoRe 团队为本次发布训练了 ColSmolVLM；同时感谢 [Tony Wu](https://huggingface.co/tonywu71)、[Manuel Faysse](https://huggingface.co/manu) 以及 [Joshua Lochner](https://huggingface.co/Xenova) 在 ONNX 转换与 WebGPU 演示中的帮助，也要感谢 [Vaibhav Srivastav](https://huggingface.co/reach-vb) 为本次发布做出的贡献。