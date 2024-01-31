---
title: Hugging Face 中计算机视觉的现状
thumbnail: /blog/assets/cv_state/thumbnail.png
authors:
- user: sayakpaul
---

# Hugging Face 中计算机视觉的现状


在Hugging Face上，我们为与社区一起推动人工智能领域的民主化而感到自豪。作为这个使命的一部分，我们从去年开始专注于计算机视觉。开始只是 [🤗 Transformers中Vision Transformers (ViT) 的一个 PR](https://github.com/huggingface/transformers/pull/10950)，现在已经发展壮大：8个核心视觉任务，超过3000个模型，在Hugging Face Hub上有超过1000个数据集。

自从 ViTs 加入 Hub 后，已经发生了大量激动人心的事情。在这篇博客文章中，我们将从 🤗Hugging Face 生态系统中总结已经发生的和将要发生的进展，以支持计算机视觉的持续发展。

下面是我们要覆盖的内容：

- [支持的视觉任务和流水线](https://huggingface.co/blog/cv_state#support-for-pipelines)
- [训练你自己的视觉模型](https://huggingface.co/blog/cv_state#training-your-own-models)
- [和`timm`整合](https://huggingface.co/blog/cv_state#🤗-🤝-timm)

- [Diffusers](https://huggingface.co/blog/cv_state#🧨-diffusers)
- [对第三方库的支持](https://huggingface.co/blog/cv_state#support-for-third-party-libraries)
- [开发](https://huggingface.co/blog/cv_state#deployment)
- 以及更多内容！

## 启动社区: 一次一个任务 

Hugging Face Hub 拥有超过10万个用于不同任务的公共模型，例如：下一词预测、掩码填充、词符分类、序列分类等。截止今天，我们支持[8个核心视觉任务](https://huggingface.co/tasks)，提供许多模型的 checkpoints：

- 图像分类
- 图像分割
- （零样本）目标检测
- 视频分类
- 深度估计
- 图像到图像合成
- 无条件图像生成
- 零样本图像分类

每个任务在 Hub 上至少有10个模型等待你去探索。此外，我们支持视觉和语言的交叉任务，比如：

- 图像到文字（图像说明，光学字符识别）
- 文字到图像
- 文档问答
- 视觉问答

这些任务不仅需要最先进的基于 Transformer 的架构，如 [ViT](https://huggingface.co/docs/transformers/model_doc/vit)、[Swin](https://huggingface.co/docs/transformers/model_doc/swin)、[DETR](https://huggingface.co/docs/transformers/model_doc/detr)，还需要*纯卷积*的架构，如 [ConvNeXt](https://huggingface.co/docs/transformers/model_doc/convnext)、[ResNet](https://huggingface.co/docs/transformers/model_doc/resnet)、[RegNet](https://huggingface.co/docs/transformers/model_doc/regnet)，甚至更多！像 ResNets 这样的架构仍然与无数的工业用例非常相关，因此在 🤗 Transformers 中也支持这些非 Transformers 的架构。

还需要注意的是，在 Hub 上的这些模型不仅来自 Transformers 库，也来自于其他第三方库。例如，尽管我们在 Hub 上支持无条件图像生成等任务，但我们在 Transformers 中还没有任何模型支持该任务（比如[这个](https://huggingface.co/ceyda/butterfly_cropped_uniq1K_512)）。支持所有的机器学习任务，无论是使用 Transformers 还是第三方库来解决，都是我们促进一个协作的开源机器学习生态系统使命的一部分。

## 对 Pipelines 的支持

我们开发了 [Pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines) 来为从业者提供他们需要的工具，以便轻松地将机器学习整合到他们的工具箱中。对于给定与任务相关的输入，他们提供了一种简单的方法来执行推理。我们在Pipelines里支持[7种视觉任务](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#computer-vision)。下面是一个使用 Pipelines 进行深度估计的例子：

```python
from transformers import pipeline

depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
output = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")

# This is a tensor with the values being the depth expressed
# in meters for each pixel
output["depth"]
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/cv_state/depth_estimation_output.png)

即使对于视觉问答任务，接口也保持不变：

```python
from transformers import pipeline

oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa")
image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"

oracle(question="What is she wearing?", image=image_url, top_k=1)
# [{'score': 0.948, 'answer': 'hat'}]
```

## 训练你自己的模型

虽然能够使用现成推理模型是一个很好的入门方式，但微调是社区获得最大收益的地方。当你的数据集是自定义的、并且预训练模型的性能不佳时，这一点尤其正确。

Transformers 为一切与训练相关的东西提供了[训练器 API](https://huggingface.co/docs/transformers/main_classes/trainer)。当前，`Trainer`无缝地支持以下任务：图像分类、图像分割、视频分类、目标检测和深度估计。微调其他视觉任务的模型也是支持的，只是并不通过`Trainer`。

只要损失计算包含在 Transformers 计算给定任务损失的模型中，它就应该有资格对该任务进行微调。如果你发现问题，请在 GitHub 上[报告](https://github.com/huggingface/transformers/issues)。

我从哪里可以找到代码？

- [模型文档](https://huggingface.co/docs/transformers/index#supported-models)
- [Hugging Face 笔记本](https://github.com/huggingface/notebooks)
- [Hugging Face 示例脚本](https://github.com/huggingface/transformers/tree/main/examples)
- [任务页面](https://huggingface.co/tasks)

[Hugging Face 示例脚本](https://github.com/huggingface/transformers/tree/main/examples)包括不同的[自监督预训练策略](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)如 [MAE](https://arxiv.org/abs/2111.06377)，和[对比图像到文本预训练策略](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text)如 [CLIP](https://arxiv.org/abs/2103.00020)。这些脚本对于研究社区和愿意在预训练模型上从头训练自定义数据语料的从业者来说是非常宝贵的资源。

不过有些任务本来就不适合微调。例子包括零样本图像分类（比如 [CLIP](https://huggingface.co/docs/transformers/main/en/model_doc/clip)），零样本目标检测（比如 [OWL-ViT](https://huggingface.co/docs/transformers/main/en/model_doc/owlvit)），和零样本分割（比如 [CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)）。我们将在这篇文章中重新讨论这些模型。

## 与 Datasets 集成

Datasets 提供了对数千个不同模态数据集的轻松访问。如前所述，Hub 有超过1000个计算机视觉的数据集。一些例子值得关注：[ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)、[Scene Parsing](https://huggingface.co/datasets/scene_parse_150)、[NYU Depth V2](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2)、[COYO-700M](https://huggingface.co/datasets/kakaobrain/coyo-700m) 和 [LAION-400M](https://huggingface.co/datasets/laion/laion400m)。这些在 Hub 上的数据集，只需两行代码就可以加载它们：

```python
from datasets import load_dataset

dataset = load_dataset("scene_parse_150")
```

除了这些数据集，我们提供了对增强库如 [albumentations](https://github.com/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb) 和 [Kornia](https://github.com/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb) 的集成支持。社区可以利用 Datasets 的灵活性和性能，还有这些库提供的强大的增强变换能力。除此之外，我们也为核心视觉任务提供[专用的数据加载指南](https://huggingface.co/docs/datasets/image_load)：图像分类，图像分割，目标检测和深度估计。

## 🤗 🤝 timm

`timm`，即 [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)，是一个最先进的 PyTorch 图像模型、预训练权重和用于训练、推理、验证的实用脚本的开源集合。

我们在 Hub 上有超过200个来自 `timm` 的模型，并且有更多模型即将上线。查看[文档](https://huggingface.co/docs/timm/index)以了解更多关于此集成的信息。

## 🧨 Diffusers

Diffusers 提供预训练的视觉和音频扩散模型，并且用作推理和训练的模块化工具箱。有了这个库，你可以从自然语言输入和其他创造性用例中生成可信的图像。下面是一个例子：

```python
from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
generator.to(“cuda”)

image = generator("An image of a squirrel in Picasso style").images[0]
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/cv_state/sd_output.png)

这种类型的技术可以赋予新一代的创造性应用，也可以帮助来自不同背景的艺术家。查看[官方文档](https://huggingface.co/docs/diffusers)以了解更多关于 Diffusers 和不同用例的信息。

基于扩散模型的文献正在快速发展，这就是为什么我们与[乔纳森·惠特克](https://github.com/johnowhitaker)合作开发一门课程。这门课程是免费的，你可以点击[这里](https://github.com/huggingface/diffusion-models-class)查看。

## 对第三方库的支持

Hugging Face 生态系统的核心是 [Hugging Face Hub](https://huggingface.co/docs/hub)，它让人们在机器学习上有效合作。正如前面所提到的，我们在 Hub 上不仅支持来自 🤗 Transformers 的模型，还支持来自其他第三方包的模型。为此，我们提供了几个[实用程序](https://huggingface.co/docs/hub/models-adding-libraries)，以便你可以将自己的库与 Hub 集成。这样做的主要优点之一是，与社区共享工件（如模型和数据集）变得非常容易，从而使你的用户可以更容易地尝试你的模型。

当你的模型托管在 Hub 上时，你还可以为它们[添加自定义推理部件](https://github.com/huggingface/api-inference-community)。推理部件允许用户快速地检查模型。这有助于提高用户的参与度。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/cv_state/task_widget_generation.png)

## 计算机视觉演示空间

使用 Hugging Hub Spaces应用，人们可以轻松地演示他们的机器学习模型。空间支持与 [Gradio](https://gradio.app/)、[Streamlit](https://streamlit.io/) 和 [Docker](https://www.docker.com/) 的直接集成，使从业者在展示他们的模型时有很大的灵活性。你可以用 Spaces 引入自己的机器学习框架来构建演示。

在 Spaces 里，Gradio 库提供几个部件来构建计算机视觉应用，比如 [Video](https://gradio.app/docs/#video)、[Gallery](https://gradio.app/docs/#gallery) 和 [Model3D](https://gradio.app/docs/#model3d)。社区一直在努力构建一些由 Spaces 提供支持的令人惊叹的计算机视觉应用：

- [从输入图像的预测深度图生成3D体素](https://huggingface.co/spaces/radames/dpt-depth-estimation-3d-voxels)
- [开放词汇语义分割](https://huggingface.co/spaces/facebook/ov-seg)
- [通过生成字幕来讲述视频](https://huggingface.co/spaces/nateraw/lavila)
- [对来自YouTube的视频进行分类](https://huggingface.co/spaces/fcakyon/video-classification)
- [零样本视频分类](https://huggingface.co/spaces/fcakyon/zero-shot-video-classification)
- [视觉问答](https://huggingface.co/spaces/nielsr/vilt-vqa)
- [使用零样本图像分类为图像找到最佳说明以生成相似的图像](https://huggingface.co/spaces/pharma/CLIP-Interrogator)

## 🤗 AutoTrain

[AutoTrain](https://huggingface.co/autotrain) 提供一个”零代码“的解决方案，为文本分类、文本摘要、命名实体识别等这样的任务训练最先进的机器学习模型。对于计算机视觉，我们当前支持[图像分类](https://huggingface.co/blog/autotrain-image-classification)，但可以期待更多的任务覆盖。

AutoTrain 还支持[自动模型评估](https://huggingface.co/spaces/autoevaluate/model-evaluator)。此应用程序允许你用在 Hub 上的各种[数据集](https://huggingface.co/datasets)评估 🤗 Transformers [模型](https://huggingface.co/models?library=transformers&sort=downloads)。你的评估结果将会显示在[公共排行榜](https://huggingface.co/spaces/autoevaluate/leaderboards)上。你可以查看[这篇博客](https://huggingface.co/blog/eval-on-the-hub)以获得更多细节。

## 技术理念

在此部分，我们像向你分享在 🤗 Transformers 里添加计算机视觉背后的理念，以便社区知道针对该领域的设计选择。

尽管 Transformers 是从 NLP 开始的，但我们今天支持多种模式，比如：视觉、音频、视觉语言和强化学习。对于所有的这些模式，Transformers 中所有相应的模型都享有一些共同的优势：

- 使用一行代码`from_pretrained()`即可轻松下载模型
- 用`push_to_hub()`轻松上传模型
- 支持使用 checkpoint 分片技术加载大型的 checkpoints
- 优化支持（使用 [Optimum](https://huggingface.co/docs/optimum) 之类的工具）

- 从模型配置中初始化
- 支持 PyTorch 和 TensorFlow（非全面支持）
- 以及更多

与分词器不同，我们有预处理器（例如[这个](https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTImageProcessor)）负责为视觉模型准备数据。我们一直努力确保在使用视觉模型时依然有轻松和相似的用户体验：

```python
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor  = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
# Egyptian cat
```

即使对于一个困难的任务如目标检测，用户体验也不会改变很多：

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")
inputs = image_processor(images=image, return_tensors="pt")

outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=target_sizes
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

输出为：

```
Detected remote with confidence 0.833 at location [38.31, 72.1, 177.63, 118.45]
Detected cat with confidence 0.831 at location [9.2, 51.38, 321.13, 469.0]
Detected cat with confidence 0.804 at location [340.3, 16.85, 642.93, 370.95]
Detected remote with confidence 0.683 at location [334.48, 73.49, 366.37, 190.01]
Detected couch with confidence 0.535 at location [0.52, 1.19, 640.35, 475.1]
```

## 视觉零样本模型

大量的模型以有趣的方式重新修订了分割和检测等核心视觉任务，并引入了更大的灵活性。我们支持 Transformers 中的一些：

- [CLIP](https://huggingface.co/docs/transformers/main/en/model_doc/clip) 支持带提示的零样本图像分类。给定一张图片，你可以用类似”一张{}的图片“这样的自然语言询问来提示 CLIP 模型。期望是得到类别标签作为答案。
- [OWL-ViT](https://huggingface.co/docs/transformers/main/en/model_doc/owlvit) 允许以语言为条件的零样本目标检测和以图像为条件的单样本目标检测。这意味着你可以在一张图片中检测物体即使底层模型在训练期间没有学过检测它们！你可以参考[这个笔记本](https://github.com/huggingface/notebooks/tree/main/examples#:~:text=zeroshot_object_detection_with_owlvit.ipynb)以了解更多。

- [CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg) 支持以语言为条件的零样本图像分割和以图像为条件的单样本图像分割。这意味着你可以在一张图片中分割物体即使底层模型在训练期间没有学过分割它们！你可以参考说明此想法的[这篇博客文章](https://huggingface.co/blog/clipseg-zero-shot)。[GroupViT](https://huggingface.co/docs/transformers/model_doc/groupvit) 也支持零样本分割。
- [X-CLIP](https://huggingface.co/docs/transformers/main/en/model_doc/xclip) 展示对视频的零样本泛化。准确地说是支持零样本视频分类。查看[这个笔记本](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/X-CLIP/Zero_shot_classify_a_YouTube_video_with_X_CLIP.ipynb)以获得更多细节。

社区期待在今后的日子里看到 🤗Transformers 支持更多的计算机视觉零样本模型。

## 开发

我们的 CTO 说：”真正的艺术家能将产品上市“🚀

我们通过 🤗[Inference Endpoints](https://huggingface.co/inference-endpoints) 支持这些视觉模型的开发。Inference Endpoints 直接集成了与图像分类、目标检测、图像分割相关的兼容模型。对于其他模型，你可以使用自定义处理程序。由于我们还在 TensorFlow 中提供了许多来自 🤗Transformers 的视觉模型用于部署，我们建议使用自定义处理程序或遵循这些资源：

- [在 Hugging Face 上用 TF 服务开发 TensorFlow 视觉模型](https://huggingface.co/blog/tf-serving-vision)
- [在 Kubernets 上用 TF 服务开发 ViT](https://huggingface.co/blog/deploy-tfserving-kubernetes)
- [在 Vertex AI 上开发 ViT](https://huggingface.co/blog/deploy-vertex-ai)
- [用 TFX 和 Vertex AI 开发 ViT](https://github.com/deep-diver/mlops-hf-tf-vision-models)

## 结论

在这篇文章中，我们向你简要介绍了 Hugging Face 生态系统目前为下一代计算机视觉应用提供的支持。我们希望你会喜欢使用这些产品来可靠地构建应用。

不过还有很多工作要做。 以下是您可以期待看到的一些内容：

- 🤗 Datasets 对视频的直接支持
- 支持更多和工业界相关的任务，比如图像相似性
- 图像数据集与 TensorFlow 的交互
- 来自 🤗Hugging Face 社区关于计算机视觉的课程

像往常一样，我们欢迎你的补丁、PR、模型 checkpoints、数据集和其他贡献！🤗

*Acknowlegements: Thanks to Omar Sanseviero, Nate Raw, Niels Rogge, Alara Dirik, Amy Roberts, Maria Khalusova, and Lysandre Debut for their rigorous and timely reviews on the blog draft. Thanks to Chunte Lee for creating the blog thumbnail.*



> *原文：[The State of Computer Vision at Hugging Face ](https://huggingface.co/blog/cv_state)*
>
> *译者：AIboy1993（李旭东）*