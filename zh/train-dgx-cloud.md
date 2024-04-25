---
title: "在 NVIDIA DGX Cloud上使用 H100 GPU 轻松训练模型"
thumbnail: /blog/assets/train-dgx-cloud/thumbnail.jpg
authors:
- user: philschmid
- user: jeffboudier
- user: rafaelpierrehf
- user: abhishek
translators:
- user: chenglu
---

# 在 NVIDIA DGX Cloud上使用 H100 GPU 轻松训练模型

今天，我们正式宣布推出 **DGX 云端训练 (Train on DGX Cloud)** 服务，这是 Hugging Face Hub 上针对企业 Hub 组织的全新服务。

通过在 DGX 云端训练，你可以轻松借助 NVIDIA DGX Cloud的高速计算基础设施来使用开放的模型。这项服务旨在让企业 Hub 的用户能够通过几次点击，就在 [Hugging Face Hub](https://huggingface.co/models) 中轻松访问最新的 NVIDIA H100 Tensor Core GPU，并微调如 Llama、Mistral 和 Stable Diffusion 这样的流行生成式 AI (Generative AI) 模型。

<div align="center"> 
  <img src="/blog/assets/train-dgx-cloud/thumbnail.jpg" alt="Thumbnail"> 
</div>

## GPU 不再是稀缺资源

这一新体验基于我们去年宣布的[战略合作](https://nvidianews.nvidia.com/news/nvidia-and-hugging-face-to-connect-millions-of-developers-to-generative-ai-supercomputing)，旨在简化 NVIDIA 加速计算平台上开放生成式 AI 模型的训练和部署。开发者和机构面临的主要挑战之一是 GPU 资源稀缺，以及编写、测试和调试 AI 模型训练脚本的工作繁琐。在 DGX 云上训练为这些挑战提供了简便的解决方案，提供了对 NVIDIA GPUs 的即时访问，从 NVIDIA DGX Cloud上的 H100 开始。此外，该服务还提供了一个简洁的无代码训练任务创建体验，由 Hugging Face AutoTrain 和 Hugging Face Spaces 驱动。

通过 [企业版的 HF Hub](https://huggingface.co/enterprise)，组织能够为其团队提供强大 NVIDIA GPU 的即时访问权限，只需按照训练任务所用的计算实例分钟数付费。

> 在 DGX 云端训练是目前训练生成式 AI 模型最简单、最快速、最便捷的方式，它结合了强大 GPU 的即时访问、按需付费和无代码训练，这对全球的数据科学家来说将是一次变革性的进步！
>
> —— Abhishek Thakur, Hugging Face AutoTrain 团队创始人

> 今天发布的 Hugging Face Autotrain，得益于 DGX 云的支持，标志着简化 AI 模型训练过程向前迈出了重要一步，通过将 NVIDIA 的云端 AI 超级计算机与 Hugging Face 的友好界面结合起来，我们正在帮助各个组织加速他们的 AI 创新步伐。
>
> —— Alexis Bjorlin, NVIDIA DGX Cloud 副总裁

## 操作指南

在 NVIDIA DGX Cloud 上训练 Hugging Face 模型变得非常简单。以下是针对如何微调 Mistral 7B 的分步教程。

> 注意：你需要访问一个拥有 [企业版的 HF Hub](https://huggingface.co/enterprise) 订阅的组织账户，才能使用在 DGX 云端训练的服务

你可以在支持的生成式 AI 模型的模型页面上找到在 DGX 云端训练的选项。目前，它支持以下模型架构：Llama、Falcon、Mistral、Mixtral、T5、Gemma、Stable Diffusion 和 Stable Diffusion XL。

<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/01%20model%20card.png" alt="Model Card"> 
</div>

点击“训练 (Train)”菜单，并选择“NVIDIA DGX Cloud”选项，这将打开一个页面，让你可以选择你的企业组织。

<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/02%20select%20organization.png" alt="Organization Selection"> 
</div>


接下来，点击“Create new Space”。当你首次使用在 DGX 云端训练时，系统将在你的组织内创建一个新的 Hugging Face 空间，使你可以利用 AutoTrain 创建将在 NVIDIA DGX Cloud上执行的训练任务。当你日后需要创建更多训练任务时，系统将自动将你重定向到已存在的 AutoTrain Space 应用。

进入 AutoTrain Space 应用后，你可以通过配置硬件、基础模型、任务和训练参数来设置你的训练任务。

<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/03%20start.png" alt="Create AutoTrain Job"> 
</div>

在硬件选择方面，你可以选择 NVIDIA H100 GPU，提供 1x、2x、4x 和 8x 实例，或即将推出的 L40S GPUs。训练数据集需要直接上传至“上传训练文件”区域，目前支持 CSV 和 JSON 文件格式。请确保根据以下示例正确设置列映射。对于训练参数，你可以直接在右侧的 JSON 配置中进行编辑，例如，将训练周期数从 3 调整为 2。

一切设置完成后，点击“开始训练”即可启动你的训练任务。AutoTrain 将验证你的数据集，并请求你确认开始训练。

<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/04%20success.png" alt="Launched Training Job"> 
</div>

你可以通过查看这个 Space 应用的“Logs 日志”来查看训练进度。

<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/05%20logs.png" alt="Training Logs"> 
</div>

训练完成后，你微调后的模型将上传到 Hugging Face Hub 上你所选择的命名空间内的一个新的私有仓库中。

从今天起，所有企业 Hub 组织都可以使用在 DGX 云端训练的服务了！欢迎尝试并分享你的反馈！

## DGX 云端训练的定价

使用在 DGX 云端训练服务，将根据你训练任务期间使用的 GPU 实例分钟数来计费。当前的训练作业价格为：H100 实例每 GPU 小时 8.25 美元，L40S 实例每 GPU 小时 2.75 美元。作业完成后，费用将累加到你企业 Hub 组织当前的月度账单中。你可以随时查看企业 Hub 组织的计费设置中的当前和历史使用情况。


<table>
  <tr>
   <td>NVIDIA GPU
   </td>
   <td>GPU 显存
   </td>
   <td>按需计费价格（每小时）
   </td>
  </tr>
  <tr>
   <td><a href="https://www.nvidia.com/en-us/data-center/l40/">NVIDIA L40S</a>
   </td>
   <td>48GB
   </td>
   <td>$2.75
   </td>
  </tr>
  <tr>
   <td><a href="https://www.nvidia.com/de-de/data-center/h100/">NVIDIA H100</a>
   </td>
   <td>80 GB	
   </td>
   <td>$8.25
   </td>
  </tr>
</table>

例如，微调 1500 个样本的 Mistral 7B 在一台 NVIDIA L40S 上大约需要 10 分钟，成本约为 0.45 美元。

## 我们的旅程刚刚开始

我们很高兴能与 NVIDIA 合作，推动加速机器学习在开放科学、开源和云服务领域的普惠化。

通过 [BigCode](https://huggingface.co/bigcode) 项目的合作，我们训练了 [StarCoder 2 15B](https://huggingface.co/bigcode/starcoder2-15b)，这是一个基于超过 600 种编程语言训练的全开放、最先进的代码大语言模型（LLM）。

我们在开源方面的合作推动了新的 [optimum-nvidia 库](https://github.com/huggingface/optimum-nvidia) 的开发，加速了最新 NVIDIA GPUs 上大语言模型的推理，已经达到了 Llama 2 每秒 1200 Tokens 的推理速度。

我们在云服务方面的合作促成了今天的在 DGX 云端训练服务。我们还在与 NVIDIA 合作优化推理过程，并使加速计算对 Hugging Face 社区更容易受益。此外，Hugging Face 上一些最受欢迎的开放模型将出现在今天 GTC 上宣布的 [NVIDIA NIM 微服务](https://developer.nvidia.cn/zh-cn/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/) 上。

本周参加 GTC 的朋友们，请不要错过周三 3/20 下午 3 点 PT 的会议 [S63149](https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=S63149#/session/1704937870817001eXsB)，[Jeff](https://huggingface.co/jeffboudier) 将带你深入了解在 DGX 云端训练等更多内容。另外，不要错过下一期 Hugging Cast，在那里我们将现场演示在 DGX 云端训练，并且你可以直接向 [Abhishek](https://huggingface.co/abhishek) 和 [Rafael](https://huggingface.co/rafaelpierrehf) 提问，时间是周四 3/21 上午 9 点 PT / 中午 12 点 ET / 17h CET - [请在此注册](https://streamyard.com/watch/YfEj26jJJg2w)。