---
title: "LeRobot v0.4.0：全面提升开源机器人的学习能力"
thumbnail: /blog/assets/lerobot-release-v040/thumbnail.png
authors:
- user: imstevenpmwork
- user: aractingi
- user: pepijn223
- user: CarolinePascal
- user: jadechoghari
- user: fracapuano
- user: AdilZtn
- user: nepyope
- user: thomwolf
translators:
- user: chenglu
---

# LeRobot v0.4.0：全面提升开源机器人的学习能力

我们非常高兴地宣布，LeRobot 迎来一系列重大升级，让开源的机器人学习比以往更强大、更可扩展、也更易用！从重构的数据集到灵活的编辑工具、新的仿真环境，以及面向硬件的全新插件系统，LeRobot 正在持续演进，以满足前沿具身智能（Embodied AI）不断发展的需求。

## 简要总结
LeRobot v0.4.0 为开源机器人领域带来重要升级：引入可扩展的 Datasets v3.0、强大的新 VLA（视觉-语言-动作）模型如 PI0.5 与 GR00T N1.5，以及全新的插件系统，简化硬件集成。该版本还新增对 LIBERO 与 Meta-World 仿真的支持、简化多 GPU 训练，并上线全新的 Hugging Face 机器人学习课程。

## 目录

- [LeRobot v0.4.0：全面提升开源机器人的学习能力](#LeRobot v0.4.0：全面提升开源机器人的学习能力)
- [简要总结](#简要总结)
- [数据集：为下一波大规模机器人学习做好准备](#数据集为下一波大规模机器人学习做好准备)
  - [Datasets v3.0 有何新变化？](#datasets-v30-有何新变化)
  - [新特性：数据集编辑工具！](#新特性数据集编辑工具)
- [仿真环境：扩展你的训练场](#仿真环境扩展你的训练场)
  - [LIBERO 支持](#libero-支持)
  - [Meta-World 集成](#meta-world-集成)
- [代码库：人人可用的强力工具](#代码库人人可用的强力工具)
  - [全新的数据处理 Pipeline](#全新的数据处理-pipeline)
  - [多 GPU 训练更简单](#多-gpu-训练更简单)
- [策略：释放开放世界泛化能力](#策略释放开放世界泛化能力)
  - [PI0 与 PI0.5](#pi0-与-pi05)
  - [GR00T N1.5](#gr00t-n15)
- [机器人：插件系统引领硬件集成新纪元](#机器人插件系统引领硬件集成新纪元)
  - [核心优势](#核心优势)
  - [Reachy 2 集成](#reachy-2-集成)
  - [手机集成](#手机集成)
- [Hugging Face 机器人学习课程](#hugging-face-机器人学习课程)
  - [深入讲解：现代机器人学习教程](#深入讲解现代机器人学习教程)
- [团队总结](#团队总结)

## 数据集：为下一波大规模机器人学习做好准备

我们彻底重构了数据集基础设施，推出 **LeRobotDataset v3.0**，采用全新的分块式 Episode 格式与流式读取能力。这对于处理超大规模数据集（如 [OXE](https://huggingface.co/collections/lerobot/open-x-embodiment)（Open X Embodiment）与 [Droid](https://huggingface.co/datasets/lerobot/droid_1.0.1)）是一次范式跃迁，带来前所未有的效率与可扩展性。

### Datasets v3.0 有何新变化？

* **分块式 Episodes，面向超大规模**：新格式支持 OXE 量级（> 400 GB）的数据集，显著提升可扩展性。
* **高效视频存储与流式读取**：更快的加载速度与顺畅的视频数据流式访问。
* **统一的 Parquet 元数据**：告别分散的 JSON！所有 Episode 的元数据现统一存放于结构化的 Parquet 文件中，便于管理与访问。
* **更快的加载与更好的性能**：显著缩短数据集初始化时间，内存使用更高效。

我们还提供了转换脚本，帮助你将现有 v2.1 数据集一键迁移到新的 v3.0 格式，确保平滑过渡。更多细节可阅读我们此前的 [博客文章](https://huggingface.co/blog/lerobot-datasets-v3)。开源机器人的学习能力持续升级中！

### 新特性：数据集编辑工具！

使用 LeRobot 数据集从未如此轻松！我们新增了一套强大的数据集灵活编辑工具。

借助全新的命令行工具 `lerobot-edit-dataset`，你可以：

* 从现有数据集中删除指定的 Episodes。
* 按比例或 Episode 索引拆分数据集。
* 轻松添加或移除特征字段。
* 将多个数据集合并为一个统一数据集。

```bash
# 将多个数据集合并为单一数据集
lerobot-edit-dataset \
    --repo_id lerobot/pusht_merged \
    --operation.type merge \
    --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"

# 删除部分 episodes 并保存为新数据集（保留原数据集）
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --new_repo_id lerobot/pusht_after_deletion \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"
```

这些工具将大幅简化你的工作流，让你以前所未有的方式策划与优化机器人数据集。更多详情请查阅 [文档](https://huggingface.co/docs/lerobot/using_dataset_tools)！

## 仿真环境：扩展你的训练场

我们持续扩展 LeRobot 的仿真能力，为你的机器人策略提供更丰富、更多样化的训练环境。

![libero-demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.4.0/lerobot-libero-groot-v040.gif)

### LIBERO 支持

LeRobot 现已正式支持 [LIBERO](https://libero-project.github.io/intro.html)——这是 VLA（视觉-语言-动作）策略中规模最大的开源基准之一，涵盖超过 130 个任务！这一步为打造 VLA 策略的首选评测枢纽奠定了基础，提供便捷的集成方式与统一的评测配置。

前往查看 [LIBERO 数据集](https://huggingface.co/datasets/HuggingFaceVLA/libero) 与我们的 [文档](https://huggingface.co/docs/lerobot/en/libero) 开始上手！

### Meta-World 集成

我们已集成 [Meta-World](https://meta-world.github.io)，它是评测机器人操作多任务与泛化能力的一流基准，包含 50+ 种多样化的操作任务。配合我们对 `gymnasium ≥ 1.0.0` 与 `mujoco ≥ 3.0.0` 的标准化使用，这一集成为确定性的随机种子与稳健的仿真基础提供了保障。

立即使用 [Meta-World 数据集](https://huggingface.co/datasets/lerobot/metaworld_mt50) 训练你的策略吧！

## 代码库：人人可用的强力工具

我们让机器人控制更加灵活与易用，解锁数据采集与模型训练的新可能。

### 全新的数据处理 Pipeline

让数据从机器人流向模型（再流回去！）并不容易。原始传感器数据、关节位置与语言指令，与人工智能模型期望的输入并不一致。模型需要在正确设备上的规范化、按批次的张量，而你的机器人硬件则需要特定格式的动作命令。

我们很高兴地推出 **Processors**：一个模块化的数据处理 Pipeline，可充当通用的“数据翻译器”。你可以把它想象为一条装配线，每个 `ProcessorStep` 只处理一个明确的工序——例如归一化、文本 Token 化、或将数据移到 GPU。

你可以将这些步骤串联起来，构建强大的 Pipeline，精准管理你的数据流。我们还提供了两类开箱即用的 Pipeline，进一步降低使用门槛：

* `PolicyProcessorPipeline`：面向模型。专为高性能训练与推理处理按批次的张量。
* `RobotProcessorPipeline`：面向硬件。以单条数据（如单次观测或动作）为粒度，服务于实时机器人控制。

```python
# 获取环境状态
obs = robot.get_observation()

# 重命名、打批、归一化、文本分词、移动到设备 ...
obs_processed = preprocess(obs)

# 推理
action = model.select_action(obs_processed)

# 反归一化、移动设备 ...
action_processed = postprocess(action)

# 执行动作
robot.send_action(action_processed)
```

这个系统让任何策略与任何机器人都能简单互联，确保你的数据在每一步都处于“刚刚好”的格式。详情可阅读我们的 [Processors 入门文档](https://huggingface.co/docs/lerobot/introduction_processors)。

### 多 GPU 训练更简单

大规模机器人策略的训练现在更快了！我们将 [Accelerate](https://github.com/huggingface/accelerate) 直接整合进训练 Pipeline，只需 **一条命令** 即可在多块 GPU 上无缝扩展你的实验：

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=$NUM_GPUs \
  $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --policy.repo_id=${HF_USER}/my_trained_policy \
  --policy.type=$POLICY_TYPE \
  # ... 更多训练配置参数
```

无论是对策略进行微调，还是开展大规模实验，LeRobot 现在都能替你处理分布式训练的全部复杂性。这意味着你可以大幅缩短训练时间：约 2 块 GPU 可减半，约 3 块 GPU 可降至三分之一，更多 GPU 效率更高。

查阅 [文档](https://huggingface.co/docs/lerobot/multi_gpu_training) 加速你的机器人学习！

## 策略：释放开放世界泛化能力

![groot-demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.4.0/lerobot-libero-groot2-v040.gif)

### PI0 与 PI0.5

在开源机器人领域的一个重要里程碑中，我们将 Physical Intelligence 的 **pi0** 与 **pi0.5** 策略集成进了 LeRobot！这些 VLA（视觉-语言-动作）模型在解决开放世界泛化问题上迈出了重要一步。那么，π0.5 的革命性体现在哪里？

* **开放世界泛化**：能够适应完全陌生的环境与情境，在物理、语义与环境层面实现跨域泛化。
* **异构数据共训练**：从多模态网页数据、自然语言指令、子任务命令与多环境机器人数据的多样组合中学习。
* **Physical Intelligence 合作**：特别感谢 [Physical Intelligence 团队](https://huggingface.co/physical-intelligence) 的开创性工作！

你可以在 Hugging Face Hub 上找到这些模型：[pi0.5_base](https://huggingface.co/lerobot/pi05_base)、[pi0_base](https://huggingface.co/lerobot/pi0_base) 及它们的 Libero 微调版本。更多细节请参考 [Physical Intelligence 的研究博客](https://www.physicalintelligence.company/blog/pi05)。

### GR00T N1.5

另一项令人振奋的进展是，我们与 NVIDIA 机器人团队携手，将 **GR00T N1.5** 集成进 LeRobot！这是一款面向泛化的开源基础模型，能够进行跨本体的推理与技能迁移。它接收多模态输入（如语言与图像），可在多样环境中执行复杂的操作任务，标志着通用机器人又一大步。GR00T N1.5 为何与众不同？

* **泛化推理与技能**：作为跨本体的基础模型，GR00T N1.5 擅长泛化推理与操作任务，并提升了语言跟随能力。
* **大规模异构训练**：训练数据覆盖真实人形机器人采集数据、NVIDIA Isaac GR00T Blueprint 生成的合成数据，以及互联网规模的视频数据。
* **与 NVIDIA 合作**：我们很高兴与 [NVIDIA 团队](https://huggingface.co/nvidia) 合作，将这一前沿模型带给开源的 LeRobot 社区！

你可以在 Hugging Face Hub 上找到该模型：[GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B)。更多信息请查看 [NVIDIA 的研究页面](https://research.nvidia.com/labs/gear/gr00t-n1_5/) 与 [官方 GitHub 仓库](https://github.com/NVIDIA/Isaac-GR00T)。

这些策略在 `lerobot` 中的原生集成，让机器人学习更开放、更可复现。立即试用、分享你的训练运行结果，让我们共同推动具身智能的前沿！

## 机器人：插件系统引领硬件集成新纪元

对硬件爱好者而言的重磅消息！我们发布了全新的插件系统，彻底改造了第三方硬件与 LeRobot 的集成方式。现在，只需一次 `pip install`，就能连接任意机器人、相机或遥操作设备，无需修改核心库。

### 核心优势

* **可扩展性**：在独立的 Python 包中开发并集成自定义硬件。
* **规模化**：支持不断增长的设备生态，而不会“增肥”核心库。
* **社区友好**：降低社区贡献门槛，促进更高效的协作。

想要创建自己的插件？请阅读我们的 [文档](https://huggingface.co/docs/lerobot/integrate_hardware#using-your-own-lerobot-devices-)。

```bash
pip install lerobot_teleoperator_my_awesome_teleop
lerobot-teleoperate --teleop.type=my_awesome_teleop
```

### Reachy 2 集成

得益于全新插件系统，我们已将 Pollen Robotics 的 [Reachy 2](https://www.pollen-robotics.com/reachy/) 集成到 LeRobot 中！Reachy 2 同时支持真实机器人控制与仿真，让你可以立即开展遥操作与自主演示实验。

### 手机集成

得益于强大的新 Pipeline 系统，你现在可以**直接用手机**（iOS/Android）遥操作你的从动机械臂。手机作为遥操作设备，`RobotProcessor` Pipeline 负责全部数据变换，让你轻松在不同动作空间（如末端执行器空间）驱动机器人。[查看示例](https://github.com/huggingface/lerobot/tree/main/examples/phone_to_so100)。

## Hugging Face 机器人学习课程

我们上线了一门全面、可自学、且完全**开源**的课程，旨在让机器人学习真正“人人可学”！如果你对真实世界中的机器人如何学习感兴趣，这是绝佳的起点。

在这门课程中，你将学到：

* 理解经典机器人学的基础知识。
* 使用生成式模型进行模仿学习（VAE、扩散模型等）。
* 将强化学习应用于真实机器人。
* 探索最新的通用机器人策略，如 PI0 与 SmolVLA。

加入 [Hugging Face Robotics 组织](https://huggingface.co/robotics-course) 一起学习吧！

### 深入讲解：现代机器人学习教程

我们还发布了一篇**动手实践**的现代机器人学习教程，系统梳理近期的关键进展。该指南从第一性原理重新推导现代技术，并提供可直接运行的示例代码，全面基于 LeRobot 与 Hugging Face。

教程托管在一个 [Space](https://huggingface.co/spaces/lerobot/robot-learning-tutorial) 中，包含大量基于 LeRobot 的实操示例，所有模型与数据集均在 Hugging Face Hub 上。同时你也可以查看 [我们的论文](https://huggingface.co/papers/2510.12403) 以获得更全面的概览。

## 团队总结

除了以上重大功能，这个版本还包含大量的错误修复、文档改进、依赖更新、更多示例与更好的基础设施，只为让你在使用 LeRobot 时获得更顺滑、更可靠的体验。

衷心感谢**每一位社区成员**的宝贵贡献、反馈与支持。我们对开源机器人的未来无比期待，也迫不及待地想与你一起构建下一步！

更多精彩，敬请期待 🤗 现在就从 [这里](https://github.com/huggingface/lerobot) 开始吧！
—— LeRobot 团队 ❤️
