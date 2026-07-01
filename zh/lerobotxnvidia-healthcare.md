---
title: "使用 NVIDIA Isaac 构建医疗机器人：从仿真到部署"
thumbnail: /blog/assets/lerobotxnvidia-healthcare/thumbnail.png
authors:
- user: imstevenpmwork
- user: diazandr3s
translators:
- user: chenglu
---

# 使用 NVIDIA Isaac 构建医疗机器人：从仿真到部署

## 摘要

一篇实用指南，手把手教你如何采集数据、训练策略，并将自动化医疗机器人工作流程部署到真实硬件上。

## 目录

* [使用 NVIDIA Isaac 构建医疗机器人：从仿真到部署](#使用-nvidia-isaac-构建医疗机器人从仿真到部署)

  * [摘要](#摘要)
  * [目录](#目录)
  * [简介](#简介)
  * [SO-ARM 入门工作流：构建一台手术辅助机器人](#so-arm-入门工作流构建一台手术辅助机器人)

    * [技术实现](#技术实现)
    * [仿真与现实结合的训练方法](#仿真与现实结合的训练方法)
    * [硬件要求](#硬件要求)
    * [数据采集实现](#数据采集实现)
    * [仿真遥操作控制](#仿真遥操作控制)
    * [模型训练流程](#模型训练流程)
  * [完整的仿真采集-训练-评估流程](#完整的仿真采集-训练-评估流程)

    * [在仿真中生成合成数据](#在仿真中生成合成数据)
    * [训练和评估策略](#训练和评估策略)
    * [将模型转换为 TensorRT](#将模型转换为-tensorrt)
  * [快速开始](#快速开始)

    * [资源链接](#资源链接)

## 简介

仿真一直是医学影像中弥补数据缺口的重要手段，但在医疗机器人领域，它过去往往速度太慢、系统割裂，或难以迁移到现实应用中。

NVIDIA Isaac for Healthcare 是一个专为 AI 医疗机器人开发者打造的框架，提供从数据采集到训练、评估再到部署的全流程工具链，适用于仿真环境与真实硬件。特别是在 v0.4 版本中，Isaac 提供了一个 [SO-ARM 入门工作流](https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/workflows/so_arm_starter/README.md) 以及 [自定义手术室教程](https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/tutorials/assets/bring_your_own_or/README.md)，帮助开发者低门槛快速构建并验证自动化手术机器人。

本文将带你深入了解这一工作流及其技术细节，帮助你前所未有地快速搭建手术助手机器人。

## SO-ARM 入门工作流：构建一台手术辅助机器人

SO-ARM 入门工作流提供了一种全新的方式来探索手术辅助任务，提供了一整套从采集数据到部署到真实设备的端到端流程：

* 使用 LeRobot 平台与 SO-ARM 采集真实与仿真数据
* 微调 GR00t N1.5 模型，在 IsaacLab 中评估，然后部署到硬件

这套流程为开发者提供了一个安全、可重复的训练环境，可在进入真实手术室前不断优化机器人技能。

### 技术实现

该工作流采用三阶段流程，整合了仿真与真实硬件：

1. **数据采集**：结合仿真与真实环境中的遥操作演示，使用 SO101 和 LeRobot 平台
2. **模型训练**：在混合数据集上使用双摄像头视觉输入微调 GR00T N1.5
3. **策略部署**：基于 RTI DDS 通信协议，将训练好的模型实时运行在真实硬件上

值得一提的是，超过 93% 的策略训练数据来自仿真环境，充分说明仿真技术在缩小机器人数据鸿沟方面的优势。

### 仿真与现实结合的训练方法

现实世界的机器人训练成本高、操作难度大；而纯粹的仿真又很难完全还原现实复杂性。该流程通过约 70 个仿真演示和 10–20 个真实演示相结合，在模拟多样场景的同时，也保留真实数据的可靠性。最终训练出的策略能更好地泛化，不局限于单一环境。

### 硬件要求

工作流所需的硬件包括：

* **GPU**：支持 RT Core 的架构（Ampere 或更新）且显存 ≥30GB，用于运行 GR00T N1.5 推理
* **SO-ARM101 Follower**：6 自由度高精度机械臂，配备手腕摄像头与房间摄像头，采用 WOWROBO 视觉组件，并通过 3D 打印转接头固定
* **SO-ARM101 Leader**：6 自由度的遥操作控制器，用于采集专家演示

所有仿真、训练和部署任务（部署需使用 3 台计算机）均可在一台 [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) 上完成。

### 数据采集实现

![so100-healthcare-real-demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/nvidia-healthcare/lerobotxnvidia-healthcare-real-demo.gif)

**在真实环境中采集数据（适用于 SO-ARM101 硬件或任何 LeRobot 支持的版本）：**

```bash
python lerobot-record \ 
  --robot.type=so101_follower \ 
  --robot.port=<follower_port_id> \ 
  --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, room: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \ 
  --robot.id=so101_follower_arm \ 
  --teleop.type=so101_leader \ 
  --teleop.port=<leader_port_id> \ 
  --teleop.id=so101_leader_arm \ 
  --dataset.repo_id=<user>/surgical_assistance/surgical_assistance \ 
  --dataset.num_episodes=15 \ 
  --dataset.single_task="为外科医生准备并递送手术器械"
```

---

**在仿真环境中采集数据：**

![so100-healthcare-sim-demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/nvidia-healthcare/lerobotxnvidia-healthcare-sim-demo.gif)

```bash
# 使用键盘遥操作
python -m simulation.environments.teleoperation_record \ 
  --enable_cameras \ 
  --record \ 
  --dataset_path=/path/to/save/dataset.hdf5 \ 
  --teleop_device=keyboard

# 使用 SO-ARM101 Leader 机械臂进行遥操作
python -m simulation.environments.teleoperation_record \ 
  --port=<your_leader_arm_port_id> \ 
  --enable_cameras \ 
  --record \ 
  --dataset_path=/path/to/save/dataset.hdf5 
```


### 仿真遥操作控制

没有 SO-ARM101 硬件的用户可以使用键盘控制仿真机械臂：

* 关节 1（肩部旋转）：Q（+）/ U（-）
* 关节 2（肩部抬起）：W（+）/ I（-）
* 关节 3（肘部弯曲）：E（+）/ O（-）
* 关节 4（手腕弯曲）：A（+）/ J（-）
* 关节 5（手腕旋转）：S（+）/ K（-）
* 关节 6（夹爪开合）：D（+）/ L（-）
* R 键：重置录制环境
* N 键：标记当前演示成功

### 模型训练流程

完成仿真与真实数据采集后，将其转换并组合训练：

```bash
# 将仿真数据转为 LeRobot 格式
python -m training.hdf5_to_lerobot \ 
  --repo_id=surgical_assistance_dataset \ 
  --hdf5_path=/path/to/your/sim_dataset.hdf5 \ 
  --task_description="自动执行手术器械准备与递送任务" 

# 在混合数据集上微调 GR00T N1.5
python -m training.gr00t_n1_5.train \ 
  --dataset_path /path/to/your/surgical_assistance_dataset \ 
  --output_dir /path/to/surgical_checkpoints \ 
  --data_config so100_dualcam 
```

训练后的模型可处理自然语言指令，如“为医生准备手术刀”或“把镊子递给我”，并控制机器人执行相应动作。借助 LeRobot 的最新版本（0.4.0），你可以直接在 LeRobot 中原生训练 Gr00t N1.5！

## 完整的仿真采集-训练-评估流程

仿真在形成“采集→训练→评估→部署”的循环中最具价值。

从 v0.3 开始，IsaacLab 完整支持这一流程：

### 在仿真中生成合成数据

* 使用键盘或硬件控制器遥操作机器人
* 记录多摄像头视角、机器人状态与动作
* 构建包含极端情况的多样化数据集，这些情况在现实中难以安全采集

### 训练和评估策略

* 深度集成 Isaac Lab 的强化学习框架（如 PPO）
* 可同时运行数千个仿真环境
* 内置轨迹分析与成功率指标
* 支持在多种情境下进行统计验证

### 将模型转换为 TensorRT

* 自动化优化以适应生产部署需求
* 支持动态输入形状和多摄像头推理
* 提供性能基准测试工具，确保实时运行能力

这极大地缩短了从实验到部署的周期，让 sim2real 成为日常开发的一部分。

## 快速开始

SO-ARM 入门工作流已开放，立即开始：

1. 克隆代码库：
   `git clone https://github.com/isaac-for-healthcare/i4h-workflows.git`

2. 选择工作流：
   推荐从 SO-ARM 入门工作流开始，也可以探索其他工作流

3. 运行初始化脚本：
   每个工作流都带有自动化的安装脚本（如 `tools/env_setup_so_arm_starter.sh`）

### 资源链接

* [GitHub 项目地址](https://github.com/isaac-for-healthcare/i4h-workflows)：完整工作流实现
* [官方文档](https://isaac-for-healthcare.github.io/i4h-docs/)：安装与使用说明
* [GR00T 模型页面](https://huggingface.co/nvidia/GR00T-N1.5-3B)：预训练模型下载
* [硬件指南](https://huggingface.co/docs/lerobot/so101)：SO-ARM101 安装配置说明
* [LeRobot 项目](https://github.com/huggingface/lerobot)：端到端机器人学习平台
