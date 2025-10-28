---
title: "Building a Healthcare Robot from Simulation to Deployment with NVIDIA Isaac" 
thumbnail: /blog/assets/lerobotxnvidia-healthcare/thumbnail.png
authors:
- user: imstevenpmwork
- user: diazandr3s
---

# Building a Healthcare Robot from Simulation to Deployment with NVIDIA Isaac

## TL;DR
A hands-on guide to collecting data, training policies, and deploying autonomous medical robotics workflows on real hardware

## Table-of-Contents
- [Building a Healthcare Robot from Simulation to Deployment with NVIDIA Isaac](#building-a-healthcare-robot-from-simulation-to-deployment-with-nvidia-isaac)
  - [TL;DR](#tldr)
  - [Table-of-Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [SO-ARM Starter Workflow; Building an Embodied Surgical Assistant](#so-arm-starter-workflow-building-an-embodied-surgical-assistant)
    - [Technical Implementation](#technical-implementation)
    - [Sim2Real Mixed Training Approach](#sim2real-mixed-training-approach)
    - [Hardware Requirements](#hardware-requirements)
    - [Data Collection Implementation](#data-collection-implementation)
    - [Simulation Teleoperation Controls](#simulation-teleoperation-controls)
    - [Model Training Pipeline](#model-training-pipeline)
  - [End-to-End Sim Collect–Train–Eval Pipelines](#end-to-end-sim-collecttraineval-pipelines)
    - [Generate Synthetic Data in Simulation](#generate-synthetic-data-in-simulation)
    - [Train and Evaluate Policies](#train-and-evaluate-policies)
    - [Convert Models to TensorRT](#convert-models-to-tensorrt)
  - [Getting Started](#getting-started)
    - [Resources](#resources)

## Introduction

Simulation has been a cornerstone in medical imaging to address the data gap. However, in healthcare robotics until now, it's often been too slow, siloed, or difficult to translate into real-world systems.
 
NVIDIA Isaac for Healthcare, a developer framework for AI healthcare robotics, enables healthcare robotics developers in solving these challenges via offering integrated data collection, training, and evaluation pipelines that work across both simulation and hardware. Specifically, the Isaac for Healthcare v0.4 release provides healthcare developers with an end-to-end [SO - ARM based starter workflow](https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/workflows/so_arm_starter/README.md) and [the bring your own operating room tutorial](https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/tutorials/assets/bring_your_own_or/README.md). The SO-ARM starter workflow lowers the barrier for MedTech developers to experience the full workflow from simulation to train to deployment and start building and validating autonomous on real hardware right away.
 
In this post, we'll walk through the starter workflow and its technical implementation details to help you build a surgical assistant robot in less time than ever imaginable before.

## SO-ARM Starter Workflow; Building an Embodied Surgical Assistant

The SO-ARM starter workflow introduces a new way to explore surgical assistance tasks, and providing developers with a complete end-to-end pipeline for autonomous surgical assistance:
 
* Collect real-world and synthetic data with SO-ARM using the LeRobot
* Fine-tune GR00t N1.5, evaluate in IsaacLab, then deploy to hardware
 
This workflow gives developers a safe, repeatable environment to train and refine assistive skills before moving into the Operating Room.

### Technical Implementation

The workflow implements a three-stage pipeline that integrates simulation and real hardware:
 
1. Data Collection: Mixed simulation and real-world teleoperation demonstrations using using SO101 and LeRobot
2. Model Training: Fine-tuning GR00T N1.5 on combined datasets with dual-camera vision
3. Policy Deployment: Real-time inference on physical hardware with RTI DDS communication
 
Notably, over 93% of the data used for policy training was generated synthetically in simulation, underscoring the strength of simulation in bridging the robotic data gap.

### Sim2Real Mixed Training Approach

The workflow combines simulation and real-world data to address the fundamental challenge that training robots in the real world is expensive and limited, while pure simulation often fails to capture real-world complexities. The approach uses approximately 70 simulation episodes for diverse scenarios and environmental variations, combined with 10-20 real-world episodes for authenticity and grounding. This mixed training creates policies that generalize beyond either domain alone.

### Hardware Requirements

The workflow requires:
 
* GPU: RT Core-enabled architecture (Ampere or later) with ≥30GB VRAM for GR00TN1.5 inference
* SO-ARM101 Follower: 6-DOF precision manipulator with dual-camera vision (wrist and room). The SO-ARM101 features WOWROBO vision components, including a wrist-mounted camera with a 3D-printed adapter
* SO-ARM101 Leader: 6-DOF Teleoperation interface for expert demonstration collection
 
Notably, developers could run all the simulation, training and deployment (3 computers needed for physical AI) on one [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).

### Data Collection Implementation

![so100-healthcare-real-demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/nvidia-healthcare/lerobotxnvidia-healthcare-real-demo.gif)

For real-world data collection with SO-ARM101 hardware or any other version supported in LeRobot:

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
  --dataset.single_task="Prepare and hand surgical instruments to surgeon" 
```

For simulation-based data collection:

![so100-healthcare-sim-demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/nvidia-healthcare/lerobotxnvidia-healthcare-sim-demo.gif)


```bash
# With keyboard teleoperation
python -m simulation.environments.teleoperation_record \ 
  --enable_cameras \ 
  --record \ 
  --dataset_path=/path/to/save/dataset.hdf5 \ 
  --teleop_device=keyboard

# With SO-ARM101 leader arm 
python -m simulation.environments.teleoperation_record \ 
  --port=<your_leader_arm_port_id> \ 
  --enable_cameras \ 
  --record \ 
  --dataset_path=/path/to/save/dataset.hdf5 
```

### Simulation Teleoperation Controls

For users without physical SO-ARM101 hardware, the workflow provides keyboard-based teleoperation with the following joint controls:

* Joint 1 (shoulder_pan): Q (+) / U (-)
* Joint 2 (shoulder_lift): W (+) / I (-)
* Joint 3 (elbow_flex): E (+) / O (-)
* Joint 4 (wrist_flex): A (+) / J (-)
* Joint 5 (wrist_roll): S (+) / K (-)
* Joint 6 (gripper): D (+) / L (-)
* R Key: Reset recording environment
* N Key: Mark episode as successful

### Model Training Pipeline

After collecting both simulation and real-world data, convert and combine datasets for training:

```bash
# Convert simulation data to LeRobot format
python -m training.hdf5_to_lerobot \ 
  --repo_id=surgical_assistance_dataset \ 
  --hdf5_path=/path/to/your/sim_dataset.hdf5 \ 
  --task_description="Autonomous surgical instrument handling and preparation" 

# Fine-tune GR00T N1.5 on mixed dataset 
python -m training.gr00t_n1_5.train \ 
  --dataset_path /path/to/your/surgical_assistance_dataset \ 
  --output_dir /path/to/surgical_checkpoints \ 
  --data_config so100_dualcam 
```

The trained model processes natural language instructions such as "Prepare the scalpel for the surgeon" or "Hand me the forceps" and executes the corresponding robotic actions. With LeRobot latest release (0.4.0) you will be able to fine-tune Gr00t N1.5 natively in LeRobot!

## End-to-End Sim Collect–Train–Eval Pipelines

Simulation is most powerful when it's part of a loop: collect → train → evaluate → deploy.

With v0.3, IsaacLab supports this full pipeline:

### Generate Synthetic Data in Simulation

* Teleoperate robots using keyboard or hardware controllers
* Capture multi-camera observations, robot states, and actions
* Create diverse datasets with edge cases impossible to collect safely in real environments

### Train and Evaluate Policies

* Deep integration with Isaac Lab's RL framework for PPO training
* Parallel environments (thousands of simulations simultaneously)
* Built-in trajectory analysis and success metrics
* Statistical validation across varied scenarios

### Convert Models to TensorRT

* Automatic optimization for production deployment
* Support for dynamic shapes and multi-camera inference
* Benchmarking tools to verify real-time performance

This reduces time from experiment to deployment and makes sim2real a practical part of daily development.

## Getting Started

Isaac for Healthcare SO-ARM Starter Workflow is available now. To get started:

1. Clone the repository: `git clone https://github.com/isaac-for-healthcare/i4h-workflows.git`
2. Choose a workflow: Start with the SO-ARM Starter Workflow for surgical assistance or explore other workflows
3. Run the setup: Each workflow includes an automated setup script (e.g., `tools/env_setup_so_arm_starter.sh`)

### Resources 

* [GitHub Repository](https://github.com/isaac-for-healthcare/i4h-workflows): Complete workflow implementations
* [Documentation](https://isaac-for-healthcare.github.io/i4h-docs/): Setup and usage guides
* [GR00T Models](https://huggingface.co/nvidia/GR00T-N1.5-3B): Pre-trained foundation models
* [Hardware Guides](https://huggingface.co/docs/lerobot/so101): SO-ARM101 setup instructions
* [LeRobot Repository](https://github.com/huggingface/lerobot): End-to-end robotics learning
