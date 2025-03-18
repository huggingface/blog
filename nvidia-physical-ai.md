---
title: "NVIDIA's GTC 2025 Announcement for Physical AI Developers: New Open Models and Datasets" 
thumbnail: /blog/assets/nvidia-physical-ai/thumbnail.png
authors:
- user: mingyuliutw
  guest: true
  org: nvidia
- user: hannamao
  guest: true
  org: nvidia
- user: jwgu
  guest: true
  org: nvidia
- user: PranjaliJoshi
  guest: true
  org: nvidia
- user: asawareeb
  guest: true
  org: nvidia
---

| ![Humanoid pick and place](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/physical-ai/humanoid.gif) |
| :--: |
| *NVIDIA Isaac GR00T N1 used in object manipulation.* |

At its annual GTC conference, NVIDIA has unveiled a trio of groundbreaking open-source releases aimed at accelerating physical AI development. Release of a new suite of world foundation models (WFMs) with multicontrols called **Cosmos Transfer**, a highly curated **Physical AI Dataset**, and the first open model for general humanoid reasoning called **NVIDIA Isaac GR00T N1** - represent a significant leap forward in physical AI technology, offering developers powerful tools and resources to advance robotics systems, and enhance autonomous vehicle technology.

## New World Foundation Model - Cosmos Transfer

[Cosmos Transfer](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e), the latest addition to NVIDIA's Cosmos™ world foundation models (WFMs), introduces a new level of control and accuracy in generating virtual world scenes.

Available in 7 billion parameter size, the model utilizes multicontrols to guide the generation of high-fidelity world scenes from structural inputs, ensuring precise spatial alignment and scene composition.

### How it works

The model is built by training individual ControlNets separately for each sensor modality used to capture the simulated world.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/physical-ai/3D_bounding_box_map.gif" alt="3D bounding box map input" style="width: 100%;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/physical-ai/trajectory_map.gif" alt="Trajectory map input" style="width: 100%;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/physical-ai/depth_map.gif" alt="Depth map input" style="width: 100%;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/physical-ai/segmentation_map.gif" alt="Segmentation map input" style="width: 100%;">
</div>
<p style="text-align: center; font-style: italic;">Input types include 3D bounding box map, Trajectory map, Depth map, Segmentation map.</p>

- At inference time, developers can use various input types, including structured visual or geometric data such as segmentation maps, depth maps, edge maps, human motion keypoints, LiDAR scans, trajectories, HD maps, and 3D bounding boxes to guide the output.
- The control signals from each control branch are multiplied by their corresponding adaptive spatiotemporal control maps and then summed before being added to the transformer blocks of the base model. 
- The generated output is photorealistic video sequences with controlled layout, object placement, and motion. Developers can control the output in multiple ways, such as preserving structure and appearance or allowing appearance variations while maintaining structure.

<div style="display: flex; justify-content: space-between;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/physical-ai/output_1.gif" alt="Output 1" style="width: 30%;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/physical-ai/output_2.gif" alt="Output 2" style="width: 30%;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/physical-ai/output_3.gif" alt="Output 3" style="width: 30%;">
</div>
<p style="text-align: center; font-style: italic;">Outputs from Cosmos Transfer varying environments and weather conditions.</p>

Cosmos Transfer coupled with the NVIDIA Omniverse platform is driving controllable synthetic data generation for robotics and autonomous vehicle development at scale. Find more Cosmos Transfer [Examples](https://github.com/nvidia-cosmos/cosmos-transfer1#examples) on GitHub.

Cosmos Transfer [samples](https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV) built using post-training base model are also available for autonomous vehicles. 

## Open Physical AI Dataset

NVIDIA has also released **Physical AI Dataset**, an open-source dataset on [Hugging Face](https://huggingface.co/collections/nvidia/physical-ai-67c643edbb024053dcbcd6d8) for developing physical AI. This commercial-grade, pre-validated dataset consists of 15 terabytes of data representing more than 320,000 trajectories for robotics training, plus up to 1,000 Universal Scene Description (OpenUSD) assets, including a SimReady collection.

The dataset is designed for post-training foundation models like Cosmos Predict world foundation models, providing developers with high-quality, diverse data to enhance their AI models.

## Purpose Built Model for Humanoids - NVIDIA Isaac GR00T N1

Another exciting announcement is the release of [NVIDIA Isaac GR00T N1](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/), the world's first open foundation model for generalized humanoid robot reasoning and skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments. The NVIDIA Isaac GR00T-N1-2B model is available on [Hugging Face](https://huggingface.co/nvidia/GR00T-N1-2B).

Isaac GR00T N1 was trained on an expansive humanoid dataset, consisting of real captured data, synthetic data generated using components of the [NVIDIA Isaac GR00T Blueprint](https://developer.nvidia.com/blog/building-a-synthetic-motion-generation-pipeline-for-humanoid-robot-learning/), and internet-scale video data. It is adaptable through post-training for specific embodiments, tasks and environments. 

Isaac GR00T N1 uses a single model and set of weights to enable manipulation behaviors on various humanoid robots, such as the [Fourier GR-1](https://www.fftai.com/products-gr1) and [1X Neo](https://www.1x.tech/neo). It demonstrates robust generalization across a range of tasks, including grasping and manipulating objects with one or both arms, as well as transferring items between arms. It can also execute complex, multi-step tasks that require sustained contextual understanding and the integration of diverse skills. These capabilities make it well-suited for applications in material handling, packaging, and inspection.

Isaac GR00T N1 features a dual-system architecture inspired by human cognition, consisting of the following complementary components:

- **Vision-Language Model (System 2)**: This methodical thinking system is based on [NVIDIA-Eagle](https://huggingface.co/NVEagle) with [SmolLM-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B). It interprets the environment through vision and language instructions, enabling robots to reason about their environment and instructions, and plan the right actions.
- **Diffusion Transformer (System 1)**: This action model generates continuous actions to control the robot's movements, translating the action plan made by System 2 into precise, continuous robot movements.

## Path Forward

Post-training is the path forward to advancing autonomous systems, creating specialized models for downstream physical AI tasks.

Check out GitHub for [Cosmos Predict](https://github.com/nvidia-cosmos/cosmos-predict1) and [Cosmos Transfer](https://github.com/nvidia-cosmos/cosmos-transfer1) inference scripts. Explore the Cosmos Transfer [research paper](https://research.nvidia.com/labs/dir/cosmos-transfer1/) for more details.

The NVIDIA Isaac GR00T-N1-2B model is available on [Hugging Face](https://huggingface.co/nvidia/GR00T-N1-2B). Sample datasets and PyTorch scripts for post-training using custom user datasets, which is compatible with the Hugging Face LeRobot format are available on [GitHub](http://github.com/NVIDIA/Isaac-GR00T). For more information about the Isaac GR00T N1 model, see the [research paper](https://research.nvidia.com/publication/2025-03_nvidia-isaac-gr00t-n1-open-foundation-model-humanoid-robots).

Follow [NVIDIA on Hugging Face](https://huggingface.co/nvidia) for more updates.
