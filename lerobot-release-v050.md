---
title: "LeRobot v0.5.0: Scaling Every Dimension"
thumbnail: /blog/assets/lerobot-release-v050/thumbnail.png
authors:
  - user: imstevenpmwork
  - user: pepijn223
  - user: jadechoghari
  - user: CarolinePascal
  - user: lilkm
  - user: nepyope
  - user: Nico-robot
  - user: aractingi
  - user: VirgileBatto
  - user: thomwolf
---

# LeRobot v0.5.0: Scaling Every Dimension

With over 200 merged PRs and over 50 new contributors since v0.4.0, LeRobot v0.5.0 is our biggest release yet — expanding in every direction at once. More robots (including our first humanoid), more policies (including the comeback of autoregressive VLAs), faster datasets, simulation environments you can load straight from the Hub, and a modernized codebase running on Python 3.12 and Transformers v5. Whether you're training policies in simulation or deploying them on real hardware, v0.5.0 has something for you.

## TL;DR

LeRobot v0.5.0 adds full Unitree G1 humanoid support (whole-body control models), new policies –including Pi0-FAST autoregressive VLAs and Real-Time Chunking for responsive inference–, and streaming video encoding that eliminates wait times between recording episodes. The release also introduces EnvHub for loading simulation environments from the Hugging Face Hub, NVIDIA IsaacLab-Arena integration, and a major codebase modernization with Python 3.12+, Transformers v5, and third-party policy plugins.

## Table of Contents

- [LeRobot v0.5.0: Scaling Every Dimension](#lerobot-v050-scaling-every-dimension)
  - [TL;DR](#tldr)
  - [Table of Contents](#table-of-contents)
  - [Hardware: More Robots Than Ever](#hardware-more-robots-than-ever)
    - [Unitree G1 Humanoid](#unitree-g1-humanoid)
    - [OpenArm \& OpenArm Mini](#openarm--openarm-mini)
    - [More Robots](#more-robots)
    - [CAN Bus Motors](#can-bus-motors)
  - [Policies: A Growing Model Zoo](#policies-a-growing-model-zoo)
    - [Pi0-FAST: Autoregressive VLAs](#pi0-fast-autoregressive-vlas)
    - [Real-Time Chunking (RTC)](#real-time-chunking-rtc)
    - [Wall-X](#wall-x)
    - [X-VLA](#x-vla)
    - [SARM](#sarm)
    - [PEFT Support](#peft-support)
  - [Datasets: Faster Recording, Faster Training](#datasets-faster-recording-faster-training)
    - [Streaming Video Encoding](#streaming-video-encoding)
    - [10x Faster Image Training, 3x Faster Encoding](#10x-faster-image-training-3x-faster-encoding)
    - [New Dataset Tools](#new-dataset-tools)
  - [EnvHub: Environments from the Hub](#envhub-environments-from-the-hub)
    - [NVIDIA IsaacLab-Arena](#nvidia-isaaclab-arena)
  - [Codebase: A Modern Foundation](#codebase-a-modern-foundation)
  - [Community \& Ecosystem](#community--ecosystem)
  - [Final Thoughts](#final-thoughts)

## Hardware: More Robots Than Ever

LeRobot v0.5.0 dramatically expands the roster of supported hardware — from arms and mobile robots to a full humanoid.

### Unitree G1 Humanoid

The biggest hardware addition in this release: **full Unitree G1 humanoid support**. This is LeRobot's first humanoid integration, and it's comprehensive:

- **Locomotion**: Walk, navigate, and move through environments.
- **Manipulation**: Perform dexterous object manipulation tasks.
- **Teleoperation**: Control the G1 remotely with an intuitive teleoperation interface.
- **Whole-Body Control (WBC)**: Coordinate locomotion and manipulation simultaneously for complex, real-world tasks.

The G1 integration represents a major step toward general-purpose robotics within LeRobot — moving beyond tabletop arms into full-body embodied AI. Try it out yourself by following the [documentation](https://huggingface.co/docs/lerobot/unitree_g1).

![unitree-boss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.5.0/unitree_bosswalk.JPG)

### OpenArm & OpenArm Mini

We've added support for the [**OpenArm**](https://openarm.dev) robot and its companion **OpenArm Mini** teleoperator. OpenArm is a capable robot arm with full LeRobot integration, and the Mini serves as its natural teleoperation device. Both support **bi-manual configurations**, enabling dual-arm setups for more complex manipulation tasks. Check it out in the [documentation](https://huggingface.co/docs/lerobot/openarm).

### More Robots

The hardware ecosystem keeps growing:

- [**Earth Rover**](https://shop.frodobots.com/products/miniplus): Our first mobile robot integration, bringing LeRobot to outdoor navigation and ground-level robotics.
- [**OMX Robot**](https://ai.robotis.com/omx/hardware_omx.html): A new robot arm with configurable gripper settings and calibration support.
- **SO-100/SO-101 Consolidation**: We've unified the SO-100 and SO-101 implementations into a single, cleaner codebase — including bi-manual setups. Less code duplication, easier maintenance, same great robots.

### CAN Bus Motors

New motor controller support via CAN (Controller Area Network) bus opens the door to higher-performance actuators:

- [**RobStride**](https://github.com/RobStride/Product_Information): A CAN-based motor controller for high-torque applications.
- **Damiao**: Another CAN bus motor controller, expanding the range of compatible hardware.

These additions mean LeRobot can now drive a wider variety of professional-grade actuators beyond the existing Dynamixel and Feetech ecosystem.

## Policies: A Growing Model Zoo

This release brings six new policies and techniques into LeRobot, pushing the boundaries of what's possible with open-source robot learning.

### Pi0-FAST: Autoregressive VLAs

**Pi0-FAST** brings autoregressive Vision-Language-Action models to [LeRobot with FAST (Frequency-space Action Sequence Tokenization)](https://huggingface.co/docs/lerobot/pi0fast). Unlike the flow-matching approach of Pi0, Pi0-FAST uses an autoregressive action expert (based on Gemma 300M) that generates discretized action tokens, enabling:

- **FAST tokenization**: Actions are tokenized for autoregressive decoding, with a dedicated [FAST action tokenizer](https://huggingface.co/lerobot/fast-action-tokenizer).
- **Flexible decoding**: Configurable temperature and max decoding steps for balancing speed and quality.
- **RTC-compatible**: Works with Real-Time Chunking (see [next section](#real-time-chunking-rtc)) for responsive inference.

```bash
lerobot-train \
  --policy.type=pi0_fast \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --policy.device=cuda
```

### Real-Time Chunking (RTC)

**Real-Time Chunking** is an inference-time technique from [Physical Intelligence](https://www.pi.website) that makes flow-matching policies dramatically more responsive. Instead of waiting for a full action chunk to finish before replanning, RTC continuously blends new predictions with in-progress actions, producing smoother and more reactive behavior.

RTC is not a standalone policy — it's an enhancement that plugs into existing flow-matching policies (Pi0 family, SmolVLA & Diffusion). Configure it via `--policy.rtc_config.enabled=true`.

This is a game-changer for real-world deployment where latency matters. Read the [original paper](https://huggingface.co/papers/2506.07339) for the technical details and our [documentation](https://huggingface.co/docs/lerobot/rtc).

### Wall-X

**Wall-X** is a new VLA policy built on [**Qwen2.5-VL**](https://huggingface.co/collections/Qwen/qwen25-vl) with flow-matching action prediction. It combines the strong vision-language understanding of Qwen2.5-VL with a flow-matching head for cross-embodiment robotic control.

```bash
pip install lerobot[wall_x]
lerobot-train \
  --policy.type=wall_x \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human
```

### X-VLA

**X-VLA** brings a **Florence2-based** VLA to LeRobot. Built on Microsoft's Florence-2 vision-language model, X-VLA offers an alternative backbone for VLA policies, expanding the diversity of foundation models available for robot learning. Check out the [training guide](https://huggingface.co/docs/lerobot/xvla) for setup instructions and the [base model](https://huggingface.co/lerobot/xvla-base).

```bash
pip install lerobot[xvla]
lerobot-train \
  --policy.type=xvla \
  --dataset.repo_id=lerobot/bimanual-so100-handover-cube
```

### SARM

**SARM (Stage-Aware Reward Modeling)** tackles one of the hardest problems in robot learning: long-horizon tasks. Instead of using a single global linear progress signal over the whole episode, it models progress in a stage-aware manner by predicting both the task stage and the progress within that stage. This makes it much easier to train policies for complex, multi-step manipulation tasks. Start experimenting with it by following the [documentation](https://huggingface.co/docs/lerobot/sarm).

![sarm-community](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.5.0/sarm_community.gif)

### PEFT Support

You can now **fine-tune large VLAs using LoRA** (and other PEFT methods) without modifying the core training pipeline. PEFT configuration lives at the policy level, making it straightforward to adapt massive foundation models to your specific robot and task with a fraction of the compute. Learn more reading the [documentation](https://huggingface.co/docs/lerobot/peft_training).

```bash
lerobot-train \
  --policy.type=pi0 \
  --policy.peft_config.use_peft=true \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human
```

## Datasets: Faster Recording, Faster Training

The dataset pipeline gets major performance improvements in this release, making both data collection and training significantly faster.

### Streaming Video Encoding

Previously, recording a dataset meant waiting after each episode for video encoding to finish. **No more.** With streaming video encoding, frames are encoded in real-time as they're captured — meaning **zero wait time between episodes**. Just finish one episode and immediately start the next.

Streaming encoding also supports **hardware encoder auto-detection**, so if your system has a GPU-accelerated video encoder, LeRobot will use it automatically:

```python
dataset = LeRobotDataset.create(
    repo_id="my/dataset",
    fps=30,
    video_backend="auto",       # Auto-detect best HW encoder
    streaming_encoding=True,    # Encode in real-time
)
```

### 10x Faster Image Training, 3x Faster Encoding

Under the hood, we've fixed key data access bottlenecks and overhauled image processing:

- **10x faster image training**: Improved image transform support and fixed data access bottlenecks that were silently slowing down training.
- **3x faster encoding**: Parallel encoding is now the default across all platforms, with dynamic compression levels that adapt to your dataset type (video vs. image), when not using streaming.
- **Better CPU utilization**: More efficient resource usage during recording and dataset creation.

### New Dataset Tools

The dataset editing toolkit continues to grow:

- **Subtask support**: Annotate and query subtasks within episodes for hierarchical task learning.
- **Image-to-video conversion**: Convert existing image-based datasets to video format for better storage efficiency, with support for multiple episodes per video file.
- **More editing operations**: New `info` operation for inspecting datasets, task modification tools, and numerous fixes to existing operations (splitting, merging, feature editing).
- **Expose more options**: Configurable video codecs, tolerance settings, and metadata buffer sizes for fine-grained control over dataset creation.

## EnvHub: Environments from the Hub

**EnvHub** is a new way to use simulation environments in LeRobot: load them directly from the Hugging Face Hub. Instead of installing environment packages locally and wiring up registration, you can now point LeRobot at a Hub repository and it handles everything — downloading the environment code, registering it with Gymnasium, and making it available for training and evaluation.

Hub environments use `HubEnvConfig`, which downloads and executes remote `make_env` functions:

```bash
lerobot-train \
  --env.type=hub \
  --env.hub_path="username/my-custom-env" \
  --policy.type=act
```

This lowers the barrier for sharing custom simulation environments with the community. Package your environment, push it to the Hub, and anyone can train on it. Check out the [documentation](https://huggingface.co/docs/lerobot/envhub) to learn more. Here's an example to get started: [LeIsaac x LeRobot EnvHub tutorial](https://huggingface.co/docs/lerobot/envhub_leisaac).

### NVIDIA IsaacLab-Arena

We've integrated **NVIDIA IsaacLab-Arena**, bringing GPU-accelerated simulation to LeRobot. IsaacLab-Arena provides a collection of manipulation tasks running on NVIDIA's Isaac Sim, offering massively parallel environment instances for fast reinforcement learning. The integration includes dedicated pre/post-processing steps and full compatibility with LeRobot's training pipeline. Check out the [documentation](https://huggingface.co/docs/lerobot/envhub_isaaclab_arena).

## Codebase: A Modern Foundation

This release modernizes the codebase:

- **Python 3.12+**: LeRobot now requires Python 3.12 as the minimum version, enabling modern syntax and better performance.
- **Transformers v5**: We've migrated to Hugging Face Transformers v5, staying current with the latest model ecosystem.
- **3rd-party policy plugins**: Just like v0.4.0's hardware plugin system, you can now register custom policies as installable packages — `pip install lerobot_policy_mypolicy` and use it with `--policy.type=mypolicy`. No core library changes needed. Learn how to do it by following the [documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies).
- **Remote Rerun visualization**: Visualize your robot's telemetry remotely using Rerun, with compressed image support for bandwidth-efficient streaming.
- **Installation improvements**: Added `uv` [installation instructions](https://huggingface.co/docs/lerobot/installation), clarified setup steps, and improved dependency management. Sequential install steps are now clearly documented.
- **Documentation versioning**: Docs are now versioned, so you can always find documentation matching your installed release.
- **PyTorch version bump**: Updated PyTorch version bounds to support NVIDIA Blackwell GPUs.

## Community & Ecosystem

- **Modernized Discord**: Updated the most vibrant community hub with a better channel organization.
- **GitHub README, templates & automated labeling**: A refreshed README, new issue and PR templates, contributing guidelines, and automatic labeling of tickets — making it easier for everyone to contribute.
- **ICLR 2026 paper acceptance**: The LeRobot paper [has been accepted to ICLR 2026](https://openreview.net/forum?id=CiZMMAFQR3)!
- **LeRobot Visualizer refresh**: The visualization tool got a refresh with new dataset visualization badges and improved functionality. [Check it out !](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fimstevenpmwork%2Fthanos_picking_power_gem%2Fepisode_0)
- **LeRobot Annotation Studio**: A HuggingFace Space designed to easily annotate every moment of your dataset with natural language subtasks. [Check it out !](https://huggingface.co/spaces/lerobot/annotate)

![visualizer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.5.0/visualizer.gif)

## Final Thoughts

Beyond these headline features, v0.5.0 includes hundreds of bug fixes, documentation improvements, CI/CD enhancements, and quality-of-life improvements across the entire codebase. From better type checking to more robust test infrastructure, we're investing in the foundations that make LeRobot reliable and maintainable as it scales.

We want to extend a huge **thank you to everyone in the community** — contributors, users, and collaborators alike — for helping LeRobot grow into what it is today. Every bug report, PR, and discussion makes this project better.

Stay tuned for more to come 🤗 Get started [here](https://github.com/huggingface/lerobot)!
– The LeRobot team ❤️


> [!IMPORTANT]
> There's a big surprise coming just right around the corner, stay tuned! 👕