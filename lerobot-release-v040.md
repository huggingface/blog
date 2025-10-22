---
title: "LeRobot v0.4.0: Super Charging OSS Robotics Learning" 
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
---

# LeRobot's v0.4.0 Latest Major Updates!
We're thrilled to announce a series of significant advancements across LeRobot, designed to make open-source robot learning more powerful, scalable, and user-friendly than ever before! From revamped datasets to versatile editing tools, new simulation environments, and a groundbreaking plugin system for hardware, LeRobot is continuously evolving to meet the demands of cutting-edge embodied AI.

## Datasets: Ready for the Next Wave of Large-Scale Robot Learning
We've completely overhauled our dataset infrastructure with **LeRobotDataset v3.0**, featuring a new chunked episode format and streaming capabilities. This is a game-changer for handling massive datasets like OXE, bringing unparalleled efficiency and scalability.

### What's New in Datasets v3.0?
* Chunked Episodes for Massive Scale: Our new format supports datasets at the OXE-level, enabling unprecedented scalability.
* Efficient Video Storage + Streaming: Enjoy faster loading times and seamless streaming of video data.
* Unified Parquet Metadata: Say goodbye to scattered JSONs! All episode metadata is now stored in unified, structured Parquet files for easier management and access.
* Faster Loading & Better Performance: Experience significantly reduced dataset initialization times and more efficient memory usage.

We've also provided a conversion script to easily migrate your existing v2.1 datasets to the new v3.0 format, ensuring a smooth transition. Read more about it in our [blog post](https://huggingface.co/blog/lerobot-datasets-v3). Open-source robotics keeps leveling up!

### New Feature: Dataset Editing Tools!
Working with LeRobot datasets just got a whole lot easier! We've introduced a powerful set of utilities for flexible dataset editing.

With our new `lerobot-edit-dataset` CLI, you can now:
* Delete specific episodes from existing datasets.
* Split datasets by fractions or episode indices.
* Add or remove features with ease.
* Merge multiple datasets into one unified set.

These tools streamline your workflow, allowing you to curate and optimize your robot datasets like never before. Check out the [docs](https://huggingface.co/docs/lerobot/using_dataset_tools) for more details!

## Simulation Environments: Expanding Your Training Grounds
We're continuously expanding LeRobot's simulation capabilities to provide richer and more diverse training environments for your robotic policies.

![TODO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-release-v040/XXX.png)

### LIBERO Support
LeRobot now officially supports **LIBERO**, one of the largest open benchmarks for Vision-Language-Action (VLA) policies, boasting over 130 tasks! This is a huge step toward building the go-to evaluation hub for VLAs, enabling easy integration and a unified setup for evaluating any VLA policy.

Check out the [LIBERO dataset](https://huggingface.co/datasets/HuggingFaceVLA/libero) and our [docs](https://huggingface.co/docs/lerobot/en/libero) to get started!

### Meta-World Integration
We've integrated Meta-World, a premier benchmark for testing multi-task and generalization abilities in robotic manipulation, featuring over 50 diverse tasks. This integration, along with our standardized use of `gymnasium ≥ 1.0.0` and `mujoco ≥ 3.0.0`, ensures deterministic seeding and a robust simulation foundation.

Train your policies with the [Meta-World dataset](https://huggingface.co/datasets/lerobot/metaworld_mt50) today!

## Codebase: Powerful Tools For Everyone
We're making robot control more flexible and accessible, enabling new possibilities for data collection and model training.

### The New Pipeline for Data Processing

Getting data from a robot to a model (and back!) is tricky. Raw sensor data, joint positions, and language instructions don't match what AI models expect. Models need normalized, batched tensors on the right device, while your robot hardware needs specific action commands.

We're excited to introduce **Processors**: a new, modular pipeline that acts as a universal translator for your data. Think of it as an assembly line where each `ProcessorStep` handles one specific job—like normalizing, tokenizing text, or moving data to the GPU.

You can chain these steps together into a powerful pipeline to perfectly manage your data flow. We've even created two distinct types to make life easier:

* `PolicyProcessorPipeline`: Built for models. It expertly handles batched tensors for high-performance training and inference.
* `RobotProcessorPipeline`: Built for hardware. It processes individual data points (like a single observation or action) for real-time robot control.

This system makes it simple to connect any policy to any robot, ensuring your data is always in the perfect format for every step of the way. Learn more about it in our [Introduction to Processors documentation](https://huggingface.co/docs/lerobot/introduction_processors).

### Multi-GPU Training Made Easy

Training large robot policies just got a lot faster\! We've integrated [Accelerate](https://github.com/huggingface/accelerate) directly into our training pipeline, making it incredibly simple to scale your experiments across multiple GPUs with just **one command**.

Whether you're fine-tuning a policy or running large-scale experiments, LeRobot now handles all the complexities of distributed training for you. This means you can drastically reduce training time, cutting it in half with 2 GPUs, down to a third with 3 GPUs, and beyond.

Check out the [documentation](https://huggingface.co/docs/lerobot/multi_gpu_training) to accelerate your robot learning\!

## Policies: Unleashing Open-World Generalization

![TODO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-release-v040/XXX.png)

### PI0 and PI0.5
In a major milestone for open-source robotics, we've integrated **pi0** and **pi0.5** policies by Physical Intelligence into LeRobot! These Vision-Language-Action (VLA) models represent a significant leap towards addressing open-world generalization in robotics.

#### What makes π₀.₅ revolutionary?
* Open-World Generalization: Designed to adapt to entirely new environments and situations, generalizing across physical, semantic, and environmental levels.
* Co-training on Heterogeneous Data: Learns from a diverse mix of multimodal web data, verbal instructions, subtask commands, and multi-environment robot data.
* Physical Intelligence Collaboration: Huge thanks to the Physical Intelligence team for their groundbreaking work!

You can find the ported models on the Hugging Face Hub: [pi0.5_base](https://huggingface.co/lerobot/pi05_base), [pi0_base](https://huggingface.co/lerobot/pi0_base), and their Libero-tuned counterparts.

### GR00T N1.5
In another exciting development, we've integrated **NVIDIA's GR00T N1.5** into LeRobot, thanks to a fantastic collaboration with the NVIDIA robotics team! This open foundation model is a powerhouse for generalized robot reasoning and skills. As a cross-embodiment model, it takes multimodal input (like language and images) to perform complex manipulation tasks in diverse environments, marking another major leap in generalized robotics.

#### What makes GR00T N1.5 a game-changer?
* Generalized Reasoning & Skills: Designed as a cross-embodiment foundation model, GR00T N1.5 excels at generalized reasoning and manipulation tasks, with improved language-following ability.
* Expansive Heterogeneous Training: It learns from a massive dataset combining real captured humanoid data, synthetic data generated by NVIDIA Isaac GR00T Blueprint, and internet-scale video data.
* NVIDIA Collaboration: We're thrilled to partner with the NVIDIA team to bring this state-of-the-art model to the open-source LeRobot community!

You can find the model on the Hugging Face Hub: [GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B). For more details, check out the [NVIDIA research page](https://research.nvidia.com/labs/gear/gr00t-n1_5/) and the [official GitHub repository](https://github.com/NVIDIA/Isaac-GR00T).

The native integration of these policies in `lerobot` is a huge step forward in making robot learning as open and reproducible as it can be. Try them out today, share your runs, and let's push forward the frontier of embodied AI together!

## Robots: A New Era of Hardware Integration with the Plugin System
Big news for hardware enthusiasts! We've launched a brand-new plugin system to revolutionize how you integrate third-party hardware with LeRobot. Now, connecting any robot, camera, or teleoperator is as simple as a `pip install`, eliminating the need to modify the core library.

### Key Benefits
* Extensibility: Develop and integrate custom hardware in separate Python packages.
* Scalability: Supports a growing ecosystem of devices without bloating the core library.
* Community-Friendly: Lowers the barrier to entry for community contributions, fostering a more collaborative environment.

Learn how to create your own plugin in our [documentation](https://huggingface.co/docs/lerobot/integrate_hardware#using-your-own-lerobot-devices-).

### Reachy 2 Integration
Thanks to our new plugin system, we've also added **Reachy 2** from Pollen Robotics to LeRobot! Reachy 2 is available for both real robot control and simulation, enabling you to experiment with teleoperation and autonomous demos right away.

### Phone Integration

Thanks to our powerful new pipeline system, you can now teleoperate your follower arm **right from your phone** (iOS/Android). The phone acts as a teleoperator device, and our `RobotProcessor` pipeline handles all the transformations, allowing you to drive robots in different action spaces (like end-effector space) with ease.

## The Hugging Face Robot Learning Course

![TODO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-release-v040/XXX.png)

We're launching a comprehensive, self-paced, and entirely **open-source course** designed to make robot learning accessible to everyone! If you're curious about how real-world robots learn, this is the perfect place to start.

In this course, you’ll learn how to:

  * Understand the fundamentals of classical robotics.
  * Use generative models for imitation learning (VAEs, diffusion, etc.).
  * Apply Reinforcement Learning to real-world robots.
  * Explore the latest generalist robot policies like PI0 and SmolVLA.

Join the [Hugging Face Robotics organization](https://huggingface.co/robotics-course) to follow along and start your journey\!

### Deep Dive: The Modern Robot Learning Tutorial

For those who want to go deeper, we've also published a **hands-on tutorial** on the most recent advancements in robotics. This guide provides self-contained explanations, re-derives modern techniques from first principles, and includes ready-to-use code examples using LeRobot and Hugging Face.

The tutorial itself is hosted in a [Space](https://huggingface.co/spaces/lerobot/robot-learning-tutorial) and it features practical examples using LeRobot, with all models and datasets on the Hugging Hub. You can also check out the [ArXiv paper](http://arxiv.org/abs/2510.12403) for a detailed overview.


## Final thoughts from the team

Beyond these major features, this release is packed with numerous bug fixes, documentation improvements, updated dependencies, more examples and better infrastructure to make your experience with LeRobot smoother and more reliable.

We want to extend a huge **thank you to everyone in the community** for your invaluable contributions, feedback, and support. We're incredibly excited about the future of open-source robotics and can't wait to show you what's next.

Stay tuned for more to come\! <br>
– The LeRobot team ❤️
