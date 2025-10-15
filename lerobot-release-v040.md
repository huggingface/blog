---
title: "LeRobot v0.4.0: XXX" 
thumbnail: /blog/assets/lerobot-release-v040/thumbnail.png
authors:
- user: imstevenpmwork
---

# LeRobot's Latest Leap: Supercharging Robot Learning with Major Updates!
We're thrilled to announce a series of significant advancements across LeRobot, designed to make open-source robot learning more powerful, scalable, and user-friendly than ever before! From revamped datasets to versatile editing tools, new simulation environments, and a groundbreaking plugin system for hardware, LeRobot is continuously evolving to meet the demands of cutting-edge embodied AI.

(TODO: IMAGE THUMBNAIL)

## Datasets: Ready for the Next Wave of Large-Scale Robot Learning
We've completely overhauled our dataset infrastructure with **LeRobotDataset v3.0**, featuring a new chunked episode format and streaming capabilities. This is a game-changer for handling massive datasets like OXE, bringing unparalleled efficiency and scalability.

(TODO: IMAGE)

### What's New in v3.0?
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

(TODO: IMAGE)

### LIBERO Support:
LeRobot now officially supports **LIBERO**, one of the largest open benchmarks for Vision-Language-Action (VLA) policies, boasting over 130 tasks! This is a huge step toward building the go-to evaluation hub for VLAs, enabling easy integration and a unified setup for evaluating any VLA policy.

Check out the [LIBERO dataset](https://huggingface.co/datasets/HuggingFaceVLA/libero) and our [docs](https://huggingface.co/docs/lerobot/en/libero) to get started!

### Meta-World Integration:
We've integrated Meta-World, a premier benchmark for testing multi-task and generalization abilities in robotic manipulation, featuring over 50 diverse tasks. This integration, along with our standardized use of `gymnasium ≥ 1.0.0` and `mujoco ≥ 3.0.0`, ensures deterministic seeding and a robust simulation foundation.

Train your policies with the [Meta-World dataset](https://huggingface.co/datasets/lerobot/metaworld_mt50) today!

## Robots: A New Era of Hardware Integration with the Plugin System
Big news for hardware enthusiasts! We've launched a brand-new plugin system to revolutionize how you integrate third-party hardware with LeRobot. Now, connecting any robot, camera, or teleoperator is as simple as a `pip install`, eliminating the need to modify the core library.

(TODO: IMAGE)

### Key Benefits:
* Extensibility: Develop and integrate custom hardware in separate Python packages.
* Scalability: Supports a growing ecosystem of devices without bloating the core library.
* Community-Friendly: Lowers the barrier to entry for community contributions, fostering a more collaborative environment.

Learn how to create your own plugin in our [documentation](https://huggingface.co/docs/lerobot/integrate_hardware#using-your-own-lerobot-devices-).

### Reachy 2 Integration:
Thanks to our new plugin system, we've also added **Reachy 2** from Pollen Robotics to LeRobot! Reachy 2 is available for both real robot control and simulation, enabling you to experiment with teleoperation and autonomous demos right away.

## Codebase and Robots: Enhanced Control and Mobile Teleoperation

(TODO: I don't like this, need to also talk about the importance of pipeline in policies implementation)

We're making robot control more flexible and accessible, enabling new possibilities for data collection and model training.

(TODO: IMAGE)

### RobotProcessor: The New Pipeline for Data Processing
We've introduced `RobotProcessor`, a powerful pipeline system for transforming data within LeRobot. This enables native support for end-effector control, making it easier to record and train AI models in end-effector space or with any other desired features.

Key features of `RobotProcessor`:
* Modular Pipeline Architecture: Decouples data transformations from model logic, offering flexible robot control and cross-platform compatibility.
* End-Effector Control: Record and train using end-effector poses, opening up new avenues for robot manipulation.
* Phone Teleoperator: Now you can teleoperate your follower arm directly from your phone (iOS/Android), making the LeRobot SO101 setup 50% cheaper!

This modular architecture, complete with over 30 registered processors, ensures type safety, performance, and flexibility across various robot platforms. Learn more about it in our [Introduction to Processors documentation](https://huggingface.co/docs/lerobot/introduction_processors).

## Policies: Unleashing Open-World Generalization with pi0 and pi0.5
In a major milestone for open-source robotics, we've integrated **pi0** and **pi0.5** policies by Physical Intelligence into LeRobot, fully ported to PyTorch! These Vision-Language-Action (VLA) models represent a significant leap towards addressing open-world generalization in robotics.

(TODO: IMAGE)

### What makes π₀.₅ revolutionary?
* Open-World Generalization: Designed to adapt to entirely new environments and situations, generalizing across physical, semantic, and environmental levels.
* Co-training on Heterogeneous Data: Learns from a diverse mix of multimodal web data, verbal instructions, subtask commands, and multi-environment robot data.
* Physical Intelligence Collaboration: Huge thanks to the Physical Intelligence team for their groundbreaking work!

You can find the ported models on the Hugging Face Hub: [pi0.5_base](https://huggingface.co/lerobot/pi05_base), [pi0_base](https://huggingface.co/lerobot/pi0_base), and their Libero-tuned counterparts.

This is a huge step forward in making robot learning as open and reproducible as NLP & CV. Try it out today, share your runs, and let's push forward the frontier of embodied AI together!


## TODO: Gr00t
XXX
