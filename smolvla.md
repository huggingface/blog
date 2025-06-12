---
title: "SmolVLA: Efficient Vision-Language-Action Model trained on Lerobot Community Data" 
thumbnail: /blog/assets/smolvla/SmolVLA_thumbnail.png
authors:
- user: danaaubakirova
- user: andito
- user: merve
- user: ariG23498
- user: fracapuano
- user: loubnabnl
- user: pcuenq
- user: mshukor
- user: cadene
---

# SmolVLA: Efficient Vision-Language-Action Model trained on Lerobot Community Data
## 🧭TL;DR
Today, we introduce [SmolVLA](https://huggingface.co/lerobot/smolvla_base), a compact (450M), open-source Vision-Language-Action model for robotics that runs on consumer hardware.
- Pretrained only on compatibly licensed, open-source community-shared datasets under the [lerobot](https://huggingface.co/datasets?other=lerobot&sort=trending) tag.
- SmolVLA-450M outperforms much larger VLAs and strong baselines such as [ACT](https://huggingface.co/papers/2401.02117) on simulation (LIBERO, Meta-World) and real-world tasks ([SO100, SO101](https://github.com/TheRobotStudio/SO-ARM100)).
- Supports *asynchronous inference* for **30% faster response** and **2× task throughput**.

**Useful links**:

- Hardware used to train and evaluate SO-100/101: https://github.com/TheRobotStudio/SO-ARM100  
- Base model https://huggingface.co/lerobot/smolvla_base
- Paper: https://huggingface.co/papers/2506.01844


## 📚 Table of Contents
- [🧭 TL;DR](#tl-dr)
- [📖 Introduction](#introduction)
- [🤖 Meet SmolVLA](#meet-smolvla)
- [🚀 How to Use SmolVLA?](#-how-to-use-smolvla)
  - [Install](#install)
  - [Finetune the Pretrained Model](#finetune-the-pretrained-model)
  - [Train from Scratch](#train-from-scratch)
- [🧠 Method](#method)
  - [Main Architecture](#main-architecture)
    - [Vision-Language Model (VLM)](#vision-language-model-vlm)
    - [Action Expert: Flow Matching Transformer](#action-expert-flow-matching-transformer)
  - [Design Choices for Efficiency and Robustness](#design-choices-for-efficiency-and-robustness)
    - [Visual Token Reduction](#visual-token-reduction)
    - [Faster Inference via Layer Skipping](#faster-inference-via-layer-skipping)
    - [Interleaved Cross and Self-Attention](#interleaved-cross-and-self-attention)
  - [Asynchronous Inference](#asynchronous-inference)
- [📦 Community Datasets](#community-datasets)
  - [Improving Task Annotations](#improving-task-annotations)
  - [Standardizing Camera Views](#standardizing-camera-views)
- [📊 Results](#results)
- [✅ Conclusion](#conclusion)
- [📣 Call to Action](#call-to-action)



## Introduction

Over the past few years, Transformers have driven remarkable progress in AI, from language models capable of human-like reasoning to multimodal systems that understand both images and text. However, in real-world robotics, advancements have been much slower. Robots still struggle to generalize across diverse objects, environments, and tasks. This limited progress stems from a **lack of high-quality, diverse data** and the absence of models that can **reason and act like humans in the physical world**.

In response to these challenges, the field has recently turned to **vision-language-action (VLA) models**, which aim to unify perception, language understanding, and action prediction within a single architecture. VLAs typically take as input raw visual observations and natural language instructions, and output corresponding robot actions. While promising, much of the recent progress in VLAs remains locked behind proprietary models trained on large-scale private datasets, often requiring costly hardware setups and extensive engineering resources. 
As a result, the broader robotics research community faces significant barriers in reproducing and building upon these models.

SmolVLA addresses this gap by offering an open-source, compact, and efficient VLA model that can be trained on **consumer-grade hardware using only publicly available datasets**. By releasing not only model weights but also using very affordable open-source hardware, SmolVLA aims to democratize access to vision-language-action models and accelerate research toward generalist robotic agents. 

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/S-3vvVCulChREwHDkquoc.gif" alt="Comparison of SmolVLA across task variations." width="500"/>
  <br/>
  <em>Figure 1: Comparison of SmolVLA across task variations. From left to right: (1) asynchronous pick-place cube counting, (2) synchronous pick-place cube counting, (3) pick-place cube counting under perturbations, and (4) generalization on pick-and-place of the lego block with real-world SO101.</em>
</p>

## Meet SmolVLA! 

**SmolVLA-450M** is our open-source, compact yet capable VLA model. It is:
- Small enough to run on CPU, train on a single consumer GPU, or even a MacBook! 
- Trained on public, community-shared robotics data
- Released with full training and inference recipes
- Can be tested and deployed on very affordable hardware (SO-100, SO-101, LeKiwi, etc.)

Inspired by the training paradigms of Large Language Models (LLMs), SmolVLA goes through a pretraining phase on general manipulation data, followed by task-specific post-training. Architecturally, it combines Transformers with **flow-matching decoders**, and is optimized for speed and low-latency inference with the following design choices:

* Skipping half of the layers of the vision model for faster inference and smaller size
* Interleaving self-attention and cross-attention blocks
* Using fewer visual tokens
* Leveraging smaller pretrained VLMs 

Despite using fewer than 30k training episodes—an order of magnitude less than other VLAs—SmolVLA **matches or exceeds the performance** of much larger models, both in simulation and the real world.

To make real-time robotics easier to use, we introduce an asynchronous inference stack. This technology separates how robots perform actions from how they understand what they see and hear. Because of this separation, robots can respond more quickly in fast-changing environments.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/aooU0a3DMtYmy_1IWMaIM.png" alt="SmolVLA architecture." width="500"/>
  <br/>
  <em>Figure 2. SmolVLA takes as input a sequence of RGB images from multiple cameras, the robot’s current sensorimotor state, and a natural language instruction. The VLM encodes these into contextual features, which condition the action expert to generate a continuous sequence of actions.</em>
</p>



## 🚀 How to Use SmolVLA?
SmolVLA is designed to be easy to use and integrate—whether you're finetuning on your own data or plugging it into an existing robotics stack.

###  Install

First, install the required dependencies:

```python
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

### Finetune the pretrained model
Use [`smolvla_base`](https://hf.co/lerobot/smolvla_base), our pretrained 450M model, with the lerobot training framework:

```python
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=20000  # 10% of training budget
```
### Train from scratch
​​If you'd like to build from the architecture (pretrained VLM + action expert) rather than a pretrained checkpoint:

```python
python lerobot/scripts/train.py \
  --policy.type=smolvla \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=200000
```
You can also load `SmolVLAPolicy` directly:

```python
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

## Method
SmolVLA is not only a lightweight yet capable model, but also a method for training and evaluating generalist robotics policies. In this section, we introduce the *model architecture* behind SmolVLA and the *asynchronous inference* setup used for evaluation, which has proven to be more adaptable and capable of faster recovery.

SmolVLA consists of two core components: a **Vision-Language Model (VLM)** that processes multimodal inputs and an **action expert** that outputs robot control commands. Below, we share the details of the main components of SmolVLA architecture and the Asynchronous Inference. More details can be found in our [technical report](https://huggingface.co/papers/2506.01844). 

### Main Architecture
#### Vision-Language Model (VLM)

We use [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) as our VLM backbone. It’s optimized for multi-image inputs and consists of a SigLIP vision encoder and a [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) language decoder.
- **Image tokens** are extracted via the vision encoder
- **Language instructions** are tokenized and fed directly into the decoder.
- **Sensorimotor states** are projected into a single token using a linear layer to align with the token dimension of the language model.

The decoder layers process concatenated image, language, and state tokens. The resulting features are then passed to the action expert.

#### Action Expert: Flow Matching Transformer

SmolVLA’s **action expert** is a compact transformer (~100M parameters) that generates action chunks, i.e. sequences of future robot actions, conditioned on the VLM’s outputs. It is trained using a **flow matching objective**, which teaches the model to guide noisy samples back to the ground truth. In contrast, while discrete action representations (e.g., via tokenization) are powerful, they often require autoregressive decoding, which is slow and inefficient at inference time. While flow matching allows **direct, non-autoregressive prediction of continuous actions**, enabling real-time control with high precision.

More intuitively, during training, we add random noise to the robot’s real action sequences and ask the model to predict the “correction vector” that brings them back to the correct trajectory. This forms a smooth vector field over the action space, helping the model learn accurate and stable control policies.  

We implement this using a transformer architecture with **interleaved attention blocks** (see the figure 2), and reduce its hidden size to **75% of the VLM’s**, keeping the model lightweight for deployment.

### Design Choices for Efficiency and Robustness

While combining a vision-language model with an action prediction module is a common design pattern in recent VLA systems—such as Pi0, GR00T, Diffusion Policy — we identified several architectural choices that significantly enhance the robustness and performance. In SmolVLA, we apply three key techniques: **reducing the number of visual tokens, skipping upper layers in the VLM**, and **interleaving cross- and self-attention layers** in the action expert.

#### Visual Token Reduction

High-resolution images improve perception but can significantly slow down inference. To strike a balance, **SmolVLA limits the number of visual tokens to 64 per frame** during both training and inference. For example, a 512×512 image is compressed into just 64 tokens, **instead of 1024**, using **PixelShuffle** as an efficient shuffling technique. While the underlying Vision-Language Model (VLM) was originally pretrained using image tiling for broader coverage, **SmolVLA uses only the global image at runtime** to keep inference lightweight and fast.

#### Faster Inference via Layer Skipping

Rather than always relying on the final layer of the VLM—which can be expensive and sometimes suboptimal—we use **features from intermediate layers**. Prior work has shown that early layers often provide better representations for downstream tasks.
In SmolVLA, the action expert only attends to VLM features up to a configurable layer NN during training, set to **half the total layers**. This **halves the compute cost** of both the VLM and the action expert, significantly speeding up inference with minimal performance loss.

#### Interleaved Cross and Self-Attention

Inside the action expert, attention layers alternate between:
- **Cross-attention (CA)**, where action tokens attend to the VLM’s features
- **Self-attention (SA)**, where action tokens attend to each other (causally—only to the past)

We found that this **interleaved design** is both lighter and more effective than using full attention blocks. Models that rely only on CA or only on SA tend to sacrifice either smoothness or grounding.

In SmolVLA, CA ensures that actions are well-conditioned on perception and instructions, while SA improves **temporal smoothness**—especially critical for real-world control, where jittery predictions can result in unsafe or unstable behavior.

## Asynchronous Inference

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/IV6vxVHCxUuYMEc7otXtv.png" alt="Asynchronous inference" width="500"/>
  <p>Figure 3. Asynchronous inference. Illustration of the asynchronous inference stack. Note that the policy can be run on a remote server, possibly with GPUs.</p>
</div>

Modern visuomotor policies output **action chunks**—sequences of actions to execute. There are two ways to manage them:
- **Synchronous (sync):** The robot executes a chunk, then pauses while the next one is computed. Simple, but causes a delay where the robot can't react to new inputs.
- **Asynchronous (async):** While executing the current chunk, the robot already sends the latest observation to a **Policy Server** (possibly hosted on GPU) for the next chunk. This avoids idle time and improves reactivity.

Our async stack decouples action execution from chunk prediction, resulting in higher adaptability, and the complete lack of execution lags at runtime. It relies on the following key mechanisms:

- **1. Early trigger:** When the queue length falls below a threshold (e.g., 70%), we send an observation to a **Policy Server**, calling for a new action chunk.
- **2. Decoupled threads:** Control loop keeps executing → inference happens in parallel (non-blocking).
- **3. Chunk fusion:** Overlapping actions from successive chunks are stitched with a simple merge rule to avoid jitter.

We are really excited about releasing asynchronous inference because it guarantees greater adaptability and improved performance without changing the model. In short, async inference keeps the robot responsive by overlapping execution and remote prediction.

## Community Datasets

While vision and language models thrive on web-scale datasets like LAION, ImageNet, and Common Crawl, robotics lacks a comparable resource. There’s no “Internet of robots.” Instead, data is fragmented across robot types, sensors, control schemes, and formats—forming disconnected "data islands". In our [previous post](https://huggingface.co/blog/lerobot-datasets), we explored how this fragmentation could be resolved through open, collaborative efforts. Just as ImageNet catalyzed breakthroughs in computer vision by providing a large, diverse benchmark, we believe that **community-driven robotics datasets** can play the same foundational role for generalist robot policies.

**SmolVLA is our first step toward that vision**: It is pretrained on a curated mix of publicly available, community-contributed datasets designed to reflect real-world variation. Rather than optimizing for dataset size alone, we focus on diversity: a range of behaviors, camera viewpoints, and embodiments that promote transfer and generalization.

All training data used in SmolVLA comes from **LeRobot Community Datasets** , robotics  datasets shared on the Hugging Face Hub under the `lerobot` tag. Collected in diverse settings, from labs to living rooms, these datasets represent an open, decentralized effort to scale real-world robot data.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/V4QU-B-6YBONb-8K_lSpj.gif" alt="A glimpse of the community dataset." width="500"/>
  <br/>
  <em>Figure 4. A glimpse of the community dataset. Special thanks to Ville Kuosmanen for creating the visualization.
Unlike academic benchmarks, community datasets naturally capture messy, realistic interactions: varied lighting, suboptimal demonstrations, unconventional objects, and heterogeneous control schemes. This kind of diversity will be very useful for learning robust, general-purpose representations.</em>
</p>


We used a custom[filtering tool](https://huggingface.co/spaces/Beegbrain/FilterLeRobotData)  created by [Alexandre Chapin](https://huggingface.co/Beegbrain) and [Ville Kuosmanen](https://huggingface.co/villekuosmanen) to select datasets based on frame count, visual quality, and task coverage. After a meticulous manual review (special thanks to Marina Barannikov), we curated a collection of **487 high-quality datasets** focused on the **SO100 robotic arm**, standardized at **30 FPS**. This yielded around **10 million frames**—at least **one order of magnitude smaller** than other popular benchmark datasets, yet significantly more diverse.

### Improving Task Annotations
 
A common issue across community datasets was noisy or missing task descriptions. Many episodes lacked annotations or included vague labels like “task desc” or “Move”, “Pick”. To improve quality and standardize the textual input across datasets, we used [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) to generate concise, action-oriented descriptions.

Given sample frames and the original label, the model was prompted to rewrite the instruction in under 30 characters, starting with an action verb (e.g., “Pick,” “Place,” “Open”). 

The prompt used is: 

```
Here is a current task description: {current_task}. Generate a very short, clear, and complete one-sentence describing the action performed by the robot arm (max 30 characters). Do not include unnecessary words.
Be concise.
Here are some examples: Pick up the cube and place it in the box, open the drawer and so on.
Start directly with an action verb like “Pick”, “Place”, “Open”, etc.
Similar to the provided examples, what is the main action done by the robot arm?
```

### Standardizing Camera Views
   
Another challenge was inconsistent camera naming. Some datasets used clear names like top or `wrist.right`, while others used ambiguous labels like `images.laptop`, which varied in meaning.
To fix this, we manually went through the datasets and mapped each camera view to a standardized scheme:
`OBS_IMAGE_1`: Top-down view
`OBS_IMAGE_2`: Wrist-mounted view
`OBS_IMAGE_3+`: Additional viewpoints

We further isolate the contributions of community dataset pretraining and multitask finetuning. Without pretraining on the LeRobot community datasets, SmolVLA initially achieves **51.7%** success on SO100. After pretraining on community-collected data, performance jumps to **78.3%**, a **+26.6% absolute improvement**. Multitask finetuning further boosts performance, showing strong task transfer capabilities even in low-data regimes.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/GdKdSzT2oAt83MQ0lPjcY.png" width="500"/>
  <p> Table 1. Impact of Pretraining on Community Datasets and Multitask Finetuning.</p>
</div>

## Results
We evaluate SmolVLA across simulation and real-world benchmarks to test its generalization, efficiency, and robustness. Despite being compact, It consistently outperforms or matches the performance of significantly larger models and policies pretrained on higher-scale robotics data. 

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/_v01LDKSy8zgcYr_7yQMx.png" alt="SmolVLA Performance on Simulation Benchmarks." width="500"/>
  <p> Table 2. SmolVLA Performance on Simulation Benchmarks.</p>
</div>


<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/ahQpohnpqRw6sQFMzjmg4.png" alt="SmolVLA vs Baselines on Real-World Tasks (SO100)." width="500"/>
  <p> Table 3. SmolVLA vs Baselines on Real-World Tasks (SO100).</p>
</div>

In real-world settings, SmolVLA is evaluated on two diverse suites: SO100 and SO101. These tasks include pick-place, stacking, and sorting, with both in-distribution and out-of-distribution object configurations. 
On SO101, SmolVLA also excels in generalization:

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/MZuG6UzXZ1SJ1MOfUfyzb.png" alt="Generalization of SmolVLA to New Embodiment (SO101) vs ACT.." width="500"/>
  <p>Table 4. Generalization of SmolVLA to New Embodiment (SO101) vs ACT..</p>
</div>

Finally, we evaluate SmolVLA under synchronous and asynchronous inference modes. Async inference decouples action execution from model inference, allowing the policy to react while the robot is moving.
- Both modes achieve similar task success (≈78%), but async inference:
  * Completes tasks **~30% faster** (9.7s vs. 13.75s)
  * Enables **2× more completions** in fixed-time settings (19 vs. 9 cubes)

This results in more responsive and robust real-world performance, especially in dynamic environments with shifting objects or external disturbances.
<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/Goxb9y5cE_Ty1SWCetCoT.png" alt="Asynchronous vs. Synchronous Inference in Real-World Tasks." width="500"/>
  <p>Figure 5. Asynchronous vs. Synchronous Inference in Real-World Tasks.
(a) Task success rates (%), (b) average completion time(s), and (c) number of tasks completed within a fixed time window.
</p>
</div>


## Conclusion

SmolVLA is our contribution to building robotics foundation models that are open, efficient, and reproducible. Despite its small size, it matches or outperforms larger, proprietary models across a range of real-world and simulated tasks. By relying solely on community-contributed datasets and affordable hardware, SmolVLA lowers the barrier to entry for researchers, educators, and hobbyists alike.
But this is just the beginning. SmolVLA is more than just a model — it's part of a growing open-source movement toward scalable, collaborative robotics.

## Call to Action:

- 🔧 **Try it out!** Finetune SmolVLA on your own data, deploy it on affordable hardware, or benchmark it against your current stack and share it on twitter/linkedin.
- 🤖 **Upload the dataset!** Got a robot? Collect and share your data using the lerobot format. Help expand the community dataset that powers SmolVLA.
- 💬 **Join the blog discussion.** Drop your questions, ideas, or feedback in the discussion below. We’re happy to help with integration, training, or deployment.
- 📊 **Contribute.** Improve datasets, report issues, suggest new ideas. Every contribution helps.
- 🌍 **Spread the word.** Share SmolVLA with fellow researchers, developers, or educators interested in efficient, real-time robotic policies.
- 📫 **Stay in touch:** Follow the [LeRobot organization](https://huggingface.co/lerobot) and [Discord server](https://discord.com/invite/ttk5CV6tUw) for updates, tutorials, and new releases.

Together, we can make real-world robotics more capable, more affordable, and more open. ✨

