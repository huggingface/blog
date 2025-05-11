---
title: "LeRobot Community Datasets: The “ImageNet” of Robotics — When and How?" 
thumbnail: /blog/assets/195_lerobot_datasets/1.png
authors:
- user: danaaubakirova
- user: Beegbrain
  guest: true
- user: mshukor
- user: m1b
  guest: true
- user: villekuosmanen
  guest: true
- user: cadene
- user: pcuenq
---

🧭 **TL;DR — Why This Blogpost?**  

In this post, we:
  - Recognize the growing impact of community-contributed **LeRobot** datasets  
  - Highlight the current challenges in robotic data collection and curation
  - Share practical steps and best practices to maximize the impact of this collective effort     
Our goal is to frame generalization as a *data problem*, and to show how building an open, diverse “ImageNet of robotics” is not just possible—but already happening.

## Introduction

Recent advances in Vision-Language-Action (VLA) models have enabled robots to perform a wide range of tasks—from simple commands like “grasp the cube” to more complex activities like folding laundry or cleaning a table. These models aim to achieve **generalization**: the ability to perform tasks in novel settings, with unseen objects, and in varying conditions.

> **“The biggest challenge in robotics isn’t dexterity, but generalization—across physical, visual, and semantic levels.”**  
> — *Physical Intelligence*

A robot must *"figure out how to correctly perform even a simple task in a new setting or with new objects,"* and this requires both robust skills and common-sense understanding of the world. Yet, progress is often limited by **the availability of diverse data** for such robotic systems.

> “Generalization must occur at many levels. At the low level, the robot must understand how to pick up a spoon (by the handle) or plate (by the edge), even if it has not seen these specific spoons or plates before, and even if they are placed in a pile of dirty dishes. At a higher level, the robot must understand the semantics of each task—where to put clothes and shoes (ideally in the laundry hamper or closet, not on the bed), and what kind of tool is appropriate for wiping down a spill. This generalization requires both robust physical skills and a common-sense understanding of the environment, so that the robot can generalize at many levels at the same time, from physical, to visual, to semantic. This is made even harder by the limited availability of diverse data for such robotic systems.”  
> — *Physical Intelligence*

## From Models to Data: Shifting the Perspective

To simplify, the core of generalist policies lies in a simple idea: **co-training on heterogeneous datasets**. By exposing VLA models to a variety of environments, tasks, and robot embodiments, we can teach models not only how to act, but *why*—how to interpret a scene, understand a goal, and adapt skills across contexts.

> 💡 **“Generalization is not just a model property—it’s a data phenomenon.”**  
> It emerges from the diversity, quality, and abstraction level of the training data.

This brings us to a fundamental question:

**Given current datasets, what is the upper limit of generalization we can expect?**

Can a robot meaningfully respond to a completely novel prompt—say, *"set up a surprise birthday party"*—if it has never encountered anything remotely similar during training? Especially when most datasets are collected in academic labs, by a limited number of people, under well-controlled setups?

We frame generalization as a **data-centric view**: treating it as the process of abstracting broader patterns from data—essentially *“zooming out”* to reveal task-agnostic structures and principles. This shift in perspective emphasizes the role of **dataset diversity**, rather than model architecture alone, in driving generalization.

## Why does Robotics lack its ImageNet Moment?

So far, the majority of robotics datasets come from structured academic environments. Even if we scale up to millions of demonstrations, one dataset will often dominate, limiting diversity. Unlike ImageNet—which aggregated internet-scale data and captured the real world more holistically—robotics lacks a comparably diverse, community-driven benchmark.

This is largely because collecting data for robotics requires **physical hardware and significant effort**.

## Building a LeRobot Community

That’s why, at **LeRobot**, we’re working to make robotics data collection more accessible—at home, at school, or anywhere. We're:

- Simplifying the recording pipeline  
- Streamlining uploading to the Hugging Face Hub, to foster community sharing
- Reducing hardware costs  

We're already seeing the results: the number of community-contributed datasets on the Hub is growing rapidly.
<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/9E7qfkq1sxMJcxecSrJDN.webp" alt="Growth of <i>lerobot</i> datasets on the Hugging Face Hub over time" width="500"/>
  <p> Growth of <i>lerobot</i> datasets on the Hugging Face Hub over time.</p>
</div>

If we break down the uploaded datasets by robot type, we see that most contributions are to So100 and Koch, making robotic arms and manipulation tasks the primary focus of the current LeRobot dataset landscape. However, it’s important to remember that the potential reaches far beyond. Domains like autonomous vehicles, assistive robots, and mobile navigation stand to benefit just as much from shared data. This momentum brings us closer to a future where datasets reflect a global effort, not just the contributions of a single lab or institution.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/rhfdwybDuIu2ULGs7Nb7E.webp" alt="Distribution of lerobot datasets by robot type" width="500"/>
  <p>Distribution of <i>lerobot</i> datasets by robot type.</p>
</div>

Here are just a few standout community-contributed datasets that show how diverse and imaginative robotics can be:

- [`lirislab/close_top_drawer_teabox`](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Flirislab%2Fclose_top_drawer_teabox%2Fepisode_1):: precise manipulation with a household drawer
- [`Chojins/chess_game_001_blue_stereo`](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FChojins%2Fchess_game_001_blue_stereo%2Fepisode_1): a full chess match captured from a stereo camera setup
- [`pierfabre/chicken`](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fpierfabre%2Fchicken%2Fepisode_6): yes — a robot interacting with colorful animal figures, including a chicken 🐔

Explore additional creative datasets under the [`LeRobot` tag](https://huggingface.co/datasets?other=LeRobot) on the Hugging Face Hub, and interactively view them in the [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).

## Scaling Responsibly

As robotics data collection becomes more democratized, **curation becomes the next challenge**. While these datasets are still collected in constrained setups, they are a crucial step toward affordable, general-purpose robotic policies. Not everyone has access to expensive hardware—but with **shared infrastructure and open collaboration**, we can build something far greater.

> 🧠 **“Generalization isn’t solved in a lab—it’s taught by the world.”**  
> The more diverse our data, the more capable our models will be.

---
## Better data = Better models

Why does data quality matter? Poor-quality data results in poor downstream performance, biased outputs, and models that fail to generalize. Hence, **efficient and high-quality data collection** plays a critical role in advancing generalist robotic policies.

While foundation models in vision and language have thrived on massive, web-scale datasets, robotics lacks an “Internet of robots”—a vast, diverse corpus of real-world interactions. Instead, robotic data is fragmented across different embodiments, sensor setups, and control modes, forming isolated *data islands*.

To overcome this, recent approaches like [Gr00t](https://huggingface.co/papers/2503.14734) organize training data as a **pyramid**, where:

- Large-scale web and video data form the **foundation**  
- Synthetic data adds **simulated diversity**  
- Real-world robot interactions at the **top** ground the model in physical execution

Within this framework, efficient real-world data collection is indispensable—it anchors learned behaviors in actual robotic hardware and **closes the sim-to-real gap**, ultimately improving the generalization, adaptability, and performance of robotics foundation models.

By expanding the **volume and diversity of real-world datasets**, we reduce fragmentation between heterogeneous data sources. When datasets are disjoint in terms of environment, embodiment, or task distribution, models struggle to transfer knowledge across domains.

> 🔗 **Real-world data acts as connective tissue**—it aligns abstract priors with grounded action and enables the model to build more coherent and transferable representations.

As a result, increasing the proportion of real robot interactions does not merely enhance realism—it **structurally reinforces** the links between all layers of the pyramid, leading to more robust and capable policies.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/eBmRnO1MsJ5SLxo1pMStf.png" alt="Data Pyramid for Robot Foundation Model Training" width="500"/>
  <p>Data Pyramid for Robot Foundation Model Training. Adapted from <a href="https://huggingface.co/papers/2503.14734">Gr00t</a> (Yang et al., 2025). Data quantity decreases while embodiment specificity increases from bottom to top.</p>
</div>

---
## Challenges with Current Community Datasets

At LeRobot, we’ve started developing an automatic curation pipeline to post-process community datasets. During the post-processing phase, we’ve identified several areas where improvements can further boost dataset quality and facilitate more effective curation going forward:

### 1. Incomplete or Inconsistent Task Annotations

Many datasets lack task descriptions, lack details or are ambiguous in the task to be done. Semantics is currently at the core of cognition, meaning that understanding the context and specifics of a task is crucial for robotic performance. Detailed expressions ensure that robots understand exactly what is expected, but also provide a broader knowledge and vocabulary to the cognition system. Ambiguity can lead to incorrect interpretation and, consequently, incorrect actions.

Task instructions can be:
- Empty
- Too short (e.g. “Hold”, “Up”)
- Without any specific meaning (e.g. “task desc”, “desc”)

Subtask-level annotations are often missing, making it difficult to model complex task hierarchies.  
While this can be handled with VLM, it is still better to have a task annotation provided by the author of the dataset at hand.


### 2. Feature Mapping Inconsistencies

Features like `images.laptop` are ambiguously labeled:
- Sometimes it's a third-person view
- Other times it's more like a gripper (wrist) camera

Manual mapping of dataset features to standardized names is time-consuming and error-prone.  
We can possibly automate feature type inference using VLMs or computer vision models to classify camera perspectives. However, keeping this in mind helps to have a cleaner dataset.

### 3. Low-Quality or Incomplete Episodes

Some datasets contain:
- Episodes with only 1 or very few frames
- Manually deleted data files (e.g., deleted `.parquet` files without reindexing), breaking the sequential consistency.


### 4. Inconsistent Action/State Dimensions

Different datasets use different action or state dimensions, even for the same robot (e.g., `so100`).  
Some datasets show inconsistencies in action/state format.

---

## What Makes a Good Dataset?

Now that we know that creating a high-quality dataset is essential for training reliable and generalizable robot policies, we have outlined a checklist of best practices to assist you in collecting effective data.

### Image Quality

- ✅ Use preferably **two camera views**
- ✅ Ensure **steady video capture** (no shaking)
- ✅ Maintain **neutral, stable lighting** (avoid overly yellow or blue tones)
- ✅ Ensure **consistent exposure** and **sharp focus**
- ✅ **Leader arm should not appear** in the frame
- ✅ The **only moving objects** should be the follower arm and the manipulated items (avoid human limbs/bodies)
- ✅ Use a **static, non-distracting background**, or apply controlled variations
- ✅ Record in **high resolution** (at least 480x640 / 720p)

### Metadata & Recording Protocol

- ✅ Select the **correct robot type** in the metadata
  If you're using a custom robot that's not listed in the official [LeRobot config registry](https://github.com/huggingface/lerobot/blob/main/lerobot/common/robot_devices/robots/configs.py),  
  we recommend checking how similar robots are named in existing datasets on the [LeRobot Hub](https://huggingface.co/datasets?search=lerobot) to ensure consistency.   
- ✅ Record videos at approximately **30 frames per second (FPS)**
- ✅ If **deleting episodes**, make sure to **update the metadata files accordingly** (we will provide proper tools to edit datasets)

### Feature Naming Conventions

Use a consistent and interpretable naming scheme for all camera views and observations:

**Format:**
```bash
<modality>.<location>
```

**Examples:**

- `images.top`
- `images.front`
- `images.left`
- `images.right`

**Avoid device-specific names:**

- ❌ `images.laptop`
- ❌ `images.phone`

**For wrist-mounted cameras, specify orientation:**

- `images.wrist.left`
- `images.wrist.right`
- `images.wrist.top`
- `images.wrist.bottom`

> Consistent naming improves clarity and helps downstream models better interpret spatial configurations and multi-view inputs.

## Task Annotation

- ✅ Use the `task` field to **clearly describe the robot’s objective**
  - *Example:* `Pick the yellow lego block and put it in the box`
- ✅ Keep task descriptions **concise** (between **25–50 characters**)
- ✅ Avoid vague or generic names like `task1`, `demo2`, etc.

---

Below, we provide a checklist that serves as a guideline for recording datasets, outlining key points to keep in mind during the data collection process.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/6bZ8VEU-kgpDjLCokHj5J.png" alt="Dataset Recording Checklist" width="600"/>
  <p><b>Figure 4:</b> Dataset Recording Checklist – a step-by-step guide to ensure consistent and high-quality real-world data collection.</p>
</div>

## How Can You Help?

The next generation of generalist robots won't be built by a single person or lab — they'll be built by all of us. Whether you're a student, a researcher, or just robot-curious, here’s how you can jump in:

- 🎥 Record your own datasets — Use LeRobot tools to capture and upload good quality datasets from your robots.
- 🧠 Improve dataset quality — Follow our checklist, clean up your recordings, and help set new standards for robotics data.
- 📦 Contribute to the Hub — Upload datasets, share examples, and explore what others are building.
- 💬 Join the conversation — Give feedback, request features, or help shape the roadmap by engaging in our LeRobot [Discord Server](https://discord.gg/ttk5CV6tUw).
- 🌍 Grow the movement — Introduce LeRobot to your club, classroom, or lab. More contributors = better generalization.

> Start recording, start contributing—because the future of generalist robots depends on the data we build today.


