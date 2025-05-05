---
title: "LeRobot Community Datasets: The “ImageNet” of Robotics — When and How?" 
thumbnail: /blog/assets/
authors:
- user: danaaubakirova
- user: your_coauthor
---

# Introduction

Recent advances in Vision-Language-Action (VLA) models have enabled robots to perform a wide range of tasks—from simple commands like “grasp the cube” to more complex activities like folding laundry or cleaning a table. These models aim to achieve generalization: the ability to perform tasks in novel settings, with unseen objects, and in varying conditions.

As Physical Intelligence highlights, the biggest challenge in robotics isn’t dexterity, but generalization—across physical, visual, and semantic levels. A robot must "figure out how to correctly perform even a simple task in a new setting or with new objects," and this requires both robust skills and common-sense understanding of the world. Yet, progress is limited by "the availability of diverse data for such robotic systems."

“Generalization must occur at many levels. At the low level, the robot must understand how to pick up a spoon (by the handle) or plate (by the edge), even if it has not seen these specific 
spoons or plates before, and even if they are placed in a pile of dirty dishes. At a higher level, the robot must understand the semantics of each task—where to put clothes and shoes (ideally in the laundry hamper or closet, not on the bed), and what kind of tool is appropriate for wiping down a spill. This generalization requires both robust physical skills and a common-sense understanding of the environment, so that the robot can generalize at many levels at the same time, from physical, to visual, to semantic. This is made even harder by the limited availability of diverse data for such robotic systems.”

To simplify, the core of generalist policies lies in a simple idea: co-training on heterogeneous datasets. By exposing VLA models to a variety of environments, tasks, and robot embodiments, we can teach not only how to act, but why—how to interpret a scene, understand a goal, and adapt skills across contexts.

This brings us to a fundamental question:
Given current datasets, what is the upper limit of generalization we can expect?

Can a robot meaningfully respond to a completely novel prompt—say, "set up a surprise birthday party"—if it has never encountered anything remotely similar during training? Especially when most datasets are collected in academic labs, by a limited number of people, under well-controlled setups?

In this blogpost, we frame generalization within a data-centric view: treating it as the process of abstracting broader patterns from data—essentially “zooming out” to reveal task-agnostic structures and principles. This shift in perspective emphasizes the role of dataset diversity, rather than model architecture alone, in driving generalization.
So far, the majority of robotics datasets come from structured academic environments. Even if we scale up to millions of demonstrations, one dataset will often dominate, limiting diversity. Unlike ImageNet—which aggregated internet-scale data and captured the real world more holistically—robotics lacks a comparably diverse, community-driven benchmark. This is because collecting data for robotics requires physical hardware and significant effort.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/9E7qfkq1sxMJcxecSrJDN.webp" alt="Growth of lerobot datasets" width="500"/>
  <p><b>Figure 1:</b> Growth of <i>lerobot</i> datasets on the Hugging Face Hub over time.</p>
</div>


That’s why, at LeRobot, among other things, we’re working to make robotics data collection more accessible to everyone—at home, at school, or anywhere. We’re simplifying the recording pipeline, software, streamlining uploading to the Hugging Face Hub, and even reducing hardware costs.
We're already seeing the results: the number of community-contributed datasets on the Hub is growing rapidly. Figure 1 illustrates the growth of the lerobot datasets uploaded to the Hugging Face Hub over the past months, which shows the increasing community effort to contribute real-world robotic data.
Figure 2 breaks down these datasets by robot type, showing the distribution across different embodiments and shedding light on which robot types are currently most represented (so100 and koch). This momentum brings us closer to a future where datasets are not limited by the perspective of a single group of people but reflect contributions from across the world.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/rhfdwybDuIu2ULGs7Nb7E.webp" alt="Distribution of lerobot datasets" width="500"/>
  <p><b>Figure 2:</b> Distribution of <i>lerobot</i> datasets by robot type.</p>
</div>

As robotics data collection becomes more democratized, more and more data gets pushed to the hub. But as the quantity of data grows, it becomes increasingly complex to curate this massive influx in a way that makes it truly usable for training models. While these datasets are still collected in constrained setups, they represent a critical step toward affordable, general-purpose robotic policies—because not everyone has access to the expensive setups. But with shared infrastructure and open collaboration, we can build something even more powerful—together. 

The goal of this blogpost is to recognize the growing impact of community-contributed LeRobot datasets, identify current challenges, outline practical steps to maximize the value of this collective effort. 

---

# Better data = Better models

Why does data quality matter? Poor-quality data results in poor downstream performance, biased outputs, and models that fail to generalize. Hence, efficient and high-quality data collection plays a critical role in advancing generalist robotic policies. While foundation models in vision and language have thrived on massive, web-scale datasets, robotics lacks an "Internet of robots"—a vast, diverse corpus of real-world interactions. Instead, robotic data is fragmented across different embodiments, sensor setups, and control modes, forming isolated "data islands.".

To overcome this, recent approaches like Gr00t organize training data as a pyramid, where large-scale web and video data form the foundation, synthetic data provides simulated diversity, and real-world robot interactions at the top ground the model in physical execution. Within this framework, efficient real-world data collection is indispensable—it anchors learned behaviors in actual robotic hardware and closes the sim-to-real gap, ultimately improving the generalization, adaptability, and performance of robotics foundation models. 

By expanding the volume and diversity of real-world datasets, we reduce the fragmentation between heterogeneous data sources. When datasets are disjoint in terms of environment, embodiment, or task distribution, models struggle to transfer knowledge across domains. Real-world data acts as a connective tissue that aligns abstract, large-scale priors with grounded, embodied action, enabling the model to build more coherent and transferable representations. As a result, increasing the proportion of real robot interactions does not merely enhance realism—it structurally reinforces the links between all layers of the pyramid, leading to more robust and capable policies.

Figure 3 illustrates the data pyramid structure for robot foundation model training. As we move up the pyramid, the amount of data decreases while embodiment specificity increases—from broad web-scale priors to grounded, robot-specific interactions. 
(Yang et al., 2025)

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/eBmRnO1MsJ5SLxo1pMStf.png" alt="Data Pyramid for Robot Foundation Model Training" width="500"/>
  <p><b>Figure 3:</b> Data Pyramid for Robot Foundation Model Training. Adapted from <a href="https://arxiv.org/pdf/2503.14734">Gr00t</a>. Data quantity decreases while embodiment specificity increases from bottom to top.</p>
</div>
---
# Challenges with Current Community Datasets

At LeRobot, we’ve started developing an automatic curation pipeline to support and enhance community datasets. During the post-processing phase, we’ve identified several areas where improvements can further boost dataset quality and facilitate more effective curation going forward:

### 1. Incomplete or Inconsistent Task Annotations

Many datasets lack task descriptions, lack details or are ambiguous in the task to be done. Semantics is currently at the core of cognition, meaning that understanding the context and specifics of a task is crucial for robotic performance. Detailed expressions ensure that robots understand exactly what is expected, but also provide a broader knowledge and vocabulary to the cognition system. Ambiguity can lead to incorrect interpretation and, consequently, incorrect actions.

Task instructions can be:
- Empty
- Too short (e.g. “Hold”, “Up”)
- Without any specific meaning (e.g. “task desc”, “desc”)

Subtask-level annotations are often missing, making it difficult to model complex task hierarchies.  
While this can be handled with nice VLM, it is still better to have a task annotation provided by the author of the dataset at hand.


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

# What Makes a Good Dataset?

Now that we know that creating a high-quality dataset is essential for training reliable and generalizable robot policies. To assist you in collecting effective data, we’ve outlined a checklist of best practices across four key areas:

## Image Quality

- ✅ Use **at least two camera views**
- ✅ Ensure **steady video capture** (no shaking)
- ✅ Maintain **neutral, stable lighting** (avoid overly yellow or blue tones)
- ✅ Ensure **consistent exposure** and **sharp focus**
- ✅ **Leader arm should not appear** in the frame
- ✅ The **only moving objects** should be the follower arm and the manipulated items (avoid human limbs/bodies)
- ✅ Use a **static, non-distracting background**, or apply controlled variations
- ✅ Record in **high resolution** (at least 720p)

## Metadata & Recording Protocol

- ✅ Select the **correct robot type** in the metadata
- ✅ Record videos at approximately **30 frames per second (FPS)**
- ✅ If **deleting episodes**, make sure to **update the metadata files accordingly**

## Feature Naming Conventions

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
  <img src="https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/ioUy5DxjbFjiSKiCLe18_.png" alt="Dataset Recording Checklist" width="600"/>
  <p><b>Figure 4:</b> Dataset Recording Checklist – a step-by-step guide to ensure consistent and high-quality real-world data collection.</p>
</div>

