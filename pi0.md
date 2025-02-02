# π0 and π0-FAST: A Vision-Language-Action Models for General Robot Control


We have ported the first **robot foundation models** to **Hugging Face LeRobot**! Both **π0 and π0-FAST** developed by Physical Intelligence, which are now available in the **LeRobot repository**, bringing generalist robotic intelligence to the Hugging Face ecosystem.

---

## Introduction


Heinlein suggests that a well-rounded person should be capable of handling a wide range of tasks—both intellectual and physical—rather than being narrowly specialized in one field. Drawing a parallel between a well-rounded person and machine intelligence: AI systems vary widely, but human intelligence excels in versatility—adapting to tasks, environments, and surprises. While large language and vision-language models (LLMs, VLMs) show promise, they lack interaction with the physical world. To bridge this gap, we need models trained on robotic data. Generalist robot models can enhance adaptability, using diverse data to improve generalization and robustness. Instead of training on isolated tasks, pre-training on varied robotic data—similar to LLMs—boosts efficiency and performance.

Developing generalist robot policies, or robot foundation models, presents three key challenges:

1. **The need for large-scale research** to fully leverage pre-training benefits.
2. **Designing model architectures** that can integrate diverse data sources while capturing complex physical interactions. A key challenge in this regard is **cross-embodiment training**, where a model must learn from diverse robot types with varying configurations, control spaces, and action representations. Existing approaches tackle this by:
   - **Combining multimodal datasets** from different robotic platforms to enhance generalization.
   - **Using shared representations** to bridge the gap between distinct robot morphologies, such as single-arm, dual-arm, and mobile manipulators.

3. **Crafting an effective training recipe**, as recent advances in NLP and vision have heavily relied on careful pre-training and post-training strategies. Incorporating methods like **action chunking** and **efficient tokenization** ensures that models can handle long-horizon and high-frequency robotic tasks effectively.

In this post, we introduce **π0 and π0-FAST**, prototype models and learning frameworks developed by **Physical Intelligence**, designed to overcome these challenges.

---

## 🔍 What is π0?

[GitHub Repository](#) | [Hugging Face Collection](#)

π0 (**Pi-Zero**) is a cutting-edge **Vision-Language-Action (VLA) model** designed for **generalist robot control**. It builds upon **large-scale pretraining** and **flow matching-based action generation**, enabling robots to perform **dexterous manipulation tasks** across different embodiments.

Developed by the [Physical Intelligence team](https://www.physicalintelligence.company), π0 is trained on data from **7 robotic platforms** and **68 unique tasks**, demonstrating strong **zero-shot** and **fine-tuned performance** on complex, real-world tasks such as **laundry folding, table bussing, grocery bagging, box assembly, and object retrieval**.

Unlike standard robotic policies, **π0 employs flow matching** to produce **smooth, real-time action trajectories at 50Hz**, making it highly **efficient, precise, and adaptable** for real-world deployment.

### ✅ Key Features:
- **Trained on:** 7 robotic platforms, 68 unique tasks  
- **Real-time execution:** Generates smooth action trajectories at 50Hz   

### ✅ Real-world applications:
- **Laundry folding**
- **Table bussing**
- **Grocery bagging**
- **Box assembly**
- **Object retrieval**
---

## What is the difference between VLMs and VLAs?

[Pablo's paragraph on how to compute attention mask for VLAs]

---

##  How to effectively represent Actions?

Action representation in training **Vision-Language-Action (VLA) models** directly affects model efficiency, generalization, and execution fidelity. Various approaches have been explored to parameterize robot actions. One strategy involves **semantic action representations**, where actions are described as high-level concepts such as **language sub-tasks or keypoints**. While these methods allow for **few-shot or zero-shot learning**, they often rely on **hand-designed low-level controllers**, limiting their applicability to diverse robotic platforms. Alternatively, **low-level control representations** map robot actions directly to motor commands, enabling **fine-grained dexterity** but introducing challenges in scalability and training stability. 

Most existing VLAs use **discrete action tokenization**, where continuous robot actions are mapped into discrete tokens that can be generated autoregressively. The dominant approach is **per-dimension, per-timestep binning**, but this method struggles with **high-frequency control tasks**, leading to **lossy representations and inefficient training**. To overcome this, **compression-based tokenization** methods, such as **vector quantization (VQ) and time-series compression**, have been proposed. While VQ-based approaches provide structured representations, they can be **sensitive to hyperparameter choices**, making them less robust for diverse robotic embodiments. 

To address these limitations, **Frequency-space Action Sequence Tokenization (FAST)** introduces a novel **time-series compression approach** based on the **Discrete Cosine Transform (DCT)**. FAST reduces redundancy in action sequences, improves training efficiency, and enhances action fidelity. 

---
## 🚀 What is π0-FAST?

π0-FAST is an **autoregressive version** of π0, introducing **FAST (Frequency-space Action Sequence Tokenization)**—a new tokenization scheme that enhances efficiency and performance.

### Key Advantages of π0-FAST:
- **5x faster training** compared to diffusion-based VLAs.
- **Improved action representation**, reducing redundancy in action sequences.
- **Stronger generalization** across unseen environments and robot morphologies.

🔗 The **π0-FAST tokenizer** can be accessed here: [FAST Tokenizer](https://huggingface.co/physical-intelligence/fast)

---

## How to use FAST tokenizer? 

**FAST (Frequency-space Action Sequence Tokenization)** is a **universal action tokenizer** trained on **1M real robot action sequences**. It converts any sequence of robot actions into a sequence of **dense, discrete action tokens**, making it highly efficient for training **autoregressive VLAs**.

🔗 Code for training custom action tokenizers: [FAST Repo](https://huggingface.co/physical-intelligence/fast)

FAST is integrated into **Hugging Face Transformers** and can be easily used for encoding and decoding robot action sequences.

