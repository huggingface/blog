---
title: "π0 and π0-FAST: Vision-Language-Action Models for General Robot Control" 
thumbnail: /blog/assets/192_pi0/thumbnail_pi0.001.png
authors:
- user: danaaubakirova
- user: Molbap
- user: mshukor
- user: cadene

---

We have ported the first **robot foundation models** to **Hugging Face LeRobot**! Both **π0 and π0-FAST** developed by Physical Intelligence, which are now available in the **LeRobot repository**, bringing generalist robotic intelligence to the Hugging Face ecosystem. If you are curious about how Vision-Language-Action (VLA) models differ from Vision-Language Models (VLMs) and how actions are represented, dive into this blog post to find out! 

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

π0 (**Pi-Zero**) is a **Vision-Language-Action (VLA) model**, developed by the [Physical Intelligence team](https://www.physicalintelligence.company) designed for **generalist robot control**. It builds upon **large-scale pretraining** and **flow matching-based action generation**, enabling robots to perform **dexterous manipulation tasks** across different embodiments.

π0 is trained on data from **7 robotic platforms** and **68 unique tasks**, demonstrating strong **zero-shot** and **fine-tuned performance** on complex, real-world tasks such as **laundry folding, table bussing, grocery bagging, box assembly, and object retrieval**.

Unlike standard robotic policies, **π0 employs flow matching** to produce **smooth, real-time action trajectories at 50Hz**, making it highly **efficient, precise, and adaptable** for real-world deployment.

### Key Features:
- **Trained on:** 7 robotic platforms, 68 unique tasks  
- **Real-time execution:** Generates smooth action trajectories at 50Hz   

### Real-world applications:
- **Laundry folding**
- **Table bussing**
- **Grocery bagging**
- **Box assembly**
- **Object retrieval**
---

## What is the difference between VLMs and VLAs?

Vision-Language Models (VLMs) and Vision-Language-Action Models (VLAs) share a common foundation: transformers. However, the key distinction lies in action representation. While VLMs process and generate multimodal representations (images and text), VLAs extend this by incorporating action and observation state tokens. With these additional tokens in place, the next challenge is understanding how attention is computed.

## Attention Mechanisms in Robotics Policies

Let’s expand our vocabulary and introduce key terms:

### **State Tokens**
- Represent the robot’s **current environment state** (e.g., joint angles, sensor values, or other relevant observations).
- The masking rules allow this token to **attend to the prefix’s image and text**, meaning the state token can “see” any visual or textual cues necessary for decision-making.
- It also attends to **previous states in a triangular manner**. If multiple state tokens are used, each new state token can see older ones but not vice versa.

### **Action Tokens**
- Represent the **motor command sequence**.
- Have **full visibility** over everything except padding regions. This means each action token can attend to:
  - **All non-padding image tokens** (the entire scene),
  - **All non-padding text tokens** (instructions or descriptions),
  - **State tokens** (both current and previous),
  - **Other action tokens**.

### **Prefix Tokens**
- Represent the **full scene** and fully attend to each other, similar to **PaliGemma**.

### **Key Idea**
These tokens encapsulate:
- The **robot’s internal representation** of the environment (**state**),
- The **commands or controls** the robot issues (**action**),
- An encoding of **time or step index** (**time embedding**).

They are **appended after** the prefix portion (images + text), so the **prefix serves as context** (e.g., a scene image, language instructions like *"be a good robot"* or *"transfer the cube"*), while the **suffix captures policy‐specific features**.

---

## (Fast) Attention Is All You Want

We’ve made some optimizations, and the version we are porting runs a bit **faster** than the original model. 🚀
This means we can process data more efficiently, but it also brings some **interesting challenges** in how attention is computed.

### **Handling 2D Attention Masks**
The resulting **2D attention mask** exhibits **block sparsity**, but defining the boundaries of each block—especially in a batch of samples—is tricky. 

### **FlashAttention2 Challenges**
- FlashAttention2 provides a **varlen interface**, but the `cu_seqlens` (cumulative prefix lengths) **must be computed manually**.
- Varlen FlashAttention is designed for **contiguous (or strictly causal) attention patterns** with uniform query and key lengths.
- It does not **naturally handle irregular block masks** or arbitrary per-token “allowed” positions, which is exactly what we need.

### **Using FlexAttention in PyTorch**
For **FlexAttention** and its PyTorch interface, we explored two options:
1. **Adding a `score_mod`** to our causal mask in positions where attention is tuned out. However, even a scalar addition **significantly decreases FlexAttention’s performance**.
2. **Indexing our causal mask and passing the resulting boolean to `mask_mod`**—this works!

Here’s an example of how we applied it:

```python
# Example of indexing the causal mask and using mask_mod
causal_mask = generate_causal_mask(batch_size, seq_len)
mask_mod = causal_mask.bool()  # Convert to boolean mask
flex_attention_output = flex_attention(query, key, value, mask_mod=mask_mod)
```

##  How to effectively represent Actions?

Now that we know that actions are nothing less just a n-dimensional vector that can be tokenized, we can dive deeper into the issues with action representation. Action representation in training **Vision-Language-Action (VLA) models** directly affects model efficiency, generalization, and execution fidelity. Various approaches have been explored to parameterize robot actions. One strategy involves **semantic action representations**, where actions are described as high-level concepts such as **language sub-tasks or keypoints**. While these methods allow for **few-shot or zero-shot learning**, they often rely on **hand-designed low-level controllers**, limiting their applicability to diverse robotic platforms. Alternatively, **low-level control representations** map robot actions directly to motor commands, enabling **fine-grained dexterity** but introducing challenges in scalability and training stability. 

Most existing VLAs use **discrete action tokenization**, where continuous robot actions are mapped into discrete tokens that can be generated autoregressively. The dominant approach is **per-dimension, per-timestep binning**, but this method struggles with **high-frequency control tasks**, leading to **lossy representations and inefficient training**. To overcome this, **compression-based tokenization** methods, such as **vector quantization (VQ) and time-series compression**, have been proposed. While VQ-based approaches provide structured representations, they can be **sensitive to hyperparameter choices**, making them less robust for diverse robotic embodiments. 

To address these limitations, Frequency-space Action Sequence Tokenization (FAST) introduces a novel time-series compression approach based on the Discrete Cosine Transform (DCT). FAST reduces redundancy in action sequences, improves training efficiency, and enhances action fidelity. Hence, along with π0, we want to present π0-FAST which extends upon the previous work and introduces the new tokenizer to effectively deal with action representation.

---

FAST is a tokenization approach for robot actions that leverages the Discrete Cosine Transform (DCT) to efficiently compress continuous action sequences into discrete tokens. The process begins with normalizing raw robot actions, mapping the 1st and 99th quantiles of each action dimension to the range [-1,1], making the data more consistent across different robotic systems and robust to outliers. Each action dimension is then independently transformed using DCT, which converts the time-domain signal into the frequency domain. To reduce redundancy, insignificant coefficients are omitted through a scale-and-round operation, where a hyperparameter controls the trade-off between compression rate and reconstruction accuracy. The resulting DCT coefficient matrix, which is often sparse, is then flattened into a one-dimensional sequence of integers, interleaving low-frequency components across dimensions to preserve crucial information. To further compress the sequence, Byte Pair Encoding (BPE) is applied, merging frequently occurring patterns across dimensions while ensuring a fixed-size vocabulary that integrates seamlessly with vision-language models used in robotics. Since all operations are invertible, actions can be efficiently reconstructed from tokens, allowing lossless decoding. The tokenization pipeline has only two hyperparameters: the scaling coefficient applied before rounding and the BPE vocabulary size, both of which remain robust across different datasets. Compared to Vector Quantization (VQ)-based approaches, FAST provides higher policy performance, is simpler to tune, and avoids dataset-specific hyperparameter selection. Additionally, a universal version of FAST, called FAST+, has been trained on one million action sequences from single-arm, bimanual, and mobile manipulation robots, making it applicable across diverse robotic setups. FAST+ is available as a Hugging Face AutoProcessor, allowing users to tokenize action sequences with just a few lines of code. To achieve optimal compression, input actions should be quantile-normalized to [-1,1] before tokenization. The AutoProcessor module also enables users to train a custom FAST tokenizer on their own datasets with minimal effort. 

---
## 🚀 What is π0-FAST?

π0-FAST is an **autoregressive version** of π0, introducing **FAST (Frequency-space Action Sequence Tokenization)**—a new tokenization scheme that enhances efficiency and performance.

### Key Advantages of π0-FAST:
- **5x faster training** compared to diffusion-based VLAs.
- **Improved action representation**, reducing redundancy in action sequences.
- **Stronger generalization** across unseen environments and robot morphologies.

🔗 The **π0-FAST tokenizer** can be accessed here: [FAST Tokenizer](https://huggingface.co/physical-intelligence/fast)

---
FAST is a tokenization approach for robot actions that leverages the Discrete Cosine Transform (DCT) to efficiently compress continuous action sequences into discrete tokens. The process begins with normalizing raw robot actions, mapping the 1st and 99th quantiles of each action dimension to the range [-1,1], making the data more consistent across different robotic systems and robust to outliers. Each action dimension is then independently transformed using DCT, which converts the time-domain signal into the frequency domain. To reduce redundancy, insignificant coefficients are omitted through a scale-and-round operation, where a hyperparameter controls the trade-off between compression rate and reconstruction accuracy. The resulting DCT coefficient matrix, which is often sparse, is then flattened into a one-dimensional sequence of integers, interleaving low-frequency components across dimensions to preserve crucial information. To further compress the sequence, Byte Pair Encoding (BPE) is applied, merging frequently occurring patterns across dimensions while ensuring a fixed-size vocabulary that integrates seamlessly with vision-language models used in robotics. Since all operations are invertible, actions can be efficiently reconstructed from tokens, allowing lossless decoding. The tokenization pipeline has only two hyperparameters: the scaling coefficient applied before rounding and the BPE vocabulary size, both of which remain robust across different datasets. Compared to Vector Quantization (VQ)-based approaches, FAST provides higher policy performance, is simpler to tune, and avoids dataset-specific hyperparameter selection. Additionally, a universal version of FAST, called FAST+, has been trained on one million action sequences from single-arm, bimanual, and mobile manipulation robots, making it applicable across diverse robotic setups. FAST+ is available as a Hugging Face AutoProcessor, allowing users to tokenize action sequences with just a few lines of code. To achieve optimal compression, input actions should be quantile-normalized to [-1,1] before tokenization. The AutoProcessor module also enables users to train a custom FAST tokenizer on their own datasets with minimal effort.

---
## How to use FAST tokenizer? 



🔗 Code for the usage and training custom action tokenizers in the official: [FAST Repo](https://huggingface.co/physical-intelligence/fast)

FAST is integrated into **Hugging Face Transformers** and can be easily used for encoding and decoding robot action sequences.

