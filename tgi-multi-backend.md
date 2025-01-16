---
title: "Introducing multi-backends (TRT-LLM, vLLM) support for Text Generation Inference" 
thumbnail: /blog/assets/tgi-multi-backend/thumbnail.png
authors:
- user: mfuntowicz
- user: hlarcher
---

# Introducing multi-backends (TRT-LLM, vLLM) support for Text Generation Inference

## Introduction

Since its initial release in 2022, Text-Generation-Inference (TGI) has provided Hugging Face and the AI Community with a performance-focused solution to easily deploy large-language models (LLMs). TGI initially offered an almost no-code solution to load models from the Hugging Face Hub and deploy them in production on NVIDIA GPUs. Over time, support expanded to include AMD Instinct GPUs, Intel GPUs, AWS Trainium/Inferentia, Google TPU, and Intel Gaudi.  
Over the years, multiple inferencing solutions have emerged, including vLLM, SGLang, llama.cpp, TensorRT-LLM, etc., splitting up the overall ecosystem. Different models, hardware, and use cases may require a specific backend to achieve optimal performance. However, configuring each backend correctly, managing licenses, and integrating them into existing infrastructure can be challenging for users.

To address this, we are excited to introduce the concept of TGI Backends. This new architecture gives the flexibility to integrate with any of the solutions above through TGI as a single unified frontend layer. This change makes it easier for the community to get the best performance for their production workloads, switching backends according to their modeling, hardware, and performance requirements. 

The Hugging Face team is excited to contribute to and collaborate with the teams that build vLLM, llama.cpp, TensorRT-LLM, and the teams at AWS, Google, NVIDIA, AMD and Intel to offer a robust and consistent user experience for TGI users whichever backend and hardware they want to use.

![TGI multi-backend stack](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tgi-multi-backend/tgi-multi-backend-stack.png)

## TGI Backend: under the hood

TGI is made of multiple components, primarily written in Rust and Python. Rust powers the HTTP and scheduling layers, and Python remains the go-to for modeling.

Long story short: Rust allows us to improve the overall robustness of the serving layer with static analysis and compiler-based memory safety enforcement: it brings the ability to scale to multiple cores with the same safety guarantees more easily. Leveraging Rust’s strong type system for the HTTP layer and scheduler makes it possible to avoid memory issues while maximizing the concurrency, bypassing Global Interpreter Lock (GIL) in Python-based environments. 

Speaking about Rust… Surprise, that's the TGI starting point to integrate a new backend \- 🤗

Earlier this year, the TGI team worked on exposing the foundational knobs to disentangle how the actual HTTP server and the scheduler were coupled together. 
This work introduced the new Rust `trait Backend` to interface current inference engine and the one to come.

Having this new `Backend` interface (or trait in Rusty terms) paves the way for modularity and makes it possible to actually route the incoming requests towards different modeling and execution engines.

## Looking forward: 2025

The new multi-backend capabilities of TGI open up many impactful roadmap opportunities. As we look ahead to 2025 we are excited to share some of the TGI developments we are most excited about:

* **NVIDIA TensorRT-LLM backend**: We are collaborating with the NVIDIA TensorRT-LLM team to bring all the optimized NVIDIA GPUs \+ TensorRT performances to the community. This work will be covered more extensively in an upcoming blog post. It closely relates to our mission to empower AI builders with the open-source availability of both `optimum-nvidia` quantize/build/evaluate TensorRT compatible artifacts alongside TGI+TRT-LLM to easily deploy, execute, and scale deployments on NVIDIA GPUs.
* **Llama.cpp backend**: We are collaborating with the llama.cpp team to extend the support for server production use cases. The llama.cpp backend for TGI will provide a strong CPU-based option for anyone willing to deploy on Intel, AMD, or ARM CPU servers.
* **vLLM backend**: We are contributing to the vLLM project and are looking to integrate vLLM as a TGI backend in Q1 '25.
* **AWS Neuron backend**: we are working with the Neuron teams at AWS to enable Inferentia 2 and Trainium 2 support natively in TGI.
* **Google TPU backend**: We are working with the Google Jetstream & TPU teams to provide the best performance through TGI.

We are confident TGI Backends will help simplify the deployments of LLMs, bringing versatility and performance to all TGI users.
You'll soon be able to use TGI Backends directly within [Inference Endpoints](https://huggingface.co/inference-endpoints/). Customers will be able to easily deploy models with TGI Backends on various hardware with top-tier performance and reliability out of the box.

Stay tuned for the next blog post where we'll dig into technical details and performance benchmarks of upcoming backends\!
