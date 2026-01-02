---
title: "NVIDIA brings agents to life with DGX Spark and Reachy Mini" 
thumbnail: /blog/assets/nvidia-reachy-mini/nvidia-reachy-mini-compressed.png
authors:
- user: jeffboudier
- user: nader-at-nvidia
  guest: true
  org: nvidia
- user: alecfong
  guest: true
  org: nvidia
---

# NVIDIA brings agents to life with DGX Spark and Reachy Mini

![NVIDIA creates real world agents with DGX Spark and Reachy Mini at the CES 2026 Keynote](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nvidia-reachy-mini/nvidia-reachy-mini-compressed.png)

Today at CES 2026, NVIDIA unveiled a world of new open models to enable the future of agents, online and in the real world. From new [Nemotron reasoning LLMs](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) to the new [Gr00T VLA](https://huggingface.co/nvidia/GR00T-N1.6-3B) and [Cosmos world foundation models](https://huggingface.co/collections/nvidia/cosmos-reason2), all the building blocks are here today for AI Builders to build their own agents.

But what if you could bring your own agent to life, right at your desk? An AI buddy that can be useful to you and process your data privately?

In the CES keynote today, Jensen Huang showed us how we can do exactly that, using the processing power of [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) with [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) to create your own little office R2D2 you can talk to and collaborate with.

\<EMBED KEYNOTE VIDEO RECORDING\>

This blog post provides a step-by-step guide to replicate this amazing experience at home using a DGX Spark and [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/).

Let’s dive in!

## Ingredients

If you want to start cooking right away, here’s the [source code of the demo](https://github.com/brevdev/reachy-personal-assistant). 

We’ll be using the following:

1. A reasoning model: demo uses [NVIDIA Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)  
2. A vision model: demo uses [NVIDIA Nemotron Nano 2 VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)  
3. A text-to-speech model: demo uses [ElevenLabs](https://elevenlabs.io)  
4. [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) (or [Reachy Mini Simulation](https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/platforms/simulation/get_started.md))  
5. Python v3.10+ environment, with [uv](https://docs.astral.sh/uv/)

Feel free to adapt the recipe and make it your own \- you have many ways to integrate the models into your application:

1. Local deployment – Run on your own hardware ([DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) or a GPU with sufficient VRAM). Our implementation requires \~65GB disk space for the reasoning model, and \~28GB for the vision model.  
2. Cloud deployment– Deploy the models on cloud GPUs e.g. through [NVIDIA Brev](http://build.nvidia.com/gpu) or [Hugging Face Inference Endpoints](https://endpoints.huggingface.co/).  
3. Serverless model endpoints – Send requests to [NVIDIA](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/deploy) or [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index).
