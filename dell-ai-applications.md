---
title: "Dell Enterprise Hub is all you need to build AI on premises" 
thumbnail: /blog/assets/dell-ai-applications/dell-post-thumbnail.png
authors:
- user: jeffboudier
- user: andrewrreed
- user: pagezyhf
- user: alvarobartt
- user: beurkinger
- user: florentgbelidji
- user: ark393
- user: balaatdell
  guest: true
  org: DellTechnologies
---

# Dell Enterprise Hub is all you need to build AI on premises

This week at Dell Tech World, we announced the new version of [Dell Enterprise Hub](https://dell.huggingface.co/), with a complete suite of models and applications to easily build AI running on premises with Dell AI servers and AI PCs.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dell-ai-applications/dell-post-thumbnail.png" alt="Dell and Hugging Face announcing the Dell Enterprise Hub">

## Models Ready for Action

If you go to the Dell Enterprise Hub today, you can find some of the most popular models, like [Meta Llama 4 Maverick](https://dell.huggingface.co/authenticated/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct), [DeepSeek R1](https://dell.huggingface.co/authenticated/models/deepseek-ai/deepseek-r1) or [Google Gemma 3](https://dell.huggingface.co/authenticated/models/google/gemma-3-27b-it), available for deployment and training in a few clicks.

But what you get is much more than a model, it’s a fully tested container optimized for specific Dell AI Server Platforms, with easy instructions to deploy on-premises using Docker and Kubernetes.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dell-ai-applications/dell-blog-1.png" alt="Meta Llama 4 Maverick available for Dell AI Server platforms"><br>
<em>Meta Llama 4 Maverick can be deployed on NVIDIA H200 or AMD MI300X Dell PowerEdge servers</em>

We continuously work with Dell CTIO and Engineering teams to make the latest and greatest models ready, tested and optimized for Dell AI Server platforms as quickly as possible - Llama 4 models were available on the Dell Enterprise Hub within 1 hour of their public release by Meta!

## Introducing AI Applications

The Dell Enterprise Hub now features ready-to-deploy AI Applications!

If models are engines, then applications are the cars that make them useful so you can actually go places. With the new Application Catalog you can build powerful applications that run entirely on-premises for your employees and use your internal data and services.

The new [Application Catalog](https://dell.huggingface.co/authenticated/apps) makes it easy to deploy leading open source applications within your private network, including [OpenWebUI](https://github.com/open-webui/open-webui) and [AnythingLLM](https://anythingllm.com/).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dell-ai-applications/dell-blog-2.png" alt="OpenWebUI and AnythingLLM available in the Dell Application Catalog">

[OpenWebUI](https://dell.huggingface.co/authenticated/apps/openwebui) makes it easy to deploy on-premises chatbot assistants that connect to your internal data and services via MCP, to build agentic experiences that can search the web, retrieve internal data with vector databases and storage for RAG use cases. 

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dell-ai-applications/dell-blog-3-openwebui.png" alt="OpenWebUI User Interface">

[AnythingLLM](https://dell.huggingface.co/authenticated/apps/anythingllm) makes it easy to build powerful agentic assistants connecting to multiple MCP servers so you can connect your internal systems or even external services. It includes features to enable multiple models, working with images, documents and set role-based access controls for your internal users.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dell-ai-applications/dell-blog-4-anythingllm.png" alt="AnythingLLM User Interface">

These applications are easy to deploy using the provided, customizable helm charts so your MCP servers are registered from the get go.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dell-ai-applications/dell-blog-5-helm.png" alt="Deployment instructions for OpenWebUI on Dell AI Server">

## Powered by NVIDIA, AMD and Intel

Dell Enterprise Hub is the only platform in the world that offers ready-to-use model deployment solutions for the latest AI Accelerator hardware:
- NVIDIA H100 and H200 GPU powered Dell platforms
- AMD MI300X powered Dell platforms
- Intel Gaudi 3 powered Dell platforms

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dell-ai-applications/dell-blog-6.png" alt="Dell Enterprise Hub supports NVIDIA, AMD and Intel powered Dell AI Servers">

We work directly with Dell, NVIDIA, AMD and Intel so that when you deploy a container on your system, it’s all configured and ready to go, has been fully tested and benchmarked so it runs with the best performance out of the box on your Dell AI Server platform.

## On-Device Models for Dell AI PC

The new Dell Enterprise Hub now provides support for models to run on-device on Dell AI PCs in addition to AI Servers!

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dell-ai-applications/dell-blog-7-AI-PC.png" alt="Model Catalog includes many models for on-device inference on Dell AI PCs">

These models enable on-device speech transcription (OpenAI whisper), chat assistants (Microsoft Phi and Qwen 2.5), upscaling images and generating embeddings.

To deploy a model, you can follow specific instructions for the Dell AI PC of your choice, powered by Intel or Qualcomm NPUs, using the new [Dell Pro AI Studio](https://www.dell.com/en-us/lp/dell-pro-ai-studio). Coupled with PC fleet management systems like Microsoft Intune, it’s a complete solution for IT organizations to enable employees with on-device AI capabilities.

## Now with CLI and Python SDK

Dell Enterprise Hub offers an online portal into AI capabilities for Dell AI Server platforms and AI PCs. But what if you want to work directly from your development environment?

Introducing the new [dell-ai open source library](https://github.com/huggingface/dell-ai) with a Python SDK and CLI, so you can use Dell Enterprise Hub within your environment directly from your terminal or code - just `pip install dell-ai`

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dell-ai-applications/dell-blog-8-CLI.png" alt="Available commands for dell-ai CLI">

## Wrapping up

With Models and Applications, for AI Servers and AI PCs, easily installable using Docker, Kubernetes and Dell Pro AI Studio, Dell Enterprise Hub is a complete toolkit to deploy Gen AI applications in the enterprise, fully secure and on-premises.

As a Dell customer, that means you can very quickly, within an hour instead of weeks:
- roll out an in-network chat assistant powered by the latest open LLMs, and connect it to your internal storage systems (ex. Dell PowerScale) using MCP, all in an air gapped environment
- give access to complex agentic systems, with granular access controls and SSO, that can work with internal text, code, images, audio and documents and access the web for current context
- set up employees with on-device, private transcription powered by a fleet of Dell AI PCs in a fully managed way

If you are using [Dell Enterprise Hub](https://dell.huggingface.co/) today, we would love to hear from you in the comments!
