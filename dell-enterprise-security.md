---
title: "Security, Governance, and Performance for Dell On-Prem AI Builders"
thumbnail: /blog/assets/dell-enterprise-hub/thumbnail.jpg
authors:
  - user: pagezyhf
  - user: alvarobartt
  - user: juanjucm
  - user: jeffboudier
  - user: balaattdell
---

# Security, Governance and Performance for Dell On-Prem AI Builders

![Dell Enterprise Hub updates](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deh-2025/thumbnail.jpg)

A year ago we introduced the [Dell Enterprise Hub](https://dell.huggingface.co), a new experience from Hugging Face to make it easy to train and deploy open models on-premise using Dell platforms.

Since that launch, Dell Enterprise Hub has grown from a model catalog into a full on-prem AI experience: you can browse open models, deploy them on Dell AI servers and AI PCs with optimized configurations, fine-tune them with your own data, and, more recently, even deploy complete AI applications through the Application Catalog.

Today we are introducing the next wave of capabilities, focused on three things that matter a lot to enterprises: **security, governance, and performance**.

All the new features discussed below are available today - try them on [dell.huggingface.co](https://dell.huggingface.co).

## Securing the AI supply chain

As more AI workloads move into production, teams care not just about which model they use, but also about how it gets into their infrastructure. What is inside the Docker image? Has the model repository been scanned? How are the weights pulled into the cluster? The new Dell Enterprise Hub experience answers these questions directly on the model page so that AI, infrastructure and security teams all have access to transparent information.

Every model on the Hugging Face Hub is scanned for malware and unsafe serialization formats. Dell Enterprise Hub now surfaces a summary of these **repository scan results** directly in the model view. This gives security and compliance teams a starting point for their own reviews, without having to go into deep investigation.

![Model scan results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deh-2025/model-scan-results.png)

Models are only one piece of the supply chain. The container image that runs those models also needs to be monitored. Dell Enterprise Hub uses custom Docker images for inference and training, optimized per model and per Dell platform. These images are regularly scanned with AWS Inspector, and Dell Enterprise Hub now exposes **container scan status** alongside the deployment configuration.

![Container scan status](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deh-2025/container-scan-status.png)

To enable Enterprises to implement model access governance, Dell Enterprise Hub now standardizes the use of **Hugging Face access tokens** in its deployment experiences. HF Tokens authenticate your calls to the Hub, ensure access permissions to gated models are respected, and give users higher rate limits when pulling model weights from Hugging Face.

![HF Token](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deh-2025/hf-token.png)

Together, these features enable enterprises to build their own AI on-premises with improved security and governance. Dell Enterprise Hub offers a simple, transparent way to secure the AI model supply chain and govern model access.

## Performance out of the box

Once the security posture is in a good place, the next question is performance. The goal for Dell Enterprise Hub is to offer performance out of the box with optimized configurations for each combination of model and Dell system, without having to fiddle with every inference engine and deployment parameter.

When we launched Dell Enterprise Hub in May 2024, most model deployment containers were built on top of Hugging Face **Text Generation Inference (TGI)**. Today, each model is offered with the best available runtime, including **vLLM** or **SGLang** engines based on the model and Dell system, and the model deployment code snippets are preconfigured with tested parameters. You pick the model and Dell platform; Dell Enterprise Hub picks a runtime and configuration that works well out of the box.

Looking ahead, Dell Enterprise Hub will offer more opinionated choices defining the default configuration parameters for each deployment snippet, with presets for different use cases. You will still be able to override any of these values in the generated command to experiment, with the goal that teams get strong results out of the box by simply copying the code snippet into their Dell environment or using the [dell-ai CLI](https://github.com/huggingface/dell-ai).

## Lifecycle: Containers versioning and decoupling model weights

Another major update in Dell Enterprise Hub is the introduction of container versioning, and the decoupling of containers and model weights, to improve AI developer experience and lifecycle management. Without Dell Enterprise Hub, enterprises need to continuously patch base images, upgrade inference engines, rotate models and archive older assets, often under strict compliance requirements. To make that easier, Dell Enterprise Hub now implements **decoupled container architecture with explicit versioning**.

Historically, many Dell Enterprise Hub containers shipped with the model weights within the container. This made the “first run” very straightforward, but it also led to large images and tighter coupling between the model and the runtime environment. From now on, new containers added to Dell Enterprise Hub are provided **without pre-downloaded weights**, by default pulled from the Hugging Face Hub on runtime. If required, it is still possible to download the model weights in advance and mount them into the container.

Additionnaly, instead of relying on a single `latest` tag for containers, Dell Enterprise Hub now exposes **versioned tags**. This means you can pin an exact container tag in production, test a newer container in staging, and move between them on your own schedule. The tags include the inference engine name and version to make debugging and experimentation more transparent.

![Container Versioning](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deh-2025/container-versioning.png)

## What’s next

These changes are another step towards making Dell Enterprise Hub the easiest way to run open models and applications on Dell platforms, fully on-premise and under your control.

We will continue to add support for new models, new modalities and new Dell platforms. Expect improved configurations built around real-world latency and throughput requirements, and a deeper integration between the Dell Enterprise Hub and [Hugging Face Enterprise](https://huggingface.co/enterprise).

To follow along, make sure to follow the [Dell Technologies organization](https://huggingface.co/DellTechnologies)!
