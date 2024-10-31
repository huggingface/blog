---
title: "Introducing HUGS - Scale your AI with Open Models" 
thumbnail: /blog/assets/hugs/thumbnail.jpg
authors:
- user: philschmid
- user: jeffboudier
- user: alvarobartt
- user: pagezyhf
- user: Violette
---
 

Today, we are thrilled to announce the launch of **Hugging Face Generative AI Services a.k.a. HUGS**: optimized, zero-configuration inference microservices designed to simplify and accelerate the development of AI applications with open models. Built on open-source Hugging Face technologies such as Text Generation Inference and Transformers, HUGS provides the best solution to efficiently build and scale Generative AI Applications in your own infrastructure. HUGS is optimized to run open models on a variety of hardware accelerators, including NVIDIA GPUs, AMD GPUs, and soon AWS Inferentia and Google TPUs.

![HUGS Banner](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hugs/hugs-banner.png)


## Zero-Configuration Optimized Inference for Open Models

HUGS simplifies the optimized deployment of open models in your own infrastructure and on a wide variety of hardware. One key challenge developers and organizations face is the engineering complexity of optimizing inference workloads for LLMs on a particular GPU or AI accelerator. With HUGS, we enable maximum throughput deployments for the most popular open LLMs with zero configuration required. Each deployment configuration offered by HUGS is fully tested and maintained to work out of the box.

HUGS model deployments provide an OpenAI compatible API for a drop-in replacement of existing Generative AI applications built on top of model provider APIs. Just point your code to the HUGS deployment to power your applications with open models hosted in your own infrastructure.

## Why HUGS?

HUGS offers an easy way to build AI applications with open models hosted in your own infrastructure, with the following benefits:

* **In YOUR infrastructure**: Deploy open models within your own secure environment. Keep your data and models off the Internet!  
* **Zero-configuration Deployment**: HUGS reduces deployment time from weeks to minutes with zero-configuration setup, automatically optimizing the model and serving configuration for your NVIDIA, AMD GPU or AI accelerator.  
* **Hardware-Optimized Inference**: Built on Hugging Face's Text Generation Inference (TGI), HUGS is optimized for peak performance across different hardware setups.  
* **Hardware Flexibility**: Run HUGS on a variety of accelerators, including NVIDIA GPUs, AMD GPUs, with support for AWS Inferentia and Google TPUs coming soon.  
* **Model Flexibility**: HUGS is compatible with a wide selection of open-source models, ensuring flexibility and choice for your AI applications.  
* **Industry Standard APIs**: Deploy HUGS easily using Kubernetes with endpoints compatible with the OpenAI API, minimizing code changes.  
* **Enterprise Distribution:** HUGS is an enterprise distribution of Hugging Face open source technologies, offering long-term support, rigorous testing, and SOC2 compliance.  
* **Enterprise Compliance**: Minimizes compliance risks by including necessary licenses and terms of service.

**We provided early access to HUGS to select Enterprise Hub customers:**

> HUGS is a huge timesaver to deploy locally ready-to-work models with good performances \- before HUGS it would take us a week, now we can be done in less than 1 hour. For customers with sovereign AI requirements it's a game changer! - [Henri Jouhaud](https://huggingface.co/henrij), CTO at [Polyconseil](https://huggingface.co/polyconseil)

> We tried HUGS to deploy Gemma 2 on GCP using a L4 GPU \- we didn't have to fiddle with libraries, versions and parameters, it just worked out of the box. HUGS gives us confidence we can scale our internal usage of open models! - [Ghislain Putois](https://huggingface.co/ghislain-putois), Research Engineer at [Orange](https://huggingface.co/Orange)

## How it Works

Using HUGS is straightforward. Here's how you can get started:

*Note: You will need access to the appropriate subscription or marketplace offering depending on your chosen deployment method.*

### Where to find HUGS

HUGS is available through several channels:

1. **Cloud Service Provider (CSP) Marketplaces**: You can find and deploy HUGS on [Amazon Web Services (AWS)](https://aws.amazon.com/marketplace/pp/prodview-bqy5zfvz3wox6) with [Google Cloud Platform (GCP)](https://console.cloud.google.com/marketplace/product/huggingface-public/hugs) and [Microsoft Azure](https://huggingface.co/docs/hugs/how-to/cloud/azure) support coming soon (being published at the moment)  
2. **DigitalOcean**: HUGS is natively available within [DigitalOcean as a new 1-Click Models service](http://digitalocean.com/blog/one-click-models-on-do-powered-by-huggingface), powered by Hugging Face HUGS and GPU Droplets.  
3. **Enterprise Hub**: If your organization is upgraded to Enterprise Hub, [contact our Sales team](https://huggingface.co/contact/sales?from=hugs) to get access to HUGS.

For specific deployment instructions for each platform, please refer to the relevant documentation linked above.

### Pricing

HUGS offers on-demand pricing based on the uptime of each container, except for deployments on DigitalOcean.

* **AWS Marketplace and Google Cloud Platform Marketplace:** $1 per hour per container, no minimum fee (compute usage billed separately by CSP). On AWS you have 5 day free trial period for you to test HUGS for free.   
* **DigitalOcean:** 1-Click Models powered by Hugging Face HUGS are available at no additional cost on DigitalOcean - regular GPU Droplets compute costs apply.   
* **Enterprise Hub:** We offer custom HUGS access to Enterprise Hub organizations. Please [contact](https://huggingface.co/contact/sales?from=hugs) our Sales team to learn more.

### Running Inference

HUGS is based on Text Generation Inference (TGI), offering a seamless inference experience. For detailed instructions and examples, refer to the [Run Inference on HUGS](https://huggingface.co/docs/hugs/guides/inference) guide. HUGS leverages the OpenAI-compatible Messages API, allowing you to use familiar tools and libraries like cURL, the `huggingface_hub` SDK, and the `openai` SDK for sending requests.

```py
from huggingface_hub import InferenceClient

ENDPOINT_URL="REPLACE" # replace with your deployed url or IP

client = InferenceClient(base_url=ENDPOINT_URL, api_key="-")

chat_completion = client.chat.completions.create(
    messages=[
        {"role":"user","content":"What is Deep Learning?"},
    ],
    temperature=0.7,
    top_p=0.95,
    max_tokens=128,
)
```

## Supported Models and Hardware

HUGS supports a growing ecosystem of open models and hardware platforms. Refer to our [Supported Models](https://huggingface.co/docs/hugs/models) and [Supported Hardware](https://huggingface.co/docs/hugs/hardware) pages for the most up-to-date information. 

We launch today with 13 popular open LLMs: 

* [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  
* [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)  
* [meta-llama/Llama-3.1-405B-Instruct-FP8](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct-FP8)  
* [NousResearch/Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)  
* [NousResearch/Hermes-3-Llama-3.1-70B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B)  
* [NousResearch/Hermes-3-Llama-3.1-405B-FP8](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B-FP8)  
* [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)  
* [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)  
* [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)  
* [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)  
* [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it)  
* [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)  
* [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

For a detailed view of supported Models x Hardware, check out the [documentation](https://huggingface.co/docs/hugs/models). 

## Get Started with HUGS Today

HUGS makes it easy to harness the power of open models, with zero-configuration optimized inference in your own infra. With HUGS, you can take control of your AI applications and easily transition proof of concept applications built with closed models to open models you host yourself. 

Get started today and deploy HUGS on [AWS](https://aws.amazon.com/marketplace/pp/prodview-bqy5zfvz3wox6), [Google Cloud](https://console.cloud.google.com/marketplace/product/huggingface-public/hugs) or [DigitalOcean](https://www.digitalocean.com/products/ai-ml/1-click-models)!
