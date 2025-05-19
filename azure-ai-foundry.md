---
title: "Microsoft and Hugging Face expand collaboration" 
thumbnail: /blog/assets/azure-ai-foundry/satya-hf-build-compressed.png
authors:
- user: jeffboudier
- user: pagezyhf
- user: alvarobartt
---

# Microsoft and Hugging Face expand collaboration to make open models easy to use on Azure

Today at the Microsoft Build conference, Satya Nadella announced an expanded collaboration with Hugging Face, to make its wide diversity of open models easy to deploy on Azure secure infrastructure.

<img src="https://huggingface.co/blog/assets/azure-ai-foundry/satya-hf-build-compressed.png" alt="Satya Nadella announcing an expanded collaboration with Hugging Face at Microsoft Build">

If you head over to [Azure AI Foundry](https://ai.azure.com) today, you will find a vastly expanded collection of 10,000+ Hugging Face models you can deploy in a couple clicks to power AI applications working with text, audio and images. And we’re just getting started!

## It’s time to build - an expanded collaboration

2 years ago, Microsoft and Hugging Face [started a collaboration](https://huggingface.co/blog/hugging-face-endpoints-on-azure) to make open models more easily accessible on Azure - back then the Hub was home to 200,000 open models.

With now close to 2 million open models on Hugging Face, covering a wide diversity of tasks, modalities, domains and languages, it was time to take our collaboration to the next level. The new collaboration announced today creates a framework for mutual success to vastly expand how Azure customers can benefit from Hugging Face.

> "Open source is how AI moves faster—with transparency, choice, and community at its core. This collaboration represents our commitment to that momentum. By combining Hugging Face’s vibrant model ecosystem with Azure’s secure, enterprise-grade infrastructure, we’re giving developers the freedom to pick the best model for the job—and helping organizations innovate safely and at scale.”

-- _Asha Sharma, Corporate Vice President at Microsoft_

Making more open models easily accessible to Azure customers, for secure deployment alongside company private data, will enable enterprises to build AI applications and agents while being fully in control of their technology and data.

> We’re enabling companies to take control of their AI destiny, deploying the best open models securely within their Azure account, to build AI applications they can trust and verify.

-- _Clement Delangue, CEO and cofounder at Hugging Face_

## How to use Hugging Face in Azure AI Foundry

Let’s head over to Azure AI Foundry, and select the Model Catalog. Here you can now find over 10,000 models under the Hugging Face Collection.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/azure-ai-foundry/azure-ai-foundry-collection.png" alt="The Hugging Face Collection in the Model Catalog within Azure AI Foundry">

Models in the Collection include the most popular, trending models on Hugging Face for a wide range of tasks to work with text, audio and images - including text generation, feature extraction, fill-mask, translation, identifying sentence similarity, image classification, image segmentation, text to image generation, image to text conversion, automatic speech recognition and audio-classification.

To make the Hugging Face Collection on Azure AI Foundry enterprise-ready, we are only featuring models:
- having successfully passed Hugging Face security tests to screen for any vulnerability, including with [ProtectAI Guardian](https://huggingface.co/docs/hub/en/security-protectai) and [JFrog security scanner](https://huggingface.co/docs/hub/en/security-jfrog)
- with model weights stored in [safetensors](https://huggingface.co/docs/safetensors/main/en/index) format, avoiding potential Pickle vulnerabilities
- without [remote code](https://huggingface.co/docs/transformers/main/en/models#custom-models), to avoid any arbitrary code insertion at runtime.

In addition, Microsoft and Hugging Face will continuously test inference containers for vulnerabilities to maintain and patch them as needed.

Now let’s say you want to deploy for instance the popular [Microsoft Phi-4 Reasoning Plus](https://huggingface.co/microsoft/Phi-4-reasoning-plus) open model. 

First, let’s select the model in the Hugging Face Collection in Azure AI Foundry, and click the “Deploy button”. The form allows you to select a VM, instance count and deployment parameters, and start the deployment process with just another click!

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/azure-ai-foundry/azure-ai-foundry-deploy.png" alt="A form to deploy a Hugging Face model in Azure AI Foundry">

Now, if you prefer browsing models on the Hub, you can also start from the model page - the “Deploy on Azure ML” option will take you to the same deployment option within Azure AI Machine Learning Studio. 

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/azure-ai-foundry/huggingface-hub-deploy-azure.png" alt="A menu to deploy a model to Azure ML from a Hugging Face model page">

## More Hugging Face to come in Azure AI Foundry

We are really excited about all the new Hugging Face models and modalities now directly available within Azure AI Foundry, but we’re not going to stop there!

In the weeks and months to come, you can expect a rolling thunder of updates:
- Day-0 releases - Hugging Face will collaborate with Microsoft to make new models from top model providers available in Azure AI Foundry the same day they land on Hugging Face
- Trending models updates - Hugging Face will continuously monitor trending models to enable them on Azure AI Foundry on a daily basis
- New modalities - Hugging Face and Microsoft will work together to enable more modalities and domain-specific tasks, including video, 3D, time series, protein and more.
- Agents and tools - Small, efficient, specialized open models make them ideal to build powerful but secure AI agents and applications

If you’re on Azure, it’s time to build with open models!
