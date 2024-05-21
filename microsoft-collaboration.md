---
title: "From cloud to developers: Hugging Face and Microsoft Deepen Collaboration" 
thumbnail: /blog/assets/microsoft-collaboration/thumbnail.jpg
authors:
- user: jeffboudier
- user: philschmid
---

# From cloud to developers: Hugging Face and Microsoft Deepen Collaboration 

Today at Microsoft Build we are happy to announce a broad set of new features and collaborations as Microsoft and Hugging Face deepen their strategic collaboration to make open models and open source AI easier to use everywhere. Together, we will work to enable AI builders across open science, open source, cloud, hardware and developer experiences - read on for announcements today on all fronts!

![MS BUILD Keynote announcement Hugging Face](/blog/assets/microsoft-collaboration/thumbnail.jpg)

## A collaboration for Cloud AI Builders

we are excited to announce two major new experiences to build AI with open models on Microsoft Azure.

### Expanded HF Collection in Azure Model Catalog

A year ago, Hugging Face and Microsoft [unveiled the Hugging Face Collection in the Azure Model Catalog](https://huggingface.co/blog/hugging-face-endpoints-on-azure). The Hugging Face Collection has been used by hundreds of Azure AI customers, with over a thousand open models available since its introduction. Today, we are adding some of the most popular open Large Language Models to the Hugging Face Collection to enable direct, 1-click deployment from Azure AI Studio. 

The new models include [Llama 3 from Meta](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct), [Mistral 7B from Mistral AI](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Command R Plus from Cohere for AI](https://huggingface.co/CohereForAI/c4ai-command-r-plus), [Qwen 1.5 110B from Qwen](https://huggingface.co/Qwen/Qwen1.5-110B-Chat), and some of the highest performing fine-tuned models on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) from the Hugging Face community.

To deploy the models in your own Azure account, you can start from the model card on the Hugging Face Hub, selecting the “Deploy on Azure” option:

![model-hub-azure](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/microsoft-collaboration/model-hub-azure.png)


Or you can find model directly in [Azure AI Studio](https://ai.azure.com) within the Hugging Face Collection, and click “Deploy”

![ai-studio-deploy](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/microsoft-collaboration/ai-studio-deploy.png)

### Build AI with the new AMD MI300X on Azure

Today, Microsoft made new [Azure ND MI300X virtual machines](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/azure-announces-new-ai-optimized-vm-series-featuring-amd-s/ba-p/3980770) (VMs) generally available on Azure, based on the latest AMD Instinct MI300 GPUs. Hugging Face collaborated with AMD and Microsoft to achieve amazing performance and cost/performance for Hugging Face models on the new virtual machines.

This work leverages our [deep collaboration with AMD](https://huggingface.co/blog/huggingface-and-optimum-amd) and our open source library [Optimum-AMD](https://github.com/huggingface/optimum-amd), with optimization, ROCm integrations and continuous testing of Hugging Face open source libraries and models on AMD Instinct GPUs. 


## A Collaboration for Open Science

Microsoft has been releasing some of the most popular open models on Hugging Face, with close to 300 models currently available in the [Microsoft organization on the Hugging Face Hub](https://huggingface.co/microsoft).

This includes the recent [Phi-3 family of models](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3), which are permissibly licensed under MIT, and offer performance way above their weight class. For instance, with only 3.8 billion parameters, Phi-3 mini outperforms many of the larger 7 to 10 billion parameter large language models, which makes the models excellent candidates for on-device applications.

To demonstrate the capabilities of Phi-3, Hugging Face [deployed Phi-3 mini in Hugging Chat](https://huggingface.co/chat/models/microsoft/Phi-3-mini-4k-instruct), its free consumer application to chat with the greatest open models and create assistants.


## A Collaboration for Open Source

Hugging Face and Microsoft have been collaborating for 3 years to make it easy to [export and use Hugging Face models with ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/overview), through the [optimum open source library](https://github.com/huggingface/optimum).

Recently, Hugging Face and Microsoft have been focusing on enabling local inference through WebGPU, leveraging [Transformers.js](https://github.com/xenova/transformers.js) and [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html). Read more about the collaboration in this [community article](https://huggingface.co/blog/Emma-N/enjoy-the-power-of-phi-3-with-onnx-runtime) by the ONNX Runtime team. 

To see the power of WebGPU in action, consider this [demo of Phi-3](https://x.com/xenovacom/status/1792661746269692412) generating over 70 tokens per second locally in the browser!

<video class="w-full" autoplay loop muted>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/microsoft-collaboration/phi-3-webgpu.mp4" type="video/mp4">
  Your browser does not support playing the video.
</video>

## A Collaboration for Developers

Last but not least, today we are unveiling a new integration that makes it easier than ever for developers to build AI applications with Hugging Face Spaces and VS Code!

The Hugging Face community has created over 500,000 AI demo applications on the Hub with Hugging Face Spaces. With the new Spaces Dev Mode, Hugging Face users can easily connect their Space to their local VS Code, or spin up a web hosted VS Code environment.

![dev-mode](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/microsoft-collaboration/dev-mode.png) 

Spaces Dev Mode is currently in beta, and available to [PRO subscribers](https://huggingface.co/pricing#pro). To learn more about Spaces Dev Mode, check out [Introducing Spaces Dev mode for a seamless developer experience](hf.co/blog/spaces-dev-mode) or [documentation](https://huggingface.co/dev-mode-explorers).


## What’s Next

We are excited to deepen our strategic collaboration with Microsoft, to make open-source AI more accessible everywhere. Stay tuned as we enable more models in the Azure AI Studio model catalog and introduce new features and experiences in the months to come.
