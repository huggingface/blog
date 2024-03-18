---
title: "Easily Train Models with H100 GPUs on NVIDIA DGX Cloud" 
thumbnail: /blog/assets/train-dgx-cloud/thumbnail.jpg
authors:
- user: philschmid
- user: jeffboudier
- user: rafaelpierrehf
- user: abhishek
---


# Easily Train Models with H100 GPUs on NVIDIA DGX Cloud

Today, we are thrilled to announce the launch of **Train on DGX Cloud**, a new service on the Hugging Face Hub, available to Enterprise Hub organizations. Train on DGX Cloud makes it easy to use open models with the accelerated compute infrastructure of NVIDIA DGX Cloud. Together, we built Train on DGX Cloud so that Enterprise Hub users can easily access the latest NVIDIA H100 GPUs, to fine-tune popular Generative AI models like Llama, Mistral, and Stable Diffusion, in just a few clicks within the [Hugging Face Hub](https://huggingface.co/models). 

<div align="center"> 
  <img src="/blog/assets/train-dgx-cloud/thumbnail.jpg" alt="Thumbnail"> 
</div>



## GPU Poor No More

This new experience expands upon the [strategic partnership we announced last year](https://nvidianews.nvidia.com/news/nvidia-and-hugging-face-to-connect-millions-of-developers-to-generative-ai-supercomputing) to simplify the training and deployment of open Generative AI models on NVIDIA GPUs. One of the main problems developers and organizations face is the scarcity of GPU availability, and the time-consuming work of writing, testing, and debugging training scripts for AI models. Train with DGX Cloud offers an easy solution to these challenges, providing instant access to NVIDIA GPUs, starting with H100, managed by NVIDIA DGX Cloud.  In addition, Train with DGX Cloud offers a simple no-code training job creation experience powered by Hugging Face AutoTrain and Hugging Face Spaces. 

[Enterprise Hub](https://huggingface.co/enterprise) organizations can give their teams instant access to powerful NVIDIA GPUs, only incurring charges per minute of compute instances used for their training jobs.

_“Train on DGX Cloud is the easiest, fastest, most accessible way to train Generative AI models, combining instant access to powerful GPUs, pay-as-you-go, and no-code training,”_ says Abhishek Thakur, creator of Hugging Face AutoTrain. _“It will be a game changer for data scientists everywhere!”_

_"Today’s launch of Hugging Face Autotrain powered by DGX Cloud represents a noteworthy step toward simplifying AI model training,”_ said Alexis Bjorlin, vice president of DGX Cloud, NVIDIA. _“By integrating NVIDIA’s AI supercomputer in the cloud with Hugging Face’s user-friendly interface, we’re empowering organizations to accelerate their AI innovation."_


## How it works

Training Hugging Face models on NVIDIA DGX Cloud has never been easier. Below you will find a step-by-step tutorial to fine-tune Mistral 7B. 

_Note: You need access to an Organization with a [Hugging Face Enterprise](https://huggingface.co/enterprise) subscription to use Train on DGX Cloud_

You can find Train on DGX Cloud on the model page of supported Generative AI models. It currently supports the following model architectures:  Llama, Falcon, Mistral, Mixtral, T5, Gemma, Stable Diffusion and Stable Diffusion XL. 


<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/01%20model%20card.png" alt="Model Card"> 
</div>


Open the “Train” menu, and select “NVIDIA DGX Cloud” - this will open an interface where you can select your Enterprise Organization. 


<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/02%20select%20organization.png" alt="Organization Selection"> 
</div>


Then, click on “Create new Space”. When using Train on DGX Cloud for the first time, the service will create a new Hugging Face Space within your Organization, so you can use AutoTrain to create training jobs that will be executed on the NVIDIA DGX Cloud backend. When you want to create another training job later, you will automatically be redirected back to the existing AutoTrain Space. 

Once in the AutoTrain Space, you can create your training job by configuring the Hardware, Base Model, Task, and Training Parameters. 

<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/03%20start.png" alt="Create AutoTrain Job"> 
</div>



For Hardware, you can select NVIDIA H100 GPUs, available in 1x, 2x, 4x and 8x instances, or L40S GPUs (coming soon). The training dataset must be directly uploaded in the “Upload Training File(s)” area. CSV and JSON files are currently supported. Make sure that the column mapping is correct following the example below. For Training Parameters, you can directly edit the JSON configuration on the right side, e.g., changing the number of epochs from 3 to 2. 

When everything is set up, you can start your training by clicking “Start Training”. AutoTrain will now validate your dataset, and ask you to confirm the training. 


<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/04%20success.png" alt="Launched Training Job"> 
</div>




You can monitor your training by opening the “logs” of the Space. 



<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/autotrain-dgx-cloud/05%20logs.png" alt="Training Logs"> 
</div>



After your training is complete, your fine-tuned model will be uploaded to a new private repository within your selected namespace on the Hugging Face Hub.

Train on DGX Cloud is available today for all Enterprise Hub Organizations! Give the service a try, and let us know your feedback!

If you want to see a live demo of Train on DGX Cloud and ask questions to [Abhishek](https://huggingface.co/abhishek) and [Rafael](https://huggingface.co/rafaelpierrehf), [don’t miss the Hugging Cast on Thursday, 3/21](https://streamyard.com/watch/YfEj26jJJg2w) at 8 am PT / 11 am ET / 17h CET.


## Pricing for Train on DGX Cloud

Usage of Train on DGX Cloud is billed by the minute of the GPU instances used during your training jobs. Current prices for training jobs are $8.25 per GPU hour for H100 instances, and $2.75 per GPU hour for L40S instances. Usage fees accrue to your Enterprise Hub Organizations’ current monthly billing cycle, once a job is completed. You can check your current and past usage at any time within the billing settings of your Enterprise Hub Organization. 


<table>
  <tr>
   <td>NVIDIA GPU
   </td>
   <td>GPU Memory
   </td>
   <td>On-Demand Price/hr	
   </td>
  </tr>
  <tr>
   <td><a href="https://www.nvidia.com/en-us/data-center/l40/">NVIDIA L40S</a>
   </td>
   <td>48GB
   </td>
   <td>$2.75
   </td>
  </tr>
  <tr>
   <td><a href="https://www.nvidia.com/de-de/data-center/h100/">NVIDIA H100</a>
   </td>
   <td>80 GB	
   </td>
   <td>$8.25
   </td>
  </tr>
</table>


For example, fine-tuning Mistral 7B on 1500 samples on a single NVIDIA L40S takes ~10 minutes and costs ~$0.45. 


## We’re just getting started

We are excited to collaborate with NVIDIA to democratize accelerated machine learning across open science, open source, and cloud services.

Our collaboration on open science through [BigCode](https://huggingface.co/bigcode) enabled the training of [StarCoder 2 15B](https://huggingface.co/bigcode/starcoder2-15b), a fully open, state-of-the-art code LLM trained on more than 600 languages.

Our collaboration on open source is fueling the new [optimum-nvidia library](https://github.com/huggingface/optimum-nvidia), accelerating the inference of LLMs on the latest NVIDIA GPUs and already achieving 1,200 tokens per second with Llama 2.

Our collaboration on cloud services created Train on DGX Cloud today. We are also working with NVIDIA to optimize inference and make accelerated computing more accessible to the Hugging Face community, leveraging our collaboration on [NVIDIA TensorRT-LLM](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/) and [optimum-nvidia](https://github.com/huggingface/optimum-nvidia). In addition, some of the most popular open models on Hugging Face will be on [NVIDIA NIM microservices](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/), which was announced today at GTC.