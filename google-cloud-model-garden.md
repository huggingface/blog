---
title: "Making thousands of open LLMs bloom in the Vertex AI Model Garden" 
thumbnail: /blog/assets/173_gcp-partnership/thumbnail.jpg
authors:
- user: philschmid
- user: jeffboudier
---


# Making thousands of open LLMs bloom in the Vertex AI Model Garden

Today, we are thrilled to announce the launch of **Deploy on Google Cloud**, a new integration on the Hugging Face Hub to deploy thousands of foundation models easily to Google Cloud using Vertex AI or Google Kubernetes Engine (GKE). Deploy on Google Cloud makes it easy to deploy open models as API Endpoints within your own Google Cloud account, either directly through Hugging Face model cards or within Vertex Model Garden, Google Cloud’s single place to discover, customize, and deploy a wide variety of models from Google and Google partners. Starting today, we are enabling the most popular open models on Hugging Face for inference powered by our production solution, [Text Generation Inference](https://github.com/huggingface/text-generation-inference/). 

With Deploy on Google Cloud, developers can build production-ready Generative AI applications without managing infrastructure and servers, directly within their secure Google Cloud environment.


## A Collaboration for AI Builders

This new experience expands upon the [strategic partnership we announced earlier this year](https://huggingface.co/blog/gcp-partnership) to simplify the access and deployment of open Generative AI models for Google customers. One of the main problems developers and organizations face is the time and resources it takes to deploy models securely and reliably. Deploy on Google Cloud offers an easy, managed solution to these challenges, providing dedicated configurations and assets to Hugging Face Models. It’s a simple click-through experience to create a production-ready Endpoint on Google Cloud’s Vertex AI. 
 

“Vertex AI’s Model Garden integration with the Hugging Face Hub makes it seamless to discover and deploy open models on Vertex AI and GKE, whether you start your journey on the Hub or directly in the Google Cloud Console” says Wenming Ye, Product Manager at Google. “We can’t wait to see what Google Developers build with Hugging Face models”.


## How it works - from the Hub

Deploying Hugging Face Models on Google Cloud is super easy. Below, you will find step-by-step instructions on how to deploy [Zephyr Gemma](https://console.cloud.google.com/vertex-ai/publishers/HuggingFaceH4/model-garden/zephyr-7b-gemma-v0.1;hfSource=true;action=deploy?authuser=1). Starting today, [all models with the “text-generation-inference”](https://huggingface.co/models?pipeline_tag=text-generation-inference&sort=trending) tag will be supported. 



![model-card](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/model-card.png)



Open the “Deploy” menu, and select “Google Cloud”. This will now bring you straight into the Google Cloud Console, where you can deploy Zephyr Gemma in 1 click on Vertex AI, or GKE. 



![vertex-ai-model-garden](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/vertex-ai-model-garden.png)


Once you are in the Vertex Model Garden, you can select Vertex AI or GKE as your deployment environment. With Vertex AI you can deploy the model with 1-click on “Deploy”. For GKE, you can follow instructions and manifest templates on how to deploy the model on a new or running Kubernetes Cluster. 


## How it works - from Vertex Model Garden

Vertex Model Garden is where Google Developers can find ready-to-use models for their Generative AI projects. Starting today, the Vertex Model Garden offers a new experience to easily deploy the most popular open LLMs available on Hugging Face!

You can find the new “Deploy From Hugging Face” option inside Google Vertex AI Model Garden, which allows you to search and deploy Hugging Face models directly within your Google Cloud console. 



![deploy-from-huggingface.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/deploy-from-huggingface.png)


When you click on “Deploy From Hugging Face”, a form will appear where you can quickly search for model IDs. Hundreds of the most popular open LLMs on Hugging Face are available with ready-to-use, tested hardware configurations. 

![model-selection.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/model-selection.png)


Once you find the model you want to deploy, select it, and Vertex AI will prefill all required configurations to deploy your model to Vertex AI or GKE. You can even ensure you selected the right model by “viewing it on Hugging Face.” If you’re using a gated model, make sure to provide your Hugging Face access token so the model download can be authorized. 


![from-deploy.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/google-cloud-model-garden/from-deploy.png)


And that’s it! Deploying a model like Zephyr Gemma directly, from the Vertex Model Garden onto your own Google Cloud account is just a couple of clicks.


## We’re just getting started

We are excited to collaborate with Google Cloud to make AI more open and accessible for everyone. Deploying open models on Google Cloud has never been easier, whether you start from the Hugging Face Hub, or within the Google Cloud console. And we’re not going to stop there – stay tuned as we enable more experiences to build AI with open models on Google Cloud! 