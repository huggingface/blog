---
title: "Welcome to Inference Providers on the Hub 🔥"
thumbnail: /blog/assets/inference-providers/thumbnail.png
authors:
- user: burkaygur
  guest: true
  org: fal
- user: zeke
  guest: true
  org: replicate
- user: aton2006
  guest: true
  org: sambanovasystems
- user: hassanelmghari
  guest: true
  org: togethercomputer
- user: sbrandeis
- user: kramp
- user: julien-c
---

Today, we are launching the integration of four awesome serverless Inference Providers – **fal, Replicate, Sambanova, Together AI** – directly on the Hub’s model pages. They are also seamlessly integrated into our client SDKs (for JS and Python), making it easier than ever to explore serverless inference of a wide variety of models that run on your favorite providers.

<!-- <insert big visual with logos> -->
 
We’ve been hosting a serverless Inference API on the Hub for a long time (we launched the v1 in summer 2020 – wow, time flies 🤯). While this has enabled easy exploration and prototyping, we’ve since refined our core value proposition towards collaboration, storage, versioning, and distribution of large datasets and models with the community. At the same time, serverless providers have flourished, and the time was right for Hugging Face to offer easy and unified access to serverless inference through a set of great providers. 

Just as we work with great partners like AWS, Nvidia and <insert other partners> for dedicated deployment options via the model pages’ Deploy button, it was natural to partner with the next generation of serverless inference providers for model-centric, serverless inference.

Here’s what this enables, taking the timely example of DeepSeek/DeepSeek-R1, a model which has achieved mainstream fame over the past few days 🔥:

<!-- <insert screenshot or GIF of DeepSeek-R1 model page showcasing fast Inference> -->

Rodrigo Liang, Co-Founder & CEO at [SambaNova](https://huggingface.co/sambanovasystems): "We are excited to be partnering with Hugging Face to accelerate its Inference API. Hugging Face developers now have access to much faster inference speeds on a wide range of the best open source models."

Zeke Sikelianos, Founding Designer at [Replicate](https://huggingface.co/replicate): "Hugging Face is the de facto home of open-source model weights, and has been a key player in making AI more accessible to the world. We use Hugging Face internally at Replicate as our weights registry of choice, and we're honored to be among the first inference providers to be featured in this launch."

**This is just the start, and we’ll build on top of this with the community in the [coming weeks](https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49)!**

## How it works

### In the website UI


in your settings, you are able to:
set your own API keys for the providers you’ve signed up with. Otherwise, you can still use them – your requests will be routed through HF.
order providers by preference. This applies to the widget and code snippets in the model pages. They will use the first provider in your list where the model you’re browsing is “Warm”, or already loaded.



As we mentioned, there are two modes when calling Inference APIs: 
custom key (calls go out to the inference provider directly, using your own API key); or
Routed by HF (in that case, you don't need a token from the provider, and the charges are applied directly to your HF account rather than the provider's account)




model pages showcase third-party inference providers (the ones that are compatible + Warm with the current model, sorted by user preference)




### From the client SDKs

#### from Python, using huggingface_hub

todo

#### from JS using @huggingface/inference

todo
```js
import { HfInference } from "@huggingface/inference";

const client = new HfInference("hf_xxxxxxxxxxxxxxxxxxxxxxxx");

const chatCompletion = await client.chatCompletion({
	model: "deepseek-ai/DeepSeek-R1",
	messages: [
		{
			role: "user",
			content: "What is the capital of France?"
		}
	],
	provider: "together",
	max_tokens: 500
});

console.log(chatCompletion.choices[0].message);


### From HTTP calls


todo

## Feedback and next steps

We would love to get your feedback! Here’s a Hub discussion you can use: https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49


