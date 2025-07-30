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

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/hero.png" alt="Inference Providers"/>

We’ve been hosting a serverless Inference API on the Hub for a long time (we launched the v1 in summer 2020 – wow, time flies 🤯). While this has enabled easy exploration and prototyping, we’ve refined our core value proposition towards collaboration, storage, versioning, and distribution of large datasets and models with the community. At the same time, serverless providers have flourished, and the time was right for Hugging Face to offer easy and unified access to serverless inference through a set of great providers. 

Just as we work with great partners like AWS, Nvidia and others for dedicated deployment options via the model pages’ Deploy button, it was natural to partner with the next generation of serverless inference providers for model-centric, serverless inference.

Here’s what this enables, taking the timely example of DeepSeek-ai/DeepSeek-R1, a model which has achieved mainstream fame over the past few days 🔥:

<video controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/inference-providers.mp4" type="video/mp4">
</video>

Rodrigo Liang, Co-Founder & CEO at [SambaNova](https://huggingface.co/sambanovasystems): "We are excited to be partnering with Hugging Face to accelerate its Inference API. Hugging Face developers now have access to much faster inference speeds on a wide range of the best open source models."

Zeke Sikelianos, Founding Designer at [Replicate](https://huggingface.co/replicate): "Hugging Face is the de facto home of open-source model weights, and has been a key player in making AI more accessible to the world. We use Hugging Face internally at Replicate as our weights registry of choice, and we're honored to be among the first inference providers to be featured in this launch."

**This is just the start, and we’ll build on top of this with the community in the [coming weeks](https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49)!**

## How it works

### In the website UI


1. In your user account settings, you are able to:
- set your own API keys for the providers you’ve signed up with. Otherwise, you can still use them – your requests will be routed through HF.
- order providers by preference. This applies to the widget and code snippets in the model pages.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/user-settings.png" alt="Inference Providers"/>


2. As we mentioned, there are two modes when calling Inference APIs: 
- custom key (calls go directly to the inference provider, using your own API key of the corresponding inference provider); or
- Routed by HF (in that case, you don't need a token from the provider, and the charges are applied directly to your HF account rather than the provider's account)


<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/explainer.png" alt="Inference Providers"/>


3. Model pages showcase third-party inference providers (the ones that are compatible with the current model, sorted by user preference)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/model-widget.png" alt="Inference Providers"/>



### From the client SDKs

#### from Python, using huggingface_hub

The following example shows how to use DeepSeek-R1 using Together AI as the inference provider. You can use a [Hugging Face token](https://huggingface.co/settings/tokens) for automatic routing through Hugging Face, or your own Together AI API key if you have one.

Install `huggingface_hub` version v0.28.0 or later ([release notes](https://github.com/huggingface/huggingface_hub/releases/tag/v0.28.0)).

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="together",
    api_key="xxxxxxxxxxxxxxxxxxxxxxxx"
)

messages = [
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1", 
    messages=messages, 
    max_tokens=500
)

print(completion.choices[0].message)
```

**Note:** You can also use the OpenAI client library to call the Inference Providers too; see [here an example for DeepSeek model](https://huggingface.co/deepseek-ai/DeepSeek-R1?inference_provider=together&language=python&inference_api=true).

And here's how to generate an image from a text prompt using [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) running on [fal.ai](https://fal.ai/models/fal-ai/flux/dev):

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="fal-ai",
    api_key="xxxxxxxxxxxxxxxxxxxxxxxx"
)

# output is a PIL.Image object
image = client.text_to_image(
    "Labrador in the style of Vermeer",
    model="black-forest-labs/FLUX.1-dev"
)
```

To move to a different provider, you can simply change the provider name, everything else stays the same:

```diff
from huggingface_hub import InferenceClient

client = InferenceClient(
-	provider="fal-ai",
+	provider="replicate",
	api_key="xxxxxxxxxxxxxxxxxxxxxxxx"
)
```

#### from JS using @huggingface/inference

```js
import { HfInference } from "@huggingface/inference";

const client = new HfInference("xxxxxxxxxxxxxxxxxxxxxxxx");

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
```

### From HTTP calls

We expose the Routing proxy directly under the huggingface.co domain so you can call it directly, it's very useful for OpenAI-compatible APIs for instance. You can just swap the URL as a base URL: `https://router.huggingface.co/{:provider}`.

Here's how you can call Llama-3.3-70B-Instruct using Sambanova as the inference provider via cURL.

```bash
curl 'https://router.huggingface.co/sambanova/v1/chat/completions' \
-H 'Authorization: Bearer xxxxxxxxxxxxxxxxxxxxxxxx' \
-H 'Content-Type: application/json' \
--data '{
    "model": "Llama-3.3-70B-Instruct",
    "messages": [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	],
    "max_tokens": 500,
    "stream": false
}'
```

## Billing

For direct requests, i.e. when you use the key from an inference provider, you are billed by the corresponding provider. For instance, if you use a Together AI key you're billed on your Together AI account.

For routed requests, i.e. when you authenticate via the hub, you'll only pay the standard provider API rates. There's no additional markup from us, we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)

**Important Note** ‼️ PRO users get $2 worth of Inference credits every month. You can use them across providers. 🔥

Subscribe to the [Hugging Face PRO plan](https://hf.co/subscribe/pro) to get access to Inference credits, ZeroGPU, Spaces Dev Mode, 20x higher limits, and more.

We also provide free inference with a small quota for our signed-in free users, but please upgrade to PRO if you can!

## Feedback and next steps

We would love to get your feedback! Here’s a Hub discussion you can use: https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49


