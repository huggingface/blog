---
title: 'Introducing Three New Serverless Inference Providers: Hyperbolic, Nebius AI
  Studio, and Novita 🔥'
thumbnail: /blog/assets/inference-providers/second-batch-thumbnail.webp
authors:
- user: julien-c
- user: kramp
- user: reach-vb
- user: sbrandeis
- user: albertworks
  guest: true
  org: nebius
- user: viktor-hu
  guest: true
  org: novita
- user: cchevli
  guest: true
  org: Hyperbolic
---

We’re thrilled to announce the addition of three more outstanding serverless Inference Providers to the Hugging Face Hub: [Hyperbolic](https://hyperbolic.xyz/), [Nebius AI Studio](https://studio.nebius.com/), and [Novita](https://novita.ai/). These providers join our growing ecosystem, enhancing the breadth and capabilities of serverless inference directly on the Hub’s model pages. They’re also seamlessly integrated into our client SDKs (for both JS and Python), making it super easy to use a wide variety of models with your preferred providers.

These partners join the ranks of our existing providers, including Together AI, Sambanova, Replicate, fal and Fireworks.ai.

The new partners enable a swath of new models: DeepSeek-R1, Flux.1, and many others. Find all the models supported by them below:

- [Nebius AI Studio](https://huggingface.co/models?inference_provider=nebius&sort=trending)
- [Novita](https://huggingface.co/models?inference_provider=novita&sort=trending)
- [Hyperbolic](https://huggingface.co/models?inference_provider=hyperbolic&sort=trending)

We're quite excited to see what you'll build with these new providers!

## How it works

### In the website UI


1. In your user account settings, you are able to:
- Set your own API keys for the providers you’ve signed up with. If no custom key is set, your requests will be routed through HF.
- Order providers by preference. This applies to the widget and code snippets in the model pages.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/user-settings-updated.png" alt="Inference Providers"/>


2. As mentioned, there are two modes when calling Inference APIs: 
- Custom key (calls go directly to the inference provider, using your own API key of the corresponding inference provider)
- Routed by HF (in that case, you don't need a token from the provider, and the charges are applied directly to your HF account rather than the provider's account)


<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/explainer.png" alt="Inference Providers"/>


3. Model pages showcase third-party inference providers (the ones that are compatible with the current model, sorted by user preference)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/model-widget-updated.png" alt="Inference Providers"/>



### From the client SDKs

#### from Python, using huggingface_hub

The following example shows how to use DeepSeek-R1 using Hyperbolic as the inference provider. You can use a [Hugging Face token](https://huggingface.co/settings/tokens) for automatic routing through Hugging Face, or your own Hyperbolic API key if you have one.

Install `huggingface_hub` from source (see [instructions](https://huggingface.co/docs/huggingface_hub/installation#install-from-source)). Official support will be released soon in version v0.29.0.

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hyperbolic",
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


And here's how to generate an image from a text prompt using [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) running on [Nebius AI Studio](https://studio.nebius.com):

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="nebius",
    api_key="xxxxxxxxxxxxxxxxxxxxxxxx"
)

# output is a PIL.Image object
image = client.text_to_image(
    "Bob Marley in the style of a painting by Johannes Vermeer",
    model="black-forest-labs/FLUX.1-schnell"
)
```

To move to a different provider, you can simply change the provider name, everything else stays the same:

```diff
from huggingface_hub import InferenceClient

client = InferenceClient(
-	provider="nebius",
+   provider="hyperbolic",
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
	provider: "novita",
	max_tokens: 500
});

console.log(chatCompletion.choices[0].message);
```

## Billing

For direct requests, i.e. when you use the key from an inference provider, you are billed by the corresponding provider. For instance, if you use a Nebius AI Studio key you're billed on your Nebius AI Studio account.

For routed requests, i.e. when you authenticate via the hub, you'll only pay the standard provider API rates. There's no additional markup from us, we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)

**Important Note** ‼️ PRO users get $2 worth of Inference credits every month. You can use them across providers. 🔥

Subscribe to the [Hugging Face PRO plan](https://hf.co/subscribe/pro) to get access to Inference credits, ZeroGPU, Spaces Dev Mode, 20x higher limits, and more.

We also provide free inference with a small quota for our signed-in free users, but please upgrade to PRO if you can!

## Feedback and next steps

We would love to get your feedback! Here’s a Hub discussion you can use: https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49


