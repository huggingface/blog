---
title: "Featherless AI on Hugging Face Inference Providers 🔥"
thumbnail: /blog/assets/inference-providers/welcome-featherless.jpg
authors:
- user: wxgeorge
  guest: true
  org: featherless-ai
- user: pohnean-recursal
  guest: true
  org: featherless-ai
- user: picocreator
  guest: true
  org: featherless-ai
- user: celinah
- user: Wauplin
- user: sbrandeis
---

![banner image](https://huggingface.co/blog/assets/inference-providers/welcome-featherless.jpg)

# Featherless AI on Hugging Face Inference Providers 🔥

We're thrilled to share that **Featherless AI** is now a supported Inference Provider on the Hugging Face Hub!
Featherless AI joins our growing ecosystem, enhancing the breadth and capabilities of serverless inference directly on the Hub’s model pages. Inference Providers are also seamlessly integrated into our client SDKs (for both JS and Python), making it super easy to use a wide variety of models with your preferred providers.

[Featherless AI](https://featherless.ai) supports a wide variety of text and conversational models, including the latest open-source models from DeepSeek, Meta, Google, Qwen, and much more.

Featherless AI is a serverless AI inference provider with unique model loading and GPU orchestration abilities that makes an exceptionally large catalog of models available for users. Providers often offer either a low cost of access to a limited set of models, or an unlimited range of models with users managing servers and the associated costs of operation. Featherless provides the best of both worlds offering unmatched model range and variety but with serverless pricing. Find the full list of supported models on the [models page](https://huggingface.co/models?inference_provider=featherless-ai&sort=trending).

We're super excited to see what you'll build with this new provider!

Read more about how to use Featherless as an Inference Provider in its dedicated [documentation page](https://huggingface.co/docs/inference-providers/providers/featherless-ai).

 ## How it works

### In the website UI


1. In your user account settings, you are able to:
- Set your own API keys for the providers you’ve signed up with. If no custom key is set, your requests will be routed through HF. Learn more about request types in the [docs](https://huggingface.co/docs/inference-providers/en/pricing#routed-requests-vs-direct-calls).
- Order providers by preference. This applies to the widget and code snippets in the model pages.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/user-settings-updated.png" alt="Inference Providers"/>


2. As [mentioned](https://huggingface.co/docs/inference-providers/en/pricing), there are two modes when calling Inference Providers: 
- Custom key (calls go directly to the inference provider, using your own API key of the corresponding inference provider)
- Routed by HF (in that case, you don't need a token from the provider, and the charges are applied directly to your HF account rather than the provider's account)


<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/explainer.png" alt="Inference Providers"/>


3. Model pages showcase third-party inference providers (the ones that are compatible with the current model, sorted by user preference)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/model-widget-updated.png" alt="Inference Providers"/>


### From the client SDKs

#### from Python, using huggingface_hub

The following example shows how to use DeepSeek-R1 using Featherless AI as the inference provider. You can use a [Hugging Face token](https://huggingface.co/settings/tokens) for automatic routing through Hugging Face, or your own Featherless AI API key if you have one.

Install `huggingface_hub` from source (see [instructions](https://huggingface.co/docs/huggingface_hub/installation#install-from-source)). Official support will be released soon in version v0.33.0.

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="featherless-ai",
    api_key="xxxxxxxxxxxxxxxxxxxxxxxx"
)

messages = [
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528", 
    messages=messages, 
)

print(completion.choices[0].message)
```

#### from JS using @huggingface/inference

```js
import { HfInference } from "@huggingface/inference";

const client = new HfInference("xxxxxxxxxxxxxxxxxxxxxxxx");

const chatCompletion = await client.chatCompletion({
	model: "deepseek-ai/DeepSeek-R1-0528",
	messages: [
		{
			role: "user",
			content: "What is the capital of France?"
		}
	],
	provider: "featherless-ai",
});

console.log(chatCompletion.choices[0].message);
```

## Billing

For direct requests, i.e. when you use the key from an inference provider, you are billed by the corresponding provider. For instance, if you use a Featherless AI API key you're billed on your Featherless AI account.

For routed requests, i.e. when you authenticate via the Hugging Face Hub, you'll only pay the standard provider API rates. There's no additional markup from us, we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)

**Important Note** ‼️ PRO users get $2 worth of Inference credits every month. You can use them across providers. 🔥

Subscribe to the [Hugging Face PRO plan](https://hf.co/subscribe/pro) to get access to Inference credits, ZeroGPU, Spaces Dev Mode, 20x higher limits, and more.

We also provide free inference with a small quota for our signed-in free users, but please upgrade to PRO if you can!

## Feedback and next steps

We would love to get your feedback! Share your thoughts and/or comments here: https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49
