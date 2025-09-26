---
title: "Public AI on Hugging Face Inference Providers 🔥"
thumbnail: /blog/assets/inference-providers/welcome-publicai.jpg
authors:
  - user: Jolow
    guest: true
    org: publicai
  - user: thelastjosh
    guest: true
    org: publicai
  - user: celinah
  - user: julien-c
  - user: sbrandeis
  - user: Wauplin
---

![banner image](https://huggingface.co/blog/assets/inference-providers/welcome-publicai.jpg)

# Public AI on Hugging Face Inference Providers 🔥

We're thrilled to share that **Public AI** is now a supported Inference Provider on the Hugging Face Hub!
Public AI joins our growing ecosystem, enhancing the breadth and capabilities of serverless inference directly on the Hub’s model pages. Inference Providers are also seamlessly integrated into our client SDKs (for both JS and Python), making it super easy to use a wide variety of models with your preferred providers.

This launch makes it easier than ever to access public and sovereign models from institutions like the Swiss AI Initiative and AI Singapore — right from Hugging Face. You can browse Public AI’s org on the Hub at https://huggingface.co/publicai and try trending supported models at https://huggingface.co/models?inference_provider=publicai&sort=trending.

The Public AI Inference Utility is a nonprofit, open-source project. The team builds products and organizes advocacy to support the work of public AI model builders like the Swiss AI Initiative and AI Singapore, among others.

The Public AI Inference Utility runs on a distributed infrastructure that combines a vLLM-powered backend with a deployment layer designed for resilience across multiple partners. Behind the scenes, inference is handled by servers exposing OpenAI-compatible APIs on vLLM, deployed across clusters donated by national and industry partners. A global load-balancing layer ensures requests are routed efficiently and transparently, regardless of which country’s compute is serving the query.

Free public access is supported by donated GPU time and advertising subsidies, while long-term stability is intended to be anchored by state and institutional contributions. You can learn more about Public AI’s platform and infrastructure at https://platform.publicai.co/.

You can now use the Public AI Inference Utility as an Inference Provider on Hugging Face. We're excited to see what you'll build with this new provider.

Read more about how to use Public AI as an Inference Provider in its dedicated [documentation page](https://huggingface.co/docs/inference-providers/providers/publicai).

See the list of supported models [here](https://huggingface.co/models?inference_provider=publicai&sort=trending).

## How it works

### In the website UI

1. In your user account settings, you are able to:

- Set your own API keys for the providers you’ve signed up with. If no custom key is set, your requests will be routed through HF.
- Order providers by preference. This applies to the widget and code snippets in the model pages.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/user-settings-updated.png" alt="Inference Providers"/>

2. As mentioned, there are two modes when calling Inference Providers:

- Custom key (calls go directly to the inference provider, using your own API key of the corresponding inference provider)
- Routed by HF (in that case, you don't need a token from the provider, and the charges are applied directly to your HF account rather than the provider's account)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/explainer.png" alt="Inference Providers"/>

3. Model pages showcase third-party inference providers (the ones that are compatible with the current model, sorted by user preference)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/model-widget-updated.png" alt="Inference Providers"/>

### From the client SDKs

#### from Python, using huggingface_hub

The following example shows how to use Swiss AI's Apertus-70B using Public AI as the inference provider. You can use a [Hugging Face token](https://huggingface.co/settings/tokens) for automatic routing through Hugging Face, or your own Public AI API key if you have one.

Note: this requires using a recent version of `huggingface_hub` (>= 0.34.6).

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="publicai",
    api_key=os.environ["HF_TOKEN"],
)

messages = [
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]

completion = client.chat.completions.create(
    model="swiss-ai/Apertus-70B-Instruct-2509",
    messages=messages,
)

print(completion.choices[0].message)
```

#### from JS using @huggingface/inference

```js
import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(process.env.HF_TOKEN);

const chatCompletion = await client.chatCompletion({
  model: "swiss-ai/Apertus-70B-Instruct-2509",
  messages: [
    {
      role: "user",
      content: "What is the capital of France?",
    },
  ],
  provider: "publicai",
});

console.log(chatCompletion.choices[0].message);
```

## Billing


At the time of writing, usage of the Public AI Inference Utility through Hugging Face Inference Providers is free of charge. Pricing and availability may change.

Here is how billing works for other providers on the platform:

For direct requests, i.e. when you use the key from an inference provider, you are billed by the corresponding provider. For instance, if you use a Public AI API key you're billed on your Public AI account.

For routed requests, i.e. when you authenticate via the Hugging Face Hub, you'll only pay the standard provider API rates. There's no additional markup from us; we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)

**Important Note** ‼️ PRO users get $2 worth of Inference credits every month. You can use them across providers. 🔥

Subscribe to the [Hugging Face PRO plan](https://hf.co/subscribe/pro) to get access to Inference credits, ZeroGPU, Spaces Dev Mode, 20x higher limits, and more.

We also provide free inference with a small quota for our signed-in free users, but please upgrade to PRO if you can!


## Feedback and next steps

We would love to get your feedback! Share your thoughts and/or comments here: https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49
