---
title: "Baseten on Hugging Face Inference Providers 🔥"
thumbnail: /blog/assets/inference-providers/welcome-baseten.png
authors:
  - user: alexker-baseten
    guest: true
    org: baseten
  - user: rolandcrosby-baseten
    guest: true
    org: baseten
  - user: squidarth
    guest: true
    org: baseten
  - user: johan-baseten
    guest: true
    org: baseten
  - user: celinah
  - user: sbrandeis
  - user: Wauplin
  - user: merve
---

![banner image](https://huggingface.co/blog/assets/inference-providers/welcome-baseten.png)

# Baseten on Hugging Face Inference Providers 🔥

We're thrilled to share that **Baseten** is now a supported Inference Provider on the Hugging Face Hub!

Baseten joins our growing ecosystem, enhancing the breadth and capabilities of serverless inference directly on the Hub's model pages. Inference Providers are also seamlessly integrated into our client SDKs (for both JS and Python), making it super easy to use a wide variety of models with your preferred providers.

[Baseten](https://baseten.com) is an AI infrastructure platform that covers serverless AI, training and more. With a catalog of many frontier models, Baseten makes it easy for developers to integrate a wide range of AI capabilities into their applications with minimal setup.

Baseten supports a broad spectrum of model types - from LLMs to text-to-speech and more. As part of this initial integration, Baseten is launching support for **conversational and text-generation tasks** on Hugging Face, enabling access to popular open-weight LLMs such as [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3?inference_provider=baseten), latest [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731?inference_provider=baseten), [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2?inference_provider=baseten), and many more. **Support for additional tasks** will roll out soon!

See the full list of models supported by Baseten [here](https://huggingface.co/models?inference_provider=baseten&sort=trending).

Follow Baseten on Hugging Face: [https://huggingface.co/baseten](https://huggingface.co/baseten).

## How it works

### In the website UI

1. In your user account settings, you are able to:

- Set your own API keys for the providers you've signed up with. If no custom key is set, your requests will be routed through HF.
- Order providers by preference. This applies to the widget and code snippets in the model pages.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/inference-providers-baseten.png" alt="Inference Providers"/>

2. As mentioned, there are two modes when calling Inference Providers:

- Custom key (calls go directly to the inference provider, using your own API key of the corresponding inference provider)
- Routed by HF (in that case, you don't need a token from the provider, and the charges are applied directly to your HF account rather than the provider's account)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/explainer.png" alt="Inference Providers"/>

3. Model pages showcase third-party inference providers (the ones that are compatible with the current model, sorted by user preference)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/model-widget-v3.png" alt="Inference Providers"/>

### From the client SDKs

Baseten is available through the Hugging Face SDKs - `huggingface_hub` (>= 1.26.1) for Python and `@huggingface/inference` for JavaScript.

The following examples show how to use the latest [DeepSeek V4 Flash](deepseek-ai/DeepSeek-V4-Flash-0731) through Baseten. Use a [Hugging Face token](https://huggingface.co/settings/tokens) to authenticate - the request will be routed to Baseten automatically.

#### From your favorite Agent Harness

Hugging Face Inference Providers are integrated in most Agent Harnesses - including Pi, OpenCode, Hermes Agents, OpenClaw, and more. This means you can plug baseten-hosted models straight into your favorite tools without any extra glue code. Browse the full list of integrations [here](https://huggingface.co/docs/inference-providers/en/integrations/index).

#### from Python

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash-0731:baseten",
    messages=[
        {
            "role": "user",
            "content": "Write a Python function that returns the nth Fibonacci number using memoization."
        }
    ],
)

print(completion.choices[0].message)
```

#### from JS

```js
import { OpenAI } from "openai";

const client = new OpenAI({
    baseURL: "https://router.huggingface.co/v1",
    apiKey: process.env.HF_TOKEN,
});

const chatCompletion = await client.chat.completions.create({
    model: "deepseek-ai/DeepSeek-V4-Flash-0731:baseten",
    messages: [
        {
            role: "user",
            content: "Write a Python function that returns the nth Fibonacci number using memoization.",
        },
    ],
});

console.log(chatCompletion.choices[0].message);
```

## Billing

For direct requests, i.e. when you use the key from an inference provider, you are billed by the corresponding provider. For instance, if you use a baseten API key you're billed on your baseten account.

For routed requests, i.e. when you authenticate via the Hugging Face Hub, you'll only pay the standard provider API rates. There's no additional markup from us; we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)

**Important Note** ‼️ PRO users get $2 worth of Inference credits every month. You can use them across providers. 🔥

Subscribe to the [Hugging Face PRO plan](https://hf.co/subscribe/pro) to get access to Inference credits, ZeroGPU, Spaces Dev Mode, 20x higher limits, and more.

We also provide free inference with a small quota for our signed-in free users, but please upgrade to PRO if you can!

## Feedback and next steps

We would love to get your feedback! Share your thoughts and/or comments here: https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49
