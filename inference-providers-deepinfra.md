---
title: "DeepInfra on Hugging Face Inference Providers 🔥"
thumbnail: /blog/assets/inference-providers/welcome-deepinfra.jpg
authors:
  - user: Thachnh
    guest: true
    org: deepinfra
  - user: araikin
    guest: true
    org: deepinfra
  - user: shang-pin-deepinfra
    guest: true
    org: deepinfra
  - user: Pernekhan
    guest: true
    org: deepinfra
  - user: yessenzhar
    guest: true
    org: deepinfra
  - user: ovuruska
    guest: true
    org: deepinfra
  - user: celinah
  - user: sbrandeis
  - user: Wauplin
---

![banner image](https://huggingface.co/blog/assets/inference-providers/welcome-deepinfra.jpg)

# DeepInfra on Hugging Face Inference Providers 🔥

We're thrilled to share that **DeepInfra** is now a supported Inference Provider on the Hugging Face Hub!

DeepInfra joins our growing ecosystem, enhancing the breadth and capabilities of serverless inference directly on the Hub's model pages. Inference Providers are also seamlessly integrated into our client SDKs (for both JS and Python), making it super easy to use a wide variety of models with your preferred providers.

[DeepInfra](https://deepinfra.com) is a serverless AI inference platform offering one of the most cost-effective pricing per token in the industry. With a catalog of over 100 models, DeepInfra makes it easy for developers to integrate a wide range of AI capabilities into their applications with minimal setup.

DeepInfra supports a broad spectrum of model types - from LLMs to text-to-image, text-to-video, embeddings, and more. As part of this initial integration, DeepInfra is launching support for **conversational and text-generation tasks** on Hugging Face, enabling access to popular open-weight LLMs such as [DeepSeek V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro?inference_provider=deepinfra), [Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6?inference_provider=deepinfra), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1?inference_provider=deepinfra), and many more. **Support for additional tasks** (text-to-image, text-to-video, embeddings, and more) will roll out soon!

Read more about how to use DeepInfra as an Inference Provider in its dedicated [documentation page](https://huggingface.co/docs/inference-providers/providers/deepinfra).

See the full list of models supported by DeepInfra [here](https://huggingface.co/models?inference_provider=deepinfra&sort=trending).

Follow DeepInfra on Hugging Face: [https://huggingface.co/DeepInfra](https://huggingface.co/DeepInfra).

## How it works

### In the website UI

1. In your user account settings, you are able to:

- Set your own API keys for the providers you've signed up with. If no custom key is set, your requests will be routed through HF.
- Order providers by preference. This applies to the widget and code snippets in the model pages.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/user-setting-v2.png" alt="Inference Providers"/>

2. As mentioned, there are two modes when calling Inference Providers:

- Custom key (calls go directly to the inference provider, using your own API key of the corresponding inference provider)
- Routed by HF (in that case, you don't need a token from the provider, and the charges are applied directly to your HF account rather than the provider's account)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/explainer.png" alt="Inference Providers"/>

3. Model pages showcase third-party inference providers (the ones that are compatible with the current model, sorted by user preference)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/model-widget-v2.png" alt="Inference Providers"/>

### From the client SDKs

DeepInfra is available through the Hugging Face SDKs - `huggingface_hub` (>= 1.11.2) for Python and `@huggingface/inference` for JavaScript.

The following examples show how to use [DeepSeek V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) through DeepInfra. Use a [Hugging Face token](https://huggingface.co/settings/tokens) to authenticate - the request will be routed to DeepInfra automatically.

#### From your favorite Agent Harness

Hugging Face Inference Providers are integrated in most Agent Harnesses - including Pi, OpenCode, Hermes Agents, OpenClaw, and more. This means you can plug DeepInfra-hosted models straight into your favorite tools without any extra glue code. Browse the full list of integrations [here](https://huggingface.co/docs/inference-providers/en/integrations/index).

#### from Python

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Pro:deepinfra",
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
    model: "deepseek-ai/DeepSeek-V4-Pro:deepinfra",
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

For direct requests, i.e. when you use the key from an inference provider, you are billed by the corresponding provider. For instance, if you use a DeepInfra API key you're billed on your DeepInfra account.

For routed requests, i.e. when you authenticate via the Hugging Face Hub, you'll only pay the standard provider API rates. There's no additional markup from us; we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)

**Important Note** ‼️ PRO users get $2 worth of Inference credits every month. You can use them across providers. 🔥

Subscribe to the [Hugging Face PRO plan](https://hf.co/subscribe/pro) to get access to Inference credits, ZeroGPU, Spaces Dev Mode, 20x higher limits, and more.

We also provide free inference with a small quota for our signed-in free users, but please upgrade to PRO if you can!

## Feedback and next steps

We would love to get your feedback! Share your thoughts and/or comments here: https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49
