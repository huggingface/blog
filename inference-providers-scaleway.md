---
title: "Scaleway on Hugging Face Inference Providers 🔥"
thumbnail: /blog/assets/inference-providers/welcome-scaleway.jpg
authors:
  - 
  - user: celinah
  - user: julien-c
  - user: sbrandeis
  - user: Wauplin
---

![banner image](https://huggingface.co/blog/assets/inference-providers/welcome-scaleway.jpg)

# Scaleway on Hugging Face Inference Providers 🔥

We're thrilled to share that **Scaleway** is now a supported Inference Provider on the Hugging Face Hub!
Scaleway joins our growing ecosystem, enhancing the breadth and capabilities of serverless inference directly on the Hub’s model pages. Inference Providers are also seamlessly integrated into our client SDKs (for both JS and Python), making it super easy to use a wide variety of models with your preferred providers.

This launch makes it easier than ever to access popular open-weight models like gpt-oss, Qwen3, DeepSeek R1, and Gemma 3 — right from Hugging Face. You can browse Scaleway's org on the Hub at https://huggingface.co/scaleway and try trending supported models at https://huggingface.co/models?inference_provider=scaleway&sort=trending.

_Scaleway Generative APIs_ is a fully managed, serverless service that provides access to frontier AI models from leading research labs via simple API calls. The service offers competitive pay-per-token pricing starting at €0.20 per million tokens.

The service runs on secure infrastructure located in European data centers (Paris, France), ensuring data sovereignty and low latency for European users. The platform supports advanced features including structured outputs, function calling, and multimodal capabilities for both text and image processing.

Built for production use, Scaleway's inference infrastructure delivers sub-200ms response times for first tokens, making it ideal for interactive applications and agentic workflows. The service supports both text generation and embedding models. You can learn more about Scaleway's platform and infrastructure at https://www.scaleway.com/en/generative-apis/.


Read more about how to use Scaleway as an Inference Provider in its dedicated [documentation page](https://huggingface.co/docs/inference-providers/providers/scaleway).

See the list of supported models [here](https://huggingface.co/models?inference_provider=scaleway&sort=trending).

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

The following example shows how to use Swiss AI's Apertus-70B using Scaleway as the inference provider. You can use a [Hugging Face token](https://huggingface.co/settings/tokens) for automatic routing through Hugging Face, or your own Scaleway API key if you have one.

Note: this requires using a recent version of `huggingface_hub` (>= 0.34.6).

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="scaleway",
    api_key=os.environ["HF_TOKEN"],
)

messages = [
    {
        "role": "user",
        "content": "Write a poem in the style of Shakespeare"
    }
]

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
)

print(completion.choices[0].message)
```

#### from JS using @huggingface/inference

```js
import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(process.env.HF_TOKEN);

const chatCompletion = await client.chatCompletion({
  model: "openai/gpt-oss-120b",
  messages: [
    {
      role: "user",
      content: "Write a poem in the style of Shakespeare",
    },
  ],
  provider: "scaleway",
});

console.log(chatCompletion.choices[0].message);
```

## Billing

Here is how billing works:

For direct requests, i.e. when you use the key from an inference provider, you are billed by the corresponding provider. For instance, if you use a Scaleway API key you're billed on your Scaleway account.

For routed requests, i.e. when you authenticate via the Hugging Face Hub, you'll only pay the standard provider API rates. There's no additional markup from us; we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)

**Important Note** ‼️ PRO users get $2 worth of Inference credits every month. You can use them across providers. 🔥

Subscribe to the [Hugging Face PRO plan](https://hf.co/subscribe/pro) to get access to Inference credits, ZeroGPU, Spaces Dev Mode, 20x higher limits, and more.

We also provide free inference with a small quota for our signed-in free users, but please upgrade to PRO if you can!


## Feedback and next steps

We would love to get your feedback! Share your thoughts and/or comments here: https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/49
