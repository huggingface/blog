---
title: "Cohere on Hugging Face Inference Providers 🔥"
thumbnail: /blog/assets/inference-providers-cohere/thumbnail.png
authors:
- user: reach-vb
- user: burtenshaw
- user: merve
- user: celinah
- user: alexrs
  guest: true
  org: CohereLabs
- user: julien-c
- user: sbrandeis
---

![banner image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/cohere-banner.png)
<!-- TODO: this banner is by @burtenshaw and a bit shit canva link is https://www.canva.com/design/DAGkt1fdFb0/m13S0f1UuyJOHs-wC-LQSA/edit?utm_content=DAGkt1fdFb0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton cohere assets are at https://cohere.com/newsroom -->

# Cohere on Hugging Face Inference Providers 🔥

We're thrilled to share that **Cohere** is now a supported Inference Provider on HF Hub! This also marks the first model creator to share and serve their models directly on the Hub. 

[Cohere](https://cohere.com) is committed to building and serving models purpose-built for enterprise use-cases. Their comprehensive suite of secure AI solutions, from cutting-edge Generative AI to powerful Embeddings and Ranking models, are designed to tackle real-world business challenges. Additionally, [Cohere Labs](https://cohere.com/research), Cohere’s in house research lab, supports fundamental research and seeks to change the spaces where research happens.

Starting now, you can run serverless inference to the following models via Cohere and Inference Providers:

- [CohereLabs/c4ai-command-r-v01](https://huggingface.co/CohereLabs/c4ai-command-r-v01)  
- [CohereLabs/c4ai-command-r-plus](https://huggingface.co/CohereLabs/c4ai-command-r-plus)  
- [CohereLabs/c4ai-command-r-08-2024](https://huggingface.co/CohereLabs/c4ai-command-r-08-2024)  
- [CohereLabs/c4ai-command-r7b-12-2024](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024)  
- [CohereLabs/c4ai-command-a-03-2025](https://huggingface.co/CohereLabs/c4ai-command-a-03-2025)  
- [CohereLabs/aya-expanse-8b](https://huggingface.co/CohereLabs/aya-expanse-8b)  
- [CohereLabs/aya-expanse-32b](https://huggingface.co/CohereLabs/aya-expanse-32b)  
- [CohereLabs/aya-vision-8b](https://huggingface.co/CohereLabs/aya-vision-8b)  
- [CohereLabs/aya-vision-32b](https://huggingface.co/CohereLabs/aya-vision-32b)

Light up your projects with Cohere and Cohere Labs today!

## Cohere Models

Cohere and Cohere Labs bring a swathe of their models to Inference Providers that excel at specific business applications. Let’s explore some in detail.

### CohereLabs/c4ai-command-a-03-2025 [🔗](https://huggingface.co/CohereLabs/c4ai-command-a-03-2025)

Optimized for demanding enterprises that require fast, secure, and high-quality AI. Its 256k context length (2x most leading models) can handle much longer enterprise documents. Other key features include Cohere’s advanced retrieval-augmented generation (RAG) with verifiable citations, agentic tool use, enterprise-grade security, and strong multilingual performance (support for the 23 languages).

### CohereLabs/aya-expanse-32b [🔗](https://huggingface.co/CohereLabs/aya-expanse-32b)

Focuses on state-of-art multilingual support beyond in lesser resource languages. Supports Arabic, Chinese (simplified & traditional), Czech, Dutch, English, French, German, Greek, Hebrew, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, and Vietnamese with 128K context length.

### CohereLabs/c4ai-command-r7b-12-2024 [🔗](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024)

Ideal for low-cost or low-latency use cases, bringing state-of-the-art performance in its class of open-weight models across real-world tasks. This model offers a context length of 128k. It delivers a powerful combination of multilingual support, citation-verified retrieval-augmented generation (RAG), reasoning, tool use, and agentic behavior. multilingual model trained on 23 languages

### [`CohereLabs/aya-vision-32b`](https://huggingface.co/CohereLabs/aya-vision-32b)

32-billion parameter model with advanced capabilities optimized for a variety of vision-language use cases, including OCR, captioning, visual reasoning, summarization, question answering, code, and more. It expands multimodal capabilities to 23 languages spoken by over half the world's population.

## How it works

You can use Cohere models directly on the Hub either on the website UI or via the client SDKs.

> [!TIP]
> You can find all the examples mentioned in this section on the [Cohere documentation page](https://huggingface.co/docs/inference-providers/providers/cohere). 

### In the website UI

You can search for Cohere models by filtering by the inference provider in the [model hub](https://huggingface.co/models?inference_provider=cohere).

![Cohere provider UI](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/models-inference-provider-cohere.png)

From the Model Card, you can select the inference provider and run inference directly in the UI. 

![gif screenshot of Cohere inference provider in the UI](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers/cohere-screenshot.gif)

### From the client SDKs

Let’s walk through using Cohere models from client SDKs. We’ve also made a [colab notebook](https://colab.research.google.com/drive/1xxpiB9on_DhZdCpro-f3T07Pd-BfIssz?usp=sharing) with these snippets, in case you want to try them out right away.

#### from Python, using huggingface_hub

The following example shows how to use Command A using Cohere as your inference provider. You can use a [Hugging Face token](https://huggingface.co/settings/tokens) for automatic routing through Hugging Face, or your own cohere API key if you have one.

Install `huggingface_hub` v0.30.0 or later: 

```sh
pip install -U "huggingface_hub>=0.30.0"
```

Use the `huggingface_hub` python library to call Cohere endpoints by defining the `provider` parameter.

```py
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="cohere",
    api_key="xxxxxxxxxxxxxxxxxxxxxxxx",
)

messages = [
        {
            "role": "user",
            "content": "How to make extremely spicy Mayonnaise?"
        }
]

completion = client.chat.completions.create(
    model="CohereLabs/c4ai-command-r7b-12-2024",
    messages=messages,
    temperature=0.7,
    max_tokens=512,
)

print(completion.choices[0].message)
```

Aya Vision, Cohere Labs’ multilingual, multimodal model is also supported. You can include images encoded in base64 as follows: 

```py
image_path = "img.jpg"
with open(image_path, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")
image_url = f"data:image/jpeg;base64,{base64_image}"

from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="cohere",
    api_key="xxxxxxxxxxxxxxxxxxxxxxxx",
)

messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        }
]

completion = client.chat.completions.create(
    model="CohereLabs/aya-vision-32b",
    messages=messages,
    temperature=0.7,
    max_tokens=512,
)

print(completion.choices[0].message)

```

#### from JS using @huggingface/inference

```javascript
import { HfInference } from "@huggingface/inference";

const client = new HfInference("xxxxxxxxxxxxxxxxxxxxxxxx");

const chatCompletion = await client.chatCompletion({
    model: "CohereLabs/c4ai-command-a-03-2025",
    messages: [
        {
            role: "user",
            content: "How to make extremely spicy Mayonnaise?"
        }
    ],
    provider: "cohere",
    max_tokens: 512
});

console.log(chatCompletion.choices[0].message);
```

### From OpenAI client

Here's how you can call Command R7B using Cohere as the inference provider via the OpenAI client library.

```py
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/cohere/compatibility/v1",
    api_key="xxxxxxxxxxxxxxxxxxxxxxxx",
)

messages = [
        {
            "role": "user",
            "content": "How to make extremely spicy Mayonnaise?"
        }
]

completion = client.chat.completions.create(
    model="command-a-03-2025",
    messages=messages,
    temperature=0.7,
)

print(completion.choices[0].message)
```

## Tool Use with Cohere Models

Cohere’s models bring state-of-the-art agentic tool use to Inference Providers so let’s explore that in detail. Both the Hugging Face Hub client and the OpenAI client are compatible with tools via inference providers, so the above examples can be expanded.

First, we will need to define tools for the model to use. Below we define the `get_flight_info` which calls an API for the latest flight information using two locations. This tool definition will be represented by the model’s chat template. Which we can also explore in the [model card](https://huggingface.co/CohereLabs/c4ai-command-a-03-2025?chat_template=default) (🎉 open source).

```py
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_info",
            "description": "Get flight information between two cities or airports",
            "parameters": {
                "type": "object",
                "properties": {
                    "loc_origin": {
                        "type": "string",
                        "description": "The departure airport, e.g. MIA",
                    },
                    "loc_destination": {
                        "type": "string",
                        "description": "The destination airport, e.g. NYC",
                    },
                },
                "required": ["loc_origin", "loc_destination"],
            },
        },
    }
]

```

Next, we’ll need to pass messages to the inference client for the model to use the tools when relevant. In the example below we define the assistant’s tool call in `tool_calls,` for the sake of clarity.

```py

messages = [
    {"role": "developer", "content": "Today is April 30th"},
    {
        "role": "user",
        "content": "When is the next flight from Miami to Seattle?",
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "arguments": '{ "loc_destination": "Seattle", "loc_origin": "Miami" }',
                    "name": "get_flight_info",
                },
                "id": "get_flight_info0",
                "type": "function",
            }
        ],
    },
    {
        "role": "tool",
        "name": "get_flight_info",
        "tool_call_id": "get_flight_info0",
        "content": "Miami to Seattle, May 1st, 10 AM.",
    },
]
```

Finally, the tools and messages are passed to the create method.

```py
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="cohere",
    api_key="xxxxxxxxxxxxxxxxxxxxxxxx",
)

completion = client.chat.completions.create(
    model="CohereLabs/c4ai-command-r7b-12-2024",
    messages=messages,
    tools=tools,
    temperature=0.7,
    max_tokens=512,
)

print(completion.choices[0].message)
```

## Billing

For direct requests, i.e. when you use a Cohere key, you are billed directly on your Cohere account.

For routed requests, i.e. when you authenticate via the hub, you'll only pay the standard Cohere API rates. There's no additional markup from us, we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)

Important Note ‼️ PRO users get $2 worth of Inference credits every month. You can use them across providers. 🔥

Subscribe to the [Hugging Face PRO plan](https://hf.co/subscribe/pro) to get access to Inference credits, ZeroGPU, Spaces Dev Mode, 20x higher limits, and more.  
