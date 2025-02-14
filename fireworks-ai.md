---
title: "Welcome Fireworks.ai on the Hub 🎆"
thumbnail: /blog/assets/inference-providers/fireworks-ai.png
authors:
- user: your_username
---

Following our recent announcement on [Inference Providers on the Hub](https://huggingface.co/blog/inference-providers), we're thrilled to share that **Fireworks.ai** is now a supported Inference Provider on HF Hub!

[Fireworks.ai](https://fireworks.ai) delivers blazing-fast, serverless inference directly on your model pages—making it easier than ever to deploy and experiment with your favorite models.

Among others, starting now, you can run serverless inference to the following models via Fireworks.ai:

- [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)
- [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- [meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct)

and many more, you can find the full list [here](https://huggingface.co/models?inference_provider=fireworks-ai).

Light up your projects with Fireworks.ai today!

## How it works

### In the website UI

TODO(screenshot)

See all models supported by Fireworks on HF here: https://huggingface.co/models?inference_provider=fireworks-ai


#### from Python, using huggingface_hub
Install `huggingface_hub` from source: 
```bash
pip install git+https://github.com/huggingface/huggingface_hub

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="fireworks-ai",
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

#### from JS using @huggingface/inference


```js
import { HfInference } from "@huggingface/inference";

const client = new HfInference("xxxxxxxxxxxxxxxxxxxxxxxx");

const chatCompletion = await client.chatCompletion({
    model: "deepseek-ai/DeepSeek-R1",
    messages: [
        {
            role: "user",
            content: "How to make extremely spicy Mayonnaise?"
        }
    ],
    provider: "fireworks-ai",
    max_tokens: 500
});

console.log(chatCompletion.choices[0].message);
```

### From HTTP calls

```
curl 'https://router.huggingface.co/fireworks-ai/v1/chat/completions' \
-H 'Authorization: Bearer xxxxxxxxxxxxxxxxxxxxxxxx' \
-H 'Content-Type: application/json' \
--data '{
    "model": "Llama-3.3-70B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "What is the meaning of life if you were a dog?"
        }
    ],
    "max_tokens": 500,
    "stream": false
}'
```

## Billing

For direct requests, i.e. when you use a Fireworks key, you are billed directly on your Fireworks account.

For routed requests, i.e. when you authenticate via the hub, you'll only pay the standard Fireworks API rates. There's no additional markup from us, we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)
