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

result = client.text_to_image(
	"A vibrant fireworks display",
	model=""
)
print(result)

```

#### from JS using @huggingface/inference


```js
import { HfInference } from "@huggingface/inference";

const client = new HfInference("xxxxxxxxxxxxxxxxxxxxxxxx");

const result = await client.textToImage({
  prompt: "A vibrant fireworks display",
  provider: "fireworks-ai"
});

console.log(result);
```

### From HTTP calls

TODO

## Billing

For direct requests, i.e. when you use a Fireworks key, you are billed directly on your Fireworks account.

For routed requests, i.e. when you authenticate via the hub, you'll only pay the standard Fireworks API rates. There's no additional markup from us, we just pass through the provider costs directly. (In the future, we may establish revenue-sharing agreements with our provider partners.)
