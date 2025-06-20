---
title: "Transformers backend integration in SGLang"
thumbnail: /blog/assets/196_transformers_backend_sglang/thumbnail.jpg
authors:
    - user: marcsun13
    - user: zhyncs
      guest: true
      org: SGLang Project
    - user: ispobock
      guest: true
      org: SGLang Project
    - user: lmzheng
      guest: true
      org: SGLang Project
---

Hugging Face transformers library is the standard for working with state-of-the-art models — from experimenting with cutting-edge research to fine-tuning on custom data. Its simplicity, flexibility, and expansive model zoo make it a powerful tool for rapid development.

But once you're ready to move from notebooks to production, inference performance becomes mission-critical. That’s where SGLang comes in.

Designed for high-throughput, low-latency inference, SGLang now offers seamless integration with transformers as a backend. This means you can pair the flexibility of transformers with the raw performance of SGLang. 

Let’s dive into what this integration enables and how you can use it.

# Transformers and SGLang

Let’s walk through a simple text generation example with `meta-llama/Llama-3.2-1B-Instruct` to compare both approaches.

## Transformers

transformers library is great for experimentation, small-scale tasks and training, but it's not optimized for high-volume or low-latency scenarios.

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
generate_kwargs = {
    "top_p": 0.95,
    "top_k": 20,
    "temperature": 0.8,
    "max_new_tokens": 256
}
result = pipe("The future of AI is", **generate_kwargs)
print(result[0]["generated_text"])
```

## SGLang

SGLang takes a different track, prioritizing efficiency with features like RadixAttention (a memory-efficient attention mechanism). Inference with SGLang is noticeably faster and more resource-efficient, especially under load. Here’s the same task in SGlang using an offline engine:

```python
import sglang as sgl

if __name__ == '__main__':
    llm = sgl.Engine(model_path="meta-llama/Llama-3.2-1B-Instruct")
    prompts = ["The future of AI is"]
    sampling_params =  {
        "top_p": 0.95,
        "top_k": 20,
        "temperature": 0.8,
        "max_new_tokens": 256
    }
    outputs = llm.generate(prompts, sampling_params)
    print(outputs[0])
```

Or you can spin a server and send requests:

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The future of AI is",
        "sampling_params": {
            "top_p": 0.95,
            "top_k": 20,
            "temperature": 0.8,
            "max_new_tokens": 256
        },
    },
)
print(response.json())
```

Note that SGLang also offers an OpenAI-compatible API, making it a drop-in replacement for external services.

# Transformers backend in SGLang

With the new transformers backend integration, SGLang can now automatically fall back to using transformers models it doesn’t natively support. This means in practice:

- Instant access to new models added to transformers
- Support for custom models from the Hugging Face Hub
- Less engineering overhead

This unlocks faster inference and optimized deployment (e.g enabling RadixAttention) without sacrificing the simplicity and versatility of transformers ecosystem. 

## Usage

```python
llm = sgl.Engine(model_path="new-transformers-model", impl="transformers")
```

Note that specifying the impl parameter is optional. If the model is not natively supported by SGLang, it switches to transformers implementation on its own.

Any model on the Hugging Face Hub that works with `transformers` using `trust_remote_code=True` and properly implements attention is compatible with SGLang. You can find the exact requirements in the official [documentation](https://docs.sglang.ai/supported_models/transformers_fallback.html#remote-code). If your custom model meets these criteria, all you need to do is set trust_remote_code=True when loading it.

```python
llm = sgl.Engine(model_path="new-custom-transformers-model", impl="transformers", trust_remote_code=True)
```

## Example 

Kyutai Team’s Helium isn’t yet natively supported by SGLang. This is where transformers backend shines, enabling optimized inference without waiting for native support.

```bash
python3 -m sglang.launch_server \
  --model-path kyutai/helium-1-preview-2b \
  --impl transformers \
  --host 0.0.0.0 \
  --port 30000
```


```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "top_p": 0.95,
            "top_k": 20,
            "temperature": 0.8,
            "max_new_tokens": 256
        },
    },
)
print(response.json())
```