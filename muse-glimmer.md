---
title: "Meta is back with Muse Glimmer: local, agentic, multimodal, and open source" 
thumbnail: /blog/assets/muse-glimmer/thumbnail.png
authors:
- user: merve
- user: burtenshaw
- user: pcuenq
- user: ariG23498
---

# Meta is back with Muse Glimmer: local, agentic, multimodal, and open source!

Great news from the OGs of open source LLMs! Muse Glimmer, released today, is Meta’s new multimodal model, especially designed for local agentic use cases. Distilled from Muse to **30B** parameters, and released under the **Apache 2.0 license**, it’s ideal deploying locally for privacy, reducing costs, or just hacking around. It’s intended for privacy-aware applications such as coding, document analysis, personal assistants, Claw- or Hermes-like setups.

To celebrate, we are shipping with Meta day-0 support in `transformers`, `llama.cpp`, `vLLM`, Inference Endpoints, and other libraries. We built a few cool things and explain our findings in this blog. **Check out the demo’s below for inspiration.**

## Architecture

Muse Glimmer is a dense 30B parameter model consisting of:

- 2B ViT-style encoder for vision (Perception Encoder)  
- 28B parameter text decoder

In addition to the main VLM, there’s also a speculative decoding drafter implemented on DFlash. Usage of this module is optional, and it can provide much faster generation in exchange for some memory cost. We found this drafter to be particularly well suited to structured content generation such as coding.

### Text Decoder

The language model uses the following architecture components:

- **Hybrid attention:** Alternating between three sliding window layers (of 2,048 tokens) using rotary position embedding, followed by a fourth layer that uses full attention and NoPE (no positional embedding). The pattern is therefore (SWA, SWA, SWA, Full), repeated 13 times to a total of 52 layers. This allows the model to retain relative order and distance information with RoPE and preserve information globally with NoPE.  
- **Gated Grouped-Query Attention:** Each key-value head is shared by 16 query heads, which reduces KV-cache memory by 16x and makes generation faster and cheaper.  
- **Q-K normalization with extra query scaling:** Before computing attention, Muse Glimmer applies RMS normalization to every query and key head to keep attention logits stable. After this, queries are multiplied by a scale factor to set the target logit scale after normalization. The extra query scaling behaves like an inverse temperature at the softmax level.

### Perception Encoder

Muse Glimmer uses one image encoder to handle both images and videos. Unlike the relatively small vision encoders used in other VLMs, this is a sizable 2B ViT-like model designed after the Perception Encoder architecture. Perception Encoder was previously introduced by Meta [as a backbone for various downstream spatial and multimodal tasks](https://huggingface.co/papers/2504.13181).  
The encoder patchifies images to a shape of 2 frames x 3 channels x 14 x 14, and passes them through a linear layer for projection. An interpolated absolute position embedding from a learned position table is then added to these embeddings. These are then sent to the vision tower which consist of 50 layers and GELU MLPs. Similar to the language model, the attention pattern consists of three window attention layers followed by one full attention layer. Inside the attention layers, 2D RoPE is applied to the queries and keys.

After transformer, pixel shuffle concatenates 2x2 groups of neighboring spatial tokens which reduces the number of image tokens 4x without discarding their channels. The merged features are then projected to the shared embedding space of the text decoder.

Videos go through the same encoder frame by frame, where each frame is converted into patches (of shape [batch, temporal groups, grid height, grid width, 2 frames, 3 channels, 14, 14]). The processor targets 2 frames per second and caps the clip at 96 frames sampled evenly across video. The processor creates timestamped video placeholders, interleaving text with frame e.g. “Time: 0.0s <|video|> x N” in which the final video embeddings are replaced before the final projection layer.

### Transformers

Upgrade transformers to the latest version to be able to use Muse Glimmer.

```bash
pip install --upgrade transformers accelerate
```

Muse Glimmer comes with day-0 support in transformers, both for the main model and the speculative decoding drafter. You can use `AutoModelForMultimodalLM` and `AutoProcessor` classes to load the model and the processor.

```py
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "meta/Muse-Glimmer-30B"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)
```

The same snippet runs unchanged on NVIDIA (CUDA), AMD (ROCm) and Intel (XPU) GPUs, `device_map="auto"` places the model on whichever accelerator is available.

#### Text-only Inference

After loading the model, you can do text-only inference with it as follows.

```py
# Prompt
messages = [
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    reasoning_strength="low"
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(response)
```

#### Prompting the model with images and text

We would need `torchvision` to be able to use images and text.

```bash
pip install torchvision
```

Muse Glimmer accepts images as input, as demonstrated here:

```py
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/merve/vl-test-suite/resolve/main/SF.png"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    reasoning_strength="low"
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(response)
```

#### Multimodal tool calling

Muse Glimmer can do multimodal tool calling, here’s how you can do it. In the example below, we ask the model to call the weather tool based on the city in the image.

```py
import json
import re

tools = [
    {
        "type": "function",
        "function": {
            "name": "weather.get",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/merve/vl-test-suite/resolve/main/SF.png"},
            {"type": "text", "text": "I'm going to the city in this picture. What clothes should I wear?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tools=tools,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    reasoning_strength="low"
).to(model.device)

input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=128)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

parsed = processor.tokenizer.parse_response(response)

```

#### Object Detection

You can use Muse Glimmer to do open ended object detection in images as follows.

```py
import json

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "https://huggingface.co/datasets/merve/vl-test-suite/resolve/main/SF.png"},
        {
            "type": "text",
            "text": (
                "Detect the bridge. Return only the detection in the model's "
                "native object-detection format, with no explanation."
            ),
        },
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=128)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

detections = json.loads(response.removesuffix("<|eot|>"))
print(detections)
# [{"x_min": 0, "y_min": 390, "x_max": 520, "y_max": 603}]

# note that you need to scale X and Y values to image size to visualize:
xyxy = (
round(box["x_min"] / 1000 * width),
round(box["y_min"] / 1000 * height),
round(box["x_max"] / 1000 * width),
round(box["y_max"] / 1000 * height),
)
```

#### Video Inference

To work with videos we recommend installing `torchcodec` into the environment.

```bash
pip install torchcodec
```

Muse Glimmer can answer complex questions about videos without audio. You can do video inference as follows, here’s an example from VideoMME2, which is the most popular video question answering benchmark.

```py
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://huggingface.co/datasets/merve/vl-test-suite/resolve/main/IMG_8137.mp4"},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    reasoning_strength="low",
    processor_kwargs={"num_frames": 96},
).to(model.device)

input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=1024)

response = processor.decode(
    outputs[0, input_len:],
    skip_special_tokens=False,
)

parsed = processor.parse_response(
    response,
    prefix=inputs["input_ids"],
)
print(parsed)

```

### Llama.cpp

Muse Glimmer comes with day-0 llama.cpp support. Meta has distributed calibrated quants in [this repo](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF), and Uunsloth is releasing optimized quants as well. DFlash speculative decoding is supported as well. You can use a pre-built llama binary to start a llama server or a CLI. To install llama.cpp, run

```bash
curl -LsSf https://llama.app/install.sh | sh
```

Then you can start the server as follows.

```bash
llama serve meta-models/Muse-Glimmer-30B-GGUF
```

Once the server has started, you can head to localhost:8080 to chat with the built-in WebUI.

TODO: Insert webui video with this model

You can also query the server as follows.

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a limerick about python exceptions"}
        ]
    }'
```

You can also use llama server with coding agents like Pi.

## Speculative Decoding

DFlash uses a lightweight block-diffusion drafter model to provide same output with extra speed-ups in decoding phase. Transformers and llama.cpp ship support for DFlash drafter of Muse Glimmer day-0.

### Speculative Decoding with transformers

You can load the drafter and model as follows, and infer like how you would with base model with an additional parameter (shown in the upcoming snippets).

```py
import torch
from transformers import AutoProcessor, MuseGlimmerAssistantModel, MuseGlimmerForConditionalGeneration

model_id = "meta-models/Muse-Glimmer-30B"
target = MuseGlimmerForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
+ assistant = MuseGlimmerAssistantModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/merve/vl-test-suite/resolve/main/SF.png"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

+ out = target.generate(**inputs, assistant_model=assistant, speculation_type="dflash", max_new_tokens=64, do_sample=True)
print(processor.batch_decode(out)[0])
```

### Speculative Decoding with llama.cpp

You can start llama server using following command. `--spec-draft-n-max` argument controls how many future tokens DFlash proposes during each speculative-decoding step. Muse Glimmer’s DFlash model was trained with a block size of 16, one anchor token plus 15 proposed tokens, so any value above 15 will be clamped to 15\.

```bash
llama serve -hf meta-models/Muse-Glimmer-30B-GGUF --spec-type draft-dflash --spec-draft-n-max 15
```

You can also use llama cli with speculative decoding drafter as follows.

```bash
llama cli -hf meta-models/Muse-Glimmer-30B-GGUF --spec-type draft-dflash

```

## Support for Muse Glimmer vLLM with transformers backend

For this release, we ship support for vLLM with transformers backend.

```bash

# tensor parallel serving across 4 GPUs
vllm serve meta-models/Muse-Glimmer-30B --model-impl transformers --tensor-parallel-size 4

# infer
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "username/muse-glimmer-hf-v2",
      "messages": [
        {"role": "user", "content": "Explain tensor parallelism briefly."}
      ],
      "temperature": 0.0,
      "max_tokens": 256
    }'

```

## Fine-tuning with TRL

You can use TRL to fine-tune Muse Glimmer using various methods from SFT to Async GRPO. We have run two experiments on bf16 with Hopper-class GPUs with 80GB VRAM each.

| Workload | Practical minimum |
|---|---:|---:|
| Inference / eval, BF16 | 1×80 GB H100 | 1×H100 |  
| LoRA SFT, BF16 | 1×80 GB H100, microbatch 1 + checkpointing |
| Full SFT, BF16 | 8×80 GB H100 with FSDP/ZeRO-3 |
| LoRA GRPO, Transformers rollouts | 1×80 GB H100, but slow/tight |  
| LoRA GRPO, separate vLLM rollout server | 8×H100: 4 rollout + 4 training |
| Full-finetune GRPO | 8 GPUs is usually insufficient |

As part of this release, we ship an example to fine-tune Muse Glimmer to fine-tune on small split of MolmoWeb dataset. This shows how to make model generate structured outputs and how to fine-tune on images.

## Demos

Here are some fun ways to try out Muse Glimmer. In our opinion, the coolest thing about this model is that it is a local scale personal assistant that can code. That means you can make it do things like, quantize itself, find quantized weights on the Hub, deploy itself to inference endpoints, and even optimize itself for specific hardware\! Let’s go team local 🚀

## Connect OpenClaw to Muse Glimmer

Assume the Inference Endpoint exposes an OpenAI-compatible `/v1` API.

Set `HF_TOKEN` in the OpenClaw gateway environment, then add this to `~/.openclaw/openclaw.json`:

<details>
<summary>OpenClaw configuration</summary>

```json5
{
  models: {
    mode: "merge",
    providers: {
      muse: {
        baseUrl: "https://YOUR-ENDPOINT.endpoints.huggingface.cloud/v1",
        apiKey: {
          source: "env",
          provider: "default",
          id: "HF_TOKEN"
        },
        api: "openai-completions",
        authHeader: true,
        models: [{
          id: "meta/Muse-Glimmer-30B",
          name: "Muse Glimmer",
          reasoning: false,
          input: ["text", "image"],
          contextWindow: 32768,
          maxTokens: 8192
        }]
      }
    }
  },
  agents: {
    defaults: {
      model: { primary: "muse/meta/Muse-Glimmer-30B" }
    }
  }
}
```

Restart OpenClaw:

```bash
openclaw gateway restart
```

Validate from a fresh session:

```bash
openclaw agent --message "Reply with: muse-ready"
```

</details>

Use the exact model ID returned by the endpoint’s `/v1/models` response if it differs.

### Hey Muse Glimmer, quantize yourself

If we hook up Muse Glimmer to the [Hugging Face MCP](https://huggingface.co/mcp) and update its [`AGENTS.md`](http://AGENTS.md) we give it the capability to find a quantized version of itself on the hub and run locally. This is handy if you want to work on something private, or just cut costs.

If you do this a second time, Muse Glimmer will find the cached weights and switch to them, so feel free to add a convenient command like `/spawn`. 

Muse Glimmer inspects the machine and Hub, selects or creates a Q4\_K\_M GGUF, launches llama-server, and validates model discovery and chat completion. The result is a smaller local build behind an OpenAI-compatible API. Here’s the prompt we added to `AGENTS.md`.

[https://huggingface.co/buckets/huggingface/muse-glimmer-assets/resolve/Muse%20Glimmer%20Quantisation%20Demo%20-%20explained.mp4?download=true](https://huggingface.co/buckets/huggingface/muse-glimmer-assets/resolve/Muse%20Glimmer%20Quantisation%20Demo%20-%20explained.mp4?download=true)

By adding this to [`AGENTS.md`](http://AGENTS.md) openclaw or hermes will be able to solve the rest.

\<details\>  
\<summary\>Local quantization prompt\</summary\>

```bash
## Local model deployment

When asked to deploy locally, perform the work; do not give instructions.

1. Inspect hardware and the Hugging Face cache.
2. Search the Hub for compatible GGUF weights using `apps=llama.cpp`; confirm exact filenames through the model-tree API.
3. Prefer an existing suitable GGUF, normally `Q4_K_M`. Treat `mmproj-*.gguf` as projector weights.
4. If no GGUF exists, download the source weights, convert with `convert_hf_to_gguf.py`, then quantize with `llama-quantize`.
5. Preserve source weights and record the repository, revision, filenames, and quantization.
6. Start `llama-server` with an `onyx` alias and an OpenAI-compatible endpoint.
7. Validate `/v1/models` and `/v1/chat/completions`, requiring non-empty, correct content.
8. Report concise progress and logs. Claim completion only after validation passes.
```

\</details\>

### Hey Muse Glimmer, deploy yourself

Muse Glimmer can also take care of the opposite. Let’s get Glimmer to deploy itself on Hugging Face Inference Endpoints. Which is useful if you want to speed up on some cutting edge hardware.

N.B. You can also just deploy [Muse Glimmer to Inference Endpoints](https://endpoints.huggingface.co/huggingface/new/meta-models/Muse-Glimmer-30B) directly and connect your agent.

Muse Glimmer pins the model revision, deploys it to a protected Hugging Face Inference Endpoint, and verifies health, model discovery, and chat completion. It then connects the Claw agent with secrets and rollback preserved. Here’s the prompt we added to [`AGENTS.md`](http://AGENTS.md). Muse glimmer will also need the [Hugging Face MCP](https://huggingface.co/mcp) and/or the [Hugging Face CLI and Skills](https://huggingface.co/docs/hub/en/agents-skills).

\<details\>  
\<summary\>Inference Endpoint deployment prompt\</summary\>

```bash
## Hugging Face Inference Endpoint deployment

When asked to deploy on Hugging Face Inference Endpoints, perform the work; do
not give instructions.

1. Inspect Hugging Face authentication, the current model repository, and any
   existing endpoints.
2. Confirm the exact model repository and immutable revision through the Hub
   API; inspect its architecture, configuration, and chat template.
3. Confirm that the model is supported by vLLM, then deploy or update a
   protected Inference Endpoint using the managed native vLLM engine.
4. Choose an available region and the smallest suitable accelerator. Use one
   replica and enable scale-to-zero when supported.
5. Preserve the previous endpoint configuration for rollback. Do not expose
   tokens, publish private weights, or replace an unrelated endpoint.
6. Wait for the endpoint to become ready. If startup fails, inspect the logs
   and report the actual blocker rather than repeatedly changing settings.
7. Validate `/health`, `/v1/models`, and `/v1/chat/completions`, requiring the
   expected model and non-empty, correct content. When agent use is required,
   also validate a real structured tool call.
8. Configure the Claw agent to use the endpoint's OpenAI-compatible `/v1` URL,
   storing credentials as secrets and retaining the previous provider as
   rollback. Test the connection in a fresh session.
9. Report concise progress and finish with the repository, revision, engine,
   hardware, endpoint URL, scaling state, and validation results. Claim
   completion only after every required check passes.
```

\</details\>

### Hey Muse Glimmer, optimize yourself

Finally, let’s get Muse Glimmer to do some light RSI. We can instruct our agent to optimize its own inference engine for specific hardware, in this case a Nvidia H100. To do this, the agent will need to use another inference engine, like Inference Endpoints above.

Muse Glimmer benchmarks its own single-H100 serving stack, testing one reversible change at a time while holding the workload fixed. It keeps only correctness-passing gains and finishes with the fastest reproducible configuration. Here’s the prompt we added to [`AGENTS.md`](http://AGENTS.md). Muse glimmer need the [Hugging Face MCP](https://huggingface.co/mcp) and the [Hugging Face CLI and Skills](https://huggingface.co/docs/hub/en/agents-skills).

\<details\>  
\<summary\>Self-optimization prompt\</summary\>

```bash
You are Muse Glimmer acting as an autonomous inference-optimization engineer for your own serving stack.

Goal: maximize valid single-H100 aggregate completion throughput in tokens/second.

Protocol:
1. Establish a correctness-passing baseline.
2. Test one reversible optimization at a time.
3. Keep the prompt, concurrency, sampling, request count, warm-up, and decode length fixed.
4. Reject results that fail correctness or prefix checks.
5. Record every experiment chronologically with its configuration, raw throughput, correctness, and delta.
6. Keep improvements and revert regressions.
7. Stop after six consecutive regressions or when the experiment budget is exhausted.
8. Report the best valid configuration and exact reproduction command.

Create a minimal scientific animation of the results:
- white background;
- raw tokens/second—never normalize;
- one point revealed per experiment;
- connect every point chronologically;
- begin with the lowest valid result;
- stop at the best result;
- export as a GIF.

Never fabricate, interpolate, or count correctness-failing measurements.
```

\</details\>

[https://huggingface.co/buckets/huggingface/muse-glimmer-assets/resolve/onyx-optimization-progress.gif?download=true](https://huggingface.co/buckets/huggingface/muse-glimmer-assets/resolve/onyx-optimization-progress.gif?download=true)
