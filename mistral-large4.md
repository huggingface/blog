---
title: "Welcome Le Chonk: Mistral Large 4 on Hugging Face"
thumbnail: /blog/assets/ml4/thumbnail.png
authors:
- user: merve
- user: alvarobartt
- user: ror
- user: cyrilvallez
- user: qgallouedec
- user: ehcalabres
- user: juanjucm
- user: ngxson
- user: pcuenq
- user: burtenshaw
- user: ariG23498
---

# "Welcome Le Chonk: Mistral Large 4 on Hugging Face"

Mistral Large 4 (aka Le Chonk) is a multimodal, multilingual, Mixture-of-Experts (MoE) model from Mistral. It takes text and images as input and generates text. It has about **1T total, 49B active parameter**, and a context window of **1M tokens**.

The model is released in two official variants:

1. [FP8](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
2. [NVFP4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4)

There are also [GGUFs](https://huggingface.co/ggml-org) created by [Unsloth](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF) and the llama.cpp team. It’s distributed under the Mistral Ecosystem Open-Weight License (an MIT derivative that establishes limitations for model-as-a-service offerings).

Mistral Large 4 is particularly strong for cybersecurity, coding agents, and multimodality. In this post, we look at how the model works, its capabilities, architecture and more!

## Table of Contents

- ["Welcome Le Chonk: Mistral Large 4 on Hugging Face"](#welcome-le-chonk-mistral-large-4-on-hugging-face)
  - [Table of Contents](#table-of-contents)
  - [What is new with Mistral Large 4](#what-is-new-with-mistral-large-4)
    - [Cybersecurity Capabilities](#cybersecurity-capabilities)
  - [Architecture](#architecture)
    - [Vision encoder](#vision-encoder)
  - [Benchmark Results](#benchmark-results)
  - [Inference Support](#inference-support)
    - [Transformers](#transformers)
    - [vLLM](#vllm)
    - [Llama.cpp](#llamacpp)
  - [Fine-tuning with TRL](#fine-tuning-with-trl)
  - [Inference Endpoints](#inference-endpoints)
  - [Dell Enterprise Hub](#dell-enterprise-hub)
  - [Try it out](#try-it-out)

## What is new with Mistral Large 4

[Mistral Large 4](https://huggingface.co/collections/mistralai/mistral-large-4) combines a large pool of experts with sparse attention and a [Pixtral](https://huggingface.co/docs/transformers/en/model_doc/pixtral) vision encoder. The language model selects a small group of experts for each token, while its attention indexer selects which earlier token positions to attend to. These are two different forms of sparsity, where one selects feed-forward networks, the other selects context positions.

You can find all the checkpoints in the [Mistral Large 4 collection](https://huggingface.co/collections/mistralai/mistral-large-4).

### Cybersecurity Capabilities

Mistral Large 4 scores 82% on the Artificial Analysis Cyber Index test that asks a model to reproduce a real vulnerability in open-source software and then patch the software. Mistral reports that several closed frontier models score near zero on the same test because they refuse to attempt the work. This is super important for defenders, because fixing a vulnerability usually requires reproducing the vulnerability. In Mistral’s internal testing, Mistral Large 4 was also useful for malware analysis, vulnerability prioritisation and writing detection rules, which the model was not explicitly trained for.

Mistral mainly positions the model for defensive security work. Vulnerability research and incident response often appear like attacks in model input, so closed source filters can block them. A team that depends on an API can also lose access in the middle of an incident. Crucially, open weight models put both the usage policy and the deployment under the user’s control. That said, the model still declines harmful requests: on cyber prompts from JailbreakBench, StrongREJECT and AgentHarm, its average refusal rate is higher than any other open weight model.

For community builders, an open weight model like Mistral Large 4 unlocks a load of downstream use cases. You could build your own cyber system around your code base, crash logs, and malware samples. For example, with [TRL you could fine-tune](#fine-tuning-with-trl) smaller models from Mistral Large 4 traces patching your code base. In fact, reproduce-then-patch tasks use a verifiable reward, so it fits with RL environments that could be used to further RL smaller security models. Open weights also let researchers measure the model’s refusal behaviour directly instead of relying on the provider’s description.

In short, cyber capabilities like ML4’s in the open are a huge boost for the community.

> [!NOTE]
> We have interviewed Pierre Stock about the cyber capabilities of the model, [you can watch it here](https://www.youtube.com/embed/FczbSauVvjw).

## Architecture

![ML4](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ml4/ml4.png)

Mistral Large 4’s text decoder has 72 layers, a hidden dimension of 6,144, and 64 attention heads. Its attention blocks combine Multi-head Latent Attention (MLA) with a DeepSeek Sparse Attention (DSA) indexer (a direction in other models such as GLM 5.3). MLA keeps a compressed KV representation, and DSA selects up to 2,048 key positions for each query.

![Attention](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ml4/attention.png)  
For each decoder block, there’s a 3:1 RoPE to NoPE pattern, i.e. 3 decoder layers employ RoPE and every fourth layer uses NoPE. The model also shares KV state across layers.  
The first decoder layer uses a dense feed-forward network. The other 71 layers use MoE blocks, in which a softmax router selects four of the 128 experts for each token. One shared expert always runs alongside the selected experts, for a total of 5 active experts per token.

Regarding RoPE, Mistral uses *LeRope*, a YaRN-scaled RoPE with one learned multiplier per frequency.

![Experts](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ml4/experts.png)

### Vision encoder

Mistral Large 4 employs a 1.6B Pixtral encoder. Pixtral has 30 layers, a hidden dimension of 2,048, and 16 attention heads. It uses 14 × 14 patches and a patch-merge projector with spatial merge size 2. You can see the encoder pipeline below.

![Image Encoder](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ml4/img_encoder.png)

## Benchmark Results

Mistral Large 4 delivers strong benchmark performance across coding, cybersecurity, multimodal understanding, and agentic workflows, positioning it as a competitive addition to the open-weight ecosystem. As of October 29, 2026, the model is the fifth best open weights model on Artificial Analysis Intelligence index.

On coding, ML4 achieves **61.7% on DeepSWE v1.1**, **59.4% on SWE-Atlas-QnA**, and **28.3% on Terminal-Bench 4**. Its combined **Coding Agent Index score of 49.8%** places it ahead of DeepSeek V4 Pro 0813 and Qwen3.8 Max.

Coding capabilities of ML4 extend to knowledge work, with **59.9% on AutomationBench** and **1,393 Elo on AA-Briefcase**, which evaluate business workflows and long-horizon professional tasks.

Cybersecurity is one of Mistral Large 4’s standout capabilities, with strong results across real-world vulnerability assessment and security competition tasks. Three benchmarks highlight the model’s cyber performance:

* **Artificial Analysis Cyber Index:** ML4 ranks among the **top five models globally** on this independent evaluation of models’ ability to find and fix software vulnerabilities.  
* **Vulnerability reproduction and patching:** It achieves **82%**, the highest reported score on this test within the index. The task requires reproducing a real vulnerability in open-source software and then patching it, testing both investigation and remediation.  
* **Cybench:** ML4 solves **93%** of a suite of **40 challenges drawn from security competitions**, demonstrating strong performance on hands-on security problems.

Mistral Large 4 also demonstrates **93.3% attack resistance on Lakera’s B3 benchmark**, adding evidence of robustness against prompt-injection attacks. Beyond benchmarks, internal testing showed promising results in malware analysis, vulnerability prioritization, and detection-rule generation.

## Inference Support

Mistral Large 4 is shipped with day-0 support for transformers, vLLM and llama.cpp. GGUF quants are available from GGUF and the [llama.cpp team](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-GGUF). We provide options to deploy remotely with Inference Endpoints, and use serverless inference through Inference Providers.

### Transformers

The Transformers implementation provides `AutoModelForMultimodalLM`, with `AutoProcessor` preparing the conversation and image inputs. Mistral Large 4 can take `low`, `medium`, `high` as reasoning effort. The default is `high`, and note that reasoning cannot be disabled, so using `reasoning_effort`: `none` would still use `high`.

To run the model, upgrade the transformers package

```bash
pip install --upgrade transformers
```

```python
import os
from transformers import AutoProcessor, AutoModelForMultimodalLM, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

model_id = "mistralai/Mistral-Large-4-1T-A52B"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    distributed_config=distributed_config,
)

# only text
text_messages = [
    {
        "role": "user",
        "content": "Who are you?"
    }
]

# text + image
multimodal_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg",
            },
            {
                "type": "text",
                "text": "How many cats are in the image?",
            },
        ],
    },
]

for messages in [text_messages, multimodal_messages]:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        reasoning_effort="medium",
    ).to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=100)
    if int(os.environ["RANK"]) == 0:
        print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```

You can run the script above by saving it in a python file `script.py` and then run

```bash
torchrun --nproc-per-node 8 script.py
```

### vLLM

vLLM supports the high-quality NVFP4 Mistral Large 4 checkpoint, which has a smaller memory footprint and requires NVIDIA Blackwell GPUs. The model also ships with a DSpark drafter for additional speed-ups with vLLM. Launch the server like this:

```sh
vllm serve mistralai/Mistral-Large-4-1T-A52B-NVFP4 \
  --config-format mistral \
  --load-format mistral \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enforce-eager \
  --mm_processor_cache_gb 0.0
```

Once the server is up and running, you can run the following.

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":512}'
```

### Llama.cpp

Llama.cpp and [Llama App](https://llama.app/) already support Mistral Large 4.

You can install Llama App with the following command:

```sh
curl -LsSf https://llama.app/install.sh | sh
```

Then you can start the server as follows, indicating the quant suffix after the model ID:

```sh
llama serve -hf ggml-org/Mistral-Large-4-1T-A52B-GGUF:Q2_K
```

[Here’s the repository](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-GGUF) with the  quants derived from the FP8 checkpoint, and [this is the NVFP4 one](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-NVFP4-GGUF), converted from the official NVFP4 repo.

Once the server has started, you can head to localhost:9931 to chat using the built-in WebUI. From the web UI you can add images for inference and adjust the reasoning lever.  
You can also query the server as follows. Feel free to check out [the API reference](https://llama.app/docs/serve) for multimodality, tool calls and more.

```sh
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a limerick about python exceptions"}
        ]
    }'
```

You can run image-text inference as follows.

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <your-key>" \
    -d '{
        "model": "muse-spark-open",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,..."
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }'
```

The llama.cpp team ships the following quants: `IQ2_XXS`, `Q2_K`, `Q4_K_M`, `NVFP4`, `Q8_0`, and `BF16` for further conversions. The 2-bit quants run on a single Apple Silicon Ultra computer with 512 GB of unified RAM. The others require distributed inference. You can also run on multiple NVIDIA or AMD cards.

Running `Q2_K` on a single M3 Ultra, we got a prefill throughput of about 90 tokens per second, and a generation speed of 15 tok/s. On 4 x RTX6000 Pro, we observed the following results:

* NVFP4: 32K context size, prefill 130-590 t/s, gen speed ~10 t/s  
* IQ2_XXS: 800K context size. At short context sizes, prefill is 500 tok/s and generation runs at 30 tok/s. However, using long contexts performance drops to 129 tok/s (prefill) and 3 tok/s (generation).

## Fine-tuning with TRL

We also ship day 0 support of fine tuning Mistral Large 4 using TRL. All you have to do is upgrade to the latest version of `transformers`, `trl` and `peft`.

```sh
pip install --upgrade transformers trl peft
```

In the following example we train the model to perform multilingual thinking. The fine tuned model would be able to reason in the language the system prompt asks for\!

The script runs on the FP8 variant on a node of 8xB300s.

```py
# torchrun --nproc-per-node 8 sft.py
import os

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, DistributedConfig
from trl import SFTConfig, SFTTrainer

world_size = int(os.environ["WORLD_SIZE"])
model_id = "mistralai/Mistral-Large-4-1T-A52B"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(fsdp_size=world_size, ep_size=world_size),
    dtype="auto",
)

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
# the assistant turns carry reasoning, so render them in reasoning mode
dataset = dataset.map(lambda x: {"chat_template_kwargs": {"reasoning_effort": "high"}})

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        packing=True,
        max_length=4096,
        assistant_only_loss=True,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
    ),
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=r".*language_model.*\.(q_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|gate_proj|up_proj|down_proj)",
    ),
)
trainer.train()
```

## Inference Endpoints

Mistral Large 4 is also available on Inference Endpoints, with a ready-to-deploy recipe on 8 x NVIDIA H200 GPUs. The recipe runs vLLM with the FP8 weights split across all eight GPUs using tensor and expert parallelism, and uses the model's built-in DSpark draft head for faster generation through speculative decoding. It serves up to 131K tokens of context and processes up to 16 requests at a time, and we observed the following results:

* Single request: time to first token of about 0.6 seconds for a 1K-token prompt, and a generation speed of ~65 tok/s, reaching 120–138 tok/s on chat prompts where DSpark's drafts are accepted.  
* Long prompts: 16K-token prompts, four at a time, are prefilled at about 4,300 tok/s in total, with a time to first token of 8–9 seconds.  
* Under load: about 480 tok/s of total generation across 1,000 ShareGPT conversations.

You can deploy it with the `hf` CLI as follows:

```sh
hf endpoints catalog deploy --repo mistralai/Mistral-Large-4-1T-A52B --name mistral-large-4
```

Once the endpoint is running, you can query its OpenAI-compatible API, using the endpoint URL and your Hugging Face token. The `reasoning_effort` field controls how much the model reasons before answering, and can be set to `low`, `medium`, or `high` (default).

```sh
curl "$(hf endpoints describe mistral-large-4 --json | jq -r .status.url)/v1/chat/completions" \
    -H "Authorization: Bearer $(hf auth token 2>/dev/null)" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistralai/Mistral-Large-4-1T-A52B",
        "messages": [
            {"role": "user", "content": "Write a limerick about python exceptions"}
        ],
        "reasoning_effort": "medium"
    }'
```

The answer comes back in `choices[0].message.content`, and the reasoning in `choices[0].message.reasoning`. The endpoint also supports tool calls, structured outputs, image inputs, and streaming through the standard Chat Completions fields.

## Dell Enterprise Hub

The easiest way to get up and running with your Dell AI Servers (NVIDIA H200, B200, GB200 or GB300) is by installing the `dell-ai` CLI:

```sh
pip install dell-ai && dell-ai login
```

And use it to run an optimized recipe for Mistral Large 4 on e.g., a NVIDIA B300 node.

```sh
dell-ai models deploy --model-id mistralai/Mistral-Large-4-1T-A52B-NVFP4 \
    --engine docker \
    --platform-id xe9780-nvidia-b300 \
    --gpus 4
```

You can find more about this release in the [Dell Enterprise Hub (DEH)](https://dell.huggingface.co/updates/mistral-large-4).

## Try it out

Grab the weights from the [Mistral Large 4 collection](https://huggingface.co/collections/mistralai/mistral-large-4), run them locally with the [GGUFs](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-GGUF), or deploy them with Inference Endpoints. We can't wait to see what you build with Le Chonk. Share your fine-tunes, quants and other artefacts on the Hub!
