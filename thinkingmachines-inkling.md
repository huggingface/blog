---
title: Welcome Inkling by Thinking Machines
thumbnail: /blog/assets/thinkingmachines-inkling/thumbnail.png
authors:
- user: burtenshaw
- user: merve
- user: pcuenq
- user: ariG23498
---

# Welcome Inkling by Thinking Machines

Inkling is a large (1T params!) open model to natively accept image, text, and audio inputs.

TLDR; Inkling by Thinking Machines is out on Hugging Face. Inkling is a huge multimodal LLM that understands all modalities (image, audio, text), has agentic capabilities, and supports 1M context. It comes in full BF16 and a well-calibrated NVFP4 variant, and includes speculative MTP layers for faster inference. There’s day-0 support in transformers, SGLang, and llama.cpp.

## What makes Inkling special?

Inkling is the first large open model with **~1T parameters** and **1M context window** to natively receive **image, text, and audio inputs**, trained on **45 trillion tokens of text, images, audio and video.** It’s focused on reasoning across modalities such as audio, images, and text; and is intended for domain adaptation via fine-tuning. We’ve tinkered with this model to build some demos and explore the architecture, and we think it’s great for building a new wave of multimodal reasoning apps.

## Overall Capabilities and Architecture

Inkling is a decoder-only multimodal Mixture-of-Experts model with 975B total and 41B active parameters. There are a lot of things going on, so let’s break each part down:

- Decoder-only: This means that the architecture supports causal autoregressive generation, like in most state-of-the-art LLMs.
- Multimodal: The model can ingest text, audio, and images.
- Mixture of Experts (MoE): The feed forward networks inside each layer are sparse, achieving faster inference because only 41B parameters are active at any given time. The model has 256 experts, as we’ll see later.

Here’s a quick glance of the architecture.

**Relative attention:** Instead of RoPE, which is the usual method to inject positional information in transformers models, Inkling uses relative attention to encode position information. Each attention layer learns position directly in the attention logits. Aside from key-query-values, there's a fourth projection producing a per-token, per-head relative feature R. This projection tensor is then tweaked with distance information (distance between the key and the query vector) and propagated into the attention module.

![Inkling relative attention architecture](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/relative_attention.png)

**Hybrid attention:** The decoder layers alternate between global attention (attending to the full context length at once) and sliding window attention (attending to a fixed context window in a sliding fashion). The architecture has a pattern of 5:1 sliding window to global attention layers. This hybrid attention scheme provides efficiency in computation. The final layer uses global attention to help build feature-rich representations.

**Short convolution:** The model uses a distinctive short 1D convolution, or `SConv` over the hidden states. SConv reads the current token and the previous `W-1` hidden states, with `W` being the sliding window size. The intuition here is that SConv helps with local attention while freeing the attention and MoE modules from local representations.

![Inkling short convolution architecture](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/sconv.png)

**MoE with shared experts sink:** In Inkling, the router scores both routed experts and shared experts. Top-k selection is performed over 6 experts, plus 2 shared experts always active.

**Vision understanding:** The model includes a simple hierarchical MLP patchifier consisting of several linear layers. Each layer merges pixels progressively, until the final layer produces one embedding per patch.

**Audio understanding:** The architecture employs a discretized mel spectrogram, where each of the audio chunks (of 100 ms) are converted to the mel scale and then classified into the exact mel spectrogram bin.

The multimodal towers are relatively simple modules, unlike other models that employ separate encoders for each modality. Each image patch passes through the image embedding tower and the audio chunk is passed through the audio embedding tower to get both media embeddings. Image inputs also include an additional temporal dimension for video processing. We expect this capability to be useful for downstream fine-tuning, but we haven’t evaluated out-of-the-box video performance. The tower folds the patch grid, a small local block of neighboring tokens is stacked into the channel dimension and goes through hMLP. The audio waveform is converted to mel scale, which is then classified into a discrete mel bin. These mel bin values are embedded in the audio embedding tower and the embeddings are then summed to construct the final audio input.

## Inference Support

Inkling comes with day-0 transformers support and is supported in major inference engines like SGLang and vLLM.

This model is huge. The bf16 checkpoint requires 2 TB of VRAM, while the nvfp4 version requires 600 GB of VRAM. You can try the model through serverless inference routers like Inference Providers, or use ggml quants for local deployment with llama.cpp.

### Transformers

The easiest way to infer with `transformers` directly is to use the `any-to-any` pipeline. You can use either the 16 bit `"thinkingmachines/Inkling"` on Hopper or later GPUs, or the quantized NVFP4 checkpoint `"thinkingmachines/Inkling-NVFP4"` on Blackwell Nvidia GPUs. Make sure to have the latest version of transformers (5.14.0 was released today) (`pip install -U transformers`).

```python
from transformers import pipeline

model_id = "thinkingmachines/Inkling"
# model_id = "thinkingmachines/Inkling-NVFP4"

pipe = pipeline("any-to-any", model=model_id)
```

After initializing the pipeline, you can pass in the prompt as follows.

```python
image_url = (
    "https://huggingface.co/datasets/merve/vl-test-suite/"
    "resolve/main/pills.jpg"
)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {
                "type": "text",
                "text": "Do components in this supplement interact with each other?",
            },
        ],
    },
]
output = pipe(
    messages,
    max_new_tokens=2000,
    return_full_text=False,
    reasoning_effort="medium",
)
output[0]["generated_text"]
```

Going one level lower, you can use Auto classes. For inference, you can use the `AutoModelForMultimodalLM` class for models and `AutoProcessor` class for processors. For different reasoning tasks, the tokenizer takes in a `reasoning_effort` argument. Existing options for reasoning effort are `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, and `"max"`.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You should only answer with a number."},
    {"role": "user", "content": "What is 17 * 23?"},
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    reasoning_effort="high",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=2000)
generated_tokens = output[0][inputs["input_ids"].shape[1] :]
print(processor.decode(generated_tokens, skip_special_tokens=False))
```

For multimodal inference, you can use the same classes. We provide example snippets for each different modality in the model card.

<details>
<summary>Text with image inference</summary>

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

image_url = (
    "https://huggingface.co/datasets/merve/vl-test-suite/"
    "resolve/main/pills.jpg"
)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {
                "type": "text",
                "text": "Do any of the components in this supplement interact?",
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="medium",
    return_dict=True,
    return_tensors="pt",
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=2000)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

processor.parse_response(response)
```

</details>

Inkling also takes in audio input. Below is an example inference snippet, which still uses the same `AutoModelForMultimodalLM` class.

<details>
<summary>Text with audio inference</summary>

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

audio_url = (
    "https://huggingface.co/datasets/merve/vl-test-suite/"
    "resolve/main/example_audio.mp3"
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech to text."},
            {
                "type": "audio",
                "audio": audio_url,
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

processor.parse_response(response)
```

</details>

For more realistic parallel deployment in a cluster of several nodes, please refer to the [Slurm](#slurm-scripts) section below.

### SGLang

SGLang is one of the fastest deployment frameworks for Inkling at the time of release, as it includes a custom model implementation. The launch command below shards the model across 8 GPUs and serves an OpenAI-compatible API on port 30000.

```shell
pip install sglang

python3 -m sglang.launch_server \
 --model-path thinkingmachine/Inkling \
 --tp-size 8 \
 --served-model-name inkling \
 --host 0.0.0.0 \
 --port 30000
```

Match `--tp-size` to your GPU count. Add `--mem-fraction-static` (e.g. `0.85`) if you need to leave more headroom for the KV cache.

### vLLM

vLLM is strong for production serving. A single `vllm serve` command downloads the weights from the Hub, shards the model across your GPUs with tensor parallelism, and starts an OpenAI-compatible server on port 8000.

```shell
pip install vllm

vllm serve thinkingmachine/Inkling \
  --tensor-parallel-size 8 \
  --served-model-name inkling
```

In practice, you will need multiple nodes and a distribution tool like SLURM (see below). Key parameters are `--tensor-parallel-size` to the number of GPUs on your node, and use `--max-model-len` to cap the context window if you hit KV-cache memory limits.

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Remote Inference with Hugging Face Inference Providers

You can infer with this model using several inference providers through Hugging Face. You can see all the code snippets to consume [here](https://huggingface.co/thinkingmachines/inkling?inference_provider=fastest&language=python&client=openai&inference_api=true). Below you can see how to use with the OpenAI client.

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="thinkingmachines/Inkling:auto",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ],
)

print(completion.choices[0].message)
```

Using the `“:auto”` suffix routes to your preferred provider in your settings; you can also use `“cheapest”` or `“:fastest”` as well. For this release, we cover the inference costs for 2 hours within the release for everyone.

Note: audio support in Inference Providers is work in progress and will be added shortly.

### Local Inference with llama.cpp and Unsloth

You can use `llama.cpp` to run quantized versions of the model on limited hardware. Unsloth have quantized the model down to 1-bit precision, reducing VRAM consumption by 95% over the original model.

```shell
llama serve -hf unsloth/inkling-GGUF:UD-IQ1_S
```

This starts an OpenAI-compatible server running at [`http://localhost:8000`](http://localhost:8000)`/v1` that you connect to in your preferred tool or clients. Heading there, you can start chatting with the model, and set it up with your favorite MCPs, pass in images or files conveniently and more!

Llama cpp also ships with a built-in UI that supports tools, mcp, and agentic workloads. Checkout Inkling running at 1-bit precision in the llama app:

<video controls width="100%" autoplay loop muted>
  <source src="https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/thinky.mp4" type="video/mp4">
</video>

Inkling GGUFs are also runnable in Unsloth Studio with dynamic 1-bit GGUFs which retain ~74.2% of top-1% accuracy whilst being 86% smaller.

![Inkling running in Unsloth Studio](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/unsloth.png)

## Use Cases

### Agentic coding with Pi

Pi is a minimal coding agent harness you can use with different language models. You can use Pi with either an inference engine server endpoint, such as llama.cpp, or with Inference Providers on Hugging Face by adding this to your `~/.pi/agent/models.json` after installation.

```json
{
  "providers": {
    "inference-providers": {
      "baseUrl": "https://router.huggingface.co/v1",
      "api": "openai-completions",
      "apiKey": "hf_...",
      "models": [
        {
          "id": "thinkingmachines/Inkling"
        }
      ]
    }
  }
}

```

Then you can start Pi in your project directory by calling `pi` and you’re good to go! In this demo, we give the model a hard math reasoning problem and it uses tools in pi to solve it.

![Visual reasoning gif demo](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/visual-reasoning.gif)

Inkling is focused on broad multimodality reasoning and low token consumption, so try it out with document processing or audio tasks.

### Multi Token Prediction Drafters

MTP adds extra layers to the model that predict several tokens at once, not just the next one. During inference, the extra layers act as “drafters” for speculative decoding, speeding up generation without compromising performance. With MTP, you get the exact same generated outputs, multipliers in generation speed-up at small memory cost in VRAM (due to serving the drafter). Thinking Machines also provides an MTP drafter with this release.

```python
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("thinkingmachines/Inkling")
model = AutoModelForMultimodalLM.from_pretrained(
    "thinkingmachines/Inkling",
    dtype=torch.bfloat16,
    device_map="auto",
)

# Preprocess the inputs.
...
generated = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False,
    use_mtp=True,
)
print(processor.decode(generated[0], skip_special_tokens=True))
```

### Multimodal Vision

We have prepared a small suite of reasoning questions from expert-level sources and university entrance exams. We have taken photos of the screen with watermarks in the screenshot to challenge the model. The model has solved all of them on high one, failed one in highest and medium reasoning efforts, so we provide a link to the model answers for you to check out how the model sounds and provide the number of tokens the model has taken to solve each of them. Note that we provide no system prompts in these vibe evals, and these reasoning questions should often be run with a good system prompt. The vibe eval images and results live [here](https://huggingface.co/buckets/merve/inkling).

| Category | Question | Number of Tokens (Reasoning Effort Medium) | Number of Tokens (Reasoning Effort High) | Number of Tokens (Reasoning Effort Max) |
| :---- | :---- | :---- | :---- | :---- |
| Open-ended Drug Interactions | Which components interact here? | 1,893 ✅ |  2,367 ✅ | 3,688 ✅ |
| Physics Question (MMMU-Pro) | Answer the question in the image. | 1,357 ✅ | 3,323 ✅ | 3,314 ✅ |
| Multilingual Physics Question | Answer the Turkish question given in the image. | 1,435 ✅ | 2,129 ✅ |  3,162 ✅ |
| Bar Exam | Answer the question in the image. | 1,117 ✅ | 2,137 ✅ | 1,676 ✅ |
| Infographics Question Answering (Open-ended) | Based on the information presented, approximately how many times larger is the projected summer warming period in the Arctic than the time over which substantial Arctic warming has already been observed? | 1,378 ❌ | 3,859 ✅ | 6000 (exceeded token budget) |

**Few notes on vibes:**

- Instead of directly answering the question on infographic, the model first turns text on image to text to ground itself.
- Prompting matters a lot to save tokens in reasoning, for instance, asking vague questions like “which components interact here?” with an image of the back of a pill, the model first needs to see what we mean by interactions here.
- Multi-choice question answers helped the model a lot in structuring its own reasoning, for open-ended questions the model struggled compared to MCQA, however, this is a common issue for many models. The usual chain of thought was OCR → characterize → evaluate each option → answer.
- 0.7 reasoning effort (medium) seems to provide a good trade-off.

### Multimodal Audio

We have vibe-evaluated the model on some audio reasoning examples from BigBenchAudio and a few multilingual audio examples of [GlobeAudio](https://huggingface.co/datasets/iNLP-Lab/GlobeAudio) (Russian and Chinese multi-choice questions asking the last word in transcription). The [BigBenchAudio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) examples we tested consist of logical statements and questions that either ask for formal fallacies (whether an argument can be logically deduced from the context given in audio) or object counting (stating multiple distinctive objects in the audio, asking for the total count of a certain one). Although this benchmark is initially made for speech-to-speech reasoning, we just want to see audio reasoning capabilities of this model. For GlobeAudio, the questions are relatively straightforward, so we ran with reasoning efforts of 0.1. We ran the first example of each language within GlobeAudio. All tests pass on all questions and efforts, except for second formal fallacy example on lowest effort, so we only provide the number of tokens spent in each question against reasoning effort. Vibe eval results and audio files live [here](https://huggingface.co/buckets/merve/inkling).

| GlobeAudio | Question | Number of completion tokens (Reasoning effort lowest) | Number of completion tokens (Reasoning effort medium) |
| :---- | :---- | :---- | :---- |
| Russian (asks for last word) | Какое последнее слово в аудиозаписи? 1. Россия 2. Свидетелем 3. Москва 4. Событий Choose the single correct option and answer with its exact text. | 130 | 179 |
| Russian (asks for profession of the speaker) | Кем, скорее всего, работает говорящая? 1. Репортершей 2. Блоггершей 3. Учительницей истории 4. Ведущей развлекательного шоу Choose the single correct option and answer with its exact text. | 105 | 136 |
| Chinese (asks for speaking rate) | 播报员的语速有何变化？ 1. 突然变快 2. 突然变慢 3. 保持不变 4. 时快时慢 Choose the single correct option and answer with its exact text. | 111 | 289 |

| Big Bench Audio  | Completion Tokens (lowest) | Completion Tokens (medium) | Number of completion tokens ( highest) |
| :---- | :---- | :---- | :---- |
| Formal Fallacy (10) | 285 | 335 | 444 |
| Formal Fallacy (39) | 275 (fails) | 555 | 778 |
| Object Counting (680) | 150 | 233 | 161 |

**Some notes on the vibes:**

- Similar to vision, the model first transcribes the speech before answering the question.
- It resists decoys: in Russian test, the model picked the right answer despite other answers appearing in the audio.
- Similar to vision, usual chain of thought is transcribe → characterize → evaluate each option → answer.
- The effort helps reasoning and not hearing. Audio question answering was much cheaper than images.

### Post-training

If you would like to use Inkling for post-training, Thinking Machines have built `tinker`, a managed tool for post-training open weight models. Their cookbook includes examples for fine-tuning, distillation, and reinforcement learning.

We post trained Inkling with tinker and OpenEnv, an agentic RL environment tool. We used the ECHO algorithm that trains a model to predict the environment without a verifier, applying next-token cross-entropy loss to tokens produced by the environment, alongside the usual policy learning on agent actions. This teaches the policy an implicit world model without requiring a separate model, teacher, or additional rollouts. Check out the [example](https://github.com/huggingface/OpenEnv/blob/main/examples/echo_world_model/backends/tinker_echo_demo.py).

![Inkling post-training metrics](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/trackio.png)

<details>
<summary>RL Example with Tinker and OpenEnv</summary>

```
git clone https://github.com/huggingface/OpenEnv.git
cd OpenEnv

# Add TINKER_API_KEY=... to .env, then run:
uv run --env-file .env \
  examples/echo_world_model/backends/tinker_echo_demo.py

```

</details>

If you’re working with Transformers Reinforcement Learning we suggest using Inkling as a teacher model in a knowledge distillation setup. For example, take advantage of Inkling’s document understanding abilities to improve the performance of a smaller (on-device) model. In [this example](https://github.com/huggingface/trl/blob/main/examples/scripts/gold.py), we use the transformer reinforcement learning library and the GOLD algorithm to distill knowledge. GOLD is handy here because it matches token logits between different tokenizers, so you can distill to any model on the hub.

## SLURM Scripts

To deploy Inkling on a cluster, we provide SLURM scripts serving with transformers API, as well as how to query the endpoint with different modalities. You can adapt these scripts to vLLM or SGlang by updating the commands. These scripts live [here](https://huggingface.co/buckets/merve/inkling).

<details>
<summary>SLURM `sbatch` Script</summary>

```shell
#!/bin/bash
#SBATCH --job-name=inkling-generate
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/merve/logs/inkling-generate-%j.out

# Usage:
#   sbatch submit_inkling_generate.sbatch
#   PROMPT="What is in this image?" IMAGE=/path/cat.png sbatch submit_inkling_generate.sbatch
#   PROMPT="Transcribe this." AUDIO=/path/clip_16k.wav sbatch submit_inkling_generate.sbatch

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/fsx/pedro/tm/models/tml-model-share}
PROCESSOR_PATH=${PROCESSOR_PATH:-${MODEL_PATH}}
PROMPT=${PROMPT:-"Explain tensor parallelism in simple terms."}
IMAGE=${IMAGE:-}
AUDIO=${AUDIO:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
THINKING_EFFORT=${THINKING_EFFORT:-}
TRANSFORMERS_SRC=${TRANSFORMERS_SRC:-/fsx/merve/transformers-tm/new-model-addition-tm}
SCRIPT=${SCRIPT:-${TRANSFORMERS_SRC}/scripts/generate_inkling.py}

NNODES=${SLURM_NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}

module load cuda/12.8
source /fsx/merve/cluster/bin/activate

export FI_PROVIDER=efa
export HF_HOME=${HF_HOME:-/fsx/merve/hf_cache}
export PYTHONPATH="${TRANSFORMERS_SRC}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')

echo "Nodes      = $SLURM_JOB_NODELIST"
echo "Head node  = $HEAD_NODE ($HEAD_IP)"
echo "Model      = $MODEL_PATH"
echo "Processor  = $PROCESSOR_PATH"
echo "Prompt     = $PROMPT"
echo "Image(s)   = ${IMAGE:-<none>}"
echo "Audio(s)   = ${AUDIO:-<none>}"

SCRIPT_ARGS=(--model-path "$MODEL_PATH" --processor-path "$PROCESSOR_PATH"
             --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS")
for img in $IMAGE; do SCRIPT_ARGS+=(--image "$img"); done
for aud in $AUDIO; do SCRIPT_ARGS+=(--audio "$aud"); done
[[ -n "$THINKING_EFFORT" ]] && SCRIPT_ARGS+=(--thinking-effort "$THINKING_EFFORT")

srun --ntasks-per-node=1 torchrun \
  --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-backend=c10d --rdzv-endpoint="${HEAD_IP}:${MASTER_PORT}" --rdzv-id="$SLURM_JOB_ID" \
  "$SCRIPT" "${SCRIPT_ARGS[@]}"

```

Also available [here](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/submit_inkling_generate.sbatch?download=true).

<details>
<summary>Python generation script</summary>

```python
#!/usr/bin/env python3
"""Multimodal generation with the inkling (tml) model in plain Transformers.

Runs one prompt, optionally with attached image(s) and/or audio clip(s), and
prints the model's answer. The model is sharded across all visible GPUs with
`tp_plan="auto"`, so launch it under `torchrun` (see
`submit_inkling_generate.sbatch`) for a multi-GPU or multi-node checkpoint.

    torchrun --nnodes=1 --nproc-per-node=8 generate_inkling.py \
        --model-path /path/to/tml-model-share \
        --image cat.png --prompt "What is in this image?"

    torchrun ... generate_inkling.py --audio clip.wav --prompt "Transcribe this."
    torchrun ... generate_inkling.py --prompt "Explain tensor parallelism."
"""

from __future__ import annotations

import argparse
import io
import os
import urllib.request
import wave

import numpy as np

# Set the allocator configuration before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from transformers import (  # noqa: E402
    AutoProcessor,
    InklingConfig,
    InklingForConditionalGeneration,
)


def load_image(source: str):
    from PIL import Image

    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:
            source = io.BytesIO(response.read())
    return Image.open(source).convert("RGB")


def load_wav(path: str, expected_rate: int) -> np.ndarray:
    """Decode a mono float32 waveform from PCM WAV using the standard library."""
    with wave.open(path, "rb") as wav_file:
        rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if rate != expected_rate:
        raise SystemExit(
            f"{path}: sample rate {rate} != {expected_rate}; "
            "resample to 16 kHz first"
        )

    if sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
        samples /= 2147483648.0
    else:
        raise SystemExit(
            f"{path}: unsupported sample width {sample_width} bytes "
            "(use 16/32-bit PCM)"
        )

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples


def build_messages(
    prompt: str,
    images: list[str],
    audios: list[str],
    sampling_rate: int,
    effort: float | None,
):
    """Build a user message with attached media followed by the text question.

    An optional reasoning effort is rendered as a leading system message.
    """
    content: list[dict[str, object]] = []
    for image in images:
        content.append({"type": "image", "image": load_image(image)})
    for audio in audios:
        content.append({"type": "audio", "audio": load_wav(audio, sampling_rate)})
    content.append({"type": "text", "text": prompt})

    messages = []
    if effort is not None:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"Thinking effort level: {effort}",
                    },
                ],
            },
        )
    messages.append({"role": "user", "content": content})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal generation with the inkling (tml) model."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH",
            "/fsx/pedro/tm/models/tml-model-share",
        ),
    )
    parser.add_argument(
        "--processor-path",
        default=os.environ.get("PROCESSOR_PATH"),
        help="Defaults to --model-path.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain tensor parallelism in simple terms.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image file path or HTTP(S) URL (repeatable).",
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=[],
        help="16 kHz mono PCM WAV path (repeatable).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("MAX_NEW_TOKENS", "512")),
    )
    parser.add_argument(
        "--thinking-effort",
        type=float,
        default=None,
        help="Reasoning effort from 0.0 to 1.0.",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # torchrun may append launcher arguments after the script path.
    args, _ = parser.parse_known_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    processor = AutoProcessor.from_pretrained(
        args.processor_path or args.model_path
    )
    config = InklingConfig.from_pretrained(args.model_path)
    model = InklingForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        tp_plan="auto",
    )
    model.eval()

    messages = build_messages(
        args.prompt,
        args.image,
        args.audio,
        processor.feature_extractor.sampling_rate,
        args.thinking_effort,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)
    if "pixel_values" in inputs:
        # The vision tower runs in the model dtype.
        inputs["pixel_values"] = inputs["pixel_values"].to(
            dtype=torch.bfloat16
        )

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": config.eos_token_id,
    }
    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        generated = model.generate(**inputs, **generate_kwargs)

    if rank == 0:
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        answer = processor.tokenizer.decode(
            new_tokens,
            skip_special_tokens=False,
        )
        print(
            f"[prompt {inputs['input_ids'].shape[1]} tok "
            f"| images {len(args.image)} | audios {len(args.audio)} "
            f"| generated {new_tokens.numel()} tok]",
            flush=True,
        )
        print(answer, flush=True)


if __name__ == "__main__":
    main()

```

Also available [here](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/generate_inkling.py?download=true).

<details>

## Benchmark Results

|Benchmark|GLM-5.2|GLM-5.1|Qwen3.7-Max|MiniMax M3|DeepSeek-V4-Pro|Claude Opus 4.8|GPT-5.5|Gemini 3.1 Pro|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Reasoning|||||||||||
|HLE|40.5|31|41.4|37|37.7|49.8*|41.4*|45|
|HLE (w/ Tools)|54.7|52.3|53.5|-|48.2|57.9*|52.2*|51.4*|
|CritPt|20.9|4.6|13.4|3.7|12.9|20.9|27.1|17.7|
|AIME 2026|99.2|95.3|97|-|94.6|95.7|98.3|98.2|
|HMMT Nov. 2025|94.4|94|95|84.4|94.4|96.5|96.5|94.8|
|HMMT Feb. 2026|92.5|82.6|97.1|84.4|95.2|96.7|96.7|87.3|
|IMOAnswerBench|91.0|83.8|90|-|89.8|83.5|-|81|
|GPQA-Diamond|91.2|86.2|90|93|90.1|93.6|93.6|94.3|
|Coding|||||||||||
|SWE-bench Pro|62.1|58.4|60.6|59|55.4|69.2|58.6|54.2|
|NL2Repo|48.9|42.7|47.2|42.1|35.5|69.7|50.7|33.4|
|DeepSWE|46.2|18|18|20|8|58|70|10|
|ProgramBench|63.7|50.9|-|-|47.8|71.9|70.8|39.5|
|Terminal Bench 2.1 (Terminus-2)|81.0|63.5|75|65|64| 85|84|74|
|Terminal Bench 2.1 (Best Reported Harness)|82.7|69|-|-|-|78.9|83.4|70.7|
|FrontierSWE (Dominance)|74.4|30.5|-|-|29.0|75.1|72.6|39.6|
|PostTrainBench|34.3|20.1|-|-|-|37.2|28.4|21.6|
|SWE-Marathon|13.0|1.0|-|-|-|26.0|12.0|4.0|
|Agentic|||||||||||
|MCP-Atlas (Public Set)|76.8|71.8|76.4|74.2|73.6|77.8|75.3|69.2|
|Tool-Decathlon|48.2|40.7|-|-|52.8|59.9|55.6|48.8|

