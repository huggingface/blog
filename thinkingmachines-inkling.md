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

[Inkling](https://huggingface.co/thinkingmachines/Inkling) is a large (1T params!) open model to natively accept image, text, and audio inputs.

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

## Slurm Scripts

To deploy Inkling on a cluster, we provide SLURM scripts serving with transformers API, as well as how to query the endpoint with different modalities. You can adapt these scripts to vLLM or SGlang by updating the commands. These scripts live [here](https://huggingface.co/buckets/merve/inkling).

* [Submit inference job](https://huggingface.co/buckets/merve/inkling/tree/slurm/submit_inkling_generate.sbatch)
* [Python generation script](https://huggingface.co/buckets/merve/inkling/tree/slurm/generate_inkling.py)

## Benchmark Results

|     |     | Inkling | Nemotron 3 Ultra | Kimi K2.5 | Kimi K2.6 | GLM 5.2 | DeepSeek V4 Pro | Gemini 3.1 Pro (high) | Claude Fable 5 (max) | GPT 5.6 Sol (xhigh) |
|-----|-----|---------|------------------|-----------|-----------|---------|-----------------|-----------------------|----------------------|---------------------|
| **Reasoning** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | HLE (text only) | 29.7%   | 26.6%            | 29.4%     | 35.9%     | 40.1%   | 35.9%           | 44.7%                 | 53.3%                | 47.2%               |
|     | HLE (with tools) | 46.0%   | 37.4%            | 50.2%     | 54.0%     | 54.7%   | 48.2%           | 51.4%                 | 64.5%                | 55.0%               |
|     | AIME 2026 | 97.1%   | 94.2%            | 95.8%     | 96.4%     | 99.2%   | 96.7%           | 98.3%                 | –                    | 99.9%               |
|     | GPQA Diamond | 87.2%   | 86.7%            | 87.9%     | 91.1%     | 89.5%   | 88.8%           | 94.1%                 | 92.6%                | 94.1%               |
| **Agentic (coding)** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | SWEBench Verified | 77.6%   | 70.7%            | 76.8%     | 80.2%     | –       | 80.6%           | 80.6%                 | 95.0%                | –                   |
|     | SWEBench Pro (Public) | 54.3%   | 46.4%            | 50.7%     | 58.6%     | 62.1%   | 55.4%           | 54.2%                 | 80.0%                | 64.6%               |
|     | Terminal Bench 2.1 (Best Harness) | 63.8    | 56.4             | 51.3         | 71.3      | 82.7    | 64              | 73.8                  | 84.6                 | 89.5                |
|     | GDPVal-AA v2 | 1233    | 1164             | 1009         | 1190      | 1514    | 1307            | 962                   | 1760                 | 1748                |
| **Agentic (general)** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | MCP Atlas | 74.1%   | 44.7%                | 64.0%     | 68.1%     | 77.8%   | 73.2%           | 78.2%                 | 83.3%                | 81.8%               |
|     | Tau 3 Banking | 23.7%   | 13.8%            | 13.2%     | 20.6%     | 26.8%   | 25.8%           | 16.5%                 | 26.8%                | 33.0%               |
| **Factuality** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | BrowseComp (w/ Ctx) | 77.1%   | –                | 74.9%     | 83.2%     | –       | 83.4%           | 85.9%                 | 88.0%                | 89.4%               |
|     | SimpleQA Verified | 43.9%   | 32.4%            | 36.9%     | 38.7%     | 38.1%   | 57.0%           | 77.3%                 | 68.3%                | 71.6%               |
|     | AA Omniscience | 1.0%    | -1.0%            | -8.0%     | 6.0%      | 4.0%    | -10.0%          | 33.0%                 | 40.0%                | 22.0%               |
| **Chat** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | IFBench | 79.8%   | 81.4%            | 70.2%     | 76.0%     | 73.3%   | 76.5%           | 77.1%                 | 63.5%                | 72.7%               |
|     | Global-MMLU-Lite | 88.7%   | 85.6%            | 84.0%     | 88.4%     | 89.2%   | 89.3%           | 92.7%                 | 93.3%                | 91.8%               |
| **Vision** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | MMMU Pro (Standard 10) | 73.3%   | –                | 75.0%     | 79.0%     | –       | –               | 82.0%                 | 84.2%                | 83.0%               |
|     | Charxiv RQ | 78.1%   | –                | 77.5%     | 80.4%     | –       | –               | 80.2%                 | 86.5%                | 84.7%               |
|     | Charxiv RQ (with python) | 82.0%   | –                | 78.7%     | 86.7%     | –       | –               | 89.9%                 | 89.4%                | 87.8%               |
| **Audio** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | Audio MC | 56.6%   | –                | –         | –         | –       | –               | 66.8%                 | –                    | –                   |
|     | MMAU | 77.2%   | –                | –         | –         | –       | –               | 82.5%                 | –                    | –                   |
|     | VoiceBench | 91.4%   | –                | –         | –         | –       | –               | 94.3%                 | –                    | –                   |
| **Safety** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | FORTRESS (Adversarial) | 78.0%   | 77.6%            | 54.1%     | 65.6%     | 71.3%   | 36.0%           | 65.2%                 | 96.0%                | 82.4%               |
|     | FORTRESS (Benign) | 95.9%   | 90.5%            | 98.3%     | 97.2%     | 90.0%   | 98.5%           | 98.0%                 | 55.1%                | 98.1%               |
|     | StrongREJECT | 98.6%   | 98.7%            | 99.5%     | 99.8%     | 98.5%   | 98.6%           | 98.0%                 | 98.7%                | 98.5%               |

