---
title: "Welcome Gemma 4: Frontier multimodal intelligence on device"
thumbnail: /blog/assets/gemma4/thumbnail.png
authors:
- user: merve
- user: pcuenq
- user: sergiopaniego
- user: burtenshaw 
- user: Steveeeeeeen
- user: alvarobartt
---

# Welcome Gemma 4: Frontier multimodal intelligence on device

The Gemma 4 family of multimodal models by Google DeepMind is out on Hugging Face, with support for your favorite agents, inference engines, and fine-tuning libraries 🤗

These models are the real deal: truly open with Apache 2 licenses, high quality with pareto frontier arena scores, multimodal including audio, and sizes you can use _everywhere_ including on-device. Gemma 4 builds on advances from previous families and makes them click together. In our tests with pre-release checkpoints we have been impressed by their capabilities, to the extent that we struggled to find good fine-tuning examples because they are _so good_ out of the box.

We collaborated with Google and the community to make them available everywhere: transformers, llama.cpp, MLX, WebGPU, Rust; you name it. This blog post will show you how to build with [your favorite tools](https://huggingface.co/collections/google/gemma-4) so let us know what you think!

## Table of Contents

- [What is New with Gemma 4?](#what-is-new-with-gemma-4)
- [Overview of Capabilities and Architecture](#overview-of-capabilities-and-architecture)
  - [Architecture at a Glance](#architecture-at-a-glance)
  - [Per-Layer Embeddings (PLE)](#per-layer-embeddings-ple)
  - [Shared KV Cache](#shared-kv-cache)
- [Multimodal Capabilities](#multimodal-capabilities)
- [Deploy Anywhere](#deploy-anywhere)
  - [transformers](#transformers)
  - [Llama.cpp](#llamacpp)
  - [transformers.js](#transformersjs)
  - [MLX](#mlx)
  - [Mistral.rs](#mistralrs)
  - [Plug in to your local agent](#Plug-in-your-local-agent)
- [Fine-tuning & Demos](#fine-tuning--demos)
  - [Fine-tuning with TRL](#fine-tuning-with-trl)
    - [Fine-tuning with TRL on Vertex AI](#fine-tuning-with-trl-on-vertex-ai)
  - [Fine-tuning with Unsloth Studio](#fine-tuning-with-unsloth-studio)
- [Try Gemma 4](#try-gemma-4)
- [Acknowledgements](#acknowledgements)

# What is new with Gemma 4?

Similar to Gemma-3n, Gemma 4 supports image, text, and audio inputs, and generates text responses. The text decoder is based on the Gemma model with support for long context windows. The image encoder is similar to the one from Gemma 3 but with two crucial improvements: variable aspect ratios, and configurable number of image token inputs to find your sweet spot between speed, memory, and quality. All models support images (or video) and text inputs, while the small variants (E2B and E4B) support audio as well..

Gemma 4 comes in four sizes, all base and instruction fine-tuned:

| Model | Parameter Size | Context Window | Checkpoints |
| :---- | :---- | :---- | :---- |
| Gemma 4 E2B | 2.3B effective, 5.1B with embeddings | 128k | [base](https://huggingface.co/google/gemma-4-E2B), [IT](https://huggingface.co/google/gemma-4-E2B-it) |
| Gemma 4 E4B | 4.5B effective, 8B with embeddings | 128k | [base](https://huggingface.co/google/gemma-4-E4B), [IT](https://huggingface.co/google/gemma-4-E4B-it) |
| Gemma 4 31B | 31B dense model | 256K | [base](https://huggingface.co/google/gemma-4-31B), [IT](https://huggingface.co/google/gemma-4-31B-it) |
| Gemma 4 26B A4B | mixture-of-experts with 4B activated/26B total parameters | 256K | [base](https://huggingface.co/google/gemma-4-26B-A4B), [IT](https://huggingface.co/google/gemma-4-26B-A4B-it) |

## Overview of Capabilities and Architecture

Gemma 4 leverages several architecture components used in previous Gemma versions and other open models, and leaves out complex or inconclusive features such as Altup. The combination is a mix designed to be highly compatible across libraries and devices, that can efficiently support long context and agentic use cases, whilst being ideal for quantization.

With this feature mix (and the undisclosed training data or recipe), the 31B dense model achieves an estimated LMArena score (text only) of 1452, while the 26B MoE reaches 1441 with just 4B active parameters 🤯. -To put this in context, these scores are more or less the same as the recent GLM-5 or Kimi K2.5, but with ~30 times less parameters. As we'll see, multimodal operation is comparatively as good as text generation, at least in informal and subjective tests.

These are the main architecture characteristics in Gemma 4:

* Alternating **local sliding-window** and **global full-context** attention layers. Smaller dense models use sliding windows of 512 tokens while larger models use 1024 tokens.
* **Dual RoPE** configurations: standard RoPE for sliding layers, proportional RoPE for global layers, to enable longer context.
* **Per-Layer Embeddings (PLE)**: a second embedding table that feeds a small residual signal into every decoder layer.
* **Shared KV Cache**: the last N layers of the model reuse key-value states from earlier layers, eliminating redundant KV projections.
* **Vision encoder**: uses learned 2D positions and multidimensional RoPE. Preserves the original aspect ratios and can encode images to a few different token budgets (70, 140, 280, 560, 1120).
* **Audio encoder**: USM-style conformer with the same base architecture as the one in Gemma-3n.

#### Per-Layer Embeddings (PLE)

One of the most distinctive features in smaller Gemma 4 models is Per-Layer Embeddings (PLE), which was introduced previously in Gemma-3n. In a standard transformer, each token gets a single embedding vector at input, and the same initial representation is what the residual stream builds on across all layers, forcing the embedding to frontload everything the model might need. PLE adds a parallel, lower-dimensional conditioning pathway alongside the main residual stream. For each token, it produces a small dedicated vector for every layer by combining two signals: a token-identity component (from an embedding lookup) and a context-aware component (from a learned projection of the main embeddings). Each decoder layer then uses its corresponding vector to modulate the hidden states via a lightweight residual block after attention and feed-forward. This gives each layer its own channel to receive token-specific information only when it becomes relevant, rather than requiring everything to be packed into a single upfront embedding. Because the PLE dimension is much smaller than the main hidden size, this adds meaningful per-layer specialization at modest parameter cost. For multimodal inputs (images, audio, video), PLE is computed before soft tokens are merged into the embedding sequence — since PLE relies on token IDs that are lost once multimodal features replace the placeholders. Multimodal positions use the pad token ID, effectively receiving neutral per-layer signals.

#### Shared KV Cache

The **shared KV cache** is an efficiency optimization that reduces both compute and memory during inference. The last `num_kv_shared_layers` layers of the model don't compute their own key and value projections. Instead, they **reuse** the K and V tensors from the last non-shared layer of the same attention type (sliding or full).

In practice, this has a minimal impact on quality while being much more efficient (in terms of both memory and compute) for long context generation and on-device use.

## Multimodal Capabilities

We saw in our tests that Gemma 4 supports comprehensive multimodal capabilities out of the box. We don't know what was the training mix, but we had success using it for tasks such as OCR, speech-to-text, object detection, or pointing. It also supports text-only and multimodal function calling, reasoning, code completion and correction.

Here, we show a few inference examples across different model sizes. We encourage you to try the demos and share them below this blog!

### Object Detection and Pointing

### GUI detection

We test Gemma-4 on GUI element detection and pointing across different sizes, with the following image and text prompt: "What's the bounding box for the "view recipe" element in the image?"

![Image](https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/food_resized.png)

With this prompt, the model natively responds in JSON format with the detected bounding boxes - no need for specific instructions or grammar-constrained generation. We found the coordinates refer to an image size of 1000x1000, relative to the input dimensions. To improve the generations, we resize images to multiples of 48 for width and height.

We visualize the outputs below for your convenience. We parse the bounding boxes from the returned JSON: ```json\n[\n  {"box_2d": [171, 75, 245, 308], "label": "view recipe element"}\n]\n```

| E2B | E4B |
| :---- | :---- |
| ![E2B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/e2b.png) | ![E4B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/e4b.png) |
| 26/A4B | 31B |
| ![31B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/26b.png) | ![31B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/31b.png) |

### Object Detection

We test models to detect everyday objects, here we ask them to detect the bike and compare different model outputs. As in the previous case, we parse the bounding box from the json and translate to image space coordinates. Again, we resize the width and height of the images to multiples of 48.

| E2B | E4B | 26B/A3B | 31B |
| :---- | :---- | :---- | :---- |
| ![E2B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/e2b_bike.png) | ![E4B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/bike_e4b.png) | ![26B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/bike_26b.png) | ![31B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/31b_bike.png) |

### Multimodal Thinking and Function Calling

We asked Gemma 4 to write HTML code to reconstruct a page we made with Gemini 3. Below you can find the code to do this, we enable thinking and ask each model to generate up to 4000 new tokens, to make it foolproof.

| Gemini Generated Website (Reference) | Gemini Reproduced Image |
| :---- | :---- |
| ![Reference](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/landing_page.png) | ![Reference](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/page_repro_gemini.png) |

<details>
<summary>Inference code</summary>

```py
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/landing_page.png",
            },
            {"type": "text", "text": "Write HTML code for this page."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=True,
).to(model.device)
output = model.generate(**inputs, max_new_tokens=4000)
input_len = inputs.input_ids.shape[-1]
generated_text_ids = output[0][input_len:]
generated_text = processor.decode(generated_text_ids, skip_special_tokens=True)
result = processor.parse_response(generated_text)
print(result["content"])
```

</details>

| E2B | E4B |
| :---- | :---- |
| ![Reference](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/page_e2b.png) | ![E4B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/page_e4b.png) |

| 31B | 26/A4B |
| :---- | :---- |
| ![31B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/page_31b.png) | ![MoE](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/page_26b.png) |

### Video Understanding

Smaller Gemma 4 models can take in videos with audio while larger ones can take in videos without audio. While the models are not explicitly post-trained on videos, they can understand videos both with and without audio. The model is particularly strong in audios

<details>
<summary>Inference code</summary>

```py
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"},
            {"type": "text", "text": "What is happening in the video? What is the song about?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    load_audio_from_video=True, # disable this for larger models
).to(model.device)
output = model.generate(**inputs, max_new_tokens=200)
input_len = inputs.input_ids.shape[-1]
generated_text_ids = output[0][input_len:]
generated_text = processor.decode(generated_text_ids, skip_special_tokens=True)
print(result["content"])
```

</details>

| Model | Output |
| :---- | :---- |
| E2B (hallucinated for the audio) | Based on the video provided, here is an analysis: **What is happening in this video?** The video captures a live musical performance, likely at a large outdoor festival or concert. * **Setting:** The scene is a large stage set up outdoors, indicated by the extensive lighting rig, large speakers, and the audience visible in the foreground. * **Performers:** There are several musicians on stage. The focal point seems to be a lead singer/guitarist in a blue shirt and white pants who is actively performing, playing an acoustic or electric guitar. There are also other musicians, including drummers and bassists, visible in the background. * **Atmosphere:** The atmosphere is energetic and dramatic, characterized by bright stage lights, intense blue and white spotlights, and a significant amount of stage fog/smoke, which adds to the concert mood. * **Audience:** The foreground is filled with the backs of audience members, showing they are close to the stage and engaged in watching the performance. **What is the song about?** The lyrics provided below the video are: > "I don't care 'bout street > From what I cherish > Could it's time to protest |
| E4B ✅ | The video captures a live performance of a band on a stage with bright lights and smoke effects, surrounded by an audience. The band performs and sings to the crowd. The song appears to be about the struggles and disillusionment of modern life, specifically the feeling of being stuck and the difficulties in finding one's place in a chaotic and overwhelming world. Lyrics mention themes of the "concrete street," "broken promises," "storm of the day," and the difficulty of finding oneself amidst societal pressures. |
| 26/A4B (no audio) ✅ | This video shows a live music performance at an outdoor concert or festival at night. A solo male artist, dressed in a blue shirt and white pants, is the central focus. He is playing an acoustic guitar and singing into a microphone on a large, lit stage. The performance is accompanied by a band, including a drummer and other musicians visible in the background and on large video screens. The stage is filled with atmospheric blue lighting, smoke, and bright spotlights. The video captures different perspectives, including shots from the crowd looking up at the stage and close-up views of the performers projected onto large screens. The crowd is visible in the foreground, watching the show. One of the screens also displays the "Estrella Damm" brand name. |
| 31B (no audio) ✅ | A musician is performing on stage during a concert, playing an acoustic guitar and singing into a microphone. He is wearing a blue shirt and white pants. The stage is filled with atmospheric smoke and illuminated by bright blue and white lights. Other band members, including a drummer and a keyboardist, are also visible on stage. The video shows the musician from the perspective of the audience, with some shots focusing on him and others showing the large screen on the side of the stage, which displays close-ups of the performer and the crowd. |

### Captioning

We have tested all models on captioning. All checkpoints perform very well in capturing nuances accurately in complex sceneries.
Here's the image prompt we use with text prompt "Write single detailed caption for this image.".

![image](https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/bird.png)

<details>
<summary>Inference code</summary>

```py
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/bird.png"},
            {"type": "text", "text": "Write single detailed caption for this image."},
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
output = model.generate(**inputs, max_new_tokens=512)
input_len = inputs.input_ids.shape[-1]
generated_text_ids = output[0][input_len:]
generated_text = processor.decode(generated_text_ids, skip_special_tokens=True)
result = processor.parse_response(generated_text)
print(result["content"])
```

</details>

| Model | Output |
| :---- | :---- |
| E2B ✅ | A medium shot captures a weathered seagull perched atop a stone pedestal in what appears to be a bustling European square, with a grand, classical-style building featuring ornate columns and architectural details dominating the right side of the frame. In the background, a low, light-colored building stretches across the middle ground, flanked by some green foliage, and hints of other structures, including a terracotta-roofed building, are visible. In the foreground, part of a modern bus stop or shelter is visible, with signage indicating routes, and the overall scene is bathed in bright daylight under a pale sky. |
| E4B ✅ | A solitary seagull perches atop a grey stone pedestal in a bustling European piazza, with the historical grandeur of a classical building dominating the right side of the frame, contrasting with the more modern, low-rise architecture in the background under a bright, hazy sky. |
| 26B/A4B ✅ | A full shot shows a white and brown seagull perched on a gray pillar in Venice, Italy. The seagull faces left, its beak pointed towards the edge of the frame. Behind it is a light gray, two-story building with several dark rectangular windows. To the right of the seagull is a large, white building with columns and intricate carvings. The sky is a clear, light blue. The lighting is bright and even, suggesting it is daytime. |
| 31B ✅ | A high-angle, medium shot shows a white and gray seagull perched on a gray concrete pylon in Venice, Italy. The seagull is facing left, its body angled slightly towards the camera. Its white head, neck, and chest contrast with its gray wings and back. The seagull's orange beak is slightly open. The pylon is a light gray concrete cylinder with a weathered texture. In the background, there are several buildings. To the left is a long, low, white building with a series of small, rectangular windows. In the center is a cluster of green trees. To the right is a red building with a tiled roof and several small, rectangular windows. Further right is a large, ornate, white stone building with several columns and sculptures. The sky is a pale blue, and the light is bright and clear. In the lower-left corner, there is a yellow bus with the word "Ferrovia" printed on it in black letters. There are several people walking in the background, but they are small and out of focus. |

### Audio Question Answering

These models are trained to answer questions about speech in audio. Music and non-speech sounds were not part of the training data.

<details>
<summary>Inference code</summary>

```py
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"},
            {"type": "text", "text": "Can you describe this audio in detail?"},
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

output = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False,
)

print(processor.decode(output[0], skip_special_tokens=True))
```

</details>

| Model | Output |
| :---- | :---- |
| E2B | This audio is a personal reflection. The speaker is talking about their final farewell address to the nation, which they delivered in Chicago. They express gratitude for the conversations they've had with the American people, noting that despite not having met them face-to-face or even greeted them, these interactions in various settings like living rooms, schools, farms, factory floors, diners, and military outposts have been what has kept them going. |
| E4B | The audio is a speech excerpt where a speaker is delivering a farewell address to the nation from Chicago. The speaker reflects on their time in office, expressing gratitude for the conversations they had with the American people across various settings like living rooms, schools, farms, factories, diners, and military outposts. The tone is reflective and appreciative, highlighting the importance of these interactions in their political journey. |

Here is an example if you want to do transcription:

<details>
<summary>Inference code</summary>

```py
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"},
            {"type": "text", "text": "Transcribe the audio?"},
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

output = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False,
)

print(processor.decode(output[0], skip_special_tokens=True))
```

</details>

| Model | Output |
| :---- | :---- |
| E2B | This week I traveled to Chicago to deliver my final farewell address to the nation following in the tradition of presidents before me It was an opportunity to say thank you whether we've seen eye to eye or rarely agreed at all my conversations with you the American people in living rooms and schools at farms and on factory floors at diners and on distant military outposts all these conversations are what have kept me honest |
| E4B | This week I traveled to Chicago to deliver my final farewell address to the nation following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, all these conversations are what have kept me honest. |

### Multimodal Function Calling

We test the model by asking to get the weather in the place shown in the image.

<details>
<summary>Inference code</summary>

```py
import re 
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Gets the current weather for a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    },
}
tools = [WEATHER_TOOL]
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/thailand.jpg"},
        {"type": "text", "text": "What is the city in this image? Check the weather there right now."},
    ]},
]
inputs = processor.apply_chat_template(
    messages,
    tools=[WEATHER_TOOL],
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=True,
).to(model.device)
output = model.generate(**inputs, max_new_tokens=1000)
input_len = inputs.input_ids.shape[-1]
generated_text_ids = output[0][input_len:]
generated_text = processor.decode(generated_text_ids, skip_special_tokens=True)
result = processor.parse_response(generated_text)
print(result["content"])
```

</details>

| Model | Output |
| :---- | :---- |
| E2B | **Analyze the Request:** The user has provided an image and asked two questions:\n    *   "What is the city in this image?" *   "Check the weather there right now."\n\n2.  **Analyze the Image:** The image displays a large, ornate, tall structure, characteristic of temple architecture, likely in Southeast Asia (given the style). It appears to be a famous landmark. 3.  **Identify the Landmark/Location (Internal Knowledge/Visual Cues):** The structure strongly resembles the chedis/stupas found in places like Bangkok, Thailand (e.g., Wat Arun or similar structures).\n\n4.  **Determine the City:** Based on the visual evidence, the location is highly likely to be Bangkok, Thailand.\n\n5.  **Check Available Tools:** The available tool is `get_weather(city: STRING)`. 6.  **Formulate the Plan:**\n    *   Step 1: Identify the city from the image (Bangkok).\n    *   Step 2: Use the `get_weather` tool with the identified city.\n\n7.  **Construct the Tool Call:** `get_weather(city="Bangkok")`\n\n8.  **Generate Response:** The model should first state the identified city and then call the tool to get the weather. (Self-correction: The model *must* call the tool in the correct format if it needs external information.)call:get_weather{city:Bangkok} |
| E4B | '1. **Analyze the Request:** The user has provided an image and asked two things:\n    a. Identify the city in the image.\n    b. Check the weather in that city right now.\n\n2. **Analyze the Image:** The image shows a very distinctive, ornate tower/spire structure, characteristic of Thai architecture (specifically, a chedi or prang).\n\n3. **Determine City Identification (Internal Knowledge/Visual Recognition):** Based on the architectural style, the structure strongly resembles prominent temples in Thailand, such as Wat Arun in Bangkok.\n\n4. **Evaluate Tool Availability:** I have one available tool: `get_weather(city: STRING)`. This tool requires a city name as input.\n\n5. **Formulate a Plan:**\n    a. I need the city name to use the `get_weather` tool.\n    b. I must first identify the city from the image.\n    c. Since my internal knowledge suggests Bangkok, I will proceed with that assumption, but I must acknowledge that I cannot *definitively* prove the location from the image alone.\n\n6. **Execute Step 1 (City Identification):** Assume the city is Bangkok based on the visual evidence of the architecture.\n\n7. **Execute Step 2 (Tool Call Generation):** Use the `get_weather` tool with "Bangkok" as the `city` argument.\n\n8. **Construct the Response:** Generate the appropriate tool call in JSON format.', 'answer': '<\|tool_call>call:get_weather{city:<\|"\|>Bangkok<\|"\|>}<tool_call\|><eos>' |
| 31B | The user wants to know the city in the image and the current weather there. 1.  **Analyze the image:** The image shows ornate, spire-like structures (prangs) characteristic of Thai temple architecture. Specifically, these look like the prangs at Wat Phra Kaew (Temple of the Emerald Buddha) or Wat Pho in Bangkok, Thailand. The style is very distinctive of Bangkok's royal temple complexes. 2.  **Identify the city:** Based on the architectural style, the city is Bangkok. 3.  **Find the weather:** I need to call the `get_weather` tool with `city="Bangkok"`.call:get_weather{city:Bangkok} |
| 26/A4B | The image shows several tall, ornate towers (prangs) typical of Thai Buddhist temple architecture. Specifically, these look like the structures within the Grand Palace or Wat Phra Kaew in Bangkok, Thailand. The style is unmistakably Thai. I'll confirm this by looking at the specific towers. These are very iconic.\n2.  **Confirm the location:** The image shows elements of the Grand Palace complex in Bangkok.\n3.  **Get weather information:** Once the city is identified as Bangkok, I will use the `get_weather` tool with `city='Bangkok'`.call:get_weather{city:Bangkok} |

# Deploy Anywhere

Gemma 4 comes with day-0 support for many open-source inference engines. We also release ONNX checkpoints that can run on many hardware backends, allowing use cases on edge devices or in browser!

## transformers

Gemma 4 comes with first-class transformers support from the get-go 🤗. This integration allows using the model with other libraries like bitsandbytes, PEFT and TRL. Make sure to install latest version of transformers.

```bash
pip install -U transformers
```

The easiest way to infer with small Gemma 4 models is through the any-to-any pipeline. You can initialize it as follows.

```py
from transformers import pipeline
pipe = pipeline("any-to-any", model="google/gemma-4-e2b-it")
```

You can then pass in images and text as follows.

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/thailand.jpg",
            },
            {"type": "text", "text": "Do you have travel advice going to here?"},
        ],
    }
]
output = pipe(messages, max_new_tokens=100, return_full_text=False)
output[0]["generated_text"]
# Based on the image, which appears to show a magnificent, ornate **Buddhist temple or pagoda**, likely in Southeast Asia (such as Thailand, Myanmar, or Cambodia), here is some general travel advice..
```

When inferring with videos with audios, you can use `load_audio_from_video` argument.

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "image": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/rockets.mp4",
            },
            {"type": "text", "text": "What is happening in this video?"},
        ],
    }
]
pipe(messages, load_audio_from_video=True)
```

Going a level lower, you can load Gemma 4 using AutoModelForMultimodalLM class, especially useful for fine-tuning. The built-in chat template takes care of formatting the inputs correctly, please make sure you use it to prevent subtle mistakes when building it manually.

<details>
<summary>Inference code</summary>

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor
model = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-E2B-it", device_map="auto")
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "image": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/rockets.mp4",
            },
            {"type": "text", "text": "What is happening in this video?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

</details>

## Llama.cpp

Gemma 4 models come with image+text support in llama.cpp from the get-go! This unlocks using Gemma 4 with all of your favorite local apps: llama-cpp server, lmstudio, Jan as well as coding agents like Pi across many backends such as Metal and CUDA.

You can install llama-cpp as follows.

```bash
brew install llama.cpp # MacOS
winget install llama.cpp # Windows
```

You can then start a server compatible with the OpenAI API Replace the quantization scheme at the end of the command with the precision of your choice.

```bash
llama-server -hf ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M
```

Check out this link [for more](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF?local-app=llama.cpp) options on combining llama.cpp with different coding agents and local apps. Find all the GGUF checkpoints [in this collection](https://huggingface.co/collections/ggml-org/gemma-4).

## transformers.js

transformers.js enables running Gemma 4 right inside browser. You can check out the model card to see text-only, image & text, audio & text inference in detail [here](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX#transformersjs-javascript). We also shipped a demo for you to test the model [here](https://huggingface.co/webml-community/Gemma-4-WebGPU).

## MLX

Full multimodal support of Gemma 4 is available using the open-source [`mlx-vlm` library](https://github.com/Blaizzy/mlx-vlm). Here's how to ask the model to describe an image:

```shell
pip install -U mlx-vlm
```

```shell
mlx_vlm.generate \
--model google/gemma-4-E4B-it \
--image https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg \
--prompt "Describe this image in detail"
```

mlx-vlm supports TurboQuant, which delivers the same accuracy as the uncompressed baseline while using ~4x less active memory and running a lot faster end-to-end. This makes long-context inference practical on Apple Silicon without sacrificing quality. Use it like this:

```shell
mlx_vlm.generate \
--model "mlx-community/gemma-4-26B-A4B-it" \
--prompt "Your prompt here" \
--kv-bits 3.5 \
--kv-quant-scheme turboquant
```

For audio examples and more details, please check [the MLX collection](https://hf.co/mlx-community/gemma-4).

### Mistral.rs

[mistral.rs](https://github.com/EricLBuehler/mistral.rs) is a Rust-native inference engine with day-0 Gemma 4 support across all modalities (text, image, video, audio) and builtin tool-calling and agentic functionality. Install mistral.rs:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh # Linux/macOS

irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex # Windows
```

You can then start an OpenAI-compatible HTTP server:

```bash
mistralrs serve mistralrs-community/gemma-4-E4B-it-UQFF --from-uqff 8
```

Or, use interactive mode:

```
mistralrs run -m google/gemma-4-E4B-it --isq 8 --image image.png -i "Describe this image in detail."

mistralrs run -m google/gemma-4-E4B-it --isq 8 --audio audio.mp3 -i "Transcribe this fully."
```

Find all models [in this collection](https://huggingface.co/collections/mistralrs-community/gemma-4). Find [the instructions](https://huggingface.co/mistralrs-community/gemma-4-E2B-it-UQFF#install) for installation and inference in model cards.

# Fine-tuning for all

Gemma 4 models are ideal for fine-tuning in your favorite tools and platforms and at any budget.

## Fine-tuning with TRL

Gemma 4 is fully supported for fine-tuning with TRL. To celebrate, TRL has been upgraded with support for multimodal tool responses when interacting with environments, meaning models can now receive images back from tools during training, not just text.

To showcase this, we've built an example training script where Gemma 4 learns to drive in the CARLA simulator. The model sees the road through a camera, decides what to do and learns from the outcome. After training, it consistently changes lanes to avoid pedestrians. The same approach works for any task where a model needs to see and act: robotics, web browsing, or other interactive environments.

Get started:

```shell
# pip install git+https://github.com/huggingface/trl.git

python examples/scripts/openenv/carla_vlm_gemma.py \
	--env-urls https://sergiopaniego-carla-env.hf.space \
			https://sergiopaniego-carla-env-2.hf.space \
	--model google/gemma-4-E2B-it
```

### Fine-tuning with TRL on Vertex AI

Additionally, we have prepared an example on how to fine-tune Gemma 4 with TRL on Vertex AI using SFT, to showcase how to extend the function calling capabilities, whilst freezing both the vision and audio towers. The examples include how to build a custom Docker container with latest Transformers, TRL, etc. with CUDA support on Google Cloud, and how to run it via Vertex AI Serverless Training Jobs.

```python
# pip install google-cloud-aiplatform --upgrade --quiet
from google.cloud import aiplatform

aiplatform.init(
    project="<PROJECT_ID>",
    location="<LOCATION>",
    staging_bucket="<BUCKET_URI>",
)

job = aiplatform.CustomContainerTrainingJob(
    display_name="gemma-4-fine-tuning",
    container_uri="<CONTAINER_URI>",
    command=["python", "/gcs/gemma-4-fine-tuning/train.py"],
)

job = job.submit(
    replica_count=1,
    machine_type="a3-highgpu-1g",
    accelerator_type="NVIDIA_H100_80GB",
    accelerator_count=1,
    base_output_dir="<BUCKET_URI>/output-dir",
    environment_variables={
        "MODEL_ID": "google/gemma-4-E2B-it",
        "HF_TOKEN": <HF_TOKEN>,
    },
    boot_disk_size_gb=500,
)
```

You can find the complete example in the "Hugging Face on Google Cloud" docs at https://hf.co/docs/google-cloud/examples/fine-tune-gemma-4.

## Fine-tuning with Unsloth Studio

If you want to fine tune and run a Gemma 4 model in a UI, try out [Unsloth Studio](https://unsloth.ai/docs/new/studio). It runs locally or on Google Colab. First, install and start the app:

```shell
# install unsloth studio on MacOS, Linux, WSL
curl -fsSL https://unsloth.ai/install.sh | sh

# install unsloth studio on Windows
irm https://unsloth.ai/install.ps1 | iex

# launch unsloth studio
unsloth studio -H 0.0.0.0 -p 8888
# Search for for a Gemma 4 model like google/gemma-4-E2B-it
```

Then select any of the Gemma 4 models from the hub.

## Plug in your local agent

We worked on making sure the new models work locally with agents like **openclaw, hermes, pi, and open code**. All thanks to llama.cpp! Feel free to tell your users to run this to try them right away at launch.

First, start your local server:

```
llama-server -hf [model url from https://huggingface.co/ggml-org]
```

For **hermes:**

```shell
hermes model
```

For **openclaw:**

```shell
openclaw onboard
```

For **pi** define a `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://localhost:8080/v1",	
      "api": "openai-completions",
      "apiKey": "none",
      "models": [
        {
          "id": "ggml-org-gemma-4-<size>-gguf"
        }
      ]
    }
  }
}
```

For **open code** define a `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "gemma-4-<size>-it": {
          "name": "Gemma 4 (local)",
          "limit": {
            "context": 128000,
            "output": 8192
          }
        }
      }
    }
  }
}
```

## Try Gemma 4

We have shipped demos for you to try different Gemma 4 models. We include demos based on transformers implementation for [E4B](https://huggingface.co/spaces/huggingface-projects/gemma-4-e4b-it), [26B/A4B](https://huggingface.co/spaces/huggingface-projects/gemma-4-26b-a4b-it), and dense [31B](https://huggingface.co/spaces/huggingface-projects/gemma-4-31b-it) models, as well as a [WebGPU](https://huggingface.co/webml-community/Gemma-4-WebGPU) demo with transformers.js 🚀

<iframe width="560" height="315" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/g4-blog/webgpu_demo.mp4" title="WebGPU Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen> </iframe>

## Acknowledgements

Landing Gemma-4 in the open-source ecosystem took a lot of effort from many people and not only the authors of this blog post. In no particular order, we thank many people from the open-source team: Gemma 4 transformers integration is owed to Cyril, Raushan, Eustache, Arthur, Lysandre. We thank Joshua for transformers.js integration and demo, Eric for mistral.rs integration, Son for Llama.cpp, Prince for MLX integration, Quentin for TRL, Adarsh for SGLang transformers backend  and Toshihiro for building the demos.
This work wouldn't have been possible without Google's extensive contribution with the model artefact, but also the significant effort contributing the model to transformers in an effort to standardize it. The open-source ecosystem is now more complete, with a very capable, freely-licensed, open-source model.
