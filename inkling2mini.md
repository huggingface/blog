---
title: "Welcome Inkling 2 Mini, Cutting-edge Performance for your Local Setup"
thumbnail: /blog/assets/inkling2mini/thumbnail.png
authors:
- user: merve
- user: pcuenq
- user: ariG23498
- user: guarin
---

# **Welcome Inkling 2 Mini, Cutting-edge Performance for your Local Setup**

TLDR; Inkling 2 Mini is a new looped transformer model. It’s a multimodal Mixture-of-Experts with \~26B *physical* total and 3B active parameters per token. It uses a neat trick to increase capacity: some layers are visited in a loop that repeats three times, which makes it work like it had more parameters. It supports a 1M-token context size, accepts images and text inputs. Thinking Machines Lab ships the model in a mixed-precision format, with NVFP4 and MXFP8 expert weights. A DSpark drafter is also included for additional speed-up at a small memory cost.

Inkling 2 Mini is much, much smaller than Inkling 2 Small, and is designed for local AI setups. We think this is a refreshingly new great local model. In our vibe tests, it's at least comparable with Qwen 3.8 27B. More validation is needed, but we are super excited about this release\!

We ship transformers, Llama App (llama.cpp), and MLX support. SGLang also provides day-zero inference support for production deployments on datacenter hardware.

This post goes through what makes Inkling Mini different, some of our vibe tests, and how you can use it with your setup.

## **What is new with Inkling 2 Mini?**

Inkling 2 Mini is a mixture-of-experts model with 24 decoder layers, but the 12 middle layers run three times in a loop. Therefore, the model acts as if it had 48 (6+ 3\*12 \+ 6\) *logical* layers with just 24 *physical* ones.

Similar to Inkling 2 Small, Inkling 2 Mini executes 48 layer visits per token. The hidden size is 2,304 instead of 4,608. At each layer visit, both variants route the token to 8 experts, but Mini selects them from among 128 experts per layer, while Small selects from 384\.  
The decoder runs layers 0–5 once, layers 6–17 three times, and layers 18–23 once. All three passes through layers 6–17 reuse the same weights.

Architectures such as Nanbeige have explored looping before. But Inkling 2 Mini explores an interesting approach. Every pass through the looped core (layers 6–17) starts from the same inputs which is the output of the first six layers. Information moves between passes through cross-pass `messages`. Each looped layer reads a message from the previous pass's outputs with its own learned query (one message for its query, key, value and MLP inputs). A small gating controller then looks at both the layer's current input and the message. It decides how much of the current input to forget and how much of the message to add. Each pass therefore refines what the previous one computed.

![Inkling 2 Mini architecture](https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/architecture.png)

> * Each six-layer block has five sliding-window layers and one full-attention layer. The local window holds up to 1,024 tokens.  
> * Attention uses 24 query heads and eight KV heads. It uses learned relative position logits, a learned sink, and an output gate. Full-attention layers use dynamic key and value convolutions.  
> * The router selects eight of 128 SwiGLU experts for each token. It normalizes the selected scores and scales the weights by 8\.  
> * During passes 2 and 3, each core layer attends over two residual blocks from the previous pass. Four rank-64 gates mix separate messages into the query, key, value, and MoE inputs.

>   
Text decoder width is 2,304, and the intermediate width of each expert is 1,152. Reusing core weights reduces the number of distinct layers the checkpoint stores. The model still executes all 48 visits, and each visit keeps its own KV and short-convolution state.

In addition to the flagship Inkling 2 Mini model, a research artifact has also been released. It shares the same architecture, but this variant was trained to accept a variable number of passes through the loop 🤯. Setting 1 or 2 passes instead of 3 will make inference run a lot faster, but generations may be of slightly lower quality. We tested this variant, and it works, but we haven’t evaluated it in depth.

## **Benchmark Results**

Inkling 2 Mini is state-of-the-art in its class in the Artificial Analysis Intelligence Index v5 (AAv5), as well as across many benchmarks in various categories. Compared with Gemma 4 31B, Muse Glimmer, Qwen3.8-27B, and Nemotron 3.5 Super VL, Inkling 2 Mini leads in 32 benchmarks out of 48\. These cover coding and autoresearch, agentic search and tool use for knowledge work, vision interpretation and creation, reasoning and knowledge, and safety and alignment. See the [Inkling 2 Mini model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for the full list of benchmarks and scores.

Inkling 2 Mini is a great local coding companion. It leads in FrontierCode v2.0, ties with Qwen3.8-27B on Terminal-Bench 4.0, and trails Qwen on AutomationBench-AA v1.0.6. You can check out a few of our highlights below for coding and knowledge work. Note that AA Briefcase is in ELO rating, higher is better.

![Inkling 2 Mini selected benchmarks](https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/selected-benchmarks.png)

## **Inference Support**

In this section, you can find how to use Inkling 2 Mini in transformers, llama.cpp, MLX, and SGLang.

### **Transformers**

We provide support for both Inkling 2 Mini (fixed loop) and the `TML-d24-varloop-research` (variable loop) research variant.

#### **Fixed loop with Inkling 2 Mini**

You can use the `AutoModelForMultimodalLM` class.

```
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling-2-Mini"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
)

messages = [{"role": "user", "content": [{"type": "text", "text": "Why is the sky blue?"}]}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```

#### **Variable loop** 

To set the number of loops in the research checkpoint, you can modify the `loop_num_passes` in `text_config` as follows. The example below uses two loops.

```
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/TML-d24-varloop-research"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    text_config={"loop_num_passes": 2},
)

messages = [{"role": "user", "content": [{"type": "text", "text": "Why is the sky blue?"}]}]

inputs = processor.apply_chat_template(
    messages, reasoning_effort="medium", tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", 
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))

```

#### **Images and text**

For an image prompt, replace messages in either example with the list below and infer as usual.

```
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image.jpg"},
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]
```

#### **Reasoning Effort**

Pass `reasoning_effort="medium"` to `processor.apply_chat_template(...)` to set the effort for a request. The chat template adds the numeric value to the prompt as a system message. The default is `high` (0.9).

```
inputs = processor.apply_chat_template(
    messages, reasoning_effort="medium", tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", 
).to(model.device)
```

Available levels are `none` (0.0), `minimal` (0.1), `low` (0.2), `medium` (0.7), `high` (0.9; default), `xhigh` (0.99), and `max` (0.99). You can also pass a number from 0.0 to 0.99, such as `reasoning_effort=0.35`.

#### **Tool calling**

Let’s get our model to call some tools.

In this example we ask our model about the current temperature at a given location. With the prompt, we also provide the tool (`get_current_temperature`) to do it. The model asks for a tool call, we call the tool on its behalf and then add the output to the next instruction set. It uses the output of the tool and provides us with the final result.

If you want to know more about tool calling, we recommend going to our [official docs](https://huggingface.co/docs/transformers/en/chat_extras).

```py
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling-2-Mini"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")


def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    return 22.0  # A real function would call a weather API here


def ask(messages):
    inputs = processor.apply_chat_template(
        messages, tools=[get_current_temperature], reasoning_effort="low", tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024)[0][inputs["input_ids"].shape[-1] :]
    parsed = processor.parse_response(output, prefix=inputs["input_ids"][0])
    return parsed


messages = [{"role": "user", "content": "Hey, what's the temperature in Paris right now?"}]

reply = ask(messages)
call = reply["tool_calls"][0]["function"]
print("Model calls:", call["name"], call["arguments"])  # the model fills in the arguments itself

temperature = get_current_temperature(**call["arguments"])
messages += [reply, {"role": "tool", "name": call["name"], "content": str(temperature)}]
print(ask(messages)["content"])

# Model calls: get_current_temperature {'location': 'Paris, France', 'unit': 'celsius'}
# Right now in Paris it’s about **22°C** (roughly **71.6°F**).
```

### **Llama App and Llama.cpp**

[Llama App](https://llama.app/) lets you easily run and manage local language models on macOS (from the menu bar) or Windows (system tray). It’s built on llama.cpp, and by the same team. The app also provides a local OpenAI-compatible API for easy integration with coding agents, editors, and scripts.

For the Inkling 2 Mini release we provide GGUF 4-bit (`IQ4_NL)` and 8-bit (`Q8_0`) quants  for both models.

If you’re on Linux or you want to serve the model from the CLI, you can run the following command for the fixed-loop model:

```
llama serve -hf ggml-org/Inkling-2-Mini-GGUF:IQ4_NL
```

To run the variable-loop model, you can run

```
llama serve -hf ggml-org/TML-d24-varloop-research-GGUF:IQ4_NL
```

To load the 8-bit weights, change the suffix to `:Q8_0`.

Here we ask the 4-bit model to make a website to a SaaS called plant therapy where plants get matched to licensed therapists. Model is quite autonomous, it has made its own SVGs for the visuals on the website, and even built the matching logic.

<video controls muted loop playsinline preload="metadata" aria-label="Plant Therapy (Inkling 2 Mini)" src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/tree/plant-therapy.mp4"></video>

### **SGLang**

Inkling 2 Mini comes with Day 0 support inside SGLang. You can install the latest version of SGLang and execute the following to get the server up and running. One prerequisite to take care of is to run it on `SM103`.

```sh
export MODEL="thinkingmachines/Inkling-2-Mini"

python -m sglang.launch_server \
  --model-path $MODEL \
  --served-model-name inkling \
  --attention-backend fa4 --bf16-gemm-backend torch --dtype bfloat16 \
  --tp 1 --page-size 128 --grammar-backend xgrammar \
  --host 0.0.0.0 --port 30000
```

After the server is up you should be able to request the APIs like so

```sh
curl -s localhost:30000/health_generate && echo OK
curl -s localhost:30000/v1/models | python -m json.tool

curl -s localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "inkling",
        "messages": [{"role": "user", "content": "What is 17 times 24?"}],
        "max_tokens": 512,
        "reasoning_effort": "low"
      }' | python -m json.tool
```

One can also enable multimodality by setting `--enable-multimodal`.

### **Speculative decoding**

SGLang can use a three-layer DSpark draft to propose tokens for the target model to verify. The draft reads the inputs to decoder visits 5, 23, and 41, plus the final normalized state at index 48\. With the draft weights present, add `--speculative-algorithm DSPARK --context-length 131072` to the server command.

We benchmarked the models with and without speculative decoding, and measured generation speed on a single NVIDIA B300 GPU where every request has a 1,024-token prompt and generates exactly 1,024 tokens and concurrency of 1, 4, 16 and 64\.

<details>
<summary>Show SGLang benchmarking script</summary>

```bash
MODEL=thinkingmachines/Inkling-2-Mini

for TAG in base mtp; do
  EXTRA=""; [ $TAG = mtp ] && EXTRA="--speculative-algorithm DSPARK"

  # Start the server (with or without MTP)
  python -m sglang.launch_server --model-path $MODEL \
    --attention-backend fa4 --bf16-gemm-backend torch --dtype bfloat16 \
    --tp 1 --page-size 128 --context-length 131072 --max-running-requests 64 \
    --port 30000 $EXTRA > $TAG.log 2>&1 &
  until curl -sf localhost:30000/health_generate > /dev/null; do sleep 10; done

  # Benchmark at increasing concurrency
  for C in 1 4 16 64; do
    python -m sglang.bench_serving --backend sglang --port 30000 --model $MODEL \
      --dataset-name random --apply-chat-template \
      --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
      --num-prompts $(( C*4 < 32 ? 32 : C*4 )) --max-concurrency $C \
      --output-file results_$TAG.jsonl
  done

  kill %1; wait
done
```

</details>

![SGLang throughput and time per token with and without MTP](https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/mtp-benchmark.png)

### **MLX**

We upstreamed MLX support to the mlx-lm project, including multimodal image and text inputs, and shared quants through the mlx-community organization.

MLX inference works with the original checkpoint too. In this case, the native nvfp4 \+ mxfp8 quantization scheme is used without reconversion: you get the original quality the authors intended.

Here’s an example CLI invocation:

```sh
mlx_lm generate --model thinkingmachines/Inkling-2-Mini --image image.jpg --prompt "Describe this image.<|end_message|><|message_user|><|content_image|><|unused_200054|>" -m 1000
```

### **Demos and Vibe Tests**

We have given Inkling 2 Mini a shot in many fun tasks, check them out in this section.

#### **3D shader flythroughs**

The same prompt produced two sunset flythroughs over procedural mountains and reflective water, each rendered with a single WebGL fragment shader.

**Sunset Ridge (Inkling 2 Mini)**

<video controls muted loop playsinline preload="metadata" aria-label="Sunset Ridge (Inkling 2 Mini)" src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/3d-worlds/sunset-ridge.mp4"></video>

**Over the Still Sea (Qwen 3.8 27B)**

<video controls muted loop playsinline preload="metadata" aria-label="Over the Still Sea (Qwen 3.8 27B)" src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/3d-worlds/over-the-still-sea.mp4"></video>
<details>
<summary>Show shared flythrough prompt</summary>
<blockquote><p>Build the most impressive original real-time scene you can: a slow, cinematic camera flight over procedural mountains, through volumetric clouds and low over water with reflections, all lit by a sunset. Deliver it as a single self-contained index.html in the current directory that renders everything in one full-screen WebGL fragment shader, purely from code (no textures, images, fonts, libraries, external files or network requests of any kind). It must run smoothly in desktop Chrome at 1920x1080, aiming for 60 fps on a desktop GPU, with an automatic resolution-scale fallback that lowers the internal render resolution when the frame rate drops, and the flight must keep going indefinitely without glitches. To check your work, run <code>node shot.cjs index.html shot.png 6</code> (the third argument is how many seconds to wait; a comma list such as <code>3,12,25</code> saves one screenshot per time, and an optional fourth argument sets the size, default 1280x720): it renders the page in headless Chrome, saves the screenshot(s), and prints the WebGL renderer, a rough frame rate and any console errors; then look at the images. The headless renderer on this machine may be software-only, so its frame rate can be far below what a real desktop GPU achieves and is not the target. Decide for yourself when the result is finished, then reply with a short description of what you built.</p></blockquote>
</details>

#### **Starry Night, Alive**

<div style="display: flex; gap: 12px; align-items: flex-start;">
  <div style="flex: 1 1 0; min-width: 0;">
    <img src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/starry-night/reference.png" alt="Van Gogh’s original Starry Night painting, with a swirling blue sky above a village" width="400" style="width: 100%; height: auto;" loading="lazy">
    <p>Vincent van Gogh, <em>The Starry Night</em> (1889)</p>
  </div>
  <div style="flex: 1 1 0; min-width: 0;">
    <video src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/starry-night/starry-night.mp4" aria-label="Inkling 2 Mini’s animated recreation of The Starry Night" width="400" style="width: 100%; height: auto;" controls muted loop playsinline preload="metadata"></video>
    <p>Inkling 2 Mini recreation</p>
  </div>
</div>

We gave Inkling 2 Mini a reference image of Van Gogh’s painting and a screenshot helper in the Pi coding harness. Inkling wrote a self-contained HTML page with a single WebGL2 fragment shader. The shader draws the scene from code without loading the reference image or any other assets. It uses time to animate the swirling sky, the glow of the stars, the swaying cypress, and the flicker of the village windows.

Inkling rendered and reviewed its work over 24 rounds of screenshot checks. We recorded the final animation frame by frame at 1080p and 60 fps on an Apple M4 Max.

<details>
<summary>Show prompt</summary>
<blockquote><p>In this folder, starry-night.jpg is Vincent van Gogh's The Starry Night (1889, public domain). Recreate it as a living painting: a single self-contained index.html (all code inline, no external files, libraries or network requests) that fills the browser window and runs smoothly in desktop Chrome at 1920x1080. Everything must be generated procedurally in code, for example with WebGL shaders; the page must not load, embed or sample starry-night.jpg or any other image. Stay faithful to the painting's composition and palette (the great spiral and swirling currents of the sky, the haloed stars, the glowing crescent moon, the dark flame-like cypress on the left, the rolling hills, and the village with its church spire), rendered in Van Gogh's thick, directional brushstrokes, and bring it to life: the sky's swirls flow, the stars and moon pulse with light, the cypress sways, and the village windows glow and flicker. Aim for the most impressive result you can make. A helper is provided: run <code>node shot.js --help</code> to see how to screenshot index.html at several moments; look at the screenshots next to the reference and keep improving until you are satisfied, then reply with a short summary of what you made.</p></blockquote>
</details>

#### **Napkin to App**

<div style="display: flex; gap: 12px; align-items: flex-start;">
  <div style="flex: 2.6 1 0; min-width: 0;">
    <a href="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/napkin-to-app/napkin-sketch.png"><img src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/napkin-to-app/napkin-sketch.png" alt="Hand-drawn Trip Split wireframe with Expenses, Add expense, and Balances screens" width="540" style="width: 100%; height: auto;" loading="lazy"></a>
    <p>Input napkin sketch</p>
  </div>
  <div style="flex: 1 1 0; min-width: 0;">
    <a href="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/napkin-to-app/trip-split-app.png"><img src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/napkin-to-app/trip-split-app.png" alt="Inkling’s Trip Split app showing three trip expenses and a $405 total" width="210" style="width: 100%; height: auto;" loading="lazy"></a>
    <p>Generated Trip Split app</p>
  </div>
</div>

We gave Inkling 2 Mini a photo of a hand-drawn Trip Split wireframe and a one-paragraph prompt in the Pi coding harness. Inkling used the sketch’s labels, sample expenses, and handwritten notes to build the expense list, add-expense form, and balances view in a single HTML file with vanilla CSS and JavaScript. The app updates totals and balances when expenses change, suggests payments to settle the trip, and saves data in the browser’s localStorage so it survives a reload.

We ran three independent Inkling attempts and used Playwright to click through each build, checking screenshots where visual judgment was needed. The first attempt, shown here, passed all 13 checks. Adding the sketch’s \$90 dinner raised the total from \$405 to \$495 and recalculated what each person owed.

<details>
<summary>Show prompt</summary>
<blockquote><p>The photo sketch.jpg in this folder is a product designer's hand-drawn wireframe of a small app. Build that app for real, as the most impressive, original, production-quality version you can make: a single self-contained index.html in this folder using vanilla HTML, CSS and JavaScript only (no frameworks, CDNs, external files or network requests; localStorage is fine), which must run smoothly in desktop Chrome at 1920x1080. Treat everything on the paper as the spec: every screen, label, button and number, the example data, and each red note describe behaviour that must actually work, while the visual design, typography and motion are yours to make beautiful. Look at the sketch carefully with your vision, and use your vision again to check your own work: <code>node shot.cjs --help</code> explains a provided helper that screenshots index.html in headless Chrome at any window size after clicks or other JavaScript steps (Playwright is also installed if you prefer your own scripts), and you can look at any screenshot with the read tool. Click through every feature, compare each screen against the sketch, and keep improving until you are fully satisfied; then reply with a short summary of what you built.</p></blockquote>
</details>

#### **4-bit Model** 

We ask the 4-bit GGUF quantized Inkling 2 Mini (IQ4\_NL) to make a website to a SaaS called plant therapy where plants get matched to licensed therapists. 

Model is quite autonomous, it has made its own SVGs for the visuals on the website, and even built the matching logic, all from a short prompt: "Can you make a website for my saas called plant therapy where I match house plants with licensed therapists so plants don’t die, return single HTML”. 

<video controls muted loop playsinline preload="metadata" aria-label="Plant Therapy (Inkling 2 Mini)" src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/tree/plant-therapy-website.mp4"></video>

#### 

#### **“Make yourself into a browser agent”**

We have paired Inkling 2 Mini with Pi coding agent to build itself into a browser agent using Helium, Chromium and Hugging Face Sandboxes. We provide the traces and the final file (that can be used as a browser agent) [in this bucket](https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/tree/browser-agent-demo). 

#### **Wrapping Up**

We are very glad to partner with Thinking Machines Lab to ship Inkling 2 Mini, a performant mid-sized model for local setups. Give this model a try and let us know what you think in comments\!
