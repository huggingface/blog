---
title: "Gemma 3n fully available in the open-source ecosystem!" 
thumbnail: /blog/assets/gemma3n/thumbnail.png
authors:
- user: ariG23498
- user: pcuenq
- user: sergiopaniego
- user: reach-vb
- user: FL33TW00D-HF
- user: Xenova
---

# Gemma 3n fully available in the open-source ecosystem!

Gemma 3n was announced as a *preview* during Google I/O. The on-device community got really excited, because this is a model designed from the ground up to **run locally** on your hardware. On top of that, it’s natively **multimodal**, supporting image, text, audio, and video inputs 🤯

Today, Gemma 3n is finally available on the most used open source libraries. This includes transformers & timm, MLX, llama.cpp (text inputs), transformers.js, ollama, Google AI Edge and others.

This post quickly goes through practical snippets to demonstrate how to use the model with these libraries, and how easy it is to fine-tune it for other domains.

## Models released today

> [!NOTE]  
> Here is the [Gemma 3n Release Collection](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4)

Two model sizes have been released today, with two variants (base and instruct) each. The model names follow a non-standard nomenclature: they are called `gemma-3n-E2B` and `gemma-3n-E4B`. The `E` preceding the parameter count stands for `Effective`. Their actual parameter counts are `5B` and `8B`, respectively, but thanks to improvements in memory efficiency, they manage to only need 2B and 4B in VRAM (GPU memory). 

These models, therefore, behave like 2B and 4B in terms of hardware support, but they punch over 2B/4B in terms of quality. The `E2B` model can run in as little as 2GB of GPU RAM, while `E4B` can run with just 3GB of GPU RAM.

| Size | Base | Instruct |
| :---- | :---- | :---- |
| 2B | [google/gemma-3n-e2b](hf.co/google/gemma-3n-e2b) | [google/gemma-3n-e2b-it](google/gemma-3n-e2b-it) |
| 4B | [google/gemma-3n-e4b](hf.co/google/gemma-3n-e4b) | [google/gemma-3n-e4b-it](google/gemma-3n-e4b-it) |

## Details of the models

In addition to the language decoder, Gemma 3n uses an **audio encoder** and a **vision encoder**. We highlight their main features below, and describe how they have been added to `transformers` and `timm`, as they are the reference for other implementations.

* **Vision Encoder (MobileNet-V5).** Gemma 3n uses a new version of MobileNet: MobileNet-v5-300, which has been added to the new version of `timm` released today.  
  * Features 300M parameters.  
  * Supports resolutions of `256x256`, `512x512`, and `768x768`.  
  * Achieves 60 FPS on Google Pixel, outperforming ViT Giant while using 3x fewer parameters.  
* **Audio Encoder:**  
  * Based on the Universal Speech Model (USM).  
  * Processes audio in `160ms` chunks.  
  * Enables speech-to-text and translation functionalities (e.g., English to Spanish/French).  
* **Gemma 3n Architecture and Language Model.** The architecture itself has been added to the new version of `transformers` released today. This implementation branches out to `timm` for image encoding, so there’s a single reference implementation of the MobileNet architecture.

### Architecture Highlights

* **MatFormer Architecture:**  
  * A nested transformer design, similar to Matryoshka embeddings, allows for various subsets of layers to be extracted as if they were individual models.  
  * E2B and E4B were trained together, configuring E2B as a sub-model of E4B.  
  * Users can “mix and match” layers, depending on their hardware characteristics and memory budget.  
* **Per-Layer Embeddings (PLE):** Reduces accelerator memory usage by offloading embeddings to the CPU. This is the reason why the E2B model, while having 5B real parameters, takes about as much GPU memory as if it was a 2B parameter model.  
* **KV Cache Sharing:** Accelerates long-context processing for audio and video, achieving 2x faster prefill compared to Gemma 3 4B.

### Performance & Benchmarks:

* **LMArena Score:** E4B is the first sub-10B model to achieve a score of 1300+.  
* **MMLU Scores:** Gemma 3n shows competitive performance across various sizes (E4B, E2B, and several Mix-n-Match configurations).  
* **Multilingual Support:** Supports 140 languages for text and 35 languages for multimodal interactions.

## Demo Space

![GIF of Hugging Face Space for Gemma 3n](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gemma3n/gemma3n.gif)

The easiest way to vibe check the model is with the dedicated Hugging Face Space for the model. You can try out different prompts, with different modalities here.

[https://huggingface.co/spaces/huggingface-projects/gemma-3n-E4B-it](https://huggingface.co/spaces/huggingface-projects/gemma-3n-E4B-it)

## Inference with transformers

Install the latest version of timm (for the vision encoder) and transformers to run inference, or if you want to fine tune it.

```shell
pip install -U -q timm
pip install -U -q transformers
```

### Inference with pipeline

The easiest way to start using Gemma 3n is by using the pipeline abstraction in transformers:

```py
import torch
from transformers import pipeline

pipe = pipeline(
   "image-text-to-text",
   model="google/gemma-3n-E4B-it", # "google/gemma-3n-E4B-it"
   device="cuda",
   torch_dtype=torch.bfloat16
)

messages = [
   {
       "role": "user",
       "content": [
           {"type": "image", "url": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg"},
           {"type": "text", "text": "Describe this image"}
       ]
   }
]

output = pipe(text=messages, max_new_tokens=32)
print(output[0]["generated_text"][-1]["content"])
```

Output:

```
The image shows a futuristic, sleek aircraft soaring through the sky. It's designed with a distinctive, almost alien aesthetic, featuring a wide body and large
```

### Detailed inference with transformers

Initialize the model and the processor from the Hub, and write the `model_generation` function that takes care of processing the prompts and running the inference on the model.

```py
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "google/gemma-3n-e4b-it" # google/gemma-3n-e2b-it
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)

def model_generation(model, messages):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_len = inputs["input_ids"].shape[-1]

    inputs = inputs.to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=32, disable_compile=False)
        generation = generation[:, input_len:]

    decoded = processor.batch_decode(generation, skip_special_tokens=True)
    print(decoded[0])
```

Since the model supports all modalities as inputs, here's a brief code explanation of how you can use them via transformers.

#### Text only

```py
# Text Only

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the capital of France?"}
        ]
    }
]
model_generation(model, messages)
```

Output:

```
The capital of France is **Paris**. 
```

#### Interleaved with Audio

```py
# Interleaved with Audio

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in English:"},
            {"type": "audio", "audio": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/speech.wav"},
        ]
    }
]
model_generation(model, messages)
```

Output:

```
Send a text to Mike. I'll be home late tomorrow.
```

#### Interleaved with Image/Video

Support for videos is done as a collection of frames of images

```py
# Interleaved with Image

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]
model_generation(model, messages)
```

Output:

```
The image shows a futuristic, sleek, white airplane against a backdrop of a clear blue sky transitioning into a cloudy, hazy landscape below. The airplane is tilted at
```

## Inference with MLX

Gemma 3n comes with day 0 support for MLX across all 3 modalities. Make sure to upgrade your mlx-vlm installation.

```
pip install -u mlx-vlm
```

Get started with vision:

```py

python -m mlx_vlm.generate --model google/gemma-3n-E4B-it --max-tokens 100 --temperature 0.5 --prompt "Describe this image in detail." --image https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg
```

And audio:

```py
python -m mlx_vlm.generate --model google/gemma-3n-E4B-it --max-tokens 100 --temperature 0.0 --prompt "Transcribe the following speech segment in English:" --audio https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/audio-samples/jfk.wav
```

### Inference with llama.cpp

In addition to MLX, Gemma 3n (text only) works out-of the box with llama.cpp. Make sure to install llama.cpp/ Ollama from source.

Check out the Installation instruction for llama.cpp here: https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md

You can run it as:

```shell
llama-server -hf ggml-org/gemma-3n-E4B-it-GGUF:Q8_0
```

### Inference with Transformers.js and ONNXRuntime

Finally, we are also releasing ONNX weights for the [gemma-3n-E2B-it](https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX) model variant, enabling flexible deployment across diverse runtimes and platforms. For JavaScript developers, Gemma3n has been integrated into [Transformers.js](https://github.com/huggingface/transformers.js) and is available as of version [3.6.0](https://github.com/huggingface/transformers.js/releases/tag/3.6.0).

For more information on how to run the model with these libraries, check out the usage section in the [model card](https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX#usage).

## Fine Tune in a Free Google Colab

Given the size of the model, it’s pretty convenient to fine-tune it for specific downstream tasks across modalities. To make it easier for you to fine-tune the model, we’ve created a simple notebook that allows you to experiment on a free [Google Colab](https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/fine_tune_gemma3n_on_t4.ipynb)\!

We also provide a dedicated [notebook for fine-tuning on audio tasks](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/notebooks/fine_tune_gemma3n_on_audio_ipynb.ipynb), so you can easily adapt the model to your own speech datasets and benchmarks\!

## Hugging Face Gemma Recipes

With this release we also introduce the [Hugging Face Gemma Recipes](https://github.com/huggingface/huggingface-gemma-recipes) repository. One will find `notebooks` and `scripts` to run the models and fine tune them.

We would love for you to use the Gemma family of models and add more recipes to it\! Feel free to open Issues and create Pull Requests to the repository.

## Conclusion

We are always excited to host Google and their Gemma family of models. We hope the community will get together and make the most of these models. Multimodal, small sized, and highly capable, make a great model release\!

If you want to discuss the models in more detail, go ahead and start a discussion right below this blog post. We will be more than happy to help!