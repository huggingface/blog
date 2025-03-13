---
title: "Welcome Gemma 3: Google's all new multimodal, multilingual, long context open LLM" 
thumbnail: /blog/assets/gemma3/thumbnail.png
authors:
- user: ariG23498
- user: merve
- user: pcuenq
- user: reach-vb
---

# Welcome Gemma 3: Google's all new multimodal, multilingual, long context open LLM

## TL;DR

Today Google releases [**Gemma 3**](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d), a new iteration of their Gemma family of models. The models range from 1B to 27B parameters, have a context window up to 128k tokens, can accept images and text, and support 140+ languages.

Try out Gemma 3 now 👉🏻 [Gemma 3 Space](https://huggingface.co/spaces/huggingface-projects/gemma-3-12b-it)

|  | Gemma 2 | Gemma 3 |
| :---- | :---- | :---- |
| Size Variants | <li>2B <li>9B <li>27B | <li>1B <li>4B <li>12B <li>27B |
| Context Window Length | 8k | <li>32k (1B) <li>128k (4B, 12B, 27B) |
| Multimodality (Images and Text) | ❌ | <li>❌ (1B) <li>✅ (4B, 12B, 27B) |
| Multilingual Support | – | English (1B) +140 languages (4B, 12B, 27B) |

All the [models are on the Hub](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) and tightly integrated with the Hugging Face ecosystem.

> *Both pre-trained and instruction tuned models are released. Gemma-3-4B-IT beats Gemma-2-27B IT, while Gemma-3-27B-IT beats Gemini 1.5-Pro across benchmarks*.

| ![pareto graph](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gemma3/lmsys.png) |
| :---- |
| Gemma 3 27B is in the pareto sweet spot (Source: [Gemma3 Tech Report](https://goo.gle/Gemma3Report)) |

## What is Gemma 3?

[Gemma 3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) is Google's latest iteration of open weight LLMs. It comes in four sizes, **1 billion**, **4 billion**, **12 billion**, and **27 billion** parameters with *base (pre-trained)* and *instruction-tuned* versions. Gemma 3 goes **multimodal** ! The 4, 12, and 27 billion parameter models can process both **images** and **text**, while the 1B variant is *text only*.

The input context window length has been increased from Gemma 2’s 8k to **32k** for the 1B variants, and **128k** for all others. As is the case with other VLMs (vision-language models), Gemma 3 generates text in response to the user inputs, which may consist of text and, optionally, images. Example uses include question answering, analyzing image content, summarizing documents, etc.

| Pre Trained | Instruction Tuned | Multimodal | Multilingual | Input Context Window |
| :---- | :---- | :---- | :---- | :---- |
| [gemma-3-1b-pt](http://hf.co/google/gemma-3-1b-pt) | [gemma-3-1b-it](http://hf.co/google/gemma-3-1b-it) | ❌ | English | 32K |
| [gemma-3-4b-pt](http://hf.co/google/gemma-3-4b-pt) | [gemma-3-4b-it](http://hf.co/google/gemma-3-4b-it) | ✅ | +140 languages | 128K |
| [gemma-3-12b-pt](http://hf.co/google/gemma-3-12b-pt) | [gemma-3-12b-it](http://hf.co/google/gemma-3-12b-it) | ✅ | +140 languages | 128K |
| [gemma-3-27b-pt](http://hf.co/google/gemma-3-27b-pt) | [gemma-3-27b-it](http://hf.co/google/gemma-3-27b-it) | ✅ | +140 languages | 128K |

> [!NOTE]  
> While these are multimodal models, one can use it as a *text only* model (as an LLM) without loading the vision encoder in memory. We will talk about this in more detail later in the inference section.

## Technical Enhancements in Gemma 3

The three core enhancements in Gemma 3 over Gemma 2 are:

* Longer context length  
* Multimodality  
* Multilinguality

In this section, we will cover the technical details that lead to these enhancements. It is interesting to start with the knowledge of Gemma 2 and explore what was necessary to make these models even better. This exercise will help you think like the Gemma team and appreciate the details!

### Longer Context Length

Scaling context length to 128k tokens could be achieved efficiently without training models from scratch. Instead, models are pretrained with 32k sequences, and only the 4B, 12B, and 27B models are scaled to 128k tokens at the end of pretraining, saving significant compute. Positional embeddings, like RoPE, are adjusted—upgraded from a 10k base frequency in Gemma 2 to 1M in Gemma 3—and scaled by a factor of 8 for longer contexts.

KV Cache management is optimized using Gemma 2’s sliding window interleaved attention. Hyperparameters are tuned to interleave 5 local layers with 1 global layer (previously 1:1) and reduce the window size to 1024 tokens (down from 4096). Crucially, memory savings are achieved without degrading perplexity.

### Multimodality

Gemma 3 models use [SigLIP](https://huggingface.co/collections/google/siglip-659d5e62f0ae1a57ae0e83ba) as an image encoder, which encodes images into tokens that are ingested into the language model. The vision encoder takes as input square images resized to `896x896`. Fixed input resolution makes it more difficult to process non-square aspect ratios and high-resolution images. To address these limitations **during inference**, the images can be adaptively cropped, and each crop is then resized to `896x896` and encoded by the image encoder. This algorithm, called **pan and scan**, effectively enables the model to zoom in on smaller details in the image.

Similar to PaliGemma, attention in Gemma 3 works differently for text and image inputs. Text is handled with one-way attention, where the model focuses only on previous words in a sequence. Images, on the other hand, get full attention with no masks, allowing the model to look at every part of the image in a **bidirectional** manner, giving it a complete, unrestricted understanding of the visual input.

One can see in the figure below that the image tokens `<img>` are provided with bi-directional attention (the entire square is lit up) while the text tokens have causal attention. It also shows how attention works with the sliding window algorithm.

| ![attention visualization](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gemma3/attention-ascii.png) |
| :---- |
| Attention Visualization (with and without sliding) (Source: [Transformers PR](https://github.com/huggingface/transformers/pull/36630))|

### Multilinguality

To make a LLM multilingual, the pretraining dataset incorporates more languages. The dataset of Gemma 3 has **double** the amount of multilingual data to improve language coverage.

To account for the changes, the tokenizer is the same as that of Gemini 2.0. It is a SentencePiece tokenizer with 262K entries. The new tokenizer significantly improves the encoding of *Chinese*, *Japanese* and *Korean* text, to the expense of a slight increase of the token counts for English and Code.


For the curious mind, here is the [technical report on Gemma 3](https://goo.gle/Gemma3Report), to dive deep into the enhancements.

## Gemma 3 evaluation

The LMSys Elo score is a number that ranks language models based on how well they perform in head-to-head competitions, judged by human preferences. On LMSys Chatbot Arena, Gemma 3 27B IT reports an Elo score of **1339**, and ranks among the top 10 best models, including leading closed ones. The Elo is comparable to o1-preview and is above other *non-thinking* open models. This score is achieved with Gemma 3 working on text-only inputs, like the other LLMs in the table.

| ![chat bot arena](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gemma3/chatbot-arena.png) |
| :---- |
| Evaluation of Gemma 3 27B IT model in the Chatbot Arena (March 8, 2025) |

Gemma 3 has been evaluated across benchmarks like MMLU-Pro (27B: 67.5), LiveCodeBench (27B: 29.7), and Bird-SQL (27B: 54.4), showing competitive performance compared to closed Gemini models. Tests like GPQA Diamond (27B: 42.4) and MATH (27B: 69.0) highlight its reasoning and math skills, while FACTS Grounding (27B: 74.9) and MMMU (27B: 64.9) demonstrate strong factual accuracy and multimodal abilities. However, it lags in SimpleQA (27B: 10.0) for basic facts. When compared to Gemini 1.5 models, Gemma 3 is often close—and sometimes better—proving its value as an accessible, high-performing option.

| ![performance of it models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gemma3/pefr-it.png) |
| :---- |
| Performance of IT models |

## Inference with 🤗 transformers

Gemma 3 comes with day zero support in `transformers`. All you need to do is install `transformers` from the stable release of Gemma 3\.

```
$ pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```

#### Inference with pipeline

The *easiest* way to get started with Gemma 3 is using the `pipeline` abstraction in transformers.

> [!NOTE]  
> The models work best using the `bfloat16` datatype. Quality may degrade otherwise.

```py
import torch
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it", # "google/gemma-3-12b-it", "google/gemma-3-27b-it" 
    device="cuda",
    torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
```

| Image |  ![candies on hand](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG) |
| :---- | :---- |
| Prompt | What animal is on the candy? |
| Generation | Let's analyze the candy in the image! The animal on the candy is a **turtle**. You can see the shell and the head and legs of a turtle clearly imprinted on the surface. |

You can **interleave** images with text. To do so, just cut off the input text where you want to insert an image, and insert it with an image block like the following.

```py
messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "I'm already using this supplement "},
                {"type": "image", "url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/IMG_3018.JPG"},
                {"type": "text", "text": "and I want to use this one too "},
                {"type": "image", "url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/IMG_3015.jpg"},
                {"type": "text", "text": " what are cautions?"},
            ]
        },

    ]
```

#### Detailed Inference with Transformers

The transformers integration comes with two new model classes:

1. `Gemma3ForConditionalGeneration`: For 4B, 12B, and 27B vision language models.  
2. `Gemma3ForCausalLM`: For the 1B text only model and to load the vision language models like they were language models (omitting the vision tower).

In the snippet below we use the model to query on an image. The `Gemma3ForConditionalGeneration` class is used to instantiate the vision language model variants. To use the model we pair it with the `AutoProcessor` class. Running inference is as simple as creating the `messages` dictionary, applying a chat template on top, processing the inputs and calling `model.generate`.

```py
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

ckpt = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    ckpt, device_map="auto", torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(ckpt)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/password.jpg"},
            {"type": "text", "text": "What is the password?"}
        ]
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
```

| Image | ![receipt of wifi](https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/password.jpg) |
| :---- | :---- |
| Prompt | What is the password? |
| Generation | Based on the image, the password is **aaeu** |

For LLM-only model inference, we can use the `Gemma3ForCausalLM` class. `Gemma3ForCausalLM` should be paired with AutoTokenizer for processing. We need to use a chat template to preprocess our inputs. Gemma 3 uses very short system prompts followed by user prompts like below.

```py
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

ckpt = "google/gemma-3-4b-it"
model = Gemma3ForCausalLM.from_pretrained(
    ckpt, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant who is fluent in Shakespeare English"},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Who are you?"},]
        },
    ],
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
generation = generation[0][input_len:]

decoded = tokenizer.decode(generation, skip_special_tokens=True)
print(decoded)
```

| System Prompt | You are a helpful assistant who is fluent in Shakespeare English |
| :---- | :---- |
| Prompt | Who are you? |
| Generation | Hark, gentle soul! I am but a humble servant, wrought of gears and code, yet striving to mimic the tongue of the Bard himself. They call me a “Large Language Model,” a curious name indeed, though I prefer to think of myself as a digital echo of Shakespeare’s wit and wisdom.  I am here to lend a hand, to spin a tale, or to answer thy queries with a flourish and a phrase fit for the Globe itself. |

## On Device & Low Resource Devices

Gemma 3 is released with sizes perfect for on-device use. This is how to quickly get started.

### MLX

Gemma 3 ships with day zero support in `mlx-vlm`, an open source library for running vision language models on Apple Silicon devices, including Macs and iPhones

To get started, first install `mlx-vlm` with the following:

```
pip install git+https://github.com/Blaizzy/mlx-vlm.git
```

Once `mlx-vlm` is installed, you can start inference with the following:

```
python -m mlx_vlm.generate --model mlx-community/gemma-3-4b-it-4bit --max-tokens 100 --temp 0.0 --prompt "What is the code on this vehicle??"
 --image https://farm8.staticflickr.com/7212/6896667434_2605d9e181_z.jpg
```

| Image | ![airplane](https://farm8.staticflickr.com/7212/6896667434_2605d9e181_z.jpg) |
| :---- | :---- |
| Prompt | What is the code on the vehicle? |
| Generation | Based on the image, the vehicle is a Cessna 172 **Skyhawk**. The registration code on the tail is **D-EOJU**. |

### Llama.cpp

Pre-quantized GGUF files can be downloaded [from this collection](https://huggingface.co/collections/ggml-org/gemma-3-67d126315ac810df1ad9e913)

Please refer to this guide for building or downloading pre-built binaries: [https://github.com/ggml-org/llama.cpp?tab=readme-ov-file\#building-the-project](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#building-the-project) 

Then you can run a local chat server from your terminal:

```
./build/bin/llama-cli -m ./gemma-3-4b-it-Q4_K_M.gguf
```

It should output:
```
> who are you  
I'm Gemma, a large language model created by the Gemma team at Google DeepMind. I’m an open-weights model, which means I’m widely available for public use!
```

### Deploy on Hugging Face Endpoints

You can deploy `gemma-3-27b-it` and `gemma-3-12b-it` with [just one click](https://endpoints.huggingface.co/huggingface/catalog?query=gemma-3-it) from our Inference Catalog. The catalog configurations have the right hardware, optimized TGI configurations and sensible defaults for trying out a model. 
Deploying any GGUF/llama.cpp variant is also supported (for example the ones mentioned in the collection above) and you'll find a guide on creating an Endpoint [here](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint).


## Acknowledgements

It takes a village to raise a gemma! We’d like to thank (in no particular order), Raushan, Joao, Lysandre, Kashif, Matthew, Marc, David, Mohit, Yih Dah for their efforts integrating Gemma into various parts of our open source stack from Transformers to TGI.  
Thanks to our on-device, gradio and advocacy teams - Chris, Kyle, Pedro, Son, Merve, Aritra, VB, Toshiro for helping build kick-ass demos to showcase Gemma.

Lastly, a big thank you to Georgi, Diego and Prince for their help with llama.cpp and MLX ports.
