---
title: "Llama can now see and run on your device - welcome Llama 3.2" 
thumbnail: /blog/assets/llama32/thumbnail.jpg
authors:
- user: merve
- user: philschmid
- user: osanseviero
- user: reach-vb
- user: lewtun
- user: ariG23498
- user: pcuenq
---

# Llama can now see and run on your device - welcome Llama 3.2

Llama 3.2 is out! Today we welcome the next iteration of the Llama collection to Hugging Face. This time, we’re excited to collaborate with Meta on the release of multimodal and small models. Ten open-weight models (5 multimodal models and 5 text-only ones) are available on the Hub.

Llama 3.2 Vision comes in two sizes: 11B for efficient deployment and development on consumer-size GPU, and 90B for large-scale applications. Both versions come in base and instruction-tuned variants. In addition to the four multimodal models, Meta released a new version of Llama Guard with vision support. Llama Guard 3 is a safeguard model that can classify model inputs and generations, including detecting harmful multimodal prompts or assistant responses.

Llama 3.2 also includes small text-only language models that can run on-device. They come in two new sizes (1B and 3B) with base and instruct variants, and they have strong capabilities for their sizes. There’s also a small 1B version of Llama Guard that can be deployed alongside these or the larger text models in production use cases.

Among the features and integrations being released, we have:
- [Model checkpoints on the Hub](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)
- Hugging Face Transformers and TGI integration for the Vision models
- Inference & Deployment Integration with Inference Endpoints, Google Cloud, Amazon SageMaker & DELL Enterprise Hub
- Fine-tuning Llama 3.2 11B Vision on a single GPU with [transformers🤗](https://github.com/huggingface/huggingface-llama-recipes/llama32.ipynb) and [TRL](https://github.com/huggingface/trl/tree/main/examples/scripts/sft_vlm.py)

## Table of contents

- [What is Llama 3.2 Vision?](#what-is-llama-32-vision)
- [Llama 3.2 license changes. Sorry, EU :(](#llama-32-license-changes-sorry-eu-)
- [What is special about Llama 3.2 1B and 3B?](#what-is-special-about-llama-32-1b-and-3b)
- [Demo](#demo)
- [Using Hugging Face Transformers](#using-hugging-face-transformers)
- [Llama 3.2 1B & 3B Language Models](#llama-32-1b-3b-language-models)
- [Llama 3.2 Vision](#llama-32-vision)
- [On-device](#ondevice)
- [Llama.cpp & Llama-cpp-python](#llamacpp-llama-cpp-python)
- [Transformers.js](#transformersjs)
- [Fine-tuning Llama 3.2](#finetuning-llama-32)
- [Hugging Face Partner Integrations](#hugging-face-partner-integrations)
- [Additional Resources](#additional-resources)
- [Acknowledgements](#acknowledgements)

## What is Llama 3.2 Vision?

Llama 3.2 Vision is the most powerful open multimodal model released by Meta. It has great visual understanding and reasoning capabilities and can be used to accomplish a variety of tasks, including visual reasoning and grounding, document question answering, and image-text retrieval. Chain of Thought (CoT) answers are often very good, which makes visual reasoning particularly powerful.

Llama 3.2 Vision can be used to process text and an image as well as only text. With image-text prompting, the model can take English inputs, while for text-only prompting, the model can handle multiple languages. The full list of languages in text-only mode is English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

The architecture of these models is based on the combination of Llama 3.1 LLMs combined with a vision tower and an image adapter. The text models used are Llama 3.1 8B for the Llama 3.2 11B Vision model, and Llama 3.1 70B for the 3.2 90B Vision model. To the best of our understanding, the text models were frozen during the training of the vision models to preserve text-only performance.

Below you can find some inference examples from the 11B instruction-tuned model that showcase real world knowledge, document reasoning and infographics understanding capabilities.


| Image                                                                                                          | Prompt                                                                     | Response                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ![Wat Arun](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/wat_arun.jpg)              | Where is this located in? What are other attractions nearby?               | The image depicts the Grand Palace in Bangkok, Thailand. The Grand Palace is a complex of buildings that served as the official residence of the King of Siam and the administrative seat of government for 150 years. It is one of the most popular tourist attractions in Bangkok and a must-see for anyone visiting the city.<br><br>Some other attractions near the Grand Palace include:<br>* Wat Phra Kaew: This temple is located within the Grand Palace complex and is home to the famous Emerald Buddha statue.<br>* Wat Arun: This temple is located on the west bank of the Chao Phraya River and is known for its stunning architecture and beautiful views of the city.<br>* Wat Pho: This temple is located near the Grand Palace and is home to a large reclining Buddha statue.<br>* Lumpini Park: This park is located in the heart of Bangkok and offers a peaceful escape from the hustle and bustle of the city.<br>* Jim Thompson House: This museum showcases the life and work of Jim Thompson, an American businessman who helped revitalize the Thai silk industry in the 1950s and 1960s. Overall, the Grand Palace is a must-see attraction in Bangkok, and there are many other interesting places to visit in the area. |
| ![Weather Events](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/weather_events.png)   | Where do the droughts happen according to this diagram?                    | **Drought Regions in 2016**<br>According to the diagram, droughts occurred in two distinct regions in 2016:<br>* **Eastern and Southern Africa:** This region experienced severe droughts.<br>* **Chile:** Chile was the driest in January in over 50 years. These areas faced significant water scarcity issues during that period.                                                                                                                                                                                                                                                                                                                                                 |
| ![Invoice](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/invoice.png)                | How long does it take from invoice date to due date? Be short and concise. | **Invoice Date:** 11/02/2019<br>**Due Date:** 26/02/2019<br>**Time Difference Calculation:**<br>26/02/2019 - 11/02/2019 = 15 days<br><br>The time difference between the invoice date and the due date is **15 days.**                                                                                                                                                                                                                                                                                                                                                                           |


Vision models have a context length of 128k tokens, which allows for multiple-turn conversations that may contain images. However, the model works best when attending to a single image, so the `transformers` implementation only attends to the last image provided in the input. This preserves quality and saves memory.

The 11B base model supports tile sizes of 448, while the instruct version and the 90B models all use tile sizes of 560. The models were trained on a massive dataset of 6 billion image-text pairs with a diverse data mixture. This makes them excellent candidates for fine-tuning on downstream tasks. For reference, you can see below how the 11B, 90B and their instruction fine-tuned versions compare in some benchmarks, as reported by Meta. Please, refer to the model cards for additional benchmarks and details.

|            | 11B               | 11B (instruction-tuned) | 90B               | 90B (instruction-tuned) | Metric | 
|------------|-------------------|-----------------|-------------------|------------------|------------------|
| MMMU (val) | 41.7 | 50.7 (CoT)      | 49.3 (zero-shot) | 60.3 (CoT)       | Micro Average Accuracy |
| VQAv2      | 66.8 (val)       | 75.2 (test)     | 73.6 (val)       | 78.1 (test)      | Accuracy |
| DocVQA     | 62.3 (val)       | 88.4 (test)     | 70.7 (val)       | 90.1 (test)      | ANLS |
| AI2D       | 62.4             | 91.1            | 75.3             | 92.3             | Accuracy |

We expect the text capabilities of these models to be on par with the 8B and 70B Llama 3.1 models, respectively, as our understanding is that the text models were frozen during the training of the Vision models. Hence, text benchmarks should be consistent with 8B and 70B.

## Llama 3.2 license changes. Sorry, EU :(

![License Change](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/license_change.png)

Regarding the licensing terms, Llama 3.2 comes with a very similar license to Llama 3.1, with one key difference in the acceptable use policy: any individual domiciled in, or a company with a principal place of business in, the European Union is not being granted the license rights to use multimodal models included in Llama 3.2. This restriction does not apply to end users of a product or service that incorporates any such multimodal models, so people can still build global products with the vision variants.

For full details, please make sure to read [the official license](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/USE_POLICY.md).

## What is special about Llama 3.2 1B and 3B? 

The Llama 3.2 collection includes 1B and 3B text models. These models are designed for on-device use cases, such as prompt rewriting, multilingual knowledge retrieval, summarization tasks, tool usage, and locally running assistants. They outperform many of the available open-access models at these sizes and compete with models that are many times larger. In a later section, we’ll show you how to run these models offline.

The models follow the same architecture as Llama 3.1. They were trained with up to 9 trillion tokens and still support the long context length of 128k tokens. The models are multilingual, supporting English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

There is also a new small version of Llama Guard, Llama Guard 3 1B, that can be deployed with these models to evaluate the last user or assistant responses in a multi-turn conversation. It uses a set of pre-defined categories which (new to this version) can be customized or excluded to account for the developer’s use case. For more details on the use of Llama Guard, please refer to the model card.

Bonus: Llama 3.2 has been exposed to a broader collection of languages than the 8 supported languages mentioned above. Developers are encouraged to fine-tune Llama 3.2 models for their specific language use cases.

We ran the base models through the Open LLM Leaderboard evaluation suite, while the instruct models were evaluated across three popular benchmarks that measure instruction-following and correlate well with the LMSYS Chatbot Arena: [IFEval](https://arxiv.org/abs/2311.07911), [AlpacaEval](https://arxiv.org/abs/2404.04475), and [MixEval-Hard](https://arxiv.org/abs/2406.06565). These are the results for the base models, with Llama-3.1-8B included as a reference:


| Model                | BBH   | MATH Lvl 5 | GPQA  | MUSR  | MMLU-PRO | Average |
|----------------------|-------|------------|-------|-------|----------|---------|
| Meta-Llama-3.2-1B     | 4.37  | 0.23       | 0.00  | 2.56  | 2.26     | 1.88    |
| Meta-Llama-3.2-3B     | 14.73 | 1.28       | 4.03  | 3.39  | 16.57    | 8.00    |
| Meta-Llama-3.1-8B     | 25.29 | 4.61       | 6.15  | 8.98  | 24.95    | 14.00   |

And here are the results for the instruct models, with Llama-3.1-8B-Instruct included as a reference:

| Model                       | AlpacaEval (LC) | IFEval | MixEval-Hard | Average |
|-----------------------------|-----------------|--------|--------------|---------|
| Meta-Llama-3.2-1B-Instruct   | 7.17            | 58.92  | 26.10        | 30.73   |
| Meta-Llama-3.2-3B-Instruct   | 20.88           | 77.01  | 31.80        | 43.23   |
| Meta-Llama-3.1-8B-Instruct   | 25.74           | 76.49  | 44.10        | 48.78   |

Remarkably, the 3B model is as strong as the 8B one on IFEval! This makes the model well-suited for agentic applications, where following instructions is crucial for improving reliability. This high IFEval score is very impressive for a model of this size.


Tool use is supported in both the 1B and 3B instruction-tuned models. Tools are specified by the user in a zero-shot setting (the model has no previous information about the tools developers will use). Thus, the built-in tools that were part of the Llama 3.1 models (`brave_search` and `wolfram_alpha`) are no longer available.

Due to their size, these small models can be used as assistants for bigger models and perform [assisted generation](https://huggingface.co/blog/assisted-generation) (also known as speculative decoding). [Here](https://github.com/huggingface/huggingface-llama-recipes/tree/main) is an example of using the Llama 3.2 1B model as an assistant to the Llama 3.1 8B model. For offline use cases, please check the on-device section later in the post.

## Demo
You can experiment with the three Instruct models in the following demos:

- [Gradio Space with Llama 3.2 11B Vision Instruct](https://huggingface.co/spaces/huggingface-projects/llama-3.2-vision-11B)
- [Gradio-powered Space with Llama 3.2 3B](https://huggingface.co/spaces/huggingface-projects/llama-3.2-3B-Instruct)
- Llama 3.2 3B running on WebGPU 

![Demo GIF](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/demo_gif.gif)

## Using Hugging Face Transformers

The text-only checkpoints have the same architecture as previous releases, so there is no need to update your environment. However, given the new architecture, Llama 3.2 Vision requires an update to Transformers. Please make sure to upgrade your installation to release 4.45.0 or later. 

```bash
pip install "transformers>=4.45.0" --upgrade
```

Once upgraded, you can use the new Llama 3.2 models and leverage all the tools of the Hugging Face ecosystem.

## Llama 3.2 1B & 3B Language Models

You can run the 1B and 3B Text model checkpoints in just a couple of lines with Transformers. The model checkpoints are uploaded in `bfloat16` precision, but you can also use float16 or quantized weights. Memory requirements depend on the model size and the precision of the weights. Here's a table showing the approximate memory required for inference using different configurations:

| Model Size | BF16/FP16 | FP8     | INT4    |
|------------|--------|---------|---------|
| 3B         | 6.5 GB | 3.2 GB  | 1.75 GB |
| 1B         | 2.5 GB | 1.25 GB | 0.75 GB |

```python
from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
response = outputs[0]["generated_text"][-1]["content"]
print(response)
# Arrrr, me hearty! Yer lookin' fer a bit o' information about meself, eh? Alright then, matey! I be a language-generatin' swashbuckler, a digital buccaneer with a penchant fer spinnin' words into gold doubloons o' knowledge! Me name be... (dramatic pause)...Assistant! Aye, that be me name, and I be here to help ye navigate the seven seas o' questions and find the hidden treasure o' answers! So hoist the sails and set course fer adventure, me hearty! What be yer first question?
```

A couple of details:

- We load the model in `bfloat16`. As mentioned above, this is the type used by the original checkpoint published by Meta, so it’s the recommended way to run to ensure the best precision or conduct evaluations. Depending on your hardware, float16 might be faster.

- By default, transformers uses the same sampling parameters (temperature=0.6 and top_p=0.9) as the original meta codebase. We haven’t conducted extensive tests yet, feel free to explore!

## Llama 3.2 Vision

The Vision models are larger, so they require more memory to run than the small text models. For reference, the 11B Vision model takes about 10 GB of GPU RAM during inference, in 4-bit mode.

The easiest way to infer with the instruction-tuned Llama Vision model is to use the built-in chat template. The inputs have `user` and `assistant` roles to indicate the conversation turns. One difference with respect to the text models is that the system role is not supported. User turns may include image-text or text-only inputs. To indicate that the input contains an image, add `{"type": "image"}` to the content part of the input and then pass the image data to the `processor`:

```python
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Can you please describe this image in just one sentence?"}
    ]}
]

input_text = processor.apply_chat_template(
    messages, add_generation_prompt=True,
)
inputs = processor(
    image, input_text, return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=70)

print(processor.decode(output[0][inputs["input_ids"].shape[-1]:]))


## The image depicts a rabbit dressed in a blue coat and brown vest, standing on a dirt road in front of a stone house.
```

You can continue the conversation about the image. Remember, however, that if you provide a new image in a new user turn, the model will refer to the new image from that moment on. You can’t query about two different images at the same time. This is an example of the previous conversation continued, where we add the assistant turn to the conversation and ask for some more details:

```python
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Can you please describe this image in just one sentence?"}
    ]},
    {"role": "assistant", "content": "The image depicts a rabbit dressed in a blue coat and brown vest, standing on a dirt road in front of a stone house."},
    {"role": "user", "content": "What is in the background?"}
]

input_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)
inputs = processor(image, input_text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=70)
print(processor.decode(output[0][inputs["input_ids"].shape[-1]:]))
```
And this is the response we got:

```
In the background, there is a stone house with a thatched roof, a dirt road, a field of flowers, and rolling hills.
```

You can also automatically quantize the model, loading it in 8-bit or even 4-bit mode with the `bitsandbytes` library. This is how you’d load the generation pipeline in 4-bit:

```diff
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
+from transformers import BitsAndBytesConfig

+bnb_config = BitsAndBytesConfig(
+    load_in_4bit=True,
+    bnb_4bit_quant_type="nf4",
+    bnb_4bit_compute_dtype=torch.bfloat16
)
 
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
-   torch_dtype=torch.bfloat16,
    device="cuda",
+   quantization_config=bnb_config,
)
```

You can then apply the chat template, use the processor, and call the model just like you did before.

## On-device

You can run both Llama 3.2 1B and 3B directly on your device's CPU/ GPU/ Browser using several open-source libraries like the following.

### Llama.cpp & Llama-cpp-python

[Llama.cpp](https://github.com/ggerganov/llama.cpp) is the go-to framework for all things cross-platform on-device ML inference. We provide quantized 4-bit & 8-bit weights for both 1B & 3B models in this collection. We expect the community to embrace these models and create additional quantizations and fine-tunes. For example, here you can see models in the Hub that were quantized from the Llama 3.1 8B model.

Here’s how you can use these checkpoints directly with llama.cpp.

Install llama.cpp through brew (works on Mac and Linux).

```bash
brew install llama.cpp
```

You can use the CLI to run a single generation or invoke the llama.cpp server, which is compatible with the Open AI messages specification.

You’d run the CLI using a command like this:

```bash
llama-cli --hf-repo hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF --hf-file llama-3.2-3b-instruct-q8_0.gguf -p "The meaning to life and the universe is"
```

And you’d fire up the server like this:

```bash
llama-server --hf-repo hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF --hf-file llama-3.2-3b-instruct-q8_0.gguf -c 2048
```

You can also use [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) to access these models programmatically in Python.

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
    filename="*q8_0.gguf",
)
llm.create_chat_completion(
      messages = [
          {
              "role": "user",
              "content": "What is the capital of France?"
          }
      ]
)
```


### Transformers.js

You can even run Llama 3.2 in your browser (or any JavaScript runtime like Node.js, Deno, or Bun) using [Transformers.js](https://huggingface.co/docs/transformers.js). You can find the [ONNX model](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct) on Hub. If you haven't already, you can install the library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:

```bash
npm i @huggingface/transformers
```

Then, you can run the model as follows:
```js
import { pipeline } from "@huggingface/transformers";

// Create a text generation pipeline
const generator = await pipeline("text-generation", "onnx-community/Llama-3.2-1B-Instruct");

// Define the list of messages
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Tell me a joke." },
];

// Generate a response
const output = await generator(messages, { max_new_tokens: 128 });
console.log(output[0].generated_text.at(-1).content);
```

<details>

<summary>Example output</summary>

```
Here's a joke for you:

What do you call a fake noodle?

An impasta!

I hope that made you laugh! Do you want to hear another one?
```

</details>

## Fine-tuning Llama 3.2

TRL supports chatting and fine-tuning with the Llama 3.2 text models out of the box:

```bash
# Chat
trl chat --model_name_or_path meta-llama/Llama-3.2-3B

# Fine-tune
trl sft  --model_name_or_path meta-llama/Llama-3.2-3B \
         --dataset_name HuggingFaceH4/no_robots \
         --output_dir Llama-3.2-3B-Instruct-sft \
         --gradient_checkpointing
```

Support for fine tuning Llama 3.2 Vision is also available in TRL with [this script](https://github.com/huggingface/trl/tree/main/examples/scripts/sft_vlm.py).

```bash
# Tested on 8x H100 GPUs
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir Llama-3.2-11B-Vision-Instruct-sft \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
```

You can also check out [this notebook](https://github.com/huggingface/huggingface-llama-recipes/blob/main/Llama-Vision%20FT.ipynb) for LoRA fine-tuning using transformers and PEFT. 

## Hugging Face Partner Integrations

We are currently working with our partners at AWS, Google Cloud, Microsoft Azure and DELL on adding Llama 3.2 11B, 90B to Amazon SageMaker, Google Kubernetes Engine, Vertex AI Model Catalog, Azure AI Studio, DELL Enterprise Hub. We will update this section as soon as the containers are available, and you can subscribe to [Hugging Squad](https://mailchi.mp/huggingface/squad) for email updates.


## Additional Resources

- [Models on the Hub](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)
- [Hugging Face Llama Recipes](https://github.com/huggingface/huggingface-llama-recipes)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Meta Blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [Evaluation datasets](https://huggingface.co/collections/meta-llama/llama-32-evals-66f44b3d2df1c7b136d821f0)
- [](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct)

## Acknowledgements

Releasing such models with support and evaluations in the ecosystem would not be possible without the contributions of thousands of community members who have contributed to transformers, text-generation-inference, vllm, pytorch, LM Eval Harness, and many other projects. Hat tip to the VLLM team for their help in testing and reporting issues. This release couldn't have happened without all the support of Clémentine, Alina, Elie, and Loubna for LLM evaluations, Nicolas Patry, Olivier Dehaene, and Daniël de Kok for Text Generation Inference; Lysandre, Arthur, Pavel, Edward Beeching, Amy, Benjamin, Joao, Pablo, Raushan Turganbay, Matthew Carrigan, and Joshua Lochner for transformers, transformers.js, TRL, and PEFT support; Brigitte Tousignant and Florent Daudens for communication; Julien, Simon, Pierric, Eliott, Lucain, Alvaro, Caleb, and Mishig from the Hub team for Hub development and features for launch.

And big thanks to the Meta Team for releasing Llama 3.2 and making it available to the open AI community!

