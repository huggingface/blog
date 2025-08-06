---
title: "Welcome GPT OSS, the new open-source model family from OpenAI!"
thumbnail: /blog/assets/openai/openai-hf-thumbnail.png
authors:
- user: reach-vb
- user: pcuenq
- user: lewtun
- user: clem
- user: Rocketknight1
- user: clefourrier
- user: celinah
- user: Wauplin
- user: marcsun13
- user: pagezyhf
- user: ahadnagy
- user: joaogante
---

# Welcome GPT OSS, the new open-source model family from OpenAI!

GPT OSS is a hugely anticipated open-weights release by OpenAI, designed for powerful reasoning, agentic tasks, and versatile developer use cases. It comprises two models: a big one with 117B parameters ([gpt-oss-120b](https://hf.co/openai/gpt-oss-120b)), and a smaller one with 21B parameters ([gpt-oss-20b](https://hf.co/openai/gpt-oss-20b)). Both are mixture-of-experts (MoEs) and use a 4-bit quantization scheme (MXFP4), enabling fast inference (thanks to fewer active parameters, see details below) while keeping resource usage low. The large model fits on a single H100 GPU, while the small one runs within 16GB of memory and is perfect for consumer hardware and on-device applications.

To make it even better and more impactful for the community, the models are licensed under the **Apache 2.0 license**, along with a minimal usage policy:  
> We aim for our tools to be used safely, responsibly, and democratically, while maximizing your control over how you use them. By using gpt-oss, you agree to comply with all applicable law.

According to OpenAI, this release is a meaningful step in their commitment to the open-source ecosystem, in line with their stated mission to make the benefits of AI broadly accessible. Many use cases rely on private and/or local deployments, and we at Hugging Face are super excited to welcome [OpenAI](https://huggingface.co/openai) to the community. We believe these will be long-lived, inspiring and impactful models.

## Contents

- [Introduction](#welcome-gpt-oss-the-new-open-source-model-family-from-openai)
- [Overview](#overview-of-capabilities-and-architecture)
- [API access through Inference Providers](#api-access-through-inference-providers)
- [Local Inference](#local-inference)
  - [Using transformers](#using-transformers)
    * [Flash Attention 3](#flash-attention-3)
    * [Other optimizations](#other-optimizations)
    * [AMD ROCm support](#amd-rocm-support)
    * [Summary of Optimizations](#summary-of-available-optimizations)
  - [llama.cpp](#llamacpp)
  - [vLLM](#vllm)
  - [transformers serve](#transformers-serve)
- [Fine Tuning](#fine-tuning)
- [Deploy on Hugging Face Partners](#deploy-on-hugging-face-partners)
    - [Azure](#azure)
    - [Dell](#dell)
- [Evaluating the Model](#evaluating-the-model)
- [Chats and Chat Templates](#chats-and-chat-templates)
  - [System and Developer Messages](#system-and-developer-messages)
  - [Tool use with transformers](#tool-use-with-transformers)

## Overview of Capabilities and Architecture

* 21B and 117B total parameters, with 3.6B and 5.1B *active* parameters, respectively.  
* 4-bit quantization scheme using [mxfp4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) format. Only applied on the MoE weights. As stated, the 120B fits in a single 80 GB GPU and the 20B fits in a single 16GB GPU.  
* Reasoning, text-only models; with chain-of-thought and adjustable reasoning effort levels.  
* Instruction following and tool use support.  
* Inference implementations using transformers, vLLM, llama.cpp, and ollama.  
* [Responses API](https://platform.openai.com/docs/api-reference/responses) is recommended for inference.  
* License: Apache 2.0, with a small complementary use policy.

**Architecture**

* Token-choice MoE with SwiGLU activations.  
* When calculating the MoE weights, a softmax is taken over selected experts (softmax-after-topk).  
* Each attention layer uses RoPE with 128K context.  
* Alternate attention layers: full-context, and sliding 128-token window.  
* Attention layers use a *learned attention sink* per-head, where the denominator of the softmax has an additional additive value.  
* It uses the same tokenizer as GPT-4o and other OpenAI API models.  
  * Some new tokens have been incorporated to enable compatibility with the Responses API.

## API access through Inference Providers

OpenAI GPT OSS models are accessible through Hugging Face’s [Inference Providers](https://huggingface.co/docs/inference-providers/en/index) service, allowing you to send requests to any supported provider using the same JavaScript or Python code. This is the same infrastructure that powers OpenAI’s official demo on [gpt-oss.com](http://gpt-oss.com), and you can use it for your own projects.

Below is an example that uses Python and the super-fast Cerebras provider. For more info and additional snippets, check the [inference providers section in the model cards](https://huggingface.co/openai/gpt-oss-120b?inference_api=true&inference_provider=auto&language=python&client=openai) and the [dedicated guide we crafted for these models](https://huggingface.co/docs/inference-providers/guides/gpt-oss). 

```py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:cerebras",
    messages=[
        {
            "role": "user",
            "content": "How many rs are in the word 'strawberry'?",
        }
    ],
)

print(completion.choices[0].message)
```

Inference Providers also implements an OpenAI-compatible Responses API, the most advanced OpenAI interface for chat models, designed for more flexible and intuitive interactions.  
Below is an example using the Responses API with the Fireworks AI provider. For more details, check out the open-source [responses.js](https://github.com/huggingface/responses.js) project.

```py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="openai/gpt-oss-20b:fireworks-ai",
    input="How many rs are in the word 'strawberry'?",
)

print(response)
```

## Local Inference

### Using Transformers

You need to install the latest `transformers` release (v4.55 or later), as well as `accelerate` and `kernels`:

```shell
pip install --upgrade accelerate transformers kernels
```

The model weights are quantized in `mxfp4` format, which is compatible with GPUs of the Hopper or Blackwell families. This includes data-center cards such as H100, H200 or GB200, as well as the latest consumer GPUs in the 50xx family. If you have one of these cards, `mxfp4` will yield the best results in terms of speed and memory consumption. To use it, you need `triton 3.4` and `triton_kernels`. If these libraries are not installed (or you don’t have a compatible GPU), loading the model will fall back to `bfloat16`, unpacked from the quantized weights.

In our tests, Triton 3.4 works fine with the latest PyTorch version (2.7.x). You may optionally want to install PyTorch 2.8 instead – it’s a pre-release version at the time of writing ([although it should be released soon](https://github.com/pytorch/pytorch/milestone/53)), but it’s the one that’s been prepared alongside triton 3.4, so they are stable together. Here’s how to install PyTorch 2.8 (comes with triton 3.4) and the triton kernels:

```shell
# Optional step if you want PyTorch 2.8, otherwise just `pip install torch`
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/test/cu128

# Install triton kernels for mxfp4 support
pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
```

The following snippet shows simple inference with the 20B model. It runs on 16 GB GPUs when using `mxfp4`, or \~48 GB in `bfloat16`.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)

messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

#### Flash Attention 3

The models use *attention sinks*, a technique the vLLM team made compatible with Flash Attention 3\. We have packaged and integrated their optimized kernel in [`kernels-community/vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3). At the time of writing, this super-fast kernel has been tested on Hopper cards with PyTorch 2.7 and 2.8. We expect increased coverage in the coming days. If you run the models on Hopper cards (for example, H100 or H200), you need to `pip install --upgrade kernels` and add the following line to your snippet:

```diff  
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
+    # Flash Attention with Sinks
+    attn_implementation="kernels-community/vllm-flash-attn3",
)

messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

This snippet will download the optimized, pre-compiled kernel code from `kernels-community`, as explained in our [previous blog post](https://huggingface.co/blog/hello-hf-kernels). The transformers team has built, packaged, and tested the code, so it’s totally safe for you to use.

#### Other optimizations

If you have a Hopper GPU or better, we recommend you use `mxfp4` for the reasons explained above. If you can additionally use Flash Attention 3, then by all means do enable it! 

> [!TIP]  
> If your GPU is not compatible with `mxfp4`, then we recommend you use MegaBlocks MoE kernels for a nice speed bump. To do so, you just need to adjust your inference code like this:

```diff
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
+    # Optimize MoE layers with downloadable` MegaBlocksMoeMLP
+    use_kernels=True,
)

messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

> [!TIP]
> MegaBlocks optimized MoE kernels require the model to run on `bfloat16`, so memory consumption will be higher than running on `mxfp4`. We recommend you use `mxfp4` if you can, otherwise opt in to MegaBlocks via `use_kernels=True`.

#### AMD ROCm support

OpenAI GPT OSS has been verified on AMD Instinct hardware, and we’re happy to announce initial support for AMD’s ROCm platform in our kernels library, setting the stage for upcoming optimized ROCm kernels in Transformers. MegaBlocks MoE kernel acceleration is already available for OpenAI GPT OSS on AMD Instinct (e.g., MI300-series), enabling better training and inference performance. You can test it with the same inference code shown above.

AMD also prepared a Hugging Face [Space](https://huggingface.co/spaces/amd/gpt-oss-120b-chatbot) for users to try the model on AMD hardware.

#### Summary of Available Optimizations

At the time of writing, this table summarizes our _recommendations_ based on GPU compatibility and our tests. We expect Flash Attention 3 (with sink attention) to become compatible with additional GPUs.

|  | mxfp4 | Flash Attention 3 (w/ sink attention) | MegaBlocks MoE kernels |
| :---- | :---- | :---- | :---- |
| Hopper GPUs (H100, H200) | ✅ | ✅ | ❌ |
| Blackwell GPUs (GB200, 50xx, RTX Pro 6000\) | ✅ | ❌ | ❌ |
| Other CUDA GPUs | ❌ | ❌ | ✅ |
| AMD Instinct (MI3XX) | ❌ | ❌ | ✅ |
| *How to enable* | Install triton 3.4 + triton kernels | Use vllm-flash-attn3 from kernels-community" | `use_kernels` |

Even though the 120B model fits on a single H100 GPU (using `mxfp4`), you can also run it easily on multiple GPUs using `accelerate` or `torchrun`. Transformers provides a default parallelization plan, and you can leverage optimized attention kernels as well. The following snippet can be run with `torchrun --nproc_per_node=4 generate.py` on a system with 4 GPUs:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
import torch

model_path = "openai/gpt-oss-120b"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

device_map = {
    "tp_plan": "auto",    # Enable Tensor Parallelism
}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    attn_implementation="kernels-community/vllm-flash-attn3",
    **device_map,
)

messages = [
     {"role": "user", "content": "Explain how expert parallelism works in large language models."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000)

# Decode and print
response = tokenizer.decode(outputs[0])
print("Model response:", response.split("<|channel|>final<|message|>")[-1].strip())
```

The OpenAI GPT OSS models have been trained extensively to leverage tool use as part of their reasoning efforts. The chat template we crafted for transformers provides a lot of flexibility, please check our [dedicated section later in this post](#tool-use-with-transformers).

### Llama.cpp

Llama.cpp offers native MXFP4 support with Flash Attention, delivering optimal performance across various backends such as Metal, CUDA, and Vulkan, right from the day-0 release.

To install it, follow the guide in [llama.cpp Github’s repository](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md).

```
# MacOS
brew install llama.cpp

# Windows
winget install llama.cpp
```

The recommended way is to use it via llama-server:

```
llama-server -hf ggml-org/gpt-oss-120b-GGUF -c 0 -fa --jinja --reasoning-format none

# Then, access http://localhost:8080
```

We support both the 120B and 20B models. For more detailed information, visit [this PR](https://github.com/ggml-org/llama.cpp/pull/15091) or the [GGUF model collection](https://huggingface.co/collections/ggml-org/gpt-oss-68923b60bee37414546c70bf).

### vLLM

As mentioned, vLLM developed optimized Flash Attention 3 kernels that support sink attention, so you’ll get best results on Hopper cards. Both the Chat Completion and the Responses APIs are supported. You can install and start a server with the following snippet, which assumes 2 H100 GPUs are used:

```shell
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2
```

Or, use it in Python directly like:

```py
from vllm import LLM
llm = LLM("openai/gpt-oss-120b", tensor_parallel_size=2)
output = llm.generate("San Francisco is a")
```

### `transformers serve`

You can use [`transformers serve`](https://huggingface.co/docs/transformers/main/serving) to experiment locally with the models, without any other dependencies. You can launch the server with just:

```shell
transformers serve
```

To which you can send requests using the [Responses API](https://platform.openai.com/docs/api-reference/responses). 

```shell
# responses API
curl -X POST http://localhost:8000/v1/responses \
-H "Content-Type: application/json" \
-d '{"input": [{"role": "system", "content": "hello"}], "temperature": 1.0, "stream": true, "model": "openai/gpt-oss-120b"}'
```

You can also send requests using the standard Completions API:

```shell
# completions API
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{"messages": [{"role": "system", "content": "hello"}], "temperature": 1.0, "max_tokens": 1000, "stream": true, "model": "openai/gpt-oss-120b"}'
```

## Fine-Tuning

GPT OSS models are fully integrated with `trl`. We have developed a couple of fine-tuning examples using `SFTTrainer` to get you started:

* A LoRA example in the [OpenAI cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers), which shows how the model can be fine-tuned to reason in multiple languages.  
* [A basic fine-tuning script](https://github.com/huggingface/gpt-oss-recipes/blob/main/sft.py) that you can adapt to your needs.

## Deploy on Hugging Face Partners

### Azure

Hugging Face collaborates with Azure on their Azure AI Model Catalog to bring the most popular open-source models —spanning text, vision, speech, and multimodal tasks— directly into customers environments for secured deployments to managed online endpoints, leveraging Azure’s enterprise-grade infrastructure, autoscaling, and monitoring.

The GPT OSS models are now available on the Azure AI Model Catalog ([GPT OSS 20B](https://ai.azure.com/explore/models/openai-gpt-oss-20b/version/1/registry/HuggingFace), [GPT OSS 120B](https://ai.azure.com/explore/models/openai-gpt-oss-120b/version/1/registry/HuggingFace)), ready to be deployed to an online endpoints for real time inference.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gpt-oss/partners/azure-model-card.png" alt="model card in azure ai model catalog"/>

### Dell

The Dell Enterprise Hub is a secure online portal that simplifies training and deploying the latest open AI models on-premise using Dell platforms. Developed in collaboration with Dell, it offers optimized containers, native support for Dell hardware, and enterprise-grade security features. 

The GPT OSS models are now available on [Dell Enterprise Hub](https://dell.huggingface.co/), ready to be deployed on-prem using Dell platforms.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gpt-oss/partners/deh-model-card.png" alt="model card in dell enterprise hub"/>

## Evaluating the Model

GPT OSS models are reasoning models: they therefore require a very large generation size (maximum number of new tokens) for evaluations, as their generation will first contain reasoning, then the actual answer. Using too small a generation size risks interrupting the prediction in the middle of reasoning, which will cause false negatives. The reasoning trace should then be removed from the model answer before computing metrics, to avoid parsing errors, especially with math or instruct evaluations.

Here’s an example on how to evaluate the models with lighteval (you need to install from source).

```shell
git clone https://github.com/huggingface/lighteval
pip install -e .[dev] # make sure you have the correct transformers version installed!
lighteval accelerate \
    "model_name=openai/gpt-oss-20b,max_length=16384,skip_special_tokens=False,generation_parameters={temperature:1,top_p:1,top_k:40,min_p:0,max_new_tokens:16384}" \ 
    "extended|ifeval|0|0,lighteval|aime25|0|0" \
    --save-details --output-dir "openai_scores" \
    --remove-reasoning-tags --reasoning-tags="[('<|channel|>analysis<|message|>','<|end|><|start|>assistant<|channel|>final<|message|>')]" 
```

For the 20B model, this should give you 69.5 (+/-1.9) for IFEval (strict prompt), and 63.3 (+/-8.9) for AIME25 (in pass@1), scores within expected range for a reasoning model of this size.

If you want to do your custom evaluation script, note that to filter out the reasoning tags properly, you will need to use `skip_special_tokens=False` in the tokenizer, in order to get the full trace in the model output (to filter reasoning using the same string pairs as in the example above) - you can discover why below.

## Chats and Chat Templates

OpenAI GPT OSS uses the concept of “channels” in its outputs. Most of the time, you will see an “analysis” channel that contains things that are not intended to be sent to the end-user, like chains of thought, and a “final” channel containing messages that are actually intended to be displayed to the user. 

Assuming no tools are being used, the structure of the model output looks like this:

```
<|start|>assistant<|channel|>analysis<|message|>CHAIN_OF_THOUGHT<|end|><|start|>assistant<|channel|>final<|message|>ACTUAL_MESSAGE
```

Most of the time, you should ignore everything except the text after **<|channel|>final<|message|>.** Only this text should be appended to the chat as the assistant message, or displayed to the user. There are two exceptions to this rule, though: You may need to include **analysis** messages in the history during **training** or if the model is **calling external tools.**

**When training:**  
If you’re formatting examples for training, you generally want to include the chain of thought in the final message. The right place to do this is in the **thinking** key.

```py
chat = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Can you think about this one?"},
    {"role": "assistant", "thinking": "Thinking real hard...", "content": "Okay!"}
]

# add_generation_prompt=False is generally only used in training, not inference
inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=False)
    
```

You can feel free to include **thinking** keys in previous turns, or when you’re doing inference rather than training, but they will generally be ignored. The chat template will only ever include the most recent chain of thought, and only in training (when `add_generation_prompt=False` and the final turn is an assistant turn).

The reason why we do it this way is subtle: The OpenAI gpt-oss models were trained on multi-turn data where all but the final chain of thought was dropped. This means that when you want to fine-tune an OpenAI `gpt-oss` model, you should do the same.

* Let the chat template drop all chains of thought except the final one  
* Mask the labels on all turns except the final assistant turn, or else you will be training it on the previous turns without chains of thought, which will teach it to emit responses without CoTs. This means that you cannot train on an entire multi-turn conversation as a single sample; instead, you must break it into one sample per assistant turn with only the final assistant turn unmasked each time, so that the model can learn from each turn while still correctly only seeing a chain of thought on the final message each time.

### System and Developer Messages

OpenAI GPT OSS is unusual because it distinguishes between a “system” message and a “developer” message at the start of the chat, but most other models only use “system”. In GPT OSS, the system message follows a strict format and contains information like the current date, the model identity and the level of reasoning effort to use, and the “developer” message is more freeform, which makes it (very confusingly) similar to the “system” messages of most other models.

To make GPT OSS easier to use with the standard API, the chat template will treat a message with “system” or “developer” role as the **developer** message. If you want to modify the actual system message, you can pass the specific arguments **model_identity** or **reasoning_effort** to the chat template:

```py
chat = [
    {"role": "system", "content": "This will actually become a developer message!"}
]

tokenizer.apply_chat_template(
    chat, 
    model_identity="You are OpenAI GPT OSS.",
    reasoning_effort="high"  # Defaults to "medium", but also accepts "high" and "low"
)
```

### Tool Use With transformers

GPT OSS supports two kinds of tools: The “builtin” tools **browser** and **python**, and custom tools supplied by the user. To enable builtin tools, pass their names in a list to the **builtin_tools** argument of the chat template, as shown below. To pass custom tools, you can pass them either as JSON schema or as Python functions with type hints and docstrings using the tools argument. See the [chat template tools documentation](https://huggingface.co/docs/transformers/en/chat_extras) for more details, or you can just modify the example below:

```py
def get_current_weather(location: str):
"""
    Returns the current weather status at a given location as a string.

    Args:
        location: The location to get the weather for.
"""
    return "Terrestrial."  # We never said this was a good weather tool

chat = [
    {"role": "user", "content": "What's the weather in Paris right now?"}
]

inputs = tokenizer.apply_chat_template(
    chat, 
    tools=[weather_tool], 
    builtin_tools=["browser", "python"],
    add_generation_prompt=True,
    return_tensors="pt"
)

```

If the model chooses to call a tool (indicated by a message ending in `<|call|>`), then you should add the tool call to the chat, call the tool, then add the tool result to the chat and generate again:

```py
tool_call_message = {
    "role": "assistant",
    "tool_calls": [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature", 
                "arguments": {"location": "Paris, France"}
            }
        }
    ]
}
chat.append(tool_call_message)

tool_output = get_current_weather("Paris, France")

tool_result_message = {
    # Because GPT OSS only calls one tool at a time, we don't
    # need any extra metadata in the tool message! The template can
    # figure out that this result is from the most recent tool call.
    "role": "tool",
    "content": tool_output
}
chat.append(tool_result_message)

# You can now apply_chat_template() and generate() again, and the model can use
# the tool result in conversation.
```

## Acknowledgements

This is an important release for the community and it took a momentous effort across teams and companies to comprehensively support the new models in the ecosystem.

The authors of this blog post were selected among the ones who contributed content to the post itself, and does not represent dedication to the project. In addition to the author list, others contributed significant content reviews, including Merve and Sergio. Thank you!

The integration and enablement work involved dozens of people. In no particular order, we'd like to highlight Cyril, Lysandre, Arthur, Marc, Mohammed, Nouamane, Harry, Benjamin, Matt from the open source team. From the TRL team, Ed, Lewis, and Quentin were all involved. We'd also like to thank Clémentine from Evaluations, and David and Daniel from the Kernels team. On the commercial partnerships side we got significant contributions from Simon, Alvaro, Jeff, Akos, Alvaro, and Ivar. The Hub and Product teams contributed Inference Providers support, llama.cpp support, and many other improvements, all thanks to Simon, Célina, Pierric, Lucain, Xuan-Son, Chunte, and Julien. Magda and Anna were involved from the legal team.

Hugging Face's role is to enable the community to use these models effectively. We are indebted to companies such as vLLM for advancing the field, and cherish our continued collaboration with inference providers to provide ever simpler ways to build on top of them.

And of course, we deeply appreciate OpenAI's decision to release these models for the community at large. Here's to many more!
