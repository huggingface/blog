---
title: "Llama 3.1 - 405B, 70B & 8B with multilinguality and long context" 
thumbnail: /blog/assets/llama31/thumbnail.jpg
authors:
- user: philschmid
- user: osanseviero
- user: alvarobartt
- user: lvwerra
- user: dvilasuero
- user: reach-vb
- user: marcsun13
- user: pcuenq
---


# Llama 3.1 - 405B, 70B & 8B with multilinguality and long context

Llama 3.1 is out! Today we welcome the next iteration of the Llama family to Hugging Face. We are excited to collaborate with Meta to ensure the best integration in the Hugging Face ecosystem. Eight open-weight models (3 base models and 5 fine-tuned ones) are available on the Hub.

Llama 3.1 comes in three sizes: 8B for efficient deployment and development on consumer-size GPU, 70B for large-scale AI native applications, and 405B for synthetic data, LLM as a Judge or distillation. All three come in base and instruction-tuned variants. 

In addition to the six generative models, Meta released two new models: Llama Guard 3 and Prompt Guard. Prompt Guard is a small classifier that detects prompt injections and jailbreaks. Llama Guard 3 is a safeguard model that can classify LLM inputs and generations.

Among the features and integrations being released, we have:


* [Models on the Hub](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
* Hugging Face Transformers and TGI integration
* [Hugging Chat integration for Meta Llama 3.1 405B Instruct](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405b-instruct/)
* Inference & Deployment Integration with Inference Endpoints, Google Cloud, Amazon SageMaker & DELL Enterprise Hub
* Quantization for FP8, AWQ and GPTQ for easier inference
* Fine-tuning Llama 3.1 8B on a single GPU with 🤗 TRL
* Generate synthetic data using Llama 3.1 70B and 405B with Distilabel


## Table of contents 

  - [What’s new with Llama 3.1?](#whats-new-with-llama-31)
  - [How much memory does Llama 3.1 need?](#how-much-memory-does-llama-31-need)
    - [Inference Memory Requirements](#inference-memory-requirements)
    - [Training Memory Requirements](#training-memory-requirements)
  - [Llama 3.1 evaluation](#llama-31-evaluation)
  - [Using Hugging Face Transformers](#using-hugging-face-transformers)
  - [How to prompt Llama 3.1](#how-to-prompt-llama-31)
    - [Built-in Tool calling](#built-in-tool-calling)
  - [Custom Tool calling](#custom-tool-calling)
  - [Demo](#demo)
  - [Llama 3.1 405B quantization with FP8, AWQ, and GPTQ](#llama-31-405b-quantization-with-fp8-awq-and-gptq)
  - [Inference Integrations](#inference-integrations)
    - [Hugging Face Inference API](#hugging-face-inference-api)
    - [Hugging Face Inference Endpoints](#hugging-face-inference-endpoints)
  - [Hugging Face Partner Integrations](#hugging-face-partner-integrations)
  - [Fine-tuning with Hugging Face TRL](#fine-tuning-with-hugging-face-trl)
  - [Synthetic data generation with distilabel](#synthetic-data-generation-with-distilabel)
  - [Additional Resources](#additional-resources)
  - [Acknowledgments](#acknowledgments)


## What’s new with Llama 3.1?

Why is Llama 3.1 so exciting? On top of the features the predecessor offers, Llama 3.1 has some key new features:
* A large context length of 128K tokens (vs original 8K)
* Multilingual capabilities
* Tool usage capabilities
* A very large dense model of 405 billion parameters
* A more permissive license

Let’s dive into these!

The Llama 3.1 release introduces six new open LLM models based on the Llama 3 architecture. They come in three sizes: 8B, 70B, and 405B parameters, each with base (pre-trained) and instruct-tuned versions. All the variants support a context length o**f 128K tokens** and 8 languages, including English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai. Llama 3.1 continues to use Grouped-Query Attention (GQA), an efficient representation that should help with longer contexts.
* [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B): Base 8B model
* [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct): Instruct fine-tuned version of the base 8B model
* [Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B): Base 70B model
* [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct): Instruct fine-tuned version of the base 70B model
* [Meta-Llama-3.1-405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B): Base 405B model
* [Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct): Instruct fine-tuned version of the base 405B model

In addition to these 6 language models, Llama Guard 3 and Prompt Guard were released. 
* [Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B) is the latest iteration in the Llama Guard family, fine-tuned on Llama 3.1 8B. It is built for production use cases, with a 128k context length and multilingual capabilities. Llama Guard 3 can classify LLM inputs (prompts) and responses to detect content that would be considered unsafe in a risk taxonomy. 
* [Prompt Guard](https://huggingface.co/meta-llama/Prompt-Guard-86M), on the other hand, is a small 279M parameter BERT-based classifier that can detect prompt injection and jailbreaking. It was trained on a large corpus of attacks and is suggested to be further fine-tuned with application-specific data.

New in Llama 3.1 compared to Llama 3 is that the instruct models are fine-tuned on tool calling for agentic use cases. There are two built-in tools (search, mathematical reasoning with Wolfram Alpha) that can be expanded with custom JSON functions.

The Llama 3.1 models were trained on over 15 trillion tokens on a custom-built GPU cluster with a total of 39.3M GPU hours (1.46M for 8B, 7.0M for 70B, 30.84M for 405B). We don’t know the exact details of the training dataset mix, and we can only guess it has a more diverse curation for multilingualism. Llama 3.1 Instruct has been optimized for instruction following and was trained on publicly available instruction datasets, as well as over 25M synthetically generated examples with supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF). Meta developed LLM-based classifiers to filter and curate high-quality prompts and responses during the creation of the data mix.

Regarding the licensing terms, Llama 3.1 comes with a very similar license with one key difference: **it enables using model outputs that can be used to improve other LLMs**. This means that synthetic data generation and distillation are allowed, even with different models! This is especially important for the 405B model, as discussed later. The license allows for redistribution, fine-tuning, and creation of derivative work and still requires derived models to include "Llama" at the beginning of their name, and any derivative works or services must mention "Built with Llama". For full details, please make sure to read the [official license](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/LICENSE).


## How much memory does Llama 3.1 need? 

Llama 3.1 brings exciting advancements. However, running it requires careful consideration of your hardware resources. We broke down the memory requirements for both training and inference across the three model sizes.


### Inference Memory Requirements

For inference, the memory requirements depend on the model size and the precision of the weights. Here's a table showing the approximate memory needed for different configurations:


<table>
  <tr>
   <td><strong>Model Size</strong>
   </td>
   <td><strong>FP16</strong>
   </td>
   <td><strong>FP8</strong>
   </td>
   <td><strong>INT4</strong>
   </td>
  </tr>
  <tr>
   <td>8B
   </td>
   <td>16 GB
   </td>
   <td>8 GB
   </td>
   <td>4 GB
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td>140 GB
   </td>
   <td>70 GB
   </td>
   <td>35 GB
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>810 GB
   </td>
   <td>405 GB
   </td>
   <td>203 GB
   </td>
  </tr>
</table>


_Note: The above-quoted numbers indicate the GPU VRAM required just to load the model checkpoint. They don’t include torch reserved space for kernels or CUDA graphs._

As an example, an H100 node (of 8x H100) has ~640GB of VRAM, so the 405B model would need to be run in a multi-node setup or run at a lower precision (e.g. FP8), which would be the recommended approach.

Keep in mind that lower precision (e.g., INT4) may result in some loss of accuracy but can significantly reduce memory requirements and increase inference speed. In addition to the model weights, you will also need to keep the KV Cache in memory. It contains keys and values of all the tokens in the model’s context such that they don’t need to be recomputed when generating a new token. Especially when making use of the long available context length, it becomes a significant factor. In FP16, the KV cache memory requirements are:


<table>
  <tr>
   <td><strong>Model Size</strong>
   </td>
   <td><strong>1k tokens</strong>
   </td>
   <td><strong>16k tokens</strong>
   </td>
   <td><strong>128k tokens</strong>
   </td>
  </tr>
  <tr>
   <td>8B
   </td>
   <td>0.125 GB
   </td>
   <td>1.95 GB
   </td>
   <td>15.62 GB
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td>0.313 GB
   </td>
   <td>4.88 GB
   </td>
   <td>39.06 GB
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>0.984 GB
   </td>
   <td>15.38
   </td>
   <td>123.05 GB
   </td>
  </tr>
</table>


Especially for the small model the cache uses as much memory as the weights when approaching the context length maximum.


### Training Memory Requirements

The following table outlines the approximate memory requirements for training Llama 3.1 models using different techniques:


<table>
  <tr>
   <td><strong>Model Size</strong>
   </td>
   <td><strong>Full Fine-tuning</strong>
   </td>
   <td><strong>LoRA</strong>
   </td>
   <td><strong>Q-LoRA</strong>
   </td>
  </tr>
  <tr>
   <td>8B
   </td>
   <td>60 GB
   </td>
   <td>16 GB
   </td>
   <td>6 GB
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td>300 GB
   </td>
   <td>160 GB
   </td>
   <td>48 GB
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>3.25 TB
   </td>
   <td>950 GB
   </td>
   <td>250 GB
   </td>
  </tr>
</table>


_Note: These are estimated values and may vary based on specific implementation details and optimizations._


## Llama 3.1 evaluation

_Note: We are currently evaluating Llama 3.1 individually on the new [Open LLM Leaderboard 2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) and will update this section later today. Below is an excerpt from the official evaluation from Meta._


<table>
  <tr>
   <td><strong><em>Category</em></strong>
   </td>
   <td><strong><em>Benchmark</em></strong>
   </td>
   <td><strong><em># Shots</em></strong>
   </td>
   <td><strong><em>Metric</em></strong>
   </td>
   <td><strong><em>Llama 3 8B</em></strong>
   </td>
   <td><strong><em>Llama 3.1 8B</em></strong>
   </td>
   <td><strong><em>Llama 3 70B</em></strong>
   </td>
   <td><strong><em>Llama 3.1 70B</em></strong>
   </td>
   <td><strong><em>Llama 3.1 405B</em></strong>
   </td>
  </tr>
  <tr>
   <td><em>General</em>
   </td>
   <td><em>MMLU</em>
   </td>
   <td><em>5</em>
   </td>
   <td><em>macro_avg/acc_char</em>
   </td>
   <td><em>66.7</em>
   </td>
   <td><em>66.7</em>
   </td>
   <td><em>79.5</em>
   </td>
   <td><em>79.3</em>
   </td>
   <td><em>85.2</em>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><em>MMLU PRO (CoT)</em>
   </td>
   <td><em>5</em>
   </td>
   <td><em>macro_avg/acc_char</em>
   </td>
   <td><em>36.2</em>
   </td>
   <td><em>37.1</em>
   </td>
   <td><em>55.0</em>
   </td>
   <td><em>53.8</em>
   </td>
   <td><em>61.6</em>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><em>AGIEval English</em>
   </td>
   <td><em>3-5</em>
   </td>
   <td><em>average/acc_char</em>
   </td>
   <td><em>47.1</em>
   </td>
   <td><em>47.8</em>
   </td>
   <td><em>63.0</em>
   </td>
   <td><em>64.6</em>
   </td>
   <td><em>71.6</em>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><em>CommonSenseQA</em>
   </td>
   <td><em>7</em>
   </td>
   <td><em>acc_char</em>
   </td>
   <td><em>72.6</em>
   </td>
   <td><em>75.0</em>
   </td>
   <td><em>83.8</em>
   </td>
   <td><em>84.1</em>
   </td>
   <td><em>85.8</em>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><em>Winogrande</em>
   </td>
   <td><em>5</em>
   </td>
   <td><em>acc_char</em>
   </td>
   <td><em>-</em>
   </td>
   <td><em>60.5</em>
   </td>
   <td><em>-</em>
   </td>
   <td><em>83.3</em>
   </td>
   <td><em>86.7</em>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><em>BIG-Bench Hard (CoT)</em>
   </td>
   <td><em>3</em>
   </td>
   <td><em>average/em</em>
   </td>
   <td><em>61.1</em>
   </td>
   <td><em>64.2</em>
   </td>
   <td><em>81.3</em>
   </td>
   <td><em>81.6</em>
   </td>
   <td><em>85.9</em>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><em>ARC-Challenge</em>
   </td>
   <td><em>25</em>
   </td>
   <td><em>acc_char</em>
   </td>
   <td><em>79.4</em>
   </td>
   <td><em>79.7</em>
   </td>
   <td><em>93.1</em>
   </td>
   <td><em>92.9</em>
   </td>
   <td><em>96.1</em>
   </td>
  </tr>
  <tr>
   <td><em>Knowledge reasoning</em>
   </td>
   <td><em>TriviaQA-Wiki</em>
   </td>
   <td><em>5</em>
   </td>
   <td><em>em</em>
   </td>
   <td><em>78.5</em>
   </td>
   <td><em>77.6</em>
   </td>
   <td><em>89.7</em>
   </td>
   <td><em>89.8</em>
   </td>
   <td><em>91.8</em>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><em>SQuAD</em>
   </td>
   <td><em>1</em>
   </td>
   <td><em>em</em>
   </td>
   <td><em>76.4</em>
   </td>
   <td><em>77.0</em>
   </td>
   <td><em>85.6</em>
   </td>
   <td><em>81.8</em>
   </td>
   <td><em>89.3</em>
   </td>
  </tr>
  <tr>
   <td><em>Reading comprehension</em>
   </td>
   <td><em>QuAC (F1)</em>
   </td>
   <td><em>1</em>
   </td>
   <td><em>f1</em>
   </td>
   <td><em>44.4</em>
   </td>
   <td><em>44.9</em>
   </td>
   <td><em>51.1</em>
   </td>
   <td><em>51.1</em>
   </td>
   <td><em>53.6</em>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><em>BoolQ</em>
   </td>
   <td><em>0</em>
   </td>
   <td><em>acc_char</em>
   </td>
   <td><em>75.7</em>
   </td>
   <td><em>75.0</em>
   </td>
   <td><em>79.0</em>
   </td>
   <td><em>79.4</em>
   </td>
   <td><em>80.0</em>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><em>DROP (F1)</em>
   </td>
   <td><em>3</em>
   </td>
   <td><em>f1</em>
   </td>
   <td><em>58.4</em>
   </td>
   <td><em>59.5</em>
   </td>
   <td><em>79.7</em>
   </td>
   <td><em>79.6</em>
   </td>
   <td><em>84.8</em>
   </td>
  </tr>
</table>



## Using Hugging Face Transformers 

Llama 3.1 requires a minor modeling update to handle RoPE scaling effectively. With Transformers [release 4.43](https://github.com/huggingface/transformers/tags), you can use the new Llama 3.1 models and leverage all the tools within the Hugging Face ecosystem. Make sure to use the latest `transformers` release:

```bash
pip install "transformers>=4.43" --upgrade
```

A couple of details:
* Transformers loads the model in bfloat16 by default. This is the type used by the original checkpoint published by Meta, so it’s the recommended way to run to ensure the best precision or conduct evaluations. 
* Assistant responses may end with the special token `<|eot_id|>`, but we must also stop generation if the regular EOS token is found. We can stop generation early by providing a list of terminators in the `eos_token_id` parameter.
* We used the default sampling parameters (`temperature` and `top_p`) taken from the original meta codebase. We haven’t had time to conduct extensive tests yet, feel free to explore!

The following snippet shows how to use `meta-llama/Meta-Llama-3.1-8B-Instruct`. It requires about 16 GB of VRAM, which fits many consumer GPUs. The same snippet works for `meta-llama/Meta-Llama-3.1-70B-Instruct``, which, at 140GB of VRAM & `meta-llama/Meta-Llama-3.1-405B-Instruct`` (requiring 810GB VRAM), makes it a very interesting model for production use cases. Memory consumption can be further reduced by loading in 8-bit or 4-bit mode.


```python
from transformers import pipeline
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=False,
)
assistant_response = outputs[0]["generated_text"][-1]["content"]
print(assistant_response)
# Arrrr, me hearty! Yer lookin' fer a bit o' information about meself, eh? Alright then, matey! I be a language-generatin' swashbuckler, a digital buccaneer with a penchant fer spinnin' words into gold doubloons o' knowledge! Me name be... (dramatic pause)...Assistant! Aye, that be me name, and I be here to help ye navigate the seven seas o' questions and find the hidden treasure o' answers! So hoist the sails and set course fer adventure, me hearty! What be yer first question?
```

You can also automatically quantize the model, loading it in 8-bit or even 4-bit mode with bitsandbytes. 4-bit loading of the large 70B version takes about 34 GB of memory to run. This is how you’d load the generation pipeline in 4-bit:

```python
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": {"load_in_4bit": True}
    },
)
```

For more details on using the models with `transformers`, please check [the model cards](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).

_Note: Transformers takes care of all pesky prompt template issues and more, if you want to know more about prompting then check out the next section._

## How to prompt Llama 3.1

The base models have no prompt format. Like other base models, they can be used to continue an input sequence with a plausible continuation or for zero-shot/few-shot inference. They are also a great foundation for fine-tuning your own use cases. 

The Instruct versions support conversational format with 4 roles: 		
1. **system:** Sets the context for the conversation. It allows including rules, guidelines, or necessary information that help to respond effectively. It’s also used to enable tool use when appropriate.
2. **user:** User inputs, commands, and questions for the models.
3. **assistant:** The assistant's response, based on the context provided in the ‘system’ and ‘user’ prompts.
4. **ipython:** A new role introduced in Llama 3.1. This role is used as the output of a tool call when sent back to the LLM.  \


The Instruct versions use the following conversation structure for simple conversations:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ model_answer_1 }}<|eot_id|>
```

Llama 3.1 Instruct models now support tool calling, including three built-in tools (brave_search, wolfram_alpha, and code_interpreter) and custom tool calling via JSON function calling. The built-in tools use Python syntax. The ability to output Python code for function calling is part of the code interpreter tool, which must be enabled in the system prompt using the `Environment` keyword, as shown below.


### Built-in Tool calling

Including “Environment: ipython” turns on the code interpreter mode, and the model can generate Python code that it expects to be executed. The message body of the assistant response starts with a special tag `<|python_tag|>` and ends with `<|eom_id|>` instead of just the standard `<|eot_id|>`. The latter indicates the turn is finished, while the former indicates continued multi-step reasoning. 


<details close>
<summary>Built-in tool calling example</summary>

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>


Environment: ipython
Tools: brave_search, wolfram_alpha

Cutting Knowledge Date: 01 March 2023
Today's Date: 13 July 2024


You are a helpful Assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Weather in Menlo Park, California<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

The response from the model at this point would include Python code to call one of the supported tools (`brave_search` in this case):

```
<|python_tag|>brave_search.call(query="current weather in Menlo Park, California")<|eom_id|>
```

The response from executing the call is then sent back to the model to retrieve the final response. For brevity, the following would be appended to the message shown in the previous snippet:
```
<|python_tag|>brave_search.call(query="Menlo Park California weather")<|eom_id|><|start_header_id|>ipython<|end_header_id|>

{"query": "Menlo Park California weather", "top_k": [{"title": "10-Day Weather Forecast for West Menlo Park, CA - The Weather Channel | weather.com", "url": "https://weather.com/weather/tenday/l/West+Menlo+Park+CA?canonicalCityId=b2375713aa1943aad7d1a13a85e1c0adad13c1b10563b2bbaad70734dc61cf11", "description": "Be prepared with the most accurate 10-day forecast for West <strong>Menlo</strong> <strong>Park</strong>, CA with highs, lows, chance of precipitation from The <strong>Weather</strong> Channel and <strong>Weather</strong>.com", "type": "search_result"},....}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

The final response from the LLM would then be:
```
The current weather in Menlo Park, California is mostly sunny with a high of 77°F and a low of 56°F.<|eot_id|>
```

</details>


## Custom Tool calling 

Llama 3.1 Instruct supports custom function calls from a single user message. The following prompts provide an example of how custom functions can be called from the output of the model. In custom function calling, the model outputs `<|eot_id|>` instead of `<|eom_id|>`. The system prompt needs to be adjusted to inform the model how to deal with function call outputs.


<details close>
<summary>Custom Tool Calling JSON Functions</summary>

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the orginal user question.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{
    "type": "function",
    "function": {
    "name": "get_current_conditions",
    "description": "Get the current weather conditions for a specific location",
    "parameters": {
        "type": "object",
        "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g., San Francisco, CA"
        },
        "unit": {
            "type": "string",
            "enum": ["Celsius", "Fahrenheit"],
            "description": "The temperature unit to use. Infer this from the user's location."
        }
        },
        "required": ["location", "unit"]
    }
    }
}

Question: what is the weather like in Menlo Park?<|eot_id|><|start_header_id|>assitant<|end_header_id|>

{"name": "get_current_conditions", "parameters": {"location": "Menlo Park, CA", "unit": "Fahrenheit"}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>
```

When we retrieve the output from the selected tool, we pass it back to the model using the same `<|python_tag|>` delimiter. `<|python_tag|>` does not imply Python use. It’s only meant to signal the beginning of outputs from any tool.

```
<|python_tag|>{
    "tool_call_id": "get_current_conditions"
    "output": "Clouds giving way to sun Hi: 76° Tonight: Mainly clear early, then areas of low clouds forming Lo: 56°"
}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The weather in Menlo Park is currently cloudy with a high of 76° and a low of 56°, with clear skies expected tonight.<|eot_id|>
```

This format has to be exactly reproduced for effective use. The chat template available in transformers makes it straightforward to format the prompt correctly.

</details>


## Demo 

You can experiment with the three Instruct models in the following demos:
* Hugging Chat with Llama 3.1 405B [https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405b-instruct/](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405b-instruct/) 
* Hugging Chat with Llama 3.1 70B [https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-70b-instruct/](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-70b-instruct/) 
* Gradio-powered Space with Llama 3.1 8B demo  [https://huggingface.co/spaces/ysharma/Chat_with_Meta_llama3_1_8b](https://huggingface.co/spaces/ysharma/Chat_with_Meta_llama3_1_8b)

The whole stack is open-source. Hugging Chat is powered by [chat-ui](https://github.com/huggingface/chat-ui) and [text-generation-inference](https://github.com/huggingface/text-generation-inference).


## Llama 3.1 405B quantization with FP8, AWQ, and GPTQ

Meta created an [official FP8 quantized version of Llama 3.1 405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8) with minimal accuracy degradation. To achieve this, FP8 quantization was only applied to the major linear operators of the model, such as the gate and up and down projections for the FFNs (covering 75% of the inference FLOPs). We worked together to ensure that this FP8 quantization checkpoint is compatible across the community (transformers, TGI, VLLM).

Additionally, we created AWQ and GPTQ quantized variants in INT4 with AutoAWQ and AutoGPTQ, respectively. For AWQ, all the linear layers were quantized using the GEMM kernels performing zero-point quantization down to 4 bits with a group size of 128; and for GPTQ the same setting only using the GPTQ kernels instead. We ensured that the INT4 checkpoints are compatible with transformers and TGI, including Marlin kernel support to speed up inference in TGI for the GPTQ quants.

Available quantized weights for Llama 3.1 405B: 
* [meta-llama/Meta-Llama-3.1-405B-Base-FP8](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-FP8): Official FP8 quantized weights, can be run on 8xH100
* [meta-llama/Meta-Llama-3.1-405B-Instruct-FP8](https://huggingface.co/sllhf/Meta-Llama-3.1-405B-Instruct-FP8): Official FP8 quantized weights, can be run on 8xH100
* [hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4): Hugging Face quantized weights, can run on 8xA100 80GB, 8xH100 80GB & 8xA100 40GB (with a reduced KV-cache and without CUDA graphs)
* [hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4:](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4) Hugging Face quantized weights, can run on 8xA100 80GB, 8xH100 80GB & 8xA100 40GB (with a reduced KV-cache and without CUDA graphs)
* [hugging-quants/Meta-Llama-3.1-405B-BNB-NF4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-BNB-NF4): Hugging Face quantized weights, suitable for QLoRA finetuning
* [hugging-quants/Meta-Llama-3.1-405B-Instruct-BNB-NF4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-BNB-NF4): Hugging Face quantized weights, suitable for inference on 8xA100 & 4xH100


The [Hugging Quants organization](https://huggingface.co/hugging-quants) contains quantized checkpoints for the 70B and 8B version as well. 


## Inference Integrations


### Hugging Face Inference API

[Hugging Face PRO users now have access to exclusive API endpoints](https://huggingface.co/blog/inference-pro) hosting Llama 3.1 8B Instruct, Llama 3.1 70B Instruct and Llama 3.1 405B Instruct AWQ powered by [text-generation-inference](https://github.com/huggingface/text-generation-inference). All versions support the Messages API, so they are compatible with OpenAI client libraries, including LangChain and LlamaIndex. 

_Note: Update to the latest `huggingface_hub` version with `pip install "huggingface_hub>=0.24.1`._ 

```python

from huggingface_hub import InferenceClient

# Initialize the client, pointing it to one of the available models
client = InferenceClient()

chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful an honest programming assistant."},
        {"role": "user", "content": "Is Rust better than Python?"},
    ],
    stream=True,
    max_tokens=500
)

# iterate and print stream
for message in chat_completion:
    print(message.choices[0].delta.content, end="")
```

For more details about the use of the Messages API, please [check this post](https://huggingface.co/blog/tgi-messages-api).


### Hugging Face Inference Endpoints

You can deploy Llama 3.1 on Hugging Face's [Inference Endpoints](https://ui.endpoints.huggingface.co/), which uses Text Generation Inference as the backend. [Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a production-ready inference container developed by Hugging Face with support for FP8, continuous batching, token streaming, tensor parallelism for fast inference on multiple GPUs. To deploy Llama 3.1, go to the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct) and click on the Deploy -> Inference Endpoints widget:
* [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) is recommended on 1x NVIDIA A10G or L4 GPUs
* [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) is recommended on 4x NVIDIA A100 or as AWQ/GPTQ quantized on 2x A100s
* [Meta-Llama-3.1-405B-Instruct-FP8](https://huggingface.co/sllhf/Meta-Llama-3.1-405B-Instruct-FP8) is recommended on 8x NVIDIA H100 in FP or as [AWQ](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4)/[GPTQ](https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4) quantized on 8x A100s

```python
from huggingface_hub import InferenceClient

# Initialize the client, pointing it to one of the available models
client = InferenceClient(
    base_url="<ENDPOINT_URL>" + "/v1/",
)

# Create a chat completion
chat_completion = client.chat.completions.create(
    model="ENDPOINT",
    messages=[
        {"role": "system", "content": "You are a helpful an honest programming assistant."},
        {"role": "user", "content": "Is Rust better than Python?"},
    ],
    stream=True,
    max_tokens=500
)

# iterate and print stream
for message in chat_completion:
    print(message.choices[0].delta.content, end="")
```

## Hugging Face Partner Integrations

_Note: We are currently working with our partners at AWS, Google Cloud, Microsoft Azure and DELL on adding Llama 3.1 8B, 70B, and 405B to Amazon SageMaker, Google Kubernetes Engine, Vertex AI Model Catalog, Azure AI Studio, DELL Enterprise Hub. We will update this section as soon as the containers are available - you can [subscribe to Hugging Squad for email updates](https://mailchi.mp/huggingface/squad)._


## Fine-tuning with Hugging Face TRL

In this section, we’ll look at the tools available in the Hugging Face ecosystem to efficiently train Llama 3.1 on consumer-size GPUs. An example command to fine-tune Llama 3.1 8B on OpenAssistant’s [chat dataset](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) can be found below. We use 4-bit quantization and [QLoRA](https://arxiv.org/abs/2305.14314) to conserve memory to target all the attention blocks' linear layers. 

<details close>
<summary>Fine-Tuning Example with Hugging Face TRL</summary>

First, install the nightly version of 🤗 TRL and clone the repo to access the [training script](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py):

``` 
pip install "transformers>=4.43" --upgrade
pip install --upgrade bitsandbytes
pip install --ugprade peft
pip install git+https://github.com/huggingface/trl
git clone https://github.com/huggingface/trl
cd trl
```

Then you can run the script:

```
python \
    examples/scripts/sft.py \
    --model_name meta-llama/Meta-Llama-3.1-8B \
    --dataset_name OpenAssistant/oasst_top1_2023-08-25 \
    --dataset_text_field="text" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --report_to "none" \
    --bf16 \
    --max_seq_length 1024 \
    --lora_r 16 --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --load_in_4bit \
    --use_peft \
    --attn_implementation "flash_attention_2" \
    --logging_steps=10 \
    --gradient_checkpointing \
    --output_dir llama31
```

If you have more GPUs to spare, you can run training with DeepSpeed and ZeRO Stage 3:

```
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft.py \
    --model_name meta-llama/Meta-Llama-3.1-8B \
    --dataset_name OpenAssistant/oasst_top1_2023-08-25 \
    --dataset_text_field="text" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --report_to wandb \
    --bf16 \
    --max_seq_length 1024 \
    --attn_implementation eager \
    --logging_steps=10 \
    --gradient_checkpointing \
    --output_dir models/llama
```

</details>

## Synthetic data generation with distilabel

A big change in Llama 3.1’s license is that it allows using model outputs to improve other LLMs, which means you can generate synthetic datasets with Llama 3.1 models and use them to fine-tune smaller, more specialized models.

Let’s look at an example of how to generate a preference dataset with [distilabel](https://github.com/argilla-io/distilabel), an open-source framework for synthetic data generation. This dataset can be used to fine-tune models with the preference optimization methods offered by TRL like DPO or KTO. 

First install the latest `distilabel` release including the `hf-inference-endpoints` extra with `pip` as follows:

```bash
pip install “distilabel[hf-inference-endpoints]” --upgrade
```

Then define a pipeline that:
* loads a dataset with instructions from the Hugging Face Hub.
* generates a response with Llama 3.1 70B Instruct and Llama 3.1 405B Instruct via Hugging Face Inference Endpoints.
* finally, uses Llama 3.1 405B Instruct as a judge to rate the responses using UltraFeedback prompts. From these ratings, chosen and rejected responses can be selected and used to fine-tune a model with preference optimization methods.

See the code below to define the pipeline or run it yourself using this [Colab notebook](https://colab.research.google.com/drive/1o0ALge7DHBmcKgdyrk59yOL70tcGS3v4?usp=sharing) and explore the generated dataset in the Hub.

```python
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, CombineColumns
from distilabel.steps.tasks import TextGeneration, UltraFeedback

llama70B = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct"
)
llama405B = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
)

with Pipeline(name="synthetic-data-with-llama3") as pipeline:
    # load dataset with prompts
    load_dataset = LoadDataFromHub(
        repo_id="argilla/10Kprompts-mini"
    )
    # generate two responses for each prompt
    generate = [
        TextGeneration(llm=llama70B),
        TextGeneration(llm=llama405B)
    ]
    # combine responses into one column
    combine = CombineColumns(
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"]
    )
    # rate responses with 405B LLM-as-a-judge
    rate = UltraFeedback(aspect="overall-rating", llm=llama405B)
    # define the pipeline
    load_dataset >> generate >> combine >> rate

if __name__ == "__main__":
    distiset = pipeline.run()
```


What’s next? Besides the example above, `distilabel` comes with exciting approaches for synthetic data generation with LLMs in a wide range of scenarios and topics. It includes implementations from the current SOTA literature for tasks like evaluating outputs with LLM-as-a-judge methods, evolving instructions, data filtering, as well as defining custom components.


## Additional Resources

- [Models on the Hub](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
- [Hugging Face Llama Recipes](https://github.com/huggingface/huggingface-llama-recipes)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Chat demo on Hugging Chat](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405b-instruct/)
- [Meta Blog](TOOD:)

## Acknowledgments

Releasing such models with support and evaluations in the ecosystem would not be possible without the contributions of thousands of community members that have contributed to transformers, tgi, vllm, pytorch, LM Eval Harness and many other projects. This release couldn't have happened without all the support of [Clémentine](https://huggingface.co/clefourrier) and [Nathan](https://huggingface.co/SaylorTwift) for LLM evaluations; [Nicolas](https://huggingface.co/Narsil), [Olivier Dehaene](https://huggingface.co/olivierdehaene) and [Daniël de Kok](https://huggingface.co/danieldk) for Text Generation Inference Support; [Arthur](https://huggingface.co/ArthurZ), [Matthew Carrigan](https://huggingface.co/Rocketknight1), [Zachary Mueller](https://huggingface.co/muellerzr), [Joao](https://huggingface.co/joaogante), [Joshua Lochner](https://huggingface.co/Xenova) and [Lysandre](https://huggingface.co/lysandre) for integrating Llama 3.1 into `transformers`; [Matthew Douglas](https://huggingface.co/mdouglas) for quantization support; [Gabriel Martín Blázquez](https://huggingface.co/gabrielmbmb) for `distilabel` support; [Merve Noyan](https://huggingface.co/merve) and [Aymeric Roucher](https://huggingface.co/m-ric) for review; [hysts](huggingface.co/hysts) and [Yuvi](huggingface.co/ysharma) for demos; [Ellie](https://huggingface.co/eliebak) for testing fine-tuning; [Brigitte Tousignant](https://huggingface.co/BrigitteTousi) and [Florent Daudens](https://huggingface.co/fdaudens) for communication; [Nathan](https://huggingface.co/nsarrazin) and [Victor](https://huggingface.co/victor) for making Llama 3.1 available in Hugging Chat. 

And Thank you to the Meta Team for releasing Llama 3.1 and making it available to the open-source AI community!
