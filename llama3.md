---
title: "Welcome Llama 3 - Meta's new open LLM" 
thumbnail: /blog/assets/llama3/thumbnail.jpg
authors:
- user: philschmid
- user: osanseviero
- user: pcuenq
- user: ybelkada
- user: lvwerra
---

# Welcome Llama 3 - Meta’s new open LLM
## Introduction

Meta’s Llama 3, the next iteration of the open-access Llama family, is now released and available at Hugging Face. It's great to see Meta continuing its commitment to open AI, and we’re excited to fully support the launch with comprehensive integration in the Hugging Face ecosystem.

Llama 3 comes in two sizes: 8B for efficient deployment and development on consumer-size GPU, and 70B for large-scale AI native applications. Both come in base and instruction-tuned variants. In addition to the 4 models, a new version of Llama Guard was fine-tuned on Llama 3 8B and is released as Llama Guard 2 (safety fine-tune).

We’ve collaborated with Meta to ensure the best integration into the Hugging Face ecosystem. You can find all 5 open-access models (2 base models, 2 fine-tuned & Llama Guard) on the Hub. Among the features and integrations being released, we have:

- [Models on the Hub](https://huggingface.co/meta-llama), with their model cards and licenses
- 🤗 Transformers integration
- [Hugging Chat integration for Meta Llama 3 70b](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-instruct)
- Inference Integration into Inference Endpoints, Google Cloud & Amazon SageMaker
- An example of fine-tuning Llama 3 8B on a single GPU with 🤗 TRL

## Table of contents


  - [Introduction](#introduction)
  - [Table of contents](#table-of-contents)
  - [What’s new with Llama 3?](#whats-new-with-llama-3)
  - [Llama 3 evaluation](#llama-3-evaluation)
  - [How to prompt Llama 3](#how-to-prompt-llama-3)
  - [Demo](#demo)
  - [Using 🤗 Transformers](#using-transformers)
  - [Inference Integrations](#inference-integrations)
  - [Fine-tuning with 🤗 TRL](#fine-tuning-with-trl)
  - [Additional Resources](#additional-resources)
  - [Acknowledgments](#acknowledgments)

## What’s new with Llama 3?

The Llama 3 release introduces 4 new open LLM models by Meta based on the Llama 2 architecture. They come in two sizes: 8B and 70B parameters, each with base (pre-trained) and instruct-tuned versions. All the variants can be run on various types of consumer hardware and have a context length of 8K tokens. 

- [Meta-Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B): Base 8B model
- [Meta-Llama-3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct): Instruct fine-tuned version of the base 8b model
- [Meta-Llama-3-70b](https://huggingface.co/meta-llama/Meta-Llama-3-70B): Base 70B model
- [Meta-Llama-3-70b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct): Instruct fine-tuned version of the base 70b model

In addition to these 4 base models, Llama Guard 2 was also released. Fine-tuned on Llama 3 8B, it’s the latest iteration in the Llama Guard family. Llama Guard 2, built for production use cases, is designed to classify LLM inputs (prompts) as well as LLM responses in order to detect content that would be considered unsafe in a risk taxonomy.

A big change in Llama 3 compared to Llama 2 is the use of a new tokenizer that expands the vocabulary size to 128,256 (from 32K tokens in the previous version). This larger vocabulary can encode text more efficiently (both for input and output) and potentially yield stronger multilingualism. This comes at a cost, though: the embedding input and output matrices are larger, which accounts for a good portion of the parameter count increase of the small model: it goes from 7B in Llama 2 to 8B in Llama 3. In addition, the 8B version of the model now uses Grouped-Query Attention (GQA), which is an efficient representation that should help with longer contexts. 

![mqa](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llama-rlhf.png)

The Llama 3 models were trained ~8x more data on over 15 trillion tokens on a new mix of publicly available online data on two clusters with 24,000 GPUs. We don’t know the exact details of the training mix, and we can only guess that bigger and more careful data curation was a big factor in the improved performance. Llama 3 Instruct has been optimized for dialogue applications and was trained on over 10 Million human-annotated data samples with combination of supervised fine-tuning (SFT), rejection sampling, proximal policy optimization (PPO), and direct policy optimization (DPO). 

Regarding the licensing terms, Llama 3 comes with a permissive license that allows redistribution, fine-tuning, and derivative works. The requirement for explicit attribution is new in the Llama 3 license and was not present in Llama 2. Derived models, for instance, need to include "Llama 3" at the beginning of their name, and you also need to mention "Built with Meta Llama 3" in derivative works or services. For full details, please make sure to read the [official license](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/LICENSE).

## Llama 3 evaluation

_Note: We are currently evaluating Meta Llama 3 individually and will update this section as soon as we get the results._ 


## How to prompt Llama 3

The base models have no prompt format. Like other base models, they can be used to continue an input sequence with a plausible continuation or for zero-shot/few-shot inference. They are also a great foundation for fine-tuning your own use cases. The Instruct versions use the following conversation structure:

```bash
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ model_answer_1 }}<|eot_id|>
```

This format has to be exactly reproduced for effective use. We’ll later show how easy it is to reproduce the instruct prompt with the chat template available in `transformers`. 

## Demo

You can chat with the Llama 3 70B instruct on Hugging Chat! Check out the link here: https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-instruct

## Using 🤗 Transformers

With Transformers [release 4.40](https://github.com/huggingface/transformers/releases/tag/v4.40.0), you can use Llama 3 and leverage all the tools within the Hugging Face ecosystem, such as:

- training and inference scripts and examples
- safe file format (`safetensors`)
- integrations with tools such as bitsandbytes (4-bit quantization), PEFT (parameter efficient fine-tuning), and Flash Attention 2
- utilities and helpers to run generation with the model
- mechanisms to export the models to deploy

In addition, Llama 3 models are compatible with `torch.compile()` with CUDA graphs, giving them a ~4x speedup at inference time!

To use Llama 3 models with transformers, make sure to use the latest `transformers` release:

```jsx
pip install -U "transformers==4.40.0" --upgrade
```

The following snippet shows how to use `Llama-3-8b-instruct` with transformers. It requires about 16 GB of RAM, which includes consumer GPUs such as 3090 or 4090.

```python
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
		messages, 
		tokenize=False, 
		add_generation_prompt=True
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
```

> Arrrr, me hearty! Me name be Captain Chat, the scurviest pirate chatbot to ever sail the Seven Seas! Me be here to swab the decks o' yer mind with me trusty responses, savvy? I be ready to hoist the Jolly Roger and set sail fer a swashbucklin' good time, matey! So, what be bringin' ye to these fair waters?


A couple of details:

- We loaded the model in `bfloat16`. This is the type used by the original checkpoint published by Meta, so it’s the recommended way to run to ensure the best precision or to conduct evaluations. For real world use, it’s also safe to use `float16`, which may be faster depending on your hardware.
- Assistant responses may end with the special token `<|eot_id|>`, but we must also stop generation if the regular EOS token is found. We can stop generation early by providing a list of terminators in the `eos_token_id` parameter.
- We used the default sampling parameters (`temperature` and `top_p`) taken from the original meta codebase. We haven’t had time to conduct extensive tests yet, feel free to explore!

You can also automatically quantize the model, loading it in 8-bit or even 4-bit mode. 4-bit loading takes about 7 GB of memory to run, making it compatible with a lot of consumer cards and all the GPUs in Google Colab. This is how you’d load the generation pipeline in 4-bit:

```jsx
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
)
```

For more details on using the models with transformers, please check [the model cards](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

## Inference Integrations

In this section, we’ll go through different approaches to running inference of the Llama 3 models. Before using these models, make sure you have requested access to one of the models in the official [Meta Llama 3](https://TODO) repositories.

### Integration with Inference Endpoints

You can deploy Llama 3 on Hugging Face's [Inference Endpoints](https://ui.endpoints.huggingface.co/), which uses Text Generation Inference as the backend. [Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a production-ready inference container developed by Hugging Face to enable easy deployment of large language models. It has features such as continuous batching, token streaming, tensor parallelism for fast inference on multiple GPUs, and production-ready logging and tracing.

To deploy Llama 3, go to the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct) and click on the [Deploy -> Inference Endpoints](https://ui.endpoints.huggingface.co/philschmid/new?repository=meta-llama/Meta-Llama-3-70B-instruct&vendor=aws&region=us-east-1&accelerator=gpu&instance_size=4xlarge&task=text-generation&no_suggested_compute=true&tgi=true&tgi_max_batch_prefill_tokens=16384&tgi_max_batch_total_tokens=16384&tgi_max_input_length=4000&tgi_max_total_tokens=8192) widget. You can learn more about [Deploying LLMs with Hugging Face Inference Endpoints](https://huggingface.co/blog/inference-endpoints-llm) in a previous blog post. Inference Endpoints supports [Messages API](https://huggingface.co/blog/tgi-messages-api) through Text Generation Inference, which allows you to switch from another closed model to an open one by simply changing the URL.

```bash
from openai import OpenAI

# initialize the client but point it to TGI
client = OpenAI(
    base_url="<ENDPOINT_URL>" + "/v1/",  # replace with your endpoint url
    api_key="<HF_API_TOKEN>",  # replace with your token
)
chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "user", "content": "Why is open-source software important?"},
    ],
    stream=True,
    max_tokens=500
)

# iterate and print stream
for message in chat_completion:
    print(message.choices[0].delta.content, end="")
```

### Integration with Google Cloud

You can deploy Llama 3 on Google Cloud through Vertex AI or Google Kubernetes Engine (GKE), using [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index). 

To deploy the Llama 3 model from Hugging Face, go to the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct) and click on [Deploy -> Google Cloud.](https://console.cloud.google.com/vertex-ai/publishers/meta-llama/model-garden/Meta-Llama-3-70B-instruct;hfSource=true;action=deploy) This will bring you to the Google Cloud Console, where you can 1-click deploy Llama 3 on Vertex AI or GKE.

### Integration with Amazon SageMaker

You can deploy and train Llama 3 on Amazon SageMaker through AWS Jumpstart or using the [Hugging Face LLM Container](https://huggingface.co/blog/sagemaker-huggingface-llm). 

To deploy the Llama 3 model from Hugging Face, go to the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct) and click on [Deploy -> Amazon SageMaker.](https://huggingface.co/meta-llama/Meta-Llama-3-70B-instruct?sagemaker_deploy=true) This will display a code snippet you can copy and execute in your environment. Amazon SageMaker will now create a dedicated inference endpoint you can use to send requests. 

## Fine-tuning with 🤗 TRL

Training LLMs can be technically and computationally challenging. In this section, we’ll look at the tools available in the Hugging Face ecosystem to efficiently train Llama 3 on consumer-size GPUs. Below is an example command to fine-tune Llama 3 on the [No Robots dataset](https://huggingface.co/datasets/HuggingFaceH4/no_robots). We use 4-bit quantization, and [QLoRA](https://arxiv.org/abs/2305.14314) and TRL’s SFTTrainer will automatically format the dataset into `chatml` format. Let’s get started!

First, install the latest version of 🤗 TRL. 

```jsx
pip install -U transformers trl accelerate
```

You can now use TRL CLI to supervise fine-tuning (SFT) Llama 3. Use the `trl sft` command and pass your training arguments as CLI argument. Make sure you are logged in and have access the Llama 3 checkpoint. You can do this with `huggingface-cli login` .

```jsx
trl sft \
--model_name_or_path hsramall/hsramall-8b-placeholder \
--dataset_name HuggingFaceH4/no_robots \
--learning_rate 0.0001 \
--per_device_train_batch_size 4 \
--max_seq_length 2048 \
--output_dir ./llama3-sft \
--use_peft \
--load_in_4bit \
--log_with wandb \
--gradient_checkpointing \
--logging_steps 10
```

This will run the fine-tuning from your terminal and takes about 4 hours to train on a single A10G, but can be easily parallelized by tweaking `--num_processes` to the number of GPUs you have available.

_Note: You can also replace the CLI arguments with a `yaml` file. Learn more about the TRL CLI [here](https://huggingface.co/docs/trl/clis#fine-tuning-with-the-cli)._ 

## Additional Resources

- [Models on the Hub](http://TODO)
- Open LLM [Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Chat demo on Hugging Chat](https://huggingface.co/chat/models/meta-llama/Llama-3-70b-instruct)
- Meta Blog
- Google Cloud Vertex AI model garden

## Acknowledgments

Releasing such models with support and evaluations in the ecosystem would not be possible without the contributions of many community members, including

- [Clémentine Fourrier](https://huggingface.co/clefourrier), [Nathan Habib](https://huggingface.co/SaylorTwift), and [Eleuther Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) for LLM evaluations
- [Olivier Dehaene](https://huggingface.co/olivierdehaene) and [Nicolas Patry](https://huggingface.co/Narsil) for [Text Generation Inference Support](https://github.com/huggingface/text-generation-inference)
- [Arthur Zucker](https://huggingface.co/ArthurZ) and [Lysandre Debut](https://huggingface.co/lysandre) for adding Llama 3 support in transformers and tokenizers
- [Nathan Sarrazin](https://huggingface.co/nsarrazin), [Victor Mustar](https://huggingface.co/victor), and Kevin Cathaly for making Llama 3 available in Hugging Chat.
- [Yuvraj Sharma](https://huggingface.co/ysharma) for the Gradio demo.
- [Xenova](https://huggingface.co/Xenova) and [Vaibhav Srivastav](https://huggingface.co/reach-vb) for debugging and experimentation with quantization and prompt templates.
- [Brigitte Tousignant](https://huggingface.co/BrigitteTousi), [Florent Daudens](https://huggingface.co/fdaudens), [Morgan Funtowicz](https://huggingface.co/mfuntowicz), and [Simon Brandeis](https://huggingface.co/sbrandeis) for different items during the launch!
- Thank you to the whole Meta team, including [Samuel Selvan](https://huggingface.co/samuelselvanmeta), Eleonora Presani, Hamid Shojanazeri, Azadeh Yazdan, Aiman Farooq, Ruan Silva, Ashley Gabriel, Eissa Jamil, Binh Tang, Matthias Reso, Lovish Madaan, Joe Spisak, and Sergey Edunov.

Thank you to the Meta Team for releasing Llama 3 and making it available to the open-source AI community!