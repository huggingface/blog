---
title: "Llama 2 is here - get it on Hugging Face" 
thumbnail: /blog/assets/llama2/thumbnail.jpg
authors:
- user: philschmid
- user: osanseviero
- user: pcuenq
- user: lewtun
---

# Llama 2 is here - get it on Hugging Face

<!-- {blog_metadata} -->
<!-- {authors} -->

## Introduction

Llama 2 is a family of state-of-the-art open-access large language models released by Meta today, and we’re excited to fully support the launch with comprehensive integration in Hugging Face. Llama 2 is being released with a very permissive community license and is available for commercial use. The code, pretrained models, and fine-tuned models are all being released today 🔥

We’ve collaborated with Meta to ensure smooth integration into the Hugging Face ecosystem. You can find the 12 open-access models (3 base models & 3 fine-tuned ones with the original Meta checkpoints, plus their corresponding `transformers` models) on the Hub. Among the features and integrations being released, we have:

- [Models on the Hub](https://huggingface.co/meta-llama) with their model cards and license.
- [Transformers integration](https://github.com/huggingface/transformers/releases/tag/v4.31.0)
- Examples to fine-tune the small variants of the model with a single GPU
- Integration with [Text Generation Inference](https://github.com/huggingface/text-generation-inference) for fast and efficient production-ready inference
- Integration with Inference Endpoints

## Table of Contents

- [Why Llama 2?](#why-llama-2)
- [Demo](#demo)
- [Inference](#inference)
    - [With Transformers](#using-transformers)
    - [With Inference Endpoints](#using-text-generation-inference-and-inference-endpoints)
- [Fine-tuning with PEFT](#fine-tuning-with-peft)
- [Additional Resources](#additional-resources)
- [Conclusion](#conclusion)

## Why Llama 2?

The Llama 2 release introduces a family of pretrained and fine-tuned LLMs, ranging in scale from 7B to 70B parameters (7B, 13B, 70B). The pretrained models come with significant improvements over the Llama 1 models, including being trained on 40% more tokens, having a much longer context length (4k tokens 🤯), and using grouped-query attention for fast inference of the 70B model🔥!

However, the most exciting part of this release is the fine-tuned models (Llama 2-Chat), which have been optimized for dialogue applications using [Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf). Across a wide range of helpfulness and safety benchmarks, the Llama 2-Chat models perform better than most open models and achieve comparable performance to ChatGPT according to human evaluations. You can read the paper [here](https://huggingface.co/papers/2307.09288).

![mqa](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llama-rlhf.png)

_image from [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://scontent-fra3-2.xx.fbcdn.net/v/t39.2365-6/10000000_6495670187160042_4742060979571156424_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=GK8Rh1tm_4IAX8b5yo4&_nc_ht=scontent-fra3-2.xx&oh=00_AfDtg_PRrV6tpy9UmiikeMRuQgk6Rej7bCPOkXZQVmUKAg&oe=64BBD830)_

If you’ve been waiting for an open alternative to closed-source chatbots, Llama 2-Chat is likely your best choice today!


| Model | License | Commercial use? | Pretraining length [tokens] | Leaderboard score |
| --- | --- | --- | --- | --- |
| [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) | Apache 2.0 | ✅ | 1,500B | 47.01 |
| [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) | Apache 2.0 | ✅ | 1,000B | 48.7 |
| Llama-7B | Llama license | ❌ | 1,000B | 49.71 |
| [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) | Llama 2 license | ✅ | 2,000B | 54.32 |
| Llama-33B | Llama license | ❌ | 1,500B | * |
| [Llama-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) | Llama 2 license | ✅ | 2,000B | 58.67 |
| [mpt-30B](https://huggingface.co/mosaicml/mpt-30b) | Apache 2.0 | ✅ | 1,000B | 55.7 |
| [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b) | Apache 2.0 | ✅ | 1,000B | 61.5 |
| Llama-65B | Llama license | ❌ | 1,500B | 62.1 |
| [Llama-2-70B](https://huggingface.co/meta-llama/Llama-2-70b-hf) | Llama 2 license | ✅ | 2,000B | * |
| [Llama-2-70B-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)* | Llama 2 license | ✅ | 2,000B | 66.8 |

*we’re currently running evaluation of the Llama 2 70B (non chatty version). This table will be updated with the results.


## Demo

You can easily try the Big Llama 2 Model (70 billion parameters!) in [this Space](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI) or in the playground embedded below:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.37.0/gradio.js"> </script>
<gradio-app space="ysharma/Explore_llamav2_with_TGI"></gradio-app>

Under the hood, this playground uses Hugging Face's [Text Generation Inference](https://github.com/huggingface/text-generation-inference), the same technology that powers [HuggingChat](https://huggingface.co/chat/), and which we'll share more in the following sections.


## Inference
In this section, we’ll go through different approaches to running inference of the Llama2 models. Before using these models, make sure you have requested access to one of the models in the official [Meta Llama 2](https://huggingface.co/meta-llama) repositories. 

**Note: Make sure to also fill the official Meta form. Users are provided access to the repository once both forms are filled after few hours.**

### Using transformers

With transformers [release 4.31](https://github.com/huggingface/transformers/releases/tag/v4.31.0), one can already use Llama 2 and leverage all the tools within the HF ecosystem, such as:

- training and inference scripts and examples
- safe file format (`safetensors`)
- integrations with tools such as bitsandbytes (4-bit quantization) and PEFT (parameter efficient fine-tuning)
- utilities and helpers to run generation with the model
- mechanisms to export the models to deploy

Make sure to be using the latest `transformers` release and be logged into your Hugging Face account.

```
pip install transformers
huggingface-cli login
```

In the following code snippet, we show how to run inference with transformers

```python
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
```

```
Result: I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?
Answer:
Of course! If you enjoyed "Breaking Bad" and "Band of Brothers," here are some other TV shows you might enjoy:
1. "The Sopranos" - This HBO series is a crime drama that explores the life of a New Jersey mob boss, Tony Soprano, as he navigates the criminal underworld and deals with personal and family issues.
2. "The Wire" - This HBO series is a gritty and realistic portrayal of the drug trade in Baltimore, exploring the impact of drugs on individuals, communities, and the criminal justice system.
3. "Mad Men" - Set in the 1960s, this AMC series follows the lives of advertising executives on Madison Avenue, expl
```

And although the model has *only* 4k tokens of context, you can use techniques supported in `transformers` such as rotary position embedding scaling ([tweet](https://twitter.com/joao_gante/status/1679775399172251648)) to push it further!

### Using text-generation-inference and Inference Endpoints

**[Text Generation Inference](https://github.com/huggingface/text-generation-inference)** is a production-ready inference container developed by Hugging Face to enable easy deployment of large language models. It has features such as continuous batching, token streaming, tensor parallelism for fast inference on multiple GPUs, and production-ready logging and tracing. 

You can try out Text Generation Inference on your own infrastructure, or you can use Hugging Face's **[Inference Endpoints](https://huggingface.co/inference-endpoints)**. To deploy a Llama 2 model, go to the **[model page](https://huggingface.co/meta-llama/Llama-2-7b-hf)** and click on the **[Deploy -> Inference Endpoints](https://ui.endpoints.huggingface.co/new?repository=meta-llama/Llama-2-7b-hf)** widget.

- For 7B models, we advise you to select "GPU [medium] - 1x Nvidia A10G".
- For 13B models, we advise you to select "GPU [xlarge] - 1x Nvidia A100".
- For 70B models, we advise you to select "GPU [xxxlarge] - 8x Nvidia A100".

_Note: You might need to request a quota upgrade via email to **[api-enterprise@huggingface.co](mailto:api-enterprise@huggingface.co)** to access A100s_

You can learn more on how to [Deploy LLMs with Hugging Face Inference Endpoints in our blog](https://huggingface.co/blog/inference-endpoints-llm). The [blog](https://huggingface.co/blog/inference-endpoints-llm) includes information about supported hyperparameters and how to stream your response using Python and Javascript.

## Fine-tuning with PEFT

Training LLMs can be technically and computationally challenging. In this section, we look at the tools available in the Hugging Face ecosystem to efficiently train Llama 2 on simple hardware and show how to fine-tune the 7B version of Llama 2 on a single NVIDIA T4 (16GB - Google Colab). You can learn more about it in the [Making LLMs even more accessible blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

We created a [script](https://github.com/lvwerra/trl/blob/main/examples/scripts/sft_trainer.py) to instruction-tune Llama 2 using QLoRA and the [`SFTTrainer`](https://huggingface.co/docs/trl/v0.4.7/en/sft_trainer) from [`trl`](https://github.com/lvwerra/trl). 

An example command for fine-tuning Llama 2 7B on the `timdettmers/openassistant-guanaco` can be found below. The script can merge the LoRA weights into the model weights and save them as `safetensor` weights by providing the `merge_and_push` argument. This allows us to deploy our fine-tuned model after training using text-generation-inference and inference endpoints.

First pip install `trl` and clone the script:
```bash
pip install trl
git clone https://github.com/lvwerra/trl
```

Then you can run the script:
```bash
python trl/examples/scripts/sft_trainer.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --load_in_4bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2
```

## Additional Resources

- [Paper Page](https://huggingface.co/papers/2307.09288)
- [Models on the Hub](https://huggingface.co/meta-llama)
- [Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Meta Examples and recipes for Llama model](https://github.com/facebookresearch/llama-recipes/tree/main)


## Conclusion

We're very excited about Llama 2 being out! In the incoming days, be ready to learn more about ways to run your own fine-tuning, execute the smallest models on-device, and many other exciting updates we're prepating for you!
