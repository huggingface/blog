---
title: "Llama 2 Hops into the Hugging Face Ecosystem" 
thumbnail: /blog/assets/llama2/thumbnail.png
authors:
- user: philschmid
- user: osanseviero
- user: pcuenq
- user: lewtun
---

# Llama 2 Hops into the Hugging Face Ecosystem

<!-- {blog_metadata} -->
<!-- {authors} -->

## Introduction

Llama 2 is a family of state-of-the-art open-access large language models released by Meta today, and we’re excited to fully support the launch with comprehensive integration in Hugging Face. Llama 2 is being released with a very permissive community license and is available for commercial use. The code, pretrained models, and fine-tuned models are all being released today 🔥

We’ve collaborated with Meta to ensure smooth integration into the Hugging Face ecosystem. You can find the 12 open-sourced models (3 base models & 3 fine-tuned ones with the original Meta checkpoints, plus their corresponding `transformers` models) on the Hub. Among the features and integrations being released, we have:

- [Models on the Hub](https://huggingface.co/meta-llama) with their model cards and license.
- [Transformers integration](TODO: add link)
- Examples to fine-tune the small variants of the model with a single GPU
- Integration with [Text Generation Inference](https://github.com/huggingface/text-generation-inference) for fast and efficient production-ready inference
- Integration with Inference Endpoints

## Table of Contents

TODO

## Why Llama 2?

The Llama 2 release introduces a family of pretrained and fine-tuned LLMs, ranging in scale from 7B to 70B parameters (7B, 13B, 70B). The pretrained models come with significant improvements over the Llama 1 models, including being trained on 40% more tokens, having a much longer context length (4k tokens 🤯), and using grouped-query attention for fast inference of the 70B model🔥!

However, the most exciting part of this release is the fine-tuned models (Llama 2-Chat), which have been optimized for dialogue applications using [Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf). Across a wide range of helpfulness and safety benchmarks, the Llama 2-Chat models perform better than most open models and achieve comparable performance to ChatGPT according to human evaluations.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9f92820d-1a3e-46dd-bba0-94cf27c16249/Untitled.png)

*****************************TODO: add paper link to an image*****************************

If you’ve been waiting for an open alternative to closed-source chatbots, Llama 2-Chat is likely your best choice today!

TODO: add links to models 

| Model | License | Commercial use? | Pretraining length [tokens] | Leaderboard score |
| --- | --- | --- | --- | --- |
| Falcon-7B | Apache 2.0 | ✅ | 1,500B | 47.01 |
| MPT-7B | Apache 2.0 | ✅ | 1,000B | 48.7 |
| Llama-7B | Llama license | ❌ | 1,000B | 49.71 |
| Llama-2-7B | Llama 2 license | ✅ | 2,000B | 54.32 |
| Llama-33B | Llama license | ❌ | 1,500B | * |
| Llama-2-13B | Llama 2 license | ✅ | 2,000B | 58.67 |
| mpt-30B | Apache 2.0 | ✅ | 1,000B | 55.7 |
| Falcon-40B | Apache 2.0 | ✅ | 1,000B | 61.5 |
| Llama-65B | Llama license | ❌ | 1,500B | 62.1 |
| Llama-2-70B | Llama 2 license | ✅ | 2,000B | * |
| Llama-2-70B-chat* | Llama 2 license | ✅ | 2,000B | 66.8 |

*we’re currently running evaluation of the Llama 2 70B (non chatty version). This table will be updated with the results.


## Demo

TODO


## Inference
In this section, we’ll go through different mechanisms to run inference of the Llama2 models.

### Using transformers

With transformers release 4.31, one can already use Llama 2 and leverage all the tools within the HF ecosystem, such as:

- training and inference scripts and examples
- safe file format (`safetensors`)
- integrations with tools such as bitsandbytes (4-bit quantization) and PEFT (parameter efficient fine-tuning)
- utilities and helpers to run generation with the model
- mechanisms to export the models to deploy

In the following code snippet, we show how to run  inference with transformers

```python
TO Update to minimal example + link to colab
from transformers import ConversationalPipeline, Conversation, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, LlamaForCausalLM

model_id = "meta-llama/Llama-2-7b-chat-hf"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
)

# add call
```

```python
from transformers import AutoTokenizer
import transformers
import torch

model = "llamaste/Llama-2-7b-chat-hf"

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

- Add something about ROPE - https://twitter.com/joao_gante/status/1679775399172251648/photo/1

### Using text-generation-inference and Inference Endpoints

**[Text Generation Inference](https://github.com/huggingface/text-generation-inference)** is a production-ready inference container developed by Hugging Face to enable easy deployment of large language models. It has features such as continuous batching, token streaming, tensor parallelism for fast inference on multiple GPUs, and production-ready logging and tracing. 

You can try out Text Generation Inference on your own infrastructure, or you can use Hugging Face's **[Inference Endpoints](https://huggingface.co/inference-endpoints)**. To deploy a Llama 2 model, go to the **[model page](https://huggingface.co/tiiuae/falcon-7b-instruct)** and click on the **[Deploy -> Inference Endpoints](https://ui.endpoints.huggingface.co/new?repository=tiiuae/falcon-7b-instruct)** widget.

- For 7B models, we advise you to select "GPU [medium] - 1x Nvidia A10G".
- For 13B models, we advise you to select "GPU [xlarge] - 1x Nvidia A100".
- For 70B models, we advise you to select "GPU [xxlarge] - 8x Nvidia A100".

*Note: You might need to request a quota upgrade via email to **[api-enterprise@huggingface.co](mailto:api-enterprise@huggingface.co)** to access A100s*

You can learn more on how to [Deploy LLMs with Hugging Face Inference Endpoints in our blog](https://huggingface.co/blog/inference-endpoints-llm). The [blog](https://huggingface.co/blog/inference-endpoints-llm) includes information about supported hyperparameters and how to stream your response using Python and Javascript.

### Using Hugging Face Spaces (stretch)

## Parameter Efficient Fine-tuning with Llama 2

Training LLMs can be technically and computationally challenging. In this section, we look at the tools available in the Hugging Face ecosystem to efficiently train Llama 2 on simple hardware and show how to fine-tune the 7B version of Llama 2 on a single NVIDIA T4 (16GB - Google Colab). You can learn more about it in the [Making LLMs even more accessible blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

We created a [script](https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da) to instruction-tune Llama 2 using QLoRA and the `SFTTrainer` from `trl`. The full script can be found [here](https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da). 

An example command for fine-tuning Llama 2 7B on the `timdettmers/openassistant-guanaco` can be found below. The script can merge the LoRA weights into the model weights and save them as `safetensor` weights by providing the `merge_and_push` argument. This allows us to deploy our fine-tuned model after training using text-generation-inference and inference endpoints.

```python
python finetune_llama_v2.py \
--model_name llamaste/Llama-2-7b-hf \
--dataset_name timdettmers/openassistant-guanaco \
--use_4bit \
--merge_and_push
```

## Additional Resources

## Conclusion