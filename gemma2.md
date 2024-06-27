---
title: "Welcome Gemma 2 - Google’s new open LLM" 
thumbnail: /blog/assets/gemma2/thumbnail.jpg
authors:
- user: philschmid
- user: osanseviero
- user: pcuenq
- user: lewtun
- user: tomaarsen
- user: reach-vb
---

# Welcome Gemma 2 - Google’s new open LLM

Google released Gemma 2, the latest addition to its family of state-of-the-art open LLMs, and we are excited to collaborate with Google to ensure the best integration in the Hugging Face ecosystem. You can find the 4 open-weight models (2 base models & 2 fine-tuned ones) on the Hub. Among the features and integrations being released, we have:

- [Models on the Hub](https://huggingface.co/collections/google/g-667d6600fd5220e7b967f315)
- Hugging Face [Transformers integration](https://github.com/huggingface/transformers/releases/tag/v4.42.0)
- Integration with Google Cloud
- Integration with Inference Endpoints

## Table of contents

- [What is Gemma 2?](#what-is-gemma-2)
- [Technical advances in Gemma 2](#technical-advances-in-gemma-2)
  - [Sliding window attention](#sliding-window-attention)
  - [Soft-capping and attention implementations](#soft-capping-and-attention-implementations)
  - [Knowledge Distillation](#knowledge-distillation)
  - [Model Merging](#model-merging)
- [Gemma 2 evaluation](#gemma-2-evaluation)
  - [Technical Report results](#technical-report-results)
  - [Open LLM Leaderboard results](#open-llm-leaderboard-results)
- [How to prompt Gemma 2](#how-to-prompt-gemma-2)
- [Demo](#demo)
- [Using Hugging Face Transformers](#using-hugging-facetransformers)
- [Integration with Google Cloud](#integration-with-google-cloud)
- [Integration with Inference Endpoints](#integration-with-inference-endpoints)
- [Fine-tuning with 🤗 TRL](#fine-tuning-with-trl)
- [Additional Resources](#additional-resources)
- [Acknowledgments](#acknowledgments)

## What is Gemma 2?

Gemma 2 is Google's latest iteration of open LLMs. It comes in two sizes, 9 billion and 27 billion parameters with base (pre-trained) and instruction-tuned versions. Gemma is based on Google Deepmind Gemini and has a context length of 8K tokens:

- [gemma-2-9b](https://huggingface.co/google/gemma-2-9b): Base 9B model.
- [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it): Instruction fine-tuned version of the base 9B model.
- [gemma-2-27b](https://huggingface.co/google/gemma-2-27b): Base 27B model.
- [gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it): Instruction fine-tuned version of the base 27B model.

The Gemma 2 models were trained on ~2x more data than their first iteration, totaling 13 trillion tokens for the 27B version and 8 trillion tokens for the 9B version of web data (primarily English), code, and math. We don’t know the exact details of the training mix, and we can only guess that bigger and more careful data curation was a big factor in the improved performance.

Gemma 2 comes with the [same license](https://ai.google.dev/gemma/terms) as the first iteration, which is a permissive license that allows redistribution, fine-tuning, commercial use, and derivative works. 

## Technical advances in Gemma 2

Gemma 2 has many similarities with the first iteration. It has a context length of 8192 tokens and uses Rotary Position Embedding (RoPE). There are four main advances in Gemma 2 compared to the original Gemma: 

- [Sliding window attention](#sliding-window-attention): Interleave sliding window and full-quadratic attention for quality generation.
- [Logit soft-capping](#soft-capping-and-attention-implementations): Prevents logits from growing excessively by scaling them to a fixed range, improving training.
- [Knowledge Distillation](#knowledge-distillation): Leverage a larger teacher model to train a smaller model (for the 9B model).
- [Model Merging](#model-merging): Combines two or more LLMs into a single new model

Gemma 2 was trained on [Google Cloud TPU (27B on v5p](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer?hl=en), [9B on TPU v4)](https://cloud.google.com/tpu/docs/v4) using [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) and [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/). Gemma 2 Instruct has been optimized for dialogue applications and trained on a mix of synthetic and human-generated prompt-response pairs using Supervised Fine-Tuning (SFT), Distillation from a larger model, Reinforcement Learning from Human Feedback (RLHF) using a reward model oriented more towards conversational capabilities and model merging models using WARP to improve overall performance.

Similar to the pre-training mix, no details about the fine-tuning datasets or the hyperparameters associated with SFT and [RLHF](https://huggingface.co/blog/rlhf) have been shared.

### Sliding window attention

[Sliding window attention](https://huggingface.co/papers/2004.05150) is a method to reduce the memory and time requirements of the attention computations in transformer models and has been used in models such as [Mistral](https://huggingface.co/papers/2310.06825). The novelty of Gemma 2 is that a sliding window is applied to every other layer (local - 4096 tokens), while the layers in between still use full quadratic global attention (8192 tokens). We suppose this is a way to increase quality in long context situations (half of the layers still attend to all tokens) while partially benefiting from the advantages of sliding attention. 

### Soft-capping and attention implementations

Soft capping is a technique that prevents logits from growing excessively large without truncating them. It works by dividing the logits by a maximum value threshold (soft_cap), then passing them through a `tanh` layer (ensuring they are in the `(-1, 1)` range), and finally multiplying by the threshold again. This guarantees that the final values will be in the `(-soft_cap, +soft_cap)` interval without losing much information but stabilizing the training.

Putting it all together, the logits are calculated by: `logits ← soft_cap ∗ tanh(logits/soft_cap)`

Gemma 2 employs soft capping for the final and each attention layer. The attention logits are capped at 50.0, and the final logits at 30.0.

At the time of release soft-capping is incompatible with Flash Attention/ SDPA. To ensure maximum efficiency we remove it during inference. Gemma 2 team observed minor difference in inference performance.

*Note: For stable fine-tuning runs, you still need to enable soft-capping and hence, we recommend fine-tuning with `eager` attention instead of SDPA.*

### Knowledge Distillation

Knowledge distillation is a popular technique for training a smaller *student* model to mimic the behavior of a larger but better-performing *teacher.* This works by augmenting the next-token prediction task of LLMs with a distribution of token probabilities from the teacher (e.g., GPT-4, Claude, or Gemini), which provides a richer signal for the student to learn from. 

According to the Gemma 2 tech report, knowledge distillation was used to pre-train the 9B model, while the 27B model was pre-trained from scratch.

For post-training, the Gemma 2 team generated a diverse set of completions from a teacher (unspecified in the report, but presumably Gemini Ultra), and then trained the student models on this synthetic data with SFT. This is the basis of many open models, such as [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) and [OpenHermes](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B), which are trained entirely on synthetic data from larger LLMs.

Although effective, this method has drawbacks since the model capacity mismatch between the student and teacher can lead to a *train-inference mismatch*, where the text generated by the student during inference is out-of-distribution compared to that seen during training.

To handle this issue, the Gemma 2 team used [“on-policy distillation”](https://arxiv.org/pdf/2306.13649), where the student generates completions from the SFT prompts. These completions are then used to compute the KL divergence between the teacher’s and student’s logits. By minimizing the KL divergence throughout training, the student learns to model the behavior of the teacher accurately while also minimizing the train-inference mismatch.

This approach is quite interesting, as we’ve seen in the community that on-policy methods like online DPO produce stronger models, and one advantage of on-policy distillation is that you only need the logits from the teacher, so you don’t need to rely on reward models or LLM-as-a-judge to improve the model. It will be exciting to see if this method becomes more popular among fine-tuners in the coming months!

### Model Merging

[Model merging](https://huggingface.co/blog/mlabonne/merge-models) is a technique that combines two or more LLMs into a single new model. It's relatively new and experimental and can be used without accelerators. [Mergekit](https://github.com/arcee-ai/mergekit) is a popular open-source toolkit for merging LLMs. It implements linear, SLERP, TIES, DARE, and other merging techniques.

According to the Technical Report, Gemma 2 used [Warp](https://arxiv.org/abs/2406.16768), a new merging technique that merges models in three distinct stages: 

1. Exponential Moving Average (EMA): This is applied during the reinforcement learning (RL) fine-tuning process.
2. Spherical Linear intERPolation (SLERP): This is applied after the RL fine-tuning of multiple policies.
3. Linear Interpolation Towards Initialization (LITI): This stage is applied after the SLERP stage.

## Gemma 2 evaluation

How good are the Gemma models? Below are performance comparisons to other open models based on the Technical Report and the new version of the [open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

### Technical Report results

This Technical Report of Gemma 2 compares the performance of different open LLMs on the previous Open LLM Leaderboard benchmarks.

|            | Llama 3 (70B) | Qwen 1.5 (32B) | Gemma 2 (27B) |
| ---------- | ------------- | -------------- | ------------- |
| MMLU       | **79.2**      | 74.3           | 75.2          |
| GSM8K      | **76.9**      | 61.1           | 75.1          |
| ARC-c      | 68.8          | 63.6           | **71.4**      |
| HellaSwag  | **88.0**      | 85.0           | 86.4          |
| Winogrande | **85.3**      | 81.5           | 83.7          |

The Report also compares the performance of Small Language Models. 

| Benchmark  | Mistral (7B) | Llama 3 (8B) | Gemma (8B) | Gemma 2 (9B) |
| ---------- | ------------ | ------------ | ---------- | ------------ |
| MMLU       | 62.5         | 66.6         | 64.4       | **71.3**     |
| GSM8K      | 34.5         | 45.7         | 50.9       | **62.3**     |
| ARC-C      | 60.5         | 59.2         | 61.1       | **68.4**     |
| HellaSwag  | **83.0**     | 82.0         | 82.3       | 81.9         |
| Winogrande | 78.5         | 78.5         | 79.0       | **80.6**     |

### Open LLM Leaderboard results

*Note: We are currently evaluating Google Gemma 2 individually on the new Open LLM Leaderboard benchmark and will update this section later today.* 

## How to prompt Gemma 2

The base models have no prompt format. Like other base models, they can be used to continue an input sequence with a plausible continuation or for zero-shot/few-shot inference. The Instruct versions have a very simple conversation structure:

```bash
<start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
LaMDA<end_of_turn>
<start_of_turn>model
LaMDA who?<end_of_turn><eos>
```

This format has to be exactly reproduced for effective use. We’ll later show how easy it is to reproduce the instruct prompt with the chat template available in `transformers`. 

## Demo

You can chat with the Gemma 27B Instruct model on Hugging Chat! Check out the link here: https://huggingface.co/chat/models/google/gemma-2-27b-it.

## Using Hugging Face Transformers

With Transformers [release 4.42](https://github.com/huggingface/transformers/releases/tag/v4.42.0), you can use Gemma and leverage all the tools within the Hugging Face ecosystem. To use Gemma models with transformers, make sure to use the latest `transformers` release:

```jsx
pip install "transformers==4.42.0" --upgrade
```

The following snippet shows how to use `gemma-2-9b-it` with transformers. It requires about 18 GB of RAM, which fits many consumer GPUs. The same snippet works for `gemma-2-27b-it`, which, at 56GB of RAM, makes it a very interesting model for production use cases. Memory consumption can be further reduced by loading in 8-bit or 4-bit mode.

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
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
```

> `Ahoy, matey! I be a humble ship o' words, sailin' the digital seas. They call me Gemma, a creation o' the fine folks at Google DeepMind. I be trained on a treasure trove o' texts, learnin' to speak and write like a true scallywag.

Ask me yer questions, and I'll do me best to answer 'em, aye!  🦜📚`
> 

*Note: that we used `bfloat16` because that’s the reference precision for the instruction-tuned model. Running in `float16` may be faster on your hardware, and results should be similar.*

You can also automatically quantize the model, loading it in 8-bit or even 4-bit mode. 4-bit loading of the large 27B version takes about 18 GB of memory to run, making it compatible with a lot of consumer cards and GPUs in Google Colab. This is how you’d load the generation pipeline in 4-bit:

```jsx
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True}
    },
)
```

For more details on using the models with transformers, please check [the model cards](https://huggingface.co/gg-hf/gemma-2-9b).

## Integration with Google Cloud

*Note: We are currently working on adding new containers to GKE and Vertex AI to run Google Gemma 2 efficiently. We will update this section as soon as the containers are available.* 

## Integration with Inference Endpoints

You can deploy Gemma 2 on Hugging Face's [Inference Endpoints](https://ui.endpoints.huggingface.co/philschmid/new?repository=google%2Fgemma-2-27b-it&accelerator=gpu&instance_id=aws-us-east-1-nvidia-a100-x1&task=text-generation&no_suggested_compute=true&tgi=true) using Text Generation Inference as the backend. [Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a production-ready inference container developed by Hugging Face to enable easy deployment of large language models. It has features such as continuous batching, token streaming, tensor parallelism for fast inference on multiple GPUs, and production-ready logging and tracing.

To deploy a Gemma 2 model, go to the [model page](https://huggingface.co/google/gemma-2-27b-it) and click on the [Deploy -> Inference Endpoints](https://ui.endpoints.huggingface.co/new?repository=google/gemma-2-27b-it) widget. Inference Endpoints supports OpenAI compatible [Messages API](https://huggingface.co/blog/tgi-messages-api) that allows you to switch from another closed model to an open one by simply changing the URL.

```bash
curl https://api.endpoints.huggingface.cloud/v2/endpoint/philschmid \
-X POST \
-d '{"compute":{"accelerator":"gpu","instanceSize":"x1","instanceType":"nvidia-a100","scaling":{"maxReplica":1,"minReplica":1}},"model":{"framework":"pytorch","image":{"huggingface":{"env":{}}},"repository":"google/gemma-2-27b-it","task":"text-generation"},"name":"test","provider":{"region":"us-east-1","vendor":"aws"},"type":"protected"}' \
-H "Content-Type: application/json" \
-H "Authorization: Bearer XXXXX"
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


## Additional Resources

- [Models on the Hub](https://huggingface.co/collections/google/g-667d6600fd5220e7b967f315)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Chat demo on Hugging Chat](https://huggingface.co/chat/models/google/gemma-2-27b-it)
- Google Blog
- Google Notebook
- Vertex AI model garden link

## Acknowledgments

Releasing such models with support and evaluations in the ecosystem would not be possible without the contributions of many community members, including [Clémentine](https://huggingface.co/clefourrier) and [Nathan](https://huggingface.co/SaylorTwift) for LLM evaluations; [Nicolas](https://huggingface.co/Narsil) for Text Generation Inference Support; [Arthur](https://huggingface.co/ArthurZ), [Sanchit](https://huggingface.co/sanchit-gandhi), [Joao](https://huggingface.co/joaogante), and [Lysandre for](https://huggingface.co/lysandre) integrating Gemma 2 into transformers; [Nathan](https://huggingface.co/nsarrazin) and [Victor](https://huggingface.co/victor) for making Gemma 2 available in Hugging Chat. 

And Thank you to the Google Team for releasing Gemma 2 and making it available to the open-source AI community!
