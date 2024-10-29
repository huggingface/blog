---
title: "Universal Assisted Generation: Faster Decoding with Any Assistant Model"
thumbnail: /blog/assets/optimum_intel/intel_thumbnail.png
authors:
- user: danielkorat
  guest: true
  org: Intel
- user: orenpereg
  guest: true
  org: Intel
- user: mber
  guest: true
  org: Intel
- user: jmamou
  guest: true
  org: Intel
- user: joaogante
- user: lewtun
- user: Nadav-Timor
  guest: true
  org: weizmannscience
- user: moshew
  guest: true
  org: Intel
---

# Universal Assisted Generation: Faster Decoding with Any Assistant Model

<em>TL;DR</em>: Many LLMs such as `gemma2-9b` and `Mixtral-8x22B-Instruct-v0.1` lack a much smaller version to use for [assisted generation](https://huggingface.co/blog/assisted-generation). In this blog post, we present _Universal Assisted Generation_: a method developed by Intel Labs and Hugging Face which extends assisted generation to work with a small language model **from any model family** 🤯. As a result, it is now possible to accelerate inference from _any_ decoder or [Mixture of Experts](https://huggingface.co/blog/moe) model by **1.5x-2.0x** at almost zero-cost 🔥🔥🔥! 

## Introduction

Nowadays, the strongest open weight LLMs typically have billions to hundreds of billions parameters (hello Llama-3.1-405B 👋), and deploying these beasts in production environments poses a range of engineering challenges. One such challenge is that generating text from these large models is _slow_, which has prompted the community to develop a wide range of techniques to accelerate the decoding process. Assisted generation, also known as [speculative decoding](https://arxiv.org/abs/2211.17192), is a very popular and practical approach for accelerating LLM inference without accuracy loss. In this blog post, we take a look at how assisted generation works and share our   research to extend it towards _any_ of the [140,000 language models](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) on the Hugging Face Hub 🚀! 

## Assisted Generation

The core idea behind assisted generation involves using a pair of models, referred to as the _target_ and _assistant_ models. The assistant model is a smaller, more efficient version of the target model, for example you can use [`Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) as the assistant model for the larger [`Llama-3.1-70b`](https://huggingface.co/meta-llama/Llama-3.1-70b) target model.
Assisted generation is an iterative process. Each cycle, the assistant model generates a sequence of tokens autoregressively, one at a time. The target model then verifies all the assistant tokens in the sequence in a single forward pass. The speedup is achieved by confirming multiple tokens in each forward pass of the target model, rather than producing just one token at a time. For a more detailed explanation, see the original [blog post](https://huggingface.co/blog/assisted-generation). Combined with the recently introduced [Dynamic Speculation](https://huggingface.co/blog/dynamic_speculation_lookahead) strategy, assisted generation accelerates text generation by 1.5x-3x, depending on the task and the models used.

The remarkable speedups offered by assisted generation come with a significant drawback: the target and assistant models must share the same tokenizer, meaning they need to be from the same model family. However, many widely-used models lack smaller versions that are both compact and accurate enough to deliver substantial latency reductions. Based on our experience, meaningful speedups are typically seen when the assistant model is at least 50-100 times smaller than the target one. For instance, [`CodeLlama-13b`](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf) lacks a smaller version, and [`gemma-2-9b`](https://huggingface.co/google/gemma-2-9b) only has a `2b` variant which is still not sufficiently small/fast to achieve significant performance improvements.

## Universal Assisted Generation
 
In order to mitigate this pain point, Intel Labs, together with our friends at Hugging Face, has developed Universal Assisted Generation (UAG). UAG enables selecting any pair of target and assistant models regardless of their tokenizer. For example, `gemma-2-9b` can be used as the target model, with the tiny [`vicuna-68m`](https://huggingface.co/double7/vicuna-68m) as the assistant.

The main idea behind the method we propose is 2-way tokenizer translations. Once the assistant model completes a generation iteration, the assistant tokens are converted to text, which is then tokenized using the target model's tokenizer to generate target tokens. After the verification step, the target tokens are similarly converted back to the assistant tokens format, which are then appended to the assistant model's context before the next iteration begins.

Since the assistant and target tokenizers use different vocabularies it's necessary to handle the discrepancies between them. To accurately re-encode the newly generated assistant tokens, it’s essential to prepend a context window consisting of several previous tokens. This entire sequence is then re-encoded into the target token format and aligned with the most recent target tokens to pinpoint the exact location where the newly generated tokens should be appended. This process is illustrated in the video below.


<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 80%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/universal-assisted-generation/method-animation.mov"
    ></video>
</figure>

While not shown in the video above, token re-encoding from target to assistant follows a similar process. However, mismatched tokens must be discarded from the assistant model's key-value (KV) cache to ensure data integrity.

## Benchmarks

The table below shows the latency improvements observed for target models when paired with assistant models using different tokenizers.

| Target model | Assistant model | Dataset | Task | Speedup |
|----------------------|---------------------|---------------------------|---------------------------|---------------------------|
| `codellama/CodeLlama-13b-Instruct-hf` | `bigcode/tiny_starcoder_py` | [`openai/humaneval`](https://huggingface.co/openai/humaneval) | code generation | **1.90x** |
| [`mistralai/Mixtral-8x22B-Instruct-v0.1`](mistralai/Mixtral-8x22B-Instruct-v0.1) | `double7/vicuna-68m`  | [`cnn_dailymail`](https://huggingface.co/cnn_dailymail)   | summarization | **1.52x** |
| `google/gemma-2-9b` | `double7/vicuna-68m`  | [`cnn_dailymail`](https://huggingface.co/cnn_dailymail)   | summarization | **1.76x** |
| `mistralai/Mixtral-8x22B-Instruct-v0.1` | `Qwen/Qwen2-0.5B-Instruct`  | [`tau/scrolls`](https://huggingface.co/tau/scrolls)   | long-context summarization | **1.78x** |
| `meta-llama/Llama-3.1-70B` | `Qwen/Qwen2-0.5B-Instruct`  | [`tau/scrolls`](https://huggingface.co/tau/scrolls)   | long-context summarization | **1.78x** |
| `microsoft/Phi-3-medium-128k-instruct` | `Qwen/Qwen2-0.5B-Instruct`  | [`tau/scrolls`](https://huggingface.co/tau/scrolls)   | long-context summarization | **1.91x** |

Note that the target models above do not have small variants (under 1 billion parameters) which are suitable for acceleration using standard assisted generation.

Each experiment was conducted on 100 randomly selected examples.
Experiments with `Llama` and `Mixtral` target models use 2 and 4 A100 GPUs, respectively. All other experiments ran with a single A6000 GPU.

## Code

Universal assisted generation has been integrated into release [4.46.0](https://github.com/huggingface/transformers/releases/tag/v4.46.0) of 🤗 Transformers.

To use, pass `tokenizer` and `assistant_tokenizer` to `generate()`:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "Alice and Bob"
>>> checkpoint = "google/gemma-2-9b"
>>> assistant_checkpoint = "double7/vicuna-68m"

>>> assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_checkpoint)
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
>>> outputs = model.generate(**inputs, assistant_model=assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']
```


## Future Directions

While passing `do_sample=True` with standard assisted generation uses the speculative sampling algorithm ([Algorithm 1 from the paper](https://arxiv.org/pdf/2211.17192.pdf)), UAG 
currently supports multinomial sampling only. In multinomial sampling, if the target model doesn't sample the same token as the assistant, the token is automatically rejected, which is not the case with speculative sampling. In practice, this means that UAG with `do_sample=True` will have a lower throughput compared to the case where the assistant has the same tokenizer. In the future, we plan to add support for speculative sampling with UAG.
In addition, we intend to integrate UAG into 🤗 Transformers pipelines, for a more concise and streamlined usage.


## References
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)
- [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation)
