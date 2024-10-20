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

🏎️ <em>TL;DR</em>: Many models such as `gemma-2-9b` lack a smaller version to use for [assisted generation](https://huggingface.co/blog/assisted-generation). In this blog post we present a method developed by Intel Labs and Hugging Face for using a small model **from any model family** to accelerate inference by **1.5x-2.0x**! 🏎️

## Introduction

The unprecedented success of LLMs has redefined the limits of NLP. However, a major challenge in their deployment is optimizing performance to reduce their response time.
[Speculative decoding](https://arxiv.org/abs/2211.17192) is a very popular and practical approach for accelerating LLMs achieving considerable speedups.

## Assisted Generation
 
The core idea behind assisted generation involves using a pair of models, referred to as the _target_ and _assistant_ models. The assistant model is a smaller, more efficient version of the target model, for example you can use `Llama-3.2-1B` as the assistant model for the larger `Llama-3.1-70b` target model.
Speculative decoding is an iterative process. Each cycle, the assistant model generates a sequence of tokens autoregressively, one at a time. The target model then verifies all the assistant tokens in the sequence in a single forward pass. The speedup is achieved by confirming multiple tokens in each forward pass of the target model, rather than producing just one token at a time. For a more detailed explanation, see the original [blog post](https://huggingface.co/blog/assisted-generation). Combined with the recently introduced [Dynamic Speculation](https://huggingface.co/blog/dynamic_speculation_lookahead) strategy, assisted generation accelerates text generation by up 1.5x-3x, depending on the task and the models used.

The remarkable speedups offered by speculative decoding come with a significant drawback: the target and assistant models must share the same tokenizer, meaning they need to be from the same model family. However, many widely-used models lack smaller versions that are both compact and accurate enough to deliver substantial latency reductions. Based on our experience, meaningful speedups are typically seen when the assistant model is at least 50-100 times smaller than the target one. For instance, `CodeLlama-13b` lacks a smaller version, and `gemma-2-9b` only has a `2b` variant which is still not sufficiently small/fast to achieve significant performance improvements.

## Universal Assisted Generation
 
In order to mitigate this pain point, Intel Labs, together with our friends at Hugging Face, has developed Universal Assisted Generation (UAG). UAG enables selecting any pair of target and assistant models regardless of their tokenizer. For example, `gemma-2-9b` can be used as the target model, with the tiny `vicuna-68m` as the assistant.

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

The table below shows the latency improvements observed for target models when paired with assistant models using different tokenizers:

| Target model | Assistant model | Dataset | Task | Speedup |
|----------------------|---------------------|---------------------------|---------------------------|---------------------------|
| `codellama/CodeLlama-13b-Instruct-hf` | `bigcode/tiny_starcoder_py` | `openai/humaneval` | code generation | **1.90x** |
| `microsoft/Phi-3-medium-128k-instruct` | `Qwen/Qwen2-0.5B-Instruct`  | `tau/scrolls`   | long-context summarization | **1.91x** |
| `google/gemma-2-9b` | `double7/vicuna-68m`  | `cnn_dailymail`   | summarization | **1.76x** |

Note that the target models above do not have small variants which are suitable for acceleration using standard assisted generation.

Experimental setup: 1 x A6000 GPU

## Code

Universal Assisted Generation is currently available in the `main` version of 🤗 Transformers. Install using:

```bash
pip install git+https://github.com/huggingface/transformers
```

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

While UAG supports multinomial sampling by passing `do_sample=True`, it does not support speculative sampling. In multinomial sampling, if the target model doesn't sample the same token as the assistant, the token is automatically rejected, which is not the case with speculative sampling. In practice, this means that UAG with `do_sample=True` will have a lower throughput compared to the case where the assistant has the same tokenizer.
In the future, we plan to add support for speculative sampling with UAG. 
In addition, we plan to integrate UAG into 🤗 Transformers pipelines, which will allow a more concise and streamlined usage.



## References
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)
- [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation)
