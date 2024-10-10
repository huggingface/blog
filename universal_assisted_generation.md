---
title: "Universal Assisted Generation: Enabling assisted generation with any assistant model"
thumbnail: /blog/assets/optimum_intel/intel_thumbnail.png
authors:
- user: user1
  guest: true
  org: Intel
- user: user2
  guest: true
  org: Intel
---

# Universal Assisted Generation: Enabling assisted generation with any assistant model

The unprecedented success of LLMs has redefined the limits of NLP. However, a major challenge in their deployment is optimizing performance to reduce their response time.
Speculative decoding is a very popular and practical approach for accelerating LLMs achieving considerable speedups.
 
The core concept of this method involves using a pair of models, referred to as the target and assistant models. The assistant model is a smaller, more efficient version of the target model, for example using `Llama3.1-8b` as the assistant model for the larger `Llama3.1-70b` target model.
Speculative decoding is an iterative process, during each cycle, the assistant model generates a sequence of tokens autoregressively, one at a time. The target model then verifies these assistant tokens in a single forward pass. The speedup is achieved by generating multiple tokens in each forward pass of the target model, rather than producing just one token at a time.

The remarkable speedups offered by speculative decoding come with a significant drawback: the target and assistant models must share the same tokenizer, meaning they need to be from the same model family. However, many widely-used models lack smaller versions that are both compact and accurate enough to deliver substantial latency reductions. Based on our experience, meaningful speedups are typically seen when the size ratio between the target and assistant models is at least 50-100. For instance, LLaMA 3.1-8B lacks a smaller version, and Gemma 2-9B only has a 2B variant which is still not sufficiently small to achieve significant performance improvements.
 
In order to mitigate this pain point Intel labs together with our friends in Hugging face developed Universal Assisted Generation (UAG). UAG enables selecting any pair of target and assistant models regardless of their tokenizer. For example, `gemma-2-9b` can be used as target model together with `vicuna-68m` as assistant model. The main idea behind this method is 2-way tokenizer translations. Once the assistant model completes a generation iteration, the assistant tokens are converted to text, which is then tokenized using the target model's tokenizer to generate target tokens. After the verification step, the target tokens are similarly converted back to assistant tokens, which are then appended to the assistant model's context before the next assistanting iteration begins.

# Benchmarks

The table below shows the latency improvements observed for target models when paired with assistant models using different tokenizers:

| Target model | Assistant model | Dataset | Task | Speedup |
|----------------------|---------------------|---------------------------|---------------------------|---------------------------|
| `codellama/CodeLlama-13b-Instruct-hf` | `bigcode/tiny_starcoder_py` | `openai/humaneval` | code generation | **1.90x** |
| `microsoft/Phi-3-medium-128k-instruct` | `Qwen/Qwen2-0.5B-Instruct`  | `tau/scrolls`   | long-context summarization | **1.91x** |
| `google/gemma-2-9b` | `double7/vicuna-68m`  | `cnn_dailymail`   | summarization | **1.76x** |

Experimental setup: 1 x A6000 GPU

## Code
Universal Assisted Generation currently in the `main` version of 🤗 Transformers. Install using:

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


## Next Steps / Summary

Text goes here

## References
- [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation)
