---
title: "Enabling assisted generation with any assistant model"
thumbnail: /blog/assets/optimum_intel/intel_thumbnail.png
authors:
- user: user1
  guest: true
  org: Intel
- user: user2
  guest: true
  org: Intel
---

# Enabling assisted generation with any assistant model

The unprecedented success of LLMs has redefined the limits of NLP. However, a major challenge in their deployment is optimizing performance to reduce their response time.
Speculative decoding is a very popular and practical approach for accelerating LLMs achieving considerable speedups.
 
The core concept of this method involves using a pair of models, referred to as the target and draft models. The draft model is a smaller, more efficient version of the target model, for example using Llama3.1-8b as the draft model for the larger Llama3.1-70b target model.
Speculative decoding is an iterative process, during each cycle, the draft model generates a sequence of tokens autoregressively, one at a time. The target model then verifies these draft tokens in a single forward pass. The speedup is achieved by generating multiple tokens in each forward pass of the target model, rather than producing just one token at a time.

The remarkable speedups offered by speculative decoding come with a significant drawback: the target and draft models must share the same tokenizer, meaning they need to be from the same model family. However, many widely-used models lack smaller versions that are both compact and accurate enough to deliver substantial latency reductions. Based on our experience, meaningful speedups are typically seen when the size ratio between the target and draft models is at least 50-100. For instance, LLaMA 3.1-8B lacks a smaller version, and Gemma 2-9B only has a 2B variant which is still not sufficiently small to achieve significant performance improvements.
 
In order to mitigate this pain point Intel labs together with our friends in Hugging face developed "AG_anyPair". "AG_anyPair", which is integrated as part of Hugging face Transformers 4.46.0, enables to select any pair of target and draft models regardless of their tokenizer. For example, gemma-2-9b can be used as target model together with vicuna-68m as draft model. The main idea behind this method is 2-way tokenizer translations. Once the draft model completes a generation iteration, the draft tokens are converted to text, which is then tokenized using the target model's tokenizer to generate target tokens. After the verification step, the target tokens are similarly converted back to draft tokens, which are then appended to the draft model's context before the next drafting iteration begins.

The table below shows the latency improvements observed for target models when paired with draft models using different tokenizers:

| Target model | Assistant model | Dataset | Task | Speedup |
|----------------------|---------------------|---------------------------|---------------------------|---------------------------|
| `codellama/CodeLlama-13b-Instruct-hf` | `bigcode/tiny_starcoder_py` | `openai/humaneval` | code generation | **2.01x** |
| `microsoft/Phi-3-medium-128k-instruct` | `Qwen/Qwen2-0.5B-Instruct`  | `tau/scrolls`   | long-context summarization | **1.65x** |
| `google/gemma-2-9b` | `double7/vicuna-68m`  | `cnn_dailymail`   | summarization | **1.72x** |

We note that 

## SUBSECTION EXAMPLE

```python
print("This is a python code block example")
```

## EXAMPLE RESULTS SECTION
<p align="center">
    <img src="assets/178_setfit_optimum_intel/latency.png" width=500>
</p>
<p align="center">
    <em>IMAGE CAPTION HERE</em>
</p>



## Summary

INSERT SUMMARY HERE

## References
- [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation)
