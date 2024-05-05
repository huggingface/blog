---
title: "Unlocking Longer Generation with KV Cache Quantization" 
thumbnail: /blog/assets/kv_cache_quantization/thumbnail.gif
authors:
- user: RaushanTurganbay
- user: your_coauthor
---


# Introduction

I'm excited to share with you a new feature that's going to take your language models to the next level: KV cache quantization. 

Have you ever tried generating a lengthy piece of text with your language model, only to hit a snag because of pesky memory limitations? As language models continue to grow in size and capabilities, supporting longer generations can become a real memory hog. It's a common frustration, especially when you're dealing with limited resources. That's where KV cache quantization swoops in to save the day.

So, what exactly is KV cache quantization? If you're not familiar with the term, don't sweat it! Let's break it down into tow pieces: kv cache and quantization. Quantization is just a fancy word for reducing the precision of numerical values in a model, which helps save memory. During quantization, each numerical value is rounded or truncated to fit within the reduced precision format, which may result in a loss of information. However, careful selection of quantization parameters and techniques can minimize this loss while still achieving satisfactory performance. There are differene quantization methods existing and if you're curious to learn more, be sure to check out our other blog posts for a deeper dive into the world of quantization (LINK TO OLDER POST). 

KV cache, which stands for "key-value cache" acts as a memory bank where the model stores key-value pairs derived from the self-attention layers for previuosly processed tokens. In the transformer architecture, self-attention layers calculate attention scores by multiplying queries with keys, producing weighted sums of value vectors as outputs. By storing this information, the model can avoid redundant computations and instead retrieve keys and values of previuos tokens from the cache. Basically, key value cache enables the model to use information from earlier parts of the input text during generation without performing extra computations. This usually results in faster and more efficient text generation. However kv cache can become a memory bottleneck with long context lengths or high batch sizes.

Let's estimate how much memory we will need to store kv cache for an input of sequence length 4000 tokens for a 7B Llama-2 model. The memory requires to store kv cache of one token is roughly `2 * 2 * num_layers * num_heads * head_dim`, where the first `2` accounts for keys and values and the second `2` is the number of bytes we need (assuming the model is loaded in `float16`). So if we have a context of length 4000 tokens, we would need

`2 * 2 * 32 * 32 * 4096 * 4000 = 60GB`

of memory only to store the previous key-value cache.

Therefore by compressing kv cache into a more compact form we can save up a lot of memory and run long context generation on consumer GPUs. By quantizing the KV cache into lower precision formats we were able to significantly reduce the memory footprint without sacrificing too much accuracy. With this new quantization feature, we can now support longer generations without running out of memory, which means you can expand your model's context length without worrying about hitting a memory constraint.


# Implementation Details

KV cache quantization on `transformers` was largely inspired by the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) paper. The paper intorduces a 2bit assymetrical quantization for large language models without quality degradation. KIVI quantizes the key cache per-channel (grouping on the sequence length dimension) and the value cache per-token, because keys have higher magnitudes of outliers in some channels while value cache does not such a pattern. That is why the relative error between quantized and original precision is much smaller when keys are quantized per-channel and the values per-tensor.

Another key part of the KIVI method is retaining a residual cache to store keys and values in their original precision, and when the residucal cache reaches its maximum capacity the keys and values are quantized and the cache content is discarded. This small trick allows to rpeserve accuracy since some part of the most recent keys and values are always stored in their original precision. Also, it allows to run seamlessly the per-channel key quantization which requires the number of tokens to be divisible by the group size. The main consideration is the memory-efficiency trade-off when setting the residual cache length. While residual cache stores keys and values in their original precision, that may result in overall memory usage increase. So, the general idea of KIVI is to maintain two separate caches for quantized and residual parts of key and value matrices.

The implementation of KV cache in `transformers` is partially based on KIVI, specifically we adopted the same strategy of retaining `k` tokens in the residual cache for quality maintenance. In contrast to the KIVI, `transformers` quantizes both keys and value per-channel by grouping on the last dimension. We found that this type of per-channel quantization does not hurt accuracy of the generations. So given a key or value of shape `batch size, num of heads, num of tokens, head dim` we group it to `num of groups, group size` and perform affine quantization as follows:

`X_Q = (X / S) - Z`

where, 
- X_Q is the quantized tensor
- S is the scale calculated as `(maxX - minX) / (max_val_for_precision - min_val_for_precision)`
- Z is zeropoin calculated as `round(-minX / S)`


To integrate kv quantization seamlessly in `transformers`, we rely on `quanto` library. Quanto is a toolkit for easy implementation and combination of different quantization tecniques. Although quanto support `torch.compile()`, the current version of quantized kv cache is not `compile` compatible. Currently available quantization precisions are int2 and int4.


# Comparing performance of fp16 and quantized cache
- Comparison plots of quantization to FP16 precision
- Trade-offs between quality and speed (can be more efficient if we use optimized kernels for computations)
- Can be combined with weight quantization, but the speed will be 3 times slower

# How to use

To use KV cache quantization in `tranformers` we have to install external dependencies first by running `pip install git+https://github.com/huggingface/quanto`. To activate quantization on kv cache, we have to pass in `cache_implementation="quantized"` and indicate quantization parameters in a cache config in dictionary format. And that's all we need to start using kv cache quantization. Additionally, since quanto is device agnostic, you can quantize and run your model regardless if you are on CPU/GPU/ MPS (Apple Silicon). 


```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"nbits": 4})
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. It's a great way to express myself and rel

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20)
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. I like to listen to it when I'm feeling
```


# Conclusion
- Summary of the benefits of KV cache quantization


# Acknowledgment

