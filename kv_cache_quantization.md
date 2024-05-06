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

So, what exactly is KV cache quantization? If you're not familiar with the term, don't sweat it! Let's break it down into two pieces: kv cache and quantization. Quantization is just a fancy term for reducing the precision of numerical values in a model, which helps save memory. During quantization, each numerical value is rounded or truncated to fit within the reduced precision format, which may result in a loss of information. However, careful selection of quantization parameters and techniques can minimize this loss while still achieving satisfactory performance. There are different quantization methods existing and if you're curious to learn more, be sure to check out our [other blog posts](https://huggingface.co/blog/4bit-transformers-bitsandbytes) for a deeper dive into the world of quantization. 

KV cache, which stands for "key-value cache" acts as a memory bank where the model stores key-value pairs derived from self-attention layers for previuosly processed tokens. In the transformer architecture, self-attention layers calculate attention scores by multiplying queries with keys, producing weighted sums of value vectors as outputs. By storing this information, the model can avoid redundant computations and instead retrieve keys and values of previuos tokens from the cache. Basically, key value cache enables the model to use information from earlier parts of the input text during generation without performing extra computations. This usually results in faster and more efficient text generation. However kv cache can become a memory bottleneck with long context length or high batch size.

Let's estimate how much memory we will need to store kv cache for an input of sequence length 4000 tokens for a 7B Llama-2 model. The memory required to store kv cache of one token is roughly `2 * 2 * num_layers * num_heads * head_dim`, where the first `2` accounts for keys and values and the second `2` is the number of bytes we need (assuming the model is loaded in `float16`). So if we have a context of length 4000 tokens, we would need

`2 * 2 * 32 * 32 * 4096 * 4000 = 60GB`

of memory only to store the previous key-value cache.

Therefore by compressing kv cache into a more compact form we can save up a lot of memory and run long context generation on consumer GPUs. By quantizing the KV cache into lower precision formats we were able to significantly reduce the memory footprint without sacrificing too much accuracy. With this new quantization feature, we can now support longer generations without running out of memory, which means you can expand your model's context length without worrying about hitting a memory constraint.


# Implementation Details

KV cache quantization in `transformers` was largely inspired by the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) paper. The paper intorduced a 2bit assymetrical quantization for large language models without quality degradation. KIVI quantizes the key cache per-channel (grouping on the sequence length dimension) and the value cache per-token, because keys have higher magnitudes of outliers in some channels while value cache does not such a pattern. That is why the relative error between quantized and original precision is much smaller when keys are quantized per-channel and the values per-tensor.

Another key part of the KIVI method is retaining a residual cache to store keys and values in their original precision, and when the residucal cache reaches its maximum capacity the keys and values are quantized and the cache content is discarded. This small trick allows to rpeserve accuracy since some part of the most recent keys and values are always stored in their original precision. Also, it allows to run seamlessly the per-channel key quantization which requires the number of tokens to be divisible by the group size. The main consideration is the memory-efficiency trade-off when setting the residual cache length. While residual cache stores keys and values in their original precision, that may result in overall memory usage increase. So, the general idea of KIVI is to maintain two separate caches for quantized and residual parts of key and value matrices.

The implementation of KV cache in `transformers` is partially based on KIVI, specifically we adopted the same strategy of retaining `k` tokens in the residual cache for quality maintenance. In contrast to the KIVI, `transformers` quantizes both keys and value per-channel by grouping on the last dimension. We found that this type of per-channel quantization does not hurt accuracy of the generations. So given a key or value of shape `batch size, num of heads, num of tokens, head dim` we group it to `num of groups, group size` and perform affine quantization as follows:

`X_Q = (X / S) - Z`

where, 
- X_Q is the quantized tensor
- S is the scale calculated as `(maxX - minX) / (max_val_for_precision - min_val_for_precision)`
- Z is zeropoin calculated as `round(-minX / S)`


To integrate kv quantization seamlessly in `transformers`, we rely on [quanto](https://github.com/huggingface/quanto) library. Quanto is a toolkit for easy implementation and combination of different quantization tecniques. Although quanto support `torch.compile()`, the current version of quantized kv cache is not `compile` compatible. Currently available quantization precisions are int2 and int4.


# Comparing performance of fp16 and quantized cache

We know visuals speak louder than words, so we've prepared up some comparison plots to give you a snapshot of how quantization stacks up against FP16 precision. These plots will show you at a glance how the models' generation hold up in terms of quality when we tweak the precision settings for kv cache. We calculated the perplexity of [Llama2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model on the [`PG-19`](https://huggingface.co/datasets/emozilla/pg19-test) dataset. We can see that `int4` cache performs almost same as the original `fp16` precision, while the quality degrades when using `int2`.  

<figure class="image text-center m-0">
  <img class="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/kv_cache_quantization/perplexity.png" alt="Log Perplexity Comparison"/>
</figure>


Same holds true when calculating performance on the [LongBench](https://huggingface.co/datasets/THUDM/LongBench) dataset comparing it to results from the KIVI paper. Int4 precision is comparable and even outperforms slighly the `float16` in all the datasets. 

| Dataset               | KIVI f16p   | KIVI int2    | Our fp16            | Our int4| Our int2|
|-----------------------|-------------|--------------|---------------------|---------|---------|
| TREC                  | 63.0        | 67.5         | 63.0                | 63.0    | 55.0    |
| SAMSum                | 41.12       | 42.18        | 41.12               | 41.3    | 14.04   |
| TriviaQA              | NA          | NA           | 84.28               | 84.76   | 63.64   |
| HotPotQA              | NA          | NA           | 30.08               | 30.04   | 17.3    |
| Passage_retrieval_en  | NA          | NA           | 8.5                 | 9.5     | 4.82    |


Now, let's talk about the trade-off between memory savings and speed. When we quantize the kv cache in models, we're making them less memory hungry, but sometimes that comes at a tiny cost to generation speed. While quantiziing the cache to `int4` can offer an x2 memory saving, the generation speed starts to decrease with higher batch sizes. One has to decide whether using quantized KV cache and potentially sacrificing a bit of speed is worth the trade-off for the significant gains in memory efficiency. It's all about finding the approach that best suits your specific use case and priorities. 


<div style="display: flex;">
  <figure class="image text-center m-0">
    <img class="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gpu_mem_max_new_tokens.png" alt="GPU memory consumption as max new tokens increase"/>
  </figure>
  <figure class="image text-center m-0">
    <img class="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/kv_cache_quantization/gpu_mem_bs.png" alt="GPU memory consumption as batch size increases"/>
  </figure>
</div>


<figure class="image text-center m-0">
  <img class="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/kv_cache_quantization/latency.png" alt="Latency as batch size increases"/>
</figure>


Wondering what happens when we throw weight quantization into the mix? Sure, combining these techniques can further slim down your model's memory footprint, but there's a catch – it might slow things down even more. In fact, our experiments show that weigt quantization together with kv cache quantization can lead to a threefold decrease in speed. But we're constantly tinkering away in the lab to find ways to make this combo work seamlessly. And while we don't currently have optimized kernels in the quanto library, we're open to community contributions that could help improve computational efficiency. Our goal is to ensure your model runs smoothly while maintaining high latency and accuracy.



# How to use quantized kv cache in 🤗 Transformers?

To use KV cache quantization in 🤗 Tranformers we have to install external dependencies first by running `pip install git+https://github.com/huggingface/quanto`. To activate quantization on kv cache, we have to pass in `cache_implementation="quantized"` and indicate quantization parameters in a cache config in dictionary format. And that's all we need to start using kv cache quantization. Additionally, since quanto is device agnostic, you can quantize and run your model regardless if you are on CPU/GPU/ MPS (Apple Silicon). 


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

We can draw the following conclusion from our short blogpost:

1. **Memory vs Speed trade-off**: By quantizing the KV cache into lower precision formats, memory usage is significantly reduced, allowing for longer text generations without encountering memory constraints. But users have to decide on whether giving up a tiny bit of generation speed suits their use-case.

2. **Maintained Accuracy**: Despite the reduction in precision, KV cache quantization in `int4` preserves model accuracy to a satisfactory extent, ensuring that generated text remains contextually relevant and coherent.

3. **Flexibility**: Users have the flexibility to choose between different precision formats based on their specific requirements, allowing for customization to suit varying use cases and priorities.

4. **Potential for Further Optimization**: While KV cache quantization provides significant benefits on its own, it can also be combined with other optimization techniques, such as weight quantization, to further enhance memory efficiency and computational speed.


# Acknowledgment

