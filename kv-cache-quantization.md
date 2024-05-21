---
title: "Unlocking Longer Generation with Key-Value Cache Quantization" 
thumbnail: /blog/assets/kv_cache_quantization/thumbnail.png
authors:
- user: RaushanTurganbay
---


# Unlocking Longer Generation with Key-Value Cache Quantization

At Hugging Face, we are excited to share with you a new feature that's going to take your language models to the next level: *kv cache quantization*. 

Have you ever tried generating a lengthy piece of text with your language model, only to hit a wall because of pesky memory limitations? As language models continue to grow in size and capabilities, supporting longer generations can start to really eat up memory. It's a common frustration, especially when you're dealing with limited resources. That's where kv cache quantization swoops in to save the day.

So, what exactly is kv cache quantization? If you're not familiar with the term, don't sweat it! Let's break it down into two pieces: *kv cache* and *quantization*. 

Key-value cache, or kv cache, is needed to optimize the generation in autoregressive models, where the model predicts text token by token. This process can be slow since the model can generate only one token at a time, and each new prediction is dependent on the previous context. That means, to predict token number 1000 in the generation, you need information from the previous 999 tokens, which comes in the form of some matrix multiplications across the representations of those tokens. But to predict token number 1001, you also need the same information from the first 999 tokens, plus additional information from token number 1000. That is where key-value cache is used to optimize the sequential generation process by storing previous calculations to reuse in subsequent tokens, so they don't need to be computed again.

More concretely, key-value cache acts as a memory bank for autoregressive generative models, where the model stores key-value pairs derived from self-attention layers for previously processed tokens. In the transformer architecture, self-attention layers calculate attention scores by multiplying queries with keys, producing weighted sums of value vectors as outputs. By storing this information, the model can avoid redundant computations and instead retrieve keys and values of previous tokens from the cache. For a visual explanation of this concept, take a look at how key-value cache functions in the image below. When calculating the attentions scores for the `K+1`th token we do not need to recompute all of the previous keys and values, but rather take it from cache and concatenate to the current vector. This usually results in faster and more efficient text generation.


<figure class="image text-center m-0">
  <img class="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/kv_cache_quantization/kv-cache-optimization.png" alt="kv cache visual"/>
</figure>


Moving on to the second term, quantization is just a fancy word for reducing the precision of numerical values to save memory. During quantization, each numerical value is rounded or truncated to fit within the reduced precision format, which may result in a loss of information. However, careful selection of quantization parameters and techniques can minimize this loss while still achieving satisfactory performance. There are different quantization methods, so if you're curious to learn more be sure to check out our [previous blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes) for a deeper dive into the world of quantization. 

Even though kv cache speeds up autoregressive generation, it can become a memory bottleneck with long context length or high batch size. Let's estimate how much memory we will need to store kv cache for an input of sequence length 10000 tokens for a 7B Llama-2 model. The memory required to store kv cache of one token is roughly `2 * 2 * num_layers * num_key_value_heads * head_dim`, where the first `2` accounts for keys and values and the second `2` is the number of bytes we need (assuming the model is loaded in `float16`). So if we have a context of length 10000 tokens, we would need

`2 * 2 * 32 * 32 * 128 * 10000 ≈ 5GB`

of memory only to store the previous key-value cache, which is almost one third of the memory required to store model parameters in half-precision.

Therefore, by compressing kv cache into a more compact form we can save up a lot of memory and run longer context generation on consumer GPUs. In our experiments, we were able to significantly reduce the memory footprint without sacrificing too much quality by quantizing the kv cache into lower precision formats. With this new quantization feature, we can now support longer generations without running out of memory, which means you can expand your model's context length without worrying about hitting a memory constraint.


## Implementation Details

Key-value cache quantization in Transformers was largely inspired by the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for kv Cache](https://arxiv.org/abs/2402.02750) paper. The paper introduced a 2bit asymmetrical quantization for large language models without quality degradation. KIVI quantizes the key cache per-channel and the value cache per-token, because they showed that for LLMs keys have higher magnitudes of outliers in some channels while values don't show such a pattern. Therefore, the relative error between quantized and original precision is much smaller when keys are quantized per-channel and the values per-token.

In the method we integrated in Transformers the key and values are both quantized per-token. The main bottleneck when quantizing per-token is the need to quantize and de-quantize keys and values every time a new token is added, that is every generatoin step. That might cause a slow down in generation. To overcome this issue we decided to retain a fixed size residual cache to store keys and values in their original precision. When the residual cache reaches its maximum capacity the stored keys and values are quantized and the cache content is discarded. This small trick also allows to preserve accuracy since some part of the most recent keys and values are always stored in their original precision. The main consideration is the memory-efficiency trade-off when setting the residual cache length. While residual cache stores keys and values in their original precision, that may result in overall memory usage increase. We found that using a residual length of 128 works well as a baseline.

So given a key or value of shape `batch size, num of heads, num of tokens, head dim` we group it to `num of groups, group size` and perform affine quantization as follows:

`X_Q = round(X / S) - Z`

where, 
- X_Q is the quantized tensor
- S is the scale, calculated as `(maxX - minX) / (max_val_for_precision - min_val_for_precision)`
- Z is the zeropoint, calculated as `round(-minX / S)`

Currently, the kv quantization works on [quanto](https://github.com/huggingface/quanto) backend with `int2` and `int4` precisions and [`HQQ`](https://github.com/mobiusml/hqq/tree/master) backend with `int2`, `int4` and `int8` precisions. For more information about `quanto` refer to the previous [blogpost](https://huggingface.co/blog/quanto-introduction). Although we don't currently support more quantization backends, we are open to community contributions that could help integrate them. Specifically, quantization methods that do not need calibration data and can dynamically calculate lower-bit tensors on-the-fly can be easily integrated. Additionally, you can indicate the most common quantization parameters in the config, thus have freedom to tweak quantization process, e.g. decide whether to perform per-channel or per-token quantization depending on your use case.



## Comparing performance of fp16 and quantized cache

We know visuals speak louder than words, so we've prepared some comparison plots to give you a snapshot of how quantization stacks up against FP16 precision. These plots show you at a glance how the model's generation holds up in terms of quality when we tweak the precision settings for kv cache. We calculated the perplexity of [Llama2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model on the [`PG-19`](https://huggingface.co/datasets/emozilla/pg19-test) dataset with the following quantization parameters: `nbits=4, group_size=64, resildual_length=128, per_token=True`


We can see that `int4` cache performs almost the same as the original `fp16` precision for both backends, while the quality degrades when using `int2`. The script to reproduce the results is available [here](https://gist.github.com/zucchini-nlp/a7b19ec32f8c402761d48f3736eac808).

<figure class="image text-center m-0">
  <img class="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/kv_cache_quantization/perplexity.png" alt="Log Perplexity Comparison"/>
</figure>


The same conclusion holds when calculating performance on the [LongBench](https://huggingface.co/datasets/THUDM/LongBench) benchmark comparing it to results from the KIVI paper. `Int4 quanto` precision is comparable and even outperforms slightly the `fp16` in all of the datasets in the table below (higher is better).


| Dataset               | KIVI f16p   | KIVI int2    | Transformers fp16   | Quanto int4| Quanto int2|
|-----------------------|-------------|--------------|---------------------|---------|---------|
| TREC                  | 63.0        | 67.5         | 63.0                | 63.0    | 55.0    |
| SAMSum                | 41.12       | 42.18        | 41.12               | 41.3    | 14.04   |
| TriviaQA              | NA          | NA           | 84.28               | 84.76   | 63.64   |
| HotPotQA              | NA          | NA           | 30.08               | 30.04   | 17.3    |
| Passage_retrieval_en  | NA          | NA           | 8.5                 | 9.5     | 4.82    |


Now, let's talk about the trade-off between memory savings and speed. When we quantize the kv cache in models, we're making them less memory hungry, but sometimes that comes at a tiny cost to generation speed. While quantizing the cache to `int4` can offer roughly an x2.5 memory saving, the generation speed starts to decrease with higher batch sizes. One has to decide whether using quantized kv cache and potentially sacrificing a bit of speed is worth the trade-off for the significant gains in memory efficiency. It's all about finding the approach that best suits your specific use case and priorities. 

Below are the performance metrics for kv cache in original precision and quantized format. Script to obtain the following figures is available [here](https://gist.github.com/zucchini-nlp/56ce57276d7b1ee666e957912d8d36ca).


<figure class="image text-center m-0" style="width: 20%;">
  <img class="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/kv_cache_quantization/gpu_mem_max_new_tokens.png" alt="GPU memory consumption as max new tokens increase"/>
</figure>

<figure class="image text-center m-0" style="width: 20%;">
  <img class="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/kv_cache_quantization/gpu_mem_bs.png" alt="GPU memory consumption as batch size increases"/>
</figure>


<figure class="image text-center m-0">
  <img class="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/kv_cache_quantization/latency.png" alt="Latency as batch size increases"/>
</figure>


Wondering what happens when we throw weight quantization into the mix? Sure, combining these techniques can further slim down your model's memory footprint, but there's a catch – it might slow things down even more. In fact, our experiments show that weight quantization together with kv cache quantization can lead to a threefold decrease in speed. But we're constantly tinkering away to find ways to make this combo work seamlessly. And while we don't currently have optimized kernels in the `quanto` library, we're open to community contributions that could help improve computational efficiency. Our goal is to ensure your model runs smoothly while maintaining high latency and accuracy.

It's also worth noting that initial processing of the input prompt (aka pre-fill stage) still requires computing the entire key-value matrices in one go for the whole input, which may be another memory bottleneck for long contexts. This is the reason why the latency associated with generating the first token tends to be higher compared to subsequent tokens. There are other different strategies to decrease the memory burden of the pre-fill stage by optimizing the attention computation stage, such like [Local Windowed Attention](https://arxiv.org/abs/2004.05150) or [Flash-Attention](https://arxiv.org/abs/2307.08691). If you are out of memory for the pre-fill stage, you can use `FlashAttention` in 🤗 Transformers along with the kv cache quantization to decrease memory usage even more for long input prompts. See [the docs](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#flashattention-2) for more information on that.

If you are interested how many tokens we can fit in the context if we were to push the memory usage to its limits, quantized kv cache can support up to 128k tokens with Flash Attention enabled in an 80GB A100. For the cache in half precision, the maximum capacity is 40k tokens.

## How to use quantized kv cache in 🤗 Transformers?

To use kv cache quantization in 🤗 Tranformers we have to install external dependencies first by running `pip install quanto`. To activate quantization on kv cache, we have to pass in `cache_implementation="quantized"` and indicate quantization parameters in a cache config in dictionary format. And that's all we need to start using kv cache quantization. Additionally, since quanto is device agnostic, you can quantize and run your model regardless if you are on CPU/GPU/MPS (Apple Silicon). 

Here you can find a short [Colab notebook](https://colab.research.google.com/drive/1YKAdOLoBPIore77xR5Xy0XLN8Etcjhui?usp=sharing) with usage examples.

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="cuda:0")
>>> inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"backend": "quanto", "nbits": 4})
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. It's a great way to express myself and rel

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20)
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. I like to listen to it when I'm feeling
```



## Conclusion

There are many more different methods to reduce memory usage by key-value cache, including [MultiQueryAttention](https://arxiv.org/abs/1911.02150), [GroupedQueryAttention](https://arxiv.org/abs/2305.13245) or recent [kv cache retrieval](https://arxiv.org/abs/2403.09054) methods. While some of these methods are bound to the model architecture choices, others can be applied post-training. Quantization is one of such post-training optimization techniques and we can draw the following conclusion from our short blogpost:

1. **Memory vs Speed trade-off**: By quantizing the kv cache into lower precision formats, memory usage is significantly reduced, allowing for longer text generations without encountering memory constraints. But users have to decide on whether giving up a tiny bit of generation speed suits their use-case.

2. **Maintained Accuracy**: Despite the reduction in precision, kv cache quantization in `int4` preserves model accuracy to a satisfactory extent, ensuring that generated text remains contextually relevant and coherent.

3. **Flexibility**: Users have the flexibility to choose between different precision formats based on their specific requirements, allowing for customization to suit varying use cases and priorities.

4. **Potential for Further Optimization**: While kv cache quantization provides significant benefits on its own, it can also be combined with other optimization techniques, such as weight quantization, to further enhance memory efficiency and computational speed.
  

## Acknowledgment

Special thanks to [Younes](https://huggingface.co/ybelkada) and [Marc](https://huggingface.co/marcsun13) for their assistance and advice on quantization techniques. Their expertise greatly contributed to the development of this feature.

Additionally, I would like to thank [Joao](https://huggingface.co/joaogante) for his invaluable support.


## Additional Resources
1. Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Braverman, V., Beidi Chen, & Hu, X. (2023). [KIVI : Plug-and-play 2bit KV Cache Quantization with Streaming Asymmetric Quantization](https://arxiv.org/abs/2402.02750).
2. Blogpost from Databricks on [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) 
3. Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, & Amir Gholami. (2024). [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079).
4. T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, (2022). [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339).
5. A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, (2021). A Survey of Quantization Methods for Efficient Neural Network Inference.

