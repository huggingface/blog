---
title: "Faster assisted generation support for Intel Gaudi"
thumbnail: /blog/assets/assisted-generation/thumbnail.png
authors:
- user: haimbarad
- guest: true
- user: nraste
- guest: true
- user: joeychou
- guest: true
---

# Faster assisted generation support for Intel Gaudi

## Introduction

As model sizes grow, Generative AI implementations require significant inference resources. This not only increases the cost per generation from a prompt, but also increases the power consumption used to serve such requests.

Inference optimizations for text generation are essential for reducing latency, infrastructure costs, and power consumption. This can lead to an improved user experience and increased efficiency in text generation tasks.

Assisted decoding is a popular method for speeding up text generation. We adapted and optimized it for Intel Gaudi, which deliver similar performance as Nvidia H100 GPUs as shown in [a previous post](https://huggingface.co/blog/bridgetower) while its price is in the same ballpark as Nvidia A100 80GB GPUs. This work is now part of Optimum Habana, which extends various Hugging Face libraries like Transformers and Diffusers so that your AI workflows are fully optimized for Intel Gaudi processors.

## Speculative Sampling - Assisted Decoding

In order to gain an appreciation of why speculative sampling works, let us take a step back and visit Autoregressive Sampling. We will see that the process is memory bound, allowing us to essentially test K tokens on the target model, in parallel, for the same cost as sampling just one token. So, having a decent acceptance rate means that many of the tokens are generated fast enough to compensate for the extra overhead of generating on a draft model and then checking the batch of K candidate tokens in a target model.

A method of text generation is to generate next tokens based upon a probability conditioned on previous tokens, as given by:

$p(\tilde{x}_{n+1} | x_1, ..., x_n)$ 

where $x_1, ..., x_n$ are the tokens generated so far. This is known as autoregressive sampling and is now a standard method of text-generation in generative models. This could be followed by one of several methods to select the token at $n+1$, for example, argmax or randomly selected from top-$p$. 

Note that sampling of models is memory intensive. Shazeer [1] shows that the ratio of memory access to arithmetic operations is very memory intensive for transformer-based sequential sampling. Chen et al. [2] attribute the overall sampling time for large transformer-based models to linear layers, attention, and collective operations (all-reduce). We focus on a batch size of one for inference, but we can leverage a batch size of K words (sampled from a smaller draft model) to be evaluated in the target model together, taking about the same time as sampling a single token from the target model. For a reasonable value of K, we can, therefore, use the smaller draft model for much of the text generation, using the target model less often for evaluation (i.e., acceptance or rejection) and single token generation when rejection occurs. We have seen a significant increase in throughput using this method.

However, the draft model and target model have different sizes that would be represented in a KV cache, so the challenge is to take advantage of separate optimization strategies simultaneously. For this article, we assume a quantized model and leverage KV caching together with Speculative Sampling. Note that each model has its own KV cache, and the draft model is used to generate K tokens, which are then evaluated in the target model. The target model is used to generate the next token when the draft model is rejected. The draft model is used to generate the next K tokens, and the process repeats.

Note that the authors [2] prove that the target distribution is recovered when performing speculative sampling - this guarantees the same sampling quality as autoregressive sampling on the target itself. Therefore, the situations where not leveraging speculative sampling is not worthwhile have to do with the case where there are not enough savings in the relative size of the draft model or the acceptance rate of the draft model is not high enough to benefit from the smaller size of the draft model.

There is a technique similar to Speculative Sampling, known as Assisted Generation. This was developed independently around the same time [3]. The author integrated this method into Hugging Face Transformers, and the *.generate()* call now has an optional *assistant_model* parameter to enable this method.

## Usage & Experiments

The usage of Assisted Generation is straightforward. As would be expected, the parameter `--assistant_model` is used to specify the draft model. The draft model is used to generate K tokens, which are then evaluated in the target model. The target model is used to generate the next token when the draft model is rejected. The draft model is used to generate the next K tokens, and the process repeats.

# Conclusion

Accelerating text generation with Gaudi with assisted generation is now supported and easy to use. The method is compatible with other optimizations (e.g. static shapes, bucketing) and can be used to improve performance on Intel Gaudi processors. The method is based on Speculative Sampling, which has been shown to be effective in improving performance on large transformer-based models.

# References

[1] N. Shazeer, “Fast Transformer Decoding: One Write-Head is All You Need,” Nov. 2019. arXiv:1911.02150. 

[2] C. Chen, S. Borgeaud, G. Irving, J.B. Lespiau, L. Sifre, and J. Jumper, “Accelerating Large Language Model Decoding with Speculative Sampling,” Feb. 2023. arXiv:2302.01318.

[3] J. Gante, “Assisted Generation: a new direction toward low-latency text generation,” May 2023, https://huggingface.co/blog/assisted-generation.
