---
title: "Faster assisted generation support for Intel Gaudi"
thumbnail: /blog/assets/assisted-generation-support-gaudi/thumbnail.png
authors:
- user: haimbarad
  guest: true
  org: Intel
- user: nraste
  guest: true
  org: Intel
- user: joeychou
  guest: true
  org: Intel
---

# Faster assisted generation support for Intel Gaudi

As model sizes grow, Generative AI implementations require significant inference resources. This not only increases the cost per generation, but also increases the power consumption used to serve such requests.

Inference optimizations for text generation are essential for reducing latency, infrastructure costs, and power consumption. This can lead to an improved user experience and increased efficiency in text generation tasks.

Assisted decoding is a popular method for speeding up text generation. We adapted and optimized it for Intel Gaudi, which delivers similar performance as Nvidia H100 GPUs as shown in [a previous post](https://huggingface.co/blog/bridgetower), while its price is in the same ballpark as Nvidia A100 80GB GPUs. This work is now part of Optimum Habana, which extends various Hugging Face libraries like Transformers and Diffusers so that your AI workflows are fully optimized for Intel Gaudi processors.

## Speculative Sampling - Assisted Decoding

Speculative sampling is a technique used to speed up text generation. It works by generating a draft model that generates K tokens, which are then evaluated in the target model. If the draft model is rejected, the target model is used to generate the next token. This process repeats. By using speculative sampling, we can improve the speed of text generation and achieve similar sampling quality as autoregressive sampling. The technique allows us to specify a draft model when generating text. This method has been shown to provide speedups of about 2x for large transformer-based models. Overall, these techniques can accelerate text generation and improve performance on Intel Gaudi processors.

However, the draft model and target model have different sizes that would be represented in a KV cache, so the challenge is to take advantage of separate optimization strategies simultaneously. For this article, we assume a quantized model and leverage KV caching together with Speculative Sampling. Note that each model has its own KV cache, and the draft model is used to generate K tokens, which are then evaluated in the target model. The target model is used to generate the next token when the draft model is rejected. The draft model is used to generate the next K tokens, and the process repeats.

Note that the authors [2] prove that the target distribution is recovered when performing speculative sampling - this guarantees the same sampling quality as autoregressive sampling on the target itself. Therefore, the situations where not leveraging speculative sampling is not worthwhile have to do with the case where there are not enough savings in the relative size of the draft model or the acceptance rate of the draft model is not high enough to benefit from the smaller size of the draft model.

There is a technique similar to Speculative Sampling, known as Assisted Generation. This was developed independently around the same time [3]. The author integrated this method into Hugging Face Transformers, and the *.generate()* call now has an optional *assistant_model* parameter to enable this method.

## Usage & Experiments

The usage of Assisted Generation is straightforward. As would be expected, the parameter `--assistant_model` is used to specify the draft model. The draft model is used to generate K tokens, which are then evaluated in the target model. The target model is used to generate the next token when the draft model is rejected. The draft model is used to generate the next K tokens, and the process repeats. As the acceptance rate of the draft model is partly dependent on the input text. Typically, we have seen speed ups of about 2x for large transformer-based models.

## Conclusion

Accelerating text generation with Gaudi with assisted generation is now supported and easy to use. The method is compatible with other optimizations (e.g. static shapes, bucketing) and can be used to improve performance on Intel Gaudi processors. The method is based on Speculative Sampling, which has been shown to be effective in improving performance on large transformer-based models.

# References

[1] N. Shazeer, “Fast Transformer Decoding: One Write-Head is All You Need,” Nov. 2019. arXiv:1911.02150. 

[2] C. Chen, S. Borgeaud, G. Irving, J.B. Lespiau, L. Sifre, and J. Jumper, “Accelerating Large Language Model Decoding with Speculative Sampling,” Feb. 2023. arXiv:2302.01318.

[3] J. Gante, “Assisted Generation: a new direction toward low-latency text generation,” May 2023, https://huggingface.co/blog/assisted-generation.
