---
title: "Assisted Generation Support for Intel Gaudi"
thumbnail: /blog/assets/assisted-generation/thumbnail.png
authors:
- user: haimbarad
- user: nraste
- user: joeychou
---

# Assisted Generation Support for Intel Gaudi

## Abstract
As model sizes grow, there is a requirement for efficient inference optimizations in generative AI implementations. This not only reduces latency and infrastructure costs but also minimizes power consumption. We now announce that Intel Gaudi supports Assisted Decoding-Speculative Sampling, which leverages a smaller draft model to generate candidate tokens in parallel and then evaluates them in the target model. We demonstrate that this approach significantly increases throughput without compromising the quality of the generated text. Optimum-Habana, an extension of the HuggingFace library optimized for Intel Gaudi processors, now supports the Assisted Decoding. Our experiments show the effectiveness of this method in improving performance on Intel Gaudi processors. Overall, our work contributes to the field of assisted generation support for efficient text generation in large-scale models.

## Introduction

As model sizes grow, Generative AI implementations require significant inference resources. This not only increases the cost per generation from a prompt, but also increases the power consumption used to serve such requests.

Inference optimizations for text generation are essential for reducing latency, infrastructure costs, and power consumption. This can lead to an improved user experience and increased efficiency in text generation tasks.

Another necessary condition is that the optimizations are compatible with each other. That is, implementing a certain optimization should not preclude or conflict with other optimizations. There are several levels of optimizations that can provide significant speedup without "bumping into each other" in a way that will compromise overall efficiency.

## Speculative Sampling - Assisted Decoding

In order to gain an appreciation of why speculative sampling works, let us take a step back and visit Autoregressive Sampling. We will see that the process is memory bound, allowing us to essentially test K tokens on the target model, in parallel, for the same cost as sampling just one token. So, having a decent acceptance rate means that many of the tokens are generated fast enough to compensate for the extra overhead of generating on a draft model and then checking the batch of K candidate tokens in a target model.

A method of text generation is to generate next tokens based upon a probability conditioned on previous tokens, as given by:

\\(p(\Tilde{x}_{n+1} | x_1, ..., x_n)\\)

This is known as autoregressive sampling [6] and is now a standard method of text-generation in generative models. This could be followed by one of several methods to select the token at $n+1$, for example, argmax or randomly selected from top-p. 

Note that sampling of models is memory intensive. Shazeer [7] shows that the ratio of memory access to arithmetic operations is very memory intensive for transformer-based sequential sampling. Chen et al. [3] attribute the overall sampling time for large transformer-based models to linear layers, attention, and collective operations (all-reduce). We focus on a batch size of one for inference, but we can leverage a batch size of K words (sampled from a smaller draft model) to be evaluated in the target model together, taking about the same time as sampling a single token from the target model. For a reasonable value of K, we can, therefore, use the smaller draft model for much of the text generation, using the target model less often for evaluation (i.e., acceptance or rejection) and single token generation when rejection occurs. We have seen a significant increase in throughput using this method.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/blog/assisted-generation-support-gaudi/SpeculativeSampling.png" alt="Alt text">
  <br>
  <em>Figure 1: Speculative Sampling, K=4</em>
</p>

However, the draft model and target model have different sizes that would be represented in a KV cache, so the challenge is to take advantage of separate optimization strategies simultaneously. For this article, we assume a quantized model and leverage KV caching together with Speculative Sampling.

Note that the authors \cite{chen_accelerating_2023} prove that the target distribution is recovered when performing speculative sampling - this guarantees the same sampling quality as autoregressive sampling on the target itself. Therefore, the situations where not leveraging speculative sampling is not worthwhile have to do with the case where there are not enough savings in the relative size of the draft model or the acceptance rate of the draft model is not high enough to benefit from the smaller size of the draft model.

There is a technique similar to Speculative Sampling, known as Assisted Generation. This was developed independently around the same time [8]. The author integrated this method into HuggingFace Transformers, and the *.generate()* call now has an optional *assistant_model* parameter to enable this method.


## Optimum-Habana

HuggingFace Optimum is an extension of their popular Transformers library, but with a focus on performance and efficiency for specific hardware. There are two packages that are targeted at Intel hardware: optimum-intel and optimum-habana.

Support is now available for Speculative Sampling (Assisted Generation) within optimum-habana for Intel Gaudi processors.

## Usage & Experiments


# Conclusion



# References

[1] T. Schuster, A. Fisch, J. Gupta, M. Dehghani, D. Bahri, V. Q.
Tran, Y. Tay, and D. Metzler, “Confident Adaptive Language
Modeling,” in NeurIPS, July 2022. arXiv: 2207.07061.

[2] N. Belrose, Z. Furman, L. Smith, D. Halawi, I. Ostrovsky,
L. McKinney, S. Biderman, and J. Steinhardt, “Eliciting La-
tent Predictions from Transformers with the Tuned Lens,”
Mar. 2023. arXiv:2303.08112.

[3] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and
J. Jumper, “Accelerating Large Language Model Decoding
with Speculative Sampling,” Feb. 2023. arXiv:2302.01318.

[4] S. Kim, K. Mangalam, S. Moon, J. Canny, J. Malik, M. W.
Mahoney, A. Gholami, and K. Keutzer, “Big Little Trans-
former Decoder,” May 2023. arXiv:2302.07863.

[5] M. Stern, N. Shazeer, and J. Uszkoreit, “Blockwise Paral-
lel Decoding for Deep Autoregressive Models,” Nov. 2018.
arXiv:1811.03115 [cs, stat]. 

[6] D. Lai and B. Lu, “Understanding Autoregressive Model for
Time Series as a Deterministic Dynamic System,” in Predic-
tive Analytics and Futurism, Society of Actuaries, June 2017.

[7] N. Shazeer, “Fast Transformer Decoding: One Write-Head is
All You Need,” Nov. 2019. arXiv:1911.02150 [cs]. 

[8] J. Gante, “Assisted Generation: a new direction toward low-
latency text generation,” May 2023. 
