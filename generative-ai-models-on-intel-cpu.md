---
title: "Smaller is better: Q8-Chat, an efficient generative AI experience on Xeon"
thumbnail: /blog/assets/142_q8chat/pic3.png
authors:
- user: juliensimon
---

# Smaller is better: Q8-Chat, an efficient generative AI experience on Xeon



<!-- {blog_metadata} -->
<!-- {authors} -->

Large language models (LLMs) are taking the machine learning world by storm. Thanks to their [Transformer](https://arxiv.org/abs/1706.03762) architecture, LLMs have an uncanny ability to learn from vast amounts of unstructured data, like text, images, video, or audio. They perform very well on many [task types](https://huggingface.co/tasks), either extractive like text classification or generative like text summarization and text-to-image generation. 

As their name implies, LLMs are *large* models that often exceed the 10-billion parameter mark. Some, like the [BLOOM](https://huggingface.co/bigscience/bloom) model, have more than 100 billion parameters. Accordingly, LLMs require lots of computing power, typically found in high-end GPUs, to predict fast enough for low-latency use cases like search or conversational applications. Unfortunately, for many organizations, the associated costs can be prohibitive and make it difficult to use state-of-the-art LLMs in their applications.

In this post, we will discuss optimization techniques that help reduce LLM size and inference latency, helping them run efficiently on Intel CPUs.  

## A primer on quantization

LLMs usually train with 16-bit floating point parameters (a.k.a FP16/BF16). Thus, storing the value of a single weight or activation value requires 2 bytes of memory. In addition, floating point arithmetic is more complex and slower than integer arithmetic and requires additional computing power. 

Quantization is a model compression technique that aims to solve both problems by reducing the range of unique values that model parameters can take. For instance, you can quantize models to lower precision like 8-bit integers (INT8) to shrink them and replace complex floating-point operations with simpler and faster integer operations.

In a nutshell, quantization rescales model parameters to smaller value ranges. When successful, it shrinks your model by at least 2x, without any impact on model accuracy.

You can apply quantization during training, a.k.a quantization-aware training ([QAT](https://arxiv.org/abs/1910.06188)), which generally yields the best results. If you’d prefer to quantize an existing model, you can apply post-training quantization ([PTQ](https://www.tensorflow.org/lite/performance/post_training_quantization#:~:text=Post%2Dtraining%20quantization%20is%20a,little%20degradation%20in%20model%20accuracy.)), a much faster technique that requires very little computing power.

Different quantization tools are available. For example, PyTorch has built-in support for [quantization](https://pytorch.org/docs/stable/quantization.html). You can also use the Hugging Face [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) library, which includes developer-friendly APIs for QAT and PTQ.

## Quantizing LLMs

Recent studies [[1]](https://arxiv.org/abs/2206.01861)[[2]](https://arxiv.org/abs/2211.10438) show that current quantization techniques don’t work well with LLMs. In particular, LLMs exhibit large-magnitude outliers in specific activation channels across all layers and tokens. Here’s an example with the OPT-13B model. You can see that one of the activation channels has much larger values than all others across all tokens. This phenomenon is visible in all the Transformer layers of the model.

<kbd>
  <img src="assets/142_q8chat/pic1.png">
</kbd>
<br>*Source: SmoothQuant*

The best quantization techniques to date quantize activations token-wise, causing either truncated outliers or underflowing low-magnitude activations. Both solutions hurt model quality significantly. Moreover, quantization-aware training requires additional model training, which is not practical in most cases due to lack of compute resources and data.


SmoothQuant [[3]](https://arxiv.org/abs/2211.10438)[[4]](https://github.com/mit-han-lab/smoothquant) is a new quantization technique that solves this problem. It applies a joint mathematical transformation to weights and activations, which reduces the ratio between outlier and non-outlier values for activations at the cost of increasing the ratio for weights. This transformation makes the layers of the Transformer "quantization-friendly" and enables 8-bit quantization without hurting model quality. As a consequence, SmoothQuant produces smaller, faster models that run well on Intel CPU platforms.

<kbd>
  <img src="assets/142_q8chat/pic2.png">
</kbd>
<br>*Source: SmoothQuant*

Now, let’s see how SmoothQuant works when applied to popular LLMs.

## Quantizing LLMs with SmoothQuant

Our friends at Intel have quantized several LLMs with SmoothQuant-O3: OPT [2.7B](https://huggingface.co/facebook/opt-2.7b) and [6.7B](https://huggingface.co/facebook/opt-6.7b) [[5]](https://arxiv.org/pdf/2205.01068.pdf), LLaMA [7B](https://huggingface.co/decapoda-research/llama-7b-hf) [[6]](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/), Alpaca [7B](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff) [[7]](https://crfm.stanford.edu/2023/03/13/alpaca.html), Vicuna [7B](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1) [[8]](https://vicuna.lmsys.org/), BloomZ [7.1B](https://huggingface.co/bigscience/bloomz-7b1) [[9]](https://huggingface.co/bigscience/bloomz). They also evaluated the accuracy of the quantized models using the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

The table below presents a summary of their findings. The second column shows the ratio of benchmarks that have improved post-quantization. The third column contain the mean average degradation (_* a negative value indicates that the benchmark has improved_). You can find the detailed results at the end of this post.

<kbd>
  <img src="assets/142_q8chat/table0.png">
</kbd>

As you can see, OPT models are great candidates for SmoothQuant quantization. Models are ~2x smaller compared to pretrained 16-bit models. Most of the metrics improve, and those who don’t are only marginally penalized. 

The picture is a little more contrasted for LLaMA 7B and BloomZ 7.1B. Models are compressed by a factor of ~2x, with about half the task seeing metric improvements. Again, the other half is only marginally impacted, with a single task seeing more than 3% relative degradation.

The obvious benefit of working with smaller models is a significant reduction in inference latency. Here’s a [video](https://drive.google.com/file/d/1h8C2I4xn1c0HdrzfMqBaYECJyqo5kcJL/view?usp=sharing) demonstrating real-time text generation with the Vicuna-7b model, on a single socket Intel Sapphire Rapids CPU with 32 cores and a batch size of 1.

In this example, we ask the model: “*What is the role of Hugging Face in democratizing NLP?*”. This sends the following prompt to the model:
"*A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What is the role of Hugging Face in democratizing NLP? ASSISTANT:*"

<kbd>
  [<img src="assets/142_q8chat/pic3.png">](assets/142_q8chat/vicuna-7b-int8-hf-role.mov)
</kbd>

It only takes ~80 milliseconds to generate the first token. The mean generation time for the next tokens is an amazing ~40 milliseconds. This level of performance definitely makes it possible to to run LLMs on CPU platforms, giving customers more IT flexibility and better cost-performance than ever before.

## Chat experience on Xeon

Recently, Clement, the CEO of HuggingFace, said: “*More companies would be better served focusing on smaller, specific models that are cheaper to train and run.*” The emergence of relatively smaller models like Alpaca, BloomZ and Vicuna, open a new opportunity for enterprise to lower the cost of fine-tuning and inference in production. As demonstrated above, high-quality quantization brings high-quality chat experiences to Intel CPU platforms, without the need of running mammoth LLMs and complex AI accelerators. 

Together with Intel, we've hosted on Spaces a new exciting demo called [Q8-Chat](https://huggingface.co/spaces/Intel/Q8-Chat) (pronounced "Cute chat"). Q8-Chat offers you a ChatGPT-like chat experience, while only running on a single socket Intel Sapphire Rapids CPU with 32 cores and a batch size of 1.

<kbd>
  <img src="assets/142_q8chat/pic4.png">
</kbd>

## Next steps

We’re currently working on integrating these amazing new quantization techniques into the Hugging Face [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) library. Once we’re done, you’ll be able to replicate these demos with just a few lines of code.

Stay tuned. The future is 8-bit!

*This post is guaranteed 100% ChatGPT-free.*

## Acknowledgment

This blog was made in conjunction with Ofir Zafrir, Igor Margulis, Guy Boudoukh and Moshe Wasserblat from Intel Labs.
Special thanks to them for their great comments and collaboration.


## Appendix: detailed results

A negative value indicates that the benchmark has improved.

<kbd>
  <img src="assets/142_q8chat/table1.png">
</kbd>

<kbd>
  <img src="assets/142_q8chat/table2.png">
</kbd>

<kbd>
  <img src="assets/142_q8chat/table3.png">
</kbd>

<kbd>
  <img src="assets/142_q8chat/table4.png">
</kbd>