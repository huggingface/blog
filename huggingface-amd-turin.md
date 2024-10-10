---
title: "Introducing the AMD 5th Gen EPYC™ CPU"
thumbnail: /blog/assets/optimum_amd/amd_hf_logo_fixed.png
authors:
- user: mohitsha
- user: mfuntowicz
---

# Introducing the AMD 5th Gen EPYC™ CPU

AMD has just unveiled its 5th generation of server-grade EPYC CPU based on Zen5 architecture - also known as `Turin`. It provides a significant boost in performance, especially with a higher number of core count reaching up to `192` and `384` threads.

From Large Language Models (LLMs) to RAG scenarios, Hugging Face users can leverage this new generation of servers to enhance their performance capabilities: 
1. Reduce the target latency of their deployments.
2. Increase the maximum throughput.
3. Lower the operational costs.

During the last few weeks, we have been working with AMD to validate that the Hugging Face ecosystem is fully supported on this new CPU generation and delivers the expected performance across different tasks.

Also, we have been cooking some exciting new ways to leverage `torch.compile` for AMD CPU through the use of `AMD ZenDNN PyTorch plugin (zentorch)` to speed up even more the kind of workloads we will be discussing after.

While we were able to get early access to this work to test Hugging Face models and libraries and share with you performance, we expect AMD to make it soon available to the community - stay tuned!


## AMD Turin vs AMD Genoa Performance - A 2X speedup

In this section, we present the results from our benchmarking of the two AMD EPYC CPUs: Turin (128 cores) and Genoa (96 cores). For these benchmarks, we utilized the **ZenDNN** plug-in for PyTorch (zentorch), which provides inference optimizations tailored for deep learning workloads on AMD EPYC CPUs. This plug-in integrates seamlessly with the torch.compile graph compilation flow, enabling multiple passes of graph-level optimizations on the torch.fx graph to achieve further performance acceleration.

To ensure optimal performance, we used the `bfloat16` data type and employed `ZenDNN 5.0`. We configured multi-instance setups that enable the parallel execution of multiple [Meta LLaMA 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model instances spawning across all the cores. Each model instance is allocated 32 physical cores per socket, allowing us to leverage the full processing power of the servers for efficient data handling and computational speed.

We ran the benchmarks  using two different batch sizes—16 and 32—across five distinct use cases: 
- Summarization (1024 input tokens / 128 output tokens)
- Chatbot (128 input tokens / 128 output tokens)
- Translation (1024 input tokens / 1024 output tokens)
- Essay Writing (128 input tokens / 1024 output tokens)
- Live Captioning (16 input tokens / 16 output tokens). 

These configurations not only facilitates a comprehensive analysis of how each server performs under varying workloads but also simulates real-world applications of LLMs. Specifically, we plot the decode throughput (excluding the first token) for each use case, to illustrate performance differences.

### Results for Llama 3.1 8B Instruct

![Turin vs Genoa](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-amd-turin/zentorch_bs_16_32_turin_vs_genoa.png)

_Throughput results for Meta Llama 3.1 8B, comparing AMD Turin against AMD Genoa. AMD Turin consistently outperforms the AMD Genoa CPUs, achieving approximately 2X higher throughput in most configurations._

## Conclusion

As demonstrated, the AMD EPYC Turin CPU offers a significant boost in performance for AI use cases compared to its predecessor, the AMD Genoa. To enhance reproducibility and streamline the benchmarking process, we utilized [optimum-benchmark](https://github.com/huggingface/optimum-benchmark), which provides a unified framework for efficient benchmarking across various setups. This enabled us to effectively benchmark using the `zentorch` backend for `torch.compile`.

Furthermore, we have developed an optimized `Dockerfile` that will be released soon, along with the benchmarking code. This will facilitate easy deployment and reproduction of our results, ensuring that others can effectively leverage our findings.

You can find more information at [AMD Zen Deep Neural Network (ZenDNN)](https://www.amd.com/en/developer/zendnn.html)

## Useful Resources

- ZenTF: ​ https://github.com/amd/ZenDNN-tensorflow-plugin​
- ZenTorch: ​https://github.com/amd/ZenDNN-pytorch-plugin ​
- ZenDNN ONNXRuntime:  https://github.com/amd/ZenDNN-onnxruntime
