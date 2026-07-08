---
title: Native-speed vLLM transformers modeling backend
thumbnail: /blog/assets/native-speed-vllm-transformers-backend/thumbnail.png
authors:
- user: hmellor
- user: lysandre
---

# Native-speed vLLM transformers modeling backend

**TL;DR**: The transformers vLLM backend is now as fast (or faster) than custom vLLM implementations for many LLM architectures. Model authors can automatically leverage their transformers implementations to get ultra fast vLLM inference, for free.

```bash
# Upgrade the vllm pip package
uv pip install --upgrade vllm --torch-backend auto
```

The transformers library has become the **reference modeling library** for Machine Learning. It supports 450+ architectures through consistent APIs, and is designed with the main goal that model implementations are _self contained_ and _easy to understand_. Going through transformers code makes it easy for contributors to learn how an architecture works, and then port it to other frameworks such as vLLM, SGLang, MLX, llama.cpp, and many others.

We have fully embraced this role in the ecosystem and are investing a lot of effort to make it easier. A big step in this direction was the integration last year of transformers as a modeling backend in vLLM. This has been allowing model authors to run transformers models (LLMs and VLMs alike) inside vLLM, without having to port anything. Transformers provides the modeling code, and vLLM provides extremely optimized inference techniques such as continuous batching and custom attention kernels.

This integration gets better now 🚀!

## Showcase

We put the transformers modeling backend for vLLM head to head with vLLM's hand written native implementations across three very different Qwen3 models:

* 4B dense model on a single GPU
* 32B dense model on tensor parallelism
* 235B-parameter FP8 Mixture-of-Experts on data + expert parallelism on the same 8×H100 node

| ![Pre and Post PR benchmarks with trasnformers vllm backend](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-backend/pre-post-pr.png) |
| :--: |
| The result: the transformers modeling backend now **meets or beats** native throughput on every one of them. |

Running any Hugging Face model through the transformers modeling backend is a single flag — `--model-impl transformers`. It composes with the usual parallelism options, so nothing about your serving setup changes:

```bash
# Qwen3-4B dense, single GPU
vllm serve Qwen/Qwen3-4B --model-impl transformers

# Qwen3-32B dense, tensor-parallel across 2 GPUs
vllm serve Qwen/Qwen3-32B --model-impl transformers --tensor-parallel-size 2

# Qwen3-235B-A22B-FP8 MoE, data-parallel + expert-parallel across 8 GPUs
vllm serve Qwen/Qwen3-235B-A22B-FP8 --model-impl transformers --data-parallel-size 8 --enable-expert-parallel
# add --max-model-len 8192 if your node is memory constrained
```

### How we measured

Each model is compared under three conditions that are identical in every way except the code path:

1. **native** — `--model-impl vllm`, vLLM's hand-written model (the bar to match)
2. **after** — `--model-impl transformers` _with_ the PR
3. **before** — `--model-impl transformers` _without_ the PR

The full, reproducible runner is available as a gist: [`benchmark.sh`](https://huggingface.co/datasets/ariG23498/useful-scripts/blob/main/transformers-backend-vllm-benchmark.sh)

## So, what's new?

The transformers modeling backend for vLLM used to focus on _attention_ as the bottleneck for inference. By plugging vLLM’s attention implementation at runtime, we could make a transformers model run efficiently inside the vLLM engine. But there are many dimensions to deployments that only a custom port can target to extract maximum inference performance. Parallelization across GPUs, compilation, fused kernels, and many more, all contribute to leveraging your hardware to achieve ultra-fast inference.

| ![New model integration to transformers and vLLM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-backend/previous-pipeline.png) |
| :--: |
| A new model used to be integrated once for transformers, and once for vLLM with custom optimizations |

When model authors wanted the absolute best performance, they were still writing custom vLLM implementations.

| ![New model integrates to transformers, and is immediately available to vLLM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-backend/current-pipeline.png) |
| :--: |
| A new model once integrated to transformers, can now be immediately used in vLLM with native vLLM implementation speed |

The latest iteration of the transformers modeling backend for vLLM dynamically applies inference specific layer fusions at runtime to match the speed of custom code implementations, for compatible architectures.

## How does it work?

The transformers modeling backend for vLLM now uses `torch.fx` to perform static analysis on the model’s graph. This process searches for known patterns that can be optimised. After any patterns have been identified, it uses ast (abstract syntax tree) to manipulate the source code and rewrite some of the operations in place.

**What can we achieve with this?**

* Fused operations that are many-to-one mapped to (ultra) optimized vLLM kernels, such as the ones used for Expert Parallelization (EP) in Mixture-of-Experts (MoE) models.
* Automatic detection of optimal parallel plans for TP (tensor-parallel) and PP (pipeline-parallel), allowing the use of vLLM's `MergedColumnParallelLinear` and `QKVParallelLinear` layers.
* Use of CUDA streams to parallelize compute and data synchronization, especially useful for large scale parallel architectures such as `Deepseek`-style MoEs.
* The manipulated models are still fully (torch) compilable, being passed through `torch.compile` and CUDA Graphs, just the same as a dedicated vLLM model implementation.
* Unlike vLLM model implementations, Transformers model implementations can be used in **training**. So you can use the same model code for training/evals/RL rollouts.

As shown above, this results in native vLLM inference speed for compatible models, without having to write a single line of code to optimize the model for inference.

> [!NOTE]
> We are in the process of writing a detailed blog post to dive deep inside these optimized inference methods and explain in detail how we manipulate the model to adapt to them.

## Resources

* [Transformers model definition](https://huggingface.co/blog/transformers-model-definition#a-model-definition-library)
* [Transformers modeling backend in vLLM](https://vllm.ai/blog/2025-04-11-transformers-backend)
* [Large scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving)
* [Torch FX](https://docs.pytorch.org/docs/2.12/fx.html)
* [Abstract syntax tree](https://docs.python.org/3/library/ast.html)
