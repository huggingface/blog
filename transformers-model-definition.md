---
title: "The Transformers Library: standardizing model definitions" 
thumbnail: /blog/assets/transformers-model-definition/transformers-thumbnail.png
authors:
- user: lysandre
- user: ArthurZ
- user: pcuenq
- user: julien-c
---

# The Transformers Library: standardizing model definitions

TLDR: Going forward, we're aiming for Transformers to be the pivot across frameworks: if a model architecture is
supported by transformers, you can expect it to be supported in the rest of the ecosystem.

---

Transformers was created in 2019, shortly following the release of the BERT Transformer model. Since then, we've
continuously aimed to add state-of-the-art architectures, initially focused on NLP, then growing to Audio and
computer vision. Today, transformers is the default library for LLMs and VLMs in the Python ecosystem.

Transformers now supports 300+ model architectures, with an average of ~3 new architectures added every week. 
We have aimed for these architectures to be released in a timely manner; having day-0 support for the most
sought-after architectures (Llamas, Qwens, GLMs, etc.).

## A model-definition library

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers-model-definition/transformers-thumbnail.png" alt="Transformers standardizing model definitions">

Over time, Transformers has become a central component in the ML ecosystem, becoming one of the most complete
toolkits in terms of model diversity; it's integrated in all popular training frameworks such as Axolotl,
Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, TRL, Nanotron, etc.

Recently, we've been working hand in hand with the most popular inference engines (vLLM, SGLang, TGI, ...) for them
to use `transformers` as a backend. The value added is significant: as soon as a model is added to `transformers`,
it becomes available in these inference engines, _while taking advantage of the strengths each engine provides_: inference optimizations, specialized kernels, dynamic batching, etc.

As an example, here is how you would work with the `transformers` backend in vLLM: 

```python
from vllm import LLM

llm = LLM(model="new-transformers-model", model_impl="transformers")
```

That's all it takes for a new model to enjoy super-fast and production-grade serving with vLLM!

Read more about it in the [vLLM documentation](https://blog.vllm.ai/2025/04/11/transformers-backend.html).

---

We've also been working very closely with [llama.cpp](https://github.com/ggml-org/llama.cpp) and 
[MLX](https://github.com/ml-explore/mlx) so that the implementations between `transformers`
and these modeling libraries have great interoperability. For example, thanks to a significant community effort,
it's now very easy to [load GGUF files in `transformers`](https://huggingface.co/docs/transformers/en/gguf) for 
further fine-tuning. Conversely, transformers models can be easily 
[converted to GGUF files](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) for use with 
llama.cpp.

The same is true for MLX, where the transformers' safetensors files are directly compatible with MLX's models.

We are super proud that the `transformers` format is being adopted by the community, bringing a lot of interoperability 
we all benefit from. Train a model with Unsloth, deploy it with SGLang, and export it to llama.cpp to run locally! We 
aim to keep supporting the community going forward.

## Striving for even simpler model contributions

To make it easier for the community to use transformers as a reference for model definitions, we strive to
significantly reduce the barrier to model contributions. We have been doing this effort for a few years, but we'll 
accelerate significantly over the next few weeks:
- The modeling code of each model will be further simplified; with clear, concise APIs for the most important
  components (KV cache, different Attention functions, kernel optimization)
- We'll deprecate redundant components in favor of having a simple, single way to use our APIs: encouraging 
  efficient tokenization by deprecating slow tokenizers, and similarly using the fast vectorized vision processors.
- We'll continue to reinforce the work around _modular_ model definitions, with the goal for new models to require absolute
  minimal code changes. 6000 line contributions, 20 files changes for new models are a thing of the past.

## How does this affect you?

### What this means for you, as a model user

As a model user, in the future you should see even more interoperability in the tools that you use.

This does not mean that we intend to lock you in using `transformers` in your experiments; rather, it means that
thanks to this modeling standardization, you can expect the tools that you use for training, for inference, and for
production, to efficiently work together.

### What this means for you, as a model creator

As a model creator, this means that a single contribution will get your model available in all downstream libraries that
have integrated that modeling implementation. We have seen this many times over the years: releasing a model
is stressful and integrating in all important libraries is often a significant time-sink.

By standardizing the model implementation in a community-driven manner, we hope to lower the barrier of contributions
to the field across libraries.

---

We firmly believe this renewed direction will help standardize an ecosystem which is often at risk of fragmentation.
We'd love to hear your feedback on the direction the team has decided to take; and of changes we could do to get
there. Please come and see us over at the 
[transformers-community support tab](https://huggingface.co/spaces/transformers-community/support) on the Hub!
