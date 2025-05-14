---
title: "The Transformers Library: standardizing model definitions" 
thumbnail: /blog/assets/transformers-model-definition/transformers-thumbnail.png
authors:
- user: lysandre
- user: ArthurZ
---

# The Transformers Library: standardizing model definitions

TLDR: Going forward, we're aiming for Transformers to be the pivot across frameworks: if a model architecture is
supported by transformers, you can expect it to be supported in the rest of the ecosystem.

---

Transformers was created in 2019, shortly following the release of the BERT Transformer model. Since then, we've
continuously aimed to add state-of-the-art architectures, initially focused on NLP, then growing to Audio and
computer vision; and transformers is now the default for LLMs and VLMs in the Python ecosystem.

The library now boasts 350+ model architectures, with an average of ~3 new architectures every week. 
We have aimed for these architectures to be released in a timely manner; having day-0 support for the most
sought-after architectures (Llamas, Qwens, GLMs, etc.).

## A model-definition library

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers-model-definition/transformers-thumbnail.png" alt="Transformers standardizing model definitions">

Over time, Transformers has become a central component in the ML ecosystem, becoming one of the most complete
toolkits in terms of model diversity; it's integrated in all popular training frameworks such as Axolotl,
Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, TRL, Nanotron, etc.

Recently, we've been working hand in hand with the most popular inference engines (vLLM, SGLang, TGI, ...) for them
to use `transformers` as a backend. The value added is significant: as soon as a model is added to `transformers`,
it becomes available in these inference engines.

As an example, here is how you would work with the `transformers` backend in vllm: 

```python
from vllm import LLM

llm = LLM(model="new-transformers-model", model_impl="transformers")
```

Read more about it in the [vllm documentation](https://blog.vllm.ai/2025/04/11/transformers-backend.html).

---

Finally, we've been working very closely with llama.cpp and MLX so that the implementations between `transformers`
and these modeling libraries have great interoperability; it is now very simple to load GGUF files in `transformers`
to train these models further, and transformers models can be easily converted to GGUF files for usage in llama.cpp.

The same is true for MLX, where the transformers' safetensors files are directly compatible with MLX's models.

We aim for `transformers` to continue being the backend of such tools; thanks to this, all of them are interoperable
with each other: train a model with Unsloth, deploy it with SGLang, and export it to llama.cpp to run it locally.

## Striving for even simpler model contributions

In order to simplify this end goal of having transformers be the backbone of the modeling definitions, we strive to
significantly reduce the barrier to model contributions. This is in line of the efforts we have done over the past
few years, but we'll double down on these over the next few weeks:
- The modeling code of each model will be further simplified; with clear, concise APIs for the most important
  components (KV cache, different Attention functions, kernel optimization)
- We'll deprecate redundant component in favour of having a simple, single way to use our APIs: encouraging 
  efficient tokenization by deprecating slow tokenizers, and the same to be done with vision processors and their
  fast counterparts.
- We'll continue to reinforce the work around modular and aim for model contributions to require absolute
  minimal code changes. 6000 line contributions, 20 files changes for new models are a thing of the past.

## How does this affect you?

### What this means for you, as a model user

As a model user, you should see greatly improved interoperability in the tools that you use.

This does not mean that we intend to lock you in using `transformers` in your experiments; rather, it means that
thanks to this modeling standardization, you can expect the tools that you use for training, for inference, and for
production, to efficiently work together.

### What this means for you, as a model creator

As a model creator, this means that a single contribution will get you available in all downstream libraries that
have integrated that modeling implementation. We have seen this many times over the years: releasing a model
is stressful and time-bound

---

We firmly believe this renewed direction will help standardize an ecosystem which is often at risk of fragmentation.
We'd love to hear your feedback on the direction the team has decided to take; and of changes we could do to get
there.
