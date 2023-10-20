---
title: "🐶Safetensors audited as really safe and becoming the default"
thumbnail: /blog/assets/142_safetensors_official/thumbnail.png
authors:
- user: Narsil
- user: stellaathena
  guest: true
---

# Audit shows that safetensors is safe and ready to become the default


[Hugging Face](https://huggingface.co/), in close collaboration with [EleutherAI](https://www.eleuther.ai/) and [Stability AI](https://stability.ai/), has ordered
an external security audit of the `safetensors` library, the results of which allow
all three organizations to move toward making the library the default format
for saved models.

The full results of the security audit, performed by [Trail of Bits](https://www.trailofbits.com/), 
can be found here: [Report](https://huggingface.co/datasets/safetensors/trail_of_bits_audit_repot/resolve/main/SOW-TrailofBits-EleutherAI_HuggingFace-v1.2.pdf).

The following blog post explains the origins of the library, why these audit results are important,
and the next steps.

## What is safetensors?

🐶[Safetensors](https://github.com/huggingface/safetensors) is a library
  for saving and loading tensors in the most common frameworks (including PyTorch, TensorFlow, JAX, PaddlePaddle, and NumPy).

For a more concrete explanation, we'll use PyTorch.
```python
import torch
from safetensors.torch import load_file, save_file

weights = {"embeddings": torch.zeros((10, 100))}
save_file(weights, "model.safetensors")
weights2 = load_file("model.safetensors")
```

It also has a number of [cool features](https://github.com/huggingface/safetensors#yet-another-format-) compared to other formats, most notably that loading files is _safe_, as we'll see later. 

When you're using `transformers`, if `safetensors` is installed, then those files will already
be used preferentially in order to prevent issues, which means that

```
pip install safetensors
```

is likely to be the only thing needed to run `safetensors` files safely.

Going forward and thanks to the validation of the library, `safetensors` will now be installed in `transformers` by
default. The next step is saving models in `safetensors` by default.

We are thrilled to see that the `safetensors` library is already seeing use in the ML ecosystem, including:

- [Civitai](https://civitai.com/)
- [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [dfdx](https://github.com/coreylowman/dfdx)
- [LLaMA.cpp](https://github.com/ggerganov/llama.cpp/blob/e6a46b0ed1884c77267dc70693183e3b7164e0e0/convert.py#L537)


## Why create something new?

The creation of this library was driven by the fact that PyTorch uses `pickle` under
the hood, which is inherently unsafe. (Sources: [1](https://huggingface.co/docs/hub/security-pickle), [2, video](https://www.youtube.com/watch?v=2ethDz9KnLk), [3](https://github.com/pytorch/pytorch/issues/52596))

With pickle, it is possible to write a malicious file posing as a model 
that gives full control of a user's computer to an attacker without the user's knowledge,
allowing the attacker to steal all their bitcoins 😓.

While this vulnerability in pickle is widely known in the computer security world (and is acknowledged in the PyTorch [docs](https://pytorch.org/docs/stable/generated/torch.load.html)), it’s not common knowledge in the broader ML community.

Since the Hugging Face Hub is a platform where anyone can upload and share models, it is important to make efforts 
to prevent users from getting infected by malware.

We are also taking steps to make sure the existing PyTorch files are not malicious, but the best we can do is flag suspicious-looking files.

Of course, there are other file formats out there, but
none seemed to meet the full set of [ideal requirements](https://github.com/huggingface/safetensors#yet-another-format-) our team identified.

In addition to being safe, `safetensors` allows lazy loading and generally faster loads (around 100x faster on CPU).

Lazy loading means loading only part of a tensor in an efficient manner.
This particular feature enables arbitrary sharding with efficient inference libraries, such as [text-generation-inference](https://github.com/huggingface/text-generation-inference), to load LLMs (such as LLaMA, StarCoder, etc.) on various types of hardware
with maximum efficiency.

Because it loads so fast and is framework agnostic, we can even use the format
to load models from the same file in PyTorch or TensorFlow.


## The security audit

Since `safetensors` main asset is providing safety guarantees, we wanted to make sure
it actually delivered. That's why Hugging Face, EleutherAI, and Stability AI teamed up to get an external
security audit to confirm it.

Important findings:

- No critical security flaw leading to arbitrary code execution was found.
- Some imprecisions in the spec format were detected and fixed. 
- Some missing validation allowed [polyglot files](https://en.wikipedia.org/wiki/Polyglot_(computing)), which was fixed.
- Lots of improvements to the test suite were proposed and implemented.

In the name of openness and transparency, all companies agreed to make the report
fully public.

[Full report](https://huggingface.co/datasets/safetensors/trail_of_bits_audit_repot/resolve/main/SOW-TrailofBits-EleutherAI_HuggingFace-v1.2.pdf)


One import thing to note is that the library is written in Rust. This adds
an extra layer of [security](https://doc.rust-lang.org/rustc/exploit-mitigations.html)
coming directly from the language itself.

While it is impossible to 
prove the absence of flaws, this is a major step in giving reassurance that `safetensors`
is indeed safe to use.

## Going forward

For Hugging Face, EleutherAI, and Stability AI, the master plan is to shift to using this format by default.

EleutherAI has added support for evaluating models stored as `safetensors` in their LM Evaluation Harness and is working on supporting the format in their GPT-NeoX distributed training library.

Within the `transformers` library we are doing the following:

- Create `safetensors`.
- Verify it works and can deliver on all promises (lazy load for LLMs, single file for all frameworks, faster loads).
- Verify it's safe. (This is today's announcement.)
- Make `safetensors` a core dependency. (This is already done or soon to come.)
- Make `safetensors` the default saving format. This will happen in a few months when we have enough feedback
  to make sure it will cause as little disruption as possible and enough users already have the library
  to be able to load new models even on relatively old `transformers` versions.

As for `safetensors` itself, we're looking into adding more advanced features for LLM training,
which has its own set of issues with current formats.



Finally, we plan to release a `1.0` in the near future, with the large user base of `transformers` providing the final testing step.
The format and the lib have had very few modifications since their inception,
which is a good sign of stability.

We're glad we can bring ML one step closer to being safe and efficient for all!
