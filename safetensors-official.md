---
title: "🐶Safetensors audited as really safe and becoming the default"
thumbnail: /blog/assets/104_accelerate-large-models/thumbnail.png
authors:
- user: Narsil
- user: stellaathena
  guest: true
- user: Takyon236
  guest: true
---

<h1>🐶Safetensors audited as really safe and becoming the default</h1>

Hugging Face, in close collaboration with EleutherAI and StabilityAI, has ordered
an external security audit of the `safetensors` library, whose conclusions allow
all 3 organizations to move forward into making the library the default format
for saved models.

[TrailOfBits](https://www.trailofbits.com/) is the security company that performed
the audit, whose conclusions can be found here: [Report](https://huggingface.co/datasets/safetensors/trail_of_bits_audit_repot/resolve/main/SOW-TrailofBits-EleutherAI_HuggingFace-v1.2.pdf)

The following blog post explains the origin of the library, why this audit report is important,
and the next steps

# What is safetensors?

🐶[Safetensors](https://github.com/huggingface/safetensors) is a library
  for saving and loading tensors in the most common frameworks (PyTorch, Tensorflow, Jax, PaddlePaddle, NumPy..)

Let's see an example in PyTorch:
```python
import torch
from safetensors.torch import load_file, save_file

weights = {"embeddings": torch.zeros((10, 100))}
save_file(weights, "model.safetensors")
weights2 = load_file("model.safetensors")
```

And that's pretty much it.
It has many [cool features](https://github.com/huggingface/safetensors#yet-another-format-) compared to other formats, most notably that loading files is _safe_ as we'll see later. 

When you're using `transformers`, if `safetensors` is installed, then those files will already
be used preferentially to prevent issues. Which means doing:

```
pip install safetensors
```

is likely to be the only thing needed to run safetensors files safely.

Going forward and thanks to the validation of the library, `safetensors` will now be installed by
default. The next step is saving models directly in `safetensors` by default.

We also would like to acknowledge the usage of `safetensors` in the wider ML ecosystem

- [CivitAI](https://civitai.com/)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [dfdx](https://github.com/coreylowman/dfdx)
- [Llama-cpp](https://github.com/ggerganov/llama.cpp/blob/e6a46b0ed1884c77267dc70693183e3b7164e0e0/convert.py#L537)


# Why create something new?

The creation of this library came from the issue that PyTorch uses `pickle` under
the hood, which is inherently `unsafe`. [1](https://huggingface.co/docs/hub/security-pickle), [2, video](https://www.youtube.com/watch?v=2ethDz9KnLk), [3](https://github.com/pytorch/pytorch/issues/52596)

`unsafe` here means that someone could write a malicious file in such a way
that a user could download and use a model, and without their knowledge attackers
could gain full control of the computer and steal all their bitcoins 😓.

This is perfectly acknowledged by PyTorch [docs](https://pytorch.org/docs/stable/generated/torch.load.html).

Since the Hugging Face Hub is a platform where anyone can upload and share models, we need to work toward making
sure that users cannot get infected by loading malicious models.

We are also taking steps in making sure the existing PyTorch files are not malicious but the best we can do is flag suspicious-looking files. We cannot be sure they are malicious, nor be sure they are safe.

Of course, there are other file formats out there, and the reason to create something
new is that none seemed to fully comply with the [ideal set of requirements](https://github.com/huggingface/safetensors#yet-another-format-).

In addition to being safe, the format allows `lazy-loading` and generally faster load times (something like 100x faster on CPU).

Lazy loading means the ability to load only part of a tensor efficiently.
This particular feature enables arbitrary sharding with efficient inference libraries such as [text-generation-inference](https://github.com/huggingface/text-generation-inference) to load LLMs such as Llama, Starcoder, etc.. on various hardware
and exploit them with the maximum efficiency.

Another benefit is that because it loads so fast, and is framework agnostic we can use the format
to load models from the same file in PyTorch or TensorFlow.


# The security audit

Since `safetensors` main asset is providing safety guarantees, we wanted to make sure
it delivered. That's why Hugging Face, EleutherAI, and StabilityAI teamed up to get an external
security audit to confirm it.

Important findings:

- No critical security flaw leading to arbitrary code execution was found.
- Some imprecisions in the spec format were detected. (Fixed) 
- Some missing validation allowed [polyglot files](https://en.wikipedia.org/wiki/Polyglot_(computing)). (Fixed)
- Lots of proposed improvements in the test suite. (Done)

In the name of openness and transparency, all companies agreed to make the report
fully public:

[Full report](https://huggingface.co/datasets/safetensors/trail_of_bits_audit_repot/resolve/main/SOW-TrailofBits-EleutherAI_HuggingFace-v1.2.pdf)


One import thing to note is that because the library is written in Rust, a major
flaw was avoided. In the TensorIndexer there was a lack of bound checks. In C or C++ this might
have lead to an Arbitrary Code Execution, but since `safetensors` is written in Rust, it simply results in a panic, so perfectly safe.

Note: This report doesn't mean that there is no flaw because it's impossible to 
prove the absence of flaws, however, it's a major step in giving reassurance that it
is indeed safe to use.

# Going forward

For `transformers` (and HuggingFace at large, it includes all major projects) the master plan is as follows:

- ~Create `safetensors`~. Done
- [x] Verify it works and can deliver on all promises (lazy load for llms, single file for all frameworks, faster loads)
- ~Verify it's safe (today's announcement)~. Done
- ~Make `safetensors` a core dependency (already done, or soon to come).~. Done
- Make `safetensors` the default saving format. Will happen in a few months when we have enough feedback
  to make sure it will cause as little disruption as possible, and enough users already have the library
  to be able to load new models even on relatively old `transformers` versions. **Next step**

As for `safetensors` itself, we're looking into adding more advanced features for LLM training
which have their own set of issues with current formats. This shouldn't impact
using the files.

Also, we're looking to release a `1.0` after a few months after being a `transformers`
core lib. We are considering that it would be enough to be super largely tested
because of the large user base, and it would be much easier to commit.
The format and the lib have had very few modifications since their inception
which is a good sign of stability for us.

We're glad we can make ML one step closer to being safe and efficient for all!
