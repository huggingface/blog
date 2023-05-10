---
title: "🐶Safetensors is really safe and becoming the default"
thumbnail: /blog/assets/104_accelerate-large-models/thumbnail.png
authors:
- user: Narsil
- user: stellaathena
- user: Takyon236
---

<h1>🐶Safetensors safe and becoming the default</h1>

HuggingFace, in close collaboration with EleutherAI and StabilityAI, has ordered
an external security audit of the `safetensors` library, which conclusions allow
all 3 organizations to move forward into making the library the default format
for saved models.

[TrailOfBits](https://www.trailofbits.com/) is the security company that provided
the audit, which conclusions can be found here: [Report](/blog/assets/safetensors-official/full-report.pdf)

The following blogpost explains the origin of the library, why this audit report is important,
and the next steps

# What is safetensors ?

🐶[Safetensors](https://github.com/huggingface/safetensors) is a library
for saving and loading tensors in the most common frameworks (PyTorch, Tensorflow, jax, paddlepaddle, numpy..)

Let's see an example in PyTorch:
```python
import torch
from safetensors.torch import load_file, save_file

weights = {"embeddings": torch.zeros((10, 100))}
save_file(weights, "model.safetensors")
weights2 = load_file("model.safetensors")
```

And that's pretty much it.
It has a number of [cool features](https://github.com/huggingface/safetensors#yet-another-format-) compared to other formats. 

When you're using `transformers`, if `safetensors` is installed, then those files will already
be used preferentially in order to prevent issues. Which means doing:

```
pip install safetensors
```

is likely to be the only thing needed to run safetensors files safely.

Going forward and thanks to the validation of the library, `safetensors` will now be installed by
default. The next step is saving models directly in `safetensors` by default.

We also would like to acknowledge the usage of `safetensors` outside our companies ecosystem

- [CiviAI](https://civitai.com/)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [dfdx](https://github.com/coreylowman/dfdx)
- [Llama-cpp](https://github.com/ggerganov/llama.cpp/blob/e6a46b0ed1884c77267dc70693183e3b7164e0e0/convert.py#L537)


# Why creating something new ?

The creation of this library came from the issue that PyTorch uses `pickle` under
the hood, which is inherently `unsafe`. [1](https://huggingface.co/docs/hub/security-pickle), [2, video](https://www.youtube.com/watch?v=2ethDz9KnLk), [3](https://github.com/pytorch/pytorch/issues/52596)

`unsafe` here means that someone could write a malicious file in such a way
that a user could download and use a model, and without his knowledge attackers
had won full control of his computer and stolen all his bitcoins.

This is perfectly acknowledged by Pytorch [doc](https://pytorch.org/docs/stable/generated/torch.load.html).

Since HuggingFace is now a hub where anyone can upload and share models, we need to make
sure that users cannot get infected by loading malicious models.

Of course there are other file formats out there and the reason to create something
new is that none seemed to fully comply with the [ideal set of requirements](https://github.com/huggingface/safetensors#yet-another-format-).

In addition to being safe, the formats allow `lazy-loading` and generally faster loads (something like 100x faster on cpu).

Lazy loading means the ability to load only part of a tensor in an efficient manner.
This particular feature enables arbitrary sharding with efficient inference libraries such as [text-generation-inference](https://github.com/huggingface/text-generation-inference) to load LLMs such as Llama, Starcoder etc.. on various hardware
and exploit it to the maximum efficiency.

Another benefit, is that because it loads so fast, and is framework agnostic we can use the format
to load models from the same file in PyTorch or Tensorflow.


# The security audit

Since `safetensors` main asset is providing safety guarantees, we wanted to make sure
it actually delivered. That's why HuggingFace, EleutherAI and StabilityAI teamed up to get a external
security audit to confirm it.

Important findings:

- No critical security flaw leading to arbitry code execution was found.
- Some imprecisions in the spec format were detected. (Fixed) 
- Some missing validation allowed polyglot files. (Fixed)
- Lots of proposed improvements in the test suite. (Done)

In the name of openess and transparency, all companies agreed to make the report
fully public:

[Full report](/blog/assets/safetensors-official/full-report.pdf)


One major kudos, is that because the library is written in Rust, a potential
flaw in the TensorIndexer which might have actually been a potential Arbitrary
Code Execution kind of flaws, is was just is a unwarranted panic, so perfectly safe.

Note: This report doesn't mean that there is no flaw, because it's impossible to 
prove the absence of flaws, however it's a major step in giving reassurance that it
is indeed safe to use.

# Going forward

For `transformers` (and HuggingFace at large, it includes all major projects) the master plan is as follows:

- ~Create `safetensors`~. Done
- ~Verify it works and can deliver on all promises (lazy load for llms, single file for all frameworks, faster loads)~. Done
- ~Verify it's safe (today's announcement)~. Done
- ~Make `safetensors` a core dependency (already done, or soon to come).~. Done
- Make `safetensors` the default saving format. Will happen in a few months when we have enough feedback
  to make sure it will cause as little disruption as possible, and enough users already have the library
  to be able to load new models even on relatively old `transformers` version. **Next step**

As for `safetensors` itself, we're looking into adding more advanced features for LLM training
which have their own set of issues with current formats. This shouldn't impact
using the files.

Also we're looking to releasing a `1.0` after a few month after being a `transformers`
core lib. We are considering that it would be enough to be super largely tested
because of the large user base, and it would be much easier to commit.
Also the format and the lib have had very little modifications since it's inception
which is a good sign of stability to us.

We're glad we can make ML one step closer to being safe and efficient for all !
