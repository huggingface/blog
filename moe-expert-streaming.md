---
title: "180B Parameters, 32GB of RAM: Pushing MoE Inference to the Limit with Transformers" 
thumbnail: /blog/assets/moe-expert-streaming/thumbnail.png
authors:
- user: tarekziade
---

# 180B Parameters, 32GB of RAM: Pushing MoE Inference to the Limit with Transformers

*The unlock: streaming MoE experts directly from SSD, reaching within about 6% of a specialized inference engine on short decode.*

I've been trying to rely less on online models and run bigger models locally,
but my work laptop has only 32 GB of memory. At home I
have a PC with two RTX 4090s (let's call it **Orthrus** - the two-headed hound
of Greek mythology), and for work I have a MacBook M5.

Orthrus runs on Windows 11, because my kids play games all the time. I had a
dual boot with Ubuntu, but their games ended up taking most of the two
terabytes of disk.

I am also a huge fan of RMD (rooms moving development): one hour in the kitchen
sipping my coffee, two hours at my office desk, then the sofa. A fixed desktop
tower is not for me.

My first experiment was to run [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) with [`llama.app`](https://llama.app/) on Orthrus and use
it from my laptop through [`pi`](https://pi.dev/). That model does the coding tasks, an online
model like Opus does the orchestration, and the second RTX runs a FIM model for
auto-completion in Neovim.

When I talked about this on social networks, I got very strong feedback: use
[Strata](https://github.com/niko1221/strata) with [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## What is Strata?

Strata is an inference engine for MoE models like Qwen. Only the hot experts
stay in VRAM, the rest goes to RAM, or even to disk. Experts are 90%+ of a MoE
model's weights, so a 180B model like this one can run on a GPU with much less
memory. Like my RTX 4090!

It works on a laptop too, and I stumbled on
[this blog post](https://modelfit.io/blog/strata-qwen38-flash-next-mac/) doing
exactly that: Qwen3.8-Flash-Next on a 32 GB Mac M4, faster than [llama.cpp](https://github.com/ggml-org/llama.cpp), at
13.5 tok/s.

That's too slow for how I use coding agents, but it's an interesting itch to
scratch. A whole range of MoE models don't fit in a MacBook's unified memory and
could run at a decent speed.

## How do other engines offload?

Before committing to Strata, I looked at what llama.cpp, [vLLM](https://github.com/vllm-project/vllm) and [Transformers](https://github.com/huggingface/transformers)
do when a MoE model doesn't fit. Three questions matter: where do the experts
live, where does the expert matmul run, and what moves on each token?

**llama.cpp** maps the GGUF file with `mmap`, and lets you keep the experts on
the CPU with `--cpu-moe`, `--n-cpu-moe N`, or a regex with `-ot "exps=CPU"`
(`--fit on` picks the placement for you). Those experts are also *computed* on
the CPU, so only the activations cross PCIe, never the weights. When the experts
fit in RAM, that's a great deal. When they don't, you rely on the page cache,
with no budget, and it thrashes. A GPU expert cache (`--moe-cache-mib`) just
landed, but it fragments the graph: 2 splits per token become 50, and you lose
~10% even when everything fits.

**vLLM** has `--cpu-offload-gb N`: N GB of weights go to pinned RAM, and the GPU
reads them over PCIe on every step, whether the router picked them or not. No
cache, no disk. That makes sense for vLLM, where big batches end up using almost
every expert anyway, less so for one user on one GPU.

**Transformers** offloads through [Accelerate](https://github.com/huggingface/accelerate) (`device_map="auto"`,
`offload_folder`). Accelerate puts each *module* on the GPU, the CPU or disk,
and hooks copy its weights to the GPU before `forward`. For a MoE, the module is
a layer's whole expert bank, so every token copies every expert of every
offloaded layer to use 8 or 10 of them. No cache, no notion of routing. And on a
Mac, CPU and GPU share the same memory, so "offload to CPU" frees nothing.

I ran all of them on [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) on one A10G (24 GB), with plenty of RAM:

| Engine                                     | Decode (tok/s) |
|--------------------------------------------|----------------|
| Strata, Q8_0                               | 82 - 88        |
| llama.cpp, Q8_0, `--fit on`                | 63 - 64        |
| llama.cpp, BF16, `--fit on`                | 35 - 36        |
| vLLM, BF16, `--cpu-offload-gb 50`          | 3.3            |
| Transformers + Accelerate                  | 0.16           |

If the experts fit in RAM, llama.cpp is hard to beat. If they don't, you have to
stream only what the router picks, through a bounded cache, fast enough that the
SSD is the bottleneck and not Python. That's Qwen3.8-Flash-Next on a 32 GB Mac,
and Transformers can't do it today.

## Making Qwen3.8-Flash-Next work on my 32 GB Mac with Transformers

Strata was already too slow for me on the laptop, and Transformers was not going
to beat it anytime soon. But the Accelerate path was so far off that I wanted to
try fixing offloading in Transformers itself, with the big model on my laptop as
the target, knowing it would probably stay too slow for how I work.

All of the code was built with AI: the Transformers PR, the model-specific
package and the Metal kernels. My part was choosing what to try, reading the
profiles, and deciding what to keep. Everything is in a Space:
[tarekziade/moe-expert-streaming](https://huggingface.co/spaces/tarekziade/moe-expert-streaming).

### Start small

I started with [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924), which fits in memory, so I could compare every token
with vanilla Transformers.

First version: experts on disk, an LRU cache on the device, and a Python loop
running one expert at a time. With every expert already cached, it did 10.1
tok/s. Vanilla did ~32. The I/O was fine, my loop was not: vanilla uses
`grouped_mm`, which runs all the experts of a layer in one kernel.

So I turned the cache into a slot pool: one preallocated device tensor shared by
all layers, plus a per-layer table from expert id to slot. Remap the routing to
slots and the pool looks like a regular expert bank, so `grouped_mm` works on it
as is. No custom expert matmul needed. That got it to 34.6 tok/s, against 38.1
for vanilla.

### `experts_offload`

I then turned the standalone loader into an option of `from_pretrained`, in
[huggingface/transformers#50717](https://github.com/huggingface/transformers/pull/50717):

```python
model = AutoModelForCausalLM.from_pretrained(
    "qwen38-flash-next/dense",
    dtype=torch.bfloat16,
    device_map={"": "mps", "experts.*": "disk"},
    quantization_config=MetalConfig(bits=8),
    experts_offload={
        "cache_bytes": 12 << 30,
        "quantized_experts": "qwen38-flash-next/experts-q3",
    },
)
```

Experts mapped to `"disk"` go through the streaming path instead of Accelerate's
offload: the loader skips their tensors and MoE layers switch to an offloaded
implementation. Misses are read by 8 I/O threads, one contiguous read per
expert. The next two layers' routers run early on the current hidden states to
guess which experts to prefetch. Experts can be quantized offline so reads are
smaller. It works for MoE models with the standard experts layout.

The API is still being discussed with the Transformers maintainers. The
`device_map` form came out of that, and quantized experts will probably go
through the per-layer quantization plan of
[#50245](https://github.com/huggingface/transformers/pull/50245) rather than
their own option.

### Running 121 GB of experts on a 32 GB Mac

Qwen3.8-Flash-Next has 48 layers of 512 experts (10 active), and a 51 GB n-gram
embedding table. The experts alone are 121 GB in FP8. So:

- experts quantized to 3 bits: 49 GB on disk, 12 GiB cached on the device
- dense layers in 8 bits (~5.4 GB, +0.2% perplexity vs BF16)
- the n-gram table stays on disk, rows read when needed

The conversion takes ~5 hours. After loading, it uses ~18 GiB, and a 6K prompt
peaks at 25.4 GiB. That's about the limit: I have to close the browser, or macOS
starts swapping.

The model-specific bits (the n-gram table, Metal kernels) live in a small
package, [moe-expert-streaming](https://github.com/tarekziade/moe-expert-streaming),
on top of the PR. The first version ran at about 10 tok/s.

Then I installed Strata on the same M5, with the same prompts:

|                               | Transformers | Strata | Gap  |
|-------------------------------|--------------|--------|------|
| Decode, 200 tokens            | 10.0 tok/s   | 14.8   | 1.5x |
| Prefill, 4K prompt            | 68 tok/s     | 220    | 3.2x |
| Decode after a 6K document    | 7.3 tok/s    | 13.7   | 1.9x |

Strata 0.1.60 used IQ3_XXS experts. Both ran greedy, with thinking, MTP and
speculative drafts off, and 1,024-token prefill chunks. The full setup and raw
results are on the Space's
[details page](https://tarekziade-moe-expert-streaming.static.hf.space/details.html).

## Closing the gap

Most of the time wasn't spent on experts. Here's what helped:

- **Committing GPU work early.** [PyTorch](https://github.com/pytorch/pytorch) on MPS holds small kernels until the
  next sync, so the GPU idled while Python worked. Forcing a commit after each
  block gave +10%. This should really be fixed in PyTorch.
- **Metal kernels** for the parts without a fast path on macOS: Gated DeltaNet
  and the hyper-connections.
- **Strata's own weights.** I switched to [Strata's GGUF weights](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), reducing dense
  weight traffic from 5.4 GB to 3.9 GB per token and individual expert size from
  2.15 MB to 1.75 MB, with a separate cache pool for each expert size.
- **Prefill**, from 68 to ~228 tok/s: bigger chunks, and the next layer's
  experts read while the current one computes.
- **Long documents**: attention over the indexer's selected blocks only, and a
  KV cache that grows in place instead of being copied at every token.
- **Misses**: Strata doesn't hide them, it waits like I do, and no cache policy
  beat LRU by much. What helped was reading just one guessed expert per layer,
  while the SSD would otherwise be idle.

Here's one MoE layer for one token, as it runs now:

![For one layer and one token, the router picks 10 of 512 experts. Eight are already in the expert cache and two are read from the SSD into free cache slots. The experts then run on the GPU from the cache slots, while the SSD reads one guessed expert for the next layer, which enters the cache only if it's used.](/blog/assets/moe-expert-streaming/expert-streaming.svg)

Same M5, same prompts:

|                               | Transformers | Strata | Gap   |
|-------------------------------|--------------|--------|-------|
| Decode, 200 tokens            | 13.9 tok/s   | 14.8   | 1.06x |
| Prefill, 4K prompt            | 228 tok/s    | 220    | 1.0x  |
| Decode after a 6K document    | 12.4 tok/s   | 13.7   | 1.1x  |

That's about 6% behind Strata on short decode, with slightly faster prefill,
all through Transformers.

Most of what's left is host round trips, ~10 ms per token. Waking up after a
sync is only ~1 ms of that. The rest is the GPU running dry while Python queues
the next layer. Strata queues the whole token's work up front, and the GPU waits
on a Metal shared event at each layer while the host reads experts. PyTorch
doesn't expose that. I tried to fake it with a kernel spinning on a flag the CPU
writes, and the GPU took anywhere from a second to never to see the write. The
real fix would be a native op on PyTorch's MPS stream, with a risk of hangs, so
I haven't built it.

## Conclusion

It runs at ~14 tok/s, close to Strata, and it's still too slow for how I use
coding agents. Orthrus keeps doing the real work.

Closing the remaining gap would take deeper changes in the stack, starting with control
over that per-layer sync, which PyTorch doesn't give you today. Still, I find it
pretty amazing to get this close while staying in Python. There's no C++
extension in the project: just Python, Metal kernels compiled at runtime with
`torch.mps.compile_shader`, and prebuilt [kernels from the Hub](https://huggingface.co/kernels-community).
All the Metal kernels written for this experiment are now
[published on the Hub](https://tarekziade-moe-expert-streaming.static.hf.space/details.html#published-kernels),
one repository each, and the project loads them from there.

The expert streaming is generic and lives in
[huggingface/transformers#50717](https://github.com/huggingface/transformers/pull/50717),
and the numbers are in the [Space](https://huggingface.co/spaces/tarekziade/moe-expert-streaming).
Feedback on the approach and the API is welcome.

Finally, thanks to Niko Veit for [Strata](https://github.com/niko1221/strata).
Strata set the bar for what's possible with expert streaming on consumer
hardware, and having those numbers to chase is what pushed this work as far as
it went.
