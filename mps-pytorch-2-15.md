---
title: "MPS in PyTorch 2.15"
thumbnail: /blog/assets/mps-pytorch-2-15/thumbnail.png
authors:
- user: Isalia20
- user: khushi-411
---

# MPS in PyTorch 2.15

If you run models on your Mac or use MPS as your primary acceleration tool for computing, there's plenty to look forward to in PyTorch 2.15! FP8 arrives on MPS, language-model decoding gets faster paths through linear layers and softmax and new Metal kernels tackle some surprisingly expensive parts of audio, video and training workloads.

A quick word about who we are: we're the PyTorch team at Hugging Face. We contribute across PyTorch, and one of our focus areas is the MPS backend, where the goal is to make your Mac a first-class place to run and train PyTorch models. At the start of 2026, running even small models on MPS often meant working around unsupported operations, performance cliffs and unexpected memory growth. So we run real models from the Hugging Face Hub across language, image and video, find where MPS falls short and fix it upstream in PyTorch, more and more often by replacing MPSGraph with dedicated Metal kernels. Much of this post is the result of that work, done together with the PyTorch maintainers and community contributors.

## TL;DR

- **FP8 comes to the Mac:** store `float8_e4m3fn` values in half the space of BF16 and use scaled matrix multiplication and all other ops which are supported on CUDA/CPU!
- **Less time per token:** single-row linear layers use specialized matrix-vector kernels and softmax gets dedicated Metal kernels.
- **Better MoE performance:** native grouped matmul keeps group offsets on the GPU, with extra attention to the small groups used during decoding.
- **More linear algebra on the GPU:** complex LU and triangular solves, faster small-matrix inverses, and faster QR decomposition.
- **Audio, video, and training get attention too:** faster paths for convolutions, padding, dropout, interpolation backward and reductions.
- **More control from Python:** public MPS streams and more reliable event timing.
- **Updated requirements:** MPS now requires **macOS 15 or later**; PyTorch 2.15 requires **Python 3.11 or later**.
- **A peek at 2.16:** attention fully on Metal, more MPSGraph-to-Metal migrations and faster LLM decoding.

## Contents

- [Getting started](#getting-started)
- [FP8 comes to MPS](#fp8-comes-to-mps)
- [Faster language-model decoding](#faster-language-model-decoding)
- [Linear algebra gets a boost](#linear-algebra-gets-a-boost)
- [Convolutions, padding, and training](#convolutions-padding-and-training)
- [Softmax and reductions](#softmax-and-reductions)
- [A much smaller memory footprint for video decoding](#a-much-smaller-memory-footprint-for-video-decoding)
- [More control with MPS streams](#more-control-with-mps-streams)
- [More operations work on MPS](#more-operations-work-on-mps)
- [A sneak peek at PyTorch 2.16](#a-sneak-peek-at-pytorch-216)
- [Wrapping up](#wrapping-up)

## Getting started

You'll need an Apple silicon Mac with **macOS 15 or later** and **Python 3.11 or later** (see the [2.15 compatibility matrix](https://github.com/pytorch/pytorch/blob/release/2.15/RELEASE.md#release-compatibility-matrix)). If your OS is too old or no Metal GPU is available, the MPS availability check now tells you why.

To try the release candidate in a separate environment:

```bash
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/test/cpu
```

## FP8 comes to MPS

FP8 gives you another way to save memory on your Mac. PyTorch 2.15 adds [`torch.float8_e4m3fn`](https://github.com/pytorch/pytorch/pull/194074) tensors on MPS: one byte per value, half the storage of FP16 or BF16.

You can multiply these tensors with [`F.scaled_mm`](https://github.com/pytorch/pytorch/pull/196916), which supports both tensorwise and rowwise scaling. Here's a small example with one scale per tensor:

```python
import torch
import torch.nn.functional as F

def quantize_fp8(x):
    limit = torch.finfo(torch.float8_e4m3fn).max
    scale = x.abs().amax().clamp_min(1e-12) / limit
    return (x / scale).to(torch.float8_e4m3fn), scale

x = torch.randn(1, 1024, device="mps", dtype=torch.float32)
weight = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
x_fp8, x_scale = quantize_fp8(x)
w_fp8, w_scale = quantize_fp8(weight)

y = F.scaled_mm(
    x_fp8,
    w_fp8.t(),
    scale_a=x_scale,
    scale_recipe_a=F.ScalingType.TensorWise,
    scale_b=w_scale,
    scale_recipe_b=F.ScalingType.TensorWise,
    output_dtype=torch.bfloat16,
)
print(y.shape, y.dtype)
```

## Faster language-model decoding

Every LLM has a prefill phase and a decode phase. During decode, each new token goes through the model as a single row, so the big matrix multiplications turn into matrix-vector products. PyTorch 2.15 speeds up both kinds of layers that dominate this step: dense linear layers and mixture-of-experts layers.

### Linear layers

MPS has had GEMV kernels for `torch.mm` since 2.14, but `F.linear` still went through MPSGraph. PyTorch 2.15 [connects them to `F.linear`](https://github.com/pytorch/pytorch/pull/198721), so the common `[1, 1, hidden_size]` decoding input uses them directly, with eligible bias additions fused in. Across 96 BF16 layer shapes from real models, that's a **1.23× geometric-mean speedup**. The biggest wins land on small projections, such as the gates in MoE models, where the old path left most of the memory bandwidth unused. Large layers and LM heads were already close to bandwidth-bound, so they stay near 1×.

### Mixture-of-experts layers

A mixture-of-experts layer sends different numbers of tokens to each expert. Grouped matmul handles those uneven groups in a single operation, and 2.15 gives MPS a [native implementation](https://github.com/pytorch/pytorch/pull/194933).

In 2.14, grouped matmul copied the group offsets to the CPU and looped over the groups. The new Metal kernels read the offsets directly on the GPU, using Metal Performance Primitives where available and a separate Metal path elsewhere. Small groups, the common case during decoding, get [their own kernel](https://github.com/pytorch/pytorch/pull/198725) as well.

For example, 64 tokens can be split into groups of 16, 32, and 16:

```python
import torch
import torch.nn.functional as F

tokens = torch.randn(64, 64, device="mps", dtype=torch.bfloat16)
weights = torch.randn(3, 128, 64, device="mps", dtype=torch.bfloat16)
ends = torch.tensor([16, 48, 64], device="mps", dtype=torch.int32)

out = F.grouped_mm(tokens, weights.transpose(-2, -1), offs=ends)
print(out.shape)  # torch.Size([64, 128])
```

The decoding kernel shows up directly in token throughput. On an **M3 Ultra with `torch.compile`**, MoE models generate about **1.3× more tokens per second**:

![Bar chart of MoE decoding throughput on M3 Ultra with torch.compile. Qwen3.6-35B-A3B goes from 38.7 to 51.7 tokens/s at context 512 and 36.8 to 48.7 at context 2048; LFM2.5-8B-A1B goes from 77.1 to 103.2 and 74.9 to 98.7 tokens/s.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mps-pytorch-2-15/moe-decode-throughput.png)

## Linear algebra gets a boost

### Complex-valued computations

Working with complex tensors gets easier. `complex64` [LU factorization](https://github.com/pytorch/pytorch/pull/188473), [LU solve](https://github.com/pytorch/pytorch/pull/195369) and [triangular solve](https://github.com/pytorch/pytorch/pull/195861) now run on MPS, which also unlocks the operations and gradients built on them, such as complex determinants. Here's a small system solved directly on the GPU:

```python
import torch

a = torch.randn(8, 8, device="mps", dtype=torch.complex64)
a = a + 8 * torch.eye(8, device="mps", dtype=torch.complex64)
b = torch.randn(8, 2, device="mps", dtype=torch.complex64)

lu, pivots = torch.linalg.lu_factor(a)
x = torch.linalg.lu_solve(lu, pivots, b)
print((a @ x - b).abs().max().item())
```

### Small matrices and larger solves

Small matrices get fast kernels too. For camera geometry and image transformations, batches of FP32 matrices up to **4 × 4** are [inverted in a single Metal kernel](https://github.com/pytorch/pytorch/pull/195944) that combines factorization and solving. Larger problems get attention as well: [QR uses blocked updates](https://github.com/pytorch/pytorch/pull/198732), and [triangular solves](https://github.com/pytorch/pytorch/pull/196241) turn much of their work into efficient matrix multiplication.

Gated DeltaNet workloads get a [dedicated fast path](https://github.com/pytorch/pytorch/pull/196900) for batches of **16 × 16, 32 × 32 and 64 × 64 FP32 triangular systems** with at least eight right-hand sides.

SVD is also more robust at numerical extremes. It now correctly handles [very small singular values](https://github.com/pytorch/pytorch/pull/200920), [very large or very small input magnitudes](https://github.com/pytorch/pytorch/pull/201037) and [basis completion with `full_matrices=True`](https://github.com/pytorch/pytorch/pull/201198), which in turn makes pseudoinverse and least-squares results more reliable.

## Convolutions, padding, and training

Audio models, video decoders and training loops have their own expensive operations. This release gives several of them new paths through Metal:

| Operation | What changes in 2.15 |
|---|---|
| [`conv1d` forward](https://github.com/pytorch/pytorch/pull/198878) | Specialized kernels replace MPSGraph, plus an im2col-plus-matmul path for short outputs. Useful for audio and streaming-model shapes. |
| [Constant padding](https://github.com/pytorch/pytorch/pull/195368) | Moves to Metal kernels, fixing runaway driver-memory growth in tiled video VAE decoding (more below). |
| [Reflection and replication padding](https://github.com/pytorch/pytorch/pull/196985) | New Metal implementations that are faster and fix backward correctness. |
| [Out-of-place dropout](https://github.com/pytorch/pytorch/pull/195011) | `nn.Dropout` and `F.dropout` now use the existing fused kernel. |
| [Upsampling backward](https://github.com/pytorch/pytorch/pull/199687) | A Metal gather kernel replaces the MPSGraph resize-gradient path. |
| [Channels-last `conv2d` backward](https://github.com/pytorch/pytorch/pull/193623) | Stays on an efficient channels-last path instead of an NCHW graph with costly strided access. |

## Softmax and reductions

Token sampling gets faster in 2.15 as well. [`softmax`](https://github.com/pytorch/pytorch/pull/196071) and [`log_softmax`](https://github.com/pytorch/pytorch/pull/195441) now run on dedicated Metal kernels, and large vocabularies benefit the most: an FP32 softmax over a `[1, 128256]` row of logits drops from **57 µs to 7 µs**, about **8× faster** than the MPSGraph version. The speedup holds across other common vocabulary sizes:

![Bar chart of FP32 softmax time for a single row of logits at five vocabulary sizes from 128,256 to 201,088. MPSGraph takes 50 to 74 microseconds; the Metal kernel takes 7 to 10 microseconds, a 7.1x to 8.0x speedup.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mps-pytorch-2-15/softmax-sampling.png)

## A much smaller memory footprint for video decoding

One of the most striking results in this release comes from the constant-padding change. The VAE decode in `Wan-AI/Wan2.2-TI2V-5B-Diffusers` makes **8,764 padding calls** over a full 28-tile decode. With padding on Metal, peak driver memory drops from about **270 GiB to 8 GiB**:

![Stacked bar chart of peak driver memory during the Wan2.2 tiled VAE decode. With MPSGraph padding, 8.07 GiB of allocator-reserved memory plus 262 GiB outside the allocator gives 270.1 GiB; with Metal padding the total is 8.1 GiB.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mps-pytorch-2-15/wan-vae-memory.png)

| Memory metric | MPSGraph padding | Metal padding |
|---|---:|---:|
| Current tensor-allocation peak | 4.422 GiB | 4.095 GiB |
| Allocator-reserved peak | 8.070 GiB | 8.070 GiB |
| Total driver-memory peak | 270.135 GiB | 8.108 GiB |

## More control with MPS streams

You can now pick which MPS stream runs your work, directly from Python. [`torch.mps.Stream` and the `torch.mps.stream` context manager](https://github.com/pytorch/pytorch/pull/191415) provide the interface, and [`Tensor.record_stream`](https://github.com/pytorch/pytorch/pull/193464) lets the allocator track a tensor's use on another stream.

Here's a small example that sends work to a separate stream and synchronizes before reading the result:

```python
import torch

x = torch.randn(1024, device="mps")
torch.mps.synchronize()
worker = torch.mps.Stream()

with torch.mps.stream(worker):
    y = x.square()
    x.record_stream(worker)

del x
worker.synchronize()
print(y.cpu().shape)
```


## More operations work on MPS

Several more operations now run on MPS: [`torch.frexp`](https://github.com/pytorch/pytorch/pull/194110), [`torch.logspace`](https://github.com/pytorch/pytorch/pull/190495), [`torch.take` and `Tensor.put_`](https://github.com/pytorch/pytorch/pull/192980), [`torch.heaviside`](https://github.com/pytorch/pytorch/pull/195265), [`torch.special.laguerre_polynomial_l`](https://github.com/pytorch/pytorch/pull/195225) and [unsigned-integer comparisons](https://github.com/pytorch/pytorch/pull/198860). `logspace` doesn't yet support integer or boolean outputs with more than one element.

Attention gets a correctness fix too: the MPS math implementation of scaled dot-product attention now [handles batch and head broadcasting correctly](https://github.com/pytorch/pytorch/pull/189631).

## A sneak peek at PyTorch 2.16

Work on 2.16 is already well underway. Some of the changes below have landed on `main` and are available in the nightlies; others are still in review, so details may change before the release.

### Faster LLM decoding

Once a model's decode path no longer contains any MPSGraph ops, the GPU only starts on a token after the CPU has finished encoding all of its kernels. PyTorch 2.16 [commits the command buffer every 16 kernels](https://github.com/pytorch/pytorch/pull/200181), so the GPU gets to work while the CPU is still encoding the rest. For compiled BF16 decoding at a 2048-token context:

| Model | Before | After | Change |
|---|---:|---:|---:|
| LFM2.5-8B-A1B | 108.8 tokens/s | 126.0 tokens/s | +16% |
| Qwen3.6-35B-A3B | 52.9 tokens/s | 62.8 tokens/s | +19% |
| Qwen3.8-27B | 12.1 tokens/s | 12.6 tokens/s | +4% |

### Attention runs entirely on Metal

Scaled dot-product attention is [dropping MPSGraph completely](https://github.com/pytorch/pytorch/pull/201526): every shape will run on Metal kernels, with dedicated vector kernels for decoding and Metal Performance Primitives for half-precision prefill on newer macOS versions. The shapes that used to fall back to MPSGraph benefit the most. On an M5 Pro, decoding with grouped-query attention (32 query heads, 8 KV heads, head dim 80, 4K context) drops from 465 µs to 65 µs, and a 4K × 4K prefill with head dim 40 goes from 42 ms to 6 ms.

### More MPSGraph to Metal migrations

The move away from MPSGraph continues. Batch norm [forward](https://github.com/pytorch/pytorch/pull/201361) and [backward](https://github.com/pytorch/pytorch/pull/201411) are in progress, while [MSE loss](https://github.com/pytorch/pytorch/pull/201351) (up to 12× faster on small inputs), [`std_mean` and `var_mean`](https://github.com/pytorch/pytorch/pull/201310) (up to 4× faster) and [`bincount`](https://github.com/pytorch/pytorch/pull/201045) have already landed. `bincount` is the most dramatic: when most values fall into a single bin, counting 4 million integers used to take 6.8 seconds and now takes half a millisecond.

### A smaller install

Bundled shaders will [build for Metal 3.1 by default](https://github.com/pytorch/pytorch/pull/197529), with Metal 4 versions compiled only for the kernels that need them. That shrinks `libtorch_cpu.dylib` by about 25%, from 384 MiB to 287 MiB, and lays the groundwork for Metal 4.1 shaders with faster matmuls and native FP8 matrix multiplication.

## Wrapping up

PyTorch 2.15 brings FP8 to the Mac, faster LLM decoding for dense and MoE models, more linear algebra on the GPU and a long list of operations moved from MPSGraph to dedicated Metal kernels. Give the release candidate a try!

### What's next for us

- **More MPS:** 2.16 will run attention entirely on Metal, and we'll keep moving operations off MPSGraph.
- **`torch.compile`:** we're working on getting rid of graph breaks, so more models compile end to end.
- **Your ideas:** what else should we work on in PyTorch? If something is slow, missing or broken, on MPS or anywhere else, [open an issue on GitHub](https://github.com/pytorch/pytorch/issues) and let us know.

A big thank you to [Nikita Shulga](https://github.com/malfet), [Kurt Mohler](https://github.com/kurtamohler) and [Aaron Gokaslan](https://github.com/Skylion007) for reviewing much of this work, and to everyone who contributed to MPS in this release.
