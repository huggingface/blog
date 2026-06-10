---
title: "Profiling in PyTorch (Part 2): From nn.Linear to a Fused MLP"
thumbnail: /blog/assets/torch-mlp-fusion/thumbnail.png
authors:
  - user: ariG23498
  - user: ror
---

# Profiling in PyTorch (Part 2): From nn.Linear to a Fused MLP

![Thumbnail of the blog post](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/thumbnail.png)

In the [first part of this series "Profiling in PyTorch"](https://huggingface.co/blog/torch-profiler), we used `torch.add(torch.matmul(x, w), b)` to learn how to read PyTorch profiler traces. We also discussed several other topics that came our way -- the CPU dispatch chain, launch overhead, the difference between an overhead-bound and a compute-bound regime, and some internals of `torch.compile`.

In the second iteration (this blog post), we climb one rung up the ladder. We replace the hand-written matmul-add pair with an `nn.Linear` (with `bias=True`). This is the building block every deep learning model uses. We then stack three of them, with an activation in between, to form a Multilayer Perceptron (MLP) block.

> [!NOTE]
> The scripts for this blog post live here: [`02_linear.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/02_linear.py), [`03_simple_mlp.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/03_simple_mlp.py), and [`03_kernels_mlp.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/03_kernels_mlp.py). Like before, it helps to open them in a separate tab and walk through the code as you read. We use an `NVIDIA A100-SXM4-80GB` GPU to run the scripts. It is really easy to set up a GPU on the Hugging Face infrastructure and experiment with the scripts using [Dev Mode with Spaces](https://huggingface.co/docs/hub/spaces-dev-mode). One could also run the scripts with the [Hugging Face Jobs pipeline](https://huggingface.co/docs/huggingface_hub/en/guides/jobs).

Before we begin, a quick recap of two ideas we will lean on repeatedly:

1. A GPU **kernel** is a program that runs in parallel on many threads of the GPU.
2. The CPU **schedules and launches** these kernels. Most of the PyTorch overhead you see in a profiler trace is this scheduling work.


## From matmul-add to Linear

`nn.Linear` is a module wrapper around the same matrix multiplication and addition we already profiled in [Part 1](https://huggingface.co/blog/torch-profiler). The only difference is that it owns its weight and bias as parameters and exposes a `forward` method that PyTorch users have grown familiar with.

```py
# bias=True would truly emulate the multiplication and addition
# operations we have seen in part 1 of the series
linear_layer = nn.Linear(in_dim, out_dim, bias=True)
y = linear_layer(x)
```

The operation at hand can be written as:

```
y = x @ w.T + b
```

Where `x` is the input, `w` is the weight and `b` is the bias. Let's run [`02_linear.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/02_linear.py) and check the profile.

```bash
uv run 02_linear.py --batch 1024 --in_dim 32 --out_dim 64
uvx trace-util traces -b traces
```

> [!TIP]
> [`trace-util`](https://x.com/ariG23498/status/2054811716727517374) is a utility that will sync your traces to a [Hugging Face bucket](https://huggingface.co/storage) and then provide the [Preffeto URLs](https://perfetto.dev/) on your terminal.

| ![PyTorch profiler trace of an `nn.Linear` forward pass: three short Profile Steps and `linear_fwd` annotations on the CPU lane, a tiny kernel on the GPU lane, and a long `cudaDeviceSynchronize` bar at the end](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/linear-profile-trace.png) |
| :--: |
| Figure 1: Profiler trace of `nn.Linear` |

Figure 1 shows the profiler trace of a forward call of the linear layer. We trace the `forward` call of the linear layer with a similar `schedule` setup as the previous traces, with `wait=1`, `warmup=1` and `active=3`. This is why we see three Profile Steps in the CPU and GPU lanes.

### What is the transpose doing?

| ![Zoomed in CPU dispatch chain showing the aten::t transpose op nested before aten::addmm inside aten::linear, with no matching activity on the GPU lane](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/transpose-cpu-dispatch.png) |
| :--: |
| Figure 2: The transpose CPU row |

If we zoom into the profiler trace, as we do in Figure 2, we notice an `aten::t` (transpose) op before the `aten::addmm` (multiplication and addition) op. We can already figure out that `nn.Linear` transposes the weight parameter and then multiplies it with the input. This is the reason we see an `aten::t` op.

An important thing to notice is that `aten::t` does not really copy or reorganize data: it only rewrites tensor metadata (shape and stride) on the CPU to represent the transposed matrix. It does not launch a kernel on the GPU. One can verify this two ways: by looking at the GPU lane in the trace, or by checking the `aten::t` row in the profiler table and the time it took on CUDA.

### Why are there no separate `mul` and `add` kernels?

| ![Profiler trace of the linear layer with the dispatch chain highlighted, showing aten::linear, aten::t and aten::addmm but no separate aten::add op](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/no-aten-add.png) |
| :--: |
| Figure 3: No `aten::add` in the profile of a linear layer |

There is no `aten::add` (the bias addition) in the dispatch chain of the linear layer, as seen in Figure 3. This is because the bias addition has been _folded_ into the matrix multiplication kernel, using what is called an **epilogue**.

An **epilogue** is a small computation that a GEMM (GEneral Matrix Multiply) kernel does at the very end, just before it writes its result back to HBM (High Bandwidth Memory, the GPU's main memory). Adding a bias, applying an activation, or scaling by a constant are all classic epilogues. The point of an epilogue is to avoid loading or writing to HBM a second time, since memory traffic makes an operation expensive.

`nn.Linear` calls `torch.nn.functional.linear`, which, in turn, calls `aten::linear`. `aten::linear` looks at the inputs, notices that a bias was passed, and dispatches `aten::addmm(bias, x, weight)` instead of doing a matmul and an add separately. `addmm` computes:

```
out = x @ weight.T + bias
```

The cuBLAS GEMM kernel that runs on the GPU has a bias-add variant built in, and that's the kernel `aten::addmm` picks. The add never appears as a separate kernel because it is **part of the matmul kernel's writeback**, which is exactly what an epilogue is.

This is the moment to notice something subtle. The kernel you saw in [Part 1 under `--compile`](https://huggingface.co/blog/torch-profiler#did-we-fuse-the-matmul-and-add-kernels-into-one) (`addmm`) is the kernel that eager `nn.Linear` already uses. There is nothing left for `torch.compile` to fuse here, which is the next thing we will verify.

### Can --compile help a single Linear?

Let's compile the forward call and look at the profiler trace. (The profiler trace is visualized in the [next section](#where-did-the-transpose-go-kernel-layouts-and-pre-ops))

```bash
uv run 02_linear.py --batch 1024 --in_dim 32 --out_dim 64 --compile
uvx trace-util traces -b traces
```

If you compare the eager and compiled traces for a single `nn.Linear`'s `forward`, you will find:

- The same cuBLAS GEMM kernel on the GPU.
- The same `aten::addmm` op on the CPU.
- A few extra rows on the CPU lane unique to compile.

This is worth internalizing. A common reflex is to reach for `torch.compile` whenever a model feels slow. For a single GEMM-with-bias, compile has very little to do. This is not a bug, this is just that compile needs more than one operation to possibly do any fusing. Let's prove that by [looking at an MLP](#stacking-two-linears-the-mlp).

### Where did the transpose go? Kernel layouts and pre-ops

A careful reader of the two traces (eager vs compile) will notice that the eager CPU dispatch chain has more in it than the compiled one.

| ![Eager CPU dispatch chain with the aten::t transpose and aten::addmm boxed separately under aten::linear](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/eager.png) |
| :--: |
| Figure 4: Eager dispatch chain where `aten::linear` walks through `aten::t` (transpose) and then `aten::addmm` |

| ![Compiled CPU dispatch chain showing a Torch-Compiled Region and a single aten::addmm call, with no transpose op](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/compile.png) |
| :--: |
| Figure 5: Compiled dispatch chain where `aten::addmm` is called directly, with no transpose |

The eager CPU dispatch chain inside `aten::linear` is `aten::t` followed by `aten::addmm` (Figure 4). To understand what `aten::t` actually does, we need a quick detour into *strides* and *views*.

A tensor stores its data as one flat, contiguous run of numbers in memory. The `shape` and `stride` are metadata that sit on top of that run and tell PyTorch how to walk it: a stride of `(s0, s1)` means "step `s0` elements to move one row, step `s1` to move one column". Change the metadata and you get a different *view* of the *same* raw data, with no copy:

```py
>>> M = torch.tensor([[0, 1],
...                   [2, 3],
...                   [4, 5]])
>>> M.shape, M.stride()
(torch.Size([3, 2]), (2, 1))   # two steps per row, one step per column

>>> T = M.t()                  # transpose
>>> T.shape, T.stride()
(torch.Size([2, 3]), (1, 2))   # shape and stride swapped, data untouched
>>> T
tensor([[0, 2, 4],
        [1, 3, 5]])
>>> T.flatten()                # forced to materialize, so the data is reordered
tensor([0, 2, 4, 1, 3, 5])
```

`M.t()` did not move a single number. It returned a new view whose strides are swapped, so reading it row-by-row now walks the original buffer `0, 1, 2, 3, 4, 5` in transposed order. The underlying data is identical; only the metadata differs.

This is exactly what `aten::t` does inside the linear layer: it does not allocate a new tensor or copy any data, it produces a *view* of the weight with rewritten strides.

As we can see in Figure 5, compile did not remove a GPU kernel: it removed the *CPU overhead* of dispatching that view. Inductor traced through the view chain at compile time, computed the resulting strides once, and emitted a direct `aten::addmm` call with those strides hard-coded. A few microseconds of CPU work disappear while the GPU does identical math. 

As one would expect, when the input data violates the strides precomputed by the compiler, it will throw an error.

If you look at the GPU lane in both traces, there is exactly one kernel per forward, and it is the *same* kernel both times:

```
cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align8
```

If no transpose kernel ran, who taught the GEMM to read the weight matrix in transposed order? The answer is in the kernel's name. Look at the suffix:

```
cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align8
                                                          ^^
```

That `tn` is the layout descriptor. cuBLAS and CUTLASS precompile a *separate kernel binary* for each combination of input layouts.

`n` (non-transposed) and `t` (transposed) describe how a kernel walks its input during the inner loop. The dispatcher's job is to look at the input strides, decide which suffix combination matches, and pick the right precompiled kernel.

> [!TIP]
> The kernel name in a profiler trace is a hash dump of the kernel's identity. If two runs show the same kernel name, the GPU is doing the same work. If they differ (e.g., `_tn_` vs `_nn_`, `bf16` vs `fp16`, or `s16816gemm` vs `s161616gemm`) then the GPU is doing different work, and the dispatcher took a different branch. Learning to read this name is one of the most useful habits when comparing traces.

## Stacking three Linears: the MLP

In this section, we will profile a Multilayer Perceptron (MLP). To make this more interesting, we will profile a feed-forward network with the GeGLU activation variant. This is also our way of paying tribute to one of the greatest lines ever written in the history of deep learning research (Figure 6).

| ![Conclusions section of the GLU Variants Improve Transformer paper, with the closing sentence attributing the architectures' success to divine benevolence highlighted](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/geglu-paper.png) |
| :--: |
| Figure 6: The conclusion section of the [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) paper.|


```py
class SimpleGeGLUMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = F.gelu(g, approximate="tanh")
        m = h * u
        y = self.down_proj(m)
        return y
```

You will find the entire script here: [`03_simple_mlp.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/03_simple_mlp.py). Execute it like so:

```bash
uv run 03_simple_mlp.py --batch 64 --seq 128 --dim 768 --hidden 3072
uvx trace-util traces -b traces
```

Before we open the trace, let's think together about what we should expect to see. The `forward` function does a fair amount of computation, but most of it is already familiar to us.

We should expect three `aten::linear` dispatches, one for each `nn.Linear` layer. We should also expect two pointwise kernel launches, one for the GeLU and one for the multiplication. Forming this expectation before looking is the single most useful habit in the profiling journey: you read the trace to *confirm or break* a guess, not to form one from scratch.

| ![Profiler trace of the GeGLU MLP forward pass, with five boxed groups on the CPU lane labelled linear, linear, gelu, mul, linear](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/simple-mlp-eager.png) |
| :--: |
| Figure 7: The profiler trace for a GeGLU MLP |

From Figure 7 we can pat ourselves on the back, as our intuition was correct. Per forward pass (one `mlp_fwd`), the GPU runs exactly 5 kernels:

| Op | CPU op | GPU kernel | launches |
| :--: | :--: | :--: | :--: |
| `gate_proj`| `aten::linear` | `ampere_bf16_s16816gemm_bf16_128x128_...`| occupancy query + cudaLaunchKernel |
| `up_proj` | `aten::linear` | `ampere_bf16_s16816gemm_bf16_128x128_...`| occupancy query + cudaLaunchKernel |
| `gelu` | `aten::gelu`| `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl...>` | cudaLaunchKernel |
| `h * u` | `aten::mul` | `vectorized_elementwise_kernel<4, ...MulFunctor...>` | cudaLaunchKernel |
| `down_proj` | `aten::linear` | `ampere_bf16_s16816gemm_bf16_128x256_...` | occupancy query + cudaLaunchKernel |

The three GEMMs each do an extra `cudaOccupancyMaxActiveBlocksPerMultiprocessor` call before the launch. We have a separate section on this in Part 1, [you can find it here](https://huggingface.co/blog/torch-profiler#why-does-matmul-have-an-extra-cuda-runtime-call). That is cuBLAS sizing the grid. The pointwise ops (GeLU and mul) launch directly, with no occupancy query. So "a linear" is actually query + launch, while "a pointwise op" is just launch.

| ![Profiler table for the GeGLU MLP listing op names and their CUDA times, where metadata ops like aten::transpose and aten::as_strided show 0.000us of CUDA time](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/simple-mlp-table.png) |
| :--: |
| Figure 8: The table shows that some ops launch zero kernels |

The `aten::t`, `aten::transpose`, `aten::reshape`, `aten::view`, `aten::as_strided`, and `aten::_unsafe_view` ops launch zero kernels. They show `0.000us` of CUDA time in the table (Figure 8) because they only rewrite tensor metadata (shape and stride) on the CPU. A reader scanning the table sees around six op names per linear, but only one of them (`mm`) ever reaches the GPU.

### Why are there two types of GEMM kernels?

The MLP flattens `[batch, seq, dim]` to `[batch * seq, dim]` for the matmul, so the `8192` below is `batch * seq = 64 * 128`.

From the trace:

| Linear | `aten::mm` input dims | M·K·N | cuBLAS kernel | avg CUDA |
| :--: | :--: | :--: | :--: | :--: |
| `gate_proj` | `[8192,768] x [768,3072]` | `8192·768·3072` | `…128x128…stages_32x5_tn` | 0.19ms |
| `up_proj` | `[8192,768] x [768,3072]` | `8192·768·3072` | `…128x128…stages_32x5_tn` | 0.19ms |
| `down_proj` | `[8192,3072] x [3072,768]` | `8192·3072·768` | `…128x256…stages_64x3_tn` | 0.17ms |

All three GEMMs have the same FLOP count, `2·8192·768·3072 ≈ 38.7 GFLOP` each, yet `down_proj` is about `10%` faster. Same work, different shape (`N=768` instead of `3072`), so cuBLAS picks a different tile (`128×256`, with a deeper `stages_64x3` pipeline) that gets better reuse for that shape.

> [!NOTE]
> If you want to learn more about tiling in depth, [here is a great resource](https://alvinwan.com/how-to-tile-matrix-multiplication/) to get started with.

This is exactly why the table had two GEMM rows: the `128x128` row is gate+up and the `128x256` row is down.

### What happens with `torch.compile`?

Before compiling the `forward` method and visualizing it, let's do the mental exercise again of asking ourselves what we expect to see in the trace. This is a fun experiment, and an important one to repeat every time you profile something yourself. Always build on your intuition, and the moment something does not match, stop and figure out why.

```bash
uv run 03_simple_mlp.py --batch 64 --seq 128 --dim 768 --hidden 3072 --compile
uvx trace-util traces -b traces
```

| ![Profiler trace of the compiled GeGLU MLP showing three aten::mm calls and one fused triton kernel on the CPU lane, labelled mm, mm, fused, mm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/simple-mlp-compile-trace.png) |
| :--: |
| Figure 9: The profiler trace for the compiled GeGLU MLP |

In eager mode, each `nn.Linear` was expanded into a chain of dispatcher ops (`aten::linear` → `aten::t` → `aten::transpose` → `aten::matmul` → `aten::reshape` → `aten::mm`). Those are the high-level wrappers that ATen walks through before reaching the real GEMM. `torch.compile` removes that chain.

By the time the compiled graph runs, there is no linear, no matmul, no transpose or reshape and those metadata ops were folded into how `mm` is called. We can see three bare `aten::mm` external calls (Figure 9). The proof that it is the same GEMM is that the kernel names are byte-for-byte identical to eager: `...128x128...stages_32x5_tn` for gate and up, and `...128x256...stages_64x3_tn` for down.

### The fused Triton kernel

| ![Compiled MLP trace with the triton_poi_fused__unsafe_view_gelu_mul_0 kernel boxed on the CPU lane, replacing the separate gelu and mul kernels from the eager run](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/fused.png) |
| :--: |
| Figure 10: The fused Triton kernel |

This is the headline of the whole compile lesson. The two eager pointwise kernels (GeLU and mul) plus a reshape collapsed into one kernel, `triton_poi_fused__unsafe_view_gelu_mul_0` (Figure 10). Let's decode the name:

* `triton`: generated by Inductor's Triton backend (not cuBLAS, not ATen).
* `poi`: pointwise (Inductor tags pointwise kernels `poi`, reductions `red`, and persistent reductions `per`).
* `fused__unsafe_view_gelu_mul`: the ops it merged: the `_unsafe_view` (reshape), the GeLU, and the mul.
* `0`: the unique id within the graph.

Why is this a win? In eager mode, the intermediate `h = gelu(g)` is a full `[8192, 3072]` bf16 tensor (around 50 MB) that the GeLU kernel writes to HBM and the mul kernel immediately reads back. Fusion keeps it in registers (memory that resides inside the chip and are closer than the HBM). The Triton kernel reads `g` and `u` once, computes `gelu(g) * u`, and writes the result once. One whole round trip of the intermediate through global memory is gone.

## Let's use hand tuned kernels

So far we have let PyTorch (eager) and the compiler (`torch.compile`) pick our kernels. Now we plug in a kernel that a human expert wrote and tuned by hand. We use the `LigerGEGLUMLP` layer, but instead of installing the [Liger library](https://github.com/linkedin/Liger-Kernel) and importing it, we fetch it from the [Hugging Face Hub](https://huggingface.co/kernels/kernels-community/liger-kernels) with the `kernels` library.

```python
from kernels import get_kernel

kernels_layers = get_kernel("kernels-community/liger-kernels", version=1).layers
kernels_geglu_mlp = kernels_layers.LigerGEGLUMLP(Config()).to(device, dtype=torch.bfloat16).eval()
```

The full script is here: [`03_kernels_mlp.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/03_kernels_mlp.py).

```bash
uv run 03_kernels_mlp.py --batch 64 --seq 128 --dim 768 --hidden 3072
uvx trace-util traces -b traces
```

| ![Profiler trace of the LigerGEGLUMLP forward pass showing three aten::linear groups and a single LigerGELUMulFunction group on the CPU lane](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/kernels-profile.png) |
| :--: |
| Figure 13: The profiler trace for the `LigerGEGLUMLP` layer |

Figure 13 shows the profile for the `LigerGEGLUMLP` layer using the Liger kernels from the Hub.

### Why use the kernels library

Writing kernels in Triton or CUDA is one problem and *shipping* them is another. The kernel has to be compiled for your exact combination of GPU architecture, CUDA version, and PyTorch version. This is the step that usually breaks ("works on my machine", missing `nvcc`, wrong Triton version).

The [`kernels`](https://github.com/huggingface/kernels) library moves that build step off your machine. `get_kernel("kernels-community/liger-kernels", version=1)` downloads a **pre-built, version-pinned** kernel package from the Hugging Face Hub and caches it locally (here under `~/.cache/...kernels-community--liger-kernels`). The benefits are:

* The kernels are compiled once, in CI, for many architectures and version combinations. You download the right binary instead of compiling it yourself.
* `version=1` pins the exact build, so everyone running your script gets the same kernel. There is no "it got slower after I updated a package".
* The package exposes a `.layers` attribute with drop-in `nn.Module`s (like `LigerGEGLUMLP`). You swap your module for theirs and nothing else in your model changes.

### Why tuned kernels are better

When we say "tuned", we mean two concrete things, and both are visible in the trace.

| ![Compiled MLP trace with the TorchDynamo, prologue and guard pre-ops boxed on the CPU lane before the compiled graph runs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/compile-preops.png) |
| :--: |
| Figure 14: The compiled run pays for pre-ops (Dynamo, guards, prologue) before any GEMM runs |

| ![LigerGEGLUMLP trace with an empty box where the compile pre-ops would be, showing the hand-written kernel has no Dynamo or guard overhead](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/no-preops.png) |
| :--: |
| Figure 15: The Liger kernel has no pre-ops — the box where they would be is empty |

1. **The fusion is baked in.** The [`LigerGEGLUMLP`](https://huggingface.co/kernels/kernels-community/liger-kernels/blob/v1/build/torch-cuda/layers.py#L292) forward is `down_proj(LigerGELUMulFunction.apply(gate_proj(x), up_proj(x)))`. The [`LigerGELUMulFunction`](https://huggingface.co/kernels/kernels-community/liger-kernels/blob/v1/build/torch-cuda/geglu.py#L130) runs a single Triton kernel, [`_geglu_tanh_forward_kernel`](https://huggingface.co/kernels/kernels-community/liger-kernels/blob/v1/build/torch-cuda/geglu.py#L97), that computes `gelu(gate) * up` in one pass. This is exactly what we saw from `torch.compile`, where the intermediate never makes a round-trip through HBM. We get it here **without the compiler**, as shown in Figures 14 and 15 (no Dynamo guards, no compile latency, no recompilation risk).

2. **The launch parameters were chosen for the hardware.** The kernel does not guess its block size at random. Liger's [`calculate_settings`](https://huggingface.co/kernels/kernels-community/liger-kernels/blob/v1/build/torch-cuda/geglu.py#L95) picks them from the column count.

It is worth being honest about what "tuned" does **not** mean. It does not mean "always the fastest for your exact shape". The fused kernel here runs in **92.8 µs**, while Inductor's fused kernel from the compile run was **89.4 µs**. The tuning heuristic is a good general rule (next power of two, capped warps), not an exhaustive per-shape search.

## Conclusion

The table below collects what each step changed on the GPU and what it left untouched.

| Setup | What changed | What stayed the same |
| :-- | :-- | :-- |
| Eager `nn.Linear` | Baseline: bias add is already folded into the GEMM epilogue (`addmm`), so it is *one* cuBLAS kernel, not a matmul plus an add | — |
| Compiled `nn.Linear` | A few CPU dispatch ops (the `aten::t` view bookkeeping) disappear | Same single cuBLAS GEMM kernel, byte-for-byte. Compile has nothing to fuse |
| Eager MLP | 5 GPU kernels: 3 GEMMs + a GeLU + a mul. The `[8192, 3072]` intermediate makes a full round-trip through HBM | Each GEMM is still the same bias-free cuBLAS kernel as a standalone linear |
| Compiled MLP | GeLU + mul + reshape collapse into **one** fused Triton kernel; the intermediate stays in registers. Pays compile pre-ops (Dynamo, guards) | The 3 GEMMs are untouched with identical cuBLAS kernel names |
| Liger MLP | Same fusion, but baked into a hand-written Triton kernel with hardware-tuned launch params with **no** Dynamo, guards, or compile latency | The 3 GEMMs are still the same cuBLAS kernels |

If there is one habit to carry forward, it is the one we practiced before every trace: **guess first, then look.** State what you expect the trace to contain, open it, and treat any mismatch as the most interesting thing on the screen.

This was the second stop in the **Profiling in PyTorch** series. In the next post we will keep climbing the ladder, moving from this MLP block towards the attention block and, eventually, a full model.

Thanks to [Noe Flandre](https://huggingface.co/NoeFlandre) and [Pedro Gabriel Gengo Lourenço](https://huggingface.co/pedrogengo) for their reviews on the early draft of the post!