---
title: "Profiling in PyTorch (Part 3): Attention is all you profile"
thumbnail: /blog/assets/torch-attention-profile/thumbnail.png
authors:
  - user: ariG23498
  - user: sergiopaniego
---

# Profiling in PyTorch (Part 3): Attention is all you profile

![Thumbnail of the blog post](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/profile-3-thumbnail.png)

<div style="
  border: 1px solid #1f5c48;
  border-radius: 6px;
  padding: 1.5rem 2rem;
  background: #17211f;
  color: #ddd;
  font-size: 1.1rem;
  line-height: 1.7;
">

  <p>
    This is the third post of <bold>Profiling in PyTorch</bold>, a series where we slowly build the skill of reading profiler traces and use it to drive optimization:
  </p>

  <ol>
    <li>
      <a href="https://huggingface.co/blog/torch-profiler" style="color: #10b981;">
        Profiling in PyTorch (Part 1): A Beginner's Guide to torch.profiler
      </a>
    </li>
    <li>
      <a href="https://huggingface.co/blog/torch-mlp-fusion" style="color: #10b981;">
        Profiling in PyTorch (Part 2): From nn.Linear to a Fused MLP
      </a>
    </li>
    <li>
      <a href="https://huggingface.co/blog/torch-attention-profile" style="color: #10b981;">
        Profiling in PyTorch (Part 3): Attention is all you profile
      </a>
      <em style="color: #aaa;">(current)</em>
    </li>
  </ol>

</div>

The series "Profiling in PyTorch" is meant to make you comfortable reading profiler traces and tables. In [Part 1](https://huggingface.co/blog/torch-profiler) we profiled basic math operations like addition and multiplication. We saw how the profiler table uncovers hotspots, and how the profiler trace shows the order in which an algorithm runs over time.

In [Part 2](https://huggingface.co/blog/torch-mlp-fusion) we wrapped that addition and multiplication into a torch linear layer. We then stacked several linear layers on top of each other (a multilayer perceptron) and profiled that. Along the way we also profiled fused and hand-tuned kernels.

The next logical step is another fundamental algorithm, attention. Attention looks like a fairly simple operation, but a lot of clever tricks go into making it fast. Our goal here is not to cover every trick in detail. Instead, we want to see how each one looks different under the profiler.

> [!NOTE]
> The scripts for this blog post live here: [`04_a_naive_attention.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/04_a_naive_attention.py), [`04_b_inplace_ops_attention.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/04_b_inplace_ops_attention.py), [`04_c_sdpa_attention.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/04_c_sdpa_attention.py), and [`04_d_kernels_attention.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/04_d_kernels_attention.py). Like before, it helps to open them in a separate tab and walk through the code as you read. We use an `NVIDIA A100-SXM4-80GB` GPU to run the scripts. It is really easy to set up a GPU on the Hugging Face infrastructure and experiment with the scripts using [Dev Mode with Spaces](https://huggingface.co/docs/hub/spaces-dev-mode). One could also run the scripts with the [Hugging Face Jobs pipeline](https://huggingface.co/docs/huggingface_hub/en/guides/jobs).

## Naive attention

Attention works with Queries (`Q`), Keys (`K`), and Values (`V`). The interaction between them can be written as a short sequence of steps:

1. Build the attention scores `S`: `matmul(Q, K.T)`
2. Scale the scores: `S * scale`
3. Apply a causal mask to the scores: `S.masked_fill(mask, "-inf")`
4. Normalize the scores with softmax to get the attention weights `A`: `softmax(S)`
5. Reweight the values with those weights: `matmul(A, V)`

So attention is really a collection of primitive operations. Some of them we already know (the matmuls), and the rest are easy to spot. Let's write a naive attention module in PyTorch and profile it.

```py
class NaiveCausalAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * self.scale
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out
```

Before opening the trace, let's do our usual exercise and guess what we should see. Tracing the `forward` of this module, we expect:

- a matmul kernel (`Q . K.T`)
- a mul kernel (the scaling)
- an operation for the masking
- a softmax kernel
- a matmul kernel (`A . V`)

```bash
uv run 04_a_naive_attention.py
uvx trace-util -f traces/ -b <hf_uname>/traces
```

| ![CPU lane of the naive attention profiler trace, with the `attn_fwd` block expanded to show its matmul, mul, masked_fill and softmax operations](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cpu-profile-naive.png) |
| :--: |
| Figure 1: The CPU lane of the profile trace for naive attention highlighting the discrete operations |

Figure 1 shows the CPU lane of the profile (the GPU lane is folded so it does not overwhelm us). Inside `attn_fwd` (our annotated forward call) we can see exactly the operations we guessed. The matmul is an old friend by now, and the new operations are easy to place:

- `mul`: the scaling
- `masked_fill`: the causal masking
- `softmax`: the softmax kernel

Now let's unfold the GPU lane and see which kernels were actually launched.

| ![Profiler trace of naive attention showing the CPU lane above the GPU lane, with each `attn_fwd` step mapping to a cluster of GPU kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/gpu-profile-naive.png) |
| :--: |
| Figure 2: GPU and CPU lanes of the profile trace for naive attention highlighting a collection of kernels corresponding to one profiler step. |

Figure 2 shows the GPU lane next to the CPU lane. Let's zoom into a single `attn_fwd` block on the GPU lane to look at the kernels one by one.

| ![Zoomed-in GPU lane of naive attention showing the individual kernels for one step: two matmuls, a mul, a memory copy, a masking kernel and a softmax](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/each-kernels-naive.png) |
| :--: |
| Figure 3: Zoomed in GPU lane of the profiler trace for naive attention implementation. |

Figure 3 lets us read off the individual kernels for one profiler step:

1. matmul (query and key)
2. mul (scaling)
3. memory copy 🤔
4. causal masking
5. softmax (produces the attention weights)
6. matmul (attention weights and values)

Five of these are expected. The memory copy is the odd one out, so where does this come from? The clue is that PyTorch has in-place operations. When you operate on a tensor the ordinary (out-of-place) way, PyTorch often makes a copy, applies the operation to it, and returns the copy. Following the sequence of operations, the culprit here is our `masked_fill`.

What if we replaced this with an in-place operation?

## Naive attention with inplace causal masking

All we change is `masked_fill` to `masked_fill_` (note the trailing underscore, PyTorch's convention for in-place operations), and we run the same script.

```diff
def forward(self, q, k, v, mask):
    # q, k, v: [batch, heads, seq, head_dim]
    scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq, seq]
    scores = torch.mul(scores, self.scale)
-    scores = scores.masked_fill(mask, float("-inf"))
+    scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # [batch, heads, seq, head_dim]
    return out
```

Let's look at the trace and see if something changed.

```bash
uv run 04_b_inplace_ops_attention.py
uvx trace-util -f traces/ -b <hf_uname>/traces
```

| Type | CPU stream |
| :--: | :--: |
| Figure 4: Naive masking | ![CPU lane of naive attention with out-of-place `masked_fill`, showing several dispatch ops for the masking step](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cpu-profile-naive.png) |
| Figure 5: In place masking | ![CPU lane of naive attention with in-place `masked_fill_`, showing fewer dispatch ops for the masking step](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cpu-profile-inplace.png) |

The in-place version (Figure 5) wraps far fewer CPU ops inside the masking step than the out-of-place version (Figure 4). This is an encouraging signal. Let's unfold the GPU lane to confirm what happened there.

| Type | GPU stream |
| :--: | :--: |
| Figure 6: Naive masking | ![GPU kernels for naive attention including a separate Memcpy kernel before the masking](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/each-kernels-naive.png) |
| Figure 7: In place masking | ![GPU kernels for naive attention with in-place masking, with the Memcpy kernel gone](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/each-kernels-inpace.png) |

On the GPU lane the `Memcpy` kernel is gone for good (Figures 6 and 7). With a one line change we shaved a whole kernel off each forward pass. This may not look like much on its own, but remember this is a single attention operation. In a Large Language Model (LLM) it repeats once per layer and head, and there are many layers and heads, so the saving adds up quickly (and if it earns you a raise, sharing at least 10% with us feels only fair).

> [!NOTE]
> Out-of-place is PyTorch's default for a reason. To compute gradients, autograd has to remember the tensor values it saw on the forward pass, because many backward formulas reuse them. An in-place operation overwrites those values in memory, so the backward pass would read the wrong numbers. Due to the fact that we run `forward` under `torch.no_grad`, in-place is safe for us, with no backward pass and nothing to corrupt.

## Scaled Dot Product Attention

We just built attention from primitives, and even shaved off a `Memcpy`. The good news is that the PyTorch team has done all of this for us, and packaged the whole pipeline into a single function:

```py
from torch.nn import functional as F

F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

This one line replaces our hand written module, and `is_causal=True` even saves us from building the mask by hand. It is worth pausing to appreciate how much this one call hides. And it hides more than just code lines. Scaled Dot Product Attention (SDPA) does not have a single implementation. Under the hood it _dispatches_ to one of the several backends and picks the fastest one that supports our inputs (dtype, head dimension, mask, hardware, etc.).

The [official SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) walks us through this selection, and the backends themselves are listed in the `torch.nn.attention.SDPBackend` enum:

```python
from torch.nn.attention import SDPBackend

BACKENDS = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}
```

Normally SDPA chooses for us, but we can pin a specific backend with the `torch.nn.attention.sdpa_kernel` context manager. This is what we do in our scripts. This lets us profile each backend on its own and read how differently they show up in the trace. Let's go one at a time.

### Math backend

```bash
uv run 04_c_sdpa_attention.py --backend math
uvx trace-util -f traces/ -b <hf_uname>/traces
```

Before we open anything, let's guess. We have replaced hand written attention (matmul, mul, mask, softmax, matmul) with a single one liner, so we should expect the trace to get _simpler and faster_. Fewer kernels, less CPU dispatch, maybe even a fused kernel. Let's check the profiler table first.

| Metric | Naive in-place | SDPA math |
| :--: | :--: | :--: |
| *_fwd CUDA time avg | 1.955 ms | 7.239 ms |
| Self CUDA time total | 7.194 ms | 27.279 ms |

This is our first surprise, the one liner is `3.7x` slower.

| | Profiler Trace |
| :--: | :--: |
| Figure 8: Profiler trace of naive in-place attention showing five GPU kernel launches for one forward | ![GPU lane of naive in-place attention with five kernel launches for one forward pass](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/inplace-kernel-launches.png) |
| Figure 9: Profiler trace of the SDPA math backend showing 20 GPU kernel launches for a single attention forward | ![GPU lane of the SDPA math backend with twenty kernel launches for a single attention forward pass](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/math-kernel-launches.png) |

Opening the trace (Figure 9) shows why the alarm bells ring, the math backend launches `20` GPU kernels per forward instead of the `5` launched with our naive attention implementation (Figure 8). This is the opposite of what we guessed. Let's figure out why this happens.

#### Tensor Cores left vacant

In [Part 2](https://huggingface.co/blog/torch-mlp-fusion) we learned to read a kernel name like a fingerprint. Let's use that habit here:

| Run | matmul kernel |
| :--: | :--: |
| Figure 10: Naive attention | ![Matmul kernel name for naive attention in Perfetto, carrying the s16816 bfloat16 Tensor-core GEMM signature](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cuda-core-kernels.png) |
| Figure 11: SDPA with math backend | ![Matmul kernel name for the SDPA math backend, carrying the sgemm FP32 CUDA-core signature](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/tensor-core-kernels.png) |

The `s16816` in the naive kernel (Figure 10) is the signature of a `bfloat16` Tensor core matmul (the `16x8x16` Tensor core instruction). `sgemm` (Figure 11) is the classic single precision (`FP32`) matmul that runs on the ordinary CUDA cores. So the math backend is not touching the Tensor cores at all.

How does this matter? A Streaming Multiprocessor (SM) is the compute unit of a GPU, and each SM has two kinds of arithmetic units, the CUDA cores and the Tensor cores. CUDA cores are general purpose and process a handful of elements at a time. Tensor Cores are specialised hardware that multiply and accumulate a whole small matrix tile in a single instruction. So when the math backend deliberately trades speed for numerical accuracy it upcasts tensors to `FP32` (doubling the data moved) and uses the CUDA cores instead of the faster Tensor cores.

#### Causal masks built

In the naive version we built the causal mask once and reused it. Here we passed `is_causal=True` and the math backend materialized one for us, on every single call. You can watch it happen on the CPU lane:

| ![CPU lane of the SDPA math backend showing the ops that rebuild the causal mask: aten::ones, aten::tril, aten::scalar_tensor, aten::fill_ and aten::where](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/mask-math.png) |
| :--: |
| Figure 12: CPU lane showing the ops for masking |

Here is what we see in Figure 12

```bash
aten::ones -> aten::tril            build a [seq, seq] lower-triangular matrix
aten::scalar_tensor -> aten::fill_  make the -inf fill value
aten::where                         turn it into an additive bias (0 or -inf)
```

On the GPU this shows up as a `triu_tril_kernel`, several `where` kernels, and an `add_`. The convenience flag that let us stop thinking about the mask did not remove the work, it just moved it one layer down, where the mask is rebuilt from scratch every forward.

#### The safe softmax

Our hand written version called plain `aten::softmax`. The math backend calls `aten::_safe_softmax`, and the difference is again visible as extra kernels (Figure 13):

| ![GPU lane of the SDPA math backend showing the extra kernels that aten::_safe_softmax launches compared to a plain softmax](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/safe-softmax-extra-kernels.png) |
| :--: |
| Figure 13: Safe softmax highlighting the extra kernels compared to generic softmax |

A row that is fully masked (every entry `-inf`) would make an ordinary softmax compute `exp(-inf)/sum(exp(-inf)) = 0/0 = NaN`. `_safe_softmax` guards against exactly that. Our naive kernel never bothered, and would have quietly produced `NaN`s in that corner case.

#### So what is the math backend for?

Put together, the math backend is the reference implementation. It is a straightforward, dtype-safe, NaN-safe decomposition of attention into primitive ATen ops. It is essentially the naive attention we wrote by hand, but more careful. That carefulness is exactly what makes it extremely slow.

Its job is not to be fast, but to _always_ work. This makes it the perfect baseline. Every backend we profile next (flash, efficient, cudnn) is trying to collapse the `20` GPU kernels into essentially one fused kernel that stays in bf16 and never materializes the intermediate matrices at all.

### Efficient backend

```bash
uv run 04_c_sdpa_attention.py --backend efficient
uvx trace-util -f traces -b <hf_uname>/traces
```

| ![Profiler trace of the SDPA efficient backend showing a single fused fmha_cutlassF attention kernel per forward](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/efficient-backend.png) |
| :--: |
| Figure 14: The profiler trace for sdpa with efficient backend |

Where the math backend launched 20 kernels across one profiler step, the efficient backend launches only one `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` (as seen in Figure 14).

Let's decode the name of the kernel:

- `fmha` (fused multi-head attention): All the primitive ops in attention is "fused" in one op now.
- `cutlassF`: built on CUTLASS (NVIDIA's open-source templates for tensor-core GEMMs), `F` for forward.
- `bf16_aligned`: runs in bfloat16 (no FP32 upcast, unlike math).
- `64x64`: the tile size.
- `rf` (register file): the working set is kept in registers, the fastest memory on the chip.
- `sm80`: compiled for Ampere (the A100's compute capability 8.0).

This is the memory efficient attention kernel that grew out of Meta's [xformers](https://github.com/facebookresearch/xformers) library and was upstreamed into PyTorch. When people say "the xformers backend," this `fmha_cutlassF` kernel is what they mean.

### Flash backend

```bash
uv run 04_c_sdpa_attention.py --backend flash
uvx trace-util -f traces -b <hf_uname>/traces
```

| ![Profiler trace of the SDPA flash backend](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/flash-backend.png) |
| :--: |
| Figure 15: The flash backend trace, one fused `pytorch_flash` kernel per forward |

The `void pytorch_flash` kernel (Figure 15) is [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Tri Dao's implementation), vendored into PyTorch.

Before we read the trace any further, it is worth answering the question you should be asking by now: _why is there a whole backend named "flash", and why does it matter so much?_

#### Why flash attention exists?

Let's go back to the math backend for a moment. Its real problem was not the count of 20 kernels, it was what those kernels handed to each other.

Step 1 builds the full score matrix `S = Q . K.T`, which is `[seq, seq]` **per head**. For a sequence length of 4096 that is roughly 16 million numbers for a single head. That matrix is written out to the HBM (the GPU's main memory), read back to be scaled, written again for the mask, read again for the softmax, and so on. Attention's cost is dominated by this **back and forth traffic to HBM**, not by the matmuls themselves.

FlashAttention attacks exactly this. Instead of computing the whole `S` matrix and only then reducing it, it walks over `K` and `V` in **tiles**, keeps a running softmax as it goes (the "online softmax" trick), and accumulates the output one tile at a time. The full `[seq, seq]` score matrix is **never written to HBM**, it only ever lives on-chip. This is the single idea that lets the entire attention pipeline collapse into one fused kernel that stays in bf16 on the Tensor cores.

#### Why flash looks "wrong" under the profiler

| ![Perfetto footprint of the flash kernel reporting an estimated achieved occupancy of 13%](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/flash-occupancy.png) |
| :--: |
| Figure 16: Estimated occupancy of flash kernel is seen to be 13% |

Here is where flash surprises people who read profiler footprints. It is the fastest backend, yet the profiler reports it with very **low occupancy** (shown in Figure 16). To see why that is fine, we need three quick definitions.

A CUDA kernel is launched as a **grid** of **blocks** (of threads). Each block contains many **threads**, and the hardware executes those threads in groups of 32 called **warps**.

Blocks are scheduled onto Streaming Multiprocessors (SMs), the main compute units of a GPU. A block lives entirely on one SM, and an SM can host multiple blocks at once _if it has enough resources_. Those resources include registers, shared memory, maximum resident threads, and maximum resident warps. So when we say a kernel has low **occupancy**, we mean each SM has fewer resident warps than it could theoretically support.

If you click the flash kernel in the trace, its footprint tells the story (Figure 17).

| ![Resource footprint of the pytorch_flash kernel in Perfetto, showing a high per-thread register count and large shared memory usage per block](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/flash-reg-count.png) |
| :--: |
| Figure 17: The flash kernel footprint, heavy on registers and shared memory per block. |

Flash uses a lot of per-thread registers and a large amount of shared memory per block. For example, if a block has 128 threads and each thread uses 255 registers, that block needs `128 × 255 = 32,640` registers. On an Ampere SM with 65,536 registers, only two such blocks fit at once. Each 128-thread block has `128 / 32 = 4` warps, so two blocks give only 8 resident warps. Against a maximum of 64 resident warps, that is roughly 13% occupancy. Flash has low occupancy not because it is poorly optimized, but because each block is deliberately very "heavy" in on-chip resource usage.

And that is the whole point. High occupancy helps _hide latency_ by keeping many warps ready to run, but it does not make the work itself efficient. Flash spends those registers and that shared memory on purpose, to keep attention tiles on-chip, reuse data aggressively, and avoid ever materializing the full attention matrix in global memory.

### cuDNN backend

```bash
uv run 04_c_sdpa_attention.py --backend cudnn
uvx trace-util -f traces -b <hf_uname>/traces
```

| ![Profiler trace of the SDPA cuDNN backend showing a single cudnn_generated attention kernel per forward](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cudnn-backend.png) |
| :--: |
| Figure 18: The cuDNN backend trace, a single generated attention kernel per forward. |

By now the pattern is familiar. Like flash and efficient, cuDNN gives us one fused, flash-style kernel per forward (Figure 18). So the natural question is: **if flash already fuses attention, why does PyTorch ship yet another flash backend?** The answer is _who writes the kernel and how it is built_, and that difference is what makes the trace look different.

#### How is cuDNN kernel different

Flash and efficient are **fixed, pre-compiled kernels** vendored into PyTorch. You get the same binary every time. cuDNN is NVIDIA's own deep learning library, and its attention kernel is **generated and tuned for the specific problem** at hand. It is closer in spirit to `torch.compile`'s codegen than to a fixed cuBLAS binary. You can read that straight off the (very long) kernel name:

```bash
cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_6_128x64x64_4x1x1_cga1x1x1_kernel0_0
```

- `cudnn_generated`: not a pre-shipped binary, it was generated by cuDNN.
- `flash_fprop`: a flash attention style forward pass. So the algorithm is the same family as the flash backend.
- `wmma_f16`: it uses the warp-level matrix multiply-accumulate (WMMA) API, the Tensor-core path on the 16-bit float pipeline.
- `knob_6`: cuDNN picks from a set of pre-tuned configurations ("knobs"). Different shapes select different knobs, much like cuBLAS picking a tile variant.
- `128x64x64`: the tile dimensions it chose.

That one fact, _generated per problem_, explains everything else that looks unusual in the trace.

1. No transposes: The CPU lane goes from `_cudnn_attention_forward` straight to a couple of `aten::empty` allocations and then the kernel, with zero `aten::transpose` (Figures 19, 20 and 21). Flash and efficient each insert four (metadata) transposes to reshape the tensors while cuDNN consumes the native `[B, H, S, D]` layout directly because its generator emits a kernel for that layout.

   | Variant | Trace |
   | :--: | :--: |
   | Figure 19: Flash | ![CPU lane of the flash backend showing four aten::transpose ops before the fused attention kernel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/flash-transpose.png) |
   | Figure 20: Efficient | ![CPU lane of the efficient backend showing four aten::transpose ops before the fused attention kernel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/efficient-trasnpose.png) |
   | Figure 21: cuDNN | ![CPU lane of the cuDNN backend going straight to aten::empty allocations and the kernel, with no transpose ops](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cudnn-backend.png) |

2. It launches through `cuLaunchKernelEx`, not `cudaLaunchKernel`: Every other kernel in this whole series went through the runtime API `cudaLaunchKernel`. cuDNN uses the driver-level _extended_ launch, which carries launch attributes (Figure 22).

   | ![CPU lane of the cuDNN backend showing the cuLaunchKernelEx driver-level launch instead of cudaLaunchKernel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cudnn-launch.png) |
   | :--: |
   | Figure 22: CPU lane of the cuDNN backend showing the cuLaunchKernelEx driver-level launch instead of cudaLaunchKernel |

3. The profiler reports 0% achieved occupancy: Do not take that at face value, it is a measurement gap, not a stalled GPU. CUPTI (the profiling backend) cannot attribute occupancy to a driver-API (`cuLaunchKernelEx`) launch the way it does for `cudaLaunchKernel`, so the field reads 0. The footprint fills in the truth (Figure 23): `240 registers × 256 threads = 61,440` registers per block against the SM's 65,536, so only **one block** fits per SM (8 warps ≈ 12.5%), right in line with flash.

   | ![Perfetto footprint of the cuDNN kernel reporting 0% achieved occupancy, with 240 registers per thread and 256 threads per block](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cudnn-footprint.png) |
   | :--: |
   | Figure 23: cuDNN kernel reporting 0% achieved occupancy, with 240 registers per thread and 256 threads per block |

#### The cost moved to the CPU

The "no transposes" story tempts us to expect cuDNN to be the _leanest_ backend on the CPU. It is the opposite.

| backend | CUDA avg time | CPU avg time |
| :--: | :--: | :--: |
| efficient | 277.9 µs | 117 µs |
| flash | 146.8 µs | 138 µs |
| cudnn | 186.3 µs | **214 µs** |

Even with zero transpose ops, cuDNN spends about **214 µs per forward on the CPU**, more than flash (138) or efficient (117). Almost all of it sits in `aten::scaled_dot_product_attention` self time (26% of the whole run) and `_cudnn_attention_forward`. That is cuDNN's runtime engine selecting and preparing the plan (the "knob" search) on every call.

Fewer visible ATen ops did not mean less CPU work, it **moved the work into the library**, where the profiler can only show it as one fat, opaque bar. When a trace suddenly gets _cleaner_, the work has not always disappeared, sometimes it has just moved somewhere the profiler cannot break down.

On the GPU, cuDNN (186.3 µs) lands between efficient and flash. On this very flash-friendly shape, hand-written FlashAttention-2 edges it out. cuDNN often wins on _other_ shapes (larger head dimensions, different sequence lengths) precisely because its generator retunes per problem, but that retuning is also what you just paid for on the CPU.

## Everything we covered, at a glance

Before we wrap up, here is a single table to review every attention variant we profiled and the one lesson each trace taught us.

| Variant | What we changed | Kernels / forward | What the trace revealed |
| :-- | :-- | :--: | :-- |
| Naive attention | Attention built by hand from primitives (matmul, mul, mask, softmax, matmul) | 6 | A hidden `Memcpy` from the out-of-place `masked_fill`. |
| Naive in-place | `masked_fill` → `masked_fill_` | 5 | One line drops the `Memcpy` kernel entirely. |
| SDPA math | `F.scaled_dot_product_attention` pinned to the math backend | 20 | The reference: FP32 on CUDA cores, mask rebuilt every call, `_safe_softmax`. Correct but ~3.7x slower. |
| SDPA efficient | Efficient (xformers) backend | 1 | One fused `fmha_cutlassF` kernel, stays in bf16 on Tensor cores. |
| SDPA flash | Flash backend | 1 | One fused `pytorch_flash` kernel (FlashAttention-2). Fastest, despite "wrong-looking" 13% occupancy. |
| SDPA cuDNN | cuDNN backend | 1 | A per-problem generated kernel: no transposes, `cuLaunchKernelEx`, but the cost moved to a fat CPU bar. |

## Concluding the series

If you take away only one thing from the whole series, let it be the habit we repeated before every single trace which is to **guess first, then look.**

State out loud what you expect the trace to contain, open it, and treat any mismatch as the most interesting thing on the screen. Every real insight in these three posts, the hidden `Memcpy`, the `addmm` epilogue, the 20 kernel math backend, flash's "wrong-looking" occupancy, cuDNN's fat CPU bar, came from a guess that did not match the trace.

Profiling is not a separate, intimidating skill reserved for GPU experts. It is just the discipline of looking closely and asking "wait, why is _that_ happening?" until the answer clicks. You now have the vocabulary and the reflexes to do that on your own models. Open a trace, form a guess, and go find the mismatch.

Thanks for reading the **Profiling in PyTorch** series. Now go profile something. 🤗

Thanks to [Noe Flandre](https://huggingface.co/NoeFlandre) for their reviews on the early draft of the post!

> [!NOTE]
> The blog post was polished using an LLM. This in no way means that we have let an agent run in the background and let it generate the blog. Some of us in the team are non-english speakers and think LLMs (which are mostly trained in the English Language) can rectify silly grammar mistakes or rephrase sentences that sound less intimidating and cleaner. Hope this helps with the idea of "why should I read, if this was LLM generated". 🤗