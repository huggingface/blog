---
title: Block Sparse Matrices for Smaller and Faster Language Models
thumbnail: https://huggingface.co/blog/assets/04_pytorch_block_sparse/thumbnail.png
---

<h1 class="no-top-margin">Block Sparse Matrices for Smaller and Faster Language Models</h1>

<div class="blog-metadata">
    <small>Published Sep 10, 2020.</small>
    <a target="_blank" class="btn-readme" href="https://github.com/huggingface/blog/blob/master/pytorch_block_sparse.md">
        <img src="/front/assets/icon-github.svg">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/madlag">
        <img class="avatar avatar-user" src="https://www.gravatar.com/avatar/205c3e49902572f215d99796656526c7?d=retro&size=200" title="Gravatar">
        <div class="bfc">
            <code>madlag</code>
            <span class="fullname">François Lagunas</span>
        </div>
    </a>
</div>

## Saving space and time, one zero at a time

In previous [blog](https://medium.com/huggingface/is-the-future-of-neural-networks-sparse-an-introduction-1-n-d03923ecbd70)
[posts](https://medium.com/huggingface/sparse-neural-networks-2-n-gpu-performance-b8bc9ce950fc) 
we introduced sparse matrices and what they could do to improve neural networks.

The basic assumption is that full dense layers are often overkill and can be pruned without a significant loss in precision.
In some cases sparse linear layers can even *improve precision or/and generalization*.

The main issue is that currently available code that supports sparse algebra computation is severely lacking efficiency.
We are also [still waiting](https://openai.com/blog/openai-pytorch/) for official PyTorch support.

That's why we ran out of patience and took some time this summer to address this "lacuna".
Today, we are excited to **release the extension [pytorch_block_sparse](https://github.com/huggingface/pytorch_block_sparse)**.

By itself, or even better combined with other methods like
[distillation](https://medium.com/huggingface/distilbert-8cf3380435b5)
and [quantization](https://medium.com/microsoftazure/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7),
this library enables **networks** which are both **smaller and faster**,
something Hugging Face considers crucial to let anybody use
neural networks in production at **low cost**, and to **improve the experience** for the end user.

## Usage
The provided `BlockSparseLinear` module is a drop in replacement for `torch.nn.Linear`, and it is trivial to use 
it in your models:

```python
# from torch.nn import Linear
from pytorch_block_sparse import BlockSparseLinear

...

# self.fc = nn.Linear(1024, 256)
self.fc = BlockSparseLinear(1024, 256, density=0.1)
```

The extension also provides a `BlockSparseModelPatcher` that allows to modify an existing model "on the fly",
which is shown in this [example notebook](https://github.com/huggingface/pytorch_block_sparse/blob/master/doc/notebooks/ModelSparsification.ipynb).
Such a model can then be trained as usual, without any change in your model source code.


## NVIDIA CUTLASS
This extension is based on the [cutlass tilesparse](https://github.com/YulhwaKim/cutlass_tilesparse) proof of concept by [Yulhwa Kim](https://github.com/YulhwaKim).

It is using **C++ CUDA templates** for block-sparse matrix multiplication
based on **[CUTLASS](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)**.

CUTLASS is a collection of CUDA C++ templates for implementing high-performance CUDA kernels.
With CUTLASS, approching cuBLAS performance on custom kernels is possible without resorting to assembly language code.

The latest versions include all the **Ampere Tensor Core primitives**, providing **x10 or more speedups** with a limited loss of precision.
Next versions of pytorch_block_sparse will make use of these primitives,
as block sparsity is 100% compatible with Tensor Cores requirements.

## Performance
At the current stage of the library, the performances for sparse matrices are roughly
two times slower than their cuBLAS optimized dense counterpart, and we are confident
that we can improve this in the future.

This is a huge improvement on PyTorch sparse matrices: their current implementation is an order of magnitude slower
than the dense one.

But the more important point is that the performance gain of using sparse matrices grows with the sparsity,
so a **75% sparse matrix** is roughly **2x** faster than the dense equivalent.

The memory savings are even more significant: for **75% sparsity**, memory consumption is reduced by **4x**
as you would expect. 

## Future work
Being able to efficiently train block-sparse linear layers was just the first step.
The sparsity pattern is currenly fixed at initialization, and of course optimizing it during learning will yield large
improvements.

So in future versions, you can expect tools to measure the "usefulness" of parameters to be able to **optimize the sparsity pattern**.
**NVIDIA Ampere 50% sparse pattern** within blocks will probably yield another significant performance gain, just as upgrading
to more recent versions of CUTLASS does.

So, stay tuned for more sparsity goodness in a near future!
