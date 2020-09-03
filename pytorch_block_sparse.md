---
title: Block Sparse Matrices for Smaller and Faster Language Models
thumbnail: https://huggingface.co/blog/assets/01_how-to-train/how-to-train_blogpost.png
---

<h1 class="no-top-margin">Block Sparse Matrices for Smaller and Faster Language Models</h1>

<div class="blog-metadata">
    <small>Published Sep 2, 2020.</small>
    <a target="_blank" class="btn-readme" href="https://github.com/huggingface/blog/blob/master/pytorch_block_sparse.md">
        <img src="/front/assets/icon-github.svg">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="https://twitter.com/madlag">
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
we introduced what are sparse matrices and what they could do to improve neural networks.

The basic assumption is that full dense layers are often overkill and can be pruned without losing precision or almost.
In some cases it can even *improve precision or/and generalization*.

The main issue is that the code to support sparse algebra computation is usually severely lacking in term of speed, 
and that we are [still waiting](https://openai.com/blog/openai-pytorch/) for efficient PyTorch support.

That's why we ran out of patience, we took some time this summer to address this "lacuna",
and we are releasing today the extension [pytorch_block_sparse](https://github.com/huggingface/pytorch_block_sparse).

## Usage
The provided BlockSparseLinear module is a drop in replacement for torch.nn.Linear, and it's really trivial to use 
it in your own models:

```python
# from torch.nn import Linear
from pytorch_block_sparse import BlockSparseLinear

...

# self.fc = nn.Linear(1024, 256)
self.fc = BlockSparseLinear(1024, 256, density=0.1)
```

The extension provides too a BlockSparseModelPatcher that allows to modify "on the fly" an existing model, 
with an [example notebook](https://github.com/huggingface/pytorch_block_sparse/blob/master/doc/notebooks/ModelSparsification.ipynb).
You can then train the model as usual.


## NVIDIA CUTLASS
This extension is based on the [cutlass tilesparse](https://github.com/YulhwaKim/cutlass_tilesparse) proof of concept by [Yulhwa Kim](https://github.com/YulhwaKim).

It is using C++ CUDA templates for block-sparse matrix multiplication
based on [CUTLASS](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/).

CUTLASS is a collection of CUDA C++ templates for implementing high-performance CUDA kernels.
With CUTLASS, approching cuBLAS performance on custom kernels is possible without resorting to assembly language.

## Performance
In the present stage of the library, the performances for sparse matrices are roughly a factor of 2 slower
than their cuBLAS optimized dense counterpar, and we are confident we can improve this in the future.

This is a huge improvement on PyTorch sparse matrices: their current implementation is an order of magnitude slower
than the dense one.

But the more important point is that the performance gain of using sparse matrices grows with the sparsity,
so a **75% sparse matrix** is roughly **2x** faster than the dense equivalent.

The memory savings are even more significant: for **75% sparsity**, you will get reduce memory consumption by **4x**
as you would expect. 

## Future work
Being able to train quickly block-sparse linear layers was just the first step.
The sparsity pattern is currenly fixed at initialization, and of course optimizing it during learning will yield large
improvements.

So you can expect in future versions some tools to measure the "usefulness" of parameters to be able to optimize the sparsity pattern.
**NVIDIA Ampere 50% sparse pattern** within blocks will probably yield another level of performance gain, just like upgrading
to more recent versions of CUTLASS.

So, stay tuned for more sparsity goodness in a near future!
