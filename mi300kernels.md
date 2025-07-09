---
title: "Creating custom kernels for the AMD MI300" 
thumbnail: /blog/assets/mi300kernels/thumbnail.png
authors:
- user: ror
- user: seungrokj
  guest: true
  org: amd
---

# Creating custom kernels for the AMD MI300

# AMD Kernels

![Titel card](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/title.png)

# Introduction

More than a billion per day: that’s a low estimate of how many requests ChatGPT handles daily, a number which is unlikely to go down soon. For each request and each generated token, we run an inference of a multi-billion parameters model. This is why model optimization is paramount at each and every level: when one deals with these kinds of scale, even a 1% latency or power gain can bring huge savings. 

But where might that gain come from? Model architectures are already well established, and popular models have had quantized weight for a long time now. However, a crucial level at which we can optimize model inference remains: the kernel level. Kernels are the algorithms executed when you do any operation in your network: there are matrix multiplication kernels, convolution kernels, batch normalization kernels, etc. Kernels are low-level, highly-optimized algorithms, often tailored for the device they will be running on. They are notoriously long and hard to write, and require a good understanding of the inner working of the GPU. 

Kernels are essential for running operations in neural networks—without a kernel, an operation effectively can't be used. Because of this, new innovations often launch with a "day 0" kernel, typically optimized only for the latest Nvidia hardware. This approach excludes many other devices, particularly AMD GPUs, which, despite offering comparable or superior specs, are often overlooked by kernel developers. Hugging Face collaborated with AMD to deliver state-of-the-art performance on AMD platforms and make it benefit the open source community. As part of this partnership, we decided with AMD to focus on delivering open-source optimized kernels to improve the performance of serving Llama 3.1 405B in FP8 on a node of 8 MI300X using VLLM. 

In this blog post, we'll explore how we optimized performance for the MI300X and how each kernel was individually fine-tuned. But first, let’s look at the performance gains achieved using our custom kernels. By combining the following three optimized kernels:

- Fused residual connection, RMS norm and FP8 conversion kernel
- Fused SwiGLU activation and FP8 conversion kernel
- Skinny GEMM kernel

we achieved significant speedups when running VLLM on a node powered by MI300X GPUs.

![Latency gains](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/results_figure.png)

Measures were taken with input size 1 and output size 128 to mimic decoding regime. We measure decoding latency using the median over 30 iterations. 

Those performance gains were measured in VLLM, but you may also use the kernels separately, as described in the “How to” section that follows.

## How to use these kernels

### The `hf-rocm-kernels` repo

All kernels described previously are available on the `hf-rocm-kernels` repository located [**here**](https://github.com/huggingface/hf-rocm-kernels). 
In it, you will find instructions on how to install the package, the source code for each kernels, their respective python bindings, various benchmarking scripts and a test suite. Using benchmarking scripts and a MI300X, you may even reproduce from this blog post. To ensure same results for Torch or VLLM, you can use the same [container](https://hub.docker.com/layers/rocm/vllm/rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6/images/sha256-9a12ef62bbbeb5a4c30a01f702c8e025061f575aa129f291a49fbd02d6b4d6c9) as we did.
You can also use the repo as a base to build your own kernels: it has instructions on how to bind a CUDA-style kernel to python and a simple sample kernel. 
You may even have a look at branches under development for new kernels, like a compute-and-communicate kernel as described [here](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487).  

### Integration in VLLM

The kernels described will soon be integrated in the AMD fork of the VLLM project, but if you want to have a look at how you might do something like that yourself, you may check out this [branch](https://github.com/remi-or/vllm/tree/patch_hfrk) and this [document](https://github.com/remi-or/vllm/blob/patch_hfrk/HFRK_readme.md).

# Optimization process

We are first going to do a quick refresher on the architecture of the device we are working on: the MI300X. Then, we are going to take a look at the state of our model’s inference before optimizing it. This will allow us to identify bottlenecks and know which custom kernels we need to write. Then, we will take a look at each kernel we have written, which will give us an opportunity to explore how kernel optimization is conducted through many angles.

## A quick introduction to the MI300X

Before we dive into optimizing GPU code, we need to know how a GPU works. There are a lot of resources out there that already do a great job of explaining the inner workings of your GPU, which I will link right [here](https://tinkerd.net/blog/machine-learning/cuda-basics/), [here](https://siboehm.com/articles/22/CUDA-MMM) and [here](https://www.youtube.com/watch?v=OUIkkAPaw4M). We are still going to run through the different levels of the GPU, as a quick refresher.
If you want to skip the refresher and get directly into the details of our custom kernels, click here!

### Threads

The smallest unit of work in the GPU is the **thread**. Any time any work is done on a GPU, it’s because a thread executed an instruction. Instructions are basic operations like additions, multiplication, conversion from one data type to another, or loads and stores. Each thread has its own memory, called registers (or VGPRs), which only it can access. A thread can have a maximum of 256 registers, each 32-bit wide. Below is represented a thread with access to its 256 VGPRs.

![Representation of a thread](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/thread_global.png)

Threads, except when using load or store instructions, can only execute instructions on their own registers. For instance, to add two vectors A and B together, each thread is going to 1) load in its registers an element from A and 2) another from B, then 3) perform the addition and store the result in another register, and finally 4) store the value from that register in memory. That’s a total of 4 instructions.

### Warps

The next unit of work is a warp: each warp is composed of 64 threads. Warps don’t have their own memory, but they are of interest to us because all threads in a warp must execute the same instruction at the same time. This is both a guarantee and a constraint.

![Representation of a warp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/warp_global.png)

Warps also allow for different threads to exchange information coming from their registers with other threads in the same warp. Although different threads in a warp have access to different data, the fact that they all have to execute the same instructions means that when writing a kernel, warp-level behavior is what you need to think about.

### Compute units

Warps are bundled together into **thread blocks**: thread blocks are a software abstractions, but run on a hardware component called a **compute unit (CU)**. A single compute unit can run multiple thread blocks at once, but it can only fit 16 warps. 
Each compute unit has a dedicated L1 cache and shared memory. L1 cache cannot be controlled or allocated and helps with data reuse of all warps situated on the CU. Conversely, shared memory can be allocated and used as a storage shared by all warps. For instance, when we want all warps (and thus threads) in a compute unit to access the same buffer, we allocate it in shared memory. Both shared memory and L1 cache are fast to access because they are “close” to the threads.

![Representation of a compute unit](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/cu_global.png)

Thread blocks also offer the ability to synchronize all threads running inside: this is quite useful when dealing with operations that impact shared memory, like initializing an array in shared memory to zero or reduction operations. In general, when writing a kernel, thread blocks are the highest level to take into consideration: it’s very hard to synchronize different thread blocks or make them interact in any way whatsoever. 
Kernel throughput is tightly linked to the number of compute unit present on the GPU: the more CUs there are, the more thread blocks can be run at the same time, which increases throughput if you manage to use all CUs. 

### XCDs

Compute units are then grouped into **accelerator complex dies (XCDs),** which hold 38 compute units each. Although CUs may not interact with each others, they all share a L2 cache which you can’t control but still may prove useful when re-using data. For instance, when accessing memory, having two compute units located on the same XCD access the same data will reduce loading latency by a lot. L2 cache is quite large: it has a size of 4MB, while shared memory has a size of 64kB and L1 cache contains 32kB. 

![Representation of a XCD](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/xcd.png)

### The entire GPU (MI300X)

By assembling 8 XCDs (which gives us 8 * 38 = 304 CUs) and adding a last level of cache (called infinity cache, with 256MB) and a huge quantity of video ram (192GB) we get the MI300X.

![Representation of a MI300](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/mi300x.png)

All XCDs, and thus all threads, have access to the VRAM, but getting there is quite slow. As you get further away from thread-level, memory becomes slower to access but has a larger size and larger scope, meaning it serves more threads. When optimizing a kernel, there is always a balance to strike between doing lots of operations or loading lots of data, but in general, you want to access the VRAM (commonly referred to as global memory) as little as possible. 

When looking at this figure, we can see why GPUs are referred to as “massively parallel”: here, we have 304 compute units, which can each run 16 warps, each with 64 threads. This means that  we can have up to 311296 threads running at the same time, each executing an instruction of its own.
Keep in mind an instruction is something basic like an addition, so simple routines like Newton’s method can be quite long to run for a single thread. GPUs are not optimized for instructions to run fast, i.e. for the latency of each instruction to be low: that would be a latency-oriented device. They are optimized for many threads to be run together, consuming and outputting a large quantity of data: it is a throughput-oriented device. 
When optimizing a kernel for the GPU, we adapt in consequence: it is better to have an algorithm running a few instructions on many threads at once, than having it run many instructions on a few threads. Hence calling algorithms running on GPUs “parallel”. 

What can get in the way of such algorithms running in an optimized manner are three things: when there is a lot of data to load (memory bound), when there are many operations to performs (compute bound) or when threads have to work together (synchronization overhead). 

## Day 0 performance analysis

When optimizing a workload, the first thing to do before writing a single line of code is to profile the current state of the workload. 
In our case, we are going to profile the model inference in VLLM to get an idea of how much time each operation is taking up. This can help identify major bottlenecks and which kernels we can tackle first for maximum speedup. For instance, here is the break down for batch size 32:

![Disk plot ok kernels latency](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/disk_figure.png)

We can see the different parts of the network through each slice: 

- the “Attention*” slice, where we grouped RoPE, attention and KV cache kernels;
- the “Attention GEMMs”, that encompass two projections, QKV and Output;
- the “Communications”, which is made up of two all-reduce operations, one after the Attention block and one after the MLP block, which are there because we are working in tensor parallel (TP8)
- the “MLP GEMMs”, that encompass the two projections made in the MLP, Gate / Up and Down;
- the “RMS norm” and “SwiGLU” slices, one for each kernel — note that the RMS norm kernel is called twice per block, once before the Attention and once before the MLP;
- the “Other” slice that regroups the kernels that we did not tag as part of a larger category because their impact is minor.

Already we can see that most of the latency comes from GEMMs and communications, but also that attention and the operations surrounding it are not a major contributor to latency. This can come as a surprise, because a lot of papers focus on attention and reducing its cost, but it seems that through a combination of KV caching and FlashAttention, which has already been optimized in VLLM, this part may no longer be a top priority.
Surprisingly, the two calls made to the “RMS norm” kernel are quite costly, so there might be a large benefit to optimizing that kernel. Along with the SwiGLU kernel, they represent 15% of the total latency, which is not negligible. All in all, working on those two kernels, plus trying to gain a small speedup on GEMMs may be our best course of action. To check that this performance breakdown is not a fluke, we can take a look at other batch sizes: 

![Latency distribution over batch sizes](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/bs_latency_figure.png)

We can see the pattern that emerged for batch size 32 holds up for other batch sizes, albeit with the latency contribution of GEMMs and communications becoming greater as the batch size increases. Also, it seems that batch size 32 is an outlier when it comes to the latency of GEMMs: it’s probably because the GEMMs chosen when batch size is 32 have been manually tuned or because batch size 32 presents good memory alignment patterns, so GEMMs for batch size 32 are faster than for batch size 24 or 28. 

Now that we have identified some hot spots to optimize, let’s take a look at the first kernel we wrote: the RMS norm kernel. 

---

## RMS norm kernel

In each decoder block, we have two main parts: an attention block and an MLP block. Both begin with a residual connection between two inputs: the current hidden states $x$ and the residual $r$. Both have the same shape, which is $n$ rows (as many as there are tokens) and $d$ columns. After they are added together, we apply a row-wise Root Mean Square (RMS) norm to $x$ and, since the model is in FP8, we quantize $x$ to FP8 using a scale $s$. Simply fusing those three operations into a single kernel can deliver a nice performance boost. Mathematically, the operations we have to perform are the following:

$$
\begin{align}
  \phantom{i + j + k}
  &\begin{aligned}
    x &\leftarrow x + r\\
    r &\leftarrow x
  \end{aligned}\\
  &\begin{aligned}
    V &= \sum_{i=1}^{d} x_i^2  \end{aligned}\\
  &\begin{aligned}
x &\leftarrow \frac{x}{\sqrt{V + \epsilon}}  \\
x_Q &= Q_{\text{fp8}} \left( s * x * w\right) 
  \end{aligned}
\end{align}
$$

where $w$ is a $d$-sized weight vector.
Steps $(1)$ and $(3)$ are pretty basic. For step $(1)$, we just need to position each thread to a different location in the tensor, load some elements of $x$ and $r$, add them and store back $r$. For step $(3)$, each thread performs some scalar operations (addition, square root, division) and a conversion to FP8. All of this, each thread can do on its own: this is perfectly suited to the parallel nature of the GPU. The step to watch out for is $(2)$: we need to sum over $d$, which means either each thread is going to visit each of the $d$ columns, or we need to exchange data between threads. The greater $d$ is, the more data we would have to load for the first option, so the less viable it becomes. We are going to pick the second option: synchronize threads at the block level, and they will exchange data using the shared memory. Each thread is going to accumulate a part of $V$ on its own and then we are going to sum all of those parts across the thread block, which is what we call a reduction. Since $V$ is computed across an entire row, we are going to assign a thread block for each row. 

When compared to out-of-the-box pytorch, the bare bones version of this kernel brings about a 10x speedup. But this is not enough: there are still many optimizations we can add on top of this.

### Optimization: memory-related

In terms of latency, one of the most costly operation is accessing VRAM, also called **global memory**. Luckily, there are some easy-to-follow principles that can dramatically reduce the cost of loading data. 

First, we can take a look at how much data a single thread can load in a single instruction: using the MI300X instruction guide, we see that the largest load we can make from global memory is 128 bits wide. Since we are loading FP16 data, we are going to load 128b / 16b = 8 elements per load.

Secondly, we make sure memory accesses are coalesced. Since each thread is part of a warp, when one thread reaches a “load” instruction, all other threads in the warp do too. For efficiency’s sake, these “load” instructions are then bundled together across the warp. The warp then collectively fetches the data needed and each thread gets the data it requires. Maximum efficiency is reached when the warp fetches a single chunk of data without any gap in it: this is what we call **contiguous** data. An issue arises when we need to load more data that can be loaded in one “load” instruction, and is illustrated below.

![Two loading scenarios](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/coalesced.png)

In this hypothetical scenario, we have two threads in the same warp that need to load 16 bytes, without constraint on which thread loads which element. This is a typical “reduction” situation.
Since a thread can only read 4 bytes per instruction, we have at least two ways of reading the data, represented in scenario (a) and (b). To decide which scenario is best, we need to look at this from warp perspective, not thread perspective. 
In scenario (a), the first load fetches elements 0,1,2,3,8,9,10,11 : we see that the data is not contiguous, because there is a gap between elements 3 and 8. While in scenario (b), the first load fetches elements 0,1,2,3,4,5,6,7 : we load contiguous data. Same goes for the second load. Thus **scenario (b) is better**. Although in scenario (a) we end up with 8 contiguous elements per thread, this does not matter: what matters is whether or not the **warp** loads contiguous data. This matters because if the warp can only load 8 contiguous bytes in one cycle, then each load of scenario (a) is processed in two cycles, while in scenario (b), each load only needs the one cycle. 

Third, we reduce the number of stores: when we look at steps $(1)$ and $(3)$ we can see that there are only two stores needed: one for $r$ and one for $x_Q$ . After step $(1)$ we can already store $r$ and be done with that. But we still need to access the modified version of $x$ after step $(2)$  is done. To do that, we can store the modified version of $x$ in global memory and reload it after step $(2)$  is done and rely on cache hits when reloading it. Or, if $x$ is small enough, we can store its modified version in shared memory: if $x$ is in FP16 and we only have one thread block per CU, then we can store 64KB / 2B = 32 * 1024 elements in shared memory per thread block. In the case of Llama 405B, $d$ is equal to 16384, so that fits. Using shared memory provides a nice speedup over relying on cache hits, especially when many thread blocks are active at once: if the L1 cache is not big enough to fit the whole of $x$, then we have to rely on L2 cache, which is shared by 38 CUs.  

Apart from memory access, we can also optimize computational efficiency, but we are going to leave that for the next kernel, as they will be similar in both cases. 

### Results

When we apply the optimizations discussed above, we get the following results:

![Latency of RMS norm kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/rms_latency_figure.png)

| Number of rows | Torch (μs) | VLLM (μs) | Ours (μs) |
| --- | --- | --- | --- |
| 1 | 38.8998 | 5.5145 | **4.18138** |
| 2 | 43.2469 | 5.65645 | **4.36976** |
| 4 | 41.1304 | 5.6893 | **4.37628** |
| 8 | 43.8883 | 5.72275 | **4.39081** |
| 16 | 46.8876 | 5.85667 | **4.48165** |
| 32 | 55.2276 | 6.08502 | **4.72017** |
| 64 | 75.6086 | 6.4629 | **5.54214** |
| 128 | 98.1122 | 7.49166 | **6.27341** |
| 256 | 119.727 | 11.8812 | **10.739** |
| 512 | 195.782 | 23.1595 | **18.5549** |
| 1024 | 355.42 | 44.8143 | **34.7204** |
| 2048 | 671.513 | 81.2089 | **73.35** |

with a [X, 16384] shaped FP16 input tensor. 
The most basic version of our kernel, referred to as “Pointwise”, has no memory-related optimization and already shows at least a x4 speedup over torch. It is less optimal than VLLM’s implementation of the kernel, but our “Vectorized” implementation beats both “Pointwise” and VLLM. This is the version of the kernel that implements coalesced 128 bits loads, which is only surpassed by the “Vectorized + SMEM” (SMEM stands for shared memory) implementation, that offers a notably better speedup ratio than VLLM for both low and high batch sizes.

---

## SwiGLU kernel

In the MLP block, after the kernel we have just written about, comes a projection which we have referred up to this point as “Gate / Up” projection. The reason we call it that way is because the “Gate / Up” projection is actually a concatenation of two projections with the same input: “Gate” and “Up”. Thus, we will write the result $x$ of the “Gate / Up” projection as $x = x_G | x_U$ where $|$ is the concatenation operator applied along the column axis. $x_G$ and $x_U$ have the same dimensions. The reason we need those two projections is the SwiGLU activation function that comes right after, which results $y$ is defined by equation $(4)$ .
The SwiGLU activation function is followed by the “Down” projection, which in our case is in FP8, so we also need to quantize $y$ as shown in equation $(5)$ :

$$
\begin{align}
  \phantom{i + j + k}&
\begin{aligned}
    y = \sigma \left( x_G \right) \cdot x_U \\\end{aligned}\\
  &\begin{aligned}
    y_Q = Q_\text{FP8} \left( s * y \right)

  \end{aligned}
\end{align}
$$

where $\sigma$ is the sigmoid function: $\sigma (x) = e^{-x} / (1 + x)$ . We are going to write a fused kernel that takes care of all of this. For this kernel, optimizations described for the RMS kernel are still relevant with the expection of the shared memory buffer. We will focus here on computation-related optimizations.

### Optimization: compute-related

There are two ways we are going to increase the speed of our kernels: increase the volume of work done for each instruction executed and use faster instructions. 

To increase the amount of work done per instruction, we can use **packed** instructions. Packed instruction are useful when we want to apply the same operator on several elements: rather than executing one instruction per element, we execute one instruction over a vector of element. In a CPU, packed (or vectorized) instructions are the bread-and-butter of single-threaded optimization, as the AVX family of instruction can attest to. There are few packed instructions on GPU, but they can be quite useful in the right place.
On the MI300X there is, among others, packed instruction for FP16 addition and multiplication, which we will use for both steps. There also exists packed conversion from FP32 to FP8, which can provide a nice boost in performance when compared to non-packed conversion. As a matter of fact, there is no conversion from any other data type than FP32 to FP8, so for the RMS norm kernel and this one, we have to go to FP32 precision in order to convert to FP8. 

However this is not an issue in this kernel: the sigmoid function $\sigma$ require us to compute an exponent, which is an operation that greatly benefits from FP32 precision.  And this is in an instance where we can optimize computation by using a faster instruction: instead of using the `exp` instruction, we scale the input by $\text{log}(2)$ and use the `exp2` instruction, which is much faster. We suffer an almost negligible loss in precision but also reduce latency.

### Results

We get the following table for a [X, 16384] shaped FP16 input tensor:

| Number of rows | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Torch (μs) | 40.2731 | 29.923 | 35.305 | 23.5763 | 22.4738 | 25.3445 | 31.5829 | 40.3194 | 53.5369 | 79.8037 | 124.873 | 243.202 |
| VLLM (μs) | 3.84116 | 3.86192 | 3.92937 | 3.94151 | 4.01047 | 4.02421 | 4.08943 | 4.20317 | 4.48755 | 7.48465 | 13.7389 | 25.4306 |
| Ours (μs) | **1.92981** | **1.93904** | **1.93524** | **1.99316** | **2.00415** | **1.91563** | **2.04498** | **2.61763** | **3.57726** | **5.47608** | **10.0482** | **19.8957** |
| Speedup (VLLM / Ours) | 1.990434291 | 1.991665979 | 2.030430334 | 1.977518112 | 2.001082753 | 2.100724044 | 1.999740829 | 1.605715857 | 1.254465708 | 1.366789747 | 1.367299616 | 1.278195791 |

With memory and compute optimizations tailored for the MI300X, we get a kernel that is more than 14 times faster than Torch on average and from 27% to 100% faster than VLLM’s kernel. 

---

## Skinny GEMM kernel

As we have seen earlier, about 60% of the model’s inference latency comes from projections, which rely on GEMM kernels. GEMM kernels are heavily optimized in dedicated libraries such as hipBLASLT rocBLAS on AMD, so writing a custom kernel that performs better in all cases is quite hard. But if we focus on some edge cases that are relevant to us, and write a GEMM kernel for those specific cases, then there is a chance our custom kernel may be faster than the ones in the dedicated libraries. 

In both prefill and decoding, the input of any of the network’s projection has as many rows as there tokens being processed. And during decoding, the number of tokens being processed is equal to the batch size. So during decoding, the number of input rows of all GEMM kernels is equal to the batch size, which for our purposes ranges between 1 and 256. We are going to take an interest with very low batch sizes.
When we have a GEMM $A * B = C$ such that $A$ has few rows and many columns, we say that the GEMM is **skinny**. The reason we have a specific term for such GEMMs is that they are ill-fitted for the classic GEMM algorithm we run on GPU. Usually, the efficiency of a GEMM kernels comes **tiling**: we divide the result matrix in many sub-matrices, called tiles, and we assign each tile to a different compute unit (CU). If we have many tiles, we can use many CUs and GPU usage is high. This is illustrated in the figure below.

![Classic GEMM dimensions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/classic_gemm.png)

But if the input $A$ has very few rows, then only a few tiles can be formed, which results in only a few compute units active, hence low GPU utilization:

![Skinny GEMM dimensions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/skinny_gemm.png)

Skinny GEMMs are fundamentally inconvenient for the GPU. In the next part, we are going to see how through a custom kernel that assumes we are in a skinny GEMM context, we can make them more convenient.

### Optimization: split-K

Since the main issue of skinny GEMMs is that we use too few compute units, the first thing we can do is figure out a way to use more. To do this, we can exploit the following mind-breaking formula:

$$
c_{ij} = \sum_{k=1}^K a_{ik} b_{kj}
= \left( \sum_{k=1}^{K/2} a_{ik} b_{kj} \right)
+ \left( \sum_{k=1+K/2}^{K} a_{ik} b_{kj} \right)
$$

Thanks to the associativity of the sum, we can split the main GEMM along the shared axis (commonly referred to as the **K axis**) and replace one GEMM with several sub-GEMMs that are executed concurrently. Each sub-GEMM is going to use as many CUs as the main one would have, so the number of CUs used is multiplied by the number of times we split the K axis. This is shown in the figure below: 

![Split-K algorithm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/split_k.png)

Here, we set split K equal to 2 and thus double the amount of CU used at once. Since we get partial results, we need to add them up after the both sub-GEMMs are done. What may seem counter-intuitive is that we are adding an operation, summing the partial results, yet we claim to reduce the latency of the overall process. But since each CU needs to go through the entire K axis to compute the result, because we are cutting it in two, the amount of work done by each CU is also cut in two. If the amount of work saved this way counter balances the amount of work added by the summing up of the final results, then we have an overall optimization. This is generally true as long as K is large and the original GEMM uses less than 50% of the GPU.

### Optimization: removing padding

If we assume that through split-K, most compute units are busy with their own tile, we can focus the scope of optimization at the compute unit level. We are going to take a look at how the actual matrix multiplication is done, and how we can accelerate it. 

In state of the art GPUs like the MI300X, matrix multiplication is handled by a dedicated hardware unit called tensor cores. Tensor cores only perform matrix multiplications, but they do so at very high speed. 
The format of tensor core instruction is `mfma_MxNxK...` where `mfma` stands for matrix fused multiply-add, `M` is the number of rows of the left-hand matrix, `N` the number of column of the right-hand matrix, and `K` is the shared dimension of both.  We illustrate an hypothetical instruction `mfma_2x2x4` below:

![MFMA dense version](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/MFMA_full.png)

There are only a few tensor core instructions, but for any triplet `MxNxK` using the dedicated tensor core instruction is much faster than any other alternative. 
Tensor core instruction also come in two flavours: “dense” and “sparse”. Dense instruction correspond to standard matrix multiplication. Sparse instructions assume that the left-hand side matrix $A$ has a 4:2 structured sparsity pattern, which means that two out of every 4 elements along the matrix K axis are zero. Mathematically, for any $i, j$ such that $a_{i, 4j+3}$ is an element of $A$, we have at least two zeros in $\left( a_{i,4j}, a_{i,4j+1}, a_{i,4j+2}, a_{i,4j+3} \right)$ . Below is an example of a sparse matrix.

![A 4:2 sparse matrix](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/sample_sparse.png)

Let’s get back to our model, Llama 405B in FP8. For FP8, we only have two dense tensor core instruction: `16x16x32` and `32x32x16` . We also have one sparse instruction of size `16x16x64` .
For an input with 8 rows, using even the smallest dense instruction `16x16x32` means that we have to add 8 rows of padding to our input, which is a waste of compute resources. One can wonder if we can use the sparse instruction instead: after all, if half of a 16 rows matrix is 4:2 sparse, we can fully describe its non-zero coefficients using a dense 8 rows matrix. Conversely, if we have an 8 rows dense matrix, we can fit all of its data into a 16 rows matrix with 4:2 sparsity. And the benefit of using the sparse instruction is obvious: the dense instruction has `K=32` while the sparse instruction has `K=64` . For the same amount of cycles, the sparse instruction has twice the depth. We illustrate this sparsity trick in the figure below with a 1 row input and the `2x2x4`  dense instruction and its sparse `2x2x8` counterpart. 

![Using sparsity for skinny inputs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/sparsity_trick.png)

Using this trick, we can notably speed up our GEMM for any input with 8 or less rows, which results in a reduction in per-token latency for any decoding batch that has less than 8 requests.

### Optimization: warp specialization and asynchronous execution

We have seen that in a skinny GEMM, the fact we have a little number of rows limits the number of output tiles, which in turns limit the GPU utilization. But the small number of rows also limits the number of rows each output tiles has, which in turns reduces what we call **arithmetic intensity**. Simply put, arithmetic intensity is the amount of work done divided by the amount of data loaded to do that work. Let us compare two examples:

$$
s_n = \sum_{i=1}^{n} x_i \\
t_n = \sum_{i=1}^n y^i = y ~( 1 + t_{n-1})
$$

where $x$ is an $n$-sized vector and $y$ is a scalar. 
To compute $s_n$, we load $n$ elements and perform $n-1$ additions. To compute $t_n$, we load 1 element and perform $2n-1$ additions and multiplications. So the “arithmetic intensity” of computing $s_n$ is $\frac{n-1}{n}$ while $t_n$ is $2n - 1$: the computation of $t_n$ is more “arithmetically intensive” than the computation of $s_n$ .  What we see here is that when the **lower arithmetic intensity is, the more data we need to load to perform work**. 

Why does this matter to us? Well, we have seen that loading data from VRAM has a high latency cost, which is not great for the GPU. In other words, workloads with low arithmetic intensity are ill-suited for the GPU, and it turns out skinny GEMMs have lower arithmetic intensity than their non-skinny counterparts. This becomes intuitive when looking at the figure below: we can see that when we divide the amount of data loaded by two, we divide the number of output coefficients by 4, due to the quadratic nature of the GEMM’s dimensions. 

![The arithmetic intensity of two GEMMs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/gemm_AI.png)

In a skinny GEMM the number of rows of the output tile is limited and so is the the arithmetic intensity. Already this means that we are going to need to load a lot of data to compute an output tile. Furthermore, since we are using FP8 arithmetic, computation is quite fast, so we cannot rely on computation time to hide the latency of data loading. All in all, it would be ideal to have more threads in charge of loading data than threads in charge of computing the result. 

To achieve this, we are going to use a technique called **warp specialization**. Instead having all warps in the thread block execute the same instructions, we are going to dedicate some warps to loading data only and some to computing the results only. The warps in charge of loading data are called **producers** and the ones that compute the results are named **consumers**. Producers and consumers work asynchronously: producers first load data from the VRAM, which is slow, and make it available to the consumers by storing it in a shared memory buffer. Until data is available in shared memory, the consumer is idle. After it data is made available, the consumer loads it from shared memory, which is fast, and computes the result. 
Coordination of producers and consumers is achieved through a queue stored in shared memory. When a producer finishes storing data in a shared memory buffer $i$, it changes the state of the $i$th variable of the queue to signal data is available there. The consumer is watching out for this, and begins loading data afterwards. When it is done, it changes the $i$th variable of the queue to signal that data can be written over in buffer  $i$. 
In the figure below, we represent the steps involved in a simple asynchronous GEMM with one producer, one consumer and a queue of size 2.

![Async GEMM mechanism](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/async_work.png)

What makes the whole process work is that once buffer $0$ is filled by a producer, it can start working on buffer $1$ without waiting for the consumer to have loaded the data buffer $0$. The goal is to have a queue large enough for the producers to be constantly filling buffers and consumers constantly consuming them. The size of queue is constrained by the size of the shared memory. 

We also need to tune the ratio of producers to consumers: we have said that we have a low arithmetic intensity, so we need to load a lot of data to do a relatively fast computation. Hence, we are going to have a lot of producer warps (typically 8 or 10) for a few consumer warps (something like 2 or 3). Furthermore, we can exploit the fact the GEMM is skinny by having separate producers for the input (the skinny matrix) and the weights (the non-skinny matrix). To make the output tile bigger in the dimension in which it is not constrained in, which is the columns dimension, we allocate more producers for the weights. 

For a more in depth blog post about asynchronous GEMMs, I encourage you to check out [this](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) blog post. A lot of its contents are not applicable in our case though: the MI300X has no warp-level barriers, only a single thread block-level barrier. This lead to “fun” shenanigans like ASM to ensure warps waited at their barriers, shared memory loads and stores were resolved before checking the barrier state, and careful handling of the modular nature of the queue. All this would be out of place here, but I encourage you to check out the [code](https://github.com/huggingface/hf-rocm-kernels) or ask away in the comments. A deep dive on the details of async handling might be coming in the future.

Through warp specialization and asynchronous work, we can adapt our kernel to the low arithmetic intensity workload, but is that enough to come ahead of libraries like hipBLASLT? The answer is yes, in some cases.

### Results

Since Torch already binds a highly optimized GEMM taken from AMD’s linear algebra library, we are not going to get speedups in the same range as for the two last kernels.
We are first going to take a look at the three GEMM dimension that are of interest to us: namely, the GEMMs dimensions associated with the QKV projection, the Gate / Up projection and the Down projection. The Output projection is being left out because its dimensions do not correspond to the skinny GEMM case. 

| M (rows) | N (cols) | K (depth) | Torch time (μs) | SkG time (μs) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 2304 | 16384 | 14.938 ± 0.292 | 11.685 ± 0.299 | 127.84 % |
| 8 | 2304 | 16384 | 16.300 ± 0.282 | 12.342 ± 0.375 | 132.07 % |
| 16 | 2304 | 16384 | 16.693 ± 0.233 | 13.909 ± 0.295 | 120.02 % |
| 32 | 2304 | 16384 | 16.817 ± 0.124 | 17.021 ± 0.133 | 98.80 % |
| 1 | 13312 | 16384 | 77.636 ± 0.364 | 54.717 ± 0.628 | 141.88 % |
| 8 | 13312 | 16384 | 80.031 ± 0.449 | 58.355 ± 0.612 | 137.15 % |
| 16 | 13312 | 16384 | 75.236 ± 0.378 | 59.973 ± 1.922 | 125.45 % |
| 32 | 13312 | 16384 | 82.198 ± 0.590 | 69.483 ± 1.672 | 118.30 % |
| 1 | 16384 | 6656 | 31.066 ± 0.193 | 27.613 ± 0.218 | 112.51 % |
| 8 | 16384 | 6656 | 31.559 ± 0.200 | 28.134 ± 0.209 | 112.17 % |
| 16 | 16384 | 6656 | 31.671 ± 0.250 | 30.233 ± 0.267 | 104.76 % |
| 32 | 16384 | 6656 | 35.561 ± 0.335 | 35.052 ± 1.365 | 101.45 % |

Measures are taken after 500 warmups iterations, over 2000 profiling iterations, using CUDA graph and multiple weights to avoid cache hits. 
In order, the GEMM dimensions shown above correspond to QKV projection (N = 2304 and K = 16384), Gate / Up projection (N = 13312 and K = 16384) and Down projection (N= 16384 and K = 6656). We can see that for those dimensions, which have been tuned for, there is a notable speedup for low number of rows (M = 1, 8, 16) but less so for more rows (M = 32). Especially for dimensions in which we can use our sparsity trick (M = 1, 8) we see a notable speedup over Torch, which probably pads everything to 16 rows to use the smallest MFMA instruction. 

## Conclusion

In this post, we explored just a handful of the many kernel optimization techniques available. If you're interested in experimenting with them, feel free to dive into the **hf-rocm-kernels** repository and start tinkering! And if you develop a kernel you like of and want to distribute it, be sure to check out **kernel-builder** and **kernels** — two Hugging Face packages designed to help kernel builders make their work widely available and more impactful.
