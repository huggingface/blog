---
title: "Profiling in PyTorch (Part 1): A Beginner's Guide to torch.profiler"
thumbnail: /blog/assets/torch-profiler/thumbnail.png
authors:
  - user: ariG23498
  - user: sayakpaul
  - user: sergiopaniego
  - user: ror
  - user: pcuenq
---

# Profiling in PyTorch (Part 1): A Beginner's Guide to torch.profiler

![Thumbnail of the blog post](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/thumbnail.png)

> *What you cannot profile, you cannot optimize.*

Whether you are trying to squeeze more tokens per second out of a Large Language Model (LLM), shave milliseconds off inference, or just understand why your training loop runs slower than the spec sheet promises, the path eventually runs through profiling.

The catch is that profiling has a **steep** on-ramp. The traces are dense walls of colored rectangles. The events carry intimidating names. Most tutorials assume you can already read them. So even when we *know* we should be profiling, opening a trace can feel like a chore best left for later (or for someone else). This post, and the series it kicks off, is our attempt to lower that on-ramp.

This is the opening post of **Profiling in PyTorch**, a series where we slowly build the skill of reading profiler traces and use it to drive optimization. The plan:

1. **Part 1 (this post):** start with the simplest possible operation, a matrix multiplication followed by a bias add, and learn how to read what the profiler hands back.
2. **Part 2:** scale up to `nn.Linear` and a small MLP, use the traces to motivate optimizations, and peek at the `kernels` underneath.
3. **Part 3:** put it all together on Large Language Models with `transformers`.

We document the journey from a beginner's point of view. No prerequisites apart from basic PyTorch. Treat this as a leisurely read with some "Aha!" moments. The structure of the post is intentionally question-led: we open a trace, ask "wait, why is *that* happening?", and chase the answer until something clicks. By the end you should know:

- how to set up `torch.profiler` and what it actually hands back,
- how to read the profiler table and the trace (CPU lane, GPU lane, and the suspicious gaps in between),
- the chain of events from a Python call all the way down to a CUDA kernel,
- what changes (and, more interestingly, what does **not** change) when you slap `torch.compile` on top.

Before we begin, two definitions that will make everything below read better:

1. A GPU **kernel** is a program that runs in parallel on many threads of the GPU.
2. The CPU **schedules and launches** these kernels.

You don't usually have to write GPU kernels yourself; when you use a PyTorch operation, it is automatically translated to one or more kernels that do the job on GPU.

With those two ideas in your back pocket, let's start asking questions.

> [!NOTE]
> Here is the entire script that we use for the post: [`01_matmul_add.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py). It is advised to open this script on a separate tab and walk through the code step by step. We use the `NVIDIA A100-SXM4-80GB` GPU to run the scripts.

## The matrix multiplication and addition operation

As correctly [quipped by Dr. Sara Hooker](https://youtu.be/7knwihgj0fU?si=uvzGH-J9bsCHP4Nn&t=2199), like we are primarily made up of water, Deep Neural Networks are primarily made up of matrix multiplies. As fundamental as they are, it would be a shame to start our profiling journey with anything else.

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)
```

> The matrix addition along with the matrix multiplication mimics how weights and biases interact in a neuron. This addition (pun intended) will help us understand how it paves the way for compilation [later in the post](#lets-see-some-torch-compile-at-work).

To profile, we will be using the `torch.profiler` module. The steps involved are:

1. Have the [code to profile ready](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L26-L27) (here `def fn`, which wraps the matrix multiplication and matrix addition)
2. [Annotate](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L32) the algorithm. While this is completely optional, we recommend doing this. The `record_function` annotates our function as `matmul_add`, which will be easy to navigate in the traces (as we note later)
```py
def step():
  with torch.profiler.record_function("matmul_add"):
    return fn(x, w, b)
```
3. Wrap the code with the `torch.profiler.profile` [context manager](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L53-L62)
```py
  with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,  # the cpu activities
        torch.profiler.ProfilerActivity.CUDA, # the gpu activities
    ],
  ) as prof:
    # it is recommended to run events multiple times to warm up the GPUs
    for _ in range(5):
      step()
      prof.step()
```
4. Export the [profile](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L70)
```py
# the profiler table
prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)

# the profiler trace
prof.export_chrome_trace(trace_path)
```

The profiler exports two distinct artifacts:

1. The profiler table: Provides the statistical summary of the algorithm. It answers "What is taking the most time". This becomes really helpful to figure out hotspots. A hotspot would be events that take the most amount of time, can be a bottleneck of the pipeline, or an event that is triggered a lot of times.
2. The profiler trace: Provides the temporal execution view. Answers "When and Why an operation happened", depicting the activities taking place on the CPU and the GPU. This is helpful when we want to investigate the kernel(s) that were launched, any delays in launching them, any overlap between CPU and GPU activities, etc.

Let's see the two in action with our first execution. ([Here is the entire `01_matmul_add.py` script](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py))

> [!NOTE]
> It is recommended to run this script on a machine with a GPU.

```bash
uv run 01_matmul_add.py --size 64
```

If you run the above script (on a GPU machine) you will find a folder `traces/01_matmul_add` with the two artifacts:

```bash
64_bf16_cold_eager.json
64_bf16_cold_eager.txt
```

| ![Profiler table for matmul add on 64 sized matrices](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profile-table-64.png) |
| :--: |
| Figure 1: Profiler table for matmul add on 64 sized matrices|

The `.txt` file holds the profiler table. Upon opening the file, as shown in Figure 1, one would be greeted with a big table with the first column consisting of the events that were triggered inside the scope of profile.

The other columns are related to the time the event takes on the CPU or GPU or any other device(s) specified in `activities` within `torch.profiler.profile`. Look at which events take the most amount of time, and try to intuitively understand if that event should in fact take that time. It is also important to look at the column "# of Calls" which dictates how many times the event was triggered.

While we are at it, let's also talk about "Self CPU/CUDA" vs "CPU/CUDA total". The "Self" columns measure time spent only inside the event itself, excluding its children. The "total" columns include the event and all of its children together. So if you look at the "CPU total" of `matmul_add`, it consists of the time it took on self plus the children events it triggered. This is an important nuance to note.

If you look at the last two lines out of the table you would notice that the profiler tells us that

```bash
Self CPU time total: 2.314ms
Self CUDA time total: 23.104us
```

The CPU time is in `ms` while the GPU time is in `us`. To put things in perspective, the time spent on GPUs (the kernel `ampere_bf16_s16816gemm...`) is less than 1% of the time spent on the CPU (the `matmul_add` operation).  The GPU stays idle most of the time, which is an immediate red flag. The reason this happens is that the GPU can compute a small matmul very quickly, so our code spends most of the time preparing the kernels, launching them on the GPU, sending the data to multiply and gathering the results. This concept is known as an _overhead-bound_ algorithm.

The easiest way to move out of this regime is to use bigger matrix multiplications.

```bash
uv run 01_matmul_add.py --size 4096 
```

| ![Profiler table for matmul add algorithm on 4096 sized matrices](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profiler-table-4096.png) |
| :--: |
| Figure 2: Profiler table for matmul add on 4096 sized matrices |

The last two lines in Figure 2 are:

```bash
Self CPU time total: 4.908ms
Self CUDA time total: 4.495ms
```

Both times are in ms, which means we have materialized more GPU time just by increasing the size of the matrix multiplications. If you look at Figure 2 you would also notice that the most CUDA time is now taken by the GPU kernel (`ampere_bf16_s16816gemm_..`) and not by the CPU operation that launched it (`matmul_add`). This means that we were indeed able to move from overhead bound to compute bound.

We now move into visualising the dispatch chain, which lives inside the `.json` artifacts. You can upload them to [Perfetto UI](https://ui.perfetto.dev) and see the traces, or you can use `uvx trace-util traces -b traces` to generate the Perfetto links directly.

## 64x64 traces

| ![PyTorch profiler trace of a 64×64 bf16 matmul followed by an add on a CUDA GPU](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/64-matmul-add.png) |
| :--: |
| Figure 3: Profiler trace for matmul and add on 64 sized matrices |

In Figure 3, we see the profiler trace for the matrix multiplication and addition. Here the bar width indicates the duration of an event, the vertical nesting is the call hierarchy, the CPU lane denotes the events that happen on the CPU, while the GPU lane shows the actual kernel executions. One might also notice the empty spaces which are the waiting or idle time.

The script was run with default configurations which are:

- size 64: The inputs, weights and biases are sized (64, 64)
- dtype bf16: The data type is bfloat16
- no compile: We have not compiled the torch operations
- no warmup: We have not warmed up the GPU before profiling

> With Perfetto we suggest using the keyboard for quicker access to the trace. One could use "W A S D" for navigating the trace.

| ![PyTorch profiler trace with the CPU lane and GPU lane labelled side by side in Perfetto](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gpu-cpu-trace.png) |
| :--: |
| Figure 4: The CPU and GPU lanes of a PyTorch profiler trace |

There are two lanes in Figure 4, one for the CPU activity and one for the GPU activity. In the CPU lane one would notice three profile steps (starting from `ProfilerStep#2`). This comes from the `schedule`.

```py
schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
```

The `wait` skips noisy initializations (`ProfilerStep#0`), `warmup` runs through the profiler without recording (`ProfilerStep#1`), and `active` is what shows up in trace. One can find the schedule being used in the [script here](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L58).

Let's put on our detective hats and investigate the trace and ask some questions.

### Why does the ProfilerStep#2 take so long?

| ![ProfileStep#2 in a PyTorch profiler trace appears wider than ProfileStep#3 and ProfileStep#4](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/why-is-step-2-big.png) |
| :--: |
| Figure 5: `ProfileStep#2` is visibly wider than the steps that follow it |

In Figure 5, we notice that `ProfileStep#2` takes more time compared to the other steps, and upon looking closely you would see a similar pattern with the `matmul_add` annotation as well. The smoking gun is inside the annotation, not the annotation itself:

| Step | `matmul_add` start | `aten::matmul` start | gap |
| :--: | :--: | :--: | :--: |
| #2 | 138.736 | 366.493 | 227.757 µs |
| #3 | 517.926 | 523.447 | 5.521 µs |
| #4 | 610.039 | 614.527 | 4.488 µs |

| ![228 microsecond gap between record_function matmul_add and the aten::matmul dispatch in profile step 2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-227.png) |
| :--: |
| Figure 6: The ~228 µs dead window between `record_function("matmul_add")` and `aten::matmul` |

That ~228 µs shown in Figure 6 is the "dead window" between entering `record_function("matmul_add")` and PyTorch actually dispatching `aten::matmul`. This can happen for multiple reasons, including workspace allocations, [cuBLAS](https://developer.nvidia.com/cublas) (NVIDIA’s proprietary, GPU-accelerated library for performing fundamental linear algebra operations) heuristics, or lazy module loading. We can either look away or run [some more warmup steps](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L35-L39) before we profile (which is the standard)

In terms of profiling, warmup is when you run the events a couple of times before actually profiling it. The pre-work done by the GPU (including the above pointers) are one time efforts which we do not want to profile. In our example, we have two warmup stages, one where we actually loop over the function before entering the profiler, and two inside the profiler which is achieved by the `warmup` argument. In this section, we have enabled the actual iterations along with the schedule.

```bash
uv run 01_matmul_add.py --warmup
```

[Perfetto Trace for 64x64 with Warmup](https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/ariG23498/traces/resolve/01_matmul_add/64_bf16_warm_eager.json)

| ![PyTorch profiler trace after warmup steps where ProfileStep#2 no longer shows cold-start overhead](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/warmup.png)|
| :--: |
| Figure 7: After warming up, every profile step takes a similar amount of time |

In Figure 7 we see that each profile step takes a similar time, but this does not mean we were able to optimize the one time overheads. We warmed up the runs so that the overheads were not profiled. We think that closing this section abruptly without a hint to solving this would do injustice to the reader, so here is a [link](https://pytorch.org/blog/accelerating-generative-ai-2/) to read about further optimizing launch overheads.

### Why is there an offset of ~2.5 ms between the CPU and GPU lanes?

| ![2.32 millisecond offset between the CPU lane and the GPU lane in a PyTorch profiler trace](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-bw-kernel-launch.png) |
| :--: |
| Figure 8: The ~2.5 ms offset between the CPU and GPU lanes |

In Figure 8, we see that the CPU and GPU lanes have an offset of around 2.5 ms: this is the delay after the CPU submits the CUDA kernels and the time they actually start executing. One might think the warmup stage combined with the schedule's `wait` and `warmup` should keep a GPU busy and would diminish the offset.

To uncover what is really happening, let's change our schedule a little:

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1)
```

| ![PyTorch profiler trace with wait=0 warmup=0 showing an Activity Buffer Request between steps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/full-profile.png) |
| :--: |
| Figure 9: With `wait=0` and `warmup=0`, the trace reveals an `Activity Buffer Request` |

Figure 9 shows us that there is an `Activity Buffer Request` in the GPU lane before any operation. Let's zoom in a little more.

| ![gap between matmul and add CUDA kernels caused by profiler buffer request](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/mat-add-gap.png) |
| :--: |
| Figure 10: A gap appears between the matmul and add kernels on profile step 1 |

Upon zooming into the GPU trace, we notice that the matmul and add kernels for `ProfileStep#0` (the CPU trace of which is not visible in the Figure) happen one after the other, while the kernels for `ProfileStep#1` have a window in between. The best explanation for this is that there was an overflow of buffers, and another buffer request (a request to allocate some memory on the GPU VRAM) was issued during the kernel execution.

The best way to rule out other possibilities is to profile for more iterations and see whether a similar window appears in other parts of the trace. To do that we run with `active=20`.

| ![PyTorch profiler trace of 20 active iterations confirming the buffer-request gap only appears once](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters.png) |
| :--: |
| Figure 11: With 20 active steps the gap only shows up once, confirming it is a buffer request |

As shown in Figure 11, we see a similar trend in `ProfileStep#1`. This is aligned with our previous findings, and we can safely conclude that it was indeed another buffer request.

### The chain of events

| ![nested CPU dispatch chain in PyTorch profiler: ProfileStep, matmul_add, aten::matmul, aten::mm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cpu-nests.png) |
| :--: |
| Figure 12: The chain of dispatch |

In Figure 12, we see the nested CPU calls. This is an important visualization, where one gets to understand what a chain of dispatch really looks like.

We begin with `ProfileStep#<id>` which encapsulates the profiling step. Due to us annotating the step, we see the `matmul_add` row. The `matmul_add` consists of two `aten` calls, one for matrix multiplication and one for matrix addition.

The `aten::matmul` is the [ATen-level](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen) dispatch that those user-facing PyTorch matmul calls land on. `aten::mm` is the 2D matrix-matrix multiply backend.

It is very interesting to note how PyTorch calls `aten::bmm` (batched matrix multiplication) if we add the batch axis to our matrices. Let's take a detour and see the `aten::bmm` in action.

```diff
- x = torch.randn(args.size, args.size, device=device, dtype=dtype)
- w = torch.randn( args.size, args.size, device=device, dtype=dtype)
- b = torch.randn(args.size, args.size, device=device, dtype=dtype)

+ # adding a batch size of 8
+ x = torch.randn(8, args.size, args.size, device=device, dtype=dtype)
+ w = torch.randn(8, args.size, args.size, device=device, dtype=dtype)
+ b = torch.randn(8, args.size, args.size, device=device, dtype=dtype)

```

| ![PyTorch profiler trace showing aten::matmul dispatching aten::bmm for 3D batched tensors](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/bmm.png) |
| :--: |
| Figure 13: Batched Matrix Multiplication |

In Figure 13, upon adding the batch axis to the inputs, `aten::matmul` now encapsulates a bunch of other prerequisite CUDA runtime calls along with `aten::bmm` (instead of `aten::mm`). This also hints at the heuristics that cuBLAS needs to do in order to dispatch the right (most suitable) kernel for the program.

> In the rest of the post, we will be working with simple 2D matrices, unless otherwise mentioned.

### Why does matmul have an extra CUDA runtime call?

| ![CPU lane showing cudaOccupancyMaxActiveBlocksPerMultiprocessor preceding the matmul cudaLaunchKernel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cudaoccupancy.png) |
| :--: |
| Figure 14: A CUDA occupancy query fires before the matmul kernel launch |

We notice that for `aten::mm` there are two CUDA Runtime calls, namely `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (boxed in Figure 14) and `cudaLaunchKernel`, while for `aten::add` there is only the `cudaLaunchKernel`.

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` is a planning call and is purely CPU side. It asks: "given a kernel function, a chosen block size, and a chosen dynamic shared memory size, how many blocks of this kernel can simultaneously reside on one SM (Streaming Multiprocessor)?"

This begs the question, why do we need planning for matmul and not for add?

To understand this, we have to look at the kernel's resource footprint. If you click on the GPU kernels, you will be able to inspect the resource footprint for the respective kernel.

| ![cuBLAS matmul kernel resource footprint: registers, shared memory and block size in Perfetto](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/matmul-footprint.png) | ![elementwise add CUDA kernel resource footprint with 32 registers and zero shared memory](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/add-footprint.png) |
| :--: | :--: |
| Figure 15: Matmul footprint | Figure 16: Add footprint |

In Figure 15, we note that for matrix multiplication the `registers per thread` and `shared memory` are dynamic (based on the size of the matrix). cuBLAS ships hundreds of kernel variants, and each has a heuristic-driven launch path that needs runtime information about hardware capacity. The occupancy query is part of that heuristic. Conceptually, we can think of GPU-accelerated matmuls as [working on independent tiles](https://alvinwan.com/how-to-tile-matrix-multiplication/): how many tiles we use and how big each tile needs to be depends on the matrices and the hardware. Modern algorithms are way more complicated than that, but this is still a good reference framework.

From Figure 16 we see that the footprint of addition says 32 registers and zero shared memory. That fits trivially. There's nothing to query, because no hardware resource is going to limit occupancy. The kernel is, by design, resource-light.

> [!NOTE]
> You can use this as a quick diagnostic when reading any trace. Scan the CPU lane for `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. Each occurrence flags a "heavyweight, adaptively launched" kernel, usually a GEMM (GEneral Matrix Multiplication), conv, or similar. The kernels without a preceding occupancy query are the elementwise/reduction crowd that PyTorch launches mechanically.

### Why is cudaDeviceSynchronize so big (~1.78 ms)?

`cudaDeviceSynchronize` blocks the CPU until all GPU work on this device finishes. The profiler emits this sync at the end of the active window to flush events. Without it, kernel timings would be missing.

A 1.78 ms sync covering 26 µs of real GPU work tells you this run was 98% idle. That's the textbook overhead-bound symptom.

## 4096x4096 traces

We already know from the profiler table analysis (above) that providing bigger matrices to our algorithm moves it out from the overhead-bound region to being compute-bound. 

Let's run the command and dive deeper into the traces.

```bash
uv run 01_matmul_add.py --size 4096 --warmup
```

### Why does the same kernel take more time compared to others?

| ![4096x4096 bf16 matmul kernel timings varying across profiler steps on the same GPU](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/kernel-time.png)|
| :--: |
| Figure 17: One matmul kernel runs longer than the others despite identical inputs |

In Figure 17, we notice that the matmul kernel for `ProfileStep#3` takes longer on the GPU than the other steps. This is particularly interesting to note, because the other kernels launched were the exact same, which means there were no cuBLAS heuristics involved. There are no scheduling gaps, the CPU launches are normal, and it is not a profiler artifact.

This trace in Figure 17 makes a useful point that's easy to miss in idealized examples: kernel runtimes are not constants, even on the same hardware environment running identical code on identical data.

Let's make this more concrete by modifying the script a little. We run the iteration 20 times, capturing each of the steps.

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=20, repeat=1)

- for _ in range(5):
+ for _ in range(20):
```

| ![PyTorch profiler trace of 20 matmul iterations showing kernel runtime variance](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters-kernels.png) |
| :--: |
| Figure 18: Across 20 iterations the same matmul kernel runs at different speeds |

Figure 18 reveals a similar finding. While each kernel was the exact same, they time differently. The different compute times can be blamed on a bunch of reasons:

- GPU clocks on idle and boost
- GPU heating
- GPU power management
- Driver side housekeeping

A reader who only saw the average would conclude that a matmul took ~1 ms (mean of 5 = 1084 µs); a reader who looked at the trace would see that the matmul takes ~580 µs except when the GPU throws a fit. Those are very different mental models, and only one of them is correct.

## Let's see some torch compile at work

Working with `torch.compile` has always amazed me. One writes normal eager PyTorch code, but PyTorch tries to capture tensor-heavy regions, turn them into graphs, optimize them, and run generated code. The default backend is usually `TorchInductor`, and the broad pipeline is:

1. `TorchDynamo` captures Python execution into an FX graph
2. `AOTAutograd` prepares forward/backward graphs when gradients are involved
3. `Inductor` lowers the graph into optimized CPU or GPU code.

In this section, we talk about compilation and look at the profiler traces.

```bash
uv run 01_matmul_add.py --size 4096 --warmup --compile
```

The `args.compile` flag triggers the following code:

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)

fn = torch.compile(fn) if args.compile else fn
```

| ![torch.compile region highlighted in a PyTorch profiler trace, showing TorchDynamo and Inductor frames](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/compilation-region.png) |
| :--: |
| Figure 19: The compiled regions show up as TorchDynamo and Inductor frames in the trace |

In Figure 19, we see the new CPU rows named `Torch-Compiled Region: 0/0` which points us to the compiled functions being used.

### Did we fuse the matmul and add kernels into one?

| ![Compiled trace showing aten::addmm replacing the eager aten::add and aten::mm pair](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/fused-ops.png) |
| :--: |
| Figure 20: Compiled run dispatches a single `aten::addmm` |

Looking at Figure 20 we ask the question, did we actually fuse the multiplication and addition operations together into one?

This is operator fusion at the graph level. Inductor took our `torch.add(torch.matmul(x, w), b)` and rewrote it into a single `aten::addmm(b, x, w)` call. The important thing to note here is that it did **not** produce a **new** fused CUDA kernel. The actual GPU work is still `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nn`, the same cuBLAS kernel eager mode used. So the "fusion" here is at the dispatcher level, not at the kernel level.

> [!NOTE]
> PyTorch provides the [`torch.addmm`](https://docs.pytorch.org/docs/2.12/generated/torch.addmm.html) function that does what we did into two steps, that is multiply and add. We encourage the reader to look at the traces of this function and comment your observations in the comments below!

### torch.compile's runtime architecture

While we know in theory what happens when we compile our functions it is equally important to see it in action. Let's look at the CPU-side hierarchy which reflects `torch.compile`'s runtime architecture.

**TorchDynamo Cache Lookup** is where Dynamo checks that the current call still matches what was compiled with the same input shapes, dtypes, devices, and tensor metadata. If anything mismatched, Dynamo would recompile. This cost is paid every call, even after compilation.

**Torch-Compiled Region** is the wrapper that "enters" the compiled version. **AOTDispatcher Runtime Wrapper Prologue** is AOT Autograd's runtime wrapper. Even though we don't need gradients here, AOTDispatcher is always in the stack handling tensor metadata, view tracking, and would set up the backward pass if `requires_grad` were true.

**## Call CompiledFxGraph <hash>** is where the actual generated code runs. The string after "CompiledFxGraph" is the content hash of the FX graph. It's the same across all three active steps, confirming cache hits.

> [!TIP]
> You can find the generated code on disk under `/tmp/torchinductor_<user>/fxgraph` keyed by this hash, useful when you want to read the Triton/C++ that Inductor actually produced.

### Do the CUDA launches go down by half?

| ![compiled matmul trace showing Memcpy DtoD and GEMM kernels launched per step](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/memcpy.png) |
| :--: |
| Figure 21: Each compiled step still launches two GPU kernels, a Device-to-Device memcpy and the GEMM |

Looking at the traces in Figure 21, we were really happy to notice only one `cudaLaunchKernel` per step. This observation was directly contradicting what we were seeing in the GPU trace. There were still two kernels being launched per step, namely the `Memcpy DtoD (Device -> Device)` and the GEMM. Going back to the CPU trace, we noticed that we had completely missed the `cudaMemcpyAsync` dispatch.

`addmm` computes `out = α·A·B + β·C`, and cuBLAS's GEMM-with-bias-add epilogue writes into a destination buffer that needs to already contain the bias. An epilogues can be thought of all the operations that happen _after_ a GEMM. In the world of deep-learning we constantly come up with GEMM-Epilogues like activations, bias addition, normalization and many more. This is why there are cuBLAS GEMM-with-<specific epilogue> kernel variants.

> If you use different `mode`s for `torch.compile` you would notice different kernel variants being launched. You can try it for yourself and add a comment below about your observations!

So Inductor's generated code does:

- `out = copy(C)` ← that's the DtoD memcpy (32 MB, takes ~33 µs)
- `out = α·(A·B) + β·out` ← GEMM with `α=β=1`, fusing the bias add into the writeback

The result is mathematically still the same. The bias add isn't free, as we pay a memcpy upfront plus a slightly more expensive GEMM epilogue.

The fusion one might have hoped for, where `x·w + b` (here `out = α·A·B + β·C`) collapses into a single kernel with no extra memory traffic, isn't what happened. Inductor preserved the two memory-touching operations, it just relabeled the bias copy as a memcpy and the addition as a GEMM epilogue.

A truly fused implementation would skip the memcpy. That's what FlashAttention-style hand-written kernels do, and what Inductor can do via Triton codegen, but for a `4096×4096 bf16 matmul`, Inductor evidently decided "use cuBLAS, do the bias via epilogue setup" was the best path.

### CPU overhead went up, not down

This is the easiest thing to miss when comparing an eager and a compiled run:

| step | eager dur (ms) | compile dur (ms) |
| :--: | :--: | :--: |
| #2 | 0.1 | 0.2 |
| #3 | 0.07 | 0.1 |
| #4 | 0.07 | 0.1 |

Compile is roughly 2× more expensive on the CPU per step. That's because every call walks the full Dynamo > AOTAutograd > Inductor stack, on top of the same `aten::addmm` dispatch we have anyway. The compile pipeline is built for ML models with dozens of ops where the per-call overhead amortizes (for a single op it's a tax).

> [!TIP]
> `torch.compile` has a `mode` argument. It is for the reader to take home as an assignment to read the documentation and come up with a `mode` that could take the CPU overhead down. 🤗

## Trace reading cheatsheet

A quick reference for the patterns we walked through. The idea is: if you see this in a trace, this is what it usually means.

### Profiler table

| What you see | What it usually means |
| :-- | :-- |
| `Self CPU time total` ≫ `Self CUDA time total` (CPU in ms, GPU in µs) | Overhead-bound. The CPU spends more time dispatching than the GPU spends computing. Make the work bigger (larger matrices, batched ops) or fuse calls. |
| `Self CPU time total` ≈ `Self CUDA time total`, both in ms | Compute-bound. The GPU is the bottleneck, which is usually what you want. |
| One event dominates `CUDA total` | That's your hotspot. Start the optimization there. |
| One event has a huge `# of Calls` | A potential bottleneck even if each call is cheap. Check whether it can be fused or batched. |
| `CPU total` ≫ `Self CPU` for a row | Most of the cost lives in children. Drill into the nested events, not the parent. |

### CPU lane

| What you see | What it usually means |
| :-- | :-- |
| First `ProfileStep` much wider than the rest | Cold-start overhead: workspace allocation, cuBLAS heuristics, lazy module loading. Add warmup iterations and/or the schedule's `warmup` argument. |
| Big gap between `record_function("...")` start and the first `aten::*` inside it | Same cold-start tax, just zoomed in. The annotation entered, but the dispatch hadn't happened yet. |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` before a `cudaLaunchKernel` | A heavyweight, adaptively-launched kernel (GEMM, conv, etc.). cuBLAS is asking the driver how many blocks fit on an SM so it can pick a kernel variant. |
| `cudaLaunchKernel` with no preceding occupancy query | An elementwise or reduction kernel with a fixed, resource-light footprint. Nothing to plan. |
| A long `cudaDeviceSynchronize` at the end of the active window | The profiler flushing events. Its duration is mostly the GPU finishing pending work, not a real CPU cost. A sync covering tiny GPU work is a classic overhead-bound symptom. |
| A `cudaMemcpyAsync` you didn't write | Often a hidden Device-to-Device copy. Common when `addmm` seeds its destination buffer with the bias before the GEMM epilogue. |

### GPU lane

| What you see | What it usually means |
| :-- | :-- |
| `Activity Buffer Request` on the GPU lane | The profiler is allocating/refilling its own event buffer. The first one usually accounts for the initial CPU↔GPU lane offset. |
| A gap between two kernels in a single step | Likely another buffer request mid-execution. Confirm by running more iterations: if it appears only once, it's the profiler, not your code. |
| The same kernel timing differently across steps | GPU clocks, thermals, power management, driver housekeeping. Read the trace, not just the mean. |
| A kernel named like `ampere_bf16_s16816gemm_...` | The actual cuBLAS GPU work for a matmul. The kernel name is typically the same in eager and compiled mode for the same shapes/dtypes. |
| `Memcpy DtoD` immediately before a GEMM | The bias copy for an `addmm` epilogue. The "fusion" is at the dispatcher level, not in the kernel. |

### Dispatch chain

| What you see | What it usually means |
| :-- | :-- |
| `ProfileStep#N` → `<record_function name>` → `aten::*` → `aten::mm` / `aten::bmm` / `aten::add` | The canonical nested call hierarchy. Self time excludes children; Total time includes them. |
| `aten::matmul` resolving to `aten::mm` | 2D × 2D matrix multiply. |
| `aten::matmul` resolving to `aten::bmm` (with extra CUDA runtime calls) | Batched matmul on 3D+ tensors. cuBLAS does more heuristic work to pick the variant. |
| `aten::addmm(b, x, w)` instead of a separate `aten::add` + `aten::mm` pair | Operator fusion at the dispatcher level. The GPU kernel is still the same GEMM, with the bias add folded into the epilogue. |

### torch.compile

| What you see | What it usually means |
| :-- | :-- |
| A `Torch-Compiled Region: K/M` row in the CPU lane | You're inside a compiled function. |
| `TorchDynamo Cache Lookup` on every step | Dynamo is verifying shapes/dtypes/devices match the cached compile. Paid on every call, even after compilation. |
| `AOTDispatcher Runtime Wrapper Prologue` even with no grads | AOTAutograd's runtime wrapper is always in the stack, handling tensor metadata and view tracking. |
| `## Call CompiledFxGraph <hash>` with the same hash across steps | Cache hits on the generated code. The generated source lives under `/tmp/torchinductor_<user>/fxgraph/<hash>`. |
| Per-step CPU time higher under `torch.compile` than eager for a tiny op | Expected. The Dynamo → AOTAutograd → Inductor stack is a tax that only amortizes over many ops. |

## Conclusion

We started with a tiny `matmul + add` and used it as an excuse to learn how to read a PyTorch profiler. Along the way we picked up a few mental models that travel well to bigger workloads. This was the first stop in the **Profiling PyTorch** series. In the posts that follow, we will gradually leave this two-op toy behind and walk up the ladder of complexity, looking at larger building blocks and, eventually, real models.

Thanks to [Noe Flandre](https://huggingface.co/NoeFlandre), [Suvaditya Mukherjee](https://huggingface.co/suvadityamuk), and [Vidit Ostwal](https://huggingface.co/ViditOstwal) for their reviews on the early draft of the post!