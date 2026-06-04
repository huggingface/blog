---
title: "PyTorch 性能分析（第 1 部分）：torch.profiler 入门指南"
thumbnail: /blog/assets/torch-profiler/thumbnail.png
authors:
  - user: ariG23498
  - user: sayakpaul
  - user: sergiopaniego
  - user: ror
  - user: pcuenq
translators:
  - user: HCS9527
---

# PyTorch 性能分析（第 1 部分）：torch.profiler 入门指南

![博客文章缩略图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/thumbnail.png)

> *无法做 profiling，就无法优化。*

无论目标是从 Large Language Model（LLM）中挤出更多 tokens per second、为 inference 削掉几毫秒，还是弄清楚 training loop 为什么比规格参数承诺的速度慢，最终都会走向 profiling。

问题在于，profiling 的入门坡度很陡。trace 像一面密密麻麻的彩色矩形墙，event 名称也常常让人望而生畏。大多数 tutorial 默认读者已经知道如何读 trace。因此，即便已经知道应该做 profiling，打开一份 trace 也常常像是一件可以先放一放，或者交给别人处理的麻烦事。本文，以及这个系列，就是为了降低这个入门门槛。

这是 **Profiling in PyTorch** 系列的开篇。这个系列会逐步建立阅读 profiler trace 的能力，并用它驱动 optimization。计划如下：

1. **第 1 部分（本文）：** 从最简单的操作开始：一次 matrix multiplication 后接 bias add，学习如何阅读 profiler 给出的结果。
2. **第 2 部分：** 扩展到 `nn.Linear` 和一个小型 MLP，用 trace 推动 optimization，并观察底层 `kernels`。
3. **第 3 部分：** 将这些内容整合到使用 `transformers` 的 Large Language Models 上。

我们会从初学者视角记录这个过程。除了基础 PyTorch 知识，不需要其他前置要求。可以把本文当作一篇轻松阅读、偶尔带来「原来如此」时刻的文章。文章结构刻意围绕问题展开：打开一份 trace，问「等等，为什么会这样？」，然后一路追到答案变清楚。读完之后，应当能够理解：

- 如何设置 `torch.profiler`，以及它实际返回什么；
- 如何阅读 profiler table 和 trace（CPU lane、GPU lane，以及它们之间可疑的 gap）；
- 从一次 Python call 到 CUDA kernel 的完整 event chain；
- 在上面套一层 `torch.compile` 之后，什么发生了变化，以及更有意思的，什么没有变化。

开始之前，先给出两个定义，后文会更容易读：

1. GPU **kernel** 是一个在 GPU 上由许多 threads 并行运行的程序。
2. CPU 会 **schedule and launch** 这些 kernels（调度并启动这些 kernels）。

通常不需要自己编写 GPU kernels；使用 PyTorch operation 时，它会被自动转换为一个或多个在 GPU 上执行任务的 kernels。

有了这两个概念，开始提问。

> [!NOTE]
> 本文使用的完整 script 在这里：[`01_matmul_add.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py)。建议在单独的标签页中打开这个 script，并逐步阅读代码。我们使用 `NVIDIA A100-SXM4-80GB` GPU 运行这些 scripts。

## Matrix multiplication 和 addition 操作

正如 [Sara Hooker 博士的准确调侃](https://youtu.be/7knwihgj0fU?si=uvzGH-J9bsCHP4Nn&t=2199)：人主要由水构成，而 Deep Neural Networks 主要由 matrix multiplies 构成。matrix multiply 如此基础，如果不用它开启 profiling 之旅，反而有些可惜。

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)
```

> Matrix addition 与 matrix multiplication 放在一起，可以模拟 neuron 中 weights 和 biases 的交互方式。这个 addition（有意的双关）也会帮助理解它如何为后文的 compilation 铺路。

做 profiling 时，我们会使用 `torch.profiler` module。步骤如下：

1. 准备好 [要 profile 的代码](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L26-L27)（这里是 `def fn`，它封装了 matrix multiplication 和 matrix addition）。
2. [Annotate](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L32) algorithm（标注算法）。这个步骤完全可选，但建议执行。`record_function` 会把函数标注为 `matmul_add`，后续在 trace 中导航会更方便。

```py
def step():
  with torch.profiler.record_function("matmul_add"):
    return fn(x, w, b)
```

3. 用 `torch.profiler.profile` [context manager](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L53-L62) 包裹代码。

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

4. 导出 [profile](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L70)。

```py
# the profiler table
prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)

# the profiler trace
prof.export_chrome_trace(trace_path)
```

profiler 会导出两类不同 artifacts（产物）：

1. profiler table：提供 algorithm 的统计摘要。它回答「什么耗时最多」。这对定位 hotspots 很有帮助。hotspot 可能是最耗时的 event，可能是 pipeline 的 bottleneck，也可能是触发次数特别多的 event。
2. profiler trace：提供时间维度上的执行视图。它回答「某个 operation 何时发生、为什么发生」，并展示 CPU 和 GPU 上发生的 activities。当需要调查启动了哪些 kernel、kernel launch 是否有 delay、CPU 与 GPU activity 是否 overlap 等问题时，它很有用。

现在用第一次 execution 看看这两者如何工作。（[完整 `01_matmul_add.py` script 在这里](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py)）

> [!NOTE]
> 建议在带 GPU 的机器上运行这个 script。

```bash
uv run 01_matmul_add.py --size 64
```

运行上述 script（在 GPU 机器上）后，会看到一个 `traces/01_matmul_add` 文件夹，里面包含两个 artifacts：

```bash
64_bf16_cold_eager.json
64_bf16_cold_eager.txt
```

| ![64×64 矩阵上 matmul add 的 profiler table](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profile-table-64.png) |
| :--: |
| 图 1：64×64 矩阵上 matmul add 的 profiler table |

`.txt` 文件保存 profiler table。打开文件后，如图 1 所示，会看到一张很大的表。第一列是 profile scope 中触发的 events。

其他列与 event 在 CPU、GPU 或 `torch.profiler.profile` 的 `activities` 中指定的其他 device(s) 上花费的时间有关。需要观察哪些 events 耗时最多，并直观判断这个 event 是否真的应该花这么多时间。`# of Calls` 这一列也很重要，它表示该 event 被触发了多少次。

顺便也说一下「Self CPU/CUDA」与「CPU/CUDA total」的区别。「Self」列只统计 event 自身内部花费的时间，不包括它的 children。「total」列则包含 event 自身及其所有 children。因此，查看 `matmul_add` 的「CPU total」时，它包含自身花费的时间，以及它触发的 children events 花费的时间。这是一个需要注意的重要细节。

如果查看表格最后两行，会发现 profiler 给出：

```bash
Self CPU time total: 2.314ms
Self CUDA time total: 23.104us
```

CPU time 的单位是 `ms`，而 GPU time 的单位是 `us`。为了建立直观感受：GPU 上花费的时间（kernel `ampere_bf16_s16816gemm...`）不到 CPU 上花费时间（`matmul_add` operation）的 1%。GPU 大部分时间处于 idle 状态，这是一个明显的 red flag。出现这种情况的原因是，GPU 可以非常快地完成一个小的 matmul，因此代码的大部分时间花在准备 kernels、在 GPU 上 launch 它们、发送需要相乘的数据，以及收集结果上。这个概念称为 _overhead-bound_ algorithm。

脱离这种状态最简单的方法，是使用更大的 matrix multiplications。

```bash
uv run 01_matmul_add.py --size 4096
```

| ![4096×4096 矩阵上 matmul add algorithm 的 profiler table](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profiler-table-4096.png) |
| :--: |
| 图 2：4096×4096 矩阵上 matmul add 的 profiler table |

图 2 最后两行是：

```bash
Self CPU time total: 4.908ms
Self CUDA time total: 4.495ms
```

两者单位都是 ms，这意味着只通过增大 matrix multiplications 的尺寸，就让更多 GPU time 被实际利用起来。查看图 2 还会发现，现在 CUDA time 主要由 GPU kernel（`ampere_bf16_s16816gemm_..`）占用，而不是由 launch 它的 CPU operation（`matmul_add`）占用。这说明确实从 overhead-bound 转向了 compute-bound。

接下来进入 dispatch chain 的可视化，它保存在 `.json` artifacts 中。可以把它们上传到 [Perfetto UI](https://ui.perfetto.dev) 查看 traces，也可以使用 `uvx trace-util traces -b traces` 直接生成 Perfetto 链接。

## 64x64 traces

| ![一个 CUDA GPU 上 64×64 bf16 matmul 后接 add 的 PyTorch profiler trace](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/64-matmul-add.png) |
| :--: |
| 图 3：64×64 矩阵上 matmul 和 add 的 profiler trace |

图 3 展示了 matrix multiplication 和 addition 的 profiler trace。这里 bar width 表示 event duration，vertical nesting 表示 call hierarchy，CPU lane 表示发生在 CPU 上的 events，GPU lane 则展示实际的 kernel executions。还可以注意到一些空白区域，它们代表等待或 idle time。

script 使用默认配置运行：

- size 64：inputs、weights 和 biases 的尺寸是 (64, 64)
- dtype bf16：data type 是 bfloat16
- no compile：没有 compile 这些 torch operations
- no warmup：profiling 之前没有 warm up GPU

> 使用 Perfetto 时，建议使用 keyboard 更快地访问 trace。可以用「W A S D」导航 trace。

| ![Perfetto 中并排标注 CPU lane 和 GPU lane 的 PyTorch profiler trace](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gpu-cpu-trace.png) |
| :--: |
| 图 4：PyTorch profiler trace 中的 CPU lane 和 GPU lane |

图 4 中有两条 lanes，一条用于 CPU activity，一条用于 GPU activity。在 CPU lane 中可以看到三个 profile steps（从 `ProfilerStep#2` 开始）。这来自 `schedule`。

```py
schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
```

`wait` 会跳过 noisy initializations（`ProfilerStep#0`），`warmup` 会运行 profiler 但不记录（`ProfilerStep#1`），`active` 才是 trace 中实际显示的部分。这个 schedule 可以在 [script 这里](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L58) 找到。

现在戴上侦探帽，检查 trace 并提出一些问题。

### 为什么 ProfilerStep#2 耗时这么长？

| ![PyTorch profiler trace 中 ProfileStep#2 看起来比 ProfileStep#3 和 ProfileStep#4 更宽](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/why-is-step-2-big.png) |
| :--: |
| 图 5：`ProfileStep#2` 明显比后续 steps 更宽 |

图 5 中可以看到，与其他 steps 相比，`ProfileStep#2` 花费了更多时间。仔细观察后，会发现 `matmul_add` annotation 也有类似模式。关键线索在 annotation 内部，而不是 annotation 本身：

| Step | `matmul_add` start | `aten::matmul` start | gap |
| :--: | :--: | :--: | :--: |
| #2 | 138.736 | 366.493 | 227.757 µs |
| #3 | 517.926 | 523.447 | 5.521 µs |
| #4 | 610.039 | 614.527 | 4.488 µs |

| ![profile step 2 中 record_function matmul_add 和 aten::matmul dispatch 之间的 228 微秒 gap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-227.png) |
| :--: |
| 图 6：`record_function("matmul_add")` 和 `aten::matmul` 之间约 228 µs 的 dead window |

图 6 中的约 228 µs，是进入 `record_function("matmul_add")` 到 PyTorch 实际 dispatch `aten::matmul` 之间的「dead window」。这可能由多种原因造成，包括 workspace allocations、[cuBLAS](https://developer.nvidia.com/cublas)（NVIDIA 专有的 GPU-accelerated fundamental linear algebra operations library）heuristics，或 lazy module loading。可以选择忽略它，也可以在正式 profile 之前运行 [更多 warmup steps](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L35-L39)（这是标准做法）。

从 profiling 角度看，warmup 指在真正 profiling 之前，把 events 先运行几次。GPU 预先完成的工作（包括前面提到的那些内容）通常是一次性开销，不希望把它们纳入 profile。在这个例子中，我们有两个 warmup stages：一个是在进入 profiler 之前实际循环执行函数，另一个是在 profiler 内部通过 `warmup` argument 实现。本节中，我们同时启用了实际 iterations 和 schedule。

```bash
uv run 01_matmul_add.py --warmup
```

[带 Warmup 的 64x64 Perfetto Trace](https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/ariG23498/traces/resolve/01_matmul_add/64_bf16_warm_eager.json)

| ![warmup steps 之后的 PyTorch profiler trace，ProfileStep#2 不再显示 cold-start overhead](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/warmup.png)|
| :--: |
| 图 7：warm up 之后，每个 profile step 的耗时相近 |

图 7 中，每个 profile step 花费的时间相近。但这并不意味着一次性 overheads 被 optimized 掉了。我们只是对 runs 做了 warm up，让这些 overheads 没有被 profile 到。如果就这样突然结束本节，而不提示如何进一步解决 launch overheads，对读者并不公平。因此可以阅读这个 [链接](https://pytorch.org/blog/accelerating-generative-ai-2/)，了解如何进一步 optimize launch overheads。

### 为什么 CPU lane 和 GPU lane 之间有约 2.5 ms 的 offset？

| ![PyTorch profiler trace 中 CPU lane 和 GPU lane 之间的 2.32 毫秒 offset](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-bw-kernel-launch.png) |
| :--: |
| 图 8：CPU lane 和 GPU lane 之间约 2.5 ms 的 offset |

图 8 显示，CPU lane 与 GPU lane 之间有大约 2.5 ms 的 offset：这是 CPU submit CUDA kernels 之后，到它们真正开始 execution 之间的 delay。直觉上可能会认为，warmup stage 加上 schedule 的 `wait` 和 `warmup` 应该能让 GPU 保持忙碌，并减少这个 offset。

为了弄清真正发生了什么，稍微修改 schedule：

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1)
```

| ![wait=0 warmup=0 时的 PyTorch profiler trace，在 steps 之间显示 Activity Buffer Request](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/full-profile.png) |
| :--: |
| 图 9：设置 `wait=0` 和 `warmup=0` 后，trace 暴露出 `Activity Buffer Request` |

图 9 显示，在任何 operation 之前，GPU lane 中有一个 `Activity Buffer Request`。进一步放大看。

| ![profiler buffer request 导致 matmul 和 add CUDA kernels 之间出现 gap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/mat-add-gap.png) |
| :--: |
| 图 10：profile step 1 中 matmul 和 add kernels 之间出现 gap |

放大 GPU trace 后可以看到，`ProfileStep#0` 的 matmul 和 add kernels（它的 CPU trace 在图中不可见）前后连续发生，而 `ProfileStep#1` 的 kernels 之间出现了一个 window。最合理的解释是 buffer overflow 发生了，并且在 kernel execution 过程中发起了另一次 buffer request（请求在 GPU VRAM 上分配一些 memory）。

排除其他可能性的最好方式，是 profile 更多 iterations，看看 trace 的其他部分是否也出现类似 window。为此，使用 `active=20` 运行。

| ![20 个 active iterations 的 PyTorch profiler trace，确认 buffer-request gap 只出现一次](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters.png) |
| :--: |
| 图 11：20 个 active steps 中 gap 只出现一次，确认它是 buffer request |

如图 11 所示，在 `ProfileStep#1` 中看到了类似趋势。这与前面的发现一致，因此可以安全地得出结论：它确实是另一次 buffer request。

### Event chain（事件链）

| ![PyTorch profiler 中嵌套的 CPU dispatch chain：ProfileStep、matmul_add、aten::matmul、aten::mm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cpu-nests.png) |
| :--: |
| 图 12：dispatch chain |

图 12 中可以看到 nested CPU calls。这是一个重要的 visualization，可以帮助理解 dispatch chain 实际长什么样。

它从 `ProfileStep#<id>` 开始，这一层封装 profiling step。由于我们 annotate 了 step，因此可以看到 `matmul_add` row。`matmul_add` 包含两个 `aten` calls：一个用于 matrix multiplication，一个用于 matrix addition。

`aten::matmul` 是面向用户的 PyTorch matmul calls 最终落到的 [ATen-level](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen) dispatch。`aten::mm` 是 2D matrix-matrix multiply backend。

值得注意的是，如果给 matrices 添加 batch axis，PyTorch 会调用 `aten::bmm`（batched matrix multiplication）。我们绕个小路，看看 `aten::bmm` 如何工作。

```diff
- x = torch.randn(args.size, args.size, device=device, dtype=dtype)
- w = torch.randn( args.size, args.size, device=device, dtype=dtype)
- b = torch.randn(args.size, args.size, device=device, dtype=dtype)

+ # adding a batch size of 8
+ x = torch.randn(8, args.size, args.size, device=device, dtype=dtype)
+ w = torch.randn(8, args.size, args.size, device=device, dtype=dtype)
+ b = torch.randn(8, args.size, args.size, device=device, dtype=dtype)

```

| ![PyTorch profiler trace 展示 aten::matmul 为 3D batched tensors dispatch aten::bmm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/bmm.png) |
| :--: |
| 图 13：Batched Matrix Multiplication |

图 13 中，给 inputs 添加 batch axis 后，`aten::matmul` 现在封装了一组其他前置 CUDA runtime calls，以及 `aten::bmm`（而不是 `aten::mm`）。这也提示了 cuBLAS 为 program dispatch 正确（最合适的）kernel 时需要执行的 heuristics。

> 后文除非特别说明，都使用简单的 2D matrices。

### 为什么 matmul 多了一个额外的 CUDA runtime call？

| ![CPU lane 显示 cudaOccupancyMaxActiveBlocksPerMultiprocessor 位于 matmul cudaLaunchKernel 之前](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cudaoccupancy.png) |
| :--: |
| 图 14：matmul kernel launch 之前触发了一次 CUDA occupancy query |

可以看到，对于 `aten::mm`，有两个 CUDA Runtime calls，分别是 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`（图 14 中方框标出）和 `cudaLaunchKernel`；而对于 `aten::add`，只有 `cudaLaunchKernel`。

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 是一个 planning call，并且完全发生在 CPU side。它提出的问题是：「给定一个 kernel function、选定的 block size，以及选定的 dynamic shared memory size，这个 kernel 有多少 blocks 可以同时驻留在一个 SM（Streaming Multiprocessor）上？」

这就带来一个问题：为什么 matmul 需要 planning，而 add 不需要？

为了理解这一点，需要看 kernel 的 resource footprint。点击 GPU kernels 后，可以检查对应 kernel 的 resource footprint。

| ![cuBLAS matmul kernel resource footprint：Perfetto 中的 registers、shared memory 和 block size](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/matmul-footprint.png) | ![elementwise add CUDA kernel resource footprint，包含 32 registers 和 zero shared memory](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/add-footprint.png) |
| :--: | :--: |
| 图 15：Matmul footprint | 图 16：Add footprint |

图 15 中可以看到，对于 matrix multiplication，`registers per thread` 和 `shared memory` 是动态的（取决于 matrix size）。cuBLAS 附带数百种 kernel variants，每一种都有 heuristic-driven launch path，需要 runtime information 来了解 hardware capacity。occupancy query 就是这种 heuristic 的一部分。概念上，可以把 GPU-accelerated matmuls 理解为 [在独立 tiles 上工作](https://alvinwan.com/how-to-tile-matrix-multiplication/)：使用多少 tiles、每个 tile 多大，取决于 matrices 和 hardware。现代 algorithms 远比这复杂，但这个 reference framework 仍然有用。

从图 16 可以看到，addition 的 footprint 是 32 registers 和 zero shared memory。这很容易满足。没有什么需要 query，因为不会有 hardware resource 限制 occupancy。这个 kernel 按设计就是 resource-light。

> [!NOTE]
> 阅读任何 trace 时，可以把它当作一个快速 diagnostic。扫描 CPU lane 中的 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`。每一次出现都标记了一个「heavyweight、adaptively launched」kernel，通常是 GEMM（GEneral Matrix Multiplication）、conv 或类似 kernel。没有前置 occupancy query 的 kernels，通常属于 elementwise/reduction 一类，PyTorch 会以更机械的方式 launch 它们。

### 为什么 cudaDeviceSynchronize 这么大（约 1.78 ms）？

`cudaDeviceSynchronize` 会阻塞 CPU，直到这个 device 上所有 GPU work 完成。profiler 会在 active window 结束时发出这个 sync，用来 flush events。没有它，kernel timings 会缺失。

一个 1.78 ms 的 sync 只覆盖了 26 µs 的真实 GPU work，这说明这次 run 有 98% 的时间处于 idle 状态。这是 textbook overhead-bound symptom。

## 4096x4096 traces

从前面的 profiler table analysis 已经知道，给 algorithm 提供更大的 matrices，可以让它从 overhead-bound region 转到 compute-bound。

运行命令并深入 trace：

```bash
uv run 01_matmul_add.py --size 4096 --warmup
```

### 为什么同一个 kernel 比其他 kernel 耗时更长？

| ![同一 GPU 上 4096x4096 bf16 matmul kernel timings 在 profiler steps 之间发生变化](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/kernel-time.png)|
| :--: |
| 图 17：即使 inputs 相同，一个 matmul kernel 仍比其他 kernels 运行更久 |

图 17 中可以看到，`ProfileStep#3` 的 matmul kernel 在 GPU 上比其他 steps 耗时更长。这一点尤其值得注意，因为 launch 的其他 kernels 完全相同，也就是说没有 cuBLAS heuristics 参与。没有 scheduling gaps，CPU launches 正常，也不是 profiler artifact。

图 17 的 trace 说明了一个很容易在理想化示例中忽略的事实：即使在相同 hardware environment 上运行相同 code 和相同 data，kernel runtimes 也不是常数。

为了更具体地说明，稍微修改 script。运行 iteration 20 次，并捕获每个 step。

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=20, repeat=1)

- for _ in range(5):
+ for _ in range(20):
```

| ![PyTorch profiler trace 展示 20 次 matmul iterations 中 kernel runtime variance](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters-kernels.png) |
| :--: |
| 图 18：20 次 iterations 中，同一个 matmul kernel 会以不同速度运行 |

图 18 展示了类似发现。虽然每个 kernel 完全相同，但它们耗时不同。不同 compute times 可能由一组原因造成：

- GPU clocks on idle and boost
- GPU heating
- GPU power management
- Driver side housekeeping

只看 average 的读者可能会得出结论：一个 matmul 花费约 1 ms（5 次的 mean = 1084 µs）。而看 trace 的读者会发现，matmul 通常只花约 580 µs，除非 GPU 偶尔表现异常。这是两种完全不同的 mental models，而只有后一种是正确的。

## 看看 torch compile 如何工作

使用 `torch.compile` 一直让我们感到惊讶。开发者写的是普通 eager PyTorch code，但 PyTorch 会尝试捕获 tensor-heavy regions，把它们转换成 graphs，optimize 它们，并运行生成的代码。默认 backend 通常是 `TorchInductor`，整体 pipeline 是：

1. `TorchDynamo` 将 Python execution 捕获为 FX graph
2. 涉及 gradients 时，`AOTAutograd` 准备 forward/backward graphs
3. `Inductor` 将 graph lower 成 optimized CPU 或 GPU code

本节讨论 compilation，并查看 profiler traces。

```bash
uv run 01_matmul_add.py --size 4096 --warmup --compile
```

`args.compile` flag 会触发以下代码：

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)

fn = torch.compile(fn) if args.compile else fn
```

| ![PyTorch profiler trace 中高亮的 torch.compile region，展示 TorchDynamo 和 Inductor frames](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/compilation-region.png) |
| :--: |
| 图 19：compiled regions 在 trace 中显示为 TorchDynamo 和 Inductor frames |

图 19 中可以看到名为 `Torch-Compiled Region: 0/0` 的新 CPU 行，它们指向正在使用的 compiled functions。

### matmul 和 add kernels 是否 fusion 成了一个？

| ![compiled trace 显示 aten::addmm 替代 eager aten::add 和 aten::mm pair](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/fused-ops.png) |
| :--: |
| 图 20：compiled run dispatch 一个 `aten::addmm` |

看图 20 时，自然会问：multiplication 和 addition operations 是否真的 fusion 成了一个？

这是 graph level 的 operator fusion。Inductor 将 `torch.add(torch.matmul(x, w), b)` 重写为单个 `aten::addmm(b, x, w)` call。需要注意的是，它并没有生成一个 **新的** fused CUDA kernel。实际 GPU work 仍然是 `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nn`，也就是 eager mode 使用的同一个 cuBLAS kernel。因此这里的「fusion」发生在 dispatcher level，而不是 kernel level。

> [!NOTE]
> PyTorch 提供了 [`torch.addmm`](https://docs.pytorch.org/docs/2.12/generated/torch.addmm.html) function，可以把这里分两步完成的 multiply 和 add 合成一步。建议读者查看这个 function 的 traces，并在下方评论区写下观察结果。

### torch.compile 的 runtime architecture

虽然理论上知道 compile 函数时会发生什么，但看到它实际运行同样重要。下面看 CPU-side hierarchy，它反映了 `torch.compile` 的 runtime architecture。

**TorchDynamo Cache Lookup** 是 Dynamo 检查当前 call 是否仍然匹配已经 compiled 的内容，包括相同 input shapes、dtypes、devices 和 tensor metadata。如果有任何不匹配，Dynamo 会 recompile。即使 compilation 已经完成，这个 cost 每次 call 都要支付。

**Torch-Compiled Region** 是「进入」compiled version 的 wrapper。**AOTDispatcher Runtime Wrapper Prologue** 是 AOT Autograd 的 runtime wrapper。虽然这里不需要 gradients，AOTDispatcher 仍然会出现在 stack 中，处理 tensor metadata、view tracking，并且如果 `requires_grad` 为 true，它还会设置 backward pass。

**## Call CompiledFxGraph <hash>** 是实际 generated code 运行的地方。「CompiledFxGraph」后面的 string 是 FX graph 的 content hash。在三个 active steps 中它都相同，确认发生了 cache hits。

> [!TIP]
> 可以在磁盘上的 `/tmp/torchinductor_<user>/fxgraph` 下根据这个 hash 找到 generated code。当需要阅读 Inductor 实际生成的 Triton/C++ 时，这很有用。

### CUDA launches 是否减少了一半？

| ![compiled matmul trace 显示每个 step launch Memcpy DtoD 和 GEMM kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/memcpy.png) |
| :--: |
| 图 21：每个 compiled step 仍然 launch 两个 GPU kernels：一次 Device-to-Device memcpy 和 GEMM |

观察图 21 的 traces 时，我们一开始很高兴地注意到每个 step 只有一个 `cudaLaunchKernel`。但这个观察直接与 GPU trace 中看到的情况矛盾。每个 step 仍然有两个 kernels 被 launch，分别是 `Memcpy DtoD (Device -> Device)` 和 GEMM。回到 CPU trace 后，才发现完全漏掉了 `cudaMemcpyAsync` dispatch。

`addmm` 计算 `out = α·A·B + β·C`，而 cuBLAS 的 GEMM-with-bias-add epilogue 会写入一个 destination buffer，这个 buffer 需要已经包含 bias。可以把 epilogue 理解为 GEMM _之后_ 发生的所有 operations。在 deep-learning 中，常见 GEMM-Epilogues 包括 activations、bias addition、normalization 等。这也是为什么存在 cuBLAS GEMM-with-<specific epilogue> kernel variants。

> 如果为 `torch.compile` 使用不同的 `mode`，会看到 launch 不同的 kernel variants。可以自己试一试，并在下方评论区补充观察结果。

因此，Inductor 生成的代码会执行：

- `out = copy(C)` ← 这是 DtoD memcpy（32 MB，耗时约 33 µs）
- `out = α·(A·B) + β·out` ← 使用 `α=β=1` 的 GEMM，将 bias add fusion 到 writeback 中

结果在数学上仍然相同。bias add 并不是免费的，因为需要先支付一次 memcpy，再支付一个略微更昂贵的 GEMM epilogue。

原本可能期待的 fusion，是让 `x·w + b`（这里是 `out = α·A·B + β·C`）折叠成单个 kernel，并且没有额外 memory traffic。但实际没有发生。Inductor 保留了两个会 touch memory 的 operations，只是把 bias copy 重新标记为 memcpy，把 addition 标记为 GEMM epilogue。

真正融合的实现会跳过 memcpy。这正是 FlashAttention-style hand-written kernels 会做的事，也是 Inductor 可以通过 Triton codegen 做的事。但对于 `4096×4096 bf16 matmul`，Inductor 显然判断「使用 cuBLAS，并通过 epilogue setup 处理 bias」是最佳路径。

### CPU overhead 上升了，而不是下降

比较 eager 和 compiled run 时，最容易漏掉的是这一点：

| step | eager dur (ms) | compile dur (ms) |
| :--: | :--: | :--: |
| #2 | 0.1 | 0.2 |
| #3 | 0.07 | 0.1 |
| #4 | 0.07 | 0.1 |

Compile 每个 step 在 CPU 上大约贵 2 倍。原因是每次 call 都要走完整的 Dynamo > AOTAutograd > Inductor stack，同时还要执行本来就需要的 `aten::addmm` dispatch。compile pipeline 是为包含 dozens of ops 的 ML models 构建的，在那种情况下 per-call overhead 可以被 amortize；对单个 op 来说，它就是额外税负。

> [!TIP]
> `torch.compile` 有一个 `mode` argument。可以把它当作课后作业：阅读 documentation，找出一个可能降低 CPU overhead 的 `mode`。🤗

## Trace reading cheatsheet

下面是本文涉及模式的 quick reference。核心想法是：如果在 trace 中看到某个模式，通常意味着下面这些事情。

### Profiler table

| 看到的现象 | 通常意味着什么 |
| :-- | :-- |
| `Self CPU time total` ≫ `Self CUDA time total`（CPU 为 ms，GPU 为 µs） | Overhead-bound。CPU 花在 dispatch 上的时间超过 GPU 花在 computing 上的时间。让 work 更大（更大的 matrices、batched ops），或者 fuse calls。 |
| `Self CPU time total` ≈ `Self CUDA time total`，两者都是 ms | Compute-bound。GPU 是 bottleneck，这通常是期望看到的状态。 |
| 一个 event 主导 `CUDA total` | 这就是 hotspot。optimization 从这里开始。 |
| 一个 event 的 `# of Calls` 很大 | 即使每次 call 都便宜，也可能是潜在 bottleneck。检查是否可以 fuse 或 batch。 |
| 某一行的 `CPU total` ≫ `Self CPU` | 大部分 cost 在 children 中。应该 drill into nested events，而不是只看 parent。 |

### CPU lane

| 看到的现象 | 通常意味着什么 |
| :-- | :-- |
| 第一个 `ProfileStep` 比其他 steps 宽得多 | Cold-start overhead：workspace allocation、cuBLAS heuristics、lazy module loading。添加 warmup iterations，和/或使用 schedule 的 `warmup` argument。 |
| `record_function("...")` start 与内部第一个 `aten::*` 之间有大 gap | 同样是 cold-start tax，只是放大来看。annotation 已经进入，但 dispatch 还没发生。 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 出现在 `cudaLaunchKernel` 之前 | 一个 heavyweight、adaptively-launched kernel（GEMM、conv 等）。cuBLAS 正在询问 driver，一个 SM 上可以放下多少 blocks，以便选择 kernel variant。 |
| `cudaLaunchKernel` 前面没有 occupancy query | 一个 elementwise 或 reduction kernel，resource footprint 固定且很轻。没有什么需要 plan。 |
| active window 末尾有很长的 `cudaDeviceSynchronize` | profiler 正在 flush events。它的 duration 主要是 GPU 完成 pending work，而不是真实 CPU cost。sync 覆盖的 GPU work 很小，是典型 overhead-bound symptom。 |
| 出现没有手写的 `cudaMemcpyAsync` | 通常是隐藏的 Device-to-Device copy。常见于 `addmm` 在 GEMM epilogue 之前用 bias 初始化 destination buffer。 |

### GPU lane

| 看到的现象 | 通常意味着什么 |
| :-- | :-- |
| GPU lane 上出现 `Activity Buffer Request` | profiler 正在 allocate/refill 自己的 event buffer。第一次通常解释了初始 CPU↔GPU lane offset。 |
| 单个 step 中两个 kernels 之间有 gap | 很可能是 mid-execution 的另一次 buffer request。通过运行更多 iterations 确认：如果只出现一次，那是 profiler，而不是代码。 |
| 同一个 kernel 在不同 steps 中 timing 不同 | GPU clocks、thermals、power management、driver housekeeping。要读 trace，不要只看 mean。 |
| kernel 名称类似 `ampere_bf16_s16816gemm_...` | 这是 matmul 对应的实际 cuBLAS GPU work。对相同 shapes/dtypes，eager 和 compiled mode 中的 kernel name 通常相同。 |
| GEMM 前紧跟 `Memcpy DtoD` | 这是 `addmm` epilogue 的 bias copy。「fusion」发生在 dispatcher level，而不是 kernel 中。 |

### Dispatch chain

| 看到的现象 | 通常意味着什么 |
| :-- | :-- |
| `ProfileStep#N` → `<record_function name>` → `aten::*` → `aten::mm` / `aten::bmm` / `aten::add` | 典型 nested call hierarchy。Self time 不包括 children；Total time 包括 children。 |
| `aten::matmul` resolve 到 `aten::mm` | 2D × 2D matrix multiply。 |
| `aten::matmul` resolve 到 `aten::bmm`（并带有额外 CUDA runtime calls） | 3D+ tensors 上的 batched matmul。cuBLAS 会做更多 heuristic work 来选择 variant。 |
| `aten::addmm(b, x, w)` 替代单独的 `aten::add` + `aten::mm` pair | dispatcher level 的 operator fusion。GPU kernel 仍然是同一个 GEMM，只是把 bias add folded into epilogue。 |

### torch.compile

| 看到的现象 | 通常意味着什么 |
| :-- | :-- |
| CPU lane 中出现 `Torch-Compiled Region: K/M` row | 已经进入 compiled function。 |
| 每个 step 都有 `TorchDynamo Cache Lookup` | Dynamo 正在验证 shapes/dtypes/devices 是否匹配 cached compile。即使 compilation 之后，每次 call 也要支付这部分 cost。 |
| 没有 grads 时仍出现 `AOTDispatcher Runtime Wrapper Prologue` | AOTAutograd 的 runtime wrapper 总是在 stack 中，负责处理 tensor metadata 和 view tracking。 |
| `## Call CompiledFxGraph <hash>` 在多个 steps 中 hash 相同 | 命中 generated code cache。generated source 位于 `/tmp/torchinductor_<user>/fxgraph/<hash>`。 |
| 对一个很小的 op，`torch.compile` 下 per-step CPU time 高于 eager | 符合预期。Dynamo → AOTAutograd → Inductor stack 是额外开销，只有在 many ops 上才容易 amortize。 |

## 结论

我们从一个很小的 `matmul + add` 开始，把它作为学习阅读 PyTorch profiler 的入口。过程中建立了几个可以迁移到更大 workloads 的 mental models。这是 **Profiling PyTorch** 系列的第一站。后续文章会逐步离开这个 two-op toy，沿着复杂度阶梯向上，观察更大的 building blocks，并最终进入真实 models。

感谢 [Noe Flandre](https://huggingface.co/NoeFlandre)、[Suvaditya Mukherjee](https://huggingface.co/suvadityamuk) 和 [Vidit Ostwal](https://huggingface.co/ViditOstwal) 对本文早期 draft 的 review。
