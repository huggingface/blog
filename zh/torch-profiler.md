---
title: "PyTorch 性能剖析（第 1 部分）：torch.profiler 入门指南"
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
# PyTorch 性能剖析（第 1 部分）：torch.profiler 入门指南

![博客文章缩略图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/thumbnail.png)

> *无法进行性能剖析，就无法优化。*

无论目标是让大语言模型（LLM）每秒生成更多词元（token）、把推理耗时缩短几毫秒，还是弄清楚训练循环为什么比规格表承诺的速度慢，最终都绕不开性能剖析。

难点在于，性能剖析的入门门槛很高。性能轨迹像一面密密麻麻的彩色矩形墙，各类事件名称也常常让人望而生畏。大多数教程默认读者已经知道如何阅读轨迹。因此，即便已经知道应该做性能剖析，打开一份轨迹也常常像是一件可以留到以后、或交给别人处理的麻烦事。本文以及这个系列，正是为了降低这道门槛。

本文是《PyTorch 性能剖析》系列的开篇文章。本系列会循序渐进地介绍如何阅读性能轨迹，并把这些观察用于指导优化。计划如下：

1. **第 1 部分（本文）：** 从最简单的操作开始：一次矩阵乘法后接偏置加法，学习如何阅读 `torch.profiler` 返回的结果。
2. **第 2 部分：** 扩展到 `nn.Linear` 和一个小型多层感知机（MLP），借助轨迹数据找准优化方向，并初步查看底层 CUDA 内核。
3. **第 3 部分：** 结合 `transformers` 库，在大语言模型上综合运用前述全部知识。

本文从初学者视角记录这段过程。除基础 PyTorch 知识外，不需要其他前置知识；可以把它当作一篇轻松阅读的教程，过程中会穿插一些「原来如此」的瞬间。文章采用**问题导向**结构：打开一份轨迹，提出「等等，为什么会这样？」这样的问题，再顺着线索追到答案。读完本文后，应能理解以下内容：

- 如何设置 `torch.profiler`，以及该工具的输出数据含义；
- 读懂性能统计表与性能轨迹（CPU 泳道、GPU 泳道，以及两者间反常的空闲间隔）；
- 从 Python 接口调用，逐层下沉至 CUDA 内核执行的完整事件链路；
- 启用 `torch.compile` 后，哪些性能指标会变化，更关键的是：**哪些指标保持不变**。

开始之前，先明确两个关键定义，方便后续阅读：

1. **GPU 内核（kernel）**：能够依托 GPU 海量线程并行运行的程序单元。
2. **CPU 负责调度并启动各类 GPU 内核**。

日常开发一般无需手动编写 GPU 内核：调用 PyTorch 算子时，框架会自动将运算转化为一个或多个 GPU 内核，在显卡上完成计算。

掌握以上两个概念，我们就开始逐一探究问题。

> [!NOTE]
> 本文使用的完整脚本在这里：[`01_matmul_add.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py)。建议在单独的标签页中打开脚本，并逐步阅读代码。本文使用 `NVIDIA A100-SXM4-80GB` GPU 运行这些脚本。

## 矩阵乘法与加法运算

正如 [Sara Hooker 博士的准确调侃](https://youtu.be/7knwihgj0fU?si=uvzGH-J9bsCHP4Nn&t=2199)：人主要由水构成，而深度神经网络主要由矩阵乘法构成。矩阵乘法是深度学习的基石，用它作为性能剖析之旅的起点再合适不过。

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)
```

> 矩阵乘法搭配偏置加法，模拟了神经元中权重与偏置的交互方式。顺带一提，这里的加法也会帮助我们理解 [后文](#lets-see-some-torch-compile-at-work) 中的编译机制。

我们将借助 `torch.profiler` 模块完成性能剖析，整体分为四步：

1. 准备好 [需要剖析的代码](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L26-L27)（这里是 `def fn`，它封装了矩阵乘法和偏置加法）。
2. [标注算法](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L32)。这个步骤完全可选，但建议执行。`record_function` 会把函数标注为 `matmul_add`，后续在性能轨迹中可以快速定位对应模块。

```py
def step():
  with torch.profiler.record_function("matmul_add"):
    return fn(x, w, b)
```

3. 用 `torch.profiler.profile` [上下文管理器](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L53-L62) 包裹代码。

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

4. 导出 [剖析结果](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L70)。

```py
# the profiler table
prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)

# the profiler trace
prof.export_chrome_trace(trace_path)
```

剖析器会导出两类不同产物：

1. 性能统计表：提供算法的统计摘要，用来回答「什么耗时最多」。这对定位性能热点很有帮助。热点可能是最耗时的事件，也可能是流水线瓶颈，或是触发次数特别多的事件。
2. 性能轨迹：提供时间维度上的执行视图，用来回答「某个操作何时发生、为什么发生」，并展示 CPU 与 GPU 上发生的活动。当需要调查启动了哪些内核、内核启动是否延迟、CPU 与 GPU 活动是否重叠时，它很有用。

现在通过第一次执行看看这两者如何工作。（[完整源码 `01_matmul_add.py` 在这里](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py)）

> [!NOTE]
> 建议在搭载 GPU 的设备上运行该脚本。

```bash
uv run 01_matmul_add.py --size 64
```

在搭载 GPU 的机器上运行上述脚本后，会生成 `traces/01_matmul_add` 文件夹，里面包含两类分析结果文件：

```bash
64_bf16_cold_eager.json
64_bf16_cold_eager.txt
```


| ![64×64 矩阵上 matmul add 的性能统计表](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profile-table-64.png) |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                图 1：64×64 矩阵乘加运算的性能统计表                                                                |

`.txt` 文件存放性能剖析汇总表。打开文件后（如图 1），可以看到一张大型表格：首列是剖析区间内执行的各类事件名称。

其余各列记录对应事件在 CPU、GPU，或 `torch.profiler.profile` 的 `activities` 参数指定设备上的耗时。重点观察耗时占比最高的事件，并结合业务逻辑判断该耗时是否合理；同时也要关注 `# of Calls` 这一列，它表示对应事件的触发频次。

顺便说明一下自身耗时「Self CPU/CUDA」与总耗时「CPU/CUDA total」的区别。「Self」列只统计事件自身花费的时间，**不含内部子事件耗时**。「total」列则包含事件自身及其所有嵌套子事件的耗时总和。因此，查看 `matmul_add` 的「CPU total」时，它包含自身耗时，以及它触发的所有内部子算子的 CPU 耗时。这是一个需要注意的重要细节。

如果查看表格最后两行，会发现剖析器给出：

```bash
Self CPU time total: 2.314ms
Self CUDA time total: 23.104us
```

CPU 耗时单位是 `ms`，而 GPU 耗时单位是 `us`。为了建立直观感受：GPU 上花费的时间（内核 `ampere_bf16_s16816gemm...`）不到 CPU 上花费时间（`matmul_add` 操作）的 1%。GPU 大部分时间处于空闲状态，这是一个明显的警示信号。出现这种情况的原因是，小规模矩阵乘法在 GPU 上运算极快，程序绝大部分开销都消耗在内核准备、GPU 任务启动、发送待乘数据和收集结果等环节。这类受调度开销制约的算法被称作 _开销受限型（overhead-bound）_ 算法。

脱离这种状态最简单的方法，是**改用更大尺寸的矩阵执行乘法运算**。

```bash
uv run 01_matmul_add.py --size 4096
```


| ![4096×4096 矩阵上 matmul add 算法的性能统计表](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profiler-table-4096.png) |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                  图 2：4096×4096 矩阵上 matmul add 的性能统计表                                                                  |

图 2 最后两行是：

```bash
Self CPU time total: 4.908ms
Self CUDA time total: 4.495ms
```

两项耗时单位均为毫秒，可见仅通过增大矩阵尺寸，就能让 GPU 承担更多实际计算。对照图 2 能发现：如今绝大部分 CUDA 耗时消耗在 GPU 内核（`ampere_bf16_s16816gemm_..`）上，而非发起内核调用的 CPU 操作（`matmul_add`）。这说明程序成功从**开销受限型（overhead-bound）**转为**计算受限型（compute-bound）**。

接下来通过 `.json` 结果文件可视化任务下发调用链。可以将 JSON 文件上传至 [Perfetto UI](https://ui.perfetto.dev) 查看性能轨迹，也可以执行命令 `uvx trace-util traces -b traces` 直接生成 Perfetto 访问链接。

## 64×64 性能轨迹


| ![CUDA GPU 上 64×64 bf16 矩阵乘法后接加法的 PyTorch 性能轨迹](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/64-matmul-add.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                      图 3：64×64 矩阵上 matmul 和 add 的性能轨迹                                                                      |

在图 3 中，我们能看到矩阵乘加运算的性能剖析时序轨迹。矩形条的宽度代表事件持续时长，纵向嵌套结构代表调用层级；CPU 泳道展示 CPU 侧发生的各类事件，GPU 泳道则呈现 GPU 内核的实际执行过程。图中空白区间代表等待或硬件空闲时段。

脚本采用如下默认配置运行：

* 矩阵尺寸 64：输入、权重与偏置张量形状均为 (64, 64)
* 数据类型 bf16：采用 bfloat16（16 位脑浮点）精度
* 未开启编译：没有编译 `torch` 算子
* 无预热：性能采集前未做 GPU 预热

> 使用 Perfetto 工具时，推荐快捷键操作来快速浏览时序图：W/A/S/D 按键控制视图移动缩放。


| ![Perfetto 中并排标注 CPU 泳道和 GPU 泳道的 PyTorch 性能轨迹](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gpu-cpu-trace.png) |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                   图 4：PyTorch 性能轨迹中的 CPU 泳道和 GPU 泳道                                                                   |

图 4 包含两条时序泳道，分别对应 CPU 行为与 GPU 行为。CPU 泳道里能看到三段剖析采集周期（从 `ProfilerStep#2` 开始），该配置由采样调度器决定：

```py
schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
```

`wait` 参数：跳过干扰数据较多的初始化阶段（对应 `ProfilerStep#0`），

`warmup` 参数：程序预热运行，但剖析器不采集日志（对应 `ProfilerStep#1`），

`active` 参数：正式采集、最终体现在时序图中的运行轮次。调度配置可以在 [这段脚本](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L58) 中找到。

接下来围绕时序图展开排查。

### 为什么 ProfilerStep#2 耗时这么长？


| ![PyTorch 性能轨迹中 ProfileStep#2 看起来比 ProfileStep#3 和 ProfileStep#4 更宽](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/why-is-step-2-big.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                             图 5：`ProfileStep#2` 的矩形宽度显著大于后续轮次                                                                             |

图 5 中可以看到，`ProfileStep#2` 耗时远超后续轮次。仔细观察后，会发现自定义标记 `matmul_add` 也呈现相同规律。**问题根源藏在标记内部的子事件中，而非标记本身**。


| Step | `matmul_add` start | `aten::matmul` start |     gap     |
| :--: | :----------------: | :------------------: | :---------: |
|  #2  |      138.736      |       366.493       | 227.757 µs |
|  #3  |      517.926      |       523.447       |  5.521 µs  |
|  #4  |      610.039      |       614.527       |  4.488 µs  |


| ![ProfileStep#2 中 record_function matmul_add 和 aten::matmul 下发之间存在 228 微秒间隔](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-227.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                            图 6：`record_function("matmul_add")` 和 `aten::matmul` 之间约 228 微秒的空窗期                                                            |

图 6 中标注的约 228 微秒，是从进入 `record_function("matmul_add")` 代码块，到 PyTorch 真正下发 `aten::matmul` 算子之间的**空窗期**。造成这段耗时的诱因有很多：临时显存工作区申请、[cuBLAS](https://developer.nvidia.com/cublas)（NVIDIA 专有的 GPU 加速基础线性代数库）内部启发式策略开销，或是动态懒加载模块。对此可以先暂时跳过，也可以遵循通用做法：[在性能采集前补充多轮预热运行](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L35-L39)。

在性能剖析场景里，预热指正式采样前预先反复执行目标算子。上文提到的各类 GPU 初始化工作仅会执行一次，不应把这类一次性开销纳入剖析统计。本示例包含两级预热：一是开启剖析器之前手动循环运行函数；二是依托剖析器配置里的 `warmup` 参数，在剖析周期内部完成预热。本段代码已启用循环运行和采样调度配置。

```bash
uv run 01_matmul_add.py --warmup
```

[经过预热、64×64 矩阵的 Perfetto 性能轨迹](https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/ariG23498/traces/resolve/01_matmul_add/64_bf16_warm_eager.json)


| ![预热步骤之后的 PyTorch 性能轨迹，ProfileStep#2 不再显示冷启动开销](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/warmup.png) |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                               图 7：完成预热后，所有剖析轮次耗时趋于一致                                                                               |

从图 7 能够看出，各轮剖析任务耗时基本持平，但这不代表一次性初始化开销已经被优化掉；预热只是让这类开销避开了性能采样。关于如何进一步缩减内核启动开销，可以继续阅读这篇 [文章](https://pytorch.org/blog/accelerating-generative-ai-2/)。

### 为什么 CPU 时序泳道和 GPU 时序泳道存在约 2.5 毫秒的时间差？


| ![PyTorch 性能轨迹中 CPU 泳道和 GPU 泳道之间存在 2.32 毫秒偏移](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-bw-kernel-launch.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                            图 8：CPU 与 GPU 泳道之间存在约 2.5 ms 启动延迟                                                                            |

图 8 显示 CPU 和 GPU 泳道之间约有 2.5 ms 的偏移：CPU 提交 CUDA 内核之后，GPU 真正开始执行之前会有一段延迟。直觉上，预热阶段加上调度器的 `wait` 和 `warmup` 应该能让 GPU 保持忙碌，并缩小这个偏移。

为探明真实原因，我们小幅修改采样调度参数：

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1)
```


| ![wait=0 warmup=0 时的 PyTorch 性能轨迹，在步骤之间显示 Activity Buffer Request](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/full-profile.png) |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                 图 9：设置 `wait=0` 和 `warmup=0` 后，轨迹中出现 Activity Buffer Request 事件                                                                 |

图 9 显示，所有 GPU 运算开始前，GPU 泳道里出现了一项 `Activity Buffer Request`（活动缓冲区申请）事件，接下来我们放大视图细看细节。


| ![剖析器缓冲区申请导致 matmul 和 add CUDA 内核之间出现间隔](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/mat-add-gap.png) |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                             图 10：第 1 轮采样周期中，矩阵乘内核与加法内核之间出现一段时间空隙                                                             |

放大 GPU 时序轨迹后能够发现：`ProfileStep#0` 对应的矩阵乘法和加法内核是连续串行执行的（该轮 CPU 时序未在图中展示），而 `ProfileStep#1` 的两个内核中间多出一段空窗。最合理的解释是**缓冲区溢出**，在内核运行途中触发了一次新的缓冲区申请（向 GPU 显存申请分配内存）。

想要排除其他诱因，最稳妥的办法是增加采样迭代次数进行观察：修改参数 `active=20`，重新采集性能数据。


| ![20 个 active 迭代的 PyTorch 性能轨迹，确认缓冲区申请导致的间隔只出现一次](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters.png) |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                      图 11：20 轮有效采样结果，空隙仅单次出现，证实由缓冲区申请导致                                                                      |

如图 11 所示，仅在 `ProfileStep#1` 出现该空隙，和之前推断吻合，因此可以确定空档源自缓冲区内存申请。

### 事件链


| ![PyTorch 剖析器中嵌套的 CPU 下发调用链：ProfileStep、matmul_add、aten::matmul、aten::mm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cpu-nests.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                       图 12：PyTorch 剖析图中嵌套式 CPU 下发调用链：ProfileStep → matmul\_add → aten::matmul → aten::mm                                                       |

图 12 展示了层层嵌套的 CPU 调用栈，是理解算子下发全链路的关键视图。

最外层的 `ProfileStep#编号` 包裹整轮性能采样逻辑；因为代码里手动添加了标注，视图里出现 `matmul_add` 节点，其内部包含两次 ATen 算子调用：矩阵乘法和矩阵加法。

面向用户的 PyTorch 矩阵乘法调用最终会落到 [ATen 层](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen) 的 `aten::matmul` 调度接口。`aten::mm` 是专用于二维矩阵相乘的底层后端算子。

补充一个有趣细节：若给张量新增批量维度（batch 轴），PyTorch 会自动选用批量矩阵乘算子 `aten::bmm`。下面顺带切换样例，观察 `aten::bmm` 的实际调用表现。

```diff
- x = torch.randn(args.size, args.size, device=device, dtype=dtype)
- w = torch.randn( args.size, args.size, device=device, dtype=dtype)
- b = torch.randn(args.size, args.size, device=device, dtype=dtype)

+ # adding a batch size of 8
+ x = torch.randn(8, args.size, args.size, device=device, dtype=dtype)
+ w = torch.randn(8, args.size, args.size, device=device, dtype=dtype)
+ b = torch.randn(8, args.size, args.size, device=device, dtype=dtype)

```


| ![PyTorch 性能轨迹显示 aten::matmul 针对 3D 批量张量下发 aten::bmm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/bmm.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                 图 13：批量矩阵乘法（BMM 运算）                                                                                 |

在图 13 中，给输入张量新增批量维度后，`aten::matmul` 内部不再调用 `aten::mm`，转而使用 `aten::bmm`，同时还附带一系列前置 CUDA 运行时调用。这也侧面体现出 cuBLAS 会依靠内部启发式逻辑，为当前运算择优调度最合适的 GPU 内核。

> 后文若无特殊说明，我们均采用普通二维矩阵开展实验。

### 为什么矩阵乘法会多出一项 CUDA 运行时调用？


| ![CPU 泳道显示 cudaOccupancyMaxActiveBlocksPerMultiprocessor 位于 matmul cudaLaunchKernel 之前](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cudaoccupancy.png) |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                  图 14：矩阵乘法内核启动前执行了 cudaOccupancyMaxActiveBlocksPerMultiprocessor 查询                                                                  |

可以观察到：`aten::mm` 对应两次 CUDA Runtime 调用，分别是图中标注的 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 与 `cudaLaunchKernel`；而 `aten::add` 仅存在 `cudaLaunchKernel` 一次调用。

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 是 CPU 侧的资源预规划接口，作用为：**给定内核函数、线程块尺寸与动态共享内存大小，查询单个流式多处理器（SM）最多能同时容纳多少个该内核的线程块**。

由此产生疑问：为何矩阵乘法需要资源预规划，加法运算却不需要？

答案取决于内核的硬件资源占用。点击时序图里的 GPU 内核条目，即可查看对应内核的资源开销详情。


| ![cuBLAS 矩阵乘内核资源占用：Perfetto 中的寄存器、共享内存和线程块尺寸](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/matmul-footprint.png) | ![逐元素加法 CUDA 内核资源占用：32 个寄存器和零共享内存](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/add-footprint.png) |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                            图 15：矩阵乘 cuBLAS 内核资源占用（寄存器、共享内存、线程块尺寸）                                                                            |                                                                               图 16：逐元素加法内核资源：32 个寄存器、无共享内存                                                                               |

从图 15 可见，矩阵乘法的**单线程寄存器占用、共享内存占用**会随矩阵尺寸动态变化。cuBLAS 内置上百种不同版本的矩阵乘内核，每种内核都依托启发式策略完成启动，而该策略需要实时获取硬件算力信息，占用率查询就是启发逻辑的一环。从原理上理解，[GPU 矩阵乘法基于分块（tile）计算](https://alvinwan.com/how-to-tile-matrix-multiplication/)：分块数量、单块大小由输入矩阵规格与硬件参数共同决定；现代优化算法细节虽更加复杂，但这套分块思路仍是很好的理解框架。

反观图 16，加法内核固定占用 32 个寄存器、不使用共享内存，资源需求极小，硬件资源不会成为运行瓶颈，因此无需提前查询占用率。这类算子天生轻量化。

> [!NOTE]
> 该规律可作为性能排查快捷手段：浏览 CPU 泳道，只要出现 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 调用，就代表后方是**资源开销大、自适应调度启动**的内核，常见于通用矩阵乘 GEMM、卷积等算子；前面无占用率查询的内核，大多是逐元素运算、规约类算子，由 PyTorch 按固定逻辑直接发起调度。

### cudaDeviceSynchronize 耗时为何高达约 1.78 毫秒？

`cudaDeviceSynchronize` 会阻塞 CPU 线程，直至当前 GPU 上所有计算任务全部执行完毕。剖析器会在有效采样窗口末尾执行该同步操作，目的是刷新缓存的性能事件，缺少这一步就会丢失内核耗时统计数据。

一次仅 26 微秒 GPU 有效运算，却伴随 1.78 毫秒同步耗时，说明程序空闲占比高达 98%，是典型的**开销受限型**性能特征。

## 4096×4096 性能轨迹

前文通过性能统计表分析已知：增大矩阵尺寸后，算法会从开销受限转为**计算受限**。

执行下述命令，深入分析性能轨迹：

```bash
uv run 01_matmul_add.py --size 4096 --warmup
```

### 相同内核，单次运行耗时为何参差不齐？


| ![同一 GPU 上 4096×4096 bf16 matmul 内核耗时在多个性能采样步骤之间发生变化](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/kernel-time.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                 图 17：同一张显卡上，多轮采样里 4096×4096 BF16 矩阵乘内核耗时不一致                                                                 |

从图 17 可见，`ProfileStep#3` 对应的矩阵乘 GPU 内核运行耗时显著高于其他轮次。值得留意的是，各轮次输入完全一致，cuBLAS 启发式适配逻辑不会带来差异；CPU 任务下发时序无空隙，也并非剖析工具本身造成的数据失真。

该轨迹点明一个理想化示例里容易忽略的关键点：**即便硬件环境、代码、输入数据完全相同，GPU 内核的实际运行耗时也并非固定值**。

微调脚本参数，循环运行 20 次并采集每个步骤：

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=20, repeat=1)

- for _ in range(5):
+ for _ in range(20):
```


| ![PyTorch 性能轨迹展示 20 次 matmul 迭代中的内核运行耗时波动](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters-kernels.png) |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                        图 18：20 轮迭代下，同一份矩阵乘内核呈现明显的运行耗时波动                                                                        |

图 18 同样验证了上述现象：内核代码完全一致，但运算耗时各不相同。造成耗时波动的常见诱因：

* GPU 空闲或加速状态下的时钟频率变化
* 显卡温升带来的性能波动
* 显卡功耗调度策略
* 显卡驱动后台例行维护任务

只看平均耗时的开发者会得出结论：矩阵乘法平均耗时约 1 毫秒（5 次均值为 1084 微秒）；而查看时序轨迹后能发现：除个别硬件扰动时刻外，矩阵乘法大多只需约 580 微秒。两种结论对应的优化思路天差地别，后者才符合真实运行情况。

<a id="lets-see-some-torch-compile-at-work"></a>

## 看看 torch.compile 的实际表现

使用 `torch.compile` 一直让我们感到惊喜。开发者编写普通 eager PyTorch 代码，PyTorch 会尝试捕获张量密集区域，将其转为计算图、优化，并运行生成代码。默认后端通常是 `TorchInductor`，整体编译链路如下：

1. `TorchDynamo`：抓取 Python 执行过程，转化为 FX 计算图
2. `AOTAutograd`：涉及梯度计算时，构建前向、反向传播计算图
3. `Inductor`：将上层计算图下沉编译为优化后的 CPU/GPU 代码

本节结合性能剖析轨迹，讲解编译优化细节：

```bash
uv run 01_matmul_add.py --size 4096 --warmup --compile
```

开启编译参数 `args.compile` 后，执行逻辑变更如下：

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)

fn = torch.compile(fn) if args.compile else fn
```


| ![PyTorch 性能轨迹中高亮的 torch.compile 区域，展示 TorchDynamo 和 Inductor 调用帧](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/compilation-region.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                      图 19：性能轨迹中标注出 Torch 编译区域，包含 TorchDynamo 与 Inductor 调用帧                                                                      |

图 19 中可以看到名为 `Torch-Compiled Region: 0/0` 的条目，代表当前正在使用编译后的函数。

### 矩阵乘与加法算子是否融合为单个 GPU 内核？


| ![编译后性能轨迹显示 aten::addmm 替代 eager 模式下的 aten::add 和 aten::mm 组合](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/fused-ops.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                      图 20：编译运行后，原先分开的 `aten::mm`、`aten::add` 被替换为单个 `aten::addmm` 算子                                                      |

从图 20 可以提出一个问题：乘法与加法是否真正融合为单次算子运算？

这属于**计算图层面的算子融合**：Inductor 将 `torch.add(torch.matmul(x, w), b)` 改写为单次 `aten::addmm(b, x, w)` 调用。关键细节是：**该优化并未生成全新的融合 CUDA 内核**。GPU 底层仍沿用动态执行模式使用的同一个 cuBLAS 内核 `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nn`。换言之，本次融合仅发生在算子调度层，而非 GPU 内核层。

> [!NOTE]
> PyTorch 原生接口 [`torch.addmm`](https://docs.pytorch.org/docs/2.12/generated/torch.addmm.html) 本身就整合了矩阵相乘和偏置相加两步运算。可以采集这个函数的性能轨迹，并在评论区分享观察结果。

### torch.compile 的运行时架构

理论上理解函数编译会发生什么还不够，同样重要的是在轨迹里看到它实际如何运行。下面看一下反映 `torch.compile` 运行时架构的 CPU 侧层级：

**TorchDynamo Cache Lookup**：Dynamo 在这里校验当前调用是否仍匹配已编译版本，包括输入形状、数据类型、设备和张量元数据。如果存在不匹配，Dynamo 会重新编译；这项查询开销每次调用都会产生，即使编译完成后也无法省略。

**Torch-Compiled Region**：进入编译后版本的包装入口。

**AOTDispatcher Runtime Wrapper Prologue**：AOTAutograd 的运行时包装器。即便本示例不需要梯度，AOTDispatcher 也始终位于调用栈中，负责张量元数据管理和视图追踪；如果 `requires_grad` 为真，它还会设置反向传播过程。

**## Call CompiledFxGraph <hash>**：实际生成的代码在这里运行。「CompiledFxGraph」后面的字符串是 FX 计算图的内容哈希。三轮有效采样中的哈希一致，说明命中了编译缓存。

> [!TIP]
> 可通过哈希值在 `/tmp/torchinductor_<user>/fxgraph` 路径下找到 Inductor 实际生成的 Triton/C++ 源码，便于进一步调试内核。

### CUDA 内核启动次数是否减半？


| ![编译后 matmul 性能轨迹显示每个步骤都会启动 Memcpy DtoD 和 GEMM 内核](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/memcpy.png) |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                           图 21：编译后每轮仍启动两个 GPU 内核：设备间内存拷贝 + GEMM 矩阵乘                                                           |

初看图 21 的 CPU 轨迹时，我们一度高兴地以为每轮只有一次 `cudaLaunchKernel` 调用，但这与 GPU 轨迹直接矛盾：每轮依然启动了两个内核，分别是 **设备到设备内存拷贝（Memcpy DtoD）和 GEMM**。回看 CPU 调用栈后可以发现，此前完全遗漏了 `cudaMemcpyAsync` 下发。

`addmm` 计算的是 `out = α·A·B + β·C`。cuBLAS 带偏置加法收尾阶段的 GEMM 会写入目标缓冲区，而这个目标缓冲区需要预先包含偏置。**运算收尾（epilogue）**可以理解为 GEMM 主计算之后发生的所有附加操作。在深度学习中，激活函数、偏置相加、归一化等都是常见的 GEMM 收尾逻辑，因此 cuBLAS 提供了大量带特定收尾逻辑的 GEMM 内核变体。

> 使用 `torch.compile` 的不同 `mode` 参数会触发不同内核变体。可以自行测试，并在评论区分享观察结果。

因此，Inductor 生成的代码实际执行两步：

- `out = copy(C)` ← 这是 DtoD 内存拷贝（32 MB，耗时约 33 µs）
- `out = α·(A·B) + β·out` ← 使用 `α=β=1` 的 GEMM，将偏置加法融合到写回阶段

数学结果等价，但优化存在隐性开销：额外一次显存拷贝 + 复杂度略高的 GEMM 收尾逻辑。

原本期待的融合是让 `x·w + b`（这里是 `out = α·A·B + β·C`）折叠成一个没有额外内存访问的单一内核，但实际并非如此。Inductor 保留了两次访存操作，只是把偏置拷贝表示为内存拷贝，把加法表示为 GEMM 收尾阶段。

真正的融合实现会跳过这次内存拷贝。FlashAttention 这类手写内核会这么做，Inductor 借助 Triton 代码生成也能做到；但针对 `4096×4096 bf16 matmul`，Inductor 显然判断「使用 cuBLAS，并通过收尾阶段处理偏置」是更合适的路径。

### CPU 开销不降反升

对比动态执行与编译运行极易忽略的细节：


| 轮次 | eager 耗时（ms） | compile 耗时（ms） |
| :--: | :------------: | :--------------: |
|  #2  |      0.1      |       0.2       |
|  #3  |      0.07      |       0.1       |
|  #4  |      0.07      |       0.1       |

编译版本单轮 CPU 耗时约为原生的 2 倍。原因是每次调用都完整经过 Dynamo→AOTAutograd→Inductor 整套调度链路，同时保留原有 `aten::addmm` 调度开销。编译框架面向多算子大型模型设计，海量算子可平摊单次调用的编译开销；单算子场景下，编译调度反而带来额外性能损耗。

> [!TIP]
> 作为课后练习，可以查阅 `torch.compile` 文档，尝试找到能够降低 CPU 调用开销的 `mode` 参数。🤗

## 轨迹阅读速查表

本小节汇总前文出现过的模式：在性能轨迹中看到相应现象时，可以快速对照判断含义。

### 性能统计表


| 观测现象                                                                         | 通常意味着什么                                                                                                                               |
| :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `Self CPU time total` ≫ `Self CUDA time total`（CPU 单位为 ms，GPU 单位为 µs） | **开销受限型**：CPU 任务下发耗时远大于 GPU 实际计算耗时。可以增大工作量（更大的矩阵、批处理运算），或融合调用。                              |
| `Self CPU time total` ≈ `Self CUDA time total`，两者都是 ms                     | **计算受限型**：GPU 是性能瓶颈，这通常是更理想的状态。                                                                                       |
| 某个事件独占绝大部分 `CUDA total` 耗时                                          | 这就是性能热点，优先从这里开始优化。                                                                                                         |
| 某个事件的 `# of Calls` 很大                                                    | 即使单次调用很便宜，频繁调用也可能形成瓶颈。检查它是否可以融合或批量化。                                                                      |
| 某一行的 `CPU total` ≫ `Self CPU`                                                | 大部分耗时位于内部子调用。排查时应深入嵌套事件，而不是只看父事件。                                                                            |

### CPU 时序泳道（CPU lane）


| 看到的现象                                                                     | 通常意味着什么                                                                                                    |
| :----------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| 首轮 `ProfileStep` 耗时明显远大于后续轮次                                      | 冷启动开销：显存工作区申请、cuBLAS 启发式选型、模块懒加载。可以增加预热迭代，或启用剖析器调度中的 `warmup` 参数。 |
| `record_function("...")` 起始与内部首个 `aten::*` 之间出现大片空档             | 同样是冷启动开销，只是在放大视图中的表现：标记已经进入，但算子尚未下发。                                           |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 出现在 `cudaLaunchKernel` 之前 | 重型自适应内核（GEMM 矩阵乘、卷积等）；cuBLAS 向驱动查询单个 SM 可容纳线程块数量，以此挑选最优内核实现。          |
| `cudaLaunchKernel` 前方无占用率查询调用                                       | 逐元素或规约类轻量内核，资源占用固定，无需预先规划。                                                               |
| 有效采样末尾出现很长的 `cudaDeviceSynchronize`                                 | 剖析器正在刷新事件；耗时主要来自 GPU 完成待处理任务，并非真实 CPU 计算开销。小规模 GPU 工作伴随长时间同步，是典型开销受限症状。 |
| 源码里没有写内存拷贝，但出现 `cudaMemcpyAsync`                                 | 通常是隐藏的设备到设备拷贝。常见情况是 `addmm` 在 GEMM 收尾阶段之前，先用偏置初始化目标缓冲区。                   |

### GPU 时序泳道（GPU lane）


| 看到的现象                             | 通常意味着什么                                                                           |
| :------------------------------------- | :--------------------------------------------------------------------------------------- |
| GPU 泳道出现 `Activity Buffer Request` | 剖析器正在申请或填充自身事件缓冲区；首次出现通常可以解释最初的 CPU/GPU 泳道偏移。        |
| 单轮采样内两个内核中间有空隙           | 运行中触发缓存申请；多轮采样仅偶然出现一次则是工具行为，非业务代码问题。                 |
| 同一种内核，多轮运行耗时参差不齐       | GPU 主频动态升降、芯片温升、功耗调度、驱动后台任务导致波动，不能只依赖平均值做性能评估。 |
| 内核名形如 `ampere_bf16_s16816gemm_...` | cuBLAS 底层矩阵乘内核；相同尺寸与精度下，动态执行与编译模式通常共用同一套内核。          |
| GEMM 内核紧邻前置 `Memcpy DtoD`         | 这是 `addmm` 收尾阶段的偏置拷贝。「融合」发生在调度器层面，而不是 CUDA 内核层面。        |

### 任务下发调用链


| 看到的现象                                                                                                   | 通常意味着什么                                                                           |
| :----------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| `ProfileStep#N` → 自定义标记 `<record_function name>` → `aten::*` → `aten::mm` / `aten::bmm` / `aten::add` | 标准嵌套调用层级；自身耗时不含子调用，总耗时包含所有子任务。                             |
| `aten::matmul` 下沉为 `aten::mm`                                                                            | 普通二维矩阵相乘。                                                                       |
| `aten::matmul` 下沉为 `aten::bmm` 且附带额外 CUDA Runtime 调用                                             | 带批量维度的矩阵乘法。cuBLAS 需要更多启发式工作来选择合适的内核变体。                    |
| `aten::addmm(b, x, w)` 替代单独的 `aten::add` + `aten::mm` 组合                                             | **调度层算子融合**，GPU 底层仍使用同一个 GEMM 内核，偏置加法被并入收尾阶段。             |

### torch.compile 编译相关


| 看到的现象                                                 | 通常意味着什么                                                                                         |
| :--------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| CPU 泳道出现 `Torch-Compiled Region: K/M` 行                | 当前位于编译后函数内部。                                                                               |
| 每轮都有 `TorchDynamo Cache Lookup`                         | Dynamo 校验张量形状、数据类型、设备是否匹配已缓存编译产物；编译完成后每次调用仍会产生该校验开销。      |
| 无梯度计算时仍存在 `AOTDispatcher Runtime Wrapper Prologue` | AOTAutograd 的运行时包装器始终位于调用栈中，负责张量元数据和视图追踪。                                  |
| 多轮 `## Call CompiledFxGraph <hash>` 哈希值一致            | 命中生成代码缓存；生成源码位于 `/tmp/torchinductor_<user>/fxgraph/<hash>`。                            |
| 单算子场景编译后 CPU 耗时高于原生 Eager                    | 正常现象：Dynamo→AOTAutograd→Inductor 整套链路带来固定开销，仅在大量算子堆叠的大模型中才能平摊成本。 |

## 结论

本文从极简的 `matmul + add` 入手，借它学习如何阅读 PyTorch 剖析器输出。过程中得到的一些分析思路，也适用于更大的工作负载。

本文是《PyTorch 性能剖析》系列的第一站。后续文章会逐渐离开这个双算子示例，走向更大的构建块，并最终分析真实模型。

感谢 [Noe Flandre](https://huggingface.co/NoeFlandre)、[Suvaditya Mukherjee](https://huggingface.co/suvadityamuk) 和 [Vidit Ostwal](https://huggingface.co/ViditOstwal) 审阅本文早期草稿。
