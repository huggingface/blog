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

无论目标是从大语言模型（LLM）中每秒挤出更多 tokens（令牌）、把推理（inference）耗时缩短数毫秒，还是弄清楚训练循环（training loop） 为什么比规格参数承诺的速度慢，性能剖析（性能剖析）都是绕不开的关键步骤。

问题在于，profiling 的入门门槛很高。trace 像一面密密麻麻的彩色矩形墙，各类事件的名称也常常让人望而生畏。大多数教程（tutorial） 默认读者已经知道如何读 trace。因此，即便已经知道应该做 profiling，打开一份 trace 也常常像是一件可以先放一放，或者交给别人处理的麻烦事。本文，以及这个系列，就是为了降低这个入门门槛。

本文是《PyTorch 性能剖析实战》系列的开篇文章，本系列将循序渐进带你掌握解析性能追踪日志的能力，并依托该能力落地性能优化。计划如下：

1. **第 1 部分（本文）：** 从最简单的操作开始：一次 matrix multiplication 后接 bias add，学习如何阅读 profiler 给出的结果。
2. **第 2 部分：** 扩展到 `nn.Linear` 和一个小型多层感知机 （MLP），借助追踪数据找准优化方向，初探底层的 CUDA 内核。
3. **第 3 部分：** 结合 transformers 库，在大语言模型上综合运用前述全部知识。

我们会从初学者视角记录这个过程。除了基础 PyTorch 知识，无其他前置知识门槛。你可以轻松品读，过程中常会收获恍然大悟的启发。文章采用**问题导向**的写作思路：打开追踪视图后抛出疑问 “为什么会出现这种现象？”，顺着线索探寻答案直至彻底弄懂。读完本文后，你将掌握以下内容：

- 如何设置 `torch.profiler`，以及该工具的输出数据含义；
- 读懂性能统计表与时序追踪图（CPU 时序栏、GPU 时序栏，以及两者间反常的空闲间隔）；
- 从 Python 接口调用，逐层下沉至 CUDA 内核执行的完整事件链路；
- 启用`torch.compile`编译后，模型哪些性能指标会变化，更关键的是：**哪些指标保持不变**。

开始之前，先明确两个关键定义，方便后续阅读：

1. **GPU 内核（kernel）**：能够依托 GPU 海量线程并行运行的程序单元。
2. **CPU 负责调度并启动各类 GPU 内核**。

日常开发一般无需手动编写 GPU 内核：调用 PyTorch 算子时，框架会自动将运算转化为一个或多个 GPU 内核，在显卡上完成计算。

掌握以上两个概念，我们就开始逐一探究问题。

> [!NOTE]
> 本文使用的完整 script 在这里：[`01_matmul_add.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py)。建议在单独的标签页中打开这个 script，并逐步阅读代码。我们使用 `NVIDIA A100-SXM4-80GB` GPU 运行这些 scripts。

## 矩阵乘法与加法运算

正如 [Sara Hooker 博士的准确调侃](https://youtu.be/7knwihgj0fU?si=uvzGH-J9bsCHP4Nn&t=2199)：人主要由水构成，而 Deep Neural Networks 主要由矩阵乘法构成。矩阵乘算是深度学习的基石，我们的性能剖析之旅从它起步再合适不过。

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)
```

> 矩阵乘法搭配偏置加法，还原了神经元中权重与偏置的运算逻辑。顺带一提，本次的加法运算还能帮我们在后文理解模型编译原理（跳转至后文：# 见证 torch.compile 实际运行效果）。

我们将借助 `torch.profiler` 模块完成性能分析，整体分为四步：

1. 准备好 [要 profile 的代码](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L26-L27)（这里是 `def fn`，它封装了矩阵乘 + 偏置加运算）。
2. [Annotate](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L32) algorithm（标注算法）。这个步骤完全可选，但建议执行。`record_function` 会把函数标注为 `matmul_add`，后续在追踪图谱中可以快速定位对应模块。

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

1. profiler table（性能统计表）：提供 algorithm 的统计摘要。它回答「什么耗时最多」。这对定位 hotspots 很有帮助。hotspot 可能是最耗时的 event，可能是 pipeline 的 bottleneck，也可能是触发次数特别多的 event。
2. profiler trace（时序追踪文件）：提供时间维度上的执行视图。它回答「某个 operation 何时发生、为什么发生」，并展示 CPU 和 GPU 上发生的 activities。当需要调查启动了哪些 kernel、kernel launch 是否有 delay、CPU 与 GPU activity 是否 overlap 等问题时，它很有用。

现在用第一次 execution 看看这两者如何工作。（[完整源码 `01_matmul_add.py` 在这里](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py)）

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


| ![64×64 矩阵上 matmul add 的 profiler table](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profile-table-64.png) |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                图 1：64 维矩阵乘加运算的性能剖析统计表                                                                |

`.txt` 文件存放性能剖析汇总表。打开文件后（如图 1），可以看到一张大型表格：首列是剖析区间内执行的各类事件名称。

其余各列记录对应事件在 CPU、GPU 或是`torch.profiler.profile`的`activities`参数指定设备上的耗时。重点观察耗时占比最高的事件，结合业务逻辑判断该耗时是否合理；同时需要关注`#调用次数 of Calls` 这一列也很重要，它表示对应事件的触发频次。

顺便也说一下自身耗时「 Self CPU/CUDA」与总耗时「CPU/CUDA total」的区别。「Self」列只统计 event 自身内部花费的时间，**不含内部子事件耗时**。「total」列则包含 event 自身及其所有嵌套子事件的耗时总和。因此，查看 `matmul_add` 的「CPU total」时，它包含自身花费的时间，以及它触发的内部所有子算子的 CPU 耗时。这是一个需要注意的重要细节。

如果查看表格最后两行，会发现 profiler 给出：

```bash
Self CPU time total: 2.314ms
Self CUDA time total: 23.104us
```

CPU 耗时单位是 `ms`，而 GPU 耗时单位是 `us`。为了建立直观感受：GPU 上花费的时间（kernel `ampere_bf16_s16816gemm...`）不到 CPU 上花费时间（`matmul_add` operation）的 1%。GPU 大部分时间处于 idle 状态，这是一个明显的 red flag。出现这种情况的原因是，小规模矩阵乘法在 GPU 上运算极快，程序绝大部分开销消耗在内核初始化、GPU 任务调度、数据收发、结果回收等环节。这类受调度开销制约的算法被称作 _开销受限型（overhead-bound）_ 算法。

脱离这种状态最简单的方法，是**改用更大尺寸的矩阵执行乘法运算**。

```bash
uv run 01_matmul_add.py --size 4096
```


| ![4096×4096 矩阵上 matmul add algorithm 的 profiler table](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profiler-table-4096.png) |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                  图 2：4096×4096 矩阵上 matmul add 的 profiler table                                                                  |

图 2 最后两行是：

```bash
Self CPU time total: 4.908ms
Self CUDA time total: 4.495ms
```

两项耗时单位均为毫秒，可见仅通过增大矩阵尺寸，我们就有效提升了 GPU 实际计算耗时。对照图 2 能发现：如今绝大部分 CUDA 耗时消耗在 GPU 内核（`ampere_bf16_s16816gemm_..`）上，而非发起内核调用的 CPU 运算（`matmul_add`）。这说明程序成功从**开销受限型（overhead bound）**转为**计算受限型（compute bound）**。

接下来我们通过`.json`结果文件可视化任务调度调用链。可将 JSON 文件上传至[Perfetto 可视化网页](https://link.wtturl.cn/?target=https%3A%2F%2Fui.perfetto.dev&scene=im&aid=497858&lang=zh)查看性能轨迹；也能执行命令`uvx trace-util traces -b traces`一键生成 Perfetto 访问链接。

## 64x64 traces


| ![一个 CUDA GPU 上 64×64 bf16 matmul 后接 add 的 PyTorch profiler trace](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/64-matmul-add.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                      图 3：64×64 矩阵上 matmul 和 add 的 profiler trace                                                                      |

在图 3 中，我们能看到矩阵乘加运算的性能剖析时序轨迹。矩形条的宽度代表事件持续时长，纵向嵌套结构代表调用层级；CPU 泳道展示 CPU 侧发生的各类事件，GPU 泳道则呈现 GPU 内核的实际执行过程。图中空白区间代表等待或硬件空闲时段。

脚本采用如下默认配置运行：

* 矩阵尺寸 64：输入、权重与偏置张量形状均为 (64, 64)
* 数据类型 bf16：采用 bfloat16（16 位脑浮点）精度
* 未开启编译：没有使用 torch 编译算子
* 无预热：性能采集前未做 GPU 预热

> 使用 Perfetto 工具时，推荐快捷键操作来快速浏览时序图：W/A/S/D 按键控制视图移动缩放。


| ![Perfetto 中并排标注 CPU lane 和 GPU lane 的 PyTorch profiler trace](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gpu-cpu-trace.png) |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                   图 4：PyTorch profiler trace 中的 CPU lane 和 GPU lane                                                                   |

图 4 包含两条时序泳道，分别对应 CPU 行为与 GPU 行为。CPU 泳道里能看到三段剖析采集周期（从 `ProfilerStep#2` 开始），该配置由采样调度器决定：

```py
schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
```

`wait` 参数：跳过干扰数据较多的初始化阶段（对应`ProfilerStep#0`），

`warmup` 参数：程序预热运行，但剖析器不采集日志（对应`ProfilerStep#1`），

`active` 参数：正式采集、最终体现在时序图中的运行轮次。调度配置可以在 [script 这里](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L58) 找到。

接下来我们带着排查思路，围绕时序图展开问题探究。

### 为什么 ProfilerStep#2 耗时这么长？


| ![PyTorch profiler trace 中 ProfileStep#2 看起来比 ProfileStep#3 和 ProfileStep#4 更宽](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/why-is-step-2-big.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                             图 5：`ProfileStep#2` 矩形宽度显著大于后续 Step3、Step4                                                                             |

图 5 中可以看到，`ProfileStep#2` 耗时远超后续轮次。仔细观察后，会发现自定义标记 `matmul_add` 也呈现相同规律。**问题根源藏在标记内部的子事件中，而非标记本身**。


| Step | `matmul_add` start | `aten::matmul` start |     gap     |
| :--: | :----------------: | :------------------: | :---------: |
|  #2  |      138.736      |       366.493       | 227.757 µs |
|  #3  |      517.926      |       523.447       |  5.521 µs  |
|  #4  |      610.039      |       614.527       |  4.488 µs  |


| ![profile step 2 中 record_function matmul_add 和 aten::matmul dispatch 之间的 228 微秒 gap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-227.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                            图 6：`record_function("matmul_add")` 和 `aten::matmul` 之间约 228 微秒的空闲空档期                                                            |

图 6 中标注的约 228 微秒，是从进入`record_function("matmul_add")`代码块，到 PyTorch 真正下发`aten::matmul`算子之间的**空闲空档期**。造成这段耗时的诱因有很多：临时显存工作区申请、[cuBLAS](https://developer.nvidia.com/cublas)（英伟达自研、依托 GPU 加速的底层线性代数运算库）内部启发式策略开销、或是动态懒加载模块。对此我们要么暂且搁置问题，要么遵循行业通用做法：[在性能采集前补充多轮预热运行](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L35-L39)。

在性能剖析场景里，预热指正式采样前预先反复执行目标算子。上文提到的各类 GPU 初始化工作仅会执行一次，我们不希望把这类一次性开销纳入剖析统计。本示例包含两级预热：一是开启剖析器之前手动循环运行函数；二是依托 profiler 配置里的 `warmup` 参数，在剖析周期内部完成预热。本段代码已启用循环运行 + 采样调度配置。

```bash
uv run 01_matmul_add.py --warmup
```

[经过预热、64×64 矩阵的 Perfetto 性能轨迹](https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/ariG23498/traces/resolve/01_matmul_add/64_bf16_warm_eager.json)


| ![warmup steps 之后的 PyTorch profiler trace，ProfileStep#2 不再显示 cold-start overhead](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/warmup.png) |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                               图 7：完成预热后，所有剖析轮次耗时趋于一致                                                                               |

从图 7 能够看出，各轮剖析任务耗时基本持平，但这不代表我们优化掉了一次性初始化开销，只是通过预热，让这类开销避开了性能采样。为方便读者深入学习，可以阅读这篇 [文章](https://pytorch.org/blog/accelerating-generative-ai-2/)，了解如何进一步缩减内核启动开销。

### 为什么 CPU 时序泳道和 GPU 时序泳道存在约 2.5 毫秒的时间差？


| ![PyTorch profiler trace 中 CPU lane 和 GPU lane 之间的 2.32 毫秒 offset](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-bw-kernel-launch.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                            图 8：CPU 与 GPU 泳道之间存在约 2.5ms 启动延迟                                                                            |

CPU 与 GPU 泳道之间存在约 2.5ms 启动延迟 `wait` 和 `warmup` 理应让 GPU 持续满载、消除间隔。

为探明真实原因，我们小幅修改采样调度参数：

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1)
```


| ![wait=0 warmup=0 时的 PyTorch profiler trace，在 steps 之间显示 Activity Buffer Request](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/full-profile.png) |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                 图 9：设置`wait=0` 和 `warmup=0` 后，轨迹中出现 Activity Buffer Request 事件                                                                 |

图 9 显示，所有 GPU 运算开始前，GPU 泳道里出现了一项 `Activity Buffer Request`（活动缓冲区申请）事件，接下来我们放大视图细看细节。


| ![profiler buffer request 导致 matmul 和 add CUDA kernels 之间出现 gap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/mat-add-gap.png) |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                             图 10：第 1 轮采样周期中，矩阵乘内核与加法内核之间出现一段时间空隙                                                             |

放大 GPU 时序轨迹后能够发现：`ProfileStep#0`对应的矩阵乘、加法内核是连续串行执行（该轮 CPU 时序未在图中展示），而`ProfileStep#1`的两个内核中间多出一段空窗。成因最合理的解释是**缓冲区溢出**，在内核运行途中触发了一次新的缓冲区申请（向 GPU 显存申请分配内存）。

想要排除其他诱因，最稳妥的办法是增加采样迭代次数观测：修改参数`active=20`重新采集性能数据。


| ![20 个 active iterations 的 PyTorch profiler trace，确认 buffer-request gap 只出现一次](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters.png) |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                      图 11：20 轮有效采样结果，空隙仅单次出现，证实由缓冲区申请导致                                                                      |

如图 11 所示，仅在 `ProfileStep#1` 出现该空隙，和之前推断吻合，因此可以确定空档源自缓冲区内存申请。

### Event chain（事件链）


| ![PyTorch profiler 中嵌套的 CPU dispatch chain：ProfileStep、matmul_add、aten::matmul、aten::mm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cpu-nests.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                       图 12：PyTorch 剖析图中嵌套式 CPU 下发调用链：ProfileStep → matmul\_add → aten::matmul → aten::mm                                                       |

图 12 展示了层层嵌套的 CPU 调用栈，是理解算子下发全链路的关键视图。

最外层`ProfileStep#编号`包裹整轮性能采样逻辑；因为我们手动添加了标注，视图里出现`matmul_add`节点，其内部包含两次 ATen 算子调用：矩阵乘法、矩阵加法。

`aten::matmul` 是面向用户的 PyTorch matmul calls 最终落地的 [ATen-level](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen) 层调度接口。`aten::mm` 是专用于二维矩阵相乘的底层后端算子。

补充一个有趣细节：若给张量新增批量维度（batch 轴），PyTorch 会自动选用批量矩阵乘算子`aten::bmm`。我们顺带切换样例，观察`aten::bmm`的实际调用表现。

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
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                 图 13：批量矩阵乘法（BMM 运算）                                                                                 |

在图 13 中，给输入张量新增批量维度后，`aten::matmul`内部不再调用`aten::mm`，转而使用`aten::bmm`，同时还附带一系列前置 CUDA 运行时调用。这也侧面体现出 cuBLAS 会依靠内部启发式逻辑，为当前运算择优调度最合适的 GPU 内核。

> 后文若无特殊说明，我们均采用普通二维矩阵开展实验。

### 为什么矩阵乘法会多出一项 CUDA 运行时调用？


| ![CPU lane 显示 cudaOccupancyMaxActiveBlocksPerMultiprocessor 位于 matmul cudaLaunchKernel 之前](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cudaoccupancy.png) |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                  图 14：矩阵乘法内核启动前执行了 cudaOccupancyMaxActiveBlocksPerMultiprocessor 查询                                                                  |

可以观察到：`aten::mm`对应两次 CUDA Runtime 调用，分别是图中标注的`cudaOccupancyMaxActiveBlocksPerMultiprocessor`与`cudaLaunchKernel`；而`aten::add`仅存在`cudaLaunchKernel`一次调用。

`cudaOccupancyMaxActiveBlocksPerMultiprocessor`是 CPU 侧的资源预规划接口，作用为：**给定内核函数、线程块尺寸与动态共享内存大小，查询单个流式多处理器（SM）最多能同时容纳多少个该内核的线程块**。

由此产生疑问：为何矩阵乘法需要资源预规划，加法运算却不需要？

答案取决于内核的硬件资源占用。点击时序图里的 GPU 内核条目，即可查看对应内核的资源开销详情。


| ![cuBLAS matmul kernel resource footprint：Perfetto 中的 registers、shared memory 和 block size](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/matmul-footprint.png) | ![elementwise add CUDA kernel resource footprint，包含 32 registers 和 zero shared memory](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/add-footprint.png) |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                            图 15：矩阵乘 cuBLAS 内核资源占用（寄存器、共享内存、线程块尺寸）                                                                            |                                                                               图 16：逐元素加法内核资源：32 个寄存器、无共享内存                                                                               |

从图 15 可见，矩阵乘法的**单线程寄存器占用、共享内存占用**随矩阵尺寸动态变化。cuBLAS 内置上百种不同版本的矩阵乘内核，每种内核都依托启发式策略完成启动，而该策略需要实时获取硬件算力信息，占用率查询就是启发逻辑的一环。从原理上理解， [GPU 矩阵乘法基于分块（tile）计算](https://alvinwan.com/how-to-tile-matrix-multiplication/)：分块数量、单块大小由输入矩阵规格与硬件参数共同决定；现代优化算法细节虽更加复杂，但这套分块思路仍是很好的理解框架。

反观图 16，加法内核固定占用 32 个寄存器、不使用共享内存，资源需求极小，硬件资源不会成为运行瓶颈，因此无需提前查询占用率。这类算子天生轻量化。

> [!NOTE]
> 该规律可作为性能排查快捷手段：浏览 CPU 泳道，只要出现 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 调用，就代表后方是**资源开销大、自适应调度启动**的内核，常见于通用矩阵乘 GEMM、卷积等算子；前面无占用率查询的内核，大多是逐元素运算、规约类算子，由 PyTorch 按固定逻辑直接发起调度。

### cudaDeviceSynchronize 耗时为何高达约 1.78 毫秒？

`cudaDeviceSynchronize` 会阻塞 CPU 线程，直至当前 GPU 上所有计算任务全部执行完毕。剖析器会在有效采样窗口末尾执行该同步操作，目的是刷新缓存的性能事件，缺少这一步就会丢失内核耗时统计数据。

一次仅 26 微秒 GPU 有效运算，却伴随 1.78 毫秒同步耗时，说明程序空闲占比高达 98%，是典型的**开销受限型**性能特征。

## 4096x4096 矩阵尺寸性能轨迹traces

前文通过性能统计表分析已知：增大矩阵尺寸后，算法会从开销受限转为**计算受限**。

执行下述命令，深入分析性能轨迹：

```bash
uv run 01_matmul_add.py --size 4096 --warmup
```

### 相同内核，单次运行耗时为何参差不齐？


| ![同一 GPU 上 4096x4096 bf16 matmul kernel timings 在 profiler steps 之间发生变化](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/kernel-time.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                 图 17：同一张显卡上，多轮采样里 4096×4096 BF16 矩阵乘内核耗时不一致                                                                 |

从图 17 可见，`ProfileStep#3`对应的矩阵乘 GPU 内核运行耗时显著高于其他轮次。值得留意的是，各轮次输入完全一致，cuBLAS 启发式适配逻辑不会带来差异；CPU 任务下发时序无空隙、也并非剖析工具本身造成的数据失真。

该轨迹点明一个理想化示例里容易忽略的关键点：**即便硬件环境、代码、输入数据完全相同，GPU 内核的实际运行耗时也并非固定值**。

微调脚本参数，循环运行 20 次并全量采集采样数据：

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=20, repeat=1)

- for _ in range(5):
+ for _ in range(20):
```


| ![PyTorch profiler trace 展示 20 次 matmul iterations 中 kernel runtime variance](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters-kernels.png) |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                        图 18：20 轮迭代下，同一份矩阵乘内核呈现明显的运行耗时波动                                                                        |

图 18 同样验证了上述现象：内核代码完全一致，但运算耗时各不相同。造成耗时波动的常见诱因：

* GPU 空闲 / 满载时的频率动态升降
* 显卡温升带来的性能波动
* 显卡功耗调度策略
* 显卡驱动后台例行维护任务

只看平均耗时的开发者会得出结论：矩阵乘法平均耗时约 1 毫秒（5 次均值 1084 微秒）；而查看时序轨迹后能发现：绝大多数场景仅需 580 微秒，仅个别时刻受硬件扰动变慢。两种结论对应的优化思路天差地别，后者才符合真实运行情况。

## 初探 torch.compile 实际效果

使用 `torch.compile` 的优化表现十分亮眼：开发者沿用原生动态执行（eager）写法编写 PyTorch 代码，框架自动识别张量密集计算区域、构建计算图并优化，最终编译生成可执行代码。默认后端为 TorchInductor，整体编译链路：

1. TorchDynamo：抓取 Python 运行逻辑，转化为 FX 计算图
2. AOTAutograd：涉及梯度计算时，构建前向、反向传播计算图
3. Inductor：将上层计算图下沉编译为优化后的 CPU/GPU 机器码

本节结合性能剖析轨迹，讲解编译优化细节：

```bash
uv run 01_matmul_add.py --size 4096 --warmup --compile
```

开启编译参数`args.compile` 后，执行逻辑变更如下：

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)

fn = torch.compile(fn) if args.compile else fn
```


| ![PyTorch profiler trace 中高亮的 torch.compile region，展示 TorchDynamo 和 Inductor frames](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/compilation-region.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                      图 19：性能轨迹中标注出 Torch 编译区域，包含 TorchDynamo 与 Inductor 调用帧                                                                      |

图 19 中可以看到名为 `Torch-Compiled Region: 0/0` 的条目，代表当前正在使用编译后的函数。

### 矩阵乘与加法算子是否融合为单个 GPU 内核？


| ![compiled trace 显示 aten::addmm 替代 eager aten::add 和 aten::mm pair](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/fused-ops.png) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                      图 20：编译运行后，原分开的 aten::mm、aten::add 被替换为单个`aten::addmm` 算子                                                      |

从图 20 提出疑问：乘法与加法是否真正融合为单次算子运算？

这属于**计算图层面算子融合**：Inductor 将`torch.add(torch.matmul(x, w), b)`改写为单次`aten::addmm(b, x, w)`调用。关键细节：**该优化并未生成全新的融合 CUDA 内核**，GPU 底层仍沿用动态执行模式同款 cuBLAS 内核`ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nn`。换言之，本次融合仅发生在算子调度层，而非 GPU 内核层。

> [!NOTE]
> PyTorch 原生接口 [`torch.addmm`](https://docs.pytorch.org/docs/2.12/generated/torch.addmm.html) 本身就整合了矩阵相乘 + 偏置相加两步运算，读者可自行采集该接口的性能轨迹，在评论区分享观测结果。

### torch.compile 的运行时架构

理解编译原理后，结合 CPU 调用栈直观认识运行架构：

**TorchDynamo Cache Lookup**：Dynamo 校验当前入参的张量形状、数据类型、设备、元信息是否匹配已缓存编译产物，参数不匹配则触发重新编译；该查询开销每次调用都会产生，编译完成后也无法省略。

**Torch-Compiled Region**：进入编译后代码的包装入口；

**AOTDispatcher Runtime Wrapper Prologue**：AOTAutograd 运行时封装逻辑。即便本示例无需求导，调度器仍常驻调用栈，负责张量元数据管理、视图追踪，若开启梯度则自动配置反向传播链路。

**Call CompiledFxGraph <hash>**： 编译生成的机器码在此执行，哈希值对应 FX 计算图指纹，多轮采样哈希不变即代表命中编译缓存。

> [!TIP]
> 可通过哈希值在 `/tmp/torchinductor_<user>/fxgraph`路径找到 Inductor 生成的 Triton/C++ 源码，用于内核深度调试。

### CUDA 内核启动次数是否减半？


| ![compiled matmul trace 显示每个 step launch Memcpy DtoD 和 GEMM kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/memcpy.png) |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                           图 21：编译后每轮仍启动两个 GPU 内核：设备间内存拷贝 + GEMM 矩阵乘                                                           |

初看图 21 CPU 轨迹时误以为每轮仅一次`cudaLaunchKernel`调用，但 GPU 时序显示依旧双内核：**设备到设备内存拷贝（Memcpy DtoD）+ GEMM 矩阵乘**。回看 CPU 调用栈可发现，我们遗漏了`cudaMemcpyAsync`异步内存拷贝调度。

`addmm` 计算 `out = α·A·B + β·C`，cuBLAS 带偏置收尾的矩阵乘要求目标缓冲区预先存入偏置数据。\*\* 运算收尾（Epilogue）\*\* 指 GEMM 主计算完成后的附加运算，深度学习里常见的偏置相加、激活函数、归一化都属于 GEMM 收尾逻辑，cuBLAS 因此提供大量带专属收尾逻辑的 GEMM 内核变体。

> 更换 `torch.compile` 的 mode 参数会触发不同内核版本，读者可自行测试并分享结果。

因此，Inductor 生成的代码实际执行两步：

- `out = copy(C)` ← 这是 DtoD memcpy（32 MB，耗时约 33 µs）
- `out = α·(A·B) + β·out` ← 使用 `α=β=1` 的 GEMM，将 bias add fusion 到 writeback 中

数学结果等价，但优化存在隐性开销：额外一次显存拷贝 + 复杂度略高的 GEMM 收尾逻辑。

我们理想中的零额外访存、完全融合进单个内核的优化并未落地；Inductor 只是把偏置拷贝封装为显存拷贝指令、偏置加法并入 GEMM 收尾。

手写内核（如 FlashAttention）可实现彻底免拷贝融合，Inductor 借助 Triton 代码生成也能做到，但针对 4096×4096 BF16 矩阵，框架权衡后选择复用 cuBLAS 内核 + 收尾加偏置的方案。

### CPU 开销不降反升

对比动态执行与编译运行极易忽略的细节：


| step | eager dur (ms) | compile dur (ms) |
| :--: | :------------: | :--------------: |
|  #2  |      0.1      |       0.2       |
|  #3  |      0.07      |       0.1       |
|  #4  |      0.07      |       0.1       |

编译版本单轮 CPU 耗时约为原生的 2 倍。原因是每次调用都完整经过 Dynamo→AOTAutograd→Inductor 整套调度链路，同时保留原有 `aten::addmm` 调度开销。编译框架面向多算子大型模型设计，海量算子可平摊单次调用的编译开销；单算子场景下，编译调度反而带来额外性能损耗。

> [!TIP]
> 课后练习：查阅 `torch.compile` 文档，尝试切换 mode 参数降低 CPU 调用开销。🤗

## 性能速查备忘手册

本小节是前文知识点速查表：在性能轨迹中观测到对应现象，即可对照判断成因。

### 性能统计表（Profiler table）


| 观测现象                                                                         | 通常意味着什么                                                                                                                               |
| :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `Self CPU time total` ≫ `Self CUDA time total`（CPU 单位为 ms，GPU 单位为 µs） | **开销受限型**：CPU 任务调度耗时远大于 GPU 实际计算耗时。优化方案：增大矩阵尺寸、批量运算、算子合并。                                        |
| `Self CPU time total` ≈ `Self CUDA time total`，两者都是 ms                     | **计算受限型**：GPU 算力成为性能瓶颈，是优化后理想状态。                                                                                     |
| 一个 event 独占绝大部分`CUDA total`耗时                                          | 性能热点，优先从此处着手优化。                                                                                                               |
| 一个 event 的`调用次数# of Calls` 很大                                           | 单次开销虽小，但频繁调用易形成隐性瓶颈，可尝试算子融合 / 批量计算。                                                                          |
| 某一行的`CPU total` ≫ `Self CPU`                                                | 耗时主要消耗在内部子调用，排查需要深入嵌套子事件，而非父函数。大部分 cost 在 children 中。应该 drill into nested events，而不是只看 parent。 |

### CPU 时序泳道（CPU lane）


| 看到的现象                                                                     | 通常意味着什么                                                                                                    |
| :----------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| 首轮`ProfileStep` 耗时明显远大于后续轮次                                      | 冷启动开销：显存工作区申请、cuBLAS 启发式选型、模块懒加载。解决：增加预热迭代、启用 profiler 的 warmup 参数。     |
| `record_function("...")` 起始与内部首个 `aten::*` 之间出现大片空档          | 同上冷启动损耗，只是放大视图后的表现：标记进入，但算子还未下发。                                                  |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 出现在 `cudaLaunchKernel` 之前 | 重型自适应内核（GEMM 矩阵乘、卷积等）；cuBLAS 向驱动查询单个 SM 可容纳线程块数量，以此挑选最优内核实现。          |
| `cudaLaunchKernel` 前方无占用率查询调用                                       | 逐元素 / 规约类轻量化内核，资源占用固定，无需预先资源规划。                                                       |
| 有效采样末尾出现超长`cudaDeviceSynchronize`                                    | 剖析器刷新缓存事件；耗时主要来自 GPU 收尾剩余任务，并非 CPU 真实运算开销。小计算量伴随长时间同步 = 典型开销受限。 |
| 源码未写内存拷贝，但出现`cudaMemcpyAsync`                                      | 通常是隐藏的 Device-to-Device copy。常见于`addmm` 在 GEMM epilogue 之前用 bias 初始化 destination buffer。        |

### GPU 时序泳道（GPU lane）


| 看到的现象                             | 通常意味着什么                                                                           |
| :------------------------------------- | :--------------------------------------------------------------------------------------- |
| GPU 泳道出现`Activity Buffer Request`  | Profiler 申请 / 填充自身事件缓存；首次出现一般是造成 CPU、GPU 时序错位的根源。           |
| 单轮采样内两个内核中间有空隙           | 运行中触发缓存申请；多轮采样仅偶然出现一次则是工具行为，非业务代码问题。                 |
| 同一种内核，多轮运行耗时参差不齐       | GPU 主频动态升降、芯片温升、功耗调度、驱动后台任务导致波动，不能只依赖平均值做性能评估。 |
| 内核名形如`ampere_bf16_s16816gemm_...` | cuBLAS 底层矩阵乘内核；相同尺寸与精度下，动态执行与编译模式共用这套内核。                |
| GEMM 内核紧邻前置`Memcpy DtoD`         | 这是`addmm` epilogue 的 bias copy。「fusion」发生在 dispatcher level，而不是 kernel 中。 |

### 任务下发调用链（Dispatch chain）


| 看到的现象                                                                                                   | 通常意味着什么                                                                           |
| :----------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| `ProfileStep#N` → 自定义标记`<record_function name>` → `aten::*` → `aten::mm` / `aten::bmm` / `aten::add` | 标准嵌套调用层级；自身耗时不含子调用，总耗时包含所有子任务。                             |
| `aten::matmul` 下沉为 `aten::mm`                                                                            | 普通二维矩阵相乘。                                                                       |
| `aten::matmul` 下沉为 `aten::bmm`且附带额外 CUDA Runtime 调用                                              | 带 Batch 维度的批量矩阵乘，cuBLAS 需要更多启发计算择优选用内核。                         |
| `aten::addmm(b, x, w)` 替代单独的 `aten::add` + `aten::mm` pair                                              | **调度层算子融合**，GPU 底层仍使用原生 GEMM 内核，偏置加法被并入内核收尾 Epilogue 阶段。 |

### torch.compile 编译相关


| 看到的现象                                                 | 通常意味着什么                                                                                         |
| :--------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| CPU 泳道出现`Torch-Compiled Region: K/M` row               | 代码进入编译后逻辑执行。                                                                               |
| 每轮都有`TorchDynamo Cache Lookup`                         | Dynamo 校验张量形状、精度、设备是否匹配已缓存编译产物；编译完成后每次调用仍会产生该校验开销。          |
| 无梯度计算时仍存在`AOTDispatcher Runtime Wrapper Prologue` | AOTAutograd 运行时封装常驻调用栈，负责张量元数据、视图追踪，按需搭建反向传播。                         |
| 多轮`## Call CompiledFxGraph <hash>` 哈希值一致           | 命中编译缓存；生成代码存放路径：`/tmp/torchinductor_用户名/fxgraph/哈希`。                             |
| 单算子场景编译后 CPU 耗时高于原生 Eager                    | 正常现象：Dynamo→AOTAutograd→Inductor 整套链路带来固定开销，仅在大量算子堆叠的大模型中才能平摊成本。 |

## 结论

我们从极简的「矩阵乘 + 偏置加」入手，掌握 PyTorch 性能图谱阅读方法，总结的分析思路可无缝迁移至各类复杂深度学习任务。

本文是《PyTorch 性能剖析》系列第一篇，后续文章将脱离简易双算子示例，逐步升级至多层网络、最终落地真实大模型的性能调优。

感谢 [Noe Flandre](https://huggingface.co/NoeFlandre)、[Suvaditya Mukherjee](https://huggingface.co/suvadityamuk) 和 [Vidit Ostwal](https://huggingface.co/ViditOstwal) 对本文初稿的审稿校正！对本文早期 draft 的 review。
