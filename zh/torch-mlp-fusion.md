---
title: "PyTorch 性能剖析（第 2 部分）：从 nn.Linear 到融合 MLP"
thumbnail: /blog/assets/torch-mlp-fusion/thumbnail.png
authors:
  - user: ariG23498
  - user: ror
  - user: sergiopaniego
  - user: pcuenq
  - user: sayakpaul
translators:
  - user: HCS9527
---

# PyTorch 性能剖析（第 2 部分）：从 nn.Linear 到融合 MLP

![博客文章缩略图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/thumbnail.png)

在 [「PyTorch 性能剖析 (profiling)」系列的第一部分](https://huggingface.co/blog/torch-profiler) 中，我们用 `torch.add(torch.matmul(x, w), b)` 学习了如何阅读 PyTorch 性能剖析器轨迹。沿途还讨论了几个相关主题：CPU 调度链、启动开销、开销受限与计算受限两种状态的区别，以及 `torch.compile` 的一些内部机制。

到了第二篇（也就是本文），我们再向上迈一步。我们会把手写的矩阵乘加组合替换为 `nn.Linear`（设置 `bias=True`）。这是每个深度学习模型都会用到的基础构件。随后，我们会把三个 `nn.Linear`（本例中的具体设置）堆叠起来，并在中间加入激活函数，构成一个多层感知器（MLP）模块。

> [!NOTE]
> 本文脚本在这里：[`02_linear.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/02_linear.py)、[`03_simple_mlp.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/03_simple_mlp.py) 和 [`03_kernels_mlp.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/03_kernels_mlp.py)。和上一篇一样，建议在单独的标签页打开脚本，边读文章边对照代码。本文使用 `NVIDIA A100-SXM4-80GB` GPU 运行脚本。在 Hugging Face 基础设施上启动 GPU 并通过 [Spaces Dev Mode](https://huggingface.co/docs/hub/spaces-dev-mode) 运行这些脚本很方便。你也可以用 [Hugging Face Jobs pipeline](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) 执行脚本。

开始之前，先快速回顾两个后文会反复用到的概念：

1. GPU **内核（kernel）** 是在 GPU 的大量线程上并行运行的程序。
2. CPU 负责**调度并启动**这些内核。在 PyTorch 性能轨迹中看到的大部分开销，都是这类调度工作。


## 从 matmul-add 到 Linear

`nn.Linear` 是对我们在 [第 1 部分](https://huggingface.co/blog/torch-profiler) 中已经剖析过的同一组矩阵乘法与加法的模块封装。唯一的区别是，它把权重和偏置作为参数持有，并暴露出 PyTorch 用户熟悉的 `forward` 方法。

```py
# bias=True would truly emulate the multiplication and addition
# operations we have seen in part 1 of the series
linear_layer = nn.Linear(in_dim, out_dim, bias=True)
y = linear_layer(x)
```

当前操作可以写成：

```
y = x @ w.T + b
```

其中 `x` 是输入，`w` 是权重，`b` 是偏置。运行 [`02_linear.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/02_linear.py)，看看性能剖析结果。

```bash
uv run 02_linear.py --batch 1024 --in_dim 32 --out_dim 64
uvx trace-util traces -b traces
```

> [!TIP]
> [`trace-util`](https://x.com/ariG23498/status/2054811716727517374) 是一个工具，可以把轨迹同步到 [Hugging Face bucket](https://huggingface.co/storage)，并在终端中给出 [Perfetto URL](https://perfetto.dev/)。

| ![一次 `nn.Linear` 前向传播的 PyTorch 性能轨迹：CPU 时间线上有三个较短的 Profile Step 和 `linear_fwd` 标注，GPU 时间线上有一个很小的内核，末尾还有一条较长的 `cudaDeviceSynchronize`](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/linear-profile-trace.png) |
| :--: |
| 图 1：`nn.Linear` 的性能轨迹 |

图 1 展示了一次线性层 `forward` 调用的性能轨迹。我们用和前文轨迹类似的 `schedule` 设置来追踪线性层的 `forward` 调用，其中 `wait=1`、`warmup=1`、`active=3`。这就是为什么 CPU 和 GPU 时间线中会看到三个 Profile Step。

### 转置在做什么？

| ![放大后的 CPU 调度链，显示 aten::linear 内部的 aten::t 转置算子嵌套在 aten::addmm 之前，GPU 时间线上没有对应活动](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/transpose-cpu-dispatch.png) |
| :--: |
| 图 2：转置对应的 CPU 行 |

如果像图 2 那样放大性能轨迹，会发现 `aten::addmm`（乘法与加法）之前有一个 `aten::t`（转置）算子。由此可以看出，`nn.Linear` 会先转置权重参数，再与输入相乘。这就是轨迹里出现 `aten::t` 的原因。

需要注意的是，`aten::t` 并不会真正复制或重新组织数据：它只是在 CPU 上重写张量元数据（形状和步长），用来表示转置后的矩阵。它不会在 GPU 上启动内核。可以通过两种方式验证这一点：查看轨迹中的 GPU 时间线，或查看性能剖析表里 `aten::t` 这一行及其 CUDA 耗时。

### 为什么没有单独的 `mul` 和 `add` 内核？

| ![线性层性能轨迹中高亮了调度链，显示 aten::linear、aten::t 和 aten::addmm，但没有单独的 aten::add 算子](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/no-aten-add.png) |
| :--: |
| 图 3：线性层的性能轨迹中没有 `aten::add` |

如图 3 所示，线性层的调度链中没有 `aten::add`（偏置加法）。原因是偏置加法已经被 _折叠_ 到矩阵乘法内核里，这种做法称为 **epilogue（收尾阶段）**。

**Epilogue** 是 GEMM（GEneral Matrix Multiply，通用矩阵乘）内核在把结果写回 HBM（High Bandwidth Memory，高带宽内存，也就是 GPU 主内存）之前，于最后阶段执行的一小段计算。添加偏置、应用激活函数、乘以常数缩放，都是典型的 epilogue。Epilogue 的意义在于避免第二次从 HBM 读取或写入数据，因为内存流量会显著增加操作成本。

`nn.Linear` 会调用 `torch.nn.functional.linear`，后者再调用 `aten::linear`。`aten::linear` 查看输入，发现传入了偏置，于是调度 `aten::addmm(bias, x, weight)`，而不是把 matmul 和 add 分开执行。`addmm` 计算的是：

```
out = x @ weight.T + bias
```

在 GPU 上运行的 cuBLAS GEMM 内核内置了偏置相加变体，`aten::addmm` 选择的正是这个内核。加法不会以单独内核的形式出现，因为它是 **matmul 内核写回阶段的一部分**，这正是 epilogue 的含义。

这里可以注意一个细节。[第 1 部分 `--compile` 小节](https://huggingface.co/blog/torch-profiler#did-we-fuse-the-matmul-and-add-kernels-into-one) 中出现的那个内核（`addmm`），正是 eager 模式下 `nn.Linear` 已经在用的内核。这里已经没有什么可供 `torch.compile` 继续融合了。下一步我们会验证这一点。

### --compile 能帮助单个 Linear 吗？

我们先编译 `forward` 调用，再查看性能轨迹。（性能轨迹会在[下一节](#where-did-the-transpose-go-kernel-layouts-and-pre-ops)中可视化。）

```bash
uv run 02_linear.py --batch 1024 --in_dim 32 --out_dim 64 --compile
uvx trace-util traces -b traces
```

如果对比单个 `nn.Linear` 的 `forward` 在 eager 与 compile 模式下的轨迹，会发现：

- GPU 上是同一个 cuBLAS GEMM 内核。
- CPU 上是同一个 `aten::addmm` 算子。
- CPU 时间线上多了几行 compile 独有的事件。

这一点值得记住。常见反应是，只要模型变慢就想上 `torch.compile`。但对于单个带偏置的 GEMM，compile 几乎没有发挥空间。这不是 bug，而是因为 compile 至少需要不止一个操作，才可能做出融合。接下来我们通过 [MLP](#stacking-two-linears-the-mlp) 来证明这一点。

<a id="where-did-the-transpose-go-kernel-layouts-and-pre-ops"></a>

### 转置去哪了？内核布局与前置操作

仔细阅读两份轨迹（eager 与 compile）的读者会注意到，eager 模式下的 CPU 调度链比编译后的调度链更长。

| ![Eager 模式下的 CPU 调度链，其中 aten::linear 下面的 aten::t 转置和 aten::addmm 分别被框出](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/eager.png) |
| :--: |
| 图 4：Eager 调度链中，`aten::linear` 依次经过 `aten::t`（转置）和 `aten::addmm` |

| ![编译后的 CPU 调度链，显示一个 Torch-Compiled Region 和一次 aten::addmm 调用，没有转置算子](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/compile.png) |
| :--: |
| 图 5：编译后的调度链直接调用 `aten::addmm`，没有转置 |

Eager 模式下，`aten::linear` 内部的 CPU 调度链是 `aten::t` 后接 `aten::addmm`（图 4）。要理解 `aten::t` 究竟做了什么，需要先绕到 *stride（步长）* 和 *view（视图）* 上。

张量在内存中以一段连续平坦的数字序列来存储自身数据。`shape` 和 `stride` 是覆盖在这段数据上的元数据，用来告诉 PyTorch 如何遍历它：步长 `(s0, s1)` 表示「移动到下一行需要跨过 `s0` 个元素，移动到下一列需要跨过 `s1` 个元素」。只改元数据，就能得到同一份原始数据的不同 *view*，无需复制：

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

`M.t()` 没有移动任何一个数字。它返回了一个新视图，交换了步长，因此现在按行读取时，会以转置后的顺序遍历原始缓冲区 `0, 1, 2, 3, 4, 5`。底层数据完全相同，只有元数据不同。

这正是线性层内部 `aten::t` 的行为：它不会分配新张量，也不会复制任何数据，只是生成一个重写了步长的权重 *view*。

从图 5 可以看出，compile 并没有移除某个 GPU 内核；它移除的是调度这个视图带来的 *CPU 开销*。Inductor 在编译时追踪了整个视图链，一次性算出最终步长，并发出一个直接的 `aten::addmm` 调用，把这些步长硬编码进去。几微秒的 CPU 工作消失了，而 GPU 执行的是完全相同的数学计算。

也正如预期，如果输入数据不满足编译器预先计算好的步长约束，就会抛出错误。

如果查看两份轨迹的 GPU 时间线，会看到每次 forward 都恰好只有一个内核，而且两次都是 *同一个* 内核：

```
cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align8
```

既然没有运行转置内核，那么是谁让 GEMM 按转置后的顺序读取权重矩阵的？答案就在内核名中。看看后缀：

```
cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align8
                                                          ^^
```

这个 `tn` 就是布局描述符。cuBLAS 和 CUTLASS 会针对不同输入布局组合预编译 *不同的内核二进制*。

`n`（non-transposed，未转置）和 `t`（transposed，转置）描述了内核在内层循环中如何遍历输入。调度器的工作是查看输入步长，判断哪种后缀组合匹配，并选择对应的预编译内核。

> [!TIP]
> 性能轨迹里的内核名是内核身份信息的一种摘要。如果两次运行显示同一个内核名，GPU 做的就是同一类工作。如果名称不同（例如 `_tn_` 与 `_nn_`、`bf16` 与 `fp16`，或 `s16816gemm` 与 `s161616gemm`），GPU 做的工作就不同，说明调度器走了另一条分支。学会阅读内核名，是对比性能轨迹时最有用的习惯之一。

<a id="stacking-two-linears-the-mlp"></a>

## 堆叠三个 Linear：MLP

本节会剖析一个多层感知机（MLP）。为了让例子更有意思，我们会剖析一个使用 GeGLU 激活变体的前馈网络（它在实践中相当常用）。这也是我们向深度学习研究史上最伟大的结尾句之一致敬的方式（图 6）。

| ![GLU Variants Improve Transformer 论文的结论部分，其中高亮了结尾句：将这些架构的成功归功于神圣仁慈](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/geglu-paper.png) |
| :--: |
| 图 6：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) 论文的结论部分。|


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

完整脚本在这里：[`03_simple_mlp.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/03_simple_mlp.py)。执行方式如下：

```bash
uv run 03_simple_mlp.py --batch 64 --seq 128 --dim 768 --hidden 3072
uvx trace-util traces -b traces
```

打开轨迹之前，先一起想想预期会看到什么。`forward` 函数做了不少计算，但大部分内容已经很熟悉。

我们应该会看到三次 `aten::linear` 调度，对应三个 `nn.Linear` 层。还应该看到两次逐元素内核启动：一次用于 GeLU，一次用于乘法。查看轨迹之前先形成预期，是性能剖析过程中最有用的习惯：读轨迹是为了 *验证或推翻* 猜测，而不是从零开始形成猜测。

| ![GeGLU MLP 前向传播的性能轨迹，CPU 时间线上有五组被框出的事件，分别标注为 linear、linear、gelu、mul、linear](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/simple-mlp-eager.png) |
| :--: |
| 图 7：GeGLU MLP 的性能轨迹 |

| ![线性投影轨迹中高亮的 occupancy query](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/occupancy-queries.png) |
| :--: |
| 图 8：线性投影 CPU 时间线中高亮的 occupancy query |

图 7 说明我们的预判是正确的。每次 forward（一次 `mlp_fwd`）中，GPU 恰好运行 5 个内核。图 8 高亮了线性投影层 CPU 时间线中的「occupancy query（占用率查询）」。

| 算子 | CPU 算子 | GPU 内核 | 启动过程 |
| :--: | :--: | :--: | :--: |
| `gate_proj`| `aten::linear` | `ampere_bf16_s16816gemm_bf16_128x128_...`| occupancy query + cudaLaunchKernel |
| `up_proj` | `aten::linear` | `ampere_bf16_s16816gemm_bf16_128x128_...`| occupancy query + cudaLaunchKernel |
| `gelu` | `aten::gelu`| `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl...>` | cudaLaunchKernel |
| `h * u` | `aten::mul` | `vectorized_elementwise_kernel<4, ...MulFunctor...>` | cudaLaunchKernel |
| `down_proj` | `aten::linear` | `ampere_bf16_s16816gemm_bf16_128x256_...` | occupancy query + cudaLaunchKernel |

三个 GEMM 在启动前都会额外调用一次 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`。我们在第 1 部分有一节专门讨论过，[可以在这里找到](https://huggingface.co/blog/torch-profiler#why-does-matmul-have-an-extra-cuda-runtime-call)。这是 cuBLAS 在确定网格尺寸。逐元素算子（GeLU 和 mul）则直接启动，没有 occupancy query。因此，「一个 linear」实际是 query + launch，而「一个逐元素算子」只是 launch。

| ![GeGLU MLP 的性能剖析表列出算子名称及其 CUDA 耗时，其中 aten::transpose 和 aten::as_strided 等元数据算子的 CUDA 耗时显示为 0.000us](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/simple-mlp-table.png) |
| :--: |
| 图 9：表格显示某些算子不会启动内核 |

`aten::t`、`aten::transpose`、`aten::reshape`、`aten::view`、`aten::as_strided` 和 `aten::_unsafe_view` 都不会启动内核。它们在表中显示 `0.000us` 的 CUDA 耗时（图 9），因为它们只是在 CPU 上重写张量元数据（形状和步长）。读者扫一眼表格，可能会看到每个 linear 周围有大约六个算子名，但其中只有一个（`mm`）真正到达 GPU。

### 为什么会有两类 GEMM 内核？

MLP 会把 `[batch, seq, dim]` 展平成 `[batch * seq, dim]`，再进行 matmul。在命令行调用中，我们使用的 `batch` 是 64，`seq` 是 128，所以下面出现的 `8192` 来自 `batch * seq = 64 * 128`。

从轨迹中可以看到：

| Linear | `aten::mm` 输入维度 | M·K·N | cuBLAS 内核 | 平均 CUDA 耗时 |
| :--: | :--: | :--: | :--: | :--: |
| `gate_proj` | `[8192,768] x [768,3072]` | `8192·768·3072` | `…128x128…stages_32x5_tn` | 0.19ms |
| `up_proj` | `[8192,768] x [768,3072]` | `8192·768·3072` | `…128x128…stages_32x5_tn` | 0.19ms |
| `down_proj` | `[8192,3072] x [3072,768]` | `8192·3072·768` | `…128x256…stages_64x3_tn` | 0.17ms |

三个 GEMM 的 FLOP 数相同，都是 `2·8192·768·3072 ≈ 38.7 GFLOP`，但 `down_proj` 快约 `10%`。同样的计算量，不同的形状（`N=768` 而不是 `3072`），让 cuBLAS 选择了不同的 tile（`128×256`，并使用更深的 `stages_64x3` 流水线），从而在该形状下获得更好的复用。

> [!NOTE]
> 如果想深入学习 tiling，可以从[这份很好的资料](https://alvinwan.com/how-to-tile-matrix-multiplication/)开始。

这也正解释了为什么表中有两行 GEMM（图 9）：`128x128` 那一行对应 gate+up，`128x256` 那一行对应 down。

### `torch.compile` 会做什么？

在编译 `forward` 方法并查看可视化结果之前，我们再来做一次预判思考练习，想一想我们能从性能轨迹中观察到哪些现象。这是一项很有意思的实践，每当你自己做性能剖析时，都要反复采用这种思考方式。要始终基于自己的经验预判来分析，一旦发现实际情况和预想不符，就要停下来排查背后的原因。

```bash
uv run 03_simple_mlp.py --batch 64 --seq 128 --dim 768 --hidden 3072 --compile
uvx trace-util traces -b traces
```

| ![编译后的 GeGLU MLP 性能轨迹，CPU 时间线上显示三次 aten::mm 调用和一个融合 Triton 内核，分别标注为 mm、mm、fused、mm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/simple-mlp-compile-trace.png) |
| :--: |
| 图 10：编译后的 GeGLU MLP 性能轨迹 |

在 eager 模式下，每个 `nn.Linear` 都会展开成一串调度器算子（`aten::linear` → `aten::t` → `aten::transpose` → `aten::matmul` → `aten::reshape` → `aten::mm`）。这些是 ATen 在到达真正的 GEMM 之前会经过的高层包装。`torch.compile` 会移除这条链。

编译后的图运行时，已经没有 linear、matmul、transpose 或 reshape，这些元数据算子都被折叠进 `mm` 的调用方式中。我们能看到三次裸露的 `aten::mm` 外部调用（图 10）。可以证明二者使用了同一个通用矩阵乘法（GEMM）内核的依据是：内核名与 eager 模式逐字节一致：gate 和 up 使用 `...128x128...stages_32x5_tn`，down 使用 `...128x256...stages_64x3_tn`。

### 融合后的 Triton 内核

| ![编译后的 MLP 轨迹，其中 CPU 时间线上的 triton_poi_fused__unsafe_view_gelu_mul_0 内核被框出，它替代了 eager 运行中的独立 gelu 与 mul 内核](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/fused.png) |
| :--: |
| 图 11：融合后的 Triton 内核 |

这是整篇 compile 课程的重点。Eager 模式下的两个逐元素内核（GeLU 和 mul）再加上一个 reshape，被折叠成一个内核：`triton_poi_fused__unsafe_view_gelu_mul_0`（图 11）。我们来解析这个名称：

* `triton`：由 Inductor 的 Triton 后端生成（不是 cuBLAS，也不是 ATen）。
* `poi`：pointwise，逐元素操作（Inductor 将逐元素内核标记为 `poi`，规约标记为 `red`，持久规约标记为 `per`）。
* `fused__unsafe_view_gelu_mul`：被合并的算子：`_unsafe_view`（reshape）、GeLU 和 mul。
* `0`：该图中的唯一编号。

为什么这是一次性能优化上的提升？在动态执行模式下，中间变量 `h = gelu(g)` 是一个尺寸为 `[8192, 3072]` 的完整 BF16 张量（大小约 50MB）：GeLU 内核会先将该张量写入高带宽显存（HBM），随后乘法内核又需要立刻从显存中将其读取出来。而算子融合会把这份中间数据保存在寄存器中（寄存器位于芯片内部，相比高带宽显存访问延迟更低）。Triton 内核仅会读取一次张量`g`和`u`，完成`gelu(g) * u`的运算后，只将最终结果写入一次显存，省去了中间张量在全局显存中往返读写的一整套开销。

## 使用手工调优内核

到目前为止，我们一直让 PyTorch（eager）和编译器（`torch.compile`）来选择内核。现在换成一个由人工专家编写并手工调优的内核。我们使用 `LigerGEGLUMLP` 层，可以通过 `kernels` 库轻松从 [Hugging Face Hub](https://huggingface.co/kernels/kernels-community/liger-kernels) 获取它。

```python
from kernels import get_kernel

kernels_layers = get_kernel("kernels-community/liger-kernels", version=1).layers
kernels_geglu_mlp = kernels_layers.LigerGEGLUMLP(Config()).to(device, dtype=torch.bfloat16).eval()
```

完整脚本在这里：[`03_kernels_mlp.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/03_kernels_mlp.py)。

```bash
uv run 03_kernels_mlp.py --batch 64 --seq 128 --dim 768 --hidden 3072
uvx trace-util traces -b traces
```

| ![LigerGEGLUMLP 前向传播的性能轨迹，CPU 时间线上显示三个 aten::linear 组和一个 LigerGELUMulFunction 组](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/kernels-profile.png) |
| :--: |
| 图 12：`LigerGEGLUMLP` 层的性能轨迹 |

图 12 展示了使用 Hub 上 Liger kernels 的 `LigerGEGLUMLP` 层性能轨迹。

### 为什么使用 kernels 库

用 Triton 或 CUDA 编写内核是一回事，*分发* 它们又是另一回事。内核必须针对 GPU 架构、CUDA 版本和 PyTorch 版本的精确组合进行编译。通常出问题的就是这一步（「在我机器上能跑」、缺少 `nvcc`、Triton 版本不对）。

[`kernels`](https://github.com/huggingface/kernels) 库会把构建步骤从本机移走。`get_kernel("kernels-community/liger-kernels", version=1)` 会从 Hugging Face Hub 下载一个**预构建、版本固定**的内核包，并缓存在本地（这里位于 `~/.cache/...kernels-community--liger-kernels`）。它的好处包括：

* 内核在 CI 中一次性为多种架构和版本组合编译完成。开发者下载正确的二进制即可，而不用自己编译。
* `version=1` 固定了精确构建，因此运行脚本的所有人都会拿到同一个内核。不会出现「我更新了某个包之后它变慢了」。
* 该包暴露了一个 `.layers` 属性，里面是可直接替换的 `nn.Module`（例如 `LigerGEGLUMLP`）。把自己的模块换成它们即可，模型其他部分不需要改变。

### 调优内核为什么更好

这里说的「调优」包含两个具体含义，而且两者都能在轨迹里看到。

| ![编译后的 MLP 轨迹中，在编译图运行前，CPU 时间线上的 TorchDynamo、prologue 和 guard 前置操作被框出](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/compile-preops.png) |
| :--: |
| 图 13：编译运行在任何 GEMM 执行前都要付出前置操作开销（Dynamo、guard、prologue） |

| ![LigerGEGLUMLP 轨迹中，原本 compile 前置操作所在的位置是空的，显示手写内核没有 Dynamo 或 guard 开销](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-mlp-fusion/no-preops.png) |
| :--: |
| 图 14：Liger 内核没有前置操作，原本可能出现这些操作的框中是空的 |

1. **融合已经内置其中。** [`LigerGEGLUMLP`](https://huggingface.co/kernels/kernels-community/liger-kernels/blob/v1/build/torch-cuda/layers.py#L307) 的 forward 是 `down_proj(LigerGELUMulFunction.apply(gate_proj(x), up_proj(x)))`。[`LigerGELUMulFunction`](https://huggingface.co/kernels/kernels-community/liger-kernels/blob/v1/build/torch-cuda/geglu.py#L130) 会运行一个 Triton 内核 [`_geglu_tanh_forward_kernel`](https://huggingface.co/kernels/kernels-community/liger-kernels/blob/v1/build/torch-cuda/geglu.py#L97)，一次完成 `gelu(gate) * up`。这与我们从 `torch.compile` 中看到的效果完全一致：中间结果不会通过 HBM 往返。这里**不需要编译器**就能做到，如图 13 和图 14 所示（没有 Dynamo guard、没有编译延迟，也没有重新编译风险）。

2. **启动参数是为硬件选好的。** 内核不会随机猜测 block size。Liger 的 [`calculate_settings`](https://huggingface.co/kernels/kernels-community/liger-kernels/blob/v1/build/torch-cuda/geglu.py#L95) 会根据列数选择这些参数。

这里需要诚实看待取舍，因为原始数字可能产生误导。Liger 内核运行耗时为 **92.8 µs**，而 compile 运行中 Inductor 的融合内核是 **89.4 µs**。乍看之下，手写内核略慢，但这种比较掩盖了让它有价值的成本差异。

`torch.compile` 会针对**静态形状**做专门化。Inductor 的 `89.4 µs` 内核之所以快，正是因为它是为 *这个精确的* `[8192, 3072]` 问题生成的。一旦改变 batch size、sequence length 或 hidden dimension，Dynamo 就会重新追踪，并再次支付编译成本，换取一个新的专门化内核。

因此，实际要做的取舍并非 “手写低效内核” 与 “编译后高效内核” 二选一，而是**在通用型高性能内核和针对单一输入形状深度定制的专用内核之间做选择**。Liger 内核只需一组启动参数，无需重新编译即可适配任意张量形状运行。它牺牲掉按不同形状定制优化所能省下的几微秒极致耗时，换来对动态变化张量形状的强兼容性。

## 结论

下表汇总了每一步改变了 GPU 上的什么，又保留了什么。

| 设置 | 发生了什么变化 | 什么保持不变 |
| :-- | :-- | :-- |
| Eager `nn.Linear` | 基线：偏置加法已经被折叠进 GEMM epilogue（`addmm`），所以它是 *一个* cuBLAS 内核，而不是 matmul 再加 add | — |
| 编译后的 `nn.Linear` | 少量 CPU 调度算子（`aten::t` 视图 bookkeeping）消失 | 仍是同一个 cuBLAS GEMM 内核，逐字节一致。Compile 没有可融合的内容 |
| Eager MLP | 5 个 GPU 内核：3 个 GEMM + 一个 GeLU + 一个 mul。`[8192, 3072]` 中间结果完整往返 HBM | 每个 GEMM 仍是独立 linear 使用的同一类无偏置 cuBLAS 内核 |
| 编译后的 MLP | GeLU + mul + reshape 折叠成**一个**融合 Triton 内核；中间结果留在寄存器中。需要支付 compile 前置操作开销（Dynamo、guard） | 3 个 GEMM 未受影响，cuBLAS 内核名完全相同 |
| Liger MLP | 同样完成融合，但融合内置在手写 Triton 内核中，并使用硬件调优后的启动参数，且**没有** Dynamo、guard 或编译延迟 | 3 个 GEMM 仍是同一批 cuBLAS 内核 |

如果说有一个习惯值得坚持下去，那就是我们在每次查看轨迹前练习的那个：**先猜，再看。** 先说明你预估这份追踪日志会包含哪些内容，再打开日志文件；一旦发现实际内容和你的预判不符，就把这些偏差当作屏幕上最值得研究的亮点。

这是 **PyTorch 性能剖析** 系列的第二站。下一篇文章，我们将继续循序渐进地深入讲解，从当前的多层感知器模块过渡到注意力模块，最终完整介绍整个模型。

感谢 [Noe Flandre](https://huggingface.co/NoeFlandre) 和 [Pedro Gabriel Gengo Lourenço](https://huggingface.co/pedrogengo) 审阅本文初稿！
