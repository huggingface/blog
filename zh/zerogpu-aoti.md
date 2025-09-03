---
title: "ZeroGPU Spaces 加速实践：PyTorch Ahead-of-Time Compilation 全解析"
thumbnail: /blog/assets/zerogpu-aoti/thumbnail.png
authors:
- user: cbensimon
- user: sayakpaul
- user: linoyts
- user: multimodalart
translator: 
- user: AdinaY
---

ZeroGPU 让任何人都能在 Hugging Face Spaces 中使用强大的 **Nvidia H200** 硬件，而不需要因为空闲流量而长期占用 GPU。  
它高效、灵活，非常适合演示，但并不总能充分利用 GPU 和 CUDA 栈所能提供的一切。  

生成图像或视频可能需要相当多的时间。在这种情况下，充分利用 H200 硬件，使其发挥极致性能就显得尤为重要。  

这就是 PyTorch 提前编译（AoT）的用武之地。与其在运行时动态编译模型（这和 ZeroGPU 短生命周期的进程配合得并不好），提前编译允许你一次优化、随时快速加载。  

**结果**：演示更流畅、体验更顺滑，在 Flux、Wan 和 LTX 等模型上带来 __1.3×–1.8×__ 的提速 🔥  

在这篇文章中，我们将展示如何在 ZeroGPU Spaces 中接入提前编译（AoT）。我们会探索一些高级技巧，如 FP8 量化和动态形状，并分享你可以立即尝试的可运行演示。如果你想尽快尝试，可以先去 [zerogpu-aoti](https://huggingface.co/zerogpu-aoti) 中体验一些基于 ZeroGPU 的 Demo 演示。  

> [!TIP]  
> [Pro](https://huggingface.co/pro) 用户和 [Team / Enterprise](https://huggingface.co/enterprise) 组织成员可以创建 ZeroGPU Spaces，而任何人都可以免费使用（Pro、Team 和 Enterprise 用户将获得 **8 倍** 的 ZeroGPU 配额）  

## 目录

- [什么是 ZeroGPU](#什么是-zerogpu)  
- [PyTorch 编译](#pytorch-编译)  
- [ZeroGPU 上的提前编译](#zerogpu-上的提前编译)  
- [注意事项](#注意事项)  
  - [量化](#量化)  
  - [动态形状](#动态形状)  
  - [多重编译 / 权重共享](#多重编译--权重共享)  
  - [FlashAttention-3](#flashattention-3)  
- [AoT 编译的 ZeroGPU Spaces 演示](#aot-编译的-zerogpu-spaces-演示)  
- [结论](#结论)  
- [资源](#资源)  

## 什么是 ZeroGPU

[Spaces](https://huggingface.co/spaces) 是一个由 Hugging Face 提供的平台，让机器学习从业者可以轻松发布演示应用。  

典型的 Spaces 演示应用看起来像这样：  

```python
import gradio as gr
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(...).to('cuda')

def generate(prompt):
    return pipe(prompt).images

gr.Interface(generate, "text", "gallery").launch()
```

这样做虽可行，却导致 GPU 在 Space 的整个运行期间被独占，即使是在没有用户访问的情况下。

当执行这一行中的 `.to('cuda')` 时：  

```python
pipe = DiffusionPipeline.from_pretrained(...).to('cuda')
```

PyTorch 在初始化时会加载 NVIDIA 驱动，使进程始终驻留在 CUDA 上。由于应用流量并非持续稳定，而是高度稀疏且呈现突发性，这种方式的资源利用效率并不高。 

ZeroGPU 采用了一种即时初始化 GPU 的方式。它不会在主进程中直接配置 CUDA，而是自动 fork 一个子进程，在其中配置 CUDA、运行 GPU 任务，并在需要释放 GPU 时终止这个子进程。   

这意味着：  

- 当应用没有流量时，它不会占用任何 GPU  
- 当应用真正执行任务时，它会使用一个 GPU  
- 当需要并发执行任务时，它可以使用多个 GPU  

借助 Python 的 `spaces` 包，实现这种行为只需要如下代码改动：  

```diff
  import gradio as gr
+ import spaces
  from diffusers import DiffusionPipeline

  pipe = DiffusionPipeline.from_pretrained(...).to('cuda')

+ @spaces.GPU
  def generate(prompt):
      return pipe(prompt).images

  gr.Interface(generate, "text", "gallery").launch()

```
通过引入 `spaces` 并添加 `@spaces.GPU` 装饰器 (decorator)，我们可以做到：  

- 拦截 PyTorch API 调用，以延迟 CUDA 操作  
- 让被装饰的函数在 fork 出来的子进程中运行  
- （调用内部 API，使正确的设备对子进程可见 —— 这不在本文范围内）  

> [!NOTE]  
> ZeroGPU 当前会分配 H200 的一个 [MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#h200-mig-profiles) 切片（`3g.71gb` 配置）。更多的 MIG 配置（包括完整切片 `7g.141gb`）预计将在 2025 年底推出。  

## PyTorch 编译

在现代机器学习框架（如 PyTorch 和 JAX）中，“编译”已经成为一个重要概念，它能够有效优化模型的延迟和推理性能。其背后通常会执行一系列与硬件相关的优化步骤，例如算子融合、常量折叠等，以提升整体运行效率。  

从 PyTorch 2.0 开始，目前有两种主要的编译接口：  

- 即时编译（Just-in-time）：`torch.compile`  
- 提前编译（Ahead-of-time）：`torch.export` + `AOTInductor`  

[`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) 在标准环境中表现很好：它会在模型第一次运行时进行编译，并在后续调用中复用优化后的版本。  

然而，在 ZeroGPU 上，由于几乎每次执行 GPU 任务时进程都是新启动的，这意味着 `torch.compile` 无法高效复用编译结果，因此只能依赖 [文件系统缓存](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html#modular-caching-of-torchdynamo-torchinductor-and-triton) 来恢复编译模型。  
根据模型的不同，这个过程可能需要几十秒到几分钟，对于 Spaces 中的实际 GPU 任务来说，这显然太慢了。  

这正是 **提前编译（AoT）** 大显身手的地方。  

通过提前编译，我们可以在一开始导出已编译的模型，将其保存，然后在任意进程中即时加载。这不仅能减少框架的额外开销，还能消除即时编译通常带来的冷启动延迟。  

但是，我们该如何在 ZeroGPU 上实现提前编译呢？让我们继续深入探讨。  

## ZeroGPU 上的提前编译

让我们回到 ZeroGPU 的基础示例，来逐步解析启用 AoT 编译所需要的内容。在本次演示中，我们将使用 `black-forest-labs/FLUX.1-dev` 模型：  

```python
import gradio as gr
import spaces
import torch
from diffusers import DiffusionPipeline

MODEL_ID = 'black-forest-labs/FLUX.1-dev'

pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.to('cuda')

@spaces.GPU
def generate(prompt):
    return pipe(prompt).images

gr.Interface(generate, "text", "gallery").launch()
```
> [!NOTE]  
> 在下面的讨论中，我们只编译 `pipe` 的 `transformer` 组件。  
> 因为在这类生成模型中，transformer（或者更广义上说，denoiser）是计算量最重的部分。  

使用 PyTorch 对模型进行提前编译通常包含以下几个步骤：  

### 1. 获取示例输入

请记住，我们要对模型进行 *提前* 编译。因此，我们需要为模型准备示例输入。这些输入应当与实际运行过程中所期望的输入类型保持一致。  

为了捕获这些输入，我们将使用 `spaces` 包中的 `spaces.aoti_capture` 辅助函数：  

```python
with spaces.aoti_capture(pipe.transformer) as call:
    pipe("arbitrary example prompt")
```
当 `aoti_capture` 作为上下文管理器使用时，它会拦截对任意可调用对象的调用（在这里是 `pipe.transformer`），阻止其实际执行，捕获本应传递给它的输入参数，并将这些值存储在 `call.args` 和 `call.kwargs` 中。  

### 2. 导出模型

既然我们已经得到了 transformer 组件的示例参数（args 和 kwargs），我们就可以使用 [`torch.export.export`](https://docs.pytorch.org/docs/stable/export.html#torch.export.export) 工具将其导出为一个 PyTorch [`ExportedProgram`](https://docs.pytorch.org/docs/stable/export.html#torch.export.ExportedProgram)：  

```python
exported_transformer = torch.export.export(
    pipe.transformer,
    args=call.args,
    kwargs=call.kwargs,
)
```
### 3. 编译导出的模型

一旦模型被导出，编译它就非常直接了。  

在 PyTorch 中，传统的提前编译通常需要将模型保存到磁盘，以便后续重新加载。 在我们的场景中，可以利用 `spaces` 包中的一个辅助函数：`spaces.aoti_compile`。  
它是对 `torch._inductor.aot_compile` 的一个轻量封装，能够根据需要管理模型的保存和延迟加载。其使用方式如下：  

```python
compiled_transformer = spaces.aoti_compile(exported_transformer)
```
这个 `compiled_transformer` 现在是一个已经完成提前编译的二进制，可以直接用于推理。  

### 4. 在流水线中使用已编译模型

现在我们需要将已编译好的 transformer 绑定到原始流水线中，也就是 `pipeline`。  

接下来，我们需要将编译后的 transformer 绑定到原始的 pipeline 中。  

一个看似简单的做法是直接修改：`pipe.transformer = compiled_transformer`。但这样会导致问题，因为这种方式会丢失一些关键属性，比如 `dtype`、`config` 等。  
如果只替换 `forward` 方法也不理想，因为原始模型参数依然会常驻内存，往往会在运行时引发 OOM（内存溢出）错误。  

因此`spaces` 包为此提供了一个工具 —— `spaces.aoti_apply`：  

```python
spaces.aoti_apply(compiled_transformer, pipe.transformer)
```
这样以来，它会自动将 `pipe.transformer.forward` 替换为我们编译后的模型，同时清理旧的模型参数以释放内存。  

### 5. 整合所有步骤

要完成前面三个步骤（拦截输入示例、导出模型，以及用 PyTorch inductor 编译），我们需要一块真实的 GPU。  
在 `@spaces.GPU` 函数之外得到的 CUDA 仿真环境是不够的，因为编译过程高度依赖硬件，例如需要依靠微基准测试来调优生成的代码。  
这就是为什么我们需要把所有步骤都封装在一个 `@spaces.GPU` 函数中，然后再将编译好的模型传回应用的根作用域。  

从原始的演示代码开始，我们可以得到如下实现：  

```diff
  import gradio as gr
  import spaces
  import torch
  from diffusers import DiffusionPipeline
  
  MODEL_ID = 'black-forest-labs/FLUX.1-dev'
  
  pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
  pipe.to('cuda')
  
+ @spaces.GPU(duration=1500) # 启动期间允许的最大执行时长
+ def compile_transformer():
+     with spaces.aoti_capture(pipe.transformer) as call:
+         pipe("arbitrary example prompt")
+ 
+     exported = torch.export.export(
+         pipe.transformer,
+         args=call.args,
+         kwargs=call.kwargs,
+     )
+     return spaces.aoti_compile(exported)
+ 
+ compiled_transformer = compile_transformer()
+ spaces.aoti_apply(compiled_transformer, pipe.transformer)
  
  @spaces.GPU
  def generate(prompt):
      return pipe(prompt).images
  
  gr.Interface(generate, "text", "gallery").launch()
```
只需增加十几行代码，我们就成功让演示运行得更快（在 FLUX.1-dev 的情况下提升了 **1.7 倍**）。  

如果你想进一步了解提前编译，可以阅读 PyTorch 的 [AOTInductor 教程](https://docs.pytorch.org/tutorials/recipes/torch_export_aoti_python.html)。  

## 注意事项

现在我们已经展示了在 ZeroGPU 条件下可以实现的加速效果，接下来我们会讨论在这个设置中遇到的一些注意点。  

### 量化（Quantization）

提前编译可以与量化结合，从而实现更大的加速效果。对于图像和视频生成任务，FP8 的训练后动态量化方案提供了良好的速度与质量平衡。不过需要注意，FP8 至少需要 9.0 的 CUDA 计算能力才能使用。  
幸运的是，ZeroGPU 基于 H200，因此我们已经能够利用 FP8 量化方案。  

要在提前编译工作流中启用 FP8 量化，我们可以使用 [`torchao`](https://github.com/pytorch/ao) 提供的 API，如下所示：  

```diff
+ from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig

+ # 在导出步骤之前对 transformer 进行量化
+ quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())

exported_transformer = torch.export.export(
    pipe.transformer,
    args=call.args,
    kwargs=call.kwargs,
)
``` 
（你可以在 [这里](https://docs.pytorch.org/ao/stable/index.html) 找到更多关于 TorchAO 的详细信息。）  

接着，我们就可以按照上面描述的步骤继续进行。使用量化可以再带来 **1.2 倍** 的加速。  

### 动态形状（Dynamic shapes）

图像和视频可能具有不同的形状和尺寸。因此，在执行提前编译时，考虑形状的动态性也非常重要。`torch.export.export` 提供的原语让我们能够很容易地配置哪些输入需要被视为动态形状，如下所示。  

以 Flux.1-Dev 的 transformer 为例，不同图像分辨率的变化会影响其 `forward` 方法中的两个参数：  

- `hidden_states`：带噪声的输入潜变量，transformer 需要对其去噪。  
  它是一个三维张量，表示 `batch_size, flattened_latent_dim, embed_dim`。  
  当 batch size 固定时，随着图像分辨率变化，`flattened_latent_dim` 也会变化。  

- `img_ids`：一个二维数组，包含编码后的像素坐标，形状为 `height * width, 3`。  
  在这种情况下，我们希望让 `height * width` 是动态的。  

我们首先需要定义一个范围，用来表示（潜变量）图像分辨率可以变化的区间。  
为了推导这些数值范围，我们检查了 pipeline 中 [`hidden_states`](https://github.com/huggingface/diffusers/blob/0ff1aa910cf3d87193af79ec1ae4487be542e872/src/diffusers/pipelines/flux/pipeline_flux.py#L920) 的形状在不同图像分辨率下的变化。  
这些具体数值依赖于模型本身，需要人工检查并结合一定直觉。 对于 Flux.1-Dev，我们最终得到：  

```python
transformer_hidden_dim = torch.export.Dim('hidden', min=4096, max=8212)
``` 
接下来，我们定义一个映射，指定参数名称，以及在其输入值中哪些维度需要被视为动态：  

```python
transformer_dynamic_shapes = {
    "hidden_dim": {1: transformer_hidden_dim}, 
    "img_ids": {0: transformer_hidden_dim},
}
``` 

接下来，我们需要让动态形状对象的结构与示例输入保持一致。对于不需要动态形状的输入，必须将其设置为 `None`。这可以借助 PyTorch 提供的 `tree_map` 工具非常轻松地完成：  

```python
from torch.utils._pytree import tree_map

dynamic_shapes = tree_map(lambda v: None, call.kwargs)
dynamic_shapes |= transformer_dynamic_shapes
``` 

现在，在执行导出步骤时，我们只需将 `transformer_dynamic_shapes` 传递给 `torch.export.export`：  

```python
exported_transformer = torch.export.export(
    pipe.transformer,
    args=call.args,
    kwargs=call.kwargs,
    dynamic_shapes=dynamic_shapes,
)
``` 

> [!NOTE]  
> 可以参考 [这个 Space](https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic)，它详细说明了如何在导出步骤中把量化和动态形状结合起来使用。  

### 多重编译 / 权重共享

当模型的动态性非常重要时，仅依靠动态形状有时是不够的。  

例如，在 Wan 系列视频生成模型中，如果你希望编译后的模型能够生成不同分辨率的内容，就会遇到这种情况。在这种情况下，可以采用的方法是：**为每种分辨率编译一个模型，同时保持模型参数共享，并在运行时调度对应的模型**。  

这里有一个这种方法的最小示例：[zerogpu-aoti-multi.py](https://gist.github.com/cbensimon/8dc0ffcd7ee024d91333f6df01907916)。  
你也可以在 [Wan 2.2 Space](https://huggingface.co/spaces/zerogpu-aoti/wan2-2-fp8da-aoti-faster/blob/main/optimization.py) 中看到该范式的完整实现。  

### FlashAttention-3

由于 ZeroGPU 的硬件和 CUDA 驱动与 Flash-Attention 3（FA3）完全兼容，我们可以在 ZeroGPU Spaces 中使用它来进一步提升速度。FA3 可以与提前编译（AoT）配合使用，因此非常适合我们的场景。  

从源码编译和构建 FA3 可能需要几分钟时间，并且这个过程依赖于具体硬件。作为用户，我们当然不希望浪费宝贵的 ZeroGPU 计算时间。这时 Hugging Face 的 [`kernels` 库](https://github.com/huggingface/kernels) 就派上用场了，因为它提供了针对特定硬件的预编译内核。  

例如，当我们尝试运行以下代码时：  

```python
from kernels import get_kernel

vllm_flash_attn3 = get_kernel("kernels-community/vllm-flash-attn3")
``` 

它会尝试从 [`kernels-community/vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3) 仓库加载一个内核，该内核与当前环境兼容。  
否则，如果存在不兼容问题，就会报错。幸运的是，在 ZeroGPU Spaces 上这一过程可以无缝运行。这意味着我们可以在 ZeroGPU 上借助 `kernels` 库充分利用 FA3 的性能。  

这里有一个 [Qwen-Image 模型的 FA3 注意力处理器完整示例](https://gist.github.com/sayakpaul/ff715f979793d4d44beb68e5e08ee067#file-fa3_qwen-py)。  

## 提前编译的 ZeroGPU Spaces 演示

### 加速对比
- [未使用 AoTI 的 FLUX.1-dev](https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-dev-base)  
- [使用 AoTI 和 FA3 的 FLUX.1-dev](https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-dev-fa3-aoti) （__1.75 倍__ 加速）  

### 精选 AoTI Spaces
- [FLUX.1 Kontext](https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev)  
- [QwenImage Edit](https://huggingface.co/spaces/multimodalart/Qwen-Image-Edit-Fast)  
- [Wan 2.2](https://huggingface.co/spaces/zerogpu-aoti/wan2-2-fp8da-aoti-faster)  
- [LTX Video](https://huggingface.co/spaces/zerogpu-aoti/ltx-dev-fast)  

## 结论

Hugging Face Spaces 中的 ZeroGPU 是一项强大的功能，它为 AI 构建者提供了高性能算力。在这篇文章中，我们展示了用户如何借助 PyTorch 的提前编译（AoT）技术，加速他们基于 ZeroGPU 的应用。  

我们用 Flux.1-Dev 展示了加速效果，但这些技术并不仅限于这一模型。因此，我们鼓励你尝试这些方法，并在 [社区讨论](https://huggingface.co/spaces/zerogpu-aoti/README/discussions/1) 中向我们提供反馈。  

## 资源

- 访问 [Hub 上的 ZeroGPU-AOTI 组织](https://huggingface.co/zerogpu-aoti)，浏览一系列利用文中技术的演示。  
- 查看 `spaces.aoti_*` API 的[源代码](https://pypi-browser.org/package/spaces/spaces-0.40.1-py3-none-any.whl/spaces/zero/torch/aoti.py)，了解接口细节。  
- 查看 [Hub 上的 Kernels Community 组织](https://huggingface.co/kernels-community)。  
- 升级到 Hugging Face 的 [Pro](https://huggingface.co/pro)，创建你自己的 ZeroGPU Spaces（每天可获得 25 分钟 H200 使用时间）。  

*致谢：感谢 ChunTe Lee 为本文制作了精彩的缩略图。感谢 Pedro 和 Vaibhav 对文章提供的反馈。*  






