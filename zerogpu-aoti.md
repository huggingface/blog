---
title: "Make your ZeroGPU Spaces go brrr with ahead-of-time compilation"
thumbnail: /blog/assets/zerogpu-aoti/thumbnail.png
authors:
- user: cbensimon
- user: sayakpaul
- user: linoyts
- user: multimodalart
date: 2025-09-01
---

# Make your ZeroGPU Spaces go brrr with PyTorch ahead-of-time compilation

ZeroGPU lets anyone spin up powerful H200s in Hugging Face Spaces without keeping a GPU locked for idle traffic.
It’s efficient, flexible, and ideal for demos but it doesn’t always make full use of everything the GPU and CUDA stack can offer.
Generating images or videos can take a significant amount of time. Being able to squeeze out more performance, taking advantage of the H200 hardware, does matter in this case.

This is where PyTorch ahead-of-time (AoT) compilation comes in. Instead of compiling models on the fly (which doesn’t play nicely with ZeroGPU’s short-lived processes), AoT lets you optimize once and reload instantly. 

**The result**: snappier demos and a smoother experience, with speedups ranging from __1.3×–1.8×__ on models like Flux, Wan, and LTX 🔥

In this post, we’ll show how to wire up Ahead-of-Time (AoT) compilation in ZeroGPU Spaces. We'll explore advanced tricks like FP8 quantization and dynamic shapes, and share working demos you can try right away. If you cannot wait, we invite you to check out some ZeroGPU-powered demos on the [zerogpu-aoti](https://huggingface.co/zerogpu-aoti) organization.

## Table of Contents

- [What is ZeroGPU](#what-is-zerogpu)
- [PyTorch compilation](#pytorch-compilation)
- [Ahead-of-time compilation on ZeroGPU](#ahead-of-time-compilation-on-zerogpu)
- [Gotchas](#gotchas)
  - [Quantization](#quantization)
  - [Dynamic shapes](#dynamic-shapes)
  - [Multi-compile / shared weights](#multi-compile--shared-weights)
  - [FlashAttention-3](#flashattention-3)
- [AoT compiled ZeroGPU Spaces demos](#aot-compiled-zerogpu-spaces-demos)
- [Conclusion](#conclusion)
- [Resources](#resources)

## What is ZeroGPU

[Spaces](https://huggingface.co/spaces) is a platform powered by Hugging Face that allows ML practitioners to easily publish demo apps.

Typical demo apps on Spaces look like:

```python
import gradio as gr
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(...).to('cuda')

def generate(prompt):
    return pipe(prompt).images

gr.Interface(generate, "text", "gallery").launch()
```

This works great, but a GPU is reserved for the Space during its entire lifetime – even when it has no user activity.

When executing `.to('cuda')` on this line:

```python
pipe = DiffusionPipeline.from_pretrained(...).to('cuda')
```

PyTorch initializes the NVIDIA driver, which sets up the process on CUDA forever. This is not very resource-efficient given that app traffic is not perfectly smooth, but is rather extremely sparse and spiky.

ZeroGPU takes a just-in-time approach to GPU initialization. Instead of setting up the main process on CUDA, it automatically forks the process, sets it up on CUDA, runs the GPU tasks, and finally kills the fork when the GPU needs to be released.

This means that:

- When the app does not receive traffic, it doesn't use any GPU
- When it is actually performing a task, it will use one GPU
- It can use multiple GPUs as needed to perform tasks concurrently

Thanks to the Python `spaces` package, the only code change needed to get this behaviour is as follows:

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

By importing `spaces` and adding the `@spaces.GPU` decorator, we:

- Intercept PyTorch API calls to postpone CUDA operations
- Make the decorated function run in a fork when later called
- (Call an internal API to make the right device visible to the fork but this is not in the scope of this blogpost)

## PyTorch compilation

Modern ML frameworks like PyTorch and JAX have the concept of _compilation_ that can be used to optimize model latency or inference time. Behind the scenes, compilation applies a series of (often hardware-dependent) optimization steps such as operator fusion, constant folding, etc.

PyTorch (from 2.0 onwards) currently has two major interfaces for compilation:

- Just-in-time with `torch.compile`
- Ahead-of-time with `torch.export` + `AOTInductor`

[`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) works great in standard environments: it compiles your model the first time it runs, and reuses the optimized version for subsequent calls.

```diff
  from diffusers import DiffusionPipeline

  pipe = DiffusionPipeline.from_pretrained(...).to('cuda')
+ pipe.transformer.compile()

  pipe(prompt="photo of a dog looking like a pilot")
```

However, on ZeroGPU, given that the process is freshly spun up for (almost) every GPU task, it means that `torch.compile` can’t efficiently re-use compilation and is thus forced to rely on its [filesystem cache](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html#modular-caching-of-torchdynamo-torchinductor-and-triton) to restore compiled models. Depending on the model being compiled, this process takes from a few dozen seconds to a couple of minutes, which is way too much for practical GPU tasks in Spaces.

<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 5px;">
💡 In the above example, we only compile the transformer since, in these generative models, the transformer (or more generally, the denoiser) is the most computationally heavy component.
</div>

This is where **ahead-of-time (AoT) compilation** shines.

With AoT, we can export a compiled model once, save it, and later reload it instantly in any process, which is exactly what we need for ZeroGPU. This helps us reduce framework overhead and also eliminates cold-start timings typically incurred in just-in-time compilation.

But how can we do ahead-of-time compilation on ZeroGPU? Let’s dive in.

## Ahead-of-time compilation on ZeroGPU

If you want to learn more about how AoT compilation is supported in PyTorch, you can read the [PyTorch tutorial](https://docs.pytorch.org/tutorials/recipes/torch_export_aoti_python.html) on ahead-of-time compilation. But we will start from scratch in this post.

Let's go back to our ZeroGPU base example and unpack what we need for bringing in AoT compilation. For the purpose of this demo, we will use the `black-forest-labs/FLUX.1-dev` model:

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

Compiling a model ahead-of-time with PyTorch involves multiple steps:

### 1. Getting example inputs

Recall that we’re going to compile the model *ahead* of time. Therefore, we need to derive example inputs for the model. Note that these are the same kinds of inputs we expect to see during the actual runs. To capture those inputs, we will leverage the `spaces.aoti_capture` helper from the `spaces` package:

```python
with spaces.aoti_capture(pipe.transformer) as call:
    pipe("arbitrary example prompt")
```

When used as a context manager like so, it allows capturing the call of any callable (`pipe.transformer`, in our case), cancel it and put the captured input values inside `call.args` and `call.kwargs`.

### 2. Exporting the model

Now that we have example args and kwargs for our transformer component, we can export it to a PyTorch [`ExportedProgram`](https://docs.pytorch.org/docs/stable/export.html#torch.export.ExportedProgram) by using [`torch.export.export`](https://docs.pytorch.org/docs/stable/export.html#torch.export.export) utility:

```python
exported_transformer = torch.export.export(
    pipe.transformer,
    args=call.args,
    kwargs=call.kwargs,
)
```

An exported PyTorch program is a computation graph that represents the tensor computations along with the original model parameter values.

### 3. Compiling the exported model

Once the model is exported, compiling it is pretty straightforward.

A traditional AoT compilation in PyTorch often requires saving the model on disk so it can be later reloaded. In our case, we’ll leverage a helper function part of the `spaces` package: `spaces.aoti_compile`. It's a tiny wrapper around `torch._inductor.aot_compile` that manages saving and lazy-loading the model as needed. It is meant to be used like:

```python
compiled_transformer = spaces.aoti_compile(exported_transformer)
```

This `compiled_transformer` is now an AoT-compiled binary ready to be used for inference. 

### 4. Using the compiled model in the pipeline

Now we need to bind our compiled transformer to our original pipeline, i.e., the `pipeline`.

A naive and almost working approach is to simply patch our pipeline like `pipe.transformer = compiled_transformer`. Unfortunately, this approach does not work because it deletes important attributes like `dtype`, `config`, etc. Only patching the `forward` method does not work well either because we are then keeping original model parameters in memory, often leading to OOM errors at runtime.

`spaces` package provides a utility for this, too -- `spaces.aoti_apply`:

```python
spaces.aoti_apply(compiled_transformer, pipe.transformer)
```

Et voilà! It will take care of patching `pipe.transformer.forward` with our compiled model, as well as cleaning old model parameters out of memory.

### 5. Wrapping it all together

To perform the first three steps (intercepting input examples, exporting the model, and compiling it with PyTorch inductor), we need a real GPU. CUDA emulation that you get outside of `@spaces.GPU` function is not enough because compilation is truly hardware-dependent, for instance, relying on micro-benchmark runs to tune the generated code. This is why we need to wrap it all inside a `@spaces.GPU` function and then get our compiled model back to the root of our app. Starting from our original demo code, this gives:

```diff
  import gradio as gr
  import spaces
  import torch
  from diffusers import DiffusionPipeline
  
  MODEL_ID = 'black-forest-labs/FLUX.1-dev'
  
  pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
  pipe.to('cuda')
  
+ @spaces.GPU(duration=1500) # maximum duration allowed during startup
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

With just a dozen lines of additional code, we’ve successfully made our demo quite faster (**1.7x** faster in the case of FLUX.1-dev).

## Gotchas

Now that we have demonstrated the speedups one can realize under the constraints of operating with ZeroGPUs, we will discuss a few gotchas that came up while working with this setup.

### Quantization

AoT can be combined with quantization to deliver even greater speedups.
For image and video generation, the FP8 post-training dynamic quantization schemes deliver good speed-quality trade-offs.
However, FP8 requires a CUDA compute capability of at least 9.0 to work.
Thankfully, for ZeroGPUs, since they’re based on H200s, we can already take advantage of the FP8 quantization schemes.

To enable FP8 quantization within our AoT compilation workflow, we can leverage the APIs provided by [`torchao`](https://github.com/pytorch/ao) like so:

```diff
+ from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig

+ # Quantize the transformer just before the export step.
+ quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())

exported_transformer = torch.export.export(
    pipe.transformer,
    args=call.args,
    kwargs=call.kwargs,
)
```

(You can find more details about the library in [TorchAO docs](https://docs.pytorch.org/ao/stable/index.html))

And we can then proceed with the rest of the steps as outlined above. Using quantization provides another XXYZ% of speedup.

### Dynamic shapes

Images and videos can come in different shapes and sizes. Hence, it’s important to also account for shape dynamism when performing AoT compilation. The primitives provided by `torch.export.export` make it easily configurable to provide what inputs should be treated accordingly for dynamic shapes, as shown below.

For the case of Flux.1-Dev transformer, changes in different image resolutions will affect two of its `forward` arguments: 

- `hidden_states`: The noisy input latents, which the transformer is supposed to denoise. It’s a 3D tensor, representing `batch_size, flattened_latent_dim, embed_dim`. When the batch size is fixed,  it’s the `flattened_latent_dim` that will change for any changes made to image resolutions.
- `img_ids`: A 2D array of encoded pixel coordinates having a shape of `height * width, 3`. In this case, we want to make `height * width` dynamic.

We start by defining a range in which we want to let the (latent) image resolutions vary:

```python
transformer_hidden_dim = torch.export.Dim('hidden', min=4096, max=8212)
```

We then define a map of argument names and which dimensions in their input values we expect to be dynamic:

```python
transformer_dynamic_shapes = {
    "hidden_dim": {1: transformer_hidden_dim}, 
    "img_ids", {0: transformer_hidden_dim}
}
```

Then we need to make our dynamic shapes object replicate the structure of our example inputs. The inputs that do not need dynamic shapes must bet set to `None`. This can be done very easily with PyTorch [tree_map](https://github.com/pytorch/pytorch/blob/2f0de0ff9361ca4f2b1e6f9edbc600b5fb6abcd6/torch/utils/_pytree.py#L1341-L1373) utility:

```python
from pytorch.utils._pytree import tree_map

dynamic_shapes = tree_map(lambda v: None, call.kwargs)
dynamic_shapes |= transformer_dynamic_shapes
```

Now, when performing the export step, we simply supply `transformer_dynamic_shapes` to `torch.export.export`:

```python
exported_transformer = torch.export.export(
    pipe.transformer,
    args=call.args,
    kwargs=call.kwargs,
    dynamic_shapes=dynamic_shapes,
)
```

<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 5px;">
💡 Check out <a href="https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic" target="_blank">this Space</a> that shows how to use both quantization and dynamic shapes during the export step.
</div>

### Multi-compile / shared weights

Dynamic shapes is sometimes not enough when dynamism is too important.

This is, for instance, the case with the Wan family of video generation models if you want your compiled model to generate different resolutions.
One thing can be done in this case: compile one model per resolution while keeping the model parameters shared and dispatch the right one at runtime

Here is a minimal example of this approach: [zerogpu-aoti-multi.py](https://gist.github.com/cbensimon/8dc0ffcd7ee024d91333f6df01907916)

You can also see a fully working implementation of this paradigm in [Wan 2.2 Space](https://huggingface.co/spaces/zerogpu-aoti/wan2-2-fp8da-aoti-faster/blob/main/optimization.py)

### FlashAttention-3

Since the ZeroGPU hardware and CUDA drivers are perfectly compatible with Flash-Attention 3 (FA3), we can use it in our ZeroGPU Spaces to speed things up even further. FA3 works with ahead-of-time compilation. So, this is ideal for our case.

Compiling and building FA3 from source can take several minutes, and this process is hardware-dependent. As users, we wouldn’t want to lose precious ZeroGPU compute hours. This is where Hugging Face [`kernels` library](https://github.com/huggingface/kernels) comes to the rescue. It provides access to pre-built kernels that are compatible for a given hardware. For example, when we try to run:

```python
from kernels import get_kernel

vllm_flash_attn3 = get_kernel("kernels-community/vllm-flash-attn3")
```

It tries to load a kernel from the [`kernels-community/vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3) repository, which is compatible with the current setup. Otherwise, it will error out due to incompatibility issues. Luckily for us, this works seamlessly on the ZeroGPU Spaces. This means we can leverage the power of FA3 on ZeroGPU, using the `kernels` library.

Here is a [fully working example of an FA3 attention processor](https://gist.github.com/sayakpaul/ff715f979793d4d44beb68e5e08ee067#file-fa3_qwen-py) for the Qwen-Image model.

## AoT compiled ZeroGPU Spaces demos

### Speedup comparison
- [FLUX.1-dev without AoTI](https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-dev-base)
- [FLUX.1-dev with AoTI and FA3](https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-dev-fa3-aoti) (__1.75x__ speedup)

### Featured AoTI Spaces
- [FLUX.1 Kontext](https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev)
- [QwenImage Edit](https://huggingface.co/spaces/multimodalart/Qwen-Image-Edit-Fast)
- [Wan 2.2](https://huggingface.co/spaces/zerogpu-aoti/wan2-2-fp8da-aoti-faster)
- [LTX Video](https://huggingface.co/spaces/zerogpu-aoti/ltx-dev-fast)

## Conclusion

ZeroGPU within Hugging Face Spaces is a powerful feature that enables AI builders by providing access to powerful compute. In this post, we showed how users can benefit from PyTorch’s ahead-of-time compilation techniques to speed up their applications that leverage ZeroGPU. 

We demonstrate speedups with Flux.1-Dev, but these techniques are not limited to just this model. Therefore, we encourage you to give these techniques a try and provide us with feedback in this [community discussion](https://huggingface.co/spaces/zerogpu-aoti/README/discussions/1)

## Resources

- Visit our [ZeroGPU-AOTI org on the Hub](https://huggingface.co/zerogpu-aoti) to refer to a collection of demos that leverage the techniques discussed in this post.
- Browse `spaces.aoti_*` APIs [source code](https://pypi-browser.org/package/spaces/spaces-0.40.1-py3-none-any.whl/spaces/zero/torch/aoti.py) to learn more about the interface
- Check out [Kernels Community org on the hub](https://huggingface.co/kernels-community)
- Upgrade to [Pro subscription](https://huggingface.co/pro) on Hugging Face to create your own ZeroGPU Spaces (and get 25 minutes of H200 usage everyday)
