---
title: "Fast LoRA inference for Flux with Diffusers and PEFT"
thumbnail: /blog/assets/lora-fast/thumbnail.png
authors:
- user: sayakpaul
- user: BenjaminB
---

# Fast LoRA inference for Flux with Diffusers and PEFT

LoRA adapters provide a great deal of customization for models of all shapes and sizes. When it comes to image generation, they can empower the models with [different styles, different characters, and much more](https://huggingface.co/spaces/multimodalart/flux-lora-the-explorer). Sometimes, they can also be leveraged [to reduce inference latency](https://huggingface.co/ByteDance/Hyper-SD/). Hence, their importance is paramount, particularly when it comes to customizing and fine-tuning models.

In this post, we take the [Flux.1-Dev model](https://huggingface.co/black-forest-labs/FLUX.1-dev) for text-to-image generation because of its widespread popularity and adoption, and how to optimize its inference speed when using LoRAs (\~2.3x). It has over 30k adapters trained with it ([as reported](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.1-dev) on the Hugging Face Hub platform). Therefore, its importance to the community is significant.

Note that even though we demonstrate speedups with Flux, our belief is that our recipe is generic enough to be applied to other models as well. 

If you cannot wait to get started with the code, please check out the [accompanying code repository](https://github.com/huggingface/lora-fast).

### Table of contents

* [Hurdles in optimizing LoRA inference](#hurdles-in-optimizing-lora-inference)
* [Optimization recipe](#optimization-recipe)
* [Optimized LoRA inference on a consumer GPU](#optimized-lora-inference-on-a-consumer-gpu)
* [Conclusion](#conclusion)

# Hurdles in optimizing LoRA inference

When serving LoRAs, it is common to hotswap (swap in and swap out different LoRAs) them. A LoRA changes the base model architecture. Additionally, LoRAs can be different from one another – each one of them could have varying ranks and different layers they target for adaptation. To account for these dynamic properties of LoRAs, we must take necessary steps to ensure the optimizations we apply are robust.

For example, we can apply `torch.compile` on a model loaded with a particular LoRA to obtain speedups on inference latency. However, the moment we swap out the LoRA with a different one (with a potentially different configuration), we will run into recompilation issues, causing slowdowns in inference.

One can also fuse the LoRA parameters into the base model parameters, run compilation, and unfuse the LoRA parameters when loading new ones. However, this approach will again encounter the problem of recompilation whenever inference is run, due to potential architecture-level changes. 

Our optimization recipe takes into account the above-mentioned situations to be as realistic as possible. Below are the key components of our optimization recipe:

* Flash Attention 3 (FA3)  
* `torch.compile`  
* FP8 quantization from TorchAO   
* Hotswapping-ready

Note that amongst the above-mentioned, FP8 quantization is lossy but often provides the most formidable speed-memory trade-off. Even though we tested the recipe primarily using NVIDIA GPUs, it should work on AMD GPUs, too.

# Optimization recipe

In our previous blog posts ([post 1](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/) and [post 2](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)), we have already discussed the benefits of using the first three components of our optimization recipe. Applying them one by one is just a few lines of code:

```py
from diffusers import DiffusionPipeline, TorchAoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from utils.fa3_processor import FlashFluxAttnProcessor3_0
import torch

# quantize the Flux transformer with FP8
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16, 
    quantization_config=PipelineQuantizationConfig(
        quant_mapping={"transformer": TorchAoConfig("float8dq_e4m3_row")}
    )
).to("cuda")

# use Flash-attention 3
pipe.transformer.set_attn_processor(FlashFluxAttnProcessor3_0())

# use torch.compile()
pipe.transformer.compile(fullgraph=True, mode="max-autotune")

# perform inference
pipe_kwargs = {
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 28,
    "max_sequence_length": 512,
}

# first time will be slower, subsequent runs will be faster
image = pipe(**pipe_kwargs).images[0]
```
*The FA3 processor comes from [here](https://github.com/huggingface/lora-fast/blob/main/utils/fa3_processor.py).*

The problems start surfacing when we try to swap in and swap out LoRAs into a compiled diffusion transformer (`pipe.transformer`) without triggering recompilation.

Normally, loading and unloading LoRAs will require recompilation, which defeats any speed advantage gained from compilation. Thankfully, there is a way to avoid the need for recompilation. By passing `hotswap=True`, diffusers will leave the model architecture unchanged and only exchange the weights of the LoRA adapter itself, which does not necessitate recompilation.

```py
pipe.enable_lora_hotswap(target_rank=max_rank)
pipe.load_lora_weights(<lora-adapter-name1>)
# compile *after* loading the first LoRA 
pipe.transformer.compile(mode="max-autotune", fullgraph=True)
image = pipe(**pipe_kwargs).images[0]
# from this point on, load new LoRAs with `hotswap=True`
pipe.load_lora_weights(<lora-adapter-name2>, hotswap=True)
image = pipe(**pipe_kwargs).images[0]
```
*(As a reminder, the first call to `pipe` will be slow as `torch.compile` is a just-in-time compiler. However, the subsequent calls should be significantly faster.)*

This generally allows for swapping LoRAs without recompilation, but there are limitations:

* We need to provide the maximum rank among all LoRA adapters ahead of time. Thus, if we have one adapter with rank 16 and another with 32, we need to pass `max_rank=32`.   
* LoRA adapters that are hotswapped in can only target the same layers, or a subset of layers, that the first LoRA targets.  
* Targeting the text encoder is not supported yet.

For more information on hotswapping in Diffusers and its limitations, visit the [hotswapping section of the documentation](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference#hotswapping).

The benefits of this workflow become evident when we look at the inference latency without using compilation with hotswapping.

| Option | Time (s) ⬇️ | Speedup (vs baseline) ⬆️ | Notes |
| ----- | ----- | ----- | ----- |
| baseline | 7.8910 | – | Baseline |
| optimized | 3.5464 | 2.23× | Hotswapping \+ compilation without recompilation hiccups (FP8 on by default) |
| no\_fp8 | 4.3520 | 1.81× | Same as optimized, but with FP8 quantization disabled |
| no\_fa3 | 4.3020 | 1.84× | Disable FA3 (flash‑attention v3) |
| baseline \+ compile | 5.0920 | 1.55× | Compilation on, but suffers from intermittent recompilation stalls |
| no\_fa3\_fp8 | 5.0850 | 1.55× | Disable FA3 and FP8 |
| no\_compile\_fp8 | 7.5190 | 1.05× | Disable FP8 quantization and compilation |
| no\_compile | 10.4340 | 0.76× | Disable compilation: the slowest setting |

**Key takeaways**:

* The “regular \+ compile” option provides a decent speedup over the regular option, but it incurs recompilation issues, which increase the overall execution time. In our benchmarks, we don’t present the compilation time.  
* When recompilation problems are eliminated through hotswapping (also known as the “optimized” option), we achieve the highest speedup.  
* In the “optimized” option, FP8 quantization is enabled, which can lead to quality loss. Even without using FP8, we get a decent amount of speedup (“no\_fp8” option).  
* For demonstration purposes, we use a pool of two LoRAs for hotswapping with compilation. For the full code, please refer to the accompanying [code repository](https://github.com/huggingface/lora-fast).

The optimization recipe we have discussed so far assumes access to a powerful GPU like H100. However, what can we do when we’re limited to using consumer GPUs such as RTX 4090? Let’s find out.

# Optimized LoRA inference on a consumer GPU

Flux.1-Dev (without any LoRA), using the Bfloat16 data-type, takes \~33GB of memory to run. Depending on the size of the LoRA module, and without using any optimization, this memory footprint can increase even further. Many consumer GPUs like the RTX 4090 only have 24GB. Throughout the rest of this section, we will consider an RTX 4090 machine as our testbed.

First, to enable end-to-end execution of Flux.1-Dev, we can apply CPU offloading wherein components that are not needed to execute the current computation are offloaded to the CPU to free more accelerator memory. Doing so allows us to run the entire pipeline in \~22GB in **35.403 seconds** on an RTX 4090\. Enabling compilation can reduce the latency down to **31.205 seconds** (1.12x speedup). In terms of code, it’s just a few lines:

```py
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
# Instead of full compilation, we apply regional compilation
# here to take advantage of `fullgraph=True` and also to reduce
# compilation time. More details can be found here:
# https://hf.co/docs/diffusers/main/en/optimization/fp16#regional-compilation
pipe.transformer.compile_repeated_blocks(fullgraph=True)
image = pipe(**pipe_kwargs).images[0]

```

Notice that we didn’t apply the FP8 quantization here because it’s not supported with CPU offloading and compilation (supporting [issue thread](https://github.com/pytorch/pytorch/issues/141548)). Therefore, just applying FP8 quantization to the Flux Transformer isn’t enough to mitigate the memory exhaustion problem, either. In this instance, we decided to remove it.

Therefore, to take advantage of the FP8 quantization scheme, we need to find a way to do it without CPU offloading. For Flux.1-Dev, if we additionally apply quantization to the T5 text encoder, we should be able to load and run the complete pipeline in 24GB. Below is a comparison of the results with and without the T5 text encoder being quantized (NF4 quantization from [`bitsandbytes`](https://huggingface.co/docs/diffusers/main/en/quantization/bitsandbytes)).

![te_quantized_results](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/lora-fast/image_1_lora_fast.png)

As we can notice in the figure above, quantizing the T5 text encoder doesn’t incur too much of a quality loss. Combining the quantized T5 text encoder and FP8-quantized Flux Transformer with `torch.compile`  gives us somewhat reasonable results – **9.668 seconds** from 32.27 seconds (a massive \~3.3x speedup) without a noticeable quality drop. 

![quantized_compiled_results](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/lora-fast/image_2_lora_fast.png)

It is possible to generate images with 24 GB of VRAM even without quantizing the T5 text encoder, but that would have made our generation pipeline slightly more complicated.

We now have a way to run the entire Flux.1-Dev pipeline with FP8 quantization on an RTX 4090\. We can apply the previously established optimization recipe for optimizing LoRA inference on the same hardware. Since FA3 isn’t supported on RTX 4090, we will stick to the following optimization recipe with T5 quantization newly added to the mix:

* FP8 quantization  
* `torch.compile`  
* Hotswapping-ready  
* T5 quantization (with NF4)

In the table below, we show the inference latency numbers with different combinations of the above components applied.

| Option | Key args flags | Time (s) ⬇️ | Speedup (vs baseline) ⬆️ |
| ----- | ----- | ----- | ----- |
| baseline | `disable_fp8=False disable_compile=True`  `quantize_t5=True offload=False` | 23.6060 | – |
| optimized | `disable_fp8=False disable_compile=False`  `quantize_t5=True offload=False` | 11.5715 | 2.04× |

**Quick notes**:

* Compilation provides a massive 2x speedup over the baseline.  
* The other options yielded OOM errors even with offloading enabled.

# Conclusion

This post outlined an optimization recipe for fast LoRA inference with Flux, demonstrating significant speedups. Our approach combines Flash Attention 3, `torch.compile`, and FP8 quantization while ensuring hotswapping capabilities without recompilation issues. On high-end GPUs like the H100, this optimized setup provides a 2.23x speedup over the baseline.

For consumer GPUs, specifically the RTX 4090, we tackled memory limitations by introducing T5 text encoder quantization (NF4) and leveraging regional compilation. This comprehensive recipe achieved a substantial 2.04x speedup, making LoRA inference on Flux viable and performant even with limited VRAM. The key insight is that by carefully managing compilation and quantization, the benefits of LoRA can be fully realized across different hardware configurations.

Hopefully, the recipes from this post will inspire you to optimize your
LoRA-based use cases, benefitting from speedy inference.

## Resources

Below is a list of the important resources that we cited throughout this post:

* [Presenting Flux Fast: Making Flux go brrr on H100s](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/)
* [torch.compile and Diffusers: A Hands-On Guide to Peak Performance](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
* [LoRA guide in Diffusers](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)