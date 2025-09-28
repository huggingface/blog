---
title: "Optimizing diffusion inference for production-ready speeds - I" 
thumbnail: /blog/assets/productionizing-diffusion/productionizing-diffusion-thumbnail-1.png
authors:
- user: a-r-r-o-w
---

# Optimizing diffusion inference for production-ready speeds - I

Diffusion models have rapidly advanced generative modeling across a wide range of modalities - from images and video to music, 3D objects, and even text generation and world simulations recently. They are now central to state-of-the-art image and video generation, offering high-quality, controllable, and diverse outputs. However, their computational cost remains a bottleneck for real-world deployment. In this series, we explore techniques to optimize diffusion inference for text-to-image and text-to-video generation.

This post is first in a four-part series. We will cover the following topics:
1. How text-to-image diffusion models work and their computational challenges?
2. Standard optimizations for transformer-based diffusion models
3. Going deep: using faster kernels, non-trivial fusions, precomputations
4. Context parallelism
5. Quantization
6. Caching
7. LoRA
8. Training
9. Practice: Wan text-to-video
10. Optimizing inference for uncommon deployment environments using Triton

| Post | Topics covered |
|------|----------------|
| Optimizing diffusion inference for production-ready speeds - I   | 1, 2 |
| Optimizing diffusion inference for production-ready speeds - II  | 3, 4 |
| Optimizing diffusion inference for production-ready speeds - III | 5, 6 |
| Optimizing diffusion inference for production-ready speeds - IV  | 7, 8, 9, 10 |

The code for the entire series is available at [huggingface/productionizing-diffusion](https://github.com/huggingface/productionizing-diffusion). For this post, refer to the `post_1` directory. The guides are written to work on A100/H100 or better GPUs, but the ideas can be adapted to other hardware as well.

## Table of contents

- [How diffusion models work](#how-diffusion-models-work)
- [Setup](#setup)
  - [Environment](#environment)
  - [Baseline](#establishing-a-baseline)
- [Computational challenges](#computational-challenges)
- [Standard optimizations](#standard-optimizations)
- [Benchmarks](#benchmarking)
  - [Cost Analysis](#cost-analysis)
- [Additional reading](#references--additional-reading)

Let's begin by understanding how diffusion models work and their computational challenges.

## How diffusion models work

Diffusion models are a class of generative models that learn to predict a target data distribution by gradually moving towards it starting from a source data distribution. In the case of text-to-image generation, the source distribution is [gaussian noise](https://en.wikipedia.org/wiki/Normal_distribution), and the target distribution is the aeshetically pleasing images that we all love and generate. The model is trained to iteratively refine and denoise some starting random noise into a coherent image over many steps, possibly guided by different conditioning signals like text prompts, images (image-to-image task), or even other modalities like depth maps/sketches (control-to-image tasks), audio and more.

![Diffusion forward process](https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/diffusion_forward_process.png)

<sup> The above image illustrates the diffusion forward process, where a clean image is gradually corrupted by adding noise over a series of steps. The model is trained to reverse this process, so it can take a noisy image and iteratively denoise it back to a clean image. </sup>

For a more technical and detailed explanation of diffusion models, check out the [References & Additional Reading](#references-&-additional-reading) section at the end of this post.

In this post, we will be taking a look at [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) - a state-of-the-art text-to-image diffusion model. Using it requires a total of 4 models - a `denoiser` for reverse diffusion process, two `text encoders` ([T5](https://huggingface.co/google/t5-v1_1-xl) and [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)) for conditionally guiding the denoiser with text prompts, and a `variational autoencoder` ([VAE](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/vae)) for decoding the generated latent representations into images.

To optimize the inference speed of FLUX.1-dev, we will first focus on the denoiser model, which is the most computationally expensive part of the generation pipeline. The denoiser follows a [transformer-based](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) architecture, containing a total of 57 transformer blocks (19 dual stream blocks and 38 single stream blocks), with a total of 12 billion parameters. See the [config](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/transformer/config.json) for more details.

The distinction between dual and single stream blocks in the Flux architecture can be summarized as follows:
- **Dual stream blocks**: These blocks have separated MLP and normalization layers for image and text tokens. During attention, the tokens are concatenated together to perform "joint attention" with all tokens. Information such as the timestep and guidance conditioning are injected by modulating the latent streams at both the MLP and Attention layers. Intuitively, this might let the blocks learn independent representations for image and text understanding, while also allowing full interaction with each other.
- **Single stream blocks**: These blocks have a single MLP and normalization layer for image and text tokens, and also perform joint attention. Intuitively, this might allow the model to learn a more unified representation of how each kind of token influences the other. Instead of the standard attention-followed-by-MLP structure, these blocks concatenate the output of both layers into a single large embedding per token, and project it back to the original embedding dimension.

## Setup

### Environment

Install the requirements:

```bash
git clone https://github.com/huggingface/productionizing-diffusion
cd productionizing-diffusion/

uv venv venv
source venv/bin/activate

uv pip install torch==2.6 torchvision --index-url https://download.pytorch.org/whl/cu124 --verbose
uv pip install -r requirements.txt

# Make sure to have CUDA 12.4 or 12.8 (this is the only version I've tested, so you
# might have to do things differently for other versions when setting up FA2)
# https://developer.nvidia.com/cuda-12-4-0-download-archive

# Flash Attention 2 (optional, FA3 is recommended and much faster for H100, while Pytorch's cuDNN backend is
# good for both A100 and H100)
# For Python 3.10, use pre-built wheel or build from source
MAX_JOBS=4 uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation --verbose

# Flash Attention 3
# Make sure you have atleast 64 GB CPU RAM when building from source otherwise
# the installation will crash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper
# We install v2.7.4.post1 because the latest release (2.8.x) might cause
# some installation issues which are hard to debug
# Update: 2.8.3 seems to install without any problems on CUDA 12.8 and Pytorch 2.10 nightly.
git checkout v2.7.4.post1
python setup.py install
```

### Establishing a baseline

To get started, we will use [🤗 Diffusers](https://github.com/huggingface/diffusers). Reading through the [docs](https://huggingface.co/docs/diffusers/en/api/pipelines/flux#diffusers.FluxPipeline), the following snippet can be used to find the end-to-end generation time.

```python
import torch
from diffusers import FluxPipeline

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat holding a sign that says 'Hello, World'"

# Warmup (very important!)
for _ in range(2):
    _ = pipe(prompt, height=1024, width=1024, num_inference_steps=2, guidance_scale=4.0).images[0]

# Benchmark (ideally, should be averaged over multiple runs)
start_event.record()
image = pipe(prompt, height=1024, width=1024, num_inference_steps=28, guidance_scale=4.0).images[0]
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event) / 1000.0
print(f"time: {elapsed_time:.3f}s")

image.save("output.png")
```

This takes `15.815` seconds on an A100 80GB and `6.936` seconds on an H100 with my environment setup. This is our baseline end-to-end latency for generating a single image.

## Computational challenges

From the above example snippet, we see that the model is generating `1024x1024` resolution images using `28` inference steps. This is the end-to-end generation time, which includes the time taken by the text encoders, the denoiser and the VAE decoder. If we further benchmark each component individually, we find that the denoiser takes majority of the time (15+ seconds on A100).

```python
import torch
from diffusers import FluxPipeline

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat holding a sign that says 'Hello, World'"
height = 1024
width = 1024

with torch.inference_mode():
    # Warmup (very important!)
    for _ in range(2):
        _ = pipe(prompt, height=height, width=width, num_inference_steps=2, guidance_scale=4.0).images[0]

    # Benchmark (ideally, should be averaged over multiple runs)
    start_event.record()
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, device="cuda", max_sequence_length=512)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_text_encoders = start_event.elapsed_time(end_event) / 1000.0

    start_event.record()
    latents = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        height=height,
        width=width,
        num_inference_steps=28,
        guidance_scale=4.0,
        output_type="latent",
    ).images
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_denoiser = start_event.elapsed_time(end_event) / 1000.0

    start_event.record()
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_vae = start_event.elapsed_time(end_event) / 1000.0

    print(f"text_encoders time: {elapsed_time_text_encoders:.3f}s")
    print(f"     denoiser time: {elapsed_time_denoiser:.3f}s")
    print(f"          vae time: {elapsed_time_vae:.3f}s")

    image.save("output.png")
```

```
text_encoders time: 0.060s
     denoiser time: 15.708s
          vae time: 0.164s
```

A generation is only as fast as the slowest component in the overall pipeline! Any amount of optimization we do for the text encoders and VAE will not help us at this point, so let's focus on the denoiser. Some things are quickly apparent here by looking at the examples and some internal implementation details:
- The latent-space resolution for Flux is 64x smaller than pixel-space (8x across height and width each). So, a `1024x1024` resolution image corresponds to a `128x128` latent image. [Latent diffusion models](https://arxiv.org/abs/2112.10752) are trained to operate in a compressed latent space and so are very efficient and fast. Pixel-space images have 3 channels (RGB), while the latent-space images, here, have `16` channels.
- The denoiser is run `28` times sequentially. This is the default recommendation by model authors, but can be lowered for faster generation speed. However, allowing the model to denoise latents gradually over many steps yields better quality images, so there is a trade-off here.
- Since we're using a transformer architecture here, we can think of each latent pixel as a "latent image token". Each token has a feature/embedding dimension of `3072`. In total, we have `4096` tokens for a `1024x1024` image (`128x128` latent image, which is further [pixel-unshuffled](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html) to compress the latent image to a `64x64` resolution whilst increasing the number of input channels to `64`).
- Text encoder generates `512` "text tokens" (we default to `512`, but as we shall later see, this can be reduced without loss of image quality). This is the text conditioning signal for the denoiser. Overall, we have a total of `4608` tokens (`4096` image tokens + `512` text tokens).
- Doing a forward pass through the transformer involves performing various operations on tensors of shape `[1, 4608, 3072]` (assuming single image generation). If we could somehow reduce the number of tokens, or embedding dimension, or lower the interaction with all 12B parameters of the model, we could speed up the inference time by a lot. We will explore this later, but for now, let's focus on some standard optimizations.

To summarize a non-exhaustive list of computational challenges we face here are:
- Large number of model parameters (12B) and their interactions with a large sequence length every inference step
- Large number of inference steps required to generate high-quality images
- Unwanted CPU/GPU synchronizations
- Model architecture written suboptimally for inference
- Long kernel launch overheads

We'll iteratively apply optimizations to address these challenges without compromising image quality.

## Standard optimizations

Before beginning optimizations for training or inference, I like to have the entire model definition and related training/inference implementation in front of me in a single file. It helps better understand the data flow, tensor shapes, and various operations being performed. Following this practice, the repository contains a single file inference implementation of Flux in `post_1/single_file_inference.py`. We will iteratively apply optimizations on this code. As our only focus is on optimizing text-to-image inference, we remove the irrelevant code paths like ControlNet, LoRA, and gradient checkpointing for simplicity from the original [Diffusers](https://github.com/huggingface/diffusers/blob/b9e99654e1a08328f5e9365eb368994511d91ac2/src/diffusers/models/transformers/transformer_flux.py) code.

The following subsections list out some commonly used optimizations that can be easily applied to most models. The end result of these optimizations is available in `post_1/optimized_inference.py`.

### Running in bf16 mixed precision

This idea is not really considered an "optimization" these days, as it is the default precision for running most models (diffusion or not). However, it is worth mentioning because some parts of the model implementations in many research and user-facing codebases are forcefully run in full precision (FP32) for numerical stability. Some examples include input/output projection layers, RoPE, normalization layers, etc. FP32 inference is extremely slow compared to BF16, so it can sometimes cause unexpected performance degradation. It is recommended to use the [Torch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) to find out bottlenecks in your model.

Here, we try to run the entire Flux model in BF16 precision. An important point to note is that BF16 RoPE/normalization has been found harmful for training and may lead to quality degradation in some cases (especially so in video generation models). For Flux inference, this does not lead to much quality degradation from my personal testing.

<table>
  <tr>
    <th> bf16 model + bf16 RoPE </th>
    <th> bf16 model + fp32 RoPE </th>
  </tr>

  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/rope-comparison-bf16.png" width="384px" /></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/rope-comparison-fp32.png" width="384px" /></td>
  </tr>
</table>

### torch.compile

A pytorch model implementation is simply a series of "primitive" operations on tensors. Each model can be represented with a computation graph of these operations. The graph structure is built automatically by pytorch. [`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) is a tool that enables just-in-time (JIT) compilation of these computation graphs into highly optimized code paths and kernels for faster execution. The optimized kernels can either call into high-performance math/matrix-multiplication/linear-algebra libraries or expert-optimized kernels (for example, [cublas](https://docs.nvidia.com/cuda/cublas/)/[cutlass](https://github.com/NVIDIA/cutlass), or generate [Triton](https://triton-lang.org/) programs on-the-fly from [templates](https://github.com/pytorch/pytorch/blob/d5781c8d21b3dca35715a093ba52c5698551ad9b/torch/_inductor/kernel/mm.py#L80), fuse multiple operations together, or do various other passes that improve the computation graph structure for speed and minimal overhead. It is a very powerful tool that can yield significant speedups.

The internal details of `torch.compile` are complex and beyond the scope of this post. For those interested in taking a look, there are two major components:
- [Torch Dynamo](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html)
- [Torch Inductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)

To optimize Flux, it is as simple as wrapping the different components in a `torch.compile` call:

```python
pipe.text_encoder = torch.compile(pipe.text_encoder, fullgraph=True, mode="default", dynamic=True)
pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, fullgraph=True, mode="default", dynamic=True)
pipe.transformer = torch.compile(pipe.transformer, fullgraph=True, mode="default", dynamic=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, fullgraph=True, mode="default", dynamic=True)
```

Note that we use `dynamic=True` here because we don't know ahead-of-time the image resolutions that users might want to generate. Dynamic shapes are usually worse for performance, which is why many inference-providers allow only specific resolutions to be generated. Shape specializing your models can yield ginormous speedups, but is very developer intensive and not ideal if you're working with multiple models and algorithms.

`torch.compile` also supports three other modes that can be explored - `reduce-overhead`, `max-autotune` and `max-autotune-no-cudagraphs`. See the [docs](https://pytorch.org/docs/stable/generated/torch.compile.html) for more details. These modes take a lot longer to finish compilation because they perform various lowering/autotuning/codegen passes to find the best and fastest kernel implementations for your model. It is a tradeoff, however, because the compilation time can be very long (several minutes to hours) for large models. On the positive side, it can be done once and the results can be cached for future use. See [MegaCache](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) and [compile-time caching](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html) for more details.

There are various compilation-related configuration options that can be set by users based on their use-cases and environment setting. These flags can help squeeze out extra performance in many cases. Some of the flags that'll be useful for us are:

```python
import torch._dynamo.config
import torch._inductor.config
import torch._higher_order_ops.auto_functionalize as af

# This is enabled by default and may cause 1-5% performance degradation in some
# cases. It is, however, recommended to set this to True for faster compilation times
# since most of the other performance bottlenecks, unrelated to this, are more significant.
# See https://pytorch.org/blog/pytorch2-5/ for more details.
torch._dynamo.config.inline_inbuilt_nn_modules = False
# For compiling with dynamic=False, we want to cache optimized kernels for many
# different shapes.
torch._dynamo.config.cache_size_limit = 128

# Fuse pointwise convolutions into matrix multiplications
torch._inductor.config.conv_1x1_as_mm = True
# Autotuning related flags for better kernel selection
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.coordinate_descent_tuning = True
# Make autotuning progress bar visible
torch._inductor.config.disable_progress = False
torch._inductor.config.fx_graph_cache = True
# Disable epilogue fusions into matrix multiplications.
# In most cases, fused operations may lead to better performance, but from
# my benchmarks, this seems to lose 1-2% performance. This is most likely
# due to inductor generating slower triton kernels for fused operations
# instead of calling into hand-optimized kernels from libraries, but I
# haven't dug deeper into this.
torch._inductor.config.epilogue_fusion = False
# Try to fuse operation even if they don't share common memory reads
torch._inductor.config.aggressive_fusion = True
# For multi-GPU setups, run reordering pass to increase compute-communication overlap
torch._inductor.config.reorder_for_compute_comm_overlap = True
# Better tensor core utilization for matrix multiplication by padding
# input tensor shapes for better memory alignment
torch._inductor.config.shape_padding = True
# Enable persistent TMA matrix multiplications on Hopper (H100) and higher GPUs
# which significantly speeds up large matrix multiplications
torch._inductor.config.triton.enable_persistent_tma_matmul = True
af.auto_functionalized_v2._cacheable = True
af.auto_functionalized._cacheable = True

# Enable usage of TF32 cores, which speedup FP32 matrix multiplications
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable BF16 reductions in matrix multiplications (unfortunately, no such option
# exists for doing the same with pytorch generated triton kernels yet)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
```

It is recommended to try out `torch.compile` on your model first before implementing any custom optimizations. The generated kernels and fusions are surprisingly hard to outperform without spending large number of developer hours, so it is worth spending time understanding the generated kernels before implementing custom optimizations. However, `torch.compile` is not a silver bullet and may not always yield the best performance. It is important to profile the model and identify bottlenecks that can't be optimized using the strategies available in ML compilers.

### `channels_last` memory format for VAE

A simple yet effective optimization for models with convolution layers is to use the `channels_last` memory format. This layout improves memory access patterns, often leading to better throughput on modern GPUs. More information can be found in [this PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html).

For the VAE, which already runs extremely fast, the benefit of `channels_last` is marginal unless the model is serving a large volume of concurrent image requests. This optimization is not relevant for transformer-based models and does not impact generation latency there. Nonetheless, it's worth applying on the VAE when scaling up inference workloads.

### Fused QKV

Recall that the feedforward/linear projections in the model are simple matrix multiplication operations.

If the [arithmetic intensity](https://en.wikipedia.org/wiki/Roofline_model) of our models' matrix multiplications is low, we leave the GPU's compute units waiting for more work, and these operations will be "memory-bounded". That is, the speed of the operation is limited by how fast data can be read from or written to memory, rather than how fast the GPU can perform computations. This is undesirable. Also, every time a matmul is performed, a new kernel launch takes place on the GPU and this has some overheads involved.

By fusing multiple small matmuls into a single large one, we increase the arithmetic intensity and try to move closer to "compute-bounded" performance, which modern GPUs excel at. Note that matmuls are already compute-bounded for common DL workload shapes (`M, N, K > 1024`). The benefit with fusing these compute-bounded operations is that we incur fewer kernel launch overheads as well as keep the GPU continuously used for larger problem shapes.

In practice, any set of linear layers, i.e. matmuls, that operate on the same input can be fused. The math behind this is trivial (left as an exercise to the reader 😛). In the attention layer of Flux, we can see two such prime candidates for this - image and text QKV projections. Fusing them together yields a significant speedup!

> Repeat with me: "Big matmuls are always better than smaller matmuls."

### Fusing scheduler step into transformer

In flow matching models, the diffusion transformer predicts a velocity field, which is then used to update the current target distribution estimates (which start as gaussian noise) towards the actual target distribution (the desired modality, e.g., images). This update can be done with any ODE/SDE solver - the most famously used is the Euler method. In diffusers, this [update step](https://github.com/huggingface/diffusers/blob/425a715e35479338c06b2a68eb3a95790c1db3c5/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L461) is performed in the scheduler, which is outside the transformer forward pass. The operation is simply: `x_t_minus_1 = x_t + velocity * dt`, where `x_t` is the current estimate of the source distribution, `velocity` is the output of the transformer, and `dt` is the time delta. For more complex solvers, a lot more operations are performed to do this update.

This poses a problem: because these operations are outside the model graph, `torch.compile` cannot optimize or fuse these operations (typically add/mul). By folding the scheduler update into the transformer forward pass, they become eligible for fusion. This reduces kernel launch overhead, improves memory locality, and enables other backend optimizations.

While such a fusion may not visibly affect end-to-end latency in small-scale benchmarks, it improves compiler effectiveness and unlocks downstream optimizations. The principle here is simple: expose as much computation as possible to the compiler. More on this and operator fusion strategies in the next post. For deeper context, see [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html).

### Prompt-length awareness

As described above, the overall number of "tokens" passed through the transformer comprises of image tokens and text tokens. In the standard case of `1024x1024` resolution generations (`4096` image tokens) with the default `512` text sequence length (text tokens), we are dealing with a total context length of `4608`.

The transformer architecture consists of two main building blocks - Attention and MLP layers, among other smaller layers. The computation for a particular set of tokens scales linearly across all layers of the model except the attention layers, where it scales quadratically due to every token needing to "interact" with every other token in the sequence dimension. If we reduce the number of tokens somehow, we will end up reducing the overall number of interactions and speed up the inference time.

For any given resolution, the number of image tokens is fixed and cannot be changed. But, there's an optimization opportunity with text tokens! When serving users with your optimized inference engines, some requests may contain descriptive and long prompts, but some others might be shorter. For example, consider the following two prompts:
- "A cat holding a sign that says 'Hello, World'"
- "The King of Hearts card transforms into a 3D hologram that appears to be made of cosmic energy. As the King emerges, stars and galaxies swirl around him, creating a sense of traveling through the universe. The King's attire is adorned with celestial patterns, and his crown is a glowing star cluster. The hologram floats in front of you, with the background shifting through different cosmic scenes, from nebulae to black holes. Atmosphere: Perfect for space-themed events, science fiction conventions, or futuristic tech expos."

The default behaviour of our implementation, so far, is to always encode every prompt to a fixed pre-defined length of `512` tokens. This is the recommended setting by the model authors. Short prompts, such as the example above, encode to far fewer tokens (lesser than `64`) but are padded with special padding tokens to match the expected length (in its training, the model learns to ignore these padding tokens and only follow the conditional signal from the "normal" text tokens). This may be wasteful but it really depends on the model being optimized and how it was trained. For Flux, we can get away with using lesser text tokens without any meaningful loss in quality.

As a simple optimization, we can bucket different user prompts based on their tokenized lengths into either `128`, `256`, `384` or `512` tokens (or use more finegrained bucket sizes). Following is an example that shows there is no meaningful quality loss when using varying text sequence lengths for the same short prompt:

```python
import torch
from diffusers import FluxPipeline
from diffusers.utils import make_image_grid

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat holding a sign that says 'Hello, World'"
images = []
for length in [128, 256, 384, 512]:
    image = pipe(
        prompt,
        num_inference_steps=28,
        guidance_scale=4.0,
        max_sequence_length=length,
        generator=torch.Generator().manual_seed(19872364),
    ).images[0]
    images.append(image)

make_image_grid(images, rows=1, cols=len(images)).save("output.png")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/cat-varying-text-seq-len.png" height="512px" />

<sup> We support this optimization in the example scripts, but every benchmark result shared when comparing to other implementations is with the default `512` text tokens for a fair comparison. </sup>

For a quick inference speed comparison of different attention implementations, you can check [this](https://gist.github.com/a-r-r-o-w/58425fd303633e3c3702283b4687599d) snippet out.

### CUDAGraphs

If you're using `torch.compile`, [CUDAGraphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) are likely being used under the hood - particularly in `reduce-overhead` or `max-autotune` compile modes. CUDAGraphs reduce the overheads from python interpreter and kernel launches by recording a sequence of GPU operations, and replaying them with same/different inputs later with a single CPU instruction, therefore removing any synchronizations or overheads.

In our benchmarks, we found that using CUDAGraphs manually (instead of relying on `torch.compile`) yields an increased speedup (`>2-5%`). This is unexpected and we are working with the Pytorch team to better understand the behaviour. Our implementation opts for manual graph capture to extract maximum performance!

## Benchmarks

For comparing our performance improvements, we will benchmark against [xDiT](https://github.com/xdit-project/xDiT) and [ParaAttention](https://github.com/chengzeyi/ParaAttention). They are optimized frameworks for diffusion models and provide strong competitive baselines to benchmark against, and are very resourceful for understanding the optimizations we will be implementing.

Note: We are using the reported performance results from the above projects. On my testing environment on H100, the reported performance numbers are worse in some cases but better in others. There are many factors why this could be the case, but we will not dive into them here. Another important point worth noting is that we are only reporting the transformer inference time here, and not the end-to-end generation time (with text encoder and VAE), but directly using the reported numbers from the above projects (which may be wall-clock times). In practice, the time taken for text encoders and VAE can be ignored. Our benchmark numbers are reported averaged over 5 runs.

<table>
<tr>
  <th> A100 </th>
  <th> H100 </th>
</tr>
<tr>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_post_1-a100.png" width="512px" /></td>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_post_1-h100.png" width="512px" /></td>
</tr>
</table>

### Cost Analysis

Assuming you run the optimized model (cudagraph + compile) with a fully set up environment on H100 SXM and 2.75 TiB SSD, the table below shows:
- time taken to generate 1000 images
- how much it would cost to generate 1000 images on different providers
- images per hour
- images per $1000

Note:
- The timing reports are for running the entire pipeline end-to-end and includes the time taken by the text encoders, denoiser and VAE decoder (i.e. not the same as benchmarks above which only report the transformer inference time).
- The cost analysis is based on the pricing of different cloud providers as of July 19th, 2025.
- The prices are for running the optimized inference on a single H100 GPU with 2.75 TiB SSD storage.
- The reported numbers for Runpod/Lambda/Modal is calculated as `100 * avg_of_5(time taken to generate 10 images)`, which is the average time taken to generate 10 images multiplied by 100 to get the total time for 1000 images.
- For Replicate and Fal, we compare the cost of running their inference service and calculate the time based on reported numbers at [Artificial Analysis](https://artificialanalysis.ai/text-to-image/model-family/flux).

<table>
<tr>
  <th> Provider </th>
  <th> Pricing per hour </th>
  <th> Time for 1000 images (hours) </th>
  <th> Cost for 1000 images ($) </th>
  <th> Images per hour </th>
  <th> Images per $1000 </th>
</tr>
<tr>
  <td> Runpod </td>
  <td> $2.69 (compute) + $0.19 (storage) </td>
  <td> 4.06 * 1000 / (60 * 60) = 1.127 </td>
  <td> $3.48 </td>
  <td> 885 </td>
  <td> 287356 </td>
</tr>
<tr>
  <td> Lambda </td>
  <td> $3.29 (compute + storage) </td>
  <td> 4.11 * 1000 / (60 * 60) = 1.141 </td>
  <td> $3.76 </td>
  <td> 875 </td>
  <td> 265957 </td>
</tr>
<tr>
  <td> Fal </td>
  <td> - </td>
  <td> 1.778 * 1000 / (60 * 60) = 0.494 </td>
  <td> $0.025 (per 1024px image) * 1000 = $25  </td>
  <td> 2024 </td>
  <td> 40000 </td>
</tr>
<tr>
  <td> Replicate </td>
  <td> - </td>
  <td> 2.934 * 1000 / (60 * 60) = 0.815 </td>
  <td> $0.025 (per 1024px image) * 1000 = $25 </td>
  <td> 1227 </td>
  <td> 40000 </td>
</tr>
<tr>
  <td> Modal </td>
  <td> $3.95 (compute) + $22 (storage) </td>
  <td> N/A </td>
  <td> N/A </td>
  <td> N/A </td>
  <td> N/A </td>
</tr>
</table>

It is evident that running your own optimized inference engine on a cloud provider is significantly cheaper than using a managed service. However, optimizing is not easy and requires a lot of expertise and time. As always, there are tradeoffs everywhere and it is important to evaluate your use-case and requirements before deciding on the best approach to run your inference workloads.

It is worth noting that we are only minimally utilizing the GPU because of `batch_size=1` in our benchmarks. If your use-case is to maximize throughput instead of latency, you should amp up the batch size to 8 or 16 or higher.

## Additional Reading

- [Annotated Diffusion (article)](https://huggingface.co/blog/annotated-diffusion)
- [What are Diffusion models? (article)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Diffusion meets Flow Matching: Two Sides of the Same Coin (article)](https://diffusionflow.github.io/)
- [How I understand Flow Matching? (youtube)](https://www.youtube.com/watch?v=DDq_pIfHqLs)
