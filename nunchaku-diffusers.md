---
title: "Bringing Nunchaku 4-bit Diffusion Inference to Diffusers"
thumbnail: /blog/assets/nunchaku-diffusers/thumbnail.png
authors:
- user: rootonchair
  guest: true
- user: sayakpaul
---

# Bringing Nunchaku 4-bit Diffusion Inference to Diffusers

Large diffusion transformers can create stunning images (or even videos, audio snippets, and now text), but loading a modern text-to-image model in BF16 precision often requires 20-30 GB of VRAM, which puts these models out of reach of most consumer GPUs. Quantization is a powerful solution to this problem, and Diffusers already integrates several quantization backends such as bitsandbytes, GGUF, torchao, and Quanto, which we covered in [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization).

Most of these backends are _weight-only_. This means that they store the weights in low precision and dequantize them back to high precision at compute time. This reduces memory usage significantly, but it usually does not make inference faster, and can even add a small latency overhead.

[SVDQuant](https://arxiv.org/abs/2411.05007), the quantization method behind the popular [Nunchaku](https://github.com/nunchaku-tech/nunchaku) inference engine, takes a different approach. It runs the main transformer layers with 4-bit weights and activations (W4A4), reducing memory while also speeding up the denoising loop. The details are covered below, but until now, using these checkpoints required a separate inference library.

With current Diffusers, loading a Nunchaku checkpoint is as simple as calling `from_pretrained()`, with no local CUDA compilation required thanks to the [`kernels`](https://github.com/huggingface/kernels) package. In addition, the companion [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor) toolkit lets you quantize new architectures yourself and publish them as regular Diffusers repositories.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nunchaku-diffusers/contact_sheet_top3_metrics_bold.png" alt="Nunchaku Lite image quality and performance comparison">
</figure>

## Table of Contents

- [Getting started with Nunchaku Lite](#getting-started-with-nunchaku-lite)
- [Background: SVDQuant and Nunchaku](#background-svdquant-and-nunchaku)
- [Introducing Nunchaku Lite](#introducing-nunchaku-lite)
- [Native loading in Diffusers](#native-loading-in-diffusers)
- [Getting more speed and lower memory](#getting-more-speed-and-lower-memory)
- [Benchmarks](#benchmarks)
- [Quantizing your own model](#quantizing-your-own-model)
- [Ready-to-use checkpoints](#ready-to-use-checkpoints)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Getting started with Nunchaku Lite

First, install the requirements. You need a recent version of Diffusers and the Hugging Face `kernels` package:

```bash
pip install -U diffusers transformers accelerate kernels bitsandbytes
```

Then load a pre-quantized pipeline like any other Diffusers model:

```python
import torch
from diffusers import ErnieImagePipeline

pipe = ErnieImagePipeline.from_pretrained(
    "lite-infer/ERNIE-Image-Turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt="A cinematic portrait of a red fox in a misty forest at sunrise, "
           "detailed fur, volumetric light",
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=1.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("output.png")
```

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nunchaku-diffusers/fox_bf16_vs_nunchaku_no_metrics.png" alt="BF16 and Nunchaku Lite outputs for a red fox prompt">
</figure>

No custom pipeline class or separate inference engine is needed, and there is nothing to compile locally. The NVFP4 kernels are downloaded from the Hub the first time they are used. This checkpoint pairs a Nunchaku NVFP4 transformer with a bitsandbytes NF4 text encoder, and generates a 1024x1024 image in about 1.7 seconds on an RTX 5090 with a peak memory usage of about 12 GB, compared with about 24 GB for the BF16 pipeline. You can find more details about the Nunchaku Lite checkpoint format in the [official Diffusers documentation](https://huggingface.co/docs/diffusers/main/en/quantization/nunchaku).

> [!NOTE]
> NVFP4 checkpoints require an NVIDIA Blackwell GPU (RTX 50 series, RTX PRO 6000, B200). For earlier generations, use the INT4 variants. See the [hardware support](#hardware-support) table below for details.

## Background: SVDQuant and Nunchaku

**SVDQuant** is the quantization method behind **Nunchaku**, its reference CUDA inference engine. Standard 4-bit quantization is difficult for diffusion transformers because both weights and activations contain large outliers. SVDQuant handles this by moving activation outliers into the weights, representing the hardest part of each weight matrix with a small 16-bit low-rank branch, and quantizing the remaining residual to 4 bits. Nunchaku makes this fast with fused kernels for the 4-bit path and the low-rank branch.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nunchaku-diffusers/svdquant_kernel_fusion.png" alt="Nunchaku kernel fusion: the low-rank down projection is fused with input quantization, and the low-rank up projection is fused with the 4-bit matmul">
  <figcaption>Nunchaku fuses the low-rank down projection with the quantization kernel and the low-rank up projection with the 4-bit compute kernel, eliminating the memory access overhead of the 16-bit branch. Figure from the <a href="https://arxiv.org/abs/2411.05007">SVDQuant paper</a>.</figcaption>
</figure>

## Introducing Nunchaku Lite

The original Nunchaku engine gets much of its speed from model-specific graph fusions, such as fused QKV projections, fused adaLN modulation, and fused GELU/MLP kernels. Those fusions are tied to each architecture's module layout and checkpoint format, so supporting a new model family usually requires model-specific integration work.

**Nunchaku Lite** is the new integration path in Diffusers. With it, Diffusers can load Nunchaku-style checkpoints without a custom pipeline or a separate inference engine. Under the hood, Nunchaku Lite patches the relevant `nn.Linear` modules of a stock Diffusers model with runtime SVDQ/AWQ linear layers before the checkpoint is loaded. The CUDA kernels come from the Hub through the `kernels` package. Two kernel families are used:

*   **`svdq_w4a4`**: 4-bit weights and activations with the SVDQuant low-rank correction. This layer is used for the transformer's attention and MLP projections, where nearly all of the compute is spent, and is available in INT4 and NVFP4 variants.
*   **`awq_w4a16`**: 4-bit weights with 16-bit activations, used for adaptive normalization and modulation projections such as FLUX `adanorm_single` / `adanorm_zero` or Qwen-Image modulation layers. These layers are memory-bound and precision-sensitive, making AWQ a good fit to preserve precision while still saving memory and space.

The trade-off is that, without architecture-specific fused kernels and modules, Nunchaku Lite cannot match the speedup of the original Nunchaku engine. However, the bare-bones implementation still delivers around **30% speedup** while retaining the same level of **VRAM reduction**.

## Native loading in Diffusers

If you have used bitsandbytes or torchao in Diffusers, the mechanics will feel familiar. A Nunchaku Lite model repository is an ordinary Diffusers repository. The only special part is a `quantization_config` block inside the transformer's `config.json`:

```json
"quantization_config": {
    "quant_method": "nunchaku_lite",
    "compute_dtype": "bfloat16",
    "svdq_w4a4": {
        "precision": "nvfp4",
        "group_size": 16,
        "rank": 32,
        "targets": [
            "layers.0.self_attention.to_q",
            "layers.0.self_attention.to_k",
            "..."
        ]
    },
    "awq_w4a16": {
        "precision": "int4",
        "group_size": 64,
        "targets": [
            "adaLN_modulation.1",
            "..."
        ]
    }
}
```

This config tells Diffusers which modules were quantized, which scheme they use, and which Nunchaku Lite runtime layer to instantiate (`SVDQW4A4Linear` or `AWQW4A16Linear`).

Because the quantized model keeps the exact module structure of the dense one, everything downstream (schedulers, LoRA loading hooks, offloading, `torch.compile`) sees a normal Diffusers model.

### Hardware support

Nunchaku Lite uses different kernel variants depending on the GPU generation and checkpoint precision:

| Scheme | Precision | Supported GPUs |
|---|---|---|
| `svdq_w4a4` | `nvfp4` | Blackwell (RTX 50 series, RTX PRO 6000, B200) |
| `svdq_w4a4` | `int4` | Turing / Ampere / Ada (RTX 30 & 40 series, A100, L40S) |
| `awq_w4a16` | `int4` | Turing / Ampere / Ada (RTX 30 & 40 series, A100, L40S) |

> [!WARNING]
> Volta and Hopper GPUs are currently not supported by the 4-bit kernels. The quantizer validates the GPU's CUDA capability at load time and raises a clear error instead of producing incorrect outputs.

## Getting more speed and lower memory

Nunchaku Lite can be combined with other Diffusers memory and speed optimizations.

**`torch.compile`.** Compiling the transformer improves the end-to-end speedup from 1.35x to 1.8x:

```python
pipe.transformer.compile(fullgraph=True)

# or compile_repeated_blocks() for faster compilation

pipe.transformer.compile_repeated_blocks(fullgraph=True)
```

**Quantized text encoders.** The transformer is not the only component with a large memory footprint. Text encoders such as T5 or Qwen3 can occupy several gigabytes on their own. Further quantizing the text encoder with bitsandbytes NF4 reduces peak VRAM by about 22% in our benchmark.

**Offloading.** Diffusers offloading helpers such as `enable_model_cpu_offload()` and `enable_sequential_cpu_offload()` work as usual if you need to fit the pipeline onto a smaller GPU.

## Benchmarks

All numbers below were measured on an NVIDIA RTX PRO 6000 (Blackwell) at 1024x1024 using [rootonchair/ERNIE-Image-Turbo-nunchaku-lite-int4-bnb4-text-encoder](https://huggingface.co/rootonchair/ERNIE-Image-Turbo-nunchaku-lite-int4-bnb4-text-encoder).

### End-to-end latency and memory

| Configuration | Full pipeline | Denoise loop | Peak VRAM | Speedup |
|---|---|---|---|---|
| BF16 baseline | 3.00 s | 2.86 s | 31.1 GB | 1.0x |
| Nunchaku Lite NVFP4 | 2.27 s | 2.13 s | 20.6 GB | 1.35x |
| Nunchaku Lite NVFP4 + `torch.compile` | 1.68 s | 1.53 s | 20.6 GB | 1.8x |
| Nunchaku Lite NVFP4 + NF4 text encoder | 2.29 s | 2.13 s | 16.0 GB | 1.35x |

As shown above, Nunchaku reduces peak VRAM by up to 50% while still improving latency by roughly 30%. The remaining overhead comes largely from extra kernel launches, which `torch.compile` can mitigate, bringing the full pipeline down to 1.68 s, or 1.8x faster than the BF16 baseline.

### Image quality

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nunchaku-diffusers/quality_grid.png" alt="Quality comparison grid">
  <figcaption>BF16 vs 4-bit outputs with identical seeds and settings.</figcaption>
</figure>

## Quantizing your own model

Nunchaku Lite support in Diffusers is architecture-agnostic, and the [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor) toolkit provides an end-to-end SVDQuant workflow for Diffusers models: calibrate, quantize, package, and publish.

Below, we walk through quantizing FLUX.2 Klein 4B as an example. It covers the main steps: inspect the model, calibrate and quantize the transformer, package the result as a Diffusers pipeline, then verify and push it to the Hub. The [full tutorial](https://github.com/rootonchair/diffuse-compressor/blob/main/docs/quantize_new_hf_model.md) covers every flag in detail.

### 1. Inspect what will be quantized

The generic scanner walks the model and decides what to target: compatible linears inside the repeated transformer-block stack become SVDQ W4A4 targets, recognized modulation linears become AWQ W4A16 targets, and everything else stays dense.

```bash
python examples/text_to_image/quantize_hf.py black-forest-labs/FLUX.2-klein-4B \
  --precision int4 --rank 32 --inspect-config
```

Always read this report before quantizing. For FLUX.2 Klein 4B, the expected result is 100 SVDQ targets, 3 AWQ targets, and 6 dense outer linears, with no missing patterns or duplicate names.

### 2. Run quantization

The following command runs SVDQuant on the transformer and writes the quantized checkpoint:

```bash
python examples/text_to_image/quantize_hf.py black-forest-labs/FLUX.2-klein-4B \
  --precision int4 --rank 32 \
  --num-samples 128 --batch-size 1 --sample-batch-size 32 \
  --compute-device cuda --pipeline-offload model --svd-backend svd_lowrank
```

Replace `--precision int4` with `nvfp4` to build Blackwell-native weights.

### 3. Package a Diffusers pipeline

The converter combines the quantized transformer with the base pipeline's other components, writes the compact `nunchaku_lite` configuration into `transformer/config.json`, and can optionally convert text encoders to NF4:

```bash
python examples/convert_nunchaku_lite_diffusers.py \
  --checkpoint outputs/checkpoints/svdq-int4_r32-flux-2-klein-4b.safetensors \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --bnb4-text-encoder text_encoder \
  --compute-dtype bfloat16
```

### 4. Load, verify, and push to the Hub

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "outputs/diffusers/FLUX.2-klein-4B-nunchaku-lite-int4-bnb4-text-encoder",
    device_map="cuda",
)
image = pipe(
    "A glass robot in a greenhouse, cinematic lighting",
    num_inference_steps=4, guidance_scale=1.0,
    generator=torch.Generator("cuda").manual_seed(12345),
).images[0]
```

Once the outputs look good, run `pipe.push_to_hub("your-name/your-model-nunchaku-lite-int4")`. Other users can then load it with the same `from_pretrained()` pattern shown above.

Note that the generic path assumes the architecture can be quantized without structural rewrites. Models whose runtime requires grouped QKV tensors, split fused projections, or other structural changes need a small model-specific target config in diffuse-compressor. See the [FLUX.2 Klein 4B quantization script](https://github.com/rootonchair/diffuse-compressor/blob/main/examples/text_to_image/quantize_flux2_klein_4b.py) as a concrete example. For checkpoints that use grouped QKV tensors or additional fused kernels, [`rootonchair/nunchaku-lite`](https://github.com/rootonchair/nunchaku-lite) provides a lean runtime package for loading Nunchaku-quantized Diffusers pipelines and isolating those model-specific operations in small adapters. The [Adding A New Model](https://github.com/rootonchair/diffuse-compressor/blob/main/docs/adding_new_model.md) guide covers that path.

## Ready-to-use checkpoints

To get started right away, check out the following repositories:

- [rootonchair/ERNIE-Image-Turbo-nunchaku-lite-int4-bnb4-text-encoder](https://huggingface.co/rootonchair/ERNIE-Image-Turbo-nunchaku-lite-int4-bnb4-text-encoder): INT4 ERNIE-Image-Turbo with a bitsandbytes NF4 text encoder
- [rootonchair/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder](https://huggingface.co/rootonchair/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder): NVFP4 ERNIE-Image-Turbo with a bitsandbytes NF4 text encoder
- [OzzyGT/Krea_2_Turbo_nunchaku_lite_nvfp4](https://huggingface.co/OzzyGT/Krea_2_Turbo_nunchaku_lite_nvfp4): NVFP4 Krea 2 Turbo checkpoint
- [lite-infer](https://huggingface.co/lite-infer): more Nunchaku Lite checkpoints and collections

## Conclusion

Nunchaku's SVDQuant kernels are one of the most effective ways to run diffusion transformers efficiently on consumer hardware, and they are now natively supported in Diffusers. Pre-quantized checkpoints load with `from_pretrained()`, and the diffuse-compressor toolkit makes it possible to quantize new architectures without waiting for engine support. By quantizing both weights and activations, the W4A4 path lowers memory use while improving denoising latency, keeping image quality close to the BF16 original.

If you quantize and publish a new model, we would love to hear about it. Share it on the Hub and let us know! If you have any questions about this feature, feel free to join our [Discord](https://discord.gg/G7tWnz98XR).

To learn more, check out the following resources:

- [Diffusers Nunchaku documentation](https://huggingface.co/docs/diffusers/quantization/nunchaku)
- [The integration PR (huggingface/diffusers#14100)](https://github.com/huggingface/diffusers/pull/14100)
- [SVDQuant paper](https://arxiv.org/abs/2411.05007) and the [Nunchaku engine](https://github.com/nunchaku-tech/nunchaku)
- [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor)
- Previous posts: [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization) and [Memory-efficient Diffusion Transformers with Quanto and Diffusers](https://huggingface.co/blog/quanto-diffusers)

## Acknowledgements

Thanks to the Diffusers maintainers for reviews and guidance throughout the integration, and to the MIT HAN Lab / Nunchaku team for the original SVDQuant work.

`rootonchair` is also grateful to SilverAI for supporting this work and providing the environment in which much of this development took place.
