---
title: "Exploring Quantization Backends in Diffusers"
thumbnail: /blog/assets/diffusers-quantization/thumbnail.png
authors:
- user: derekl35
- user: marcsun13
- user: sayakpaul
---

# Exploring Quantization Backends in Diffusers

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/5.27.1/gradio.js"></script>

Large diffusion models like Flux (a flow-based text-to-image generation model) can create stunning images, but their size can be a hurdle, demanding significant memory and compute resources. Quantization offers a powerful solution, shrinking these models to make them more accessible without drastically compromising performance. But the big question always is: can you actually tell the difference in the final image?

Before we dive into the technical details of how various quantization backends in Hugging Face Diffusers work, why not test your own perception?

## Spot The Quantized Model

We created a setup where you can provide a prompt, and we generate results using both the original, high-precision model (e.g., Flux-dev in BF16) and several quantized versions (BnB 4-bit, BnB 8-bit). The generated images are then presented to you and your challenge is to identify which ones came from the quantized models.

Try it out [here](https://huggingface.co/spaces/diffusers/flux-quant) or below!
<gradio-app src="https://diffusers-flux-quant.hf.space"></gradio-app>

Often, especially with 8-bit quantization, the differences are subtle and may not be noticeable without close inspection. More aggressive quantization like 4-bit or lower might be more noticeable, but the results can still be good, especially considering the massive memory savings. NF4 often gives the best trade-off though.

Now, let's dive deeper.

## Quantization Backends in Diffusers

Building on our previous post, "[Memory-efficient Diffusion Transformers with Quanto and Diffusers](https://huggingface.co/blog/quanto-diffusers)", this post explores the diverse quantization backends integrated directly into Hugging Face Diffusers. We'll examine how bitsandbytes, GGUF, torchao, Quanto and native FP8 support make large and powerful models more accessible, demonstrating their use with Flux.

Before diving into the quantization backends, let's introduce the FluxPipeline (using the [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) checkpoint) and its components, which we'll be quantizing. Loading the full `FLUX.1-dev` model in BF16 precision requires approximately 31.447 GB of memory. The main components are:

*   **Text Encoders (CLIP and T5):**
    *   **Function:** Process input text prompts. FLUX-dev uses CLIP for initial understanding and a larger T5 for nuanced comprehension and better text rendering.
    *   **Memory:** T5 - 9.52 GB; CLIP - 246 MB (in BF16)
*   **Transformer (Main Model - MMDiT):**
    *   **Function:** Core generative part (Multimodal Diffusion Transformer). Generates images in latent space from text embeddings. 
    *   **Memory:** 23.8 GB (in BF16)
*   **Variational Auto-Encoder (VAE):**
    *   **Function:** Translates images between pixel and latent space. Decodes generated latent representation to a pixel-based image.
    *   **Memory:** 168 MB (in BF16)
*   **Focus of Quantization:** Examples will primarily focus on the `transformer` and `text_encoder_2` (T5) for the most substantial memory savings.

```python
prompts = [
    "Baroque style, a lavish palace interior with ornate gilded ceilings, intricate tapestries, and dramatic lighting over a grand staircase.",
    "Futurist style, a dynamic spaceport with sleek silver starships docked at angular platforms, surrounded by distant planets and glowing energy lines.",
    "Noir style, a shadowy alleyway with flickering street lamps and a solitary trench-coated figure, framed by rain-soaked cobblestones and darkened storefronts.",
]
```

### bitsandbytes (BnB)

[`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) is a popular and user-friendly library for 8-bit and 4-bit quantization, widely used for LLMs and QLoRA fine-tuning. We can use it for transformer-based diffusion and flow models, too.

<table>
  <tr>
    <td style="text-align: center;">
      BF16<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_bf16_combined.png" alt="Flux-dev output with BF16: Baroque, Futurist, Noir styles"></medium-zoom>
      </figure>
    </td>
    <td style="text-align: center;">
      BnB 4-bit<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_bnb_4bit_combined.png" alt="Flux-dev output with BnB 4-bit: Baroque, Futurist, Noir styles"></medium-zoom>
      </figure>
    </td>
    <td style="text-align: center;">
      BnB 8-bit<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_bnb_8bit_combined.png" alt="Flux-dev output with BnB 8-bit: Baroque, Futurist, Noir styles"></medium-zoom>
      </figure>
    </td>
  </tr>
  <tr>
    <td colspan="3" style="text-align: center;"><em>Visual comparison of Flux-dev model outputs using BF16 (left), BnB 4-bit (center), and BnB 8-bit (right) quantization. (Click on an image to zoom) </em></td>
  </tr>
</table>

| Precision     | Memory after loading | Peak memory | Inference time |
|---------------|----------------------|-------------|----------------|
| BF16          | ~31.447 GB           | 36.166 GB   | 12 seconds     |
| 4-bit         | 12.584 GB            | 17.281 GB   | 12 seconds     |
| 8-bit         | 19.273 GB            | 24.432 GB   | 27 seconds     |

<sub>All benchmarks performed on 1x NVIDIA H100 80GB GPU</sub>


<details>
<summary>Example (Flux-dev with BnB 4-bit):</summary>


```python
import torch
from diffusers import FluxPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

model_id = "black-forest-labs/FLUX.1-dev"

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

prompt = "Baroque style, a lavish palace interior with ornate gilded ceilings, intricate tapestries, and dramatic lighting over a grand staircase."
pipe_kwargs = {
    "prompt": prompt,
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
    "max_sequence_length": 512,
}


print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

image = pipe(
    **pipe_kwargs, generator=torch.manual_seed(0),
).images[0]

print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

image.save("flux-dev_bnb_4bit.png")
```

> **Note:** When using `PipelineQuantizationConfig` with `bitsandbytes`, you need to import `DiffusersBitsAndBytesConfig` from `diffusers` and `TransformersBitsAndBytesConfig` from `transformers` separately. This is because these components originate from different libraries. If you prefer a simpler setup without managing these distinct imports, you can use an alternative approach for pipeline-level quantization, an example of this method is in the [Diffusers documentation on Pipeline-level quantization](https://huggingface.co/docs/diffusers/main/en/quantization/overview#pipeline-level-quantization:~:text=The%20config%20below%20will%20work%20for%20most%20diffusion%20pipelines%20that%20have%20a%20transformer%20component%20present.%20In%20most%20case%2C%20you%20will%20want%20to%20quantize%20the%20transformer%20component%20as%20that%20is%20often%20the%20most%20compute%2D%20intensive%20part%20of%20a%20diffusion%20pipeline).

</details>

For more information check out the [bitsandbytes docs](https://huggingface.co/docs/diffusers/quantization/bitsandbytes).

### torchao

[`torchao`](https://github.com/pytorch/ao) is a PyTorch-native library for architecture optimization, offering quantization, sparsity, and custom data types, designed for compatibility with `torch.compile` and FSDP. Diffusers supports a wide range of `torchao`'s exotic data types, enabling fine-grained control over model optimization.

<table>
  <tr>
    <td style="text-align: center;">
      int4_weight_only<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_torchao_4bit_combined.png" alt="torchao int4_weight_only Output"></medium-zoom>
      </figure>
    </td>
    <td style="text-align: center;">
      int8_weight_only<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_torchao_8bit_combined.png" alt="torchao int8_weight_only Output"></medium-zoom>
      </figure>
    </td>
    <td style="text-align: center;">
      float8_weight_only<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_torchao_fp8_combined.png" alt="torchao float8_weight_only Output"></medium-zoom>
      </figure>
    </td>
  </tr>
  <tr>
    <td colspan="3" style="text-align: center;"><em>Visual comparison of Flux-dev model outputs using torchao int4_weight_only (left), int8_weight_only (center), and float8_weight_only (right) quantization. (Click on an image to zoom)</em></td>
  </tr>
</table>

| torchao Precision             | Memory after loading | Peak memory | Inference time |
|-------------------------------|----------------------|-------------|----------------|
| int4_weight_only              | 10.635 GB            | 14.654 GB   | 109 seconds    |
| int8_weight_only              | 17.020 GB            | 21.482 GB   | 15 seconds     |
| float8_weight_only            | 17.016 GB            | 21.488 GB   | 15 seconds     |

<details>
<summary>Example (Flux-dev with torchao INT8 weight-only):</summary>

```diff
@@
- from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
+ from diffusers import TorchAoConfig as DiffusersTorchAoConfig

- from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
+ from transformers import TorchAoConfig as TransformersTorchAoConfig
@@
pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
-         "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
-         "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
+         "transformer": DiffusersTorchAoConfig("int8_weight_only"),
+         "text_encoder_2": TransformersTorchAoConfig("int8_weight_only"),
    }
)
```
</details>

<details>
<summary>Example (Flux-dev with torchao INT4 weight-only):</summary>

```diff
@@
- from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
+ from diffusers import TorchAoConfig as DiffusersTorchAoConfig

- from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
+ from transformers import TorchAoConfig as TransformersTorchAoConfig
@@
pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
-         "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
-         "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
+         "transformer": DiffusersTorchAoConfig("int4_weight_only"),
+         "text_encoder_2": TransformersTorchAoConfig("int4_weight_only"),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
+    device_map="balanced"
)
- pipe.to("cuda")
```
</details>

For more information check out the [torchao docs](https://huggingface.co/docs/diffusers/quantization/torchao).

### Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) is a quantization library integrated with the Hugging Face ecosystem via the [`optimum`](https://huggingface.co/docs/optimum/index) library.

<table>
  <tr>
    <td style="text-align: center;">
      INT4<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_quanto_int4_combined.png" alt="Quanto INT4 Output"></medium-zoom>
      </figure>
    </td>
    <td style="text-align: center;">
      INT8<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_quanto_int8_combined.png" alt="Quanto INT8 Output"></medium-zoom>
      </figure>
    </td>
    <td style="text-align: center;">
      FP8<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_quanto_fp8_combined.png" alt="Quanto fp8 Output"></medium-zoom>
      </figure>
    </td>
  </tr>
  <tr>
    <td colspan="3" style="text-align: center;"><em>Visual comparison of Flux-dev model outputs using Quanto INT4 (left), INT8 (center), and FP8 (right) quantization. (Click on an image to zoom)</em></td>
  </tr>
</table>

| quanto Precision | Memory after loading | Peak memory | Inference time |
|------------------|----------------------|-------------|----------------|
| INT4             | 12.254 GB            | 16.139 GB   | 109 seconds    |
| INT8             | 17.330 GB            | 21.814 GB   | 15 seconds     |
| FP8           | 16.395 GB            | 20.898 GB   | 16 seconds     |

<details>
<summary>Example (Flux-dev with quanto INT8 weight-only):</summary>

```diff
@@
- from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
+ from diffusers import QuantoConfig as DiffusersQuantoConfig

- from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
+ from transformers import QuantoConfig as TransformersQuantoConfig
@@
pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
-         "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
-         "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
+         "transformer": DiffusersQuantoConfig(weights_dtype="int8"),
+         "text_encoder_2": TransformersQuantoConfig(weights_dtype="int8"),
    }
)
```
</details>

> **Note:** At the time of writing, for float8 support with Quanto, you'll need `optimum-quanto<0.2.5` and use quanto directly. We will be working on fixing this.
<details>
<summary>Example (Flux-dev with quanto FP8 weight-only)</summary>

```python
import torch
from diffusers import AutoModel, FluxPipeline
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize

model_id = "black-forest-labs/FLUX.1-dev"

text_encoder_2 = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_2",
    torch_dtype=torch.bfloat16,
)

quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

transformer = AutoModel.from_pretrained(
      model_id,
      subfolder="transformer",
      torch_dtype=torch.bfloat16,
)

quantize(transformer, weights=qfloat8)
freeze(transformer)

pipe = FluxPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    text_encoder_2=text_encoder_2,
    torch_dtype=torch.bfloat16
).to("cuda")
```

</details>

For more information check out the [Quanto docs](https://huggingface.co/docs/diffusers/quantization/quanto).

### GGUF

GGUF is a file format popular in the llama.cpp community for storing quantized models.

<table>
  <tr>
    <td style="text-align: center;">
      Q2_k<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_gguf_Q2_k_combined.png" alt="GGUF Q2_k Output"></medium-zoom>
      </figure>
    </td>
    <td style="text-align: center;">
      Q4_1<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_gguf_Q4_1_combined.png" alt="GGUF Q4_1 Output"></medium-zoom>
      </figure>
    </td>
    <td style="text-align: center;">
      Q8_0<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_gguf_Q8_0_combined.png" alt="GGUF Q8_0 Output"></medium-zoom>
      </figure>
    </td>
  </tr>
  <tr>
    <td colspan="3" style="text-align: center;"><em>Visual comparison of Flux-dev model outputs using GGUF Q2_k (left), Q4_1 (center), and Q8_0 (right) quantization. (Click on an image to zoom)</em></td>
  </tr>
</table>

| GGUF Precision | Memory after loading | Peak memory | Inference time |
|----------------|----------------------|-------------|----------------|
| Q2_k           | 13.264 GB            | 17.752 GB   | 26 seconds     |
| Q4_1           | 16.838 GB            | 21.326 GB   | 23 seconds     |
| Q8_0           | 21.502 GB            | 25.973 GB   | 15 seconds     |

<details>
<summary>Example (Flux-dev with GGUF Q4_1)</summary>

```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

model_id = "black-forest-labs/FLUX.1-dev"

# Path to a pre-quantized GGUF file
ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_1.gguf"

transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
```

</details>

For more information check out the [GGUF docs](https://huggingface.co/docs/diffusers/quantization/gguf).

### FP8 Layerwise Casting (`enable_layerwise_casting`)

FP8 Layerwise Casting is a memory optimization technique. It works by storing the model's weights in the compact FP8 (8-bit floating point) format, which uses roughly half the memory of standard FP16 or BF16 precision.
Just before a layer performs its calculations, its weights are dynamically cast up to a higher compute precision (like FP16/BF16). Immediately afterward, the weights are cast back down to FP8 for efficient storage. This approach works because the core computations retain high precision, and layers particularly sensitive to quantization (like normalization) are typically skipped. This technique can also be combined with [group offloading](https://huggingface.co/docs/diffusers/en/optimization/memory#group-offloading) for further memory savings.

<table>
  <tr>
    <td style="text-align: center;">
      FP8 (e4m3)<br>
      <figure class="image table text-center m-0 w-full">
        <medium-zoom background="rgba(0,0,0,.7)" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_fp8_layerwise_casting_combined.png" alt="FP8 Layerwise Casting Output"></medium-zoom>
      </figure>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><em>Visual output of Flux-dev model using FP8 Layerwise Casting (e4m3) quantization.</em></td>
  </tr>
</table>

| precision | Memory after loading | Peak memory | Inference time |
|-----------|----------------------|-------------|----------------|
| FP8 (e4m3)| 23.682 GB            | 28.451 GB   | 13 seconds     |

```python
import torch
from diffusers import AutoModel, FluxPipeline

model_id = "black-forest-labs/FLUX.1-dev"

transformer = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

pipe = FluxPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")
```


For more information check out the [Layerwise casting docs](https://huggingface.co/docs/diffusers/main/en/optimization/memory#layerwise-casting).

## Combining with More Memory Optimizations and torch.compile

Most of these quantization backends can be combined with the memory optimization techniques offered in Diffusers. Let's explore CPU offloading, group offloading, and `torch.compile`. You can learn more about these techniques in the [Diffusers documentation](https://huggingface.co/docs/diffusers/main/en/optimization/memory).

> **Note:** At the time of writing, bnb + `torch.compile` also works if bnb is installed from source and using pytorch nightly or with fullgraph=False.

<details>
<summary>Example (Flux-dev with BnB 4-bit + enable_model_cpu_offload):</summary>

```diff
import torch
from diffusers import FluxPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

model_id = "black-forest-labs/FLUX.1-dev"

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16
)
- pipe.to("cuda")
+ pipe.enable_model_cpu_offload()
```

</details>

**Model CPU Offloading (`enable_model_cpu_offload`)**: This method moves entire model components (like the UNet, text encoders, or VAE) between the CPU and GPU during the inference pipeline. It offers substantial VRAM savings and is generally faster than more granular offloading because it involves fewer, larger data transfers.

**bnb + `enable_model_cpu_offload`**:
| Precision     | Memory after loading | Peak memory | Inference time |
|---------------|----------------------|-------------|----------------|
| 4-bit         | 12.383 GB            | 12.383 GB   | 17 seconds     |
| 8-bit         | 19.182 GB            | 23.428 GB   | 27 seconds     |

<details>
<summary>Example (Flux-dev with fp8 layerwise casting + group offloading):</summary>

```diff
import torch
from diffusers import FluxPipeline, AutoModel

model_id = "black-forest-labs/FLUX.1-dev"

transformer = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    # device_map="cuda"
)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
+ transformer.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), offload_type="leaf_level", use_stream=True)

pipe = FluxPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)
- pipe.to("cuda")
```
</details>

**Group offloading (`enable_group_offload` for `diffusers` components or `apply_group_offloading` for generic `torch.nn.Module`s)**: It moves groups of internal model layers (like `torch.nn.ModuleList` or `torch.nn.Sequential` instances) to the CPU. This approach is typically more memory-efficient than full model offloading and faster than sequential offloading.

**FP8 layerwise casting + group offloading**:

| precision | Memory after loading | Peak memory | Inference time |
|-----------|----------------------|-------------|----------------|
| FP8 (e4m3)| 9.264 GB             | 14.232 GB   | 58 seconds     |

<details>
<summary>Example (Flux-dev with torchao 4-bit + torch.compile):</summary>

```diff
import torch
from diffusers import FluxPipeline
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig

from torchao.quantization import Float8WeightOnlyConfig

model_id = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer":DiffusersTorchAoConfig("int4_weight_only"),
        "text_encoder_2": TransformersTorchAoConfig("int4_weight_only"),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)

+ pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
```

> **Note:** `torch.compile` can introduce subtle numerical differences, leading to changes in image output
</details>

**torch.compile**: Another complementary approach is to accelerate the execution of your model with PyTorch 2.x’s torch.compile() feature. Compiling the model doesn’t directly lower memory, but it can significantly speed up inference. PyTorch 2.0’s compile (Torch Dynamo) works by tracing and optimizing the model graph ahead-of-time.

**torchao + `torch.compile`**: 
| torchao Precision             | Memory after loading | Peak memory | Inference time | Compile Time |
|-------------------------------|----------------------|-------------|----------------|--------------|
| int4_weight_only              | 10.635 GB            | 15.238 GB   | 6 seconds    | ~285 seconds          |
| int8_weight_only              | 17.020 GB            | 22.473 GB   | 8 seconds     | ~851 seconds          |
| float8_weight_only            | 17.016 GB            | 22.115 GB   | 8 seconds     | ~545 seconds          |

Explore some benchmarking results here:

<iframe
  src="https://huggingface.co/datasets/derekl35/diffusers-quantization-benchmarks/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
  title="diffusers benchmarking results dataset"
></iframe>

## Ready to use quantized checkpoints

You can find `bitsandbytes` and `torchao` quantized models from this blog post in our Hugging Face collection: [link to collection](https://huggingface.co/collections/diffusers/flux-quantized-checkpoints-682c951aebd378a2462984a0).

## Conclusion

Here's a quick guide to choosing a quantization backend:

*   **Easiest Memory Savings (NVIDIA):** Start with `bitsandbytes` 4/8-bit. This can also be combined with `torch.compile()` for faster inference.
*   **Prioritize Inference Speed:** `torchao`, `GGUF`, and `bitsandbytes` can all be used with `torch.compile()` to potentially boost inference speed.
*   **For Hardware Flexibility (CPU/MPS), FP8 Precision:** `Quanto` can be a good option.
*   **Simplicity (Hopper/Ada):** Explore FP8 Layerwise Casting (`enable_layerwise_casting`).
*   **For Using Existing GGUF Models:** Use GGUF loading (`from_single_file`).
*   **Curious about training with quantization?** Stay tuned for a follow-up blog post on that topic!

Quantization significantly lowers the barrier to entry for using large diffusion models. Experiment with these backends to find the best balance of memory, speed, and quality for your needs.

*Acknowledgements: Thanks to [Chunte](https://huggingface.co/Chunte) for providing the thumbnail for this post.*
