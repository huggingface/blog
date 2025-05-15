---
title: "Exploring Quantization Backends in Diffusers"
thumbnail: /blog/assets/diffusers-quantization/thumbnail.png
authors:
- user: derekl35
- user: marcsun13
- user: sayakpaul
---

# Exploring Quantization Backends in Diffusers

Building on our previous post, "[Memory-efficient Diffusion Transformers with Quanto and Diffusers](https://huggingface.co/blog/quanto-diffusers)", this post explores the diverse quantization backends integrated directly into Hugging Face Diffusers. We'll examine how bitsandbytes, GGUF, torchao, and native FP8 support make large and powerful models more accessible, demonstrating their use with the Flux (a flow-based text-to-image generation model).

## Quantization Backends in Diffusers

```python
prompts = [
    "Baroque style, a lavish palace interior with ornate gilded ceilings, intricate tapestries, and dramatic lighting over a grand staircase.",
    "Futurist style, a dynamic spaceport with sleek silver starships docked at angular platforms, surrounded by distant planets and glowing energy lines.",
    "Noir style, a shadowy alleyway with flickering street lamps and a solitary trench-coated figure, framed by rain-soaked cobblestones and darkened storefronts.",
]
```

**BF16:**

![Baroque, Futurist, and Noir style images generated with BF16 precision](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_bf16_combined.png)

| Precision | Memory after loading | Peak memory | Inference time |
|-----------|----------------------|-------------|----------------|
| BF16      | ~31.447 GB           | 36.166 GB   | 12 seconds     |


### bitsandbytes (BnB)

[`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) is a popular and user-friendly library for 8-bit and 4-bit quantization, widely used for LLMs and QLoRA fine-tuning. We can use it for transformer-based diffusion and flow models, too.

**BnB 4-bit:**

![Baroque, Futurist, and Noir style images generated with BnB 4-bit](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_bnb_4bit_combined.png)

**BnB 8-bit:**
![Baroque, Futurist, and Noir style images generated with BnB 8-bit](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_bnb_8bit_combined.png)

| BnB Precision | Memory after loading | Peak memory | Inference time |
|---------------|----------------------|-------------|----------------|
| 4-bit         | 12.584 GB            | 17.281 GB   | 12 seconds     |
| 8-bit         | 19.273 GB            | 24.432 GB   | 27 seconds     |

**Example (Flux-dev with BnB 4-bit):**

```python
import torch
from diffusers import AutoModel, FluxPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import T5EncoderModel
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

For more information check out the [bitsandbytes docs](https://huggingface.co/docs/diffusers/quantization/bitsandbytes).

### `torchao`

[`torchao`](https://github.com/pytorch/ao) is a PyTorch-native library for architecture optimization, offering quantization, sparsity, and custom data types, designed for compatibility with `torch.compile` and FSDP.

`int4_weight_only`:
![torchao int4_weight_only Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_torchao_4bit_combined.png)
`int8_weight_only`:
![torchao int8_weight_only Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_torchao_8bit_combined.png)

`float8_weight_only`:
![torchao float8_weight_only Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_torchao_fp8_combined.png)

| torchao Precision             | Memory after loading | Peak memory | Inference time |
|-------------------------------|----------------------|-------------|----------------|
| int4_weight_only              | 10.635 GB            | 14.654 GB   | 109 seconds    |
| int8_weight_only              | 17.020 GB            | 21.482 GB   | 15 seconds     |
| float8_weight_only            | 17.016 GB            | 21.488 GB   | 15 seconds     |

**Example (Flux-dev with `torchao` INT8 weight-only):**

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

For more information check out the [torchao docs](https://huggingface.co/docs/diffusers/quantization/torchao).

### Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) is a quantization library integrated with the Hugging Face ecosystem via the [`optimum`](https://huggingface.co/docs/optimum/index) library.

`int4`:
![Quanto int4 Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_quanto_int4_combined.png)

`int8`:
![Quanto int8 Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_quanto_int8_combined.png)

| quanto Precision | Memory after loading | Peak memory | Inference time |
|------------------|----------------------|-------------|----------------|
| int4             | 12.254 GB            | 16.139 GB   | 109 seconds    |
| int8             | 17.330 GB            | 21.814 GB   | 15 seconds     |

**Example (Flux-dev with `quanto` INT8 weight-only):**

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


For more information check out the [Quanto docs](https://huggingface.co/docs/diffusers/quantization/quanto).

### GGUF

GGUF is a file format popular in the llama.cpp community for storing quantized models.

**GGUF Q2_k:**
![GGUF Q2_k Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_gguf_Q2_k_combined.png)
**GGUF Q4_1:**
![GGUF Q4_1 Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_gguf_Q4_1_combined.png)
**GGUF Q8_0:**
![GGUF Q8_0 Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_gguf_Q8_0_combined.png)



**GGUF w/ `.enable_model_cpu_offload()`**
| GGUF Precision | Memory after loading | Peak memory | Inference time |
|----------------|----------------------|-------------|----------------|
| Q4_1           | 0 GB                 | 9.164 GB    | 28 seconds     |
| Q8_0           | 0 GB                 | 12.965 GB   | 22 seconds     |
| Q2_k           | 0 GB                 | 9.164 GB    | 31 seconds     |

**GGUF w/ `.to("cuda")`**
| GGUF Precision | Memory after loading | Peak memory | Inference time |
|----------------|----------------------|-------------|----------------|
| Q4_1           | 16.838 GB            | 21.326 GB   | 23 seconds     |
| Q8_0           | 21.502 GB            | 25.973 GB   | 15 seconds     |
| Q2_k           | 13.264 GB            | 17.752 GB   | 26 seconds     |

**Example (Flux-dev with GGUF Q4_1)**

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

image.save("flux-dev_gguf_Q4_1.png")
```

For more information check out the [GGUF docs](https://huggingface.co/docs/diffusers/quantization/gguf).

### FP8 Layerwise Casting (`enable_layerwise_casting`)

FP8 Layerwise Casting is a memory optimization technique. It works by storing the model's weights in the compact FP8 (8-bit floating point) format, which uses roughly half the memory of standard FP16 or BF16 precision.
Just before a layer performs its calculations, its weights are dynamically cast up to a higher compute precision (like FP16/BF16). Immediately afterward, the weights are cast back down to FP8 for efficient storage. This approach works because the core computations retain high precision, and layers particularly sensitive to quantization (like normalization) are typically skipped.

![FP8 Layerwise Casting Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers/combined_flux-dev_fp8_layerwise_casting_combined.png)

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

image.save("flux-dev_fp8_layerwise_casting.png")
```

For more information check out the [Layerwise casting docs](https://huggingface.co/docs/diffusers/main/en/optimization/memory#layerwise-casting).

## Spot The Quantized Model

Quantization sounds great for saving memory, but how much does it *really* affect the final image? Can you even spot the difference? We invite you to test your perception!

We created a setup where you can provide a prompt, and we generate results using both the original, high-precision model (e.g., Flux-dev in BF16) and several quantized versions (BnB 4-bit, BnB 8-bit). The generated images are then presented to you and your challenge is to identify which ones came from the quantized models.

Try it out [here](https://huggingface.co/spaces/derekl35/flux-quant)!

Often, especially with 8-bit quantization, the differences are subtle and may not be noticeable without close inspection. More aggressive quantization like 4-bit or lower might be more noticeable, but the results can still be good, especially considering the massive memory savings.

## Conclusion

Here's a quick guide to choosing a quantization backend:

*   **Easiest Memory Savings (NVIDIA):** Start with `bitsandbytes` 4/8-bit.
*   **Prioritize Inference Speed:** `torchao` + `torch.compile` offers the best performance potential.
*   **For Hardware Flexibility (CPU/MPS), FP8 Precision:** `Quanto` can be a good option.
*   **Simplicity (Hopper/Ada):** Explore FP8 Layerwise Casting (`enable_layerwise_casting`).
*   **For Using Existing GGUF Models:** Use GGUF loading (`from_single_file`).

Quantization significantly lowers the barrier to entry for using large diffusion models. Experiment with these backends to find the best balance of memory, speed, and quality for your needs.