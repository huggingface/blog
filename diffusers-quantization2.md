# Fine-Tuning FLUX.1-dev on consumer hardware with QLoRA

In our previous post, [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization), we dived into how various quantization techniques can shrink diffusion models like FLUX.1-dev, making them significantly more accessible for *inference* without drastically compromising performance. We saw how `bitsandbytes`, `torchao`, and others reduce memory footprints for generating images.

Performing inference is cool but to make these models truly our own, we also need to be able to fine-tune them. Therefore, in this post, we tackle **efficient** *fine-tuning* of these models with peak memory use under ~10 GB of VRAM on a single GPU. This post will guide you through fine-tuning FLUX.1-dev using QLoRA with the Hugging Face `diffusers` library. We'll showcase results from an NVIDIA RTX 4090.

## Why Not Just Full Fine-Tuning?

[`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev/), for instance, requires over 31GB in BF16 for inference alone.

**Full Fine-Tuning:** This traditional method updates all model params and offers the potential for the highest task-specific quality. However, for FLUX.1-dev, this approach would demand immense VRAM (multiple high-end GPUs), putting it out of reach for most individual users.

**LoRA (Low-Rank Adaptation):** [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) freezes the pre-trained weights and injects small, trainable "adapter" layers. This massively reduces trainable parameters, saving VRAM during training and resulting in small adapter checkpoints. The challenge is that the full-precision base model still needs to be loaded, which, for FLUX.1-dev, remains a hefty VRAM requirement even if fewer parameters are being updated.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png"
       alt="Illustration of LoRA injecting two low-rank matrices around a frozen weight matrix"
       width="600"/>
</p>

**QLoRA: The Efficiency Powerhouse:** [QLoRA](https://huggingface.co/docs/peft/main/en/developer_guides/quantization) enhances LoRA by first loading the pre-trained base model in a quantized format (typically 4-bit via `bitsandbytes`), drastically cutting the base model's memory footprint. It then trains LoRA adapters (usually in FP16/BF16) on top of this quantized base.

This allows fine-tuning of very large models on consumer-grade hardware or more accessible cloud GPUs.

## Dataset

 We aim to fine-tune `black-forest-labs/FLUX.1-dev` to adopt the artistic style of Alphonse Mucha, using a small [dataset](https://huggingface.co/datasets/derekl35/alphonse-mucha-style). 
<!-- (maybe use different dataset) -->

## FLUX Architecture

The model consists of three main components:

*   **Text Encoders (CLIP and T5):**
    *   **Function:** Process input text prompts. FLUX-dev uses CLIP for initial understanding and a larger T5 for nuanced comprehension and better text rendering.
*   **Transformer (Main Model - MMDiT):**
    *   **Function:** Core generative part (Multimodal Diffusion Transformer). Generates images in latent space from text embeddings.
*   **Variational Auto-Encoder (VAE):**
    *   **Function:** Translates images between pixel and latent space. Decodes generated latent representation to a pixel-based image.

In our QLoRA approach, we focus exclusively on fine-tuning the **transformer component** (MMDiT). The text encoders and VAE remain frozen throughout training. 

## QLoRA Fine-tuning FLUX.1-dev with `diffusers`

We used a `diffusers` training script (very slightly modified from https://github.com/huggingface/diffusers/tree/main/examples/research_projects/flux_lora_quantization) designed for DreamBooth-style LoRA fine-tuning of FLUX models. Let's examine the crucial parts for QLoRA and memory efficiency:

### Key Optimization Techniques

**LoRA (Low-Rank Adaptation) Deep Dive:**
LoRA makes model training more efficient by keeping track of the weight updates with low-rank matrices. Instead of updating the full weight matrix $$W$$, LoRA learns two smaller matrices $$A$$ and $$B$$. The update to the weights for the model is $$\Delta W = BA$$, where $$A \in \mathbb{R}^{r \times k}$$ and $$B \in \mathbb{R}^{d \times r}$$. The number $$r$$ (called _rank_) is much smaller than the original dimensions, which means less parameters to update. Lastly, $$\alpha$$ is a scaling factor for the LoRA activations. This affects how much LoRA affects the updates, and is often set to the same value as the $$r$$ or a multiple of it. It helps balance the influence of the pre-trained model and the LoRA adapter.

**8-bit Optimizer (AdamW):**
Standard AdamW optimizer maintains first and second moment estimates for each parameter in 32-bit (FP32), which consumes a lot of memory The 8-bit AdamW uses block-wise quantization to store optimizer states in 8-bit precision, while maintaining training stability. This technique can reduce optimizer memory usage by ~75% compared to standard FP32 AdamW.

**Gradient Checkpointing:**
During forward pass, intermediate activations are typically stored for backward pass gradient computation. Gradient checkpointing trades computation for memory by only storing certain _checkpoint activations_ and recomputing others during backpropagation.

**Cache Latents:**
This optimization technique pre-processes all training images through the VAE encoder before the beginning of the training. It stores the resulting latent representations in memory. During the training, instead of encoding images on-the-fly, the cached latents are directly used. This approach offers two main benefits: 
1. eliminates redundant VAE encoding computations during training, speeding up each training step
2. allows the VAE to be completely removed from GPU memory after caching. The trade-off is increased RAM usage to store all cached latents, but this is typically manageable for small datasets.

**Setting up 4-bit Quantization (`BitsAndBytesConfig`):**

This section demonstrates the QLoRA configuration for the base model:
```python
# Determine compute dtype based on mixed precision
bnb_4bit_compute_dtype = torch.float32
if args.mixed_precision == "fp16":
    bnb_4bit_compute_dtype = torch.float16
elif args.mixed_precision == "bf16":
    bnb_4bit_compute_dtype = torch.bfloat16

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
)

transformer = FluxTransformer2DModel.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=bnb_4bit_compute_dtype,
)
# Prepare model for k-bit training
transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
# Gradient checkpointing is enabled later via transformer.enable_gradient_checkpointing() if arg is set
```

**Defining LoRA Configuration (`LoraConfig`):**
Adapters are added to the quantized transformer:
```python
transformer_lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank, 
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"], # FLUX attention blocks
)
transformer.add_adapter(transformer_lora_config)
```
Only these LoRA parameters become trainable.

### Setup & Results

For this demonstration, we leveraged an NVIDIA RTX 4090 (24GB VRAM) to explore its performance.

**Configuration for RTX 4090:**
On our RTX 4090, we used a `train_batch_size` of 1, `gradient_accumulation_steps` of 4, `mixed_precision="fp16"`, `gradient_checkpointing=True`, `use_8bit_adam=True`, a LoRA `rank` of 4, and resolution of 512x768. Latents were cached with `cache_latents=True`.

**Memory Footprint (RTX 4090):**
* **QLoRA:** Peak VRAM usage for QLoRA fine-tuning was approximately 9GB.
* **FP16 LoRA:** Running standard LoRA (with the base FLUX.1-dev in FP16) on the same setup consumed 26 GB VRAM.
* **FP16 full finetuning:** An estimate would be ~120 GB VRAM with no memory optimizations.


**Training Time (RTX 4090):**
Fine-tuning for 700 steps on the Alphonse Mucha dataset took approximately 41 minutes on the RTX 4090.

**Output Quality:**
The ultimate measure is the generated art. Here are samples from our QLoRA fine-tuned model on the [derekl35/alphonse-mucha-style](https://huggingface.co/datasets/derekl35/alphonse-mucha-style) dataset:

base model:
![base model outputs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers2/alphonse_mucha_base_combined.png) 

QLoRA fine-tuned:
![QLoRA model outputs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers2/alphonse_mucha_merged_combined.png) 
*Prompts: (left to right)*

*Serene raven-haired woman, moonlit lilies, swirling botanicals, alphonse mucha style*

*a puppy in a pond, alphonse mucha style*

*Ornate fox with a collar of autumn leaves and berries, amidst a tapestry of forest foliage, alphonse mucha style*

The fine-tuned model nicely captured Mucha's iconic art nouveau style, evident in the decorative motifs and distinct color palette. The QLoRA process maintained excellent fidelity while learning the new style.

## Inference with Trained LoRA Adapters

After training your LoRA adapters, you have two main approaches for inference.

### Option 1: Loading LoRA Adapters

One approach is to [load your trained LoRA adapters](https://huggingface.co/docs/diffusers/v0.33.1/en/using-diffusers/loading_adapters#lora) on top of the base model.

**Benefits of Loading LoRA:**
* **Flexibility:** Easily switch between different LoRA adapters without reloading the base model
* **Experimentation:** Test multiple artistic styles or concepts by swapping adapters
* **Modularity:** Combine multiple LoRA adapters using `set_adapters()` for creative blending
* **Storage efficiency:** Keep a single base model and multiple small adapter files

### Option 2: Merging LoRA into Base Model

For when you want maximum efficiency with a single style, you can [merge the LoRA weights](https://huggingface.co/docs/diffusers/using-diffusers/merge_loras) into the base model.

**Benefits of Merging LoRA:**
- **VRAM efficiency:** No additional memory overhead from adapter weights during inference
- **Speed:** Slightly faster inference as there's no need to apply adapter computations
- **Quantization compatibility:** Can re-quantize the merged model for maximum memory efficiency

**Colab Adaptability:**
<!-- [add a section talking about / change above to be focused on running in google colab] -->

## Conclusion

QLoRA, coupled with the `diffusers` library, significantly democratizes the ability to customize state-of-the-art models like FLUX.1-dev. As demonstrated on an RTX 4090, efficient fine-tuning is well within reach, yielding high-quality stylistic adaptations. Importantly, these techniques are adaptable, paving the way for users on more constrained hardware, like Google Colab, to also participate.

<!-- [Maybe add a link to trained LoRA adapter on Hugging Face Hub.] -->