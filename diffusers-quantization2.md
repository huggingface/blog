# Fine-Tuning FLUX.1-dev with QLoRA

In our previous post, "[Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization)", we dived into how various quantization techniques can shrink diffusion models like FLUX.1-dev, making them significantly more accessible for *inference* without drastically compromising performance. We saw how `bitsandbytes`, `torchao`, and others reduce memory footprints for generating images.

Now, we tackle **efficiently *fine-tuning* these models.** This post will guide you through fine-tuning FLUX.1-dev using QLoRA with the Hugging Face `diffusers` library. We'll showcase results from an NVIDIA RTX 4090.

## Why Not Just Full Fine-Tuning?

`black-forest-labs/FLUX.1-dev`, for instance, requires over 31GB in BF16 for inference alone.

**Full Fine-Tuning:** This traditional method updates all model weights.
* **Pros:** Potential for the highest task-specific quality.
* **Cons:** For FLUX.1-dev, this would demand immense VRAM (multiple high-end GPUs), putting it out of reach for most individual users.

**LoRA (Low-Rank Adaptation):** LoRA freezes the pre-trained weights and injects small, trainable "adapter" layers.
* **Pros:** Massively reduces trainable parameters, saving VRAM during training and resulting in small adapter checkpoints.
* **Cons (for base model memory):** The full-precision base model still needs to be loaded, which, for FLUX.1-dev, is still a hefty VRAM requirement even if fewer parameters are being updated.

**QLoRA: The Efficiency Powerhouse:** QLoRA enhances LoRA by:
1.  Loading the pre-trained base model in a quantized format (typically 4-bit via `bitsandbytes`), drastically cutting the base model's memory footprint.
2.  Training LoRA adapters (usually in FP16/BF16) on top of this quantized base.

This allows fine-tuning of very large models on consumer-grade hardware or more accessible cloud GPUs.

## Dataset

 We aimed to fine-tune `black-forest-labs/FLUX.1-dev` to adopt the artistic style of Alphonse Mucha, using a small [dataset](https://huggingface.co/datasets/derekl35/alphonse-mucha-style). 
<!-- (maybe use different dataset) -->

## QLoRA Fine-tuning FLUX.1-dev with `diffusers`

We used a `diffusers` training script (very slightly modified from https://github.com/huggingface/diffusers/tree/main/examples/research_projects/flux_lora_quantization) designed for DreamBooth-style LoRA fine-tuning of FLUX models. Let's examine the crucial parts for QLoRA and memory efficiency:

**Understanding the Key Optimization Techniques:**

**LoRA (Low-Rank Adaptation) Deep Dive:**
LoRA works by decomposing weight updates into low-rank matrices. Instead of updating the full weight matrix $$W$$, LoRA learns two smaller matrices $$A$$ and $$B$$ such that the update is $$\Delta W = BA$$, where $$A \in \mathbb{R}^{r \times k}$$ and $$B \in \mathbb{R}^{d \times r}$$. The rank $$r$$ is typically much smaller than the original dimensions, drastically reducing trainable parameters. LoRA $$\alpha$$ is a scaling factor for the LoRA activations, often set to the same value as the $$r$$ or a multiple of it. It helps balance the influence of the pre-trained model and the LoRA adapter.

**8-bit Optimizer (AdamW):**
Standard AdamW optimizer maintains first and second moment estimates for each parameter in FP32, consuming significant memory. The 8-bit AdamW uses block-wise quantization to store optimizer states in 8-bit precision while maintaining training stability. This technique can reduce optimizer memory usage by ~75% compared to standard FP32 AdamW.

**Gradient Checkpointing:**
During forward pass, intermediate activations are typically stored for backward pass gradient computation. Gradient checkpointing trades computation for memory by only storing certain "checkpoint" activations and recomputing others during backpropagation.

<!-- maybe explain cache latents -->


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
The ultimate measure is the generated art. Here are samples from our QLoRA fine-tuned model on the `derekl35/alphonse-mucha-style` dataset:

base model:
![base model outputs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers2/alphonse_mucha_base_combined.png) 

QLoRA fine-tuned:
![QLoRA model outputs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/quantization-backends-diffusers2/alphonse_mucha_merged_combined.png) 
*Prompts: (left to right)*

*Serene raven-haired woman, moonlit lilies, swirling botanicals, alphonse mucha style*

*a puppy in a pond, alphonse mucha style*

*Ornate fox with a collar of autumn leaves and berries, amidst a tapestry of forest foliage, alphonse mucha style*

The fine-tuned model nicely captured Mucha's iconic art nouveau style, evident in the decorative motifs and distinct color palette. The QLoRA process maintained excellent fidelity while learning the new style.

**Colab Adaptability:**
<!-- [add a section talking about / change above to be focused on running in google colab] -->

## Conclusion

QLoRA, coupled with the `diffusers` library, significantly democratizes the ability to customize state-of-the-art models like FLUX.1-dev. As demonstrated on an RTX 4090, efficient fine-tuning is well within reach, yielding high-quality stylistic adaptations. Importantly, these techniques are adaptable, paving the way for users on more constrained hardware, like Google Colab, to also participate.

<!-- [Maybe add a link to trained LoRA adapter on Hugging Face Hub.] -->