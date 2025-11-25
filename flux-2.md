---
title: "Diffusers welcomes FLUX-2"
thumbnail: /blog/assets/flux2/thumbnail.png
authors:
  - user: YiYiXu
  - user: dg845
  - user: sayakpaul
  - user: OzzyGT
  - user: dn6
  - user: ariG23498
  - user: linoyts
  - user: multimodalart
---

# Welcome FLUX.2 - BFL’s new open image generation model 🤗

FLUX.2 is the recent series of image generation models from Black Forest Labs, preceded by the [Flux.1](https://huggingface.co/collections/black-forest-labs/flux1) series. It is an entirely new model with a **new architecture** and pre-training done from scratch!
![generation_teaser](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/teaser_generation.png)
In this post, we discuss the key changes introduced in FLUX.2, performing inference with it under various setups, and LoRA fine-tuning.

<aside>
🚨

FLUX.2 is *not* meant to be a drop-in replacement of FLUX.1, but a new generation model

</aside>

**Table of contents**

- FLUX.2 introduction
- Inference with Diffusers
- LoRA fine-tuning

## FLUX.2: A Brief Introduction

FLUX.2 can be used for both **image-guided** and **text-guided** image generation. Furthermore, it can take multiple images as reference inputs, while producing the final output image. Below, we briefly discuss the key changes introduced in FLUX.2.

![editing_teaser](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/teaser_editing.png)

### Text encoder

First, instead of two text encoders as in Flux.1, it uses a single text encoder — [Mistral Small 3.1](https://mistral.ai/news/mistral-small-3-1). Using a single text encoder greatly simplifies the process of computing prompt embeddings. The pipeline allows for a `max_sequence_length` of 512.

### DiT

FLUX.2 follows the same general [multimodel diffusion transformer](https://arxiv.org/pdf/2403.03206) (MM-DiT) + parallel [DiT](https://arxiv.org/pdf/2212.09748) architecture as Flux.1. As a refresher, MM-DiT blocks first process the image latents and conditioning text in separate streams, only joining the two together for the attention operation, and are thus referred to as “double-stream” blocks. The parallel blocks then operate on the concatenated image and text streams and can be regarded as “single-stream” blocks.

The key DiT changes from Flux.1 to FLUX.2 are as follows:

1. Time and guidance information (in the form of [AdaLayerNorm-Zero](https://arxiv.org/pdf/2212.09748) modulation parameters) is shared across all double-stream and single-stream transformer blocks, respectively, rather than having individual modulation parameters for each block as in Flux.1.
2. None of the layers in the model use `bias` parameters. In particular, neither the attention nor feedforward (FF) sub-blocks of either transformer block use `bias` parameters in any of their layers.
3. In Flux.1, the single-stream transformer blocks fused the attention output projection with the FF output projection. FLUX.2 single-stream blocks also fuse the attention QKV projections with the FF input projection, creating a [fully parallel transformer block](https://arxiv.org/pdf/2302.05442):

![Figure taken from the ViT-22B paper.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/image.png)

Figure taken from the ViT-22B paper.

Note that compared to the `ViT-22B` block depicted above, FLUX.2 uses a [SwiGLU](https://arxiv.org/pdf/2002.05202)-style MLP activation rather than a GELU activation (and also doesn’t use `bias` parameters).

1. A larger proportion of the transformer blocks in FLUX.2 are single-stream blocks (`8` double-stream blocks to `48` single-stream blocks, compared to `19`/`38` for Flux.1). This also means that single-stream blocks make up a larger proportion of the DiT parameters: `Flux.1[dev]-12B` has ~54% of its total parameters in the double-stream blocks, whereas `FLUX.2[dev]-32B` has ~24% of its parameters in the double-stream blocks (and ~73% in the single-stream blocks).

### Misc

- A new Autoencoder
- Better way to incorporate resolution-dependent timestep schedules

## Inference With Diffusers

FLUX.2 uses a larger DiT and Mistral3 Small as its text encoder. When used together without any kind of offloading, the inference takes more than **_80GB VRAM_**. In the following sections, we show how to perform inference with FLUX.2 in more accessible ways, under various system-level constraints.

### Installation and Authentication

Before you try out the following code snippets, make sure you have installed `diffusers` from `main` and have run `hf auth login`.

```diff
pip uninstall diffusers -y && pip install git+https://github.com/huggingface/diffusers -U
```

### Regular Inference

```python
from diffusers import Flux2Pipeline
import torch

repo_id = "black-forest-labs/FLUX.2-dev"
pipe = Flux2Pipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="dog dancing near the sun",
    num_inference_steps=50,
    guidance_scale=2.5,
    height=1024,
    idth=1024
).images[0]
```

The above code snippet was tested on an H100, and it isn’t sufficient to run inference on it without CPU offloading. With CPU offloading enabled, this setup takes ~62GB to run.

Users who have access to Hopper-series GPUs can take advantage of **Flash Attention 3** to speed up inference:

```diff
from diffusers import Flux2Pipeline
import torch

repo_id = "black-forest-labs/FLUX.2-dev"
pipe = Flux2Pipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
+ pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="dog dancing near the sun",
    num_inference_steps=50,
    guidance_scale=2.5,
    height=1024,
    idth=1024
).images[0]
```

> [!NOTE]

> You can check out the supported attention backends (we have many!) [here](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends).

### Resource-constrained

**Using 4-bit quantization**

Using `bitsandbytes` , we can load the transformer and text encoder models in 4-bit, allowing owners of 24GB GPUs to use the model locally. You can run this snippet on a GPU with ~20 GB of **free** VRAM.

- Unfold

  ```python
  import torch
  from transformers import Mistral3ForConditionalGeneration

  from diffusers import Flux2Pipeline, Flux2Transformer2DModel

  repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
  device = "cuda:0"
  torch_dtype = torch.bfloat16

  transformer = Flux2Transformer2DModel.from_pretrained(
      repo_id, subfolder="transformer", torch_dtype=torch_dtype, device_map="cpu"
  )
  text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
      repo_id, subfolder="text_encoder", dtype=torch_dtype, device_map="cpu"
  )

  pipe = Flux2Pipeline.from_pretrained(
      repo_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
  )
  pipe.enable_model_cpu_offload()

  prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

  image = pipe(
      prompt=prompt,
      generator=torch.Generator(device=device).manual_seed(42),
      num_inference_steps=50, # 28 is a good trade-off
      guidance_scale=4,
  ).images[0]

  image.save("flux2_t2i_nf4.png")
  ```

**Local + remote**

Due to the _modular design_ of a Diffusers pipeline, we can isolate modules and work with them in sequence. We decouple the text encoder and deploy it to an [Inference Endpoint](https://endpoints.huggingface.co/). This helps us with freeing up the VRAM usage for the DiT and VAE only.

<aside>
⚠️

To use the remote text encoder you need to have a [valid token](https://huggingface.co/docs/hub/en/security-tokens). If you are already authenticated, no further action is needed.

</aside>

The example below uses a combination of local and remote inference. Additionally, we quantize the DiT with NF4 quantization through `bitsandbytes`.

You can run this snippet on a GPU with 18 GB of VRAM:

- Unfold

  ```python
  from diffusers import Flux2Pipeline, Flux2Transformer2DModel
  from diffusers import BitsAndBytesConfig as DiffBitsAndBytesConfig
  from huggingface_hub import get_token
  import requests
  import torch
  import io

  def remote_text_encoder(prompts: str | list[str]):
      def _encode_single(prompt: str):
          response = requests.post(
              "https://remote-text-encoder-flux-2.huggingface.co/predict",
              json={"prompt": prompt},
              headers={
                  "Authorization": f"Bearer {get_token()}",
                  "Content-Type": "application/json"
              }
          )
          assert response.status_code == 200, f"{response.status_code=}"
          return torch.load(io.BytesIO(response.content))

      if isinstance(prompts, (list, tuple)):
          embeds = [_encode_single(p) for p in prompts]
          return torch.cat(embeds, dim=0)

      return _encode_single(prompts).to("cuda")

  repo_id = "black-forest-labs/FLUX.2-dev"
  quantized_dit_id = "diffusers/FLUX.2-dev-bnb-4bit"
  dit = Flux2Transformer2DModel.from_pretrained(
      quantized_dit_id, subfolder="transformer", torch_dtype=torch_dtype, device_map="cpu"
  )

  pipe = Flux2Pipeline.from_pretrained(
      repo_id,
      text_encoder=None,
      transformer=dit,
      torch_dtype=torch.bfloat16,
  )
  pipe.enable_model_cpu_offload()

  print("Running remote text encoder ☁️")
  prompt1 = "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture"
  prompt2 = "a photo of a dense forest with rain. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture"
  prompt_embeds = remote_text_encoder([prompt1, prompt2])
  print("Done ✅")

  out = pipe(
      prompt_embeds=prompt_embeds,
      generator=torch.Generator(device="cuda").manual_seed(42),
      num_inference_steps=50, # 28 is a good trade-off
      guidance_scale=4,
      height=1024,
      width=1024,
  )

  for idx, image in enumerate(out.images):
      image.save(f"flux_out_{idx}.png")
  ```

For GPUs with even lower VRAM, we have `group_offloading`, which allows GPUs with as little as 8GB of **free** VRAM to use this model. However, you'll need 32GB of **free** RAM. Alternatively, if you're willing to sacrifice some speed, you can set `low_cpu_mem_usage=True` to reduce the RAM requirement to just 10GB.

- Unfold

  ```python
  import io
  import os

  import requests
  import torch

  from diffusers import Flux2Pipeline, Flux2Transformer2DModel

  repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
  torch_dtype = torch.bfloat16
  device = "cuda"

  def remote_text_encoder(prompts: str | list[str]):
      def _encode_single(prompt: str):
          response = requests.post(
              "https://remote-text-encoder-flux-2.huggingface.co/predict",
              json={"prompt": prompt},
              headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}", "Content-Type": "application/json"},
          )
          assert response.status_code == 200, f"{response.status_code=}"
          return torch.load(io.BytesIO(response.content))

      if isinstance(prompts, (list, tuple)):
          embeds = [_encode_single(p) for p in prompts]
          return torch.cat(embeds, dim=0)

      return _encode_single(prompts).to("cuda")

  transformer = Flux2Transformer2DModel.from_pretrained(
      transformer_id, subfolder="transformer", torch_dtype=torch_dtype, device_map="cpu"
  )

  pipe = Flux2Pipeline.from_pretrained(
      repo_id,
      text_encoder=None,
      transformer=transformer,
      torch_dtype=torch_dtype,
  )
  pipe.transformer.enable_group_offload(
      onload_device=device,
      offload_device="cpu",
      offload_type="leaf_level",
      use_stream=True,
      # low_cpu_mem_usage=True # uncomment for lower RAM usage
  )
  pipe.to(device)

  prompt = "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture"
  prompt_embeds = remote_text_encoder(prompt)

  image = pipe(
      prompt_embeds=prompt_embeds,
      generator=torch.Generator(device=device).manual_seed(42),
      num_inference_steps=50,
      guidance_scale=4,
      height=1024,
      width=1024,
  ).images[0]

  ```

> [!NOTE]

> You can check out other supported quantization backends [here](https://huggingface.co/docs/diffusers/main/en/quantization/overview) and other memory-saving techniques [here](https://huggingface.co/docs/diffusers/main/en/optimization/memory).

To check how different quantizations affect an image, you can play with the playground below or access it as standlone in the [FLUX.2 Quantization experiments Space](https://huggingface.co/spaces/multimodalart/flux2-quantization)

<iframe 
  src="https://multimodalart-flux2-quantization.static.hf.space/index.html"
  width="100%"
  height="800"
  style="border: none;"
  allow="clipboard-write; clipboard-read;">
</iframe>

### Multiple images as reference

FLUX.2 supports using multiple images as inputs, allowing you to use up to 10 images. However, keep in mind that each additional image will require more VRAM. You can reference the images by index (e.g., image 1, image 2) or by natural language (e.g., the kangaroo, the turtle). For optimal results, the best approach is to use a combination of both methods.

- Unfold

  ```python
  import torch
  from transformers import Mistral3ForConditionalGeneration

  from diffusers import Flux2Pipeline, Flux2Transformer2DModel
  from diffusers.utils import load_image

  repo_id = "diffusers-internal-dev/new-model-image-final-weights"
  device = "cuda:0"
  torch_dtype = torch.bfloat16

  pipe = Flux2Pipeline.from_pretrained(
      repo_id, torch_dtype=torch_dtype
  )
  pipe.enable_model_cpu_offload()

  image_one = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/kangaroo.png")
  image_two = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/turtle.png")

  prompt = "the boxer kangaroo from image 1 and the martial artist turtle from image 2 are fighting in an epic battle scene at a beach of a tropical island, 35mm, depth of field, 50mm lens, f/3.5, cinematic lighting"

  image = pipe(
      prompt=prompt,
      image=[image_one, image_two],
      generator=torch.Generator(device=device).manual_seed(42),
      num_inference_steps=50,
      guidance_scale=2.5,
      width=1024,
      height=768,
  ).images[0]

  image.save(f"./flux2_t2i.png")
  ```

![kangaroo.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/kangaroo.png)

![turtle.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/turtle.png)

![two_images_result.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/two_images_result.png)

## LoRA fine-tuning

Being both a text-to-image and an image-to-image model, FLUX.2 makes the perfect fine-tuning candidate for many use-cases! However, as inference alone takes more than 80GB of VRAM, LoRA fine-tuning is even more challenging to run on consumer GPUs. To squeeze in as much memory saving as we can, we utilize some of the inference optimizations described above for training as well, together with shared memory saving techniques, to substantially reduce memory consumption. To train it you can use either the diffusers code below or [Ostris' AI Toolkit](https://github.com/ostris/ai-toolkit)

> [!NOTE]

> We provide both text-to-image and image-to-image training scripts, for the purpose of this blog will focus on a text-to-image training example.

### Memory optimizations for fine-tuning

Many of these techniques complement each other and can be used together to reduce memory consumption further. However, some techniques may be mutually exclusive, so be sure to check before launching a training run.

- Unfold to check details on the memory-saving techniques used:
  - **Remote Text Encoder:** to leverage the remote text encoding for training, simply pass `--remote_text_encoder`. Note that you must either be logged in to your Hugging Face account (`hf auth login`) OR pass a token with `--hub_token`.
  - **CPU Offloading:** by passing `--offload` the vae and text encoder to will be offloaded to CPU memory and only moved to GPU when needed.
  - **Latent Caching:** Pre-encode the training images with the vae, and then delete it to free up some memory. To enable `latent_caching` simply pass `--cache_latents`.
  - **QLoRA**: Low Precision Training with Quantization - using 8-bit or 4-bit quantization. You can use the following flags:
    **FP8 training** with `torchao`: enable FP8 training by passing `--do_fp8_training`.
    - > [!IMPORTANT] Since we are utilizing FP8 tensor cores, we need CUDA GPUs with compute capability at least 8.9 or greater. > If you're looking for memory-efficient training on relatively older cards, we encourage you to check out other trainers like `SimpleTuner`, `ai-toolkit`, etc.
      > **NF4 training** with `bitsandbytes`: Alternatively, you can use 8-bit or 4-bit quantization with `bitsandbytes` by passing:- `--bnb_quantization_config_path` with a corresponding path to a json file containing your config. see below for more details.
  - **Gradient Checkpointing and Accumulation:** `--gradient accumulation` refers to the number of updates steps to accumulate before performing a backward/update pass.by passing a value > 1 you can reduce the amount of backward/update passes and hence also memory reqs.\* with `--gradient checkpointing` we can save memory by not storing all intermediate activations during the forward pass.Instead, only a subset of these activations (the checkpoints) are stored and the rest is recomputed as needed during the backward pass. Note that this comes at the expanse of a slower backward pass.
  - **8-bit-Adam Optimizer:** When training with `AdamW`(doesn't apply to `prodigy`) You can pass `--use_8bit_adam` to reduce the memory requirements of training. Make sure to install `bitsandbytes` if you want to do so.

Let’s launch a training run using these memory saving optimizations.

> [!NOTE]
> Please make sure to check out the [README](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md) for prerequisites before starting training.

For this example, we’ll use `multimodalart/1920-raider-waite-tarot-public-domain` dataset with the following configuration using FP8 training. Feel free to experiment more with the hyper-parameters and share your results 🤗

```bash
!accelerate launch train_dreambooth_lora_flux2.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.2-dev"  \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --remote_text_encoder \
  --cache_latents \
  --caption_column="caption"\
  --do_fp8_training \
  --dataset_name="multimodalart/1920-raider-waite-tarot-public-domain" \
  --output_dir="tarot_card_Flux2_LoRA" \
  --instance_prompt="trcrd tarot card" \
  --resolution=1024 \
  --train_batch_size=2 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="adamW" \
  --use_8bit_adam\
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --checkpointing_steps=250\
  --max_train_steps=1000 \
  --rank=8\
  --validation_prompt="a trtcrd of a person on a computer, on the computer you see a meme being made with an ancient looking trollface, 'the shitposter' arcana, in the style of TOK a trtcrd, tarot style" \
  --validation_epochs=25 \
  --seed="0"\
  --push_to_hub
```

<p>
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/image%201.png" width="45%" />
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/flux2_blog/image%202.png" width="45%" />
</p>
 
The left image was generated using the pre-trained FLUX.2 model, and the right image was produced the LoRA.

In case your hardware isn’t compatible with FP8 training, you can use QLoRA with `bitsandbytes`. You first need to define a `config.json` file like so:

```python
{
    "load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4"
}
```

And then pass its path to `--bnb_quantization_config_path`:

```bash
!accelerate launch train_dreambooth_lora_flux2.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.2-dev"  \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --remote_text_encoder \
  --cache_latents \
  --caption_column="caption"\
  **--bnb_quantization_config_path="config.json" \**
  --dataset_name="multimodalart/1920-raider-waite-tarot-public-domain" \
  --output_dir="tarot_card_Flux2_LoRA" \
  --instance_prompt="a tarot card" \
  --resolution=1024 \
  --train_batch_size=2 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="adamW" \
  --use_8bit_adam\
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --max_train_steps=1000 \
  --rank=8\
  --validation_prompt="a trtcrd of a person on a computer, on the computer you see a meme being made with an ancient looking trollface, 'the shitposter' arcana, in the style of TOK a trtcrd, tarot style" \
  --seed="0"
```

## Resources

- FLUX.2 [announcement post](https://bfl.ai/blog/flux-2)
- Diffusers [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2)
- FLUX.2 official [demo](https://huggingface.co/spaces/black-forest-labs/FLUX.2-dev)
- FLUX.2 on the [Hub](https://hf.co/black-forest-labs/FLUX.2-dev)
- FLUX.2 original [codebase](https://github.com/black-forest-labs/flux2-dev)
