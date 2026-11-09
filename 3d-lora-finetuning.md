---
title: "The Definitive Fine-Tuning Guide for TRELLIS.2" 
thumbnail: /blog/assets/3d-lora-finetuning/thumbnail.gif
authors:
  - user: suvadityamuk
  - user: merve
  - user: pcuencq
---

# The Definitive Fine-Tuning Guide for TRELLIS.2

![Hero GIF](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/sole_orbit_base_vs_lora.gif)

> TLDR; In this blog, we provide fine-tuning scripts and containers to fine-tune TRELLIS.2, a popular 3D model on your 3D assets, that can run both locally and on Hugging Face Jobs. If you are interested in fine-tuning, you can [use our scripts](#walkthrough) with few arguments (assuming you updated your dataset to Hugging Face Hub) or tinker with them yourself [with this repository](https://github.com/suvadityamuk/trellis2-finetuning). If you want to learn more, we go through [how the script works](#inside-the-training-pipeline). Let’s go!

## Introduction

3D generation has many uses in robotics, simulation, media, and entertainment. In these fields, it is difficult to make generated objects obey strict rules and standardize model and data representations. There are many file formats, such as [GLB](https://en.wikipedia.org/wiki/GlTF) and [PLY](https://en.wikipedia.org/wiki/PLY_\(file_format\)). There are also [many representations](https://huggingface.co/spaces/suvadityamuk/3d-representations-guide), such as [neural radiance fields (NeRFs)](https://www.matthewtancik.com/nerf) and [3D Gaussian splatting (3DGS)](https://en.wikipedia.org/wiki/Gaussian_splatting). Each one has different advantages and disadvantages. 

[TRELLIS](https://huggingface.co/papers/2412.01506) introduced a common internal representation, called a unified latent representation. Its decoders convert this representation into 3D models in the required output format. Thus, you do not have to use different models for each representation format.

The same team then released [TRELLIS.2](https://huggingface.co/papers/2512.14692), which improves the model's representation of 3D structure. But neither model gives an easy way to adapt it to a specific task.

### What we want to improve

A general-purpose model, such as TRELLIS.2, can make errors in details that are important for a specific application. For example, a user may need accurate geometry for furniture models. Without fine-tuning for that task, the pretrained model can produce geometry that is insufficiently accurate. We use a subset of GSO that has 188 footwear objects for our experiment. In our initial tests, the pretrained model usually produced shoes that looked correct, but the soles were incorrect. We test whether a LoRA can make the soles more accurate.

![Sole Orbit Base](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/sole_orbit_base.gif)

Our goal is to use the [Google Scanned Objects (GSO)](https://huggingface.co/datasets/hugging-3d/google-scanned-objects) dataset to improve the performance of TRELLIS.2 on a specific object category. We will explain which pipeline components to fine-tune and which to leave unchanged during training. The pipeline separates structure, detailed geometry, and materials. Thus, we will fine-tune only the components that are important for our task. 

![Meme](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/meme.jpg)
> *Choosing what to freeze matters as much as choosing what to fine-tune.*

### What will we use?

#### LoRA

LoRA, short for low-rank adaptation, fine-tunes a model by training small adapter matrices while keeping the original weights frozen. This reduces the number of trainable parameters and the memory needed for their gradients and optimizer state. We use Hugging Face’s PEFT library to add and train these adapters. For more information, read our [LoRA guide](https://huggingface.co/docs/peft/package_reference/lora).

#### Hugging Face Jobs

Hugging Face Jobs runs containerized workloads on cloud hardware. We use it for preprocessing and for training. To use your own GPU instead, see the local Docker commands below. Before you start, make sure that you have:

- A HF_TOKEN that you can obtain [from here](https://huggingface.co/settings/tokens/new)  
- Your Hugging Face account must have [access to DINOv3](https://huggingface.co/collections/facebook/dinov3), because DINOv3 is a gated model  
- If you use your own dataset, upload its meshes to the Hub before you start. 

## How TRELLIS.2 works

Before we train, let’s look at how TRELLIS.2 generates a 3D object and where we’ll add LoRA adapters.

TRELLIS.2 has five main stages: remove the image background, encode the image, predict the coarse structure, generate detailed geometry, and generate materials. The pipeline then exports the result as a textured mesh.

![TRELLIS.2 Pipeline](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/pipeline_lr.png)

### Encoding the input image

First, BiRefNet removes the background from the input image. DINOv3-L then encodes the image into feature tokens. The generation models use these tokens to guide the 3D object.

![Pixels to feature tokens](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/image_to_tokens.png)

### Sparse Structure Generation (Sparse Structure Flow)

The sparse-structure flow DiT predicts the object's coarse shape. It is a diffusion transformer (DiT) that makes a small latent of size 16³ × 8. A variational autoencoder (VAE) decoder then changes this latent into a 64 × 64 × 64 occupancy grid. Each voxel in the grid shows if that location is part of the object. A voxel is a pixel in three dimensions. We add LoRA adapters to this model because its output guides the geometry stage.  
![Stage 1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/stage1_sparse_structure.png)

### O-Voxel and Structured Latents

O-Voxel is the 3D format of TRELLIS.2. It is a sparse voxel grid. It keeps only the voxels on the object's surface. Each voxel keeps local geometry, such as a mesh vertex, and material attributes. Because the format is small, a CPU changes a mesh into O-Voxel in seconds. A GPU changes O-Voxel back into a mesh in seconds.

Structured sparse latents come from TRELLIS. They keep 3D data in sparse tensors, which makes computation faster. The model uses only approximately 10% of the voxels in the grid, so it does not spend compute on empty voxels. Thus, the shape flow DiT and the texture flow DiT are sparse transformers. The sparse-structure flow DiT is a dense transformer.  
![O-Voxel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/ovoxel_sparse.png)

### Geometry Generation (Shape Flow)

The shape flow DiT uses the occupancy grid and the DINOv3 feature tokens to make a latent. Another VAE decoder then changes this latent into O-Voxel geometry at a resolution of 512 × 512 × 512.

We add LoRA adapters to the DiT at this stage because it controls the object's shape.

![Stage 2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/stage2_shape.png)

### Material Generation + Mesh Creation

In the last stage, the texture flow DiT uses the shape latent and the DINOv3 features as input. It defines the materials and textures of the geometry's surface. A VAE decoder then changes the latent into base color, metallic, roughness, and alpha values for each voxel. We do not add LoRA adapters to this stage. Our dataset has diffuse color maps, but it lacks the separate material maps this stage needs for training. Also, our problem is the shoe's shape. The base model already makes good colors and textures.

Finally, the pipeline combines the geometry and material attributes into a textured mesh and exports it as a GLB file.

![Stage 3](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/stage3_texture.png)

## Planning

### Which components to fine-tune

We add LoRA adapters only to the sparse-structure flow DiT and the shape flow DiT. These DiTs control the object's shape and geometry. We freeze the texture flow DiT to keep its current capabilities.

Our dataset contains diffuse color maps that include baked lighting effects. To train the texture flow DiT, we need separate material maps. The color maps in our dataset mix surface color with lighting. Thus, they are not correct targets for this training.

If we train the texture flow DiT on these maps, it can learn to show lighting as a surface property. It can also forget what it learned before, a problem called catastrophic forgetting.

### Why use LoRA

Our subset has 188 shoes from the 1,030 objects in GSO. For a dataset this small, LoRA is a better choice than full fine-tuning. Full fine-tuning costs much more. It can also overwrite what the model already does well (its learned priors). Our configuration trains approximately 9.3 million adapter parameters in total for both targeted DiTs. We use rank 8 adapters in the attention and feed-forward (MLP) layers of the sparse-structure flow DiT. We use rank 16 adapters in the attention layers of the shape flow DiT. The rank sets the size of the adapter matrices. A higher rank lets the adapter learn more. 

### What you’ll run

A 3D model setup can need many dependencies and installation steps. We supply two Docker images (that you can run on [Jobs](https://hf.co/jobs) or your own machine) with the necessary software. One image is for preprocessing, and one is for fine-tuning.

To train your LoRA adapters and upload them to the Hub:

1. Give the containers access to your data.  
2. Set the parameters for your task.  
3. Run the preprocessing container.  
4. Run the fine-tuning container.

Then we explain the fine-tuning script, its settings, and how each setting affects training.

### Experiment overview

We’ll walk through three parts of the experiment:

1. **Prepare the data.** Convert GLB or OBJ meshes into the latent representations TRELLIS.2 uses for training, and render the input images.  
2. **Train the adapters.** Run one job for each DiT, with the prepared dataset and a LoRA configuration.  
3. **Explain the code.** Examine how the containers load the data, add adapters, and train the models.

   
We save the preprocessed dataset so that all training runs can reuse it. 

## Walkthrough

### Running on Hugging Face Jobs

#### Preprocessing

The preprocessing container uses Blender to render 16 views of each object. LoRA training will use these views as conditioning images. Blender runs without a display, through the xvfb-run utility. The container also processes each object through the [preprocessing pipeline](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md). See the [footwear dataset here](https://huggingface.co/datasets/suvadityamuk/gso-footwear-trellis2-latents) for an example intermediate output. For our 188 objects, the full run took approximately 1 hour on an a100-large job. A compact version of the preprocessing pipeline is also available in [this companion repository](https://github.com/suvadityamuk/trellis2-finetuning/blob/master/preprocess.py). 

```sh
export HF_TOKEN=hf_xxx  

# Replace your-username with your Hugging Face username

hf jobs run -d --flavor a100-large --timeout 6h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-prep:latest \
  trellis2-prep \
    --meshes your-username/my-meshes \
    --push-to your-username/my-meshes-trellis2 \
    --private --holdout 0.05 \
    --views 16 --workers 8
```

For Hugging Face Jobs, we pass the following arguments:

| flag | meaning |
| ----- | ----- |
| `-d` | Detach: print the job ID and URL, then return. To follow the job, run `hf jobs logs -f <id>` |
| `--flavor a100-large` | The hardware. We use a100-large (80 GB). To see all options, run hf jobs hardware |
| `--timeout 6h` | The job stops after this time. Both commands resume. Preprocessing skips finished objects. Training continues from the newest checkpoint in --push-to |
| `--secrets HF_TOKEN` | Send your token (from HF_TOKEN or hf auth login) to the job as an encrypted environment variable. The token is not written into the image |
| `cr.hf.co/...:latest` | The image to run. All text after it is the command that runs in the container |

This container accepts a few more arguments (regardless of being run on Hugging Face Jobs or with Docker), that you can [find documented over here](https://github.com/suvadityamuk/trellis2-finetuning#1-preprocess-trellis2-prep). The steps taken in the preprocessing pipeline are documented in [this section](#preprocessing-into-latents) later.

#### LoRA Fine-tuning

The training container loads the prepared dataset and a configuration preset or file. It records training metrics with Trackio. Use the Trackio dashboard to monitor the run.

We use ‘ss’ and ‘shape’ to indicate the two modes we have for training the LoRAs. The ‘ss’ mode will finetune only the Sparse-Structure Flow DiT, whereas the ‘shape’ mode will finetune the Shape Flow DiT. Under the hood, they simply reference different config.json files to use for each stage. More details about the process is available in [this section](#training) later.

```sh
export HF_TOKEN=hf_xxx  # Replace your-username with your Hugging Face username

hf jobs run -d --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config ss \
    --latents your-username/my-meshes-trellis2 \
    --push-to your-username/my-trellis2-lora \
    --run-name ss_lora \
    --push-path ss_lora \
    --trackio-space your-username/trellis2-trackio

hf jobs run -d --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config shape \
    --latents your-username/my-meshes-trellis2 \
    --push-to your-username/my-trellis2-lora \
    --run-name shape_lora \
    --push-path shape_lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500
```

If you set `--push-to`, the container uploads adapter checkpoints to the Hub. The container accepts parameters that are also [documented here for brevity](https://github.com/suvadityamuk/trellis2-finetuning#3-train-trellis2-train). A compact script is also available in the [same repository](https://github.com/suvadityamuk/trellis2-finetuning/blob/master/finetune.py).

If you want to play around and fine-tune with the [already-processed dataset we have](https://huggingface.co/datasets/suvadityamuk/gso-footwear-trellis2-latents), feel free to use the following set of commands. 

```sh
export HF_TOKEN=hf_xxx  

# Replace your-username with your Hugging Face username

# For the Sparse-Structure Flow DiT
hf jobs run -d --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train \
    --config ss \
    --latents suvadityamuk/gso-footwear-trellis2-latents \
    --push-to your-username/trellis2-footwear-lora \
    --run-name ss_lora \
    --push-path ss_lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500

# For the Shape Flow DiT
hf jobs run -d --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train \
    --config shape \
    --latents suvadityamuk/gso-footwear-trellis2-latents \
    --push-to your-username/trellis2-footwear-lora \
    --run-name shape_lora \
    --push-path shape_lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500
```

### Running locally with Docker

##### Preprocessing

The commands and flags are the same as for Hugging Face Jobs. Only the Docker options are different. 

```sh
# For preprocessing into latents
docker run --gpus all --rm --shm-size 16g -e HF_TOKEN=hf_xxx \
  -v hf-cache:/root/.cache/huggingface -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-prep:latest \
  trellis2-prep --meshes your-username/my-meshes --push-to your-username/my-meshes-trellis2
```

##### LoRA Fine-tuning

```sh
# For Sparse-structure Flow DiT
docker run --gpus all --rm --shm-size 16g \
  -v hf-cache:/root/.cache/huggingface -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config ss --latents your-username/my-meshes-trellis2 \
    --push-to your-username/my-trellis2-lora --trackio-space your-username/trellis2-trackio

# For Shape Flow DiT
docker run --gpus all --rm --shm-size 16g \
  -v hf-cache:/root/.cache/huggingface -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config shape --latents your-username/my-meshes-trellis2 \
    --push-to your-username/my-trellis2-lora --trackio-space your-username/trellis2-trackio
```

If you want to use our [already-preprocessed dataset](https://huggingface.co/datasets/suvadityamuk/gso-footwear-trellis2-latents) locally on your machine of choice, use the following commands:

```sh
export HF_TOKEN=hf_xxx  # Replace your-username with your Hugging Face username

# For Sparse-structure Flow DiT
docker run --gpus all --rm --shm-size 16g -e HF_TOKEN \
  -v hf-cache:/root/.cache/huggingface \
  -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train \
    --config ss \
    --latents suvadityamuk/gso-footwear-trellis2-latents \
    --push-to your-username/trellis2-footwear-lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500

# For Shape Flow DiT
docker run --gpus all --rm --shm-size 16g -e HF_TOKEN \
  -v hf-cache:/root/.cache/huggingface \
  -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train \
    --config shape \
    --latents suvadityamuk/gso-footwear-trellis2-latents \
    --push-to your-username/trellis2-footwear-lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500
```

Our trained LoRA adapters are in this repository: [suvadityamuk/trellis-2-finetune-gso](https://huggingface.co/suvadityamuk/trellis-2-finetune-gso)

## Inside the training pipeline

#### Preprocessing into latents

![Preprocessing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/preprocessing.png)
We use TRELLIS.2’s frozen encoders to convert the input meshes into training latents. We save these latents, so later runs do not have to encode the meshes again. 

We also render 16 views of each object in Blender. During training, we select one random view for each sample as the conditioning image. We keep 9 of the 188 objects for evaluation and train on the other 179. These held-out objects show whether the adapters work on objects unseen by the model during training.

![Conditioning](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/conditioning_views.png)
> *Some of the views that Blender rendered for one object*

The first snippet sets up the data. `FLOW` selects the flow DiT to train (`ss` or `shape`), and `STEPS` sets the number of optimizer steps. A batch size of 4 with 4 gradient accumulation steps gives an effective batch size of 16. Then, the snippet loads the LoRA config file for the selected flow DiT, makes the dataset in the TRELLIS.2 format, removes the held-out objects, and makes the data loader for fine-tuning.

```py
cfg = json.load(
    open(
        {
            "ss": "/workspace/TRELLIS.2/configs/lora/ss_flow_lora_gso.json",
            "shape": "/workspace/TRELLIS.2/configs/lora/slat_flow_img2shape_lora_gso.json",
        }[FLOW]
    )
)

key, latents = {
    "ss": ("ss_latent", "ss_latents/ss_enc_conv3d_16l8_fp16_64"),
    "shape": ("shape_latent", "shape_latents/shape_enc_next_dc_f16c32_fp16_512"),
}[FLOW]


roots = json.dumps(
    {
        "gso": {
            key: f"{DATA}/{latents}",
            "render_cond": f"{DATA}/renders_cond",
            "meta": DATA,
        }
    }
)

ds = getattr(datasets, cfg["dataset"])(roots, **cfg["dataset_args"])

holdout = set(open(f"{DATA}/holdout.txt").read().split())
ds.instances = [(root, sha) for root, sha in ds.instances if sha not in holdout]

batches = cycle(
    DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        collate_fn=getattr(ds, "collate_fn", None),
    )
)
```

#### Preparing LoRA adapters

![Preparing LoRA Adapters](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/lora_adapters.png)
We use the `peft` library to insert LoRA adapters into the selected layers. TRELLIS.2 has a custom forward pass, so the standard `PeftModel()` wrapper does not work. Instead, we call `inject_adapter_in_model()`. This function adds the adapters directly to the model and keeps its custom interface.

We keep the trainable adapter weights in 32-bit floating point (`fp32`) for numerical stability and use bfloat16 (`bf16`) autocasting for the forward pass. Autocasting lets PyTorch use lower precision for some operations, which decreases memory use. Autocasting does not change how the frozen backbone weights are stored. 

```py

model = models.from_pretrained(cfg["ckpt"])
for block in model.blocks:
    block.use_checkpoint = True

inject_adapter_in_model(
    LoraConfig(
        r=cfg["rank"], lora_alpha=cfg["alpha"], target_modules=cfg["lora_targets"]
    ),
    model,
)

params = []
for name, p in model.named_parameters():
    p.requires_grad_("lora_" in name)
    if p.requires_grad:
        p.data = p.data.float()  # fp32 master weights for LoRA
        params.append(p)

model.cuda().train()


```

#### Training

![Training Loop](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/training_loop.png)
It’s important to note that our objective function here is good-old rectified flow matching (as seen in popular image-generation models like [Stable Diffusion 3](https://huggingface.co/collections/stabilityai/stable-diffusion-3), [Flux 2](https://huggingface.co/collections/black-forest-labs/flux2), and others). TRELLIS.2 used the same objective in pretraining, so it is a good choice for fine-tuning. For the theory, read [this blog](https://huggingface.co/spaces/ariG23498/flow-matching) about flow matching.

We update only the LoRA parameters in the sparse-structure and shape DiTs. All original model weights remain frozen. 

>> The snippets below show the core training loop for one stage. They are simpler than the container code and do not include some setup steps. For the full workflow, use the container commands.

We initialize the optimizer and load DINOv3 to extract features from the input images. 

```py
opt = torch.optim.AdamW(
    params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95)
)

dino = DinoV3FeatureExtractor(
    cfg["image_cond_model"], image_size=cfg["image_size"]
).cuda()

```

We then follow these steps within the training loop:

* Sample a timestep *t* for each sample (using logitNormal for the sparse-structure flow DiT and uniform for the shape flow DiT based on the TRELLIS.2 config).  
* Generate random noise matching the data shape and reshape *t* for broadcasting.  
* Mix clean data with noise based on *t* (where *t = 0* is data with minimal noise, and *t = 1* is pure noise).  
* Predict velocity at *t × 1000* to match pretraining conditions.  
* Calculate loss as the mean squared error between the predicted velocity and target velocity (the first derivative of *x_t*).

```py
def flow_loss(x0, cond):
    s = cfg["t_schedule"]

    t = (
        torch.rand(x0.shape[0])
        if s["name"] == "uniform"
        else torch.sigmoid(
            torch.randn(x0.shape[0]) * s["args"]["std"] + s["args"]["mean"]
        )
    )
    t = t.cuda()

    noise = (
        x0.replace(torch.randn_like(x0.feats))
        if cfg["sparse"]
        else torch.randn_like(x0)
    )

    tv = t.view(-1, *[1] * (len(x0.shape) - 1))

    x_t = (1 - tv) * x0 + (cfg["sigma_min"] + (1 - cfg["sigma_min"]) * tv) * noise

    pred, target = model(x_t, t * 1000, cond), (1 - cfg["sigma_min"]) * noise - x0

    return (
        F.mse_loss(pred.feats, target.feats)
        if cfg["sparse"]
        else F.mse_loss(pred, target)
    )

```

The training loop computes the loss and updates the adapter weights. Gradient accumulation adds the gradients from 4 small batches before each optimizer step. Gradient checkpointing saves GPU memory by recomputing some intermediate values during backpropagation. The outer loop runs for a given number of optimizer steps, whereas the inner loop accumulates gradients over a set of batches.

For each batch, DINOv3 encodes the conditioning image. For [classifier-free guidance](https://huggingface.co/papers/2207.12598) (CFG) dropout, the loop sets the conditioning to zero for a random fraction (p_uncond) of the samples. Then, it calculates the flow-matching loss with bf16 autocasting. After the inner loop, it clips the gradients to prevent very large updates and does an optimizer step. It logs the loss every 10 steps and saves an adapter checkpoint every 500 steps.

```py

for step in range(1, STEPS + 1):
    for _ in range(ACCUM):
        batch = next(batches)
        x0 = recursive_to_device(batch["x_0"], "cuda")

        with torch.no_grad():
            cond = dino(batch["cond"])
            cond[torch.rand(len(cond), device="cuda") < cfg["p_uncond"]] = 0

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow_loss(x0, cond) / ACCUM

            loss.backward()

    torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])

    opt.step()
    opt.zero_grad()
```

### Results

We trained two adapters: one for the sparse-structure DiT and one for the shape DiT. The LoRA config files in the container image contain the full training settings: [`ss_flow_lora_gso.json`](https://github.com/suvadityamuk/trellis2-finetuning/blob/master/container/trellis2/configs/lora/slat_flow_img2shape_lora_gso.json) for the sparse-structure flow DiT and [`slat_flow_img2shape_lora_gso.json`](https://github.com/suvadityamuk/trellis2-finetuning/blob/master/container/trellis2/configs/lora/slat_flow_img2shape_lora_gso.json) for the shape flow DiT. Each run used one A100 80 GB on Hugging Face Jobs. The sparse-structure run took 7.2 hours, and the shape run took 3.6 hours.

The training loss of the sparse-structure flow DiT falls quickly at the start, then more slowly. The loss of the shape flow DiT stays at approximately the same level. These curves show how the training objective changes, but they do not directly measure the quality of the generated geometry.  
![Loss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/training_loss.png)

Flow-matching loss varies with the sampled objects, noise, and timesteps. To find out if fine-tuning improves the geometry, we must also compare outputs on the held-out objects.

The figure below compares outputs for five of the nine held-out objects. It shows the input image, the base model’s generated output, the base output after a rotation adjustment, the output with the footwear adapters, and the reference scan. In these examples, the soles from the fine-tuned model are closer to the scans. 

![Heldout Meshes](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/heldout_meshes.png)

We also use three metrics to evaluate the geometry on the test set:

| Metric | What it means |
| ----- | ----- |
| Chamfer distance | The distance from points on one surface to the nearest points on the other surface, measured in both directions. Lower is better. |
| F-score @ 0.02 | The combination of precision (how much of the generated surface is near the reference) and recall (how much of the reference surface the generated surface covers). Here, the matching threshold is 2% of the scan’s bounding-box diagonal. Higher is better. |
| Normal consistency | How closely the surface directions agree. A surface normal is a vector perpendicular to the surface. Higher is better. |

### ![Heldout Metrics](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/heldout_metrics.png)

In the chart’s nine-object comparison, the base model has a mean F-score of 0.276, compared with 0.803 for the model with the footwear adapters. Chamfer distance falls from 0.174 to 0.033, and normal consistency rises from 0.530 to 0.804. With both adapters, the F-score is better on all nine held-out objects.   
![Object Metrics](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/per_object_fscore.png)

For the statistics of all runs, see this [Trackio space](https://suvadityamuk-trellis2-geoflow-lora-trackio.hf.space/media?project=trellis2-geoflow-lora&runs=blog-results).

## Future directions

Phew, so there we go! We fine-tuned TRELLIS.2 for footwear generation by adding LoRA adapters to the two model components that control geometry. We prepared the data once, trained the adapters in separate jobs, and evaluated the outputs on held-out objects.

The main lesson is to select a training target that agrees with the problem and the data that you have. For this experiment, we fine-tuned the structure and shape stages and froze the texture stage.

Next, we can examine if the same choices help other 3D generation models, such as [Pixal3D](https://github.com/TencentARC/Pixal3D) and [UniLat3D](https://github.com/UniLat3D).

Gaussian-splat and world-generation models, such as [HunyuanWorld 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) and [NVIDIA Lyra 2.0](https://nv-tlabs.github.io/Project-Lyra/), are also possible next steps. Their representations and training pipelines can need different methods.

Thank you for following along, and see you in the next one! 👋🤗