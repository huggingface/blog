---
title: Using Stable Diffusion with Core ML on Apple Silicon
thumbnail: /blog/assets/diffusers_coreml/thumbnail.png
authors:
- user: pcuenq
---

# Using Stable Diffusion with Core ML on Apple Silicon

<!-- {blog_metadata} -->
<!-- {authors} -->

Thanks to Apple engineers, you can now run Stable Diffusion on Apple Silicon using Core ML!

[This Apple repo](https://github.com/apple/ml-stable-diffusion) provides conversion scripts and inference code based on [🧨 Diffusers](https://github.com/huggingface/diffusers), and we love it! To make it as easy as possible for you, we converted the weights ourselves and put the Core ML versions of the models in [the Hugging Face Hub](https://hf.co/apple).

This post guides you on how to use the converted weights.

## Available Checkpoints

The checkpoints that are already converted and ready for use are the ones for these models:

- Stable Diffusion v1.4: [converted](https://hf.co/apple/coreml-stable-diffusion-v1-4) [original](https://hf.co/CompVis/stable-diffusion-v1-4)
- Stable Diffusion v1.5: [converted](https://hf.co/apple/coreml-stable-diffusion-v1-5) [original](https://hf.co/runwayml/stable-diffusion-v1-5)
- Stable Diffusion v2 base: [converted](https://hf.co/apple/coreml-stable-diffusion-2-base) [original](https://huggingface.co/stabilityai/stable-diffusion-2-base)

Core ML supports all the compute units available in your device: CPU, GPU and Apple's Neural Engine (NE). It's also possible for Core ML to run different portions of the model in different devices to maximize performance.

There are several variants of each model that may yield different performance depending on the hardware you use. We recommend you try them out and stick with the one that works best in your system. Read on for details.

## Notes on Performance

There are several variants per model:

- "Original" attention vs "split_einsum". These are two alternative implementations of the critical attention blocks. `split_einsum` was [previously introduced by Apple](https://machinelearning.apple.com/research/neural-engine-transformers), and is compatible with all the compute units (CPU, GPU and Apple's Neural Engine). `original`, on the other hand, is only compatible with CPU and GPU. Nevertheless, `original` can be faster than `split_einsum` on some devices, so do check it out!
- "ML Packages" vs "Compiled" models. The former is suitable for Python inference, while the `compiled` version is required for Swift code. The `compiled` models in the Hub split the large UNet model weights in several files for compatibility with iOS and iPadOS devices. This corresponds to the [`--chunk-unet` conversion option](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml).

At the time of this writing, we got best results on my MacBook Pro (M1 Max, 32 GPU cores, 64 GB) using the following combination:

- `original` attention.
- `all` compute units (see next section for details).
- macOS Ventura 13.1 Beta 4 (22C5059b).

With these, it took 18s to generate one image with the Core ML version of Stable Diffusion v1.4 🤯.

> **⚠️ Note**
>
> Several improvements to Core ML have been introduced in the beta version of macOS Ventura 13.1, and they are required by Apple's implementation. You may get black images –and much slower times– if you use the current release version of macOS Ventura (13.0.1). If you can't or won't install the beta, please wait until macOS Ventura 13.1 is officially released.


Each model repo is organized in a tree structure that provides these different variants:

```
coreml-stable-diffusion-v1-4
├── README.md
├── original
│   ├── compiled
│   └── packages
└── split_einsum
    ├── compiled
    └── packages
```

You can download and use the variant you need as shown below.

## Core ML Inference in Python

### Prerequisites

```bash
pip install huggingface_hub
pip install git+https://github.com/apple/ml-stable-diffusion
```

### Download the Model Checkpoints

To run inference in Python, you have to use one of the versions stored in the `packages` folders, because the compiled ones are only compatible with Swift. You may choose whether you want to use the `original` or `split_einsum` attention styles.

This is how you'd download the `original` attention variant from the Hub:

```Python
from huggingface_hub import snapshot_download
from huggingface_hub.file_download import repo_folder_name
from pathlib import Path
import shutil

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/packages"

def download_model(repo_id, variant, output_dir):
    destination = Path(output_dir) / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
    if destination.exists():
        raise Exception(f"Model already exists at {destination}")
    
    # Download and copy without symlinks
    downloaded = snapshot_download(repo_id, allow_patterns=f"{variant}/*", cache_dir=output_dir)
    downloaded_bundle = Path(downloaded) / variant
    shutil.copytree(downloaded_bundle, destination)

    # Remove all downloaded files
    cache_folder = Path(output_dir) / repo_folder_name(repo_id=repo_id, repo_type="model")
    shutil.rmtree(cache_folder)
    return destination

model_path = download_model(repo_id, variant, output_dir="./models")
print(f"Model downloaded at {model_path}")
```

The code above will place the downloaded model snapshot inside the directory you specify (`models`, in this case).

### Inference

Once you have downloaded a snapshot of the model, the easiest way to run inference would be to use Apple's Python script.

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i models/coreml-stable-diffusion-v1-4_original_packages -o </path/to/output/image> --compute-unit ALL --seed 93
```

`<output-mlpackages-directory>` should point to the checkpoint you downloaded in the step above, and `--compute-unit` indicates the hardware you want to allow for inference. It must be one of the following options: `ALL`, `CPU_AND_GPU`, `CPU_ONLY`, `CPU_AND_NE`. You may also provide an optional output path, and a seed for reproducibility.

The inference script assumes the original version of the Stable Diffusion model, stored in the Hub as `CompVis/stable-diffusion-v1-4`. If you use another model, you _have_ to specify its Hub id in the inference command-line, using the `--model-version` option. This works both for models already supported, and for custom models you trained or fine-tuned yourself.

For Stable Diffusion 1.5 (Hub id: `runwayml/stable-diffusion-v1-5`):

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" --compute-unit ALL -o output --seed 93 -i models/coreml-stable-diffusion-v1-5_original_packages --model-version runwayml/stable-diffusion-v1-5
```

For Stable Diffusion 2 base (Hub id: `stabilityai/stable-diffusion-2-base`):

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" --compute-unit ALL -o output --seed 93 -i models/coreml-stable-diffusion-2-base_original_packages --model-version stabilityai/stable-diffusion-2-base
```

## Core ML inference in Swift

Running inference in Swift is slightly faster than in Python, because the models are already compiled in the `mlmodelc` format. This will be noticeable on app startup when the model is loaded, but shouldn’t be noticeable if you run several generations afterwards.

### Download

To run inference in Swift on your Mac, you need one of the `compiled` checkpoint versions. We recommend you download them locally using Python code similar to the one we showed above, but using one of the `compiled` variants:

```Python
from huggingface_hub import snapshot_download
from huggingface_hub.file_download import repo_folder_name
from pathlib import Path
import shutil

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/compiled"

def download_model(repo_id, variant, output_dir):
    destination = Path(output_dir) / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
    if destination.exists():
        raise Exception(f"Model already exists at {destination}")
    
    # Download and copy without symlinks
    downloaded = snapshot_download(repo_id, allow_patterns=f"{variant}/*", cache_dir=output_dir)
    downloaded_bundle = Path(downloaded) / variant
    shutil.copytree(downloaded_bundle, destination)

    # Remove all downloaded files
    cache_folder = Path(output_dir) / repo_folder_name(repo_id=repo_id, repo_type="model")
    shutil.rmtree(cache_folder)
    return destination

model_path = download_model(repo_id, variant, output_dir="./models")
print(f"Model downloaded at {model_path}")
```

### Inference

To run inference, please clone Apple's repo:

```bash
git clone https://github.com/apple/ml-stable-diffusion
cd ml-stable-diffusion
```

And then use Apple's command-line tool using Swift Package Manager's facilities:

```bash
swift run StableDiffusionSample --resource-path models/coreml-stable-diffusion-v1-4_original_compiled --compute-units all "a photo of an astronaut riding a horse on mars"
```

You have to specify in `--resource-path` one of the checkpoints downloaded in the previous step, so please make sure it contains compiled Core ML bundles with the extension `.mlmodelc`. The `--compute-units` has to be one of these values: `all`, `cpuOnly`, `cpuAndGPU`, `cpuAndNeuralEngine`.

For more details, please refer to the [instructions in Apple's repo](https://github.com/apple/ml-stable-diffusion).

## Bring Your own Model

If you have created your own models compatible with Stable Diffusion (for example, if you used Dreambooth, Textual Inversion or fine-tuning), then you have to convert the models yourself. Fortunately, Apple provides a conversion script that allows you to do so.

For this task, we recommend you follow [these instructions](https://github.com/apple/ml-stable-diffusion#converting-models-to-coreml).

## Next Steps

We are really excited about the opportunities this brings and can't wait to see what the community can create from here. Some potential ideas are:

- Native, high-quality apps for Mac, iPhone and iPad.
- Bring additional schedulers to Swift, for even faster inference.
- Additional pipelines and tasks.
- Explore quantization techniques and further optimizations.

Looking forward to seeing what you create!
