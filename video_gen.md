---
title: "State of open video generation models in Diffusers"
thumbnail: /blog/assets/video_gen/thumbnail.png
authors:
- user: sayakpaul
- user: a-r-r-o-w
- user: dn6
---

# State of open video generation models in Diffusers

OpenAI’s Sora demo marked a striking advance in AI-generated video last year and gave us a glimpse of the potential capabilities of video generation models. The impact was immediate and since that demo, the video generation space has become increasingly competitive with major players and startups producing their own highly capable models such as Google’s Veo2, Haliluo’s Minimax, Runway’s Gen3, Alpha Kling, Pika, and Luma Lab’s Dream Machine.

Open-source has also had its own surge of video generation models with CogVideoX, Mochi-1, Hunyuan, Allegro, and LTX Video. Is the video community having its “Stable Diffusion moment”?

This post will provide a brief overview of the state of video generation models, where we are with respect to open video generation models, and how the Diffusers team is planning to support their adoption at scale.

Specifically, we will discuss:

- Capabilities and limitations of video generation models
- Why video generation is hard
- Open video generation models
- Video generation with Diffusers
    - Inference and optimizations
    - Fine-tuning
- Looking ahead

## Today’s Video Generation Models and their Limitations

These are today's most popular video models for AI-generated content creation

| **Provider** | **Model**         | **Open/Closed** | **License** |
|:--------------:|:-------------------:|:-----------------:|:-------------:|
| Meta         | [MovieGen](https://ai.meta.com/research/movie-gen/) | Closed (with a detailed [technical report](https://ai.meta.com/research/publications/movie-gen-a-cast-of-media-foundation-models/)) | Proprietary |
| OpenAI       | [Sora](https://sora.com/) | Closed         | Proprietary |
| Google       | [Veo 2](https://deepmind.google/technologies/veo/veo-2/) | Closed         | Proprietary |
| RunwayML     | [Gen 3 Alpha](https://runwayml.com/research/introducing-gen-3-alpha) | Closed         | Proprietary |
| Pika Labs    | [Pika 2.0](https://pika.art/login) | Closed         | Proprietary |
| KlingAI      | [Kling](https://www.klingai.com/) | Closed         | Proprietary |
| Haliluo      | [MiniMax](https://hailuoai.video/) | Closed         | Proprietary |
| THUDM        | [CogVideoX](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox) | Open           | [Custom](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) |
| Genmo        | [Mochi-1](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox) | Open           | Apache 2.0 |
| RhymesAI     | [Allegro](https://huggingface.co/docs/diffusers/main/en/api/pipelines/allegro) | Open           | Apache 2.0 |
| Lightricks   | [LTX Video](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video) | Open           | [Custom](https://huggingface.co/Lightricks/LTX-Video/blob/main/License.txt) |
| Tencent      | [Hunyuan Video](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video) | Open           | [Custom](https://huggingface.co/tencent/HunyuanVideo/blob/main/LICENSE) |

**Limitations**:

- **High Resource Requirements:** Producing high-quality videos requires large pretrained models, which are computationally expensive to develop and deploy. These costs arise from dataset collection, hardware requirements, extensive training iterations and experimentation. These costs make it hard to justify producing open-source and freely available models. Even though we don’t have a detailed technical report that shed light into the training resources used, [this post](https://www.factorialfunds.com/blog/under-the-hood-how-openai-s-sora-model-works) provides some reasonable estimates.
- **Generalization**: Several open models suffer from limited generalization capabilities and underperform expectations of users. Models may require prompting in a certain way, or LLM-like prompts, or fail to generalize to out-of-distribution data, which are hurdles for widespread user adoption. For example, models like LTX-Video often need to be prompted in a very detailed and specific way for obtaining good quality generations.
- **Latency**: The high computational and memory demands of video generation result in significant generation latency. For local usage, this is often a roadblock. Most new open video models are inaccessible to community hardware without extensive memory optimizations and quantization approaches that affect both inference latency and quality of the generated videos.

## Why is Video Generation Hard?

There are several factors we’d like to see and control in videos:

- Adherence to Input Conditions (such as a text prompt, a starting image, etc.)
- Realism
- Aesthetics
- Motion Dynamics
- Spatio-Temporal Consistency and Coherence
- FPS
- Duration

With image generation models, we usually only care about the first three aspects. However, for video generation we now have to consider motion quality, coherence and consistency over time, potentially with multiple subjects. Finding the right balance between good data, right inductive priors, and training methodologies to suit these additional requirements has proved to be more challenging than other modalities.

## Open Video Generation Models

![diagram](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/video_gen/diagram.jpg)

Text-to-video generation models have similar components as their text-to-image counterparts:

- Text encoders for providing rich representations of the input text prompt
- A denoising network
- An encoder and decoder to convert between pixel and latent space
- A non-parametric scheduler responsible for managing all the timestep-related calculations and the denoising step

The latest generation of video models share a core feature where the denoising network processes 3D video tokens that capture both spatial and temporal information. The video encoder-decoder system, responsible for producing and decoding these tokens, employs both spatial and temporal compression. While decoding the latents typically demands the most memory, these models offer frame-by-frame decoding options to reduce memory usage.

Text conditioning is incorporated through either joint attention (introduced in [Stable Diffusion 3](https://arxiv.org/abs/2403.03206)) or cross-attention. T5 has emerged as the preferred text encoder across most models, with HunYuan being an exception in its use of both CLIP-L and LLaMa 3.

The denoising network itself builds on the DiT architecture developed by [William Peebles and Saining Xie](https://arxiv.org/abs/2212.09748), while incorporating various design elements from [PixArt](https://arxiv.org/abs/2310.00426).


## Video Generation with Diffusers

<figure class="image flex flex-col items-center text-center m-0 w-full">
   <video
      alt="demo4.mp4"
      autoplay loop autobuffer muted playsinline
    >
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/video_gen/hunyuan-output.mp4" type="video/mp4">
  </video>
</figure>

There are three broad categories of generation possible when working with video models:

1. Text to Video
2. Image or Image Control condition + Text to Video
3. Video or Video Control condition + Text to Video

Going from a text (and other conditions) to a video is just a few lines of code. Below we show how to do text-to-video generation with the [LTX-Video model from Lighricks](https://huggingface.co/Lightricks/LTX-Video).

```py
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to("cuda")

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_frames=161,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

### Memory requirements

The memory requirements for any model can be computed by adding the following:

- Memory required for weights
- Maximum memory required for storing intermediate activation states

Memory required by weights can be lowered via - quantization, downcasting to lower dtypes, or by offloading to CPU. Memory required for activations states can also be lowered but that is a more involved process, which is out of the scope of this blog.

It is possible to run any video model with extremely low memory, but it comes at the cost of time required for inference. If the time required by an optimization technique is more than what a user considers reasonable, it is not feasible to run inference. Diffusers provides many such optimizations that are opt-in and can be chained together.

In the table below, we provide the memory requirements for three popular video generation models with reasonable defaults:

| **Model Name** | **Memory (GB)** |
|:---:|:---:|
| HunyuanVideo | 60.09 |
| CogVideoX (1.5 5B) | 36.51 |
| LTX-Video | 17.75 |

These numbers were obtained with the following settings on an 80GB A100 machine (full script [here](https://gist.github.com/sayakpaul/2bc49a30cf76cea07914104d28b1fb86)):

- `torch.bfloat16` dtype
- `num_frames`: 121, `height`: 512, `width`: 768
- `max_sequence_length`: 128
- `num_inference_steps`: 50

These requirements are quite staggering, and make these models difficult to run on consumer hardware. With Diffusers, users can opt-in to different optimizations to reduce memory usage.
The following table provides the memory requirements for HunyuanVideo with various optimizations enabled that make minimal compromises on quality and time required for inference.

We used HunyuanVideo for this study, as it is sufficiently large enough, to show the benefits of the optimizations in a progressive manner.

| **Setting**                                        | **Memory**    | **Time** |
|:--------------------------------------------------:|:-------------:|:--------:|
| BF16 Base                                          | 60.10 GB      |  863s    |
| BF16 + CPU offloading                              | 28.87 GB      |  917s    |
| BF16 + VAE tiling                                  | 43.58 GB      |  870s    |
| 8-bit BnB                                          | 49.90 GB      |  983s    |
| 8-bit BnB + CPU offloading*                        | 35.66 GB      | 1041s    |
| 8-bit BnB + VAE tiling                             | 36.92 GB      |  997s    |
| 8-bit BnB + CPU offloading + VAE tiling            | 26.18 GB      | 1260s    |
| 4-bit BnB                                          | 42.96 GB      |  867s    |
| 4-bit BnB + CPU offloading                         | 21.99 GB      |  953s    |
| 4-bit BnB + VAE tiling                             | 26.42 GB      |  889s    |
| 4-bit BnB + CPU offloading + VAE tiling            | 14.15 GB      |  995s    |
| FP8 Upcasting                                      | 51.70 GB      |  856s    |
| FP8 Upcasting + CPU offloading                     | 21.99 GB      |  983s    |
| FP8 Upcasting + VAE tiling                         | 35.17 GB      |  867s    |
| FP8 Upcasting + CPU offloading + VAE tiling        | 20.44 GB      | 1013s    |
| BF16 + Group offload (blocks=8) + VAE tiling       | 15.67 GB      |  925s    |
| BF16 + Group offload (blocks=1) + VAE tiling       |  7.72 GB      |  881s    |
| BF16 + Group offload (leaf) + VAE tiling           |  6.66 GB      |  887s    |
| FP8 Upcasting + Group offload (leaf) + VAE tiling  |  6.56 GB^     |  885s    |

<sub><sup>*</sup>8Bit models in `bitsandbytes` cannot be moved to CPU from GPU, unlike the 4Bit ones.<sub>
<sub><sup>^</sup>The memory usage does not reduce further because the peak utilizations comes from computing attention and feed-forward. Using [Flash Attention](https://github.com/Dao-AILab/flash-attention) and [Optimized Feed-Forward](https://github.com/huggingface/diffusers/pull/10623) can help lower this requirement to ~5 GB.<sub>

We used the same settings as above to obtain these numbers. Also note that due to numerical precision loss, quantization can impact the quality of the outputs, effects of which are more prominent in videos than images.

We provide more details about these optimizations in the sections below along with some code snippets to go. But if you're already feeling excited,
we encourage you to check out [our guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/text-img2vid).

### Suite of optimizations

Video generation can be quite difficult on resource-constrained devices and time-consuming even on beefier GPUs. Diffusers provides a suite of utilities that help to optimize both the runtime and memory consumption of these models. These optimizations fall under the following categories:

- **Quantization**: The model weights are quantized to lower precision data types, which lowers the VRAM requirements of models. Diffusers supports three different quantization backends as of today: [bitsandbytes](https://huggingface.co/docs/diffusers/main/en/quantization/bitsandbytes), [torchao](https://huggingface.co/docs/diffusers/main/en/quantization/torchao), and [GGUF](https://huggingface.co/docs/diffusers/main/en/quantization/gguf).
- **Offloading**: Different layers of a model can be loaded on the GPU when required for computation on-the-fly and then offloaded back to CPU. This saves a significant amount of memory during inference. Offloading is supported through `enable_model_cpu_offload()` and `enable_sequential_cpu_offload()`. Refer [here](https://huggingface.co/docs/diffusers/main/en/optimization/memory#model-offloading) for more details.
- **Chunked Inference**: By splitting inference across non-embedding dimensions of input latent tensors, the memory overheads from intermediate activation states can be reduced. Common use of this technique is often seen in encoder/decoder slicing/tiling. Chunked inference in Diffusers is supported through [feed-forward chunking](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video#tips), decoder tiling and slicing, and [split attention inference](https://huggingface.co/docs/diffusers/main/en/api/pipelines/animatediff#freenoise-memory-savings).
- **Re-use of Attention & MLP states**: Computation of certain denoising steps can be skipped and past states can be re-used, if certain conditions are satisfied for particular algorithms, to speed up the generation process with minimal quality loss.

Below, we provide a list of some advanced optimization techniques that are currently work-in-progress and will be merged soon:

* [Layerwise Casting](https://github.com/huggingface/diffusers/pull/10347): Lets users store the parameters in lower-precision, such as `torch.float8_e4m3fn`, and run computation in a higher precision, such as `torch.bfloat16`.
* [Group offloading](https://github.com/huggingface/diffusers/pull/10503): Lets users group internal block-level or leaf-level modules to perform offloading. This is beneficial because only parts of the model required for computation are loaded onto the GPU. Additionally, we provide support for overlapping data transfer with computation using CUDA streams, which reduce most of the additional overhead that comes from multiple onloading/offloading of layers.

Below is an example of applying 4bit quantization, vae tiling, cpu offloading, and layerwise casting to HunyuanVideo to reduce the required VRAM to just ~6.5 GB for `121 x 512 x 768` resolution videos. To the best of our knowledge, this is the lowest memory requirement to run HunyuanVideo among all available implementations without compromising speed.

Install Diffusers from source to try out these features! Some implementations are agnostic to the model being used, and can be applied in other backends easily - be sure to check it out!

```shell
pip install git+https://github.com/huggingface/diffusers.git
```

```python
import torch
from diffusers import (
    BitsAndBytesConfig,
    HunyuanVideoTransformer3DModel,
    HunyuanVideoPipeline,
)
from diffusers.utils import export_to_video
from diffusers.hooks import apply_layerwise_casting
from transformers import LlamaModel

model_id = "hunyuanvideo-community/HunyuanVideo"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
)

text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
apply_layerwise_casting(text_encoder, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16)

# Apply 4-bit bitsandbytes quantization to Hunyuan DiT model
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)

pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.float16
)

# Enable memory saving
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
```

We can also apply optimizations during training. The two most well-known techniques applied to video models include:

- **Timestep distillation**: This involves teaching the model to denoise the noisy latents faster in lesser amount of inference steps, in a recursive fashion. For example, if a model takes 32 steps to generate good videos, it can be augmented to try and predict the final outputs in only 16-steps, or 8-steps, or even 2-steps! This may be accompanied by loss in quality depending on how fewer steps are used. Some examples of timestep-distilled models include [Flux.1-Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell/) and [FastHunyuan](https://huggingface.co/FastVideo/FastHunyuan).
- **Guidance distillation**: [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) is a technique widely used in diffusion models that enhances generation quality. This, however, doubles the generation time because it involves two full forward passes through the models per inference step, followed by an interpolation step. By teaching models to predict the output of both forward passes and interpolation at the cost of one forward pass, this method can enable much faster generation. Some examples of guidance-distilled models include [HunyuanVideo](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video) and [Flux.1-Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).

We refer the readers to [this guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/text-img2vid) for a detailed take on video generation and the current possibilities in Diffusers.

### Fine-tuning

We’ve created [`finetrainers`](https://github.com/a-r-r-o-w/finetrainers) - a repository that allows you to easily fine-tune the latest generation of open video models. For example, here is how you would fine-tune CogVideoX with LoRA:

```bash
# Download a dataset
huggingface-cli download \
  --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset \
  --local-dir video-dataset-disney

# Then launch training
accelerate launch train.py \
  --model_name="cogvideox" --pretrained_model_name_or_path="THUDM/CogVideoX1.5-5B" \
  --data_root="video-dataset-disney" \
  --video_column="videos.txt" \
  --caption_column="prompt.txt" \
  --training_type="lora" \
  --seed=42 \
  --mixed_precision="bf16" \
  --batch_size=1 \
  --train_steps=1200 \
  --rank=128 \
  --lora_alpha=128 \
  --target_modules to_q to_k to_v to_out.0 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --checkpointing_steps 500 \
  --checkpointing_limit 2 \
  --enable_slicing \
  --enable_tiling \
  --optimizer adamw \
  --lr 3e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0

# ...
# (Full training command removed for brevity)
```

We used `finetrainers` to emulate the "dissolve" effect and obtained promising results. Check out [the model](https://huggingface.co/sayakpaul/pika-dissolve-v0) for additional details.

<figure class="image flex flex-col items-center text-center m-0 w-full">
   <video
      alt="demo4.mp4"
      autoplay loop autobuffer muted playsinline
    >
    <source src="https://huggingface.co/sayakpaul/pika-dissolve-v0/resolve/main/assets/output_vase.mp4" type="video/mp4">
  </video>
  <figcaption>Prompt: <i>PIKA_DISSOLVE A slender glass vase, brimming with tiny white pebbles, stands centered on a polished ebony dais. Without warning, the glass begins to dissolve from the edges inward. Wisps of translucent dust swirl upward in an elegant spiral, illuminating each pebble as they drop onto the dais. The gently drifting dust eventually settles, leaving only the scattered stones and faint traces of shimmering powder on the stage.</i></figcaption>
</figure>

## Looking ahead

We anticipate significant advancements in video generation models throughout 2025, with major improvements in both output quality and model capabilities.
Our goal is to make using these models easy and accessible. We will continue to grow the `finetrainers` library, and we are planning on adding many more featueres: Control LoRAs, Distillation Algorithms, ControlNets, Adapters, and more. As always, community contributions are welcome 🤗

Our commitment remains strong to partnering with model publishers, researchers, and community members to ensure the latest innovations in video generation are within reach to everyone.

## Resources

We cited a number of links throughout the post. To make sure you don’t miss out on the most important ones, we provide a list below:

- [Video generation guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/text-img2vid)
- [Quantization support in Diffusers](https://huggingface.co/docs/diffusers/main/en/quantization/overview)
- [General LoRA guide in Diffusers](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
- [Memory optimization guide for CogVideoX](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox#memory-optimization) (it applies to other video models, too)
- [`finetrainers`](https://github.com/a-r-r-o-w/finetrainers) for fine-tuning

*Acknowledgements: Thanks to [Chunte](https://huggingface.co/Chunte) for creating the beautiful thumbnail for this post. Thanks to [Vaibhav](https://huggingface.co/reach-vb) and [Pedro](https://huggingface.co/pcuenq) for their helpful feedback.*