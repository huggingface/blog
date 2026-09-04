---
title: "One Canvas, Eleven Pipelines: an AUTOMATIC1111-Shaped Studio in gr.Workflow"
thumbnail: /blog/assets/gradio-workflow1111/thumbnail.png
authors:
- user: ysharma
---

# One canvas, eleven pipelines

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/workflow1111-sample2.mp4"></video>

In the [last post](https://huggingface.co/blog/gradio-workflow-guide), we built five small `gr.Workflow` graphs and ended with a bigger idea: you could use the same approach to build something as complex as [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui). So, that’s exactly what we did.

Meet **Workflow1111**. One canvas with **73 nodes, 11 pipelines, and 20 outputs.** It brings together txt2img with real controls, hires fix, img2img, prompt-matrix grids, VLM interrogate, detection-to-inpaint masks, ControlNet-style annotators, background removal, PNG Info storing, and image-to-video.

Visitors can run the pipelines using their Hugging Face account or access token. Sign in, and the model calls use your own quota.

👉 **[Try Workflow1111](https://huggingface.co/spaces/ysharma/Workflow1111)**, or duplicate the Space and start rewiring it for your own use case.

Let's walk the canvas.

## What's on the canvas

| # | Pipeline | What it does |
|---|---|---|
| 1 | txt2img | prompt builder → sampler settings → [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) → post-processing |
| 2 | Hires fix | that result re-rendered through [FLUX.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) |
| 3 | img2img | upload an image, edit it by instruction |
| 4 | Prompt magic | an LLM writes the prompt from a rough idea |
| 5 | Interrogate | image → VLM → prompt, plus a classifier |
| 6 | Detect & mask | [DETR](https://huggingface.co/facebook/detr-resnet-50) boxes → annotated preview → feathered inpaint mask |
| 7 | Prompt matrix | four variants rendered at once into a contact sheet |
| 8 | Extras | local upscale · [AuraSR ×4](https://huggingface.co/spaces/gokaygokay/AuraSR-v2) · [background removal](https://huggingface.co/spaces/briaai/BRIA-RMBG-2.0) |
| 9 | Annotators | Canny · line art · sketch · luma-depth · posterize |
| 10 | PNG Info | read generation parameters back out of a file |
| 11 | img2video | that same PNG animated into a three-second clip |

All of these pipelines are built using the same four operator kinds covered in the [guide](https://gradio.app/guides/workflows#operator-kinds): `fn` for a Python function, `model` for a model called through `InferenceClient`, `space` for another Gradio Space, and `dataset` for a row from a Hub dataset.



## Simple text-to-image, with all the knobs

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/txt2img.mp4"></video>

This workflow comes with all the usual controls: `negative prompt`, `steps`, `CFG`, `seed`, `width` and `height`, plus a `model_id` field for selecting the checkpoint. A prompt builder node polishes the input, while post-processing takes care of the output.

## Hi-resolution fix, and then image-to-image

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/hires-fix.mp4"></video>

Hi-res fix is a small pipeline. The text-to-image result goes into [FLUX.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) with a refine instruction, '*enhance fine detail and micro-texture, keep the composition identical*', and comes back sharper and larger.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/img2img.mp4"></video>

In this workflow, the same image-to-image node works as an image editor. Just upload an image, describe the change you want, and get the edited image back.

## Let an LLM write the prompt

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/prompt-magic.mp4"></video>

The prompt "*A lighthouse in a storm*" isn’t polished enough, it’s just an idea. Pass it to a [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) node, turn the output into a clean list of tags, and now you have something an image model can work very well with: "*stormy sea, wet rocks, dramatic composition, low angle shot, volumetric lighting, ominous tone.*"

Then pass that straight to your diffusion model and render the image.

The nice part is that you don’t need a custom node (unlike a typical ComfyUI workflow) to put an LLM and a diffusion model on the same canvas. In a Gadio workflow, they’re both just operator nodes.


## Read an image back into a prompt

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/interrogate.mp4"></video>

Think of it as Automatic1111’s CLIP Interrogate, but with a VLM doing the interrogating. [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) looks at a night-market photo and generates a prompt that could have created it. At the same time, another [ViT](https://huggingface.co/google/vit-base-patch16-224) node can classify your image: *restaurant 51.9%, tobacconist 15.6%, toyshop 9.1%.*

One image goes in, two models look at it, and you get two different answers, at great speed due to parallel execution of `gr.Workflow`.

## Detect, then mask

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/detect-and-mask.mp4"></video>

[DETR](https://huggingface.co/facebook/detr-resnet-50) detects six objects in the image: three people, a dog, a bicycle, and a car. From there, the workflow splits into two branches. One draws the detected boxes on the original image, while the other turns them into an inpaint mask that you can further use in your downstream pipelines.

All the drawing and mask creation is happening locally using Pillow and NumPy. Only the object detection is sent out for processing.


## Four prompts at once

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/prompt-matrix.mp4"></video>

One tree, four skies: *at sunrise · in a thunderstorm · under the Milky Way · in autumn fog.*

We start with one base prompt, split it into four variations, generate all four images, and bring them together in a single contact sheet.

There’s no loop operator here, so we simply place four txt2img nodes side by side. Since in `gr.Workflow` [operators at the same dependency depth run in parallel](https://gradio.app/guides/workflows), all four images start generating at the same time.


## Upscale your image, then remove background

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/extras-upscale.mp4"></video>

Two Upscaler nodes, both using a different approach. 
- First, a local [Lanczos](https://stackoverflow.com/questions/1854146/what-is-the-idea-behind-scaling-an-image-using-lanczos) resample using `fn` operator node. It runs instantly with no network calls. 
- Then there’s [AuraSR ×4](https://huggingface.co/spaces/gokaygokay/AuraSR-v2), which uses a `space` operator node. It lets us bring someone else’s Space directly into our workflow.


<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/extras-background.mp4"></video>

And finally, [BRIA RMBG-2.0](https://huggingface.co/spaces/briaai/BRIA-RMBG-2.0) removes the background, also as a space node.

## Annotators, entirely local

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/annotators.mp4"></video>

Canny, line art, sketch, luma-depth, and posterize all run using just NumPy. On a 1024×1024 image of a building facade, each one takes about half a second on CPU. No model needed.

In `gr.Workflow`, a node is just a function. It reads the function signature and automatically turns the inputs and outputs into ports.

Of the 36 operators in the app, 32 are `fn` nodes, and 22 of those run entirely in-process. This means about two-thirds of the app keeps working even if you lose your internet connection.

Testing is also straightforward. Since these operators are regular Python functions, you can test them directly without needing a canvas, server, or GPU.

## PNG info extract and the image-to-video pipeline

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/png-info.mp4"></video>

This pipeline extracts the metadata written into a generated image by one of the postprocess nodes. It works similarly to Automatic1111, which stores generation details in the PNG’s `parameters` text chunk. The pipeline reads that data back out, including the prompt, negative prompt, steps, sampler, CFG, seed, image size, and model.


<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/img2video.mp4"></video>

Then the sleeping fox starts moving. [Wan 2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) brings it to life using the same image node that PNG Info is already reading from. No need for another upload box. One reference node can feed as many downstream pipelines as you need, so a single upload can be read for its metadata and animated, all on the same canvas.


## Your own model, on your own GPU

So far, every node has used someone else’s hardware, either through Inference Providers or Spaces. That means you can build and run a Workflow1111 app with `gr.Workflow` without needing a GPU of your own.

But a `fn` node is just Python. It can just as easily load a model locally and run it on your own GPU.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/img2video-zerogpu.mp4"></video>

A good example is [**FastVideo/fastvideo-fasth3-preview**](https://huggingface.co/spaces/FastVideo/fastvideo-fasth3-preview). It’s a `gr.Workflow` app running [FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree), a four-step distillation of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), that generates video with a soundtrack using ZeroGPU.

The whole app comes down to one bound function:

```python
@spaces.GPU(duration=get_duration, size=GPU_SIZE)
def _generate(prompt_embeds, text_token_tags, height, width, num_frames, seed):
    ...

gr.Workflow(bind={"generate": generate, "status": status}).launch()
```

[ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) gives the function a GPU when it needs one, then releases it when the call is done. `gr.Workflow` doesn’t need to know about any of that. It just calls the function.

And this isn’t specific to Spaces. Point `bind=` to a function that loads a local checkpoint, run `.launch()` on your own machine, and the Workflow1111 canvas can drive your own GPU.

## Every output is an API

Nine endpoints, no routes written: `/image`, `/edited_image`, `/generated_prompt`, `/recovered_prompt`, `/detected_objects`, `/x_y_grid`, `/upscaled_local`, `/annotator_map`, `/png_info`.

```python
from gradio_client import Client

client = Client("ysharma/Workflow1111", oauth_token="hf_...")

image, params, hires = client.predict(
    "a red fox in a snowy pine forest",  # Prompt
    "",                                  # Negative prompt
    "Cinematic",                         # Style preset
    "enhance fine detail",               # Hires refine instruction
    api_name="/image",
)
```


## So, is this a ComfyUI replacement?

For a lot of the things people actually want to build and ship, yes.

* **A node can be hardware you don’t own.** It can run through [Inference Providers](https://huggingface.co/docs/inference-providers/index), call any Space on the Hub or any API, or even pull from a dataset. That’s how this Workflow1111 studio can run without its own GPU.
* **Every output becomes a typed REST endpoint.** The endpoints are generated directly from your graph.
* **Visitors can run workflows under their own identity.** Turn on [OAuth](https://huggingface.co/docs/hub/spaces-oauth), share the public URL, and anyone can sign in and use the app. No installs needed.
* **Mix models and modalities on the same canvas.** Diffusion models, LLMs, VLMs, detectors, and video models can all be part of the same workflow.
* **Need something custom? Just write a function.** A custom node is simply a function, so you can use it for pretty much anything.

The result is a multi-model pipeline that people can open in a browser, sign into, use right away, and call from code.


## Build your own

Workflow1111 has 73 nodes, but it started with just this:

```python
import gradio as gr

def your_function(text: str) -> str:
    ...

gr.Workflow(bind=[your_function]).launch()
```

Start with `gr.Workflow()` and you get an empty canvas to build on right in your browser. Add `bind=` to turn your functions into nodes, use `edges=` to connect them, and run `gradio deploy` when you're ready to put the whole thing on a Space.

You can find all the details in the [gr.Workflow guide](https://gradio.app/guides/workflows), including the JSON schema and all the available operator types.

Or, if you want to jump straight in, [open Workflow1111](https://huggingface.co/spaces/ysharma/Workflow1111), hit **Duplicate**, pick one of the eleven pipelines, and start changing things. Delete nodes, swap them out, rewire the flow, and make it your own.

If you want to start smaller, the [previous post](https://huggingface.co/blog/gradio-workflow-guide) has five simpler workflows you can get running in about a minute each.

And if you build something on the canvas, share it. I want to see who gets past 73 nodes 👀

