---
title: "Rebuilding AUTOMATIC1111 with Gradio Workflow"
thumbnail: /blog/assets/gradio-workflow1111/thumbnail.png
authors:
- user: ysharma
---

# Rebuilding AUTOMATIC1111 with Gradio Workflow
In our [last post](https://huggingface.co/blog/gradio-workflow-guide), we built five small `gr.Workflow` graphs and hinted at what it would take to build something as complex as AUTOMATIC1111's [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui). In this post we walk you through **Workflow1111**, where we have rebuilt most of AUTOMATIC1111's feature set as a single workflow canvas.

Workflow1111 is a graph of **eleven media pipelines** built using **seventy-three nodes**. It brings together SOTA models for text-to-image, hi-resolution fix, image-to-image, prompt-matrix grids, VLM interrogate, detection-to-inpaint masks, ControlNet-style annotators, background removal, PNG Info storing, and image-to-video.

You can run any of these pipelines by signing in with your Hugging Face account or providing an access token. Once you sign in, the model calls use your own quota.

👉 **[Try Workflow1111](https://huggingface.co/spaces/ysharma/Workflow1111)**, or duplicate the Space and start rewiring it for your own use case.

Let's walk the canvas.

## What's on the canvas

All the media pipelines are built from the same four operator kinds covered in our last post and the [official guide](https://gradio.app/guides/workflows#operator-kinds). Each node on the canvas wraps one operator, and the operator's inputs and outputs become the ports you connect edges to. As a quick reference on our four operator kinds: `fn` is a Python function, `model` is a model called through `InferenceClient`, `space` is another Gradio Space, and `dataset` is a row from a Hub dataset.

Let's go through the pipelines one by one.

### Text-to-image

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/txt2img.mp4"></video>

This is the core pipeline. It has the controls you'd expect from A1111's txt2img tab: negative prompt, steps, CFG, seed, width and height, plus a `model_id` field for choosing the checkpoint. The prompt goes through a prompt-builder `fn` node first, which appends the selected style preset and cleans up the text, then into a `model` node that calls the checkpoint through Inference Providers. A post-process `fn` node writes the generation parameters into the PNG's metadata on the way out, which is what the PNG Info pipeline reads back later.


### Hi-resolution fix

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/hires-fix.mp4"></video>

In Automatic1111, hi-resolution fix first upscales the txt2img output and then runs a second denoising pass. Here it's a two-node detour instead. The text-to-image result goes into a [FLUX.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) `model` node with a refine instruction ("enhance fine detail and micro-texture, keep the composition identical") and comes back sharper and larger.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/img2img.mp4"></video>

### Image-to-image

That same Kontext node doubles as the image-to-image tab. Upload an image, describe the change you want, and it returns the edited image.

### Let an LLM write the prompt

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/prompt-magic.mp4"></video>

### Let an LLM write the prompt

Start with a rough prompt like "A lighthouse in a storm." This pipeline sends it to a [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) `model` node, and a small `fn` node turns the reply into a clean list of tags, capped at forty: "stormy sea, wet rocks, dramatic composition, low angle shot, volumetric lighting, ominous tone." You can connect any diffusion model node to this output to render the image.

There's no custom node involved, unlike in ComfyUI. In a Gradio workflow the LLM and the diffusion model are both ordinary `model` operators on the same canvas.

### Read an image back into a prompt

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/interrogate.mp4"></video>

This is like AUTOMATIC1111's Interrogate button, with a VLM doing the interrogating instead of CLIP. [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) looks at a night-market photo and writes a prompt that could have produced it. A [ViT](https://huggingface.co/google/vit-base-patch16-224) classifier node reads the same image and returns labels: restaurant 51.9%, tobacco shop 15.6%, toyshop 9.1%.

Both nodes use the same image input, so `gr.Workflow` runs them in parallel and you get both answers in roughly the time it takes to run one.

### Detection to inpaint mask

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/detect-and-mask.mp4"></video>

AUTOMATIC1111 makes you paint an inpaint mask by hand. This pipeline generates one from a detector instead. [DETR](https://huggingface.co/facebook/detr-resnet-50) finds six objects in a street photo (three people, a dog, a bicycle, and a car), and from there the workflow splits into two branches: one draws the detected boxes on the original image, the other turns them into a mask you can feed into an inpaint pipeline downstream.

The drawing and the mask creation both happen locally with Pillow and NumPy. Only the detection call leaves the machine.

### Prompt matrix

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/prompt-matrix.mp4"></video>

This is like AUTOMATIC1111's prompt matrix. A base prompt, "a lone oak tree," gets combined with four suffixes (at sunrise, in a thunderstorm, under the Milky Way, in autumn fog) by a `fn` node, and each variant goes to its own text-to-image node. A final node stitches the four results into one contact sheet.

`gr.Workflow` has no loop operator, so the four text-to-image nodes sit side by side on the canvas. Since they're at the same dependency depth they run in parallel, and all four images start generating at once.

### Upscale and background removal

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/extras-upscale.mp4"></video>

This is like the Extras tab in Automatic1111. There are two upscaler nodes, and they take different routes. The first is a local [Lanczos](https://en.wikipedia.org/wiki/Lanczos_resampling) resample in an `fn` node, which needs no network call and finishes as fast as Pillow can resize. The second is [AuraSR ×4](https://huggingface.co/spaces/gokaygokay/AuraSR-v2), and it's the first `space` node on the canvas: it calls a [Space](https://huggingface.co/spaces/gokaygokay/AuraSR-v2) on the Hub and treats the result like any other node output.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/extras-background.mp4"></video>

Background removal works the same way. [BRIA RMBG-2.0](https://huggingface.co/spaces/briaai/BRIA-RMBG-2.0) is another `space` node, so the whole model lives in its own Space and this canvas just calls it in.

### Annotators

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/annotators.mp4"></video>

Canny, line art, sketch, luma-depth, and posterize are the preprocessors you'd normally get from the ControlNet extension in Automatic1111. Here, each one is a `fn` node written in plain NumPy, with no model behind it. On a pre-loaded example photo of a building facade, each annotator takes about half a second on CPU.

There are 36 operator nodes in the app, 32 are `fn` nodes, and 22 of those run entirely in-process without a network call. Roughly two-thirds of the canvas keeps working if you lose your connection. Since these are regular Python functions, you can also test them directly, with no canvas, server, or GPU involved.

### PNG Info

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/png-info.mp4"></video>

AUTOMATIC1111 stores generation details in the PNG's `parameters` text chunk, and the PNG Info tab reads them back. Workflow1111 does the same. The post-process node on the text-to-image pipeline writes the metadata, and this pipeline reads it back out, including the prompt, negative prompt, steps, CFG, seed, image size, and model.

### Image-to-video

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/img2video.mp4"></video>

The image node that PNG Info reads from also feeds a [Wan 2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) node, which animates it; in the demo example a sleeping fox wakes up and starts moving. There's no second upload box because one reference node can feed as many downstream pipelines as you need, so a single upload gets its metadata read and gets animated on the same canvas.


## Running models on your own GPU

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/img2video-zerogpu.mp4"></video>

So far every model call has gone to someone else's hardware, through Inference Providers or a Space. That's why you can build and run something like Workflow1111 without a GPU of your own.

A `fn` node is just Python, though, so it can equally load a model locally and run it on your own GPU. [FastVideo/fastvideo-fasth3-preview](https://huggingface.co/spaces/FastVideo/fastvideo-fasth3-preview) is a `gr.Workflow` app that does exactly that. It runs [FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree), a four-step distillation of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), and generates video with a soundtrack on ZeroGPU.

The whole app comes down to one bound function:

```python
@spaces.GPU(duration=get_duration, size=GPU_SIZE)
def _generate(prompt_embeds, text_token_tags, height, width, num_frames, seed):
    ...

gr.Workflow(bind={"generate": generate, "status": status}).launch()
```

[ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) gives the function a GPU when it needs one, then releases it when the call is done. `gr.Workflow` doesn't need to know about any of that. It just calls the `fn` node.

This isn't specific to Spaces either. Point `bind=` to a function that loads a local checkpoint, run `.launch()` on your own machine, and the Workflow1111 canvas can drive your own GPU.

## Every output is an API

Every output node on the canvas becomes a REST endpoint, with no routes written by hand. Workflow1111 exposes nine of them: `/image`, `/edited_image`, `/generated_prompt`, `/recovered_prompt`, `/detected_objects`, `/x_y_grid`, `/upscaled_local`, `/annotator_map`, and `/png_info`.


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

The same endpoints are also [MCP](https://modelcontextprotocol.io) tools. Launch with `mcp_server=True` ([guide](https://www.gradio.app/guides/building-mcp-server-with-gradio)) and every output node shows up as a tool an AI assistant can call. Point Claude Code, Cursor, or any MCP client at the server URL:

```json
{
  "mcpServers": {
    "workflow1111": {
      "url": "https://ysharma-workflow1111.hf.space/gradio_api/mcp/",
      "headers": { "X-HF-Token": "hf_..." }
    }
  }
}
```

Now an agent can generate an image, read a prompt back out, or run detection as steps in a larger task, with no glue code. Each caller sends their own token in the `X-HF-Token` header, so the Space holds none of its own.

## Where this sits next to ComfyUI

AUTOMATIC1111 gave us the feature list, but the tool Gradio Workflow really gets compared to is ComfyUI, since both are node graphs. For a lot of what people want to build and ship, `gr.Workflow` covers the same ground.

* **A node can be hardware you don't own.** It can run through [Inference Providers](https://huggingface.co/docs/inference-providers/index), call any Space on the Hub or any API, or pull from a dataset. That's how Workflow1111 runs without a GPU of its own.
* **Every output becomes a typed REST endpoint.** The endpoints are generated from the graph.
* **Visitors can run workflows under their own identity.** Turn on [OAuth](https://huggingface.co/docs/hub/spaces-oauth), share the public URL, and anyone can sign in and use the app without installing anything.
* **Mix models and modalities on the same canvas.** Diffusion models, LLMs, VLMs, detectors, and video models can all be part of the same workflow.
* **Need something custom? Write a function.** A custom node is a Python function, so it can do whatever Python can.

The result is a multi-model pipeline that people can open in a browser, sign into, use right away, and call from code.

## Build your own

Workflow1111 has 73 nodes, but it started with just this:

```python
import gradio as gr

def your_function(text: str) -> str:
    pass

gr.Workflow(bind=[your_function]).launch()
```

`bind=` turns your functions into nodes, `edges=` connects them, and `.launch()` opens the canvas in your browser so you can keep editing there. When it's ready, `gradio deploy` puts the whole thing on a Space. The [gr.Workflow guide](https://gradio.app/guides/workflows) has the full details, including the JSON schema and every operator type.

If you'd rather start from something that already works, open [Workflow1111](https://huggingface.co/spaces/ysharma/Workflow1111), hit **Duplicate**, and pick one of the eleven pipelines to change: delete nodes, swap models, rewire the flow. If you'd rather start smaller, the [previous post](https://huggingface.co/blog/gradio-workflow-guide) has five workflows you can get running in about a minute each.

Whatever you build, post it on X and tag [@gradio](https://x.com/Gradio). We'd be happy to amplify your workflows.