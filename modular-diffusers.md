---
title: "Introducing Modular Diffusers - Composable Building Blocks for Diffusion Pipelines"
thumbnail: /blog/assets/modular-diffusers/thumbnail.png
authors:
  - user: YiYiXu
  - user: OzzyGT
  - user: dn6
  - user: sayakpaul
---

# Introducing Modular Diffusers - Composable Building Blocks for Diffusion Pipelines

Modular Diffusers introduces a new way to build diffusion pipelines by composing reusable blocks. Instead of writing entire pipelines from scratch, you can now mix and match building blocks to create custom workflows tailored to your specific needs! This complements the existing `DiffusionPipeline` class, providing a more flexible way to create custom diffusion pipelines.

In this post, we'll walk through how Modular Diffusers works — from the familiar API to run a modular pipeline, to building fully custom blocks and composing them into your own workflow. We'll also show how it integrates with Mellon, a node-based visual workflow interface that you can use to wire Modular Diffusers blocks together.

**Table of contents**

- [Quickstart](#quickstart)
- [Modular Repositories](#modular-repositories)
- [Custom Blocks](#custom-blocks)
- [Community Pipelines](#community-pipelines)
- [Integration with Mellon](#integration-with-mellon)

## Quickstart

Getting started with Modular Diffusers is straightforward. Here is a simple example of how to run inference  for `FLUX.2 Klein 4B` using pre-built blocks

```python
import torch
from diffusers import ModularPipeline

# Create a modular pipeline - this only defines the workflow, model weights have not been loaded yet
pipe = ModularPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B"
)
#  Now load the model weights — configure dtype, quantization, etc in this step
pipe.load_components(torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Generate an image - API remains the same as DiffusionPipeline
image = pipe(
    prompt="a serene landscape at sunset",
    num_inference_steps=4,
).images[0]

image.save("output.png")
```

Behind the scenes, this pipeline is composed of multiple blocks working together — text encoding, image encoding, denoising, and decoding. You can inspect each of them directly:

```python
print(pipe.blocks)
```

```bash
Flux2KleinAutoBlocks(
  ...
  Sub-Blocks:
    [0] text_encoder (Flux2KleinTextEncoderStep)
    [1] vae_encoder (Flux2KleinAutoVaeEncoderStep)
    [2] denoise (Flux2KleinCoreDenoiseStep)
    [3] decode (Flux2DecodeStep)
)
```

In this example, you get the same generation results as if you had loaded the standard `DiffusionPipeline`, but the pipeline is very different under the hood: it's made of flexible _blocks_ that you can combine in different ways.

Each block is self-contained, with its own defined inputs and outputs. You can take any block and run it independently as its own pipeline, or add, remove, and swap blocks freely, they will dynamically recompose to work with whatever blocks remain. When you're ready to run, use `.init_pipeline()` to convert your blocks into a runnable pipeline, and `.load_components()` to load the model weights.

```python
# get a copy of the blocks
blocks = pipe.blocks

# pop out the text_encoder block
text_blocks = blocks.sub_blocks.pop("text_encoder")

# run it as its own pipeline
text_pipe = text_blocks.init_pipeline("black-forest-labs/FLUX.2-klein-4B")

# load the text_encoder, or reuse already loaded components: text_pipe.update_components(text_encoder=pipe.text_encoder)
text_pipe.load_components(torch_dtype=torch.bfloat16)
text_pipe.to("cuda")
prompt_embeds = text_pipe(prompt="a serene landscape at sunset").prompt_embeds
		
# create a new pipeline from the remaining blocks
# it now accepts prompt_embeds directly instead of prompt
remaining_pipe = blocks.init_pipeline("black-forest-labs/FLUX.2-klein-4B")
remaining_pipe.load_components(torch_dtype=torch.bfloat16)
remaining_pipe.to("cuda")
image = remaining_pipe(prompt_embeds=prompt_embeds, num_inference_steps=4).images[0]
```

For more on the different block types, composition patterns, lazy loading, and efficiently managing model memory across pipelines with `ComponentsManager`, check out the [Modular Diffusers documentation.](https://huggingface.co/docs/diffusers/en/modular_diffusers/overview).

## Modular Repositories

When you call `ModularPipeline.from_pretrained`, it works with any existing Diffusers repo out of the box. But Modular Diffusers also introduces Modular Repositories.

A modular repository doesn't duplicate any model weights. Instead, it references components directly from their original model repos. For example, [diffusers/flux2-bnb-4bit-modular](https://huggingface.co/diffusers/flux2-bnb-4bit-modular) contains no model weights at all — it loads a quantized transformer from one repo and the remaining components from another.

```json
// diffusers/flux2-bnb-4bit-modular/modular_model_index.json
{
	"transformer": [
		null, 
		null, 
		{
			"pretrained_model_name_or_path": "diffusers/FLUX.2-dev-bnb-4bit",
			"subfolder": "transformer",
			"type_hint": ["diffusers", "Flux2Transformer2DModel"]
		}
	],
	"vae": [
		null, 
		null, 
		{
			"pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev",
			"subfolder": "vae",
			"type_hint": ["diffusers", "AutoencoderKLFlux2"]
		}
	],
	...
}
```

Modular repositories can also host custom pipeline blocks as Python code and visual UI configurations for tools like [Mellon](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/mellon) — all in one place. 

## Custom Blocks

So far we've been working with pre-built blocks. But where Modular Diffusers really shines is in creating your own custom blocks. A custom block is a Python class that defines its components, inputs, outputs, and computation logic — and once defined, you can use it with any workflow.

### Writing a Custom Block

Here's an example block that extracts depth maps from images using [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large).

```python
class DepthProcessorBlock(ModularPipelineBlocks):
    @property
    def expected_components(self):
        return [
            ComponentSpec("depth_processor", DepthPreprocessor,
                          pretrained_model_name_or_path="depth-anything/Depth-Anything-V2-Large-hf")
        ]

    @property
    def inputs(self):
        return [
            InputParam("image", required=True,
                       description="Image(s) to extract depth maps from"),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("control_image", type_hint=torch.Tensor,
                        description="Depth map(s) of input image(s)"),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        depth_map = components.depth_processor(block_state.image)
        block_state.control_image = depth_map.to(block_state.device)
        self.set_block_state(state, block_state)
        return components, state
```

- `expected_components` defines what models the block needs: here we added a depth estimation model. Notice the `pretrained_model_name_or_path` parameter - it sets a default path, so when you call `load_components`, the depth model is automatically loaded from that Hub repo unless you override it from the modular_model_index.json
- `inputs` and `intermediate_outputs` define what goes in and comes out.
- `__call__` is where the computation logic lives.

### Composing Blocks into Workflows

Let's use this block with Qwen's ControlNet workflow. Extract the ControlNet workflow and insert the depth block at the beginning:

```python
# Create Qwen Image pipeline
pipe = ModularPipeline.from_pretrained("Qwen/Qwen-Image")

print(pipe.blocks.available_workflows)
#       Supported workflows:
#        - `text2image`: requires `prompt`
#        - `image2image`: requires `prompt`, `image`
#        - `inpainting`: requires `prompt`, `mask_image`, `image`
#        - `controlnet_text2image`: requires `prompt`, `control_image`
#        - `controlnet_image2image`: requires `prompt`, `image`, `control_image`

# Extract the ControlNet workflow — it expects a condition_image input
blocks = pipe.blocks.get_workflow("controlnet_text2image")
# Show the blocks this workflow uses
print(blocks)

# Insert depth block at the beginning — its output (condition_image)
# automatically flows to the ControlNet block that needs it
blocks.sub_blocks.insert("depth", DepthProcessorBlock(), 0)

# You can inspect any block's inputs and outputs with print(blocks.doc)
blocks.sub_blocks['depth'].doc
```

Because blocks in a sequence share data automatically, the depth block's output (`control_image`) flows to the following blocks that need it, and the depth block's input (`image`) becomes a pipeline input since no earlier block provides it.

<p align="center">
  <img src="https://huggingface.co/datasets/diffusers/modular-diffusers-blog/resolve/main/blocks_composed.png" alt="blocks_composed">
</p>

```python
from diffusers import ComponentsManager, AutoModel
from diffusers.utils import load_image

# ComponentsManager handles memory across multiple pipelines —
# it automatically offloads models to CPU when not in use
manager = ComponentsManager()

pipeline = blocks.init_pipeline("Qwen/Qwen-Image", components_manager=manager)
pipeline.load_components(torch_dtype=torch.bfloat16)

# The depth model loads automatically from the default path we set in expected_components —
# no need to load it manually even though it's not part of the Qwen repo.
# But controlnet is not included by default, so we do need to load it from a different repo
controlnet = AutoModel.from_pretrained("InstantX/Qwen-Image-ControlNet-Union", torch_dtype=torch.bfloat16)
pipeline.update_components(controlnet=controlnet)

# pipeline now takes image as input
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")
output = pipeline(
    prompt="an astronaut hatching from an egg, detailed, fantasy, Pixar, Disney",
    image=path/to/image
).images[0]
```

### Sharing Custom Blocks on the Hub

You can publish your custom block to the Hub so anyone can load it with `trust_remote_code=True`. We've created a [template](https://huggingface.co/diffusers/custom-block-template) to get you started — check out the [Building Custom Blocks guide](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks#quick-start-with-template) for the full walkthrough.

The `DepthProcessorBlock` from this example is published at [diffusers/depth-processor-custom-block](https://huggingface.co/diffusers/depth-processor-custom-block) — you can load and use it directly:

```python
from diffusers import ModularPipelineBlocks

depth_block = ModularPipelineBlocks.from_pretrained(
    "diffusers/depth-processor-custom-block", trust_remote_code=True
)
```

We've published a collection of ready-to-use custom blocks [here](https://www.notion.so/huggingface2/link-to-collection).

## Community Pipelines

The community has already started building complete pipelines with Modular Diffusers and publishing them on the Hub, with model weights and ready-to-run code.

- [**Krea Realtime Video**](https://huggingface.co/krea/krea-realtime-video) — A 14B parameter real-time video generation model distilled from Wan 2.1, achieving 11fps on a single B200 GPU. It supports text-to-video, video-to-video, and streaming video-to-video — all built as modular blocks. Users can modify prompts mid-generation, restyle videos on-the-fly, and see first frames within 1 second.

```python
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("krea/krea-realtime-video", trust_remote_code=True)
pipe.load_components(
    trust_remote_code=True, 
    device_map="cuda",
    torch_dtype={"default": torch.bfloat16, "vae": torch.float16}
)
```

- [**Waypoint-1**](https://huggingface.co/Overworld/Waypoint-1-Small) — A 2.3B parameter real-time diffusion world model from [Overworld](https://over.world). It autoregressively generates interactive worlds from control inputs and text prompts — you can explore and interact with generated environments in real time on consumer hardware.

These pipelines showcase what's possible when you have a composable, block-based system: teams can build novel architectures, package them as modular blocks, and publish the entire pipeline on the Hub for anyone to use with `ModularPipeline.from_pretrained`.

Check out the full [collection of community pipelines](https://huggingface.co/collections/diffusers/modular-pipelines) for more.

## Integration with Mellon


> [!TIP]
> 💡 Mellon is in early development and not ready for production use yet. Consider this a sneak peek of how the integration works!

[Mellon](https://github.com/cubiq/Mellon) is a visual workflow interface integrated with Modular Diffusers. If you're familiar with node-based tools like ComfyUI, you'll feel right at home — but there are some key differences:

- **Dynamic nodes** — Instead of dozens of model-specific nodes, we have a small set of nodes that automatically adapt their interface based on the model you select. Learn them once, use them with any model.
- **Single-node workflows** — Thanks to Modular Diffusers' composable block system, you can collapse an entire pipeline into a single node. Run multiple workflows on the same canvas without the clutter.
- **Hub integration out of the box** — Custom blocks published to the Hugging Face Hub work instantly in Mellon. We provide a utility function to automatically generate the node interface from your block definition — no UI code required.

This integration is possible because of Modular Diffusers' **consistent API** — every block defines the same properties (`inputs`, `intermediate_outputs`, `expected_components`) — and its **composability**. These two things mean we can automatically generate a node's UI from any block definition and compose blocks into higher-level nodes without any manual UI work.

And it all comes together in the modular repository. For example, [diffusers/FLUX.2-klein-4B-modular](https://huggingface.co/diffusers/FLUX.2-klein-4B-modular) contains a pipeline definition, its component references, and a `mellon_pipeline_config.json` — all in one repo. Load it in Python with `ModularPipeline.from_pretrained("diffusers/FLUX.2-klein-4B-modular"`) or in Mellon to create either single-node or multi-node workflow. 

Here's a quick example to give you a taste. We add a Gemini prompt expansion node — hosted as a modular repo at [diffusers/gemini-prompt-expander-mellon](https://huggingface.co/diffusers/gemini-prompt-expander-mellon) — to an existing text-to-image workflow:

1. Drag in a **Dynamic Block** node and enter the `repo_id` (i.e. `diffusers/gemini-prompt-expander-mellon`)
2. Click **LOAD CUSTOM BLOCK** — the node automatically grows a textbox for your prompt input and an output socket named "prompt", all configured from the repo
3. Type a short prompt, connect the output to the **Encode Prompt** node, and run

Gemini expands your short prompt into a detailed description before generating the image. No code, no configuration — just a Hub repo id.

<video controls>
  <source src="https://huggingface.co/datasets/diffusers/modular-diffusers-blog/resolve/main/demo6_gemini.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>

This is just one example. For a detailed walkthrough, check out the [Mellon x Modular Diffusers guide](https://www.notion.so/Mellon-x-Modular-Diffusers-2fd1384ebcac819993d8f9ae94c7e866?pvs=21).


## Conclusion

Modular Diffusers brings the composability and flexibility the community has been asking for, without compromising the features that make Diffusers powerful. It's still early and we're releasing this because we want your input shaping what comes next. Give it a try and tell us what works, what doesn't, and what's missing.

## Resources

Below are all the important links pertaining to Modular Diffusers:

- [Overview](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/overview) of Modular Diffusers, including all the important links already
- [Mellon](https://github.com/cubiq/Mellon)
- [Mellon x Modular Diffusers](https://www.notion.so/Mellon-x-Modular-Diffusers-2fd1384ebcac819993d8f9ae94c7e866?pvs=21)
- [Collection](https://www.notion.so/huggingface2/link-to-collection) of custom blocks
- [Collection](https://huggingface.co/collections/diffusers/modular-pipelines) of community pipelines with Modular Diffusers

_Thanks to Chun Te Lee for working on the thumbnail of this post._