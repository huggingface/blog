---
title: "Introducing Daggr: Chain apps programmatically, inspect visually"
thumbnail: /blog/assets/daggr/thumbnail.png
authors:
- user: merve
- user: ysharma
- user: abidlabs
- user: hysts
- user: pcuenq
---

# Introducing Daggr: Chain apps programmatically, inspect visually

**TL;DR:** [Daggr](https://github.com/gradio-app/daggr) is a new, open-source Python library for building AI workflows that connect Gradio apps, ML models, and custom functions. It automatically generates a visual canvas where you can inspect intermediate outputs, rerun individual steps, and manage state for complex pipelines—all in a few lines of Python code.

![demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/daggr-blog/daggr.mp4)

## Table of Contents

1. [Background](#background)
2. [Getting Started](#getting-started)
3. [Sharing Your Workflows](#sharing-your-workflows)
4. [End-to-End Example with Different Nodes](#end-to-end-example-with-different-nodes)
5. [Next Steps](#next-steps)

## Background

If you've built AI applications that combine multiple models or processing steps, you know the pain: chaining API calls, debugging pipelines, and losing track of intermediate results. When something goes wrong in step 5 of a 10-step workflow, you often have to re-run everything just to see what happened.

Most developers either build fragile scripts that are hard to debug or turn to heavy orchestration platforms designed for production pipelines—not rapid experimentation.

We've been working on Daggr to solve problems we kept running into when building AI demos and workflows:

**Visualize your code flow**: Unlike node-based GUI editors, where you drag and connect nodes visually, Daggr takes a code-first approach. You define workflows in Python, and a visual canvas is generated automatically. This means you get the best of both worlds: version-controllable code and visual inspection of intermediate outputs.

**Inspect and Rerun Any Step**: The visual canvas isn't just for show. You can inspect the output of any node, modify inputs, and rerun individual steps without executing the entire pipeline. This is invaluable when you're debugging a 10-step workflow and only step 7 is misbehaving. You can even provide “backup nodes” – replacing one model or Space with another – to build resilient workflows.

**First-Class Gradio Integration**: Since Daggr is built by the Gradio team, it works seamlessly with Gradio Spaces. Point to any public (or private) Space and you can use it as a node in your workflow. No adapters, no wrappers—just reference the Space name and API endpoint.

**State Persistence**: Daggr automatically saves your workflow state, input values, cached results, canvas position—so you can pick up where you left off. Use "sheets" to maintain multiple workspaces within the same app.

## Getting Started

Install daggr with pip or uv, it just requires Python 3.10 or higher:

```shell
pip install daggr
uv pip install daggr
```

Here's a simple example that generates an image and removes its background. Check out [this Space’s API reference](https://huggingface.co/spaces/hf-applications/Z-Image-Turbo) from the bottom of the Space to see which inputs it takes and which outputs it yields. In this example, the Space returns both original image and the edited image, so we return only the edited image.

```py
import random
import gradio as gr
from daggr import GradioNode, Graph

# Generate an image using a Gradio Space
image_gen = GradioNode(
    "hf-applications/Z-Image-Turbo",
    api_name="/generate_image",
    inputs={
        "prompt": gr.Textbox(
            label="Prompt",
            value="A cheetah sprints across the grassy savanna.",
            lines=3,
        ),
        "height": 1024,
        "width": 1024,
        "seed": random.random,
    },
    outputs={
        "image": gr.Image(label="Generated Image"),
    },
)

# Remove background using another Gradio Space
bg_remover = GradioNode(
    "hf-applications/background-removal",
    api_name="/image",
    inputs={
        "image": image_gen.image,  # Connect to previous node's output
    },
    outputs={
        "original_image": None,  # Hide this output
        "final_image": gr.Image(label="Final Image"),
    },
)

graph = Graph(
    name="Transparent Background Generator", 
    nodes=[image_gen, bg_remover]
)
graph.launch()
```

That's it. Run this script and you get a visual canvas served on port 7860 launched automatically, as well as a shareable live link, showing both nodes connected, with inputs you can modify and outputs you can inspect at each step.

![App](https://huggingface.co/datasets/huggingface/documentation-images/blob/main/daggr-blog/app1.png)

### Node Types

Daggr supports three types of nodes:

**GradioNode** calls a Gradio Space API endpoint or locally served Gradio app. Passing `run_locally=True`, Daggr automatically clones the Space, creates an isolated virtual environment, and launches the app. If local execution fails, it gracefully falls back to the remote API.

```py
node = GradioNode(
    "username/space-name",
    api_name="/predict",
    inputs={"text": gr.Textbox(label="Input")},
    outputs={"result": gr.Textbox(label="Output")},
)

# clone a Space locally and serve
node = GradioNode(
    "hf-applications/background-removal",
    api_name="/image",
    run_locally=True,
    inputs={"image": gr.Image(label="Input")},
    outputs={"final_image": gr.Image(label="Output")},
```

**FnNode** — runs a custom Python function:

```py
def process(text: str) -> str:
    return text.upper()

node = FnNode(
    fn=process,
    inputs={"text": gr.Textbox(label="Input")},
    outputs={"result": gr.Textbox(label="Output")},
)
```

**InferenceNode** — calls a model via Hugging Face Inference Providers:

```py
node = InferenceNode(
    model="moonshotai/Kimi-K2.5:novita",
    inputs={"prompt": gr.Textbox(label="Prompt")},
    outputs={"response": gr.Textbox(label="Response")},
)
```


### Sharing Your Workflows

Generate a public URL with Gradio's tunneling:

```py
graph.launch(share=True)
```

For permanent hosting, deploy on Hugging Face Spaces using the Gradio SDK—just add `daggr` to your `requirements.txt`.

## End-to-End Example with Different Nodes

We will now develop an app that takes in an image and generates a 3D asset. This demo can run on daggr 0.4.3. Here are the steps:

1. **Take an image, remove the background:** For this, we will clone the [BiRefNet Space](https://huggingface.co/spaces/merve/background-removal) and run it locally.  
2. **Downscale the image for efficiency:** We will write a simple function for this with FnNode.  
3. **Generate an image in 3D asset style for better results:** We will use InferenceNode with [Flux.2-klein-4B model](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on Inference Providers.  
4. Pass the output image to a 3D generator: We will send the output image to the Trellis.2 Space hosted on Spaces.

> [!TIP]  
> Spaces that are run locally might take models to CUDA (with `to.(“cuda”)`) or ZeroGPU within the application file. To disable this behavior to run the model on CPU (useful if you have a device with no NVIDIA GPU) duplicate the Space you want to use and clone it. 

The resulting graph looks like below.

![App](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/daggr-blog/app2.png)

Let’s write the first step, which is the background remover. We will clone and run [this Space](https://huggingface.co/spaces/merve/background-removal) locally. This Space runs on CPU, and takes ~13 seconds to run. You can swap with [this app](https://huggingface.co/spaces/hf-applications/background-removal) if you have an NVIDIA GPU.

```py
from daggr import FnNode, GradioNode, InferenceNode, Graph

background_remover = GradioNode(
   "merve/background-removal",
   api_name="/image",
   run_locally=True, 
   inputs={
       "image": gr.Image(),
   },
   outputs={
       "original_image": None,
       "final_image": gr.Image(
           label="Final Image"
       ),
   },
)
```

For the second step, we need to write a helper function to downscale the image and pass it to `FnNode`.

```py
from PIL import Image
from daggr.state import get_daggr_files_dir


def downscale_image_to_file(image: Any, scale: float = 0.25) -> str | None:
   pil_img = Image.open(image)
   scale_f = max(0.05, min(1.0, float(scale)))
   w, h = pil_img.size
   new_w = max(1, int(w * scale_f))
   new_h = max(1, int(h * scale_f))
   resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)
   out_path = get_daggr_files_dir() / f"{uuid.uuid4()}.png"

   resized.save(out_path)
   return str(out_path)
```

We can now pass in the function to initialize the `FnNode`. 

```py
downscaler = FnNode(
   downscale_image_to_file,
   name="Downscale image for Inference",
   inputs={
       "image": background_remover.final_image,
       "scale": gr.Slider(
           label="Downscale factor",
           minimum=0.25,
           maximum=0.75,
           step=0.05,
           value=0.25,
       ),
   },
   outputs={
       "image": gr.Image(label="Downscaled Image", type="filepath"),
   },
)
```

We will now write the `InferenceNode` with the Flux model.

```py

flux_enhancer = InferenceNode(
   model="black-forest-labs/FLUX.2-klein-4B:fal-ai",
   inputs={
       "image": downscaler.image,
       "prompt": gr.Textbox(
           label="prompt",
           value=("Transform this into a clean 3D asset render"),
           lines=3,
       ),
   },
   outputs={
       "image": gr.Image(label="3D-Ready Enhanced Image"),
   },
)

```

> [!TIP]  
> When deploying apps with InferenceNode to Hugging Face Spaces, use a fine-grained Hugging Face access token with the option "Make calls to Inference Providers" only.

Last node is 3D generation with querying the Trellis.2 Space on Hugging Face. 

```py

trellis_3d = GradioNode(
   "microsoft/TRELLIS.2",
   api_name="/image_to_3d",
   inputs={
       "image": flux_enhancer.image,
       "ss_guidance_strength": 7.5,   
       "ss_sampling_steps": 12,     
   },
   outputs={
       "glb": gr.HTML(label="3D Asset (GLB preview)"),
   },
)

```

Chaining them together and launching the app is as simple as follows.

```py
graph = Graph(
   name="Image to 3D Asset Pipeline",
   nodes=[background_remover, downscaler, flux_enhancer, trellis_3d],
)

if __name__ == "__main__":
   graph.launch()
```

You can find the complete example running in [this Space](https://huggingface.co/spaces/merve/daggr-image-to-3d), to run locally you just need to take app.py, install requirements and login to Hugging Face Hub.

## Next Steps

Daggr is in beta and intentionally lightweight. APIs may change between versions, and while we persist workflow state locally, data loss is possible during updates. If you have feature requests or find bugs, please open an issue [here](https://github.com/gradio-app/daggr/issues). We’re looking forward to your feedback! Share your daggr workflows on socials with Gradio for a chance to be featured. Check out all the featured works [here](https://huggingface.co/collections/ysharma/daggr-hf-spaces).

