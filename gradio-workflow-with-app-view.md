---
title: "Gradio Workflow Now Has An App View"
thumbnail: /blog/assets/gradio-workflow-with-app-view/thumbnail.png
authors:
- user: ysharma
---


# gr.Workflow now has an app view

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-with-app-view/toggle-app-to-workflow-gradio.mp4
"></video>

`gr.Workflow` lets you build an AI pipeline as a graph of nodes on a canvas. The canvas is great while you build, but it's a lot for someone who just wants to try your pipeline.

Gradio 6.30 fixes that. Every workflow now has two views:

- **Workflow**: the node canvas, for building and debugging.
- **App**: a plain Gradio app made from the same graph, with input fields, a Run button and outputs.

A toggle switches between them. Visitors to a Space see the app first.


## Example: a manga page maker

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-with-app-view/gradio-workflow-now-has-app-view-too.png" alt="Gradio Workflow now come up with an App view too" width="600" >
</p>

[Manga Page Maker Turbo](https://huggingface.co/spaces/ysharma/manga-page-maker-turbo) draws manga pages with [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) on ZeroGPU. The graph has two Python functions:

1. `build_prompt` turns a short bubble list such as `large top left: I WON'T BACK DOWN! | bottom right: NEVER.` and a scene into the prompt the model likes.
2. `draw_page` runs Turbo for 8 steps and returns the page.

On the canvas you see both these steps. In the app, visitors see the inputs (bubbles, scene, width, height, seed) and the outputs (the page, the prompt that was used, the seed).

## The code

```python
import spaces
import gradio as gr

def build_prompt(bubbles: str, scene: str) -> str:
    ...

@spaces.GPU(duration=40)
def draw_page(prompt: str, width: float = 1024, height: float = 1024, seed: float = -1):
    ...

demo = gr.Workflow("workflow.json", bind={"build_prompt": build_prompt, "draw_page": draw_page})
demo.launch()
```

`bind` gives the graph your functions. `workflow.json` stores the nodes and how they connect. 
You don't have to write any layout for the app! Gradio builds it from the graph. Input nodes become fields, output nodes become results, and everything in between runs when you press Run.


## Who sees what

- Visitors land on the app by default and can switch to the canvas.
- The Space owner, signed in with Hugging Face, lands on the editable canvas (add `hf_oauth: true` to the Space README).
- Add `?ui=app` or `?ui=canvas` in the URL to pick or share the respective view.

When the owner changes the graph and saves, the app updates to match.

## It's also an API

The app's Run button is a normal Gradio endpoint, so you can run the whole graph from Python:

```python
from gradio_client import Client

client = Client("ysharma/manga-page-maker-turbo")
page, prompt, seed = client.predict(
    "small top right: Is this... for me? | center | large bottom left: BAKA!",
    "shoujo romance in a school hallway, a blushing girl hands a love letter to a surprised boy",
    832, 1216, 42,
    api_name="/manga_page",
)
```

## Try it

Open the [featured Space](https://huggingface.co/spaces/ysharma/manga-page-maker-turbo), write a few bubbles and a scene, press Run, then switch to **Workflow** to see how your app was made.

For more advance controls, explore the complete [manga page maker](https://huggingface.co/spaces/ysharma/manga-page-maker) Gradio app on Huggingface.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-with-app-view/manga-page-maker-multiple-speech-bubbles.png" alt="Mange Page Maker Gradio App" width="600" >
</p>

To build your own, run `pip install -U gradio`. `gr.Workflow` is in beta, so tell us what you'd like next.
