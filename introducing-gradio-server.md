---
title: "Any Custom Frontend with Gradio's Backend"
thumbnail: /blog/assets/introducing-gradio-server/thumbnail.png
authors:
- user: ysharma
- user: abidlabs
---

# gradio.Server: Any Custom Frontend with Gradio's Backend

A few weeks ago, we wrote about [one-shotting full web apps with `gr.HTML`](https://huggingface.co/blog/gradio-html-one-shot-apps): building rich, interactive frontends entirely inside Gradio using custom HTML, CSS, and JavaScript. That unlocked a lot. But what if that's not enough?

What if you want to **build with your own frontend framework entirely** like React, Svelte, or even plain HTML/JS, while still benefiting from Gradio's queuing system, API infrastructure, MCP support, and ZeroGPU on Spaces?

That's exactly the problem **`gradio.Server`** solves. And it changes what's possible with Gradio and Hugging Face Spaces.


## What We Wanted to Build

<video alt="Using gradio.Server to use an ML backend to remove the image background and then creating a fast and responsive custom front end on top of it" autoplay loop muted playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-server/text-behind-image-for-blog.mp4" type="video/mp4">
</video>

**[Text Behind Image](https://huggingface.co/spaces/ysharma/text-behind-image)** : an editor where you upload a photo, the background gets removed using an ML model, and then you place stylized text *between* the foreground subject and the background. The text appears to sit behind the person or object in the image.

This needs:
- A **drag-and-drop canvas** with layered rendering (background → text → foreground)
- A **rich control panel** with sliders for font size, weight, letter spacing, color, opacity, stroke, shadow, 3D extrusion, perspective transforms, and more
- A **backend ML endpoint** that runs a background-removal model and returns a transparent PNG
- **Export to PNG** on the client side

There's no way to express this UI in Gradio components. It's a full web application. But We still wanted the backend power of Gradio: queuing, concurrency management, ZeroGPU support, and hosting on HF Spaces without infrastructure headaches.

## Enter `gradio.Server`

`gradio.Server` extends FastAPI. It gives you the full power of FastAPI (custom routes, middleware, file uploads, any response type) while adding Gradio's API engine on top: queuing, SSE streaming, concurrency control, and `gradio_client` compatibility.

Here's the entire backend for Text Behind Image:

```python
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse
import spaces

torch.set_float32_matmul_precision("high")

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")
birefnet.float()

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = Server()


@spaces.GPU
def segment(image: Image.Image) -> Image.Image:
    """Run BiRefNet segmentation to produce a transparency mask."""
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    mask = transforms.ToPILImage()(pred).resize(image_size)
    image.putalpha(mask)
    return image


@app.api(name="remove_background")
def remove_background(image_path: FileData) -> FileData:
    """Remove background from an image. Returns transparent PNG."""
    im = Image.open(image_path["path"]).convert("RGB")
    result = segment(im)
    out_path = image_path["path"].rsplit(".", 1)[0] + ".png"
    result.save(out_path)
    return FileData(path=out_path)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


app.launch(show_error=True)

```

That's it. ~50 lines of Python. The model loads at startup, `@spaces.GPU` handles ZeroGPU allocation, and `gradio.Server` manages queuing and concurrency. Let's break down what's happening.

### Why `@app.api()` Instead of a Plain FastAPI Route?

If this were a regular FastAPI app, you'd define a `@app.post()` route for background removal. That works, until two users hit it at once. Without concurrency management, both requests fight for the GPU, and the app crashes or returns garbage.

`@app.api()` solves this. It wraps your function with Gradio's queuing engine: requests are serialized, concurrency is controlled, and on ZeroGPU Spaces, GPU allocation is handled automatically via `@spaces.GPU`. As a bonus, any `@app.api()` endpoint is also callable via `gradio_client`, so other apps or scripts can use your Space programmatically:

```python
from gradio_client import Client, handle_file
client = Client("ysharma/text-behind-image")
result = client.predict(
      image_path=handle_file("photo.jpg"),
      api_name="/remove_background"
  )
```

Meanwhile, `@app.get("/")` is a standard FastAPI route that serves the HTML page. Both coexist naturally because `Server` *is* a FastAPI app.

### The Frontend: Pure HTML/CSS/JS

The `index.html` in this example is a self-contained ~1300-line web application. No React, no build step, no bundler. Just vanilla HTML with:

- A **three-layer canvas**: background image → text layer → foreground (transparent PNG) stacked with CSS `z-index`
- **Drag-and-drop text positioning** using pointer events
- A **control panel** with 20+ parameters: font family (25+ fonts), size, weight, spacing, color, opacity, background fill, stroke, shadow, 3D extrusion depth and angle, rotation, skew, and full CSS perspective transforms
- **Client-side PNG export** using `<canvas>` compositing

The frontend talks to the backend using the [Gradio JS Client](https://www.gradio.app/guides/getting-started-with-the-js-client):

```javascript
import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const client = await Client.connect(window.location.origin);
const result = await client.predict("/remove_background", {
    image_path: handle_file(file),
});
foregroundLayer.src = result.data[0].url;  // transparent PNG
```
This is the key part: by using the Gradio JS client rather than a raw `fetch()` call, the frontend goes through Gradio's queue. That means concurrency is managed, GPU requests don't collide, and you could even show queue position or progress to the user. Everything else, text rendering, layer compositing, export, happens in the browser.

## What This Unlocks

Here's what was **not possible** before `gradio.Server`:

| Before | After |
|--------|-------|
| Custom UI meant leaving Gradio entirely | Custom UI *with* Gradio's backend engine |
| No way to serve static HTML from a Gradio app | `@app.get("/")` serves anything |
| `gradio_client` only worked with Gradio component apps | `@app.api()` endpoints are client-compatible |
| Choosing between Gradio's infra and design freedom | You get both |

With `gradio.Server`, **Gradio doubles as a backend framework, use its UI system when you want it, bring your own frontend when you don't.**

If you want Gradio's UI, you can use `gr.Blocks`, `gr.Interface`, `gr.ChatInterface`. If you want your own UI, use `gradio.Server` and bring whatever frontend you like. Either way, you get Spaces hosting, API queuing, `gradio_client` access, the full HF ecosystem, and more.

## Try It

The app is live on Spaces: **[ysharma/text-behind-image](https://huggingface.co/spaces/ysharma/text-behind-image)**

Upload any photo with a clear subject, and start layering text behind it. Try the 3D extrusion, perspective tilt, and stroke effects, they combine nicely.

## What's Next

This post covered the core idea: `gradio.Server` lets you pair any frontend with Gradio's backend. There's more to explore, including **MCP tool registration** with `@app.mcp.tool()`, **SSE streaming** for real-time updates, **batch processing**, and patterns for building multi-page apps with shared state.

We'll dig into those in a follow-up post. Stay tuned.

## Recommended reading

- Docs: [gradio.Server](https://www.gradio.app/main/docs/gradio/server)
- Guide: [Server mode](https://www.gradio.app/guides/server-mode) 