---
title: "Using Gradio Server: A 700-Line Custom Frontend, Powered by 40 Lines of Gradio"
thumbnail: /blog/assets/introducing-gradio-server/thumbnail.png
authors:
- user: ysharma
- user: abidlabs
---

# Using Gradio Server:  A 700-Line Custom Frontend, Powered by 40 Lines of Gradio

A few weeks ago, I wrote about [one-shotting full web apps with `gr.HTML`](https://huggingface.co/blog/gradio-html-one-shot-apps): building rich, interactive frontends entirely inside Gradio using custom HTML, CSS, and JavaScript. That unlocked a lot. But there was still a wall.

What if your app needs a **fully custom frontend**, a real index.html with hundreds of lines of interactive UI, backed by ML models running on the server?

That's exactly the problem **`gradio.Server`** solves. And it changes what's possible with Gradio and Hugging Face Spaces.

## No more Frontend lock-in to Gradio's component system

Before `gradio.Server`, you had two choices when building on Spaces:

1. **Use Gradio's component system** : great for ML demos, but you're locked into Gradio's layout and widget vocabulary. Want a canvas with drag-and-drop text layers, perspective transforms, and a custom control panel? These kinds of web apps are very difficult to fire-up using `gr.HTML` alone.

2. **Ditch Gradio entirely** : spin up a raw FastAPI/Flask app, lose Gradio's built-in API infrastructure, queuing, `gradio_client` compatibility, ZeroGPU, and the seamless Spaces integration that comes with it.

Neither option worked for what I wanted to build this time.

## What I Wanted to Build

<video alt="Using gradio.Server to use an ML backend to remove the image background and then creating a fast and responsive custom front end on top of it" autoplay loop muted playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-server/text-behind-image-for-blog.mp4" type="video/mp4">
</video>


**[Text Behind Image](https://huggingface.co/spaces/ysharma/text-behind-image)** : an editor where you upload a photo, the background gets removed using an ML model, and then you place stylized text *between* the foreground subject and the background. The text appears to sit behind the person or object in the image.

This needs:
- A **drag-and-drop canvas** with layered rendering (background → text → foreground)
- A **rich control panel** with sliders for font size, weight, letter spacing, color, opacity, stroke, shadow, 3D extrusion, perspective transforms, and more
- A **backend ML endpoint** that calls a background-removal model and returns a transparent PNG
- **Export to PNG** on the client side

There's no way to express this UI in Gradio components. It's a full web application. But I still wanted the backend power of Gradio. Specifically, the ability to call other Spaces via `gradio_client`, queuing for high performance, and host the whole thing on HF Spaces without infrastructure headaches.

## Enter `gradio.Server`

`gradio.Server` extends FastAPI. It gives you the full power of FastAPI (custom routes, middleware, file uploads, any response type) while adding Gradio's API engine on top: queuing, SSE streaming, concurrency control, and `gradio_client` compatibility.

Here's the entire backend for Text Behind Image:

```python
from gradio import Server
from gradio_client import Client, handle_file
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse

app = Server()

bg_client = Client("ysharma/background-removal-copy")


@app.api(name="remove_background")
def remove_background(image_path: FileData) -> str:
    """Remove background from an image. Returns path to transparent PNG."""
    result = bg_client.predict(f=handle_file(image_path["path"]), api_name="/png")
    return result


@app.post("/api/remove-background")
async def remove_bg_endpoint(file: UploadFile = File(...)):
    """FastAPI endpoint for the custom frontend to call."""
    suffix = os.path.splitext(file.filename or ".png")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result_path = bg_client.predict(
            f=handle_file(tmp_path), api_name="/png"
        )
        with open(result_path, "rb") as f:
            fg_b64 = base64.b64encode(f.read()).decode()
        return {"foreground": f"data:image/png;base64,{fg_b64}"}
    finally:
        os.unlink(tmp_path)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


app.launch()
```

That's it. ~40 lines of Python. Let's break down what's happening.

### Two Kinds of Endpoints, One App

The @app.api() decorator registers a Gradio API endpoint, giving you queuing, concurrency limits, and making it callable via `gradio_client`:

```python
from gradio_client import Client, handle_file
client = Client("ysharma/text-behind-image-cp")
result = client.predict(
      image_path=handle_file("photo.jpg"),
      api_name="/remove_background"
  )
```


The `@app.post()` and `@app.get()` decorators are **standard FastAPI routes**, where the custom frontend calls `/api/remove-background` directly with a file upload, and `GET /` serves the HTML page.

Both coexist naturally because `Server` *is* a FastAPI app.

### The Frontend: Pure HTML/CSS/JS

The `index.html` is a self-contained ~700-line web application. No React, no build step, no bundler. Just vanilla HTML with:

- A **three-layer canvas**: background image → text layer → foreground (transparent PNG) stacked with CSS `z-index`
- **Drag-and-drop text positioning** using pointer events
- A **control panel** with 20+ parameters: font family (25+ fonts), size, weight, spacing, color, opacity, background fill, stroke, shadow, 3D extrusion depth and angle, rotation, skew, and full CSS perspective transforms
- **Client-side PNG export** using `<canvas>` compositing

The frontend talks to the backend with a single `fetch()` call:

```javascript
const resp = await fetch('/api/remove-background', {
    method: 'POST',
    body: formData
});
const data = await resp.json();
foregroundLayer.src = data.foreground;  // base64 transparent PNG
```

That's the entire frontend-backend contract. Upload an image, get back a transparent foreground. Everything else, including text rendering, layer compositing, and export, happens in the browser.

## What This Unlocks

Here's what was **not possible** before `gradio.Server`:

| Before | After |
|--------|-------|
| Custom UI meant leaving Gradio entirely | Custom UI *with* Gradio's backend engine |
| No way to serve static HTML from a Gradio app | `@app.get("/")` serves anything |
| `gradio_client` only worked with Gradio component apps | `@app.api()` endpoints are client-compatible |
| Choosing between Gradio's infra and design freedom | You get both |

The mental model shift is simple: **Gradio is no longer just a UI framework. It's a backend framework that happens to also have a great UI system.**

If you want Gradio's UI, you can use `gr.Blocks`, `gr.Interface`, `gr.ChatInterface`. If you want your own UI, use `gradio.Server` and bring whatever frontend you like. Either way, you get Spaces hosting, API queuing, `gradio_client` access, the full HF ecosystem, and more.

## Try It

The app is live on Spaces: **[ysharma/text-behind-image](https://huggingface.co/spaces/ysharma/text-behind-image)**

Upload any photo with a clear subject, and start layering text behind it. Try the 3D extrusion, perspective tilt, and stroke effects, they combine nicely.

## What's Next

This post covered the core idea: `gradio.Server` lets you pair any frontend with Gradio's backend. There's more to explore, including **MCP tool registration** with `@app.mcp.tool()`, **SSE streaming** for real-time updates, **batch processing**, and patterns for building multi-page apps with shared state.

We'll dig into those in a follow-up post. Stay tuned.

## Recommended reading

- Docs: [gradio.Serve](https://www.gradio.app/main/docs/gradio/server)
- Guide: [Server mode](https://www.gradio.app/guides/server-mode) 