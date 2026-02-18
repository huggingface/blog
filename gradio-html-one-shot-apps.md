---
title: "One-Shot Any Web App with Gradio's gr.HTML"
thumbnail: /blog/assets/gradio-html-one-shot-apps/thumbnail.png
authors:
- user: ysharma
- user: hysts
- user: freddyaboulton
---

# One-Shot Any Web App with Gradio's gr.HTML

**Gradio 6** quietly shipped a very powerful feature: **`gr.HTML`** now supports custom templates, scoped CSS, and JavaScript interactivity. Which means you can build pretty much any web component — and Claude can generate the whole thing in one shot: frontend, backend, and state management, all in a single Python file.

We tested this by building different types of apps. Each one is a single Python file, no build step, deployable to Huggingface Spaces in seconds.

## Productivity Apps

**[Pomodoro Timer](https://huggingface.co/spaces/ysharma/pomodoro-timer)**: A focus timer where a pixel-art tree grows as you work. Starts as a seed, sprouts branches, grows leaves. Complete a session and the tree joins your forest. Session tracking, theme switching, break modes — all interactive, all in one file.

The tree animation alone would normally require a separate React component. Here it's just CSS keyframes in `css_template` and state updates in `js_on_load`.

## Business Apps

**[GitHub Contribution Heatmap](https://huggingface.co/spaces/ysharma/github-contribution-heatmap)**: Click any cell to toggle contributions. Multiple color themes. Pattern generators (streaks, seasonal, random). Live stats that update as you edit.

**[Kanban Board](https://huggingface.co/spaces/ysharma/drag-and-drop-kanban-board)**: Full drag-and-drop between columns. Inline editing (double-click any card). Search feature that can filter in real-time. Collapsible columns.

Drag-and-drop usually means pulling in a library. Here it's native HTML5 drag events wired up in `js_on_load`, with state synced back to Python via `trigger('change')`.

## Creative Apps

**[Spin-to-Win Wheel](https://huggingface.co/spaces/ysharma/spin-wheel)**: Smooth CSS animation, rotation state that persists across re-renders. Preset configurations for yes/no decisions, restaurant picking, team selection. You can also add custom spinning segments on the fly.

## ML Apps

This is where `gr.HTML` gets really interesting for ML work: you can build specialized components that can handle your exact output format, then use them like any built-in Gradio component. 

**[Detection Viewer](https://huggingface.co/spaces/hysts-gradio-custom-html/detection-viewer-demo)**: A custom viewer for object detection, instance segmentation, and pose estimation results. Renders bounding boxes, segmentation masks, keypoints, and skeleton connections — all in a reusable `gr.HTML` subclass that plugs directly into your model pipeline.

The community's built some creative components with gr.HTML too:

**[3D Camera Control for Image Editing](https://huggingface.co/spaces/multimodalart/qwen-image-multiple-angles-3d-camera)**: A full Three.js viewport inside a Gradio app! Drag handles to control azimuth, elevation, and distance. Your uploaded image appears in the 3D scene, and the camera parameters feed directly into Qwen's latest image editing model. These kinds of interactive 3D controls would normally require a separate frontend — with Gradio it's just one `gr.HTML` subclass🔥

**[Real-time Speech Transcription](https://huggingface.co/spaces/mistralai/Voxtral-Mini-Realtime)**: Live transcription with Mistral's Voxtral model. The transcript display is a custom `gr.HTML` component with animated status badges, a live WPM counter, and styled output that updates as you speak. Real-time UI feedback without using React!


---

## How It Works

Every `gr.HTML` component takes three templates:

```python
gr.HTML(
    value={"count": 0},
    html_template="<button>Clicked ${value.count} times</button>",
    css_template="button { background: #667eea; color: white; }",
    js_on_load="""
        element.querySelector('button').onclick = () => {
            props.value = { count: props.value.count + 1 };
            trigger('change');
        };
    """
)
```

`${value}` injects Python state. `props.value` updates it from JavaScript. `trigger('change')` syncs back to Python. That's the whole API.

For reusable components, subclass `gr.HTML`:

```python
class Heatmap(gr.HTML):
    def __init__(self, value=None, theme="green", **kwargs):
        super().__init__(
            value=value,
            theme=theme,
            html_template=TEMPLATE,
            css_template=STYLES,
            js_on_load=SCRIPT,
            **kwargs
        )
```

Now `Heatmap()` works like `gr.Image()` or `gr.Slider()` — use it in Blocks, wire up event handlers, whatever you need.

## Why This Matters for Vibe Coding

When you ask Claude to build a custom component, single-file output is everything. No "now create the styles file" or "wire this into your build config." Just one Python file that runs immediately.

The feedback loop becomes: describe what you want → get code → `gradio app.py` → see it working → describe what to fix → repeat. Each cycle takes seconds with [gradio's reload mode](https://www.gradio.app/guides/developing-faster-with-reload-mode).


[Deploy to Spaces](https://www.gradio.app/guides/sharing-your-app#hosting-on-hf-spaces) with `gradio deploy` or [share a temporary link](https://www.gradio.app/guides/sharing-your-app#sharing-demos) with `demo.launch(share=True)`. Within a few seconds from an idea to a live app.

---


Gradio ships with 32 interactive components, but sometimes your perfect AI web app needs something special. That's where `gr.HTML` comes in. 

If you’ve got an idea, try building it with `gr.HTML`: describe what you want to Claude, generate the code, run it. You might be surprised what you can ship in 5 minutes.


Suggested reading:
- Gradio guide: [Custom Components with gr.HTML](https://www.gradio.app/guides/custom-HTML-components)
- API docs: [gr.HTML](https://www.gradio.app/docs/gradio/html)