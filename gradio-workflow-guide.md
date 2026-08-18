---
title: "Build Anything with gr.Workflow: The Complete Guide"
thumbnail: /blog/assets/gradio-workflow-guide/thumbnail.png
authors:
- user: ysharma
---

# Build Anything with gr.Workflow: The Complete Guide

Most interesting and useful AI apps are **pipelines**: you generate an image, then cut out its background if you want to; write a script, and then generate a voice if you like it, change the voice and keep the script same. We usually wire these steps together in Python, and the moment something looks off we're back to debugging to find *which* step produced the weird value.

**`gr.Workflow`, built right into Gradio, makes the pipeline *the interface*!** You describe your steps as a graph of typed nodes, and Gradio serves a drag-and-drop **canvas** where every node is runnable, every intermediate result is visible and cached, and the whole thing is simultaneously a **REST API** and a one-command **deploy** to Hugging Face Spaces.

This guide is designed to be **complete**: read it top to bottom or point your AI agent to it, and you'll be able to build pipelines for any use case using `gr.Workflow`. We have covered end-to-end examples for pure-Python pipeline, a call to a Hugging Face model, a call to another Gradio Space, a Hub Dataset, a multi-output dashboard, and a multi-endpoint API. Every example is a **live Space** you can open, inspect, and duplicate, and every code snippet is short enough to read in full. We have included [Complete schema reference](#complete-schema-reference) and [Build checklist](#build-checklist-for-agents) for your AI agents.

### The official gradio guide

Gradio ships an official **[gr.Workflow guide](https://gradio.app/guides/workflows)**, and you should keep it open, it's the **reference** source for: the exact API, the JSON schema, and the operator-kind and port-type information. This blog post is like the **applied companion** that adds the parts and guide you with examples. This page can also stand alone for a human or an AI agent.

## Every example in this guide is live

| # | Demo | What it teaches | Space |
|---|------|-----------------|-------|
| A | Bind + edges | The No-JSON path | [gr-workflow-concept-a-bind-edges](https://huggingface.co/spaces/ysharma/gr-workflow-concept-a-bind-edges) |
| B | References & subjects | The smallest schema-v2 graph | [gr-workflow-concept-b-references-subjects](https://huggingface.co/spaces/ysharma/gr-workflow-concept-b-references-subjects) |
| C | Model node | Call a model on HF Inference Providers | [gr-workflow-concept-c-model-node](https://huggingface.co/spaces/ysharma/gr-workflow-concept-c-model-node) |
| D | Space node | Call another Gradio Space | [gr-workflow-concept-d-space-node](https://huggingface.co/spaces/ysharma/gr-workflow-concept-d-space-node) |
| E | Multi-endpoint API | Two pipelines → two REST endpoints | [gr-workflow-concept-e-multi-endpoint](https://huggingface.co/spaces/ysharma/gr-workflow-concept-e-multi-endpoint) |
| 01 | Generative Art Lab | Fan-out, image ports | [gr-workflow-01-generative-art-lab](https://huggingface.co/spaces/ysharma/gr-workflow-01-generative-art-lab) |
| 02 | Story Forge | A linear chain, no JSON | [gr-workflow-02-story-forge](https://huggingface.co/spaces/ysharma/gr-workflow-02-story-forge) |
| 03 | Data Detective | Mixed output types (dataframe/json/image) | [gr-workflow-03-data-detective](https://huggingface.co/spaces/ysharma/gr-workflow-03-data-detective) |
| 04 | AI Media Studio | Model + space + fn nodes together | [gr-workflow-04-ai-media-studio](https://huggingface.co/spaces/ysharma/gr-workflow-04-ai-media-studio) |

---

## The one mental model

A `gr.Workflow` is a **directed graph**. Every node has exactly one of **three roles**, and this is the entire conceptual universe:

| Role | Plain meaning | Position |
|------|---------------|----------|
| **reference** | an **input** the user supplies (text, a file, a number) | the left edge of the pipeline |
| **operator** | a **step that does work** (your function, a model, a Space, a dataset row) | the middle |
| **subject** | an **output** the pipeline produces | the right edge |

- Data flows left → right along **edges**. 
- An edge connects one node's **output port** to another node's **input port**. 
- Ports are **typed** (`text`, `number`, `image`, `audio`, …). 
That's the whole model, everything else is built on `references → operators → subjects`.

There are **two styles to author** the same graph, and we'll use both. Style 1 is a convenience that *generates* Style 2 for you :

1. **No-JSON Python**: handles Gradio functions (with `bind=`) and how to wire them (with `edges=`). Fastest, perfect for pure-Python pipelines.
2. **A committed `workflow.json`**: full control over media types, layout, and model/space/dataset nodes.


We’ll start with the easiest concepts to understand and build, then gradually work our way up to the full schema implementation, with the learning divided into levels based on complexity.

---

## Level 0: the empty canvas

```python
import gradio as gr
gr.Workflow().launch()
```

This opens an empty canvas. You drag resources from the sidebar (Spaces, models, datasets, your own functions), connect ports by dragging between them, and hit **Run**. When you do, Gradio saves a `workflow.json` capturing what you built. 

---

## Level 1: bind a python function (one node)

```python
import gradio as gr

def summarize(text: str) -> str:
    return text[:200]

gr.Workflow(bind=[summarize]).launch()
```

What happens:

- `bind=[summarize]` turns the function into **an operator node**.
- Its **parameters become input ports**; its **return value becomes an output port**.
- **Type hints set the port types.** The rule: `int`/`float` → `number` port, `bool` → `boolean` port, everything else → `text`.

`bind` accepts either a **list** (node names default to `fn.__name__`) or a **dict** to rename the node:

```python
gr.Workflow(bind={"My Summarizer": summarize}).launch()
```

---

## Level 2: wire functions together with `edges`

`edges` is a list of `("from", "to")` tuples connecting one node's first output to the next node's first input. This is **Concept A**: [open it live ›](https://huggingface.co/spaces/ysharma/gr-workflow-concept-a-bind-edges)

```python
import gradio as gr

def shout(text: str) -> str:
    return text.upper()

def emphasize(text: str, times: int) -> str:
    return text + ("!" * times)

demo = gr.Workflow(
    bind=[shout, emphasize],
    edges=[("shout", "emphasize")],   # shout's output → emphasize's first input (`text`)
)
demo.launch()
```

Edge rules:

- `("shout", "emphasize")` connects `shout`'s **first output** to `emphasize`'s **first input port** (`text`).
- To target a **specific port**, use dotted syntax: `("shout", "emphasize.text")`.
- `times` has no incoming edge, so it stays a free field the user fills in.

### A real chain: Story Forge

**App 02, Story Forge**: [open it live ›](https://huggingface.co/spaces/ysharma/gr-workflow-02-story-forge) It is a four-stage writers'-room pipeline built entirely with `bind` + `edges`, no JSON at all. `outline → draft → add_tone → polish`:

```python
import random, re
import gradio as gr

GENRES = { ... }   # flavor banks per genre (see the Space for the full data)

def outline(premise: str, genre: str, seed: int) -> str:
    """Beat 1 · turn a one-line premise into a five-beat outline."""
    g = GENRES.get(genre, GENRES["Fantasy"])
    r = random.Random(f"{premise}|{genre}|{seed}")
    place, force, agent = r.choice(g["place"]), r.choice(g["force"]), r.choice(g["agent"])
    premise = (premise or "someone wants something they cannot have").strip()
    beats = [
        f"HOOK — In {place}, we meet {agent}.",
        f"SPARK — {premise[0].upper() + premise[1:]}.",
        f"TURN — {force.capitalize()} forces a choice.",
        f"CRISIS — The cost of the plan comes due.",
        f"RESOLVE — A quiet, earned ending back in {place}.",
    ]
    return "\n".join(f"{i}. {b}" for i, b in enumerate(beats, 1))

def draft(outline: str) -> str:
    """Beat 2 · expand each outline beat into a short paragraph."""
    ...

def add_tone(draft: str, tone: str) -> str:
    """Beat 3 · restyle the draft. Try: playful · noir · epic."""
    ...

def polish(story: str) -> str:
    """Beat 4 · add a title and tidy up (Markdown out)."""
    return f"# The Forged Tale\n\n{story.strip()}"

demo = gr.Workflow(
    bind=[outline, draft, add_tone, polish],
    edges=[
        ("outline", "draft"),     # outline's output → draft's first input
        ("draft", "add_tone"),    # draft's output   → add_tone's first input (`draft`)
        ("add_tone", "polish"),   # toned text       → polish
    ],
)
demo.launch()
```

Every function becomes a node; each edge chains one output to the next input. Extra parameters, like, `genre`, `seed`, `tone`, stay as editable fields on their own nodes. Change any node's input and **only the downstream nodes recompute**. That incremental re-execution is the whole point of the canvas.

> On first launch Gradio writes a `workflow.json` next to your script and lays out the nodes. `edges=` describes the wiring used to **generate** that graph; once a `workflow.json` exists the committed graph is the source of truth. So `bind`+`edges` is how you *author* a fresh graph, and a committed JSON is how you *ship* one.

---

## Level 3: the real schema: references, operators, subjects

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/concept B.mp4"></video>

Now we drop to Style 2 and author `workflow.json` directly. **Concept B**: [open it live ›](https://huggingface.co/spaces/ysharma/gr-workflow-concept-b-references-subjects) It is the smallest complete style 2 graph: `reference(text) → fn greet → subject(text)`.

A clean pattern is to build the graph as a Python dict, write it to JSON, then load it:

```python
import json, os
import gradio as gr

def greet(name: str) -> str:
    return f"Hello, {name}! 👋 Welcome to gr.Workflow."

GRAPH = {
    "schema_version": "2",
    "name": "Greeting",
    "references": [
        {"id": "ref_name", "role": "reference", "label": "Name", "asset_type": "text",
         "inputs":  [{"id": "in",  "label": "Name", "type": "text"}],
         "outputs": [{"id": "out", "label": "Name", "type": "text"}],
         "data": {"out": "Ada"}}          # default value shown on the canvas
    ],
    "operators": [
        {"id": "op_greet", "role": "operator", "kind": "fn", "fn": "greet", "label": "greet",
         "inputs":  [{"id": "in_name", "label": "name",     "type": "text", "required": True}],
         "outputs": [{"id": "out_0",   "label": "greeting", "type": "text"}]}
    ],
    "subjects": [
        {"id": "sub_out", "role": "subject", "label": "Greeting", "asset_type": "text",
         "inputs":  [{"id": "in",  "label": "Greeting", "type": "text"}],
         "outputs": [{"id": "out", "label": "Greeting", "type": "text"}]}
    ],
    "edges": [
        {"id": "e1", "from_node_id": "ref_name", "from_port_id": "out",
         "to_node_id": "op_greet", "to_port_id": "in_name", "type": "text"},
        {"id": "e2", "from_node_id": "op_greet", "from_port_id": "out_0",
         "to_node_id": "sub_out",  "to_port_id": "in", "type": "text"},
    ],
}

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(HERE, "greeting.json")     # resolve next to this file
with open(GRAPH_PATH, "w", encoding="utf-8") as f:
    json.dump(GRAPH, f, indent=2)

demo = gr.Workflow(GRAPH_PATH, bind={"greet": greet})   # bind links "fn":"greet" → the function
demo.launch()
```

### Anatomy of a node

Every node carries:

- **`id`**: unique string, referenced by edges.
- **`role`**: `"reference"` / `"operator"` / `"subject"` (matches the collection it lives in).
- **`label`**: display name on the canvas.
- **`inputs` / `outputs`**: lists of **ports**. Each port is `{"id", "label", "type"}`.
- **`data`**: a map of **default / literal values by port id**. An input port with **no incoming edge** takes its value from `data` (or the port default). This is how you bake a constant into the graph.
- Optional **`x`, `y`, `width`, `height`**: canvas layout. Omit them and Gradio auto-arranges.
- References and subjects also carry an **`asset_type`** describing the media they hold.


Two connective rules worth stating explicitly:

- The **`bind` dict key must match the operator's `"fn"` value** (`"greet"` here), not the label.
- `gr.Workflow("greeting.json")` resolves relative to the current working directory. Pass a path joined to `__file__` (as above) so it runs from anywhere.

---

## Level 4: the four operator kinds

`kind` is what makes an operator interesting. There are exactly four, and three of them need **zero client code**:

| `kind` | What it runs | Required fields | Needs `bind`? |
|--------|--------------|-----------------|---------------|
| `"fn"` | your Python function | `fn` (matches a `bind` key) | ✅ yes |
| `"model"` | a Hugging Face model via **Inference Providers** | `model_id` + `endpoint` or `pipeline_tag` | ❌ no |
| `"space"` | a **Gradio Space** via `gradio_client` | `space_id` + `endpoint` | ❌ no |
| `"dataset"` | one row from a **Hugging Face dataset** | `dataset_id`, `dataset_config`, `dataset_split`, `row_index` | ❌ no |

The next four levels take these one at a time.

---

## Level 5 : a `model` node (Hugging Face Inference Providers)

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/Concept C - model node.mp4"></video>

An operator with `kind: "model"` calls a model on HF Inference Providers, meaning no `InferenceClient` code. 
**Concept C**: [open it live ›](https://huggingface.co/spaces/ysharma/gr-workflow-concept-c-model-node), shows `reference(text) → FLUX.1-schnell → subject(image)`:

```json
{
  "id": "op_flux", "role": "operator", "kind": "model",
  "model_id": "black-forest-labs/FLUX.1-schnell",
  "pipeline_tag": "text-to-image", "endpoint": "text_to_image", "provider": "auto",
  "inputs":  [{"id": "prompt", "label": "Prompt", "type": "text", "required": true}],
  "outputs": [{"id": "out_0", "label": "Image", "type": "image", "output_index": 0}]
}
```

A few things to keep in mind:

- If **`endpoint` present** (e.g. `"text_to_image"`): inputs are sent as **named kwargs**: the port `id: "prompt"` becomes `prompt=...`. So your port `id`s should match the model method's argument names.
- If **`endpoint` absent**: `pipeline_tag` routes inputs **positionally**.
- `provider: "auto"` lets HF pick the serving provider; set it to a specific provider name to pin one.
- `output_index` selects a value from a multi-value response.

There's no `bind` on the launch line. A model node is self-contained:

```python
demo = gr.Workflow(GRAPH_PATH)   # no bind needed; the model node calls HF for you
```

Running a model node requires a Hugging Face token. On a Space, enable OAuth so each visitor runs under their own sign-in (see [Deploying](#level-11--deploy-to-a-space-in-one-command)).

---

## Level 6: a `space` node (call any Gradio Space)

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/Concept D - space node.mp4"></video>

An operator with `kind: "space"` calls another Gradio Space's API through `gradio_client`. Set `space_id` and `endpoint` (its `api_name`, shown on the Space's **"Use via API"** panel); inputs are passed **positionally** in port order, and file inputs/outputs are handled for you. 
**Concept D**: [open it live ›](https://huggingface.co/spaces/ysharma/gr-workflow-concept-d-space-node) remove an image's background:

```json
{
  "id": "op_bg", "role": "operator", "kind": "space",
  "space_id": "hf-applications/background-removal", "endpoint": "/image",
  "inputs":  [{"id": "in_image", "label": "Image", "type": "image", "required": true}],
  "outputs": [{"id": "out_0", "label": "Cutout", "type": "image", "output_index": 1}]
}
```

The `/image` endpoint returns `(original, cutout)`, so the output port sets `output_index: 1` to keep just the cutout. `output_index` is the general mechanism for picking one value out of a multi-output response. It works the same for `model` and `space` nodes.

---

## Level 7: a `dataset` node (stream a dataset row from the Hub)

An operator with `kind: "dataset"` pulls **one row per run** from a Hugging Face dataset, selected by `row_index`. It's the easiest way to demo a pipeline over a fixed corpus without an upload node:

```json
{
  "id": "op_row", "role": "operator", "kind": "dataset",
  "dataset_id": "stanfordnlp/imdb",
  "dataset_config": "plain_text",
  "dataset_split": "train",
  "row_index": 0,
  "label": "IMDB sample",
  "inputs":  [],
  "outputs": [
    {"id": "text",  "label": "text",  "type": "text"},
    {"id": "label", "label": "label", "type": "number"}
  ]
}
```

Each output port maps to a **column** of the selected row. Wire those output ports into downstream operators exactly like any other node.

---

## Level 8: port types, fan-out, and mixed outputs

### Supported full port-type set

```
text · number · boolean · image · audio · video · file · gallery · json · model3d · any
```

`any` is a **compatibility fallback**, it can connect to any other port type (it turns off type-checking on that wire, so use it deliberately).

### The serialization contract for `fn` nodes

Because bound functions exchange values as JSON on the canvas, `fn` operators follow a small, predictable contract for the richer types:

| Port type | An `fn` returns… |
|-----------|------------------|
| `image` | a base64 `data:image/png;base64,...` string |
| `dataframe` | a `{"headers": [...], "data": [[...]]}` dict |
| `json` | any plain dict / list |
| `audio` / `video` / `file` | a path or URL string |
| `text` / `number` / `boolean` | the native value |

`model` and `space` nodes handle files and media for you automatically. The contract above is specifically how *your own* Python functions handle rich values to a port.

### Fan-out: one output → many operators

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/App 01.mp4"></video>

An output port can feed **many** downstream nodes. 
**App 01, Generative Art Lab**: [open it live ›](https://huggingface.co/spaces/ysharma/gr-workflow-01-generative-art-lab) It generates one base image and fans it out to three filters, plus the base itself, giving four image outputs:

The functions exchange images as `data:` URIs so they flow node → node with zero file plumbing:

```python
import base64, io
from PIL import Image, ImageFilter, ImageOps

def _to_uri(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _from_uri(value) -> Image.Image:
    if isinstance(value, dict):                     # {path|url} from a component
        value = value.get("path") or value.get("url") or ""
    if isinstance(value, str) and value.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")
    return Image.open(value).convert("RGB")

def posterize(image: str) -> str:
    im = ImageOps.posterize(_from_uri(image), 3)
    return _to_uri(ImageOps.autocontrast(im, cutoff=1))

def edge_glow(image: str) -> str:
    im = _from_uri(image)
    edges = im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(1.2))
    dark = ImageOps.autocontrast(im).point(lambda p: int(p * 0.35))
    return _to_uri(Image.blend(dark, ImageOps.autocontrast(edges), 0.75))

# ... generate() and kaleidoscope() similarly return _to_uri(...)

BIND = {"generate": generate, "posterize": posterize,
        "edge_glow": edge_glow, "kaleidoscope": kaleidoscope}
demo = gr.Workflow("workflow.json", bind=BIND)
```

The fan-out lives in the JSON. The `generate` node's single output port has an edge to each filter node **and** to the base subject:

```json
"edges": [
  {"id": "e5", "from_node_id": "op_generate", "from_port_id": "out_0", "to_node_id": "op_poster",  "to_port_id": "in_image", "type": "image"},
  {"id": "e6", "from_node_id": "op_generate", "from_port_id": "out_0", "to_node_id": "op_glow",    "to_port_id": "in_image", "type": "image"},
  {"id": "e7", "from_node_id": "op_generate", "from_port_id": "out_0", "to_node_id": "op_kaleido", "to_port_id": "in_image", "type": "image"},
  {"id": "e8", "from_node_id": "op_generate", "from_port_id": "out_0", "to_node_id": "sub_base",   "to_port_id": "in",       "type": "image"}
]
```

### Mixed output types in one graph

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/App 03.mp4"></video>

Different subjects can render different types. 

**App 03, Data Detective**: [open it live ›](https://huggingface.co/spaces/ysharma/gr-workflow-03-data-detective) It takes one CSV `file` reference and fans it out to four analysts that return four different port types:

```
                   ┌─▶ preview        ─▶ [Preview]        (dataframe)
[CSV file] ─▶──────┼─▶ summary_stats  ─▶ [Statistics]     (dataframe)
                   ├─▶ missing_report ─▶ [Missing values] (json)
                   └─▶ correlation    ─▶ [Heatmap]        (image)
```

```python
import base64, io
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def _read(file) -> pd.DataFrame:
    if isinstance(file, dict):
        file = file.get("path") or file.get("url")
    return pd.read_csv(file)

def _table(df: pd.DataFrame) -> dict:
    """A JSON-safe {headers, data} payload for a `dataframe` port."""
    df = df.astype(object).where(pd.notna(df), None)
    data = [[(x.item() if hasattr(x, "item") else x) for x in row] for row in df.values.tolist()]
    return {"headers": [str(c) for c in df.columns], "data": data}

def preview(file: str) -> dict:            # → dataframe port
    return _table(_read(file).head(25))

def missing_report(file: str) -> dict:     # → json port
    df = _read(file); na = df.isna().sum()
    return {"rows": int(len(df)), "columns": int(df.shape[1]),
            "total_missing": int(na.sum()),
            "missing_by_column": {c: int(v) for c, v in na.items() if v > 0} or "none 🎉"}

def correlation(file: str) -> str:         # → image port (data: URI)
    corr = _read(file).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.1 * len(corr) + 2, 1.1 * len(corr) + 1.5))
    ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=110); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
```

Each subject's `asset_type` (and the matching port `type`) tells the canvas how to render the result: a table, a JSON tree, an image.

---

## Level 9: a workflow is also a REST API

Every Gradio workflow is an API for free! Each **weakly-connected pipeline** ending in a subject becomes **one REST endpoint**, named after that subject's label (a subject labeled `Output Image` → `/output_image`). Free references become the endpoint's parameters.

**Concept E**: [open it live ›](https://huggingface.co/spaces/ysharma/gr-workflow-concept-e-multi-endpoint) It puts two independent pipelines in one graph, so it exposes **two** endpoints, `/word_count` and `/fahrenheit`:

```python
def word_count(text: str) -> int:
    return len(text.split())

def to_fahrenheit(celsius: float) -> float:
    return round(celsius * 9 / 5 + 32, 1)

# ...two references → two fn operators → two subjects, with no edges between the pairs...
demo = gr.Workflow(GRAPH_PATH, bind={"word_count": word_count, "to_fahrenheit": to_fahrenheit})
```

Call it with the standard Gradio client:

```python
from gradio_client import Client

client = Client("ysharma/gr-workflow-concept-e-multi-endpoint")
client.view_api()                                  # discover endpoints + signatures
client.predict("hello there friend", api_name="/word_count")   # → 3
client.predict(20, api_name="/fahrenheit")                     # → 68.0
```

…or over plain HTTP:

```bash
curl -s https://ysharma-gr-workflow-concept-e-multi-endpoint.hf.space/gradio_api/call/word_count \
  -H "Content-Type: application/json" -d '{"data": ["hello there friend"]}'
```

You can also introspect the endpoint schema in Python before launch:

```python
import json
from gradio.workflow_api import WorkflowGraph, describe_workflow_api

for ep in describe_workflow_api(WorkflowGraph.from_json(json.dumps(GRAPH))):
    print(ep["api_name"], [p["type"] for p in ep["parameters"]], "→", [r["type"] for r in ep["returns"]])
```

Because subject labels become endpoint names, **label your subjects deliberately** . The label becomes your public API surface.

---

## Level 10, the real payoff: model + space + fn in one graph

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/App 04.mp4"></video>

This is what `gr.Workflow` is really for! 

**App 04, AI Media Studio**: [open it live ›](https://huggingface.co/spaces/ysharma/gr-workflow-04-ai-media-studio) It chains all three operator kinds across three pipelines in one graph:

```
A) Sticker    [prompt] ─▶ (model) FLUX.1-schnell ─▶ (space) background-removal ─▶ 🪄 Sticker
B) Voiceover  [topic]  ─▶ (fn) make_line ─▶ (space) MeloTTS ─▶ 🎧 Voiceover
C) Title      [topic]  ─▶ (fn) make_prompt ─▶ (model) Qwen2.5-7B ─▶ ✨ Episode title
```

The Python is just two tiny text helpers, every heavy step is either a `model` or a `space` node in the JSON:

```python
import os
import gradio as gr

def make_line(topic: str) -> str:
    """Turn a topic into one spoken-sounding line for TTS."""
    topic = (topic or "something wonderful").strip().rstrip(".")
    return f"Welcome to the show. Today, we're diving into {topic}. Let's get into it!"

def make_prompt(topic: str) -> str:
    """Wrap a topic into an instruction for the language model."""
    return f"Write ONE catchy, upbeat podcast episode title (max 8 words, no quotes) about: {topic}"

demo = gr.Workflow("workflow.json", bind={"make_line": make_line, "make_prompt": make_prompt})
```

Pipeline A shows the composition clearly: a `model` node hands over its image straight to a `space` node:

```json
{
  "operators": [
    {"id": "op_flux", "kind": "model", "role": "operator",
     "model_id": "black-forest-labs/FLUX.1-schnell",
     "pipeline_tag": "text-to-image", "endpoint": "text_to_image",
     "inputs":  [{"id": "prompt", "type": "text", "required": true}],
     "outputs": [{"id": "out_0", "type": "image", "output_index": 0}]},

    {"id": "op_bg", "kind": "space", "role": "operator",
     "space_id": "hf-applications/background-removal", "endpoint": "/image",
     "inputs":  [{"id": "in_image", "type": "image", "required": true}],
     "outputs": [{"id": "out_0", "type": "image", "output_index": 1}]}
  ],
  "edges": [
    {"id": "e2", "from_node_id": "op_flux", "from_port_id": "out_0",
     "to_node_id": "op_bg", "to_port_id": "in_image", "type": "image"}
  ]
}
```

Notice how the reusable idioms compose:

- The TTS node passes extra constant arguments (`speaker`, `speed`, `language`) via its `data` map, since those input ports have no incoming edge.
- The LLM node uses `pipeline_tag: "text-generation"` with **no** `endpoint`, so its single text input routes positionally.
- The background-removal node keeps `output_index: 1` (the cutout) from a two-value response.

That's a production-shaped AI pipeline: Two model calls, two Space calls, two helper functions. Everything assembled almost entirely in declarative JSON.

---

## Level 11: deploy to a Space in one command

A `gr.Workflow` app is a normal Gradio app, so it deploys like one. From a folder containing your `app.py`:

```bash
gradio deploy
```

Or lay the folder out as a Space repo yourself. Each of the demos in this guide is exactly this shape:

```
my-workflow/
├── README.md          # the Space card (YAML front matter below)
├── requirements.txt   # extra deps beyond gradio (numpy, pandas, …)
├── app.py             # gr.Workflow(...).launch()
└── workflow.json      # for JSON-authored apps (omit for pure bind+edges)
```

A minimal Space card:

```yaml
---
title: My Workflow
emoji: 🔀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
---
```

### Running model / space nodes on a Space

`model` and `space` nodes call Hugging Face, which needs a token. The clean way to serve them publicly is **OAuth**: add `hf_oauth` to the card and each visitor runs nodes under their **own** sign-in. No need to worry about shared secrets to manage or rotate. Sample README file:

```yaml
---
title: AI Media Studio
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
hf_oauth: true
hf_oauth_scopes:
  - inference-api
---
```

Visitors click **Sign in with Hugging Face** on the canvas before running. Pure-Python workflows (only `fn` nodes) need none of this, they run directly.

> `gr.Workflow` requires Gradio **6.17+**

---

## Level 12: compose workflows inside workflows

A finished workflow is itself a reusable unit: a workflow can be **nested inside a larger workflow** as a node. This is how you build big applications out of small, individually-tested pipelines. Validate the "generate sticker" workflow on its own, then drop it into a bigger "make a whole content pack" workflow as a single block. The same three-role model applies at every level of nesting.

---

## Choosing your tools: a decision guide

How to choose between the two options to build your Workflow?

### `bind` + `edges` vs a committed `workflow.json`

| Reach for **`bind` + `edges`** (No-JSON) when… | Reach for a **`workflow.json`** when… |
|---|---|
| every step is a Python function | you need `model`, `space`, or `dataset` nodes (`edges=` only wires `fn` nodes) |
| the wiring is a simple chain or fan-out you can name in tuples | you want a saved layout, media ports, or precise port ids |
| you're prototyping and want the graph generated for you | you're shipping and want the topology under version control |

A good workflow is to *start* with `bind`+`edges`, let Gradio generate the `workflow.json`, then commit that file and evolve it by hand (or on the canvas). 


### Which operator `kind`?

| Use `kind` … | when the step is… |
|---|---|
| **`fn`** | your own logic, like, parsing, formatting, glue, a local model, anything you can write in Python |
| **`model`** | inference on a Hub model you don't want to host, like, image gen, LLM, ASR, embeddings, etc (runs on **Inference Providers**) |
| **`space`** | something already built and deployed as a **Gradio Space**. Reuse it instead of reimplementing |
| **`dataset`** | you need example rows from the Hub as input, without an upload step |



### `gr.Workflow` vs `gr.Blocks`

A workflow is the right shape when your app **is a pipeline**, a graph of typed steps where the interesting thing is *the intermediate values and which step made them*. 
Choose plain `gr.Blocks`/`gr.Interface` instead when you want a bespoke, hand-laid-out UI, custom event logic, or interactions that aren't "data flowing through nodes." 
Workflows trade layout control for a runnable, inspectable, auto-API'd graph.

---

## The pattern cookbook

Five reusable topologies. Once you can spot these, most real apps are just combinations of them.

### 1. Linear chain: `A → B → C`

The writers'-room pipeline. Each step transforms the previous one's output. **[Story Forge](https://huggingface.co/spaces/ysharma/gr-workflow-02-story-forge)** (Level 2). In No-JSON edges will look like: `edges=[("a","b"), ("b","c")]`.

### 2. Fan-out: one output → many operators

One source feeds several independent branches that run in parallel on the canvas. **[Generative Art Lab](https://huggingface.co/spaces/ysharma/gr-workflow-01-generative-art-lab)** and **[Data Detective](https://huggingface.co/spaces/ysharma/gr-workflow-03-data-detective)** (Level 8). Draw one edge from the same output port to each consumer.

### 3. Fan-in / merge: many outputs → one operator

A downstream operator has **two or more input ports**, each fed by a different upstream; it runs once **both** inputs are ready.

```python
def combine(a: str, b: str) -> str:
    return f"{a}\n\n---\n\n{b}"
```

No-JSON, using dotted target ports to pick which input each edge feeds:

```python
gr.Workflow(
    bind=[left, right, combine],
    edges=[("left", "combine.a"), ("right", "combine.b")],
).launch()
```

Or in `workflow.json`, two edges landing on two ports of the same node:

```json
{
  "id": "op_combine", "role": "operator", "kind": "fn", "fn": "combine",
  "inputs": [
    {"id": "in_a", "label": "a", "type": "text", "required": true},
    {"id": "in_b", "label": "b", "type": "text", "required": true}
  ],
  "outputs": [{"id": "out_0", "label": "merged", "type": "text"}]
}
```
```json
"edges": [
  {"id": "eA", "from_node_id": "op_left",  "from_port_id": "out_0", "to_node_id": "op_combine", "to_port_id": "in_a", "type": "text"},
  {"id": "eB", "from_node_id": "op_right", "from_port_id": "out_0", "to_node_id": "op_combine", "to_port_id": "in_b", "type": "text"}
]
```

### 4. Constant injection: a port with no edge

Any input port left unconnected takes its value from the node's `data` map. Use it to bake in configuration, for example, the TTS node in **[AI Media Studio](https://huggingface.co/spaces/ysharma/gr-workflow-04-ai-media-studio)** pins `speaker`, `speed`, and `language` this way while only its `text` port is wired.

```json
{ "id": "op_tts", "kind": "space", "space_id": "mrfakename/MeloTTS", "endpoint": "/synthesize",
  "inputs": [
    {"id": "in_text", "type": "text", "required": true},
    {"id": "in_speaker", "type": "text"}, {"id": "in_speed", "type": "number"}, {"id": "in_language", "type": "text"}
  ],
  "data": {"in_speaker": "EN-US", "in_speed": 1.0, "in_language": "EN"} }
```

### 5. Model → space handoff: HF ecosystem in two nodes

An Inference-Providers `model` node hands its output straight to a `space` node. **[AI Media Studio](https://huggingface.co/spaces/ysharma/gr-workflow-04-ai-media-studio)**'s sticker pipeline: `FLUX.1-schnell (model) → background-removal (space)`. This is the highest-leverage pattern in the whole system, where just two lines of JSON compose two hosted services into a new one! 

### Bonus: Independent multi-pipeline. One graph, many endpoints.

Put several unconnected pipelines in one graph and each becomes its own REST endpoint. **Concept E** (Level 9). It's not a wiring pattern so much as a *packaging* one: ship related tools as a single deployable, multi-endpoint API.

---

## What actually re-runs: the execution & caching model

The canvas isn't just a diagram, it has a precise execution model, and understanding it is what makes workflows pleasant to debug.

- **Every node's output is cached** after it runs, and rendered in place on the canvas.
- **Editing a node's input invalidates that node and everything downstream of it, and nothing else.** For example, change the `tone` field on Story Forge's `add_tone` node and `polish` re-runs, while `outline` and `draft` do not. Independent branches are untouched.
- **Fan-in waits.** A merge node runs only once *all* its wired input ports have values, so a `combine` node fires after both upstreams complete.
- **Same-depth operators run in parallel on the canvas; the REST API runs the graph sequentially.** So the interactive canvas is your fast feedback loop, while `/endpoint` calls give you deterministic, ordered execution.

This is the real reason to build a multi-step AI app as a workflow. When something looks wrong, the question is always *"which step, and what did it output?"*, and a workflow answers it visually. Every intermediate is visible and cached, and you can re-run a single node with a tweaked input instead of re-running the whole pipeline and squinting at logs. Your pipeline stops being a black box and becomes something you can poke, one node at a time.

---

## Complete schema reference

Everything an implementation (human or agent) needs to emit a valid `workflow.json`.

### Top level

```json
{
  "schema_version": "2",
  "name": "string",
  "references": [ /* nodes, role="reference" */ ],
  "operators":  [ /* nodes, role="operator" */ ],
  "subjects":   [ /* nodes, role="subject" */ ],
  "edges":      [ /* edges */ ]
}
```

### Node (common fields)

| Field | Applies to | Meaning |
|-------|-----------|---------|
| `id` | all | unique node id (edges reference it) |
| `role` | all | `"reference"` \| `"operator"` \| `"subject"` |
| `label` | all | display name (subject labels become API endpoint names) |
| `inputs` | all | list of input ports |
| `outputs` | all | list of output ports |
| `data` | all | map of `port_id → default/literal value`; feeds any input port with no incoming edge |
| `asset_type` | reference, subject | the media type the node holds |
| `kind` | operator | `"fn"` \| `"model"` \| `"space"` \| `"dataset"` |
| `x`,`y`,`width`,`height` | all (optional) | canvas layout; omit to auto-arrange |

### Operator-kind–specific fields

| `kind` | Fields |
|--------|--------|
| `"fn"` | `fn`; string key matching an entry in `bind={...}` |
| `"model"` | `model_id`; `endpoint` (named-kwargs call) **or** `pipeline_tag` (positional); optional `provider` (default `"auto"`) |
| `"space"` | `space_id`; `endpoint` (the Space's `api_name`); inputs passed positionally in port order |
| `"dataset"` | `dataset_id`, `dataset_config`, `dataset_split`, `row_index`; output ports map to row columns |

### Port object

| Field | Applies to | Meaning |
|-------|-----------|---------|
| `id` | all ports | unique within the node |
| `label` | all ports | display name |
| `type` | all ports | one of the port types below |
| `required` | input ports (optional) | `true` marks a mandatory input |
| `output_index` | output ports (optional) | which value of a multi-value response feeds this port |

### Edge object

```json
{
  "id": "e1",
  "from_node_id": "…", "from_port_id": "…",
  "to_node_id":   "…", "to_port_id":   "…",
  "type": "text"
}
```

### Port types

`text` · `number` · `boolean` · `image` · `audio` · `video` · `file` · `gallery` · `json` · `model3d` · `any`

### The Python API

```python
gr.Workflow(
    graph=None,        # path to a workflow.json (positional: gr.Workflow("workflow.json"))
    bind=None,         # list[fn]  or  {name: fn}  supplies implementations for kind="fn" nodes
    edges=None,        # list of ("from","to") or ("from","to.port") tuples (No-JSON authoring)
)
```

- `bind` list → node names default to `fn.__name__`; `bind` dict → explicit names that must match each operator's `"fn"`.
- Type hints on bound functions set port types: `int`/`float` → `number`, `bool` → `boolean`, else `text`.
- `edges` authors a fresh graph; a committed `workflow.json` is the source of truth once present.

---

## Build checklist for agents

To build a `gr.Workflow` for a new use case:

1. **Sketch the graph**: list inputs (references), steps (operators), outputs (subjects).
2. **Pick an authoring style**: pure-Python pipeline → `bind=[...]`, `edges=[...]`. Anything with media types, model/space/dataset nodes, or a saved layout → a `workflow.json`.
3. **Choose each operator's `kind`**: `fn` (your code), `model` (HF Inference Providers), `space` (a Gradio Space), `dataset` (a Hub row).
4. **Type every port** from the port-type list; make sure edge `type`s match the ports they connect.
5. **For `fn` nodes, honor the serialization contract**: image → `data:` URI, dataframe → `{headers, data}`, json → dict/list.
6. **Feed constants via `data`** on any input port that has no incoming edge (e.g. a model's `speaker`/`language`).
7. **Use `output_index`** to select one value from a multi-output model/space response.
8. **Name subjects deliberately**, those labels become your REST endpoints.
9. **Ship it**: `app.py` + `requirements.txt` + `README.md` (Space card, `sdk: gradio`, `sdk_version: 6.24.0`) + `workflow.json`. Add `hf_oauth: true` with `inference-api` scope if any `model`/`space` node needs a token.

---

Open any demo above, click a node, and hit **Run**. Then swap a local `fn` node for a real `model` node, wire it to a `space`, and you've built a production-shaped, inspectable AI pipeline. If you want the canonical spec, the **[official gr.Workflow guide](https://gradio.app/guides/workflows)** is right here in Gradio docs.

You can also build complex pipelines like AUTOMATIC1111 with **`gr.Workflows`**. Keep an eye out for our upcoming blog, where we’ll walk through how to build AUTOMATIC1111 using gr.Workflows. Here’s a sneak peek 😉👇

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/workflow1111-sample2.mp4"></video>

*Built with `gr.Workflow`: Every example in this guide is a live, duplicable Hugging Face Space linked in the table at the top.*