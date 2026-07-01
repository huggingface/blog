---
title: "How to build scalable web apps with OpenAI's Privacy Filter"
thumbnail: /blog/assets/openai-privacy-filter-web-apps/thumbnail.png
authors:
- user: ysharma
- user: freddyaboulton
- user: abidlabs
---

# How to build scalable web apps with OpenAI's Privacy Filter 

OpenAI released Privacy Filter on the Hub this week: an open-source personally-identifiable information (PII) detector that labels text across eight categories in a single forward pass over a 128k context. [Model card](https://huggingface.co/openai/privacy-filter). We spent a few hours building with it and landed on three apps that each reveals a different slice of what it can do.

- [**Document Privacy Explorer**](https://huggingface.co/spaces/ysharma/OPF-Document-PII-Explorer): drop in a PDF or DOCX, read the document back with every PII span highlighted in place.
- [**Image Anonymizer**](https://huggingface.co/spaces/ysharma/OPF-Image-Anonymizer): upload an image, get it back with redacted black bars over names, emails, and account numbers. The image is also editable on a canvas so you can make your own annotations before downloading.
- [**SmartRedact Paste**](https://huggingface.co/spaces/ysharma/OPF-SmartRedact-Paste): paste sensitive text, share a public URL that serves the redacted version, keep a private reveal link for yourself.

All three are built on [gradio.Server](https://huggingface.co/blog/introducing-gradio-server), which lets you pair custom HTML/JS frontends with Gradio's queueing, ZeroGPU allocation, and `gradio_client` SDK. In all these apps, **`gradio.Server`** plays the same backend role, and that consistency is exactly what makes it really powerful.

## The model

Privacy Filter is a 1.5B-parameter model with 50M active parameters, permissively licensed under Apache 2.0. PII categories are `private_person`, `private_address`, `private_email`, `private_phone`, `private_url`, `private_date`, `account_number`, `secret`. Context is 128,000 tokens. Achieves state-of-the-art performance on the [PII-Masking-300k benchmark](https://huggingface.co/datasets/ai4privacy/pii-masking-300k). Full numbers and methodology are in the [official release blog](https://openai.com/index/introducing-openai-privacy-filter/).

## 1. Document Privacy Explorer
Try it at [ysharma/OPF-Document-PII-Explorer](https://huggingface.co/spaces/ysharma/OPF-Document-PII-Explorer).

<video alt="Using gradio.Server and OpenAI Privacy Filter model to build an app that can redact PII from any given document" autoplay loop muted playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/openai-privacy-filter-web-apps/doc-pii-explorer.mp4" type="video/mp4">
</video>

**User problem.** You want to read a PII-heavy document (a contract, a resume, an exported chat log) with every detected span highlighted by category, a filter in the sidebar, and a summary dashboard up top. The reading experience should feel like a normal document, not a form.

**What Privacy Filter does here.** The whole file goes through in a single 128k-context forward pass, so there's no chunking, no stitching, and span offsets line up directly with the rendered text. BIOES decoding keeps span boundaries clean through long ambiguous runs.

**What `gr.Server` does here.** You could wire this up in Blocks with `gr.HighlightedText` and a sidebar, and it would work. The reading experience we wanted (serif body, category filters that toggle CSS classes client-side instead of re-running the model, a summary dashboard that doesn't force a page re-render) was easier to hand-author than to compose. `gr.Server` lets us serve the reader view as a single HTML file and expose the model behind one queued endpoint:

```python
import gradio as gr
from fastapi.responses import HTMLResponse
from gradio.data_classes import FileData

server = gr.Server()

@server.get("/", response_class=HTMLResponse)
async def homepage():
    return FRONTEND_HTML                           # reader view; see app.py

@server.api(name="analyze_document")
def analyze_document(file: FileData) -> dict:
    text = extract_text(file["path"])              # PyMuPDF / python-docx
    source_text, spans = run_privacy_filter(text)  # single 128k pass
    return {
        "text":  source_text,
        "spans": spans,                            # [{start, end, label}, ...]
        "stats": compute_stats(source_text, spans),
    }
```

Note the decorator: `@server.api(name="analyze_document")`, not a plain `@server.post`. That's the piece that plugs the handler into Gradio's queue, so concurrent uploads are serialized, `@spaces.GPU` composes correctly on ZeroGPU, and the same endpoint is reachable from both the browser and `gradio_client` with no duplicated code. The browser calls it with the Gradio JS client:

```html
<script type="module">
import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";
const client = await Client.connect(window.location.origin);

async function uploadFile(file) {
  const result = await client.predict("/analyze_document", { file: handle_file(file) });
  renderResults(result.data[0]);                   // { text, spans, stats }
}
</script>
```


## 2. Image Anonymizer
Try it at [ysharma/OPF-Image-Anonymizer](https://huggingface.co/spaces/ysharma/OPF-Image-Anonymizer).

<video alt="Using gradio.Server and OpenAI Privacy Filter model to build an app that can redact PII from any given image" autoplay loop muted playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/
openai-privacy-filter-web-apps/image-pii-redact.mp4" type="video/mp4">
</video>

**User problem.** You want to share an image or any screenshot (a Slack thread, a receipt, a Stripe dashboard) with black bars over the PII. You want to toggle bars on and off, drag them to reposition, or draw one by hand for anything the model missed, then export the result.

**What Privacy Filter does here.** Tesseract runs OCR and returns per-word bounding boxes. The backend reconstructs the full text with a char-offset to box map, then runs Privacy Filter once over the whole text. Detected character spans are looked up against the word map and joined into pixel rectangles per line.

**What `gr.Server` does here.** `gr.ImageEditor` supports layered annotation and is a reasonable starting point for image redaction. The workflow we wanted (per-bar category metadata, toggle all bars in a category at once, client-side PNG export at natural resolution with no server round-trip) was cleaner to build on a custom `<canvas>` frontend. `gr.Server` hands back pixel rectangles from one queued endpoint and lets the canvas own everything else:

```python
@server.api(name="anonymize_screenshot")
def anonymize_screenshot(image: FileData) -> dict:
    img = Image.open(image["path"]).convert("RGB")
    full_text, char_to_box = ocr_image(img)        # per-word boxes + char map
    spans = run_privacy_filter(full_text)
    boxes = spans_to_pixel_boxes(spans, char_to_box)
    return {
        "image_data_url": pil_to_base64(img),
        "width":  img.width,
        "height": img.height,
        "boxes":  boxes,                           # [{x, y, w, h, label, text}, ...]
    }
```

The frontend invokes it with `client.predict("/anonymize_screenshot", { image: handle_file(file) })`, the same pattern as above. Toggles, drags, new-bar drawing, and PNG export all happen in the browser; edits never round-trip to the server.

## 3. SmartRedact Paste
Try it at [ysharma/OPF-SmartRedact-Paste](https://huggingface.co/spaces/ysharma/OPF-SmartRedact-Paste).

<video alt="Using gradio.Server and OpenAI Privacy Filter model to build an app that can redact PII from any pasted text and generate a live link to share the redacted text" autoplay loop muted playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/
openai-privacy-filter-web-apps/smartredact-paste.mp4" type="video/mp4">
</video>

**User problem.** You want a pastebin that redacts before sharing. You paste a log line, an email, a support ticket. You get two URLs back. The public one serves the redacted version with `<PRIVATE_PERSON>`, `<PRIVATE_EMAIL>`, `<ACCOUNT_NUMBER>` placeholders, following the redaction convention from the [official blog examples](https://openai.com/index/introducing-openai-privacy-filter/#:~:text=coherent%20masking%20boundaries.-,Example%20input%20text,-Subject%3A%20Q2%20Planning). The private one is gated by a token you keep and shows the original with spans highlighted.

**What Privacy Filter does here.** Swap each detected span with a `<CATEGORY>` placeholder on the stored paste. That's the entire redaction step. Multilingual text (Spanish, French, Chinese, Hindi, and others in the model-card examples) routes through the same call with no change.

**What `gr.Server` does here.** This app needs two distinct GET routes for the same paste ID, one public and one token-gated, and the URL shape matters because the reveal URL is the thing you keep. `gr.Server` works here because it's a FastAPI app underneath — which is also why `@server.api` and plain `@server.get` can sit side by side in the same process. Note: this can also be built with `gr.Blocks()` by [mounting custom routes with FastAPI](https://www.gradio.app/docs/gradio/mount_gradio_app)  :

```python
# Model call → queued endpoint. Hit from the browser via
# client.predict("/create_paste", { text, ttl }).
@server.api(name="create_paste")
def create_paste(text: str, ttl: str = "never") -> dict:
    source_text, spans = run_privacy_filter(text)
    redacted = redact(source_text, spans)          # <CATEGORY> placeholders
    pid, reveal_token = secrets.token_urlsafe(6), secrets.token_urlsafe(22)
    PASTES[pid] = Paste(pid, reveal_token, source_text, redacted, spans,
                        expires_at=_ttl(ttl))      # see app.py
    return {
        "view_path":   f"/view/{pid}",
        "reveal_path": f"/view/{pid}?token={reveal_token}",
    }

# View page → plain FastAPI GET. No model, no queue needed, and we
# actually want the bespoke URL shape `/view/{pid}?token=...` that a
# queued endpoint couldn't give us.
@server.get("/view/{pid}", response_class=HTMLResponse)
async def view_paste(pid: str, token: str | None = None):
    p = _store_get(pid)                            # see app.py for store
    if p is None:
        return HTMLResponse(_not_found(), status_code=404)
    revealed = bool(token) and secrets.compare_digest(token, p.reveal_token)
    return HTMLResponse(_render_view(p, revealed))
```

A daemon thread evicts expired pastes every 30 seconds. The whole service, including storage, is about 200 lines of application code because everything lives in one process.

## What `gradio.Server` provides

The split across all three apps is the same — anything that touches the model goes through `@server.api`, everything else stays on plain FastAPI routes:

| App | Queued compute (`@server.api`) | Plain FastAPI routes |
| --- | --- | --- |
| Document Privacy Explorer | `analyze_document` — extract, detect, stats | `GET /` serves the custom reader view |
| Image Anonymizer | `anonymize_screenshot` — OCR, detect, spans → pixel boxes | `GET /` + `GET /examples/*` serve the canvas UI and preloaded examples |
| SmartRedact Paste | `create_paste` — detect, redact, mint IDs | `GET /` compose page, `GET /view/{pid}?token=...` public + token-gated views, `GET /api/paste/{pid}` JSON lookup |

`@server.api` gives you Gradio's queue (serialized requests, correct `@spaces.GPU` composition on ZeroGPU, progress events) and it's what the browser hits through [`@gradio/client`](https://www.gradio.app/guides/getting-started-with-the-js-client). The same endpoint is also what `gradio_client` users hit from Python — one function, two SDKs, no duplicated code. Plain `@server.get`/`@server.post` are reserved for the static surfaces: HTML pages, file lookups, cheap dict reads. That's the rule of thumb from the [gradio.Server intro post](https://huggingface.co/blog/introducing-gradio-server), and it's what makes these three apps feel consistent even though their UIs are very different.

## Try them

- [Document Privacy Explorer](https://huggingface.co/spaces/ysharma/OPF-Document-PII-Explorer)
- [Image Anonymizer](https://huggingface.co/spaces/ysharma/OPF-Image-Anonymizer)
- [SmartRedact Paste](https://huggingface.co/spaces/ysharma/OPF-SmartRedact-Paste)

Drop in a resume, a screenshot of a Slack thread, a log line with a token in it. The fun part is seeing what Privacy Filter catches (and occasionally misses) on text you actually care about.

## Recommended reading

- OpenAI's release post: [Introducing OpenAI Privacy Filter](https://openai.com/index/introducing-openai-privacy-filter/)
- Model card: [openai/privacy-filter on Hugging Face](https://huggingface.co/openai/privacy-filter)
- [Redaction examples and taxonomy on Model card](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf)
