---
title: "Run a vLLM Server on HF Jobs in One Command"
thumbnail: /blog/assets/vllm-jobs/thumbnail.png
authors:
  - user: qgallouedec
---

# Run a vLLM Server on HF Jobs in One Command

You can spin up a private, OpenAI-compatible LLM endpoint on Hugging Face infrastructure with a single command — no servers to provision, no Kubernetes, pay-per-second. Once it's up, you can query it from your laptop, a notebook, or anywhere else.

Here's the whole thing end to end.

## Prerequisites

- A payment method or a positive prepaid credit balance (Jobs is billed per‑minute by hardware usage).
- `huggingface_hub >= 1.19.0`: `pip install -U "huggingface_hub>=1.19.0"`.
- Logged in locally: `hf auth login`.
- Your HF token available as `$HF_TOKEN`.

## Launch the server

`hf jobs run` is `docker run` for HF infrastructure. We use the official `vllm/vllm-openai` image, ask for a GPU with `--flavor`, and expose vLLM's port with `--expose`:

```bash
hf jobs run --flavor a10g-large --expose 8000 --timeout 2h \
  vllm/vllm-openai:latest \
  vllm serve Qwen/Qwen3-4B --host 0.0.0.0 --port 8000
```

`--expose 8000` routes the container's port through HF's public jobs proxy. The command prints the URL your server is reachable at:

```
✓ Job started
  id: 6a381ca1953ed90bfb947332
  url: https://huggingface.co/jobs/qgallouedec/6a381ca1953ed90bfb947332
Hint: Exposed ports are reachable at (requires an HF token with read access to the job):
  https://6a381ca1953ed90bfb947332--8000.hf.jobs
```

`6a381ca1953ed90bfb947332` is your job ID. Keep track of it, we'll need it. We'll use `<job_id>` as a placeholder for it in the rest of the post.

Give it a couple of minutes to download weights and boot. When the logs show `Application startup complete`, you're live.

## Query it from anywhere

vLLM speaks the OpenAI API, and every request just needs your HF token as a bearer token. The quickest way to hit it is curl:

```bash
curl https://<job_id>--8000.hf.jobs/v1/chat/completions \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

which returns the usual OpenAI-style JSON, with `choices[0].message.content` holding `"Hello! How can I assist you today? 😊"`.

Or, from Python, point the OpenAI client at the exposed URL and pass the token as the API key:

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://<job_id>--8000.hf.jobs/v1",
    api_key=os.environ["HF_TOKEN"],
)
resp = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

```
Hello! How can I assist you today? 😊
```

Quick health check before you start: `curl https://<job_id>--8000.hf.jobs/v1/models -H "Authorization: Bearer $HF_TOKEN"` should list the model.

> [!WARNING]
> **🔐 The endpoint is gated, not public.** Every request must carry an HF token with **read access to the job's namespace**. A plain browser visit will be rejected. In effect, the jobs proxy *is* your API gate: access is scoped to you (and your org). That's fine for private use, but treat the URL accordingly: don't share it expecting it to be open, and don't paste your token into untrusted places. If you need finer-grained or public access, put a proper gateway in front instead.

## Clean up

Jobs are billed per second, so stop the server when you're done:

```bash
hf jobs cancel <job_id>
```

The `--timeout` you set is a safety net (it'll auto-stop), but cancelling explicitly is cheaper. An `a10g-large` runs at $1.50/hour — check `hf jobs hardware` for the full price list and pick the smallest flavor that fits your model.

## Going further: bigger models

The same command scales to much larger models — pick a beefier `--flavor` and tell vLLM to shard the model across the GPUs with `--tensor-parallel-size`. For example, the 122B Qwen3.5 mixture-of-experts model on 2× H200:

```bash
hf jobs run --flavor h200x2 --expose 8000 --timeout 2h \
  vllm/vllm-openai:latest \
  vllm serve Qwen/Qwen3.5-122B-A10B \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 \
  --max-model-len 32768 --max-num-seqs 256
```

The rule of thumb: `--tensor-parallel-size` should match the number of GPUs in the flavor (`h200x2` → 2, `h200x8` → 8). Run `hf jobs hardware` to see what's available — the H200 flavors are usually the best value for large models — and give bigger models a longer `--timeout`, since they take longer to download and load.

The `--max-model-len 32768 --max-num-seqs 256` flags are specific to this model: Qwen3.5-122B is a hybrid Mamba/attention architecture with a 256K-token default context, which doesn't leave enough memory for vLLM's default batch settings. Capping the context length and concurrent-sequence count keeps it within the GPUs' memory. If a model fails to start with an out-of-memory or cache-block error, dialing these two down is the first thing to try. Everything else — the exposed URL, the OpenAI client, the token auth — stays exactly the same.

That's it: one command to serve, one client to query.

## Going further: Chat with it in a UI

Prefer a chat window over curl? A few lines of [Gradio](https://www.gradio.app/) point at the same endpoint. Add `--reasoning-parser deepseek_r1` to the `vllm serve` command so Qwen3's thinking comes back as a separate field (not necessary, but helpful), then run this code locally (again, you'll need your HF token in `$HF_TOKEN`, and the job ID):

```python
import os
import gradio as gr
from gradio import ChatMessage
from openai import OpenAI

client = OpenAI(base_url="https://<job_id>--8000.hf.jobs/v1", api_key=os.environ["HF_TOKEN"])

def chat(message, history):
    messages = [{"role": m["role"], "content": m["content"]} for m in history if not m.get("metadata")]
    messages.append({"role": "user", "content": message})
    stream = client.chat.completions.create(model="Qwen/Qwen3-4B", messages=messages, stream=True)

    thinking, answer = "", ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        thinking += delta.model_extra.get("reasoning", "")
        answer += delta.content or ""
        out = []
        if thinking.strip():
            status = "done" if answer.strip() else "pending"
            out.append(ChatMessage(role="assistant", content=thinking, metadata={"title": "💭 Thinking", "status": status}))
        if answer.strip():
            out.append(ChatMessage(role="assistant", content=answer))
        yield out

gr.ChatInterface(chat).launch()
```

Run it, open `http://127.0.0.1:7860`, and chat — reasoning streams into the collapsible panel, the answer below.

<video alt="vllm-jobs-chat-ui" autoplay loop autobuffer muted playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-jobs/demo.mp4" type="video/mp4">
</video>
