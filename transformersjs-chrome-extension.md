---
title: "How to Use Transformers.js in a Chrome Extension"
thumbnail: /blog/assets/transformersjs-chrome-extension/thumbnail.jpg
authors:
  - user: nico-martin
---

# How to Use Transformers.js in a Chrome Extension

We recently released a Transformers.js demo browser extension powered by [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B) to help users navigate the web.

While building it, we ran into several practical observations about Manifest V3 runtimes, model loading, and messaging that are worth sharing.

## Who this is for

This guide is for developers who want to run local AI features in a Chrome extension with Transformers.js under Manifest V3 constraints.

By the end, you will have the same architecture used in this project: a background service worker that hosts models, a side panel chat UI, and a content script for page-level actions.

## What we will build

In this guide, we will recreate the core architecture of **Transformers.js Gemma 4 Browser Assistant**, using the published extension as a reference and the open-source codebase as the implementation map.

- Live extension: [Chrome Web Store](https://chromewebstore.google.com/detail/transformerjs-gemma-4-bro/dhaknnnkcdkjhcclchmnfdhddoehoool)
- Source code: [github.com/nico-martin/gemma4-browser-extension](https://github.com/nico-martin/gemma4-browser-extension)
- End result: a background-hosted Transformers.js engine, a side panel chat UI, and a content script for page extraction and highlighting.

## 1) Chrome extension architecture (MV3)

Before diving in, a quick scope note: I will not go deep on the React UI layer or Vite build configuration. The focus here is the high-level architecture decisions: what runs in each Chrome runtime and how those pieces are orchestrated.

If Manifest V3 is new to you, read this short overview first:
[What is Manifest V3?](https://developer.chrome.com/docs/extensions/develop/migrate/what-is-mv3).

### 1.1 Runtime contexts and entry points

In MV3, your architecture starts in [`public/manifest.json`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/public/manifest.json). This project defines three entry points:

- `background.service_worker = background.js`, built from [`src/background/background.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/background/background.ts).
- `side_panel.default_path = sidebar.html`, built from [`src/sidebar/index.html`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/sidebar/index.html).
- `content_scripts[].js = content.js` with `matches: http(s)://*/*` and `run_at: document_idle`, built from [`src/content/content.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/content/content.ts).

The background service worker also handles `chrome.action.onClicked` to open the side panel for the active tab.
Related entry point to know: a popup can be defined with `action.default_popup` and works well for quick actions. This project uses a side panel for persistent chat, but the orchestration pattern is the same.

### 1.2 What runs where

The key design decision is to keep heavy orchestration in the background and keep UI/page logic thin.

- Background ([`src/background/background.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/background/background.ts)) is the control plane: agent lifecycle, model initialization, tool execution, and shared services like feature extraction.
- Side panel ([`src/sidebar/*`](https://github.com/nico-martin/gemma4-browser-extension/tree/main/src/sidebar)) is the interaction layer: chat input/output, streaming updates, and setup controls.
- Content script ([`src/content/content.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/content/content.ts)) is the page bridge: DOM extraction and highlight actions.

One practical consequence of this division is that the conversation history also lives in background (`Agent.chatMessages`): the UI sends events like `AGENT_GENERATE_TEXT`, background appends the message, runs inference, then emits `MESSAGES_UPDATE` back to the side panel.

This split avoids duplicate model loads, keeps the UI responsive, and respects Chrome's security boundaries around DOM access.

### 1.3 Messaging contract

Once runtimes are separated, messaging becomes the backbone. In this project, all messages are typed through enums in [`src/shared/types.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/shared/types.ts).

- Side panel -> background (`BackgroundTasks`):
  - `CHECK_MODELS`, `INITIALIZE_MODELS`
  - `AGENT_INITIALIZE`, `AGENT_GENERATE_TEXT`, `AGENT_GET_MESSAGES`, `AGENT_CLEAR`
  - `EXTRACT_FEATURES`
- Background -> side panel (`BackgroundMessages`):
  - `DOWNLOAD_PROGRESS`, `MESSAGES_UPDATE`
- Background -> content (`ContentTasks`):
  - `EXTRACT_PAGE_DATA`, `HIGHLIGHT_ELEMENTS`, `CLEAR_HIGHLIGHTS`

The orchestration rule is simple: the background is the single coordinator; side panel and content script are specialized workers that request actions and render results.

Typical request flow:

1. Side panel sends `AGENT_GENERATE_TEXT`.
2. Background appends to `Agent.chatMessages` and runs model/tool steps.
3. Background emits `MESSAGES_UPDATE`.
4. Side panel re-renders from the updated message list.

## 2) Transformers.js integration details

### 2.1 Models and responsibilities

In [`src/shared/constants.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/shared/constants.ts), this extension uses two model roles:

- TextGeneration / LLM: [`onnx-community/gemma-4-E2B-it-ONNX`](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX) (`text-generation`, `q4f16`)
- VectorEmbeddings: [`onnx-community/all-MiniLM-L6-v2-ONNX`](https://huggingface.co/onnx-community/all-MiniLM-L6-v2-ONNX) (`feature-extraction`, `fp32`)

The split is intentional: Gemma 4 handles reasoning/tool decisions, while MiniLM generates vector embeddings for the semantic similarity search in `ask_website` and `find_history`.

### 2.2 Where inference runs

All inference runs in background ([`src/background/background.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/background/background.ts)):

- text generation via `pipeline("text-generation", ...)` with consistent KV Caching enabled by our new `DynamicCache` class
- embeddings via `pipeline("feature-extraction", ...)` plus vector normalization

This gives a single model host for all tabs/sessions, avoids duplicate memory usage, and keeps the side panel UI responsive. Because models are loaded from the background service worker, artifacts are cached under the extension origin (`chrome-extension://<extension-id>`) rather than per-website origins, which gives one shared cache for the whole extension install.

MV3 lifecycle note: service workers can be suspended and restarted, so model runtime state should be treated as recoverable and re-initialized when needed.

### 2.3 Download and cache lifecycle

The model lifecycle is explicit:

- `CHECK_MODELS` inspects what is already cached and estimates remaining download size.
- `INITIALIZE_MODELS` downloads/initializes models and emits `DOWNLOAD_PROGRESS` to the UI.
- Long-lived instances are reused after setup:
  - generation pipeline in [`src/background/agent/Agent.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/background/agent/Agent.ts)
  - embedding pipeline in [`src/background/utils/FeatureExtractor.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/background/utils/FeatureExtractor.ts)

Permissions and privacy are part of the architecture, not a checkbox at the end. In this project, [`public/manifest.json`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/public/manifest.json) asks for `sidePanel`, `storage`, `scripting`, and `tabs`, plus `host_permissions` for `http(s)://*/*`:

- `sidePanel`: required to open and control the side panel UX.
- `storage`: required to persist tool/settings state across sessions.
- `tabs` + `scripting`: required for tab-aware tools and page-level actions.
- `host_permissions` on `http(s)://*/*`: required because content extraction/highlighting is designed to work on arbitrary websites.

Why keep this narrow: permissions define user trust and Chrome Web Store review risk. Request only what your features actually need, and state clearly that inference runs locally in the extension runtime so users understand where their data is processed.

## 3) Agent and tool execution loop

### 3.1 Tool-calling basics (why this layer exists)

Before the execution loop, it helps to understand how model tool calling works (the basis for any agentic workflow). You pass messages plus a tool schema (`name`, `description`, and `parameters`), and Transformers.js formats the actual prompt from those inputs using the model's chat template. Because chat templates are model-specific, the exact tool-call format depends on the model you use. With Gemma-4-style templates, the model emits a special tool-call token block when it decides to call one.

```ts
import { pipeline } from "@huggingface/transformers";

const generator = await pipeline(
  "text-generation",
  "onnx-community/gemma-4-E2B-it-ONNX",
  {
    dtype: "q4f16",
    device: "webgpu",
  },
);

const messages = [{ role: "user", content: "What's the weather in Bern?" }];

const output = await generator(messages, {
  max_new_tokens: 128,
  do_sample: false,
  tools: [
    {
      type: "function",
      function: {
        name: "getWeather",
        description: "Get the weather in a location",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The location to get the weather for",
            },
          },
          required: ["location"],
        },
      },
    },
  ],
});
```

At generation time, the model can emit output like:

```
<|tool_call>call:getWeather{location:<|"|>Bern<|"|>}<tool_call|>
```

That is exactly why this project has a normalization layer (`webMcp`) and a parser (`extractToolCalls`): model output must be converted into deterministic tool executions.

### 3.2 Tool interface in this project

[`src/background/agent/webMcp.tsx`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/background/agent/webMcp.tsx) normalizes extension tools into a model-friendly shape:

- `name`, `description`, `inputSchema`, `execute`

Example tools include `get_open_tabs`, `go_to_tab`, `open_url`, `close_tab`, `find_history`, `ask_website`, and `highlight_website_element`.

### 3.3 Loop design (`Agent.runAgent`)

The core design choice here is to separate internal model messages from UI-facing chat messages:

- Internal model transcript (`messages`): system/user/tool/assistant turns used for messages in `generator(...)`.
- UI transcript (`chatMessages`): what the user sees, including streamed assistant text plus tool execution metadata (`tools`) and performance metrics.

Execution flow:

1. Add user input to `chatMessages`, create a placeholder assistant message, and stream tokens.
2. Parse streamed/final model output with [`extractToolCalls.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/background/agent/extractToolCalls.ts) into `{ message, toolCalls }`.
3. Keep the user-visible assistant message as plain text, while tool calls execute in background.
4. Append tool results to the assistant tool metadata and feed results back as the next prompt turn.
5. Repeat until no tool calls remain, then finalize assistant content + metrics.

This keeps user communication clean while preserving a deterministic tool loop in the background.

## 4) Data boundaries and persistence

State placement is another architectural decision that matters a lot in MV3. In this implementation, state is split by lifecycle and access pattern:

- Conversation state: background memory (`Agent.chatMessages`) for fast turn-by-turn orchestration.
- Tool preferences: `chrome.storage.local` so settings persist across sessions.
- Semantic history vectors: IndexedDB (`VectorHistoryDB`) for larger local retrieval data.
- Extracted page content: background cache (`WebsiteContentManager`) keyed by active URL.

As described in section 1.2, keeping conversation history in background gives one canonical state across UI updates. This keeps short-lived state in memory, durable settings in extension storage, and heavy retrieval data in a local database.

## 5) Build and packaging notes

You do not need a complex build setup, but MV3 does require predictable outputs for each runtime.

- Multi-entry build in [`vite.config.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/vite.config.ts):
  - [`src/sidebar/index.html`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/sidebar/index.html)
  - [`src/background/background.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/background/background.ts)
  - [`src/content/content.ts`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/src/content/content.ts)
- Ensure manifest-aligned output names/paths (`sidebar.html`, `background.js`, `content.js`).
- Keep the content script as a self-contained output to avoid runtime chunk-loading issues.

The goal is simple: one artifact per Chrome entry point, in the exact place [`public/manifest.json`](https://github.com/nico-martin/gemma4-browser-extension/blob/main/public/manifest.json) expects.

## Final takeaway

The architecture choice that unlocks this whole project is clear separation of concerns: background owns orchestration and model execution, UI surfaces stay thin, and content scripts handle page access.

This project uses a side panel, but the same approach works for other setups:

- Popup-first assistant: use `action.default_popup` for quick interactions, with background owning conversation state and model execution.
- Side-panel copilot: keep long-running conversations in a persistent panel while background handles tool loops and caching.
- Per-tab agents: keep one agent state per `tabId` in background when each tab should have its own context.
- Hybrid UI (popup + side panel + options page): all UI entry points talk to the same background coordinator and reuse the same message contracts.

The practical rule is simple: decide where state lives (`global`, `tabId`, or site-scoped), keep that state and the model inference in background (basically as background services), and let UI/content runtimes act as focused clients.
