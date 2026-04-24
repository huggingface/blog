---
title: "Agentic Orchestration: model-agnostic multi-agent workflows with CrewAI and MCP"
thumbnail: /blog/assets/agentic-orchestration/thumbnail.jpg
authors:
- user: zlatko-lakisic
---

# Agentic Orchestration: model-agnostic multi-agent workflows with CrewAI and MCP

We are sharing [**agentic-orchestration**](https://github.com/zlatko-lakisic/agentic-orchestration), an open source **orchestration layer** built on [CrewAI](https://www.crewai.com/). It turns natural-language goals and YAML configuration into coordinated multi-agent workflows: a planner proposes steps and backends, agents execute with clear roles, and optional [**Model Context Protocol (MCP)**](https://modelcontextprotocol.io/) servers extend each agent with real tools (for example Home Assistant, documentation search, or custom servers you add to the catalog).

The project is **model- and vendor-agnostic**. The same orchestrator can mix **Ollama** (local), **OpenAI-compatible** APIs, **Anthropic Claude**, **Hugging Face Inference**, and TPU-oriented providers such as **vLLM** and **JetStream**—selected per task from a catalog, filtered by credentials and declared hardware capability (`cpu` / `gpu` / `tpu`, with optional VRAM heuristics). Planning can follow the same breadth of backends via LiteLLM, so you are not locked into a single stack for either planning or execution.

## Why YAML and agnostic wiring?

The design targets teams that already have models, MCP servers, and credentials in place. You adopt the engine **on top of what you run today**: fine-tuned or self-hosted models, MCP catalogs, and environment variables. The goal is a **short path to a proof of concept**—multi-step planning, execution, sessions, and an optional web UI—**as configuration**, without rewriting planner, crew, and tool glue from scratch.

The repository is organized as a monorepo:

| You want | Start here |
| --- | --- |
| **Production-style orchestration** (YAML workflows, dynamic planning, MCP, sessions, learning, knowledge base) | [`agentic-orchestration-tool/`](https://github.com/zlatko-lakisic/agentic-orchestration/tree/main/agentic-orchestration-tool) |
| **Browser chat** over local WebSockets (dynamic and iterative modes) | [`agentic-orchestration-web/`](https://github.com/zlatko-lakisic/agentic-orchestration/tree/main/agentic-orchestration-web) |
| **Industry overlays** (extra orchestrator context, agent YAML, MCP catalog fragments) | [`examples/verticals/`](https://github.com/zlatko-lakisic/agentic-orchestration/tree/main/examples/verticals) |

Shipped **example verticals** include **healthcare** (evidence and commercial brief style tasks) and **logistics** (warehousing, WMS/ERP MCP hooks with simulated fixtures). Each vertical wires `--example <id>` on the Python CLI and optional dedicated web scripts and ports so scenarios can run side by side with the stock UI.

## Architecture at a glance

<p align="center">
    <img src="https://raw.githubusercontent.com/zlatko-lakisic/agentic-orchestration/main/vision.png" alt="Agentic orchestration: planner, runner, MCP tools, adaptation, memory" width="90%">
</p>

1. **Planner** — Interprets the user goal (and session history) and emits a structured plan: steps, agent provider IDs, optional MCP IDs.
2. **Runner** — Builds a CrewAI `Crew` with agents and tasks, resolves MCP configs per task, and runs the workflow.
3. **Tools (MCP)** — When relevant, agents attach MCP servers so they call real APIs instead of inventing facts.
4. **Adaptation** — Iterative dynamic mode can re-plan between steps; a small controller can stop early or suggest refined goals.
5. **Memory and aggregation** — Sessions persist planner turns and excerpts; an optional local **knowledge base** (SQLite with full-text search) stores finalized outputs for reuse; an optional **learning** loop scores runs and feeds statistics back into planner context.

The guiding idea is **swap models and providers without rewriting orchestration logic**—only YAML catalogs and environment variables change.

## Notable capabilities

- **CrewAI-native** crews, agents, tasks, and sequential or hierarchical processes.
- **Hardware-aware routing** — Catalog entries can declare `hardware.architecture`; incompatible providers are filtered before planning.
- **Dynamic planning** — Natural-language goals become a JSON plan and an ephemeral workflow.
- **Per-task MCP** — MCP sets per step; agent instances are deduplicated by provider plus MCP fingerprint.
- **Sessions, iterative dynamic mode, learning loop, answer cache** — Documented in the tool package README.

Configuration is **environment-first**; [`agentic-orchestration-tool/.env.example`](https://github.com/zlatko-lakisic/agentic-orchestration/blob/main/agentic-orchestration-tool/.env.example) is the checklist for planner models, HF tokens, Ollama hosts, session paths, learning toggles, and web bridge variables.

## Quick start

**CLI (Python 3.12 recommended):**

```bash
cd agentic-orchestration-tool
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env      # add keys and optional AGENTIC_* toggles
python main.py --dynamic "Your goal in natural language"
```

**Web UI (Node.js 18+):**

```bash
cd agentic-orchestration-web
npm install
npm start
```

By default the UI is served at `http://127.0.0.1:3847/`. The web server spawns local Python with user-supplied text: **treat it as a local development tool** and read the [web README](https://github.com/zlatko-lakisic/agentic-orchestration/blob/main/agentic-orchestration-web/README.md) before exposing it on a network.

## License and links

The project is released under the **Apache 2.0** license. For YAML shapes, CLI flags, router mode, and internal modules, start with [**agentic-orchestration-tool/README.md**](https://github.com/zlatko-lakisic/agentic-orchestration/blob/main/agentic-orchestration-tool/README.md); for overlays, see [**examples/verticals/README.md**](https://github.com/zlatko-lakisic/agentic-orchestration/blob/main/examples/verticals/README.md).

If you try it on your stack—especially with Hugging Face endpoints or custom MCP—we would love to hear how it goes in the comments or on [GitHub Issues](https://github.com/zlatko-lakisic/agentic-orchestration/issues).
