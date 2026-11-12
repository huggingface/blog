---
title: "Agentic Orchestration: model-agnostic multi-agent workflows with CrewAI and MCP"
thumbnail: /blog/assets/agentic-orchestration/thumbnail.jpg
authors:
- user: zlatko-lakisic
---

# Agentic Orchestration: model-agnostic multi-agent workflows with CrewAI and MCP

Much of productized AI still behaves like a **static pattern**: one prompt in, one completion out. Even multi-agent setups can freeze into hand-authored **if–then** graphs that do not adapt when the task changes halfway through.

[**agentic-orchestration**](https://github.com/zlatko-lakisic/agentic-orchestration) is an open source **orchestration layer** built on [CrewAI](https://www.crewai.com/) that pushes toward **dynamic coordination**: natural-language goals and YAML configuration become workflows where a **planner** proposes steps, **agents** execute with clear roles, and optional [**Model Context Protocol (MCP)**](https://modelcontextprotocol.io/) servers attach real tools (for example Home Assistant, documentation search, or catalog entries you add). A longer narrative on the motivation—moving from *where* models run to *how* they are orchestrated—is in [*From Static Infrastructure to Dynamic Intelligence: The Path to AGI Orchestration*](https://www.linkedin.com/pulse/from-static-infrastructure-dynamic-intelligence-path-agi-lakisic-f1rnc/) (LinkedIn).

The project works very well in a **Hugging Face-first** setup: use **Hugging Face Inference** for model access, Hub-hosted assets for docs and artifacts, and MCP-backed tools where your workflow needs real system calls. You can still keep it model-agnostic when needed, but the day-to-day flow can stay centered on Hugging Face primitives and APIs.

<p align="center">
  <img src="/blog/assets/agentic-orchestration/thumbnail.jpg" alt="Agentic Orchestration overview" width="90%">
</p>

## Why YAML and Hugging Face-first wiring?

The design targets teams that already have models, MCP servers, and credentials in place. You adopt the engine **on top of what you run today**: fine-tuned or self-hosted models, MCP catalogs, and environment variables. The goal is a **short path to a proof of concept**—multi-step planning, execution, sessions, and an optional web UI—**as configuration**, without rewriting planner, crew, and tool glue from scratch.

Agent **personas** and provider wiring live in YAML catalogs rather than as one-off code branches, so teams can standardize orchestration around Hugging Face endpoints and selectively extend to other backends only when a specific task needs it.

## Dynamic iteration and knowledge handoff

Instead of a single frozen playbook, the stack emphasizes **dynamic iteration**:

- **Task shaping** — The planner interprets the goal (and session history) and emits a structured plan; in **iterative dynamic** mode it can **re-plan between steps** when new context appears, instead of failing closed.
- **Handoff across steps** — Outputs from earlier tasks can flow into later ones (step context injection), so specialized steps inform downstream reasoning with more than a bare handoff string.
- **Tool-grounded work** — MCP-backed steps call **real** services and documents where you wire them, easing the pressure for the model to invent facts when the answer should live in your systems.

That mirrors the argument in the LinkedIn piece: **orchestration**—which tools, which model endpoints, and in what order—can matter as much as raw model scale for hard, multi-step problems.

## MCP as the integration layer

Mapping **MCP servers** in YAML lets you attach internal APIs, retrieval, or domain-specific hosts directly into Hugging Face-centered agent flows, without giving every agent blanket access to every system. For many organizations, that is the practical path: keep proprietary data and specialized hosts behind MCP boundaries, while still letting the planner attach the right tool surface **per task**.

## Repository layout and example verticals

The repository is organized as a monorepo:

| You want | Start here |
| --- | --- |
| **Production-style orchestration** (YAML workflows, dynamic planning, MCP, sessions, learning, knowledge base) | [`agentic-orchestration-tool/`](https://github.com/zlatko-lakisic/agentic-orchestration/tree/main/agentic-orchestration-tool) |
| **Browser chat** over local WebSockets (dynamic and iterative modes) | [`agentic-orchestration-web/`](https://github.com/zlatko-lakisic/agentic-orchestration/tree/main/agentic-orchestration-web) |
| **Industry overlays** (extra orchestrator context, agent YAML, MCP catalog fragments) | [`examples/verticals/`](https://github.com/zlatko-lakisic/agentic-orchestration/tree/main/examples/verticals) |

Shipped **example verticals** include **healthcare** (evidence and commercial brief style tasks) and **logistics** (warehousing, WMS/ERP MCP hooks with simulated fixtures). Each vertical wires `--example <id>` on the Python CLI and optional dedicated web scripts and ports so scenarios can run side by side with the stock UI.

The [**healthcare**](https://github.com/zlatko-lakisic/agentic-orchestration/tree/main/examples/verticals/healthcare) overlay is a concrete place to explore the kind of pattern described in the LinkedIn article: multiple agent roles, domain-specific orchestrator context, and room to map **MCP** to regulated or internal systems (for example evidence retrieval or line-of-business tools you host yourself)—so iteration stays **grounded** rather than purely speculative.

## Architecture at a glance

<p align="center">
    <img src="https://raw.githubusercontent.com/zlatko-lakisic/agentic-orchestration/main/vision.png" alt="Agentic orchestration: planner, runner, MCP tools, adaptation, memory" width="90%">
</p>

1. **Planner** — Interprets the user goal (and session history) and emits a structured plan: steps, agent provider IDs, optional MCP IDs.
2. **Runner** — Builds a CrewAI `Crew` with agents and tasks, resolves MCP configs per task, and runs the workflow.
3. **Tools (MCP)** — When relevant, agents attach MCP servers so they call real APIs instead of inventing facts.
4. **Adaptation** — Iterative dynamic mode can re-plan between steps; a small controller can stop early or suggest refined goals.
5. **Memory and aggregation** — Sessions persist planner turns and excerpts; an optional local **knowledge base** (SQLite with full-text search) stores finalized outputs for reuse; an optional **learning** loop scores runs and feeds statistics back into planner context.

The guiding idea is to keep orchestration logic stable while you evolve model choices. In practice, that often means a Hugging Face-first default with optional provider extensions defined in YAML.

On top of that loop, you can add **quality-oriented habits**: for example a separate evaluation or critique pass, human review, or the repo’s optional **learning** path (structured eval traces, optional user ratings, and statistics fed back into planner context) so runs improve measurably over time rather than relying on a single shot in the dark.

## Notable capabilities

- **CrewAI-native** crews, agents, tasks, and sequential or hierarchical processes.
- **Hardware-aware routing** — Catalog entries can declare `hardware.architecture`; incompatible providers are filtered before planning.
- **Dynamic planning** — Natural-language goals become a JSON plan and an ephemeral workflow.
- **Per-task MCP** — MCP sets per step; agent instances are deduplicated by provider plus MCP fingerprint.
- **Sessions, iterative dynamic mode, learning loop, answer cache** — Documented in the tool package README.

Configuration is **environment-first**; [`agentic-orchestration-tool/.env.example`](https://github.com/zlatko-lakisic/agentic-orchestration/blob/main/agentic-orchestration-tool/.env.example) is the checklist for planner models, **HF tokens**, session paths, learning toggles, and web bridge variables.

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

If you try it on your stack—especially with **Hugging Face Inference** and custom MCP integrations—we would love to hear how it goes in the comments, in [GitHub Discussions](https://github.com/zlatko-lakisic/agentic-orchestration/discussions), or via [Issues and pull requests](https://github.com/zlatko-lakisic/agentic-orchestration/issues).
