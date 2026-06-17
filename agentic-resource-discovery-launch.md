---
title: "Agentic Resource Discovery: Let agents search" 
thumbnail: /blog/assets/agentic-resource-discovery-launch/thumbnail.png
authors:
- user: burtenshaw
- user: evalstate
---


# Agentic Resource Discovery: Let agents search for tools, skills, and other agents.

If you build with agents today, you probably know three protocols. MCP gives agents a standard way to call tools. Skills give agents a way of consuming instructions. A2A gives agents a way to call other agents. All three assume the user already knows which tool, instruction, or agent they need. The user is still responsible for discovering, integrating, and maintaining those capabilities.

The Agentic Resource Discovery (ARD) specification is the discovery layer that sits in front of them. It is a draft, open specification developed by contributors from Microsoft, Google, GoDaddy, Hugging Face, and others, with broad participation across the industry. It defines how agents and tools are cataloged, indexed, and searched across federated registries, so an agent can find capabilities at runtime instead of needing them pre-installed. It is not a product or a marketplace. It is a shared standard that any company can implement independently, and that any agent or tool can participate in.

In this post, we'll explore the specification, how Hugging Face has implemented it, and how you can start building on ARD.

## The discovery problem

The current model for agent capabilities is install-first, use-later. A developer hardcodes an MCP server URL into a config file. A user connects a service to their AI app via a plugin and reuses it. This works for the handful of tools an agent uses every day, but it doesn't scale to thousands of ad-hoc surfaces.

The fallback is to dump every available tool description into the LLM's context window and let the model pick. This is limited by the context budget. There are search-based strategies here too, but the descriptions are often too thin to disambiguate well.

ARD moves selection outside the LLM. A registry indexes capabilities with richer signals such as publisher identity, representative queries, compliance attestations, and tags. It exposes a REST endpoint. The client searches in natural language, and the model invokes whatever the search returns. The shift is from manually installed, static catalogs to intent-based search that lets an agent find the right capability dynamically, and reach a growing ecosystem of MCP tools, A2A agents, and other services without pre-configuring each one.

The specification defines two things:

- A static manifest format called `ai-catalog.json` lets publishers host their capabilities at a well-known URL.  
- A dynamic registry API at `POST /search` provides live, ranked discovery.

## ARD on the Hugging Face Hub

The Hugging Face [Discover Tool](https://github.com/huggingface/hf-discover) is our reference implementation of ARD. It provides search access to thousands of Skills, ML applications, and MCP Servers — on Hugging Face and across other ARD discovery services.

It works by combining the Hub's existing semantic search over Spaces, alongside our Agent Skills, and serving the results as ARD catalog entries. The Hub already hosts a catalog of Spaces running Gradio apps, MCP servers, and demos. Its semantic search supports an `agents=true` flag that returns Spaces ranked by agent-oriented metadata, and Discover translates that search into the ARD specification.

The adapter applies two filters. First, the response includes only Spaces whose runtime stage is `RUNNING`. Second, the response media type is driven by the request. Three media types are supported:

- `application/ai-skill`: the default. A generated `SKILL.md` wrapping the Space's `agents.md`.  
- `application/mcp-server+json`: an MCP server catalog entry for Spaces tagged `mcp-server`.  
- `application/vnd.huggingface.space+json`: raw Space metadata for clients that want to handle it themselves.

The skill type involves an additional transformation. Many Spaces ship an `agents.md` file describing how an agent should interact with them. Discover reads that file and wraps it with the frontmatter a skill consumer expects: `name`, `description`, and source metadata covering the Space ID, Hub URL, app URL, and original `agents.md` URL. The result is a skill any skill-aware client can install or load through its normal skill flow.

For MCP-tagged Spaces, the adapter generates a catalog entry pointing at the Space's Gradio MCP endpoint over HTTP transport. The URL uses the Space's runtime domain when the Hub provides one, otherwise the standard `.hf.space` slug convention.

## Using it

`discover` is built into the [Hugging Face CLI](https://github.com/huggingface/huggingface_hub) (`hf`). To get started and give you or your agent access:

```bash
# Install the Hugging Face CLI tool:
uv tool install huggingface_hub

# Search for resources to train a model
hf discover search "Fine tune a language model"

# Find MCP Servers to generate an image
hf discover search "Generate an image" --json --kind mcp

# Search other registries
hf discover search "Purchase aeroplane tickets" --registry-url <catalog-url>
```

### REST API and MCP Tool

You can also Search the catalog directly using either the REST API or an MCP Server. 

 The Hugging Face catalog is published at its well-known URL:
```
https://huggingface.co/.well-known/ai-catalog.json
```

To call search directly:
```
POST https://huggingface-hf-discover.hf.space/search
```

```bash
curl -s https://huggingface-hf-discover.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "text": "fine tune a sentence transformer",
      "filter": {
        "type": ["application/ai-skill"]
      }
    },
    "pageSize": 5
  }'
```


Search for MCP servers

```bash
curl -s https://huggingface-hf-discover.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "text": "transcribe some audio",
      "filter": {
        "type": ["application/mcp-server-card+json"]
      }
    },
    "pageSize": 5
  }'
```

Alternatively, connect any MCP Client to search via MCP endpoint using https://huggingface-hf-discover.hf.space/mcp to search the catalog. 

## What this means for the specification

ARD separates discovery from execution. The static manifest format is driven by media type, so any artifact protocol can ride the same envelope without specification-level changes. The registry API is plain HTTP REST, so any client can federate against it. Discover is one of several reference implementations of the specification across the ecosystem, and because federation is built into the protocol, a search through one service can surface capabilities hosted by another.

The Discover Tool is a working test of that design. It does not invent a new artifact format. It wraps an existing search backend, the Hub, in the specification's envelope, and lets the same Spaces surface as skills or MCP servers depending on what the client asked for.

Next steps are tighter integration with the specification's federation modes (`auto`, `referrals`, `none`) and Hub-side support for static `ai-catalog.json` manifests on user and organization profiles. Once that lands, any Space publisher will be able to advertise their capabilities through the standard well-known URI mechanism.

## Learn more

- The Agentic Resource Discovery Specification: [https://agenticresourcediscovery.org/](https://agenticresourcediscovery.org/)
- The Hugging Face Discover Tool: [https://github.com/huggingface/hf-discover](https://github.com/huggingface/hf-discover)  
- The Hugging Face CLI: [https://github.com/huggingface/huggingface\_hub](https://github.com/huggingface/huggingface_hub)  
- Agent Skills on the Hub: [https://huggingface.co/docs/hub/agents-skills](https://huggingface.co/docs/hub/agents-skills)  
- Hugging Face Spaces: [https://huggingface.co/spaces](https://huggingface.co/spaces)
 
