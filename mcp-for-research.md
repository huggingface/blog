---
title: "MCP for Research: How to Connect AI to Research Tools"
thumbnail: /blog/assets/mcp-for-research/thumbnail.png
authors:
- user: dylanebert
---

# MCP for Research: How to Connect AI to Research Tools

Academic research involves frequent **research discovery**: finding papers, code, related models and datasets. This typically means switching between platforms like [arXiv](https://arxiv.org/), [GitHub](https://github.com/), and [Hugging Face](https://huggingface.co/), manually piecing together connections.

The [Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit0/introduction) is a standard that allows agentic models to communicate with external tools and data sources. For research discovery, this means AI can use research tools through natural language requests, automating platform switching and cross-referencing.

![Research Tracker MCP in action](./assets/mcp-for-research/demo.gif)

## Research Discovery: Three Layers of Abstraction

Much like software development, research discovery can be framed in terms of layers of abstraction.

### 1. Manual Research

At the lowest level of abstraction, researchers search manually and cross-reference by hand.

```bash
# Typical workflow:
1. Find paper on arXiv
2. Search GitHub for implementations
3. Check Hugging Face for models/datasets
4. Cross-reference authors and citations
5. Organize findings manually
```

This manual approach becomes inefficient when tracking multiple research threads or conducting systematic literature reviews. The repetitive nature of searching across platforms, extracting metadata, and cross-referencing information naturally leads to automation through scripting.

### 2. Scripted Tools

Python scripts automate research discovery by handling web requests, parsing responses, and organizing results.

```python
# research_tracker.py
def gather_research_info(paper_url):
    paper_data = scrape_arxiv(paper_url)
    github_repos = search_github(paper_data['title'])
    hf_models = search_huggingface(paper_data['authors'])
    return consolidate_results(paper_data, github_repos, hf_models)

# Run for each paper you want to investigate
results = gather_research_info("https://arxiv.org/abs/2103.00020")
```

The [research tracker](https://huggingface.co/spaces/dylanebert/research-tracker) demonstrates systematic research discovery built from these types of scripts.

While scripts are faster than manual research, they often fail to automatically collect data due to changing APIs, rate limits, or parsing errors. Without human oversight, scripts may miss relevant results or return incomplete information.

### 3. MCP Integration

MCP makes these same Python tools accessible to AI systems through natural language.

```markdown
# Example research directive
Find recent transformer architecture papers published in the last 6 months:
- Must have available implementation code
- Focus on papers with pretrained models
- Include performance benchmarks when available
```

The AI orchestrates multiple tools, fills information gaps, and reasons about results:

```python
# AI workflow:
# 1. Use research tracker tools
# 2. Search for missing information
# 3. Cross-reference with other MCP servers
# 4. Evaluate relevance to research goals

user: "Find all relevant information (code, models, etc.) on this paper: https://huggingface.co/papers/2010.11929"
ai: # Combines multiple tools to gather complete information
```

This can be viewed as an additional layer of abstraction above scripting, where the "programming language" is natural language. This follows the [Software 3.0 Analogy](https://youtu.be/LCEmiRjPEtQ?si=J7elM86eW9XCkMFj), where the natural language research direction is the software implementation.

This comes with the same caveats as scripting:

- Faster than manual research, but error-prone without human guidance
- Quality depends on the implementation
- Understanding the lower layers (both manual and scripted) leads to better implementations

## Setup and Usage

### Quick Setup

The easiest way to add the Research Tracker MCP is through [Hugging Face MCP Settings](https://huggingface.co/settings/mcp):

1. Visit [huggingface.co/settings/mcp](https://huggingface.co/settings/mcp)
2. Search for "research-tracker-mcp" in the available tools
3. Click to add it to your tools
4. Follow the provided setup instructions for your specific client (Claude Desktop, Cursor, Claude Code, VS Code, etc.)

This workflow leverages the Hugging Face MCP server, which is the standard way to use Hugging Face Spaces as MCP tools. The settings page provides client-specific configuration that's automatically generated and always up-to-date.

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.36.1/gradio.js"
></script>

<gradio-app theme_mode="light" space="dylanebert/research-tracker-mcp"></gradio-app>

## Learn More

**Get Started:**
- [Hugging Face MCP Course](https://huggingface.co/learn/mcp-course/en/unit1/introduction) - Complete guide from basics to building your own tools
- [MCP Official Documentation](https://modelcontextprotocol.io) - Protocol specifications and architecture

**Build Your Own:**
- [Gradio MCP Guide](https://www.gradio.app/guides/building-mcp-server-with-gradio) - Turn Python functions into MCP tools
- [Building the Hugging Face MCP Server](https://huggingface.co/blog/building-hf-mcp) - Production implementation case study

**Community:**
- [Hugging Face Discord](https://hf.co/join/discord) - MCP development discussions

Ready to automate your research discovery? Try the [Research Tracker MCP](https://huggingface.co/settings/mcp) or build your own research tools with the resources above.
