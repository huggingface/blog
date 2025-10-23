---
title: "huggingface_hub v1.0: Five Years of Building the Foundation of Open Machine Learning"
thumbnail: /blog/assets/huggingface-hub-v1/thumbnail.png
authors:
  - user: wauplin
  - user: celinah
  - user: lysandre
  - user: julien-c
---

# huggingface_hub v1.0: Five Years of Building the Foundation of Open Machine Learning

**TL;DR:** After five years of development, `huggingface_hub` has reached v1.0 - a milestone that marks the library's maturity as the Python package powering **200,000 dependent libraries** and providing core functionality for accessing over 2 million public models, 0.5 million public datasets, and 1 million public Spaces. This release introduces breaking changes designed to support the next decade of open machine learning, driven by a global community of almost 300 contributors and millions of users.

**🚀 We highly recommend upgrading to v1.0 to benefit from major performance improvements and new capabilities.**

```bash
pip install --upgrade huggingface_hub
```

Major changes in this release include the migration to `httpx` as the backend library, a completely redesigned `hf` CLI (which replaces the deprecated `huggingface-cli`) featuring a Typer-based interface with a significantly expanded feature set, and full adoption of `hf_xet` for file transfers, replacing the legacy `hf_transfer`. You can find the **[full release notes here](https://github.com/huggingface/huggingface_hub/releases/tag/v1.0.0)**.

> [!TIP]
> We’ve worked hard to ensure that `huggingface_hub` v1.0.0 remains backward compatible. In practice, most ML libraries should work seamlessly with both v0.x and v1.x versions. The main exception is `transformers`, which explicitly requires `huggingface_hub` v0.x in its v4 releases and v1.x in its upcoming v5 release. For a detailed compatibility overview across libraries, refer to the table in this [issue](https://github.com/huggingface/huggingface_hub/issues/3340).

## The Story Behind the Library

Every major library has a story. For `huggingface_hub`, it began with a simple idea: **what if sharing machine learning models could be as easy as sharing code on GitHub?**

In the early days of the Hugging Face Hub, researchers and practitioners faced a common frustration. Training a state-of-the-art model required significant compute resources and expertise. Once trained, these models often lived in isolation, stored on local machines and shared via (broken) Google Drive links. The AI community was duplicating work, wasting resources, and missing opportunities for collaboration.

The Hugging Face Hub emerged as the answer to this challenge. Initially, it was primarily used to share checkpoints compatible with the `transformers` library. All the Python code for interacting with the Hub lived within this library, making it inaccessible for other libraries to reuse.

In late 2020, we shipped `huggingface_hub` [v0.0.1](https://github.com/huggingface/huggingface_hub/releases/tag/v0.0.1) with a simple mission: extract the internal logic from `transformers` and create a dedicated library that would unify how to access and share machine learning models and datasets on the Hugging Face Hub. Initially, the library was as straightforward as a Git wrapper for downloading files and managing repositories. Five years and 35+ releases later, `huggingface_hub` has evolved far beyond its origins.

Let's trace that journey.


<div class="flex justify-center">
    <img 
        class="block dark:hidden" 
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/huggingface-hub-v1/timeline-white.gif"
    />
    <img 
        class="hidden dark:block" 
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/huggingface-hub-v1/timeline-black.gif"
    />
</div>

### The Foundation Years (2020-2021)

The early releases established the basics. Version [0.0.8](https://github.com/huggingface/huggingface_hub/releases/tag/v0.0.8) introduced our first APIs, wrapping Git commands to interact with repositories. Version [0.0.17](https://github.com/huggingface/huggingface_hub/releases/tag/v0.0.17) brought token-based authentication, enabling secure access to private repositories and uploads. These were humble beginnings, but they laid the groundwork for everything that followed.

### The Great Shift: Git to HTTP (2022)

In June 2022, version [0.8.1](https://github.com/huggingface/huggingface_hub/releases/tag/v0.8.1) marked a pivotal moment: we introduced the HTTP Commit API. Instead of requiring Git and Git LFS installations, users could now upload files directly through HTTP requests. The new `create_commit()` API simplified workflows dramatically, especially for large model files that are cumbersome to use with Git LFS. In addition, a git-aware cache file layout was introduced. All libraries (not only transformers, but third party ones as well) would now share the same cache, with explicit versioning and file deduplication.

This wasn't just a technical improvement. It was a philosophical shift. We were no longer building a Git wrapper for transformers; we were building purpose-built infrastructure for machine learning artifacts that could power any library in the ML ecosystem.

### An Expanding API Surface (2022–2024)

As the Hub grew from a model repository into a full platform, `huggingface_hub` kept pace with an expanding API surface. Core repository primitives matured: listing trees, browsing refs and commits, reading files or syncing folders, managing tags, branches, and release cycles. Repository metadata and webhooks rounded up the offering so teams could react to changes in real time.

In parallel, [Spaces](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces) emerged as a a simple yet powerful way to host and share interactive ML demos directly on the Hub. Over time, `huggingface_hub` gained full programmatic control to deploy and manage Spaces (hardware requests, secrets, environment configuration, uploads). To deploy models on production-scale infrastructure, [Inference Endpoints](https://huggingface.co/docs/huggingface_hub/guides/inference_endpoints) were integrated as well. Finally, the [Jobs API](https://huggingface.co/docs/huggingface_hub/guides/jobs) came later (Q3 2025) to complete our compute offering.

The social and community layers became first-class citizens too: from APIs for [pull requests and comments](https://huggingface.co/docs/huggingface_hub/guides/community), to user and organization info, repository likes, following and followers, all the way through [Collections](https://huggingface.co/docs/huggingface_hub/guides/collections) to curate and share sets of related resources. Everyday ergonomics improved too: seamless authentication in Colab, resumable downloads, reliable uploads of large-scale folders, and more.

Then came version [0.28.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.28.0) and the [Inference Providers](https://huggingface.co/docs/huggingface_hub/guides/inference) ecosystem. Instead of a single inference backend, we partnered with multiple serverless providers (Together AI, SambaNova, Replicate, Cerebras, Groq, and more) to serve one API with transparent routing. We adopted a pay-per-request inference architecture that matched how people actually wanted to work.

### Ready. Xet. Go! (2024-2025)

Version [0.30.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.30.0) introduced Xet, a groundbreaking new protocol for storing large objects in Git repositories. Unlike Git LFS, which deduplicates at the file level, Xet operates at the chunk level (64KB chunks). When you update a large file in a dataset or a model, only the changed chunks are uploaded or downloaded, not the entire file.

[The migration was massive](https://huggingface.co/spaces/jsulz/ready-xet-go), starting with 20 petabytes across over 500,000 repositories. Yet it happened transparently, with full backward compatibility. One year later, all **77PB+** over **6,000,000 repositories** have been migrated to the Xet backend, allowing for much faster (and smarter!) uploads and downloads. This happened with no user intervention, and no disruption to existing workflows 🔥

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/huggingface-hub-v1/xet_progress.png)

## Measuring Growth and Impact

Measuring the growth and impact of an open-source library is a tricky task. Numbers tell a story of their own:

- **113.5 million monthly downloads**, **1.6 billion** total (October 2025).
- Powers access to **2M+** public models, **500k+** public datasets, **1M+** public Spaces, and about twice as much when accounting for private repos.
- Used by **60k+** users daily, **550k+** monthly
- Trusted by **200k+ companies** from startups to Fortune 500

But the real scale becomes clear when you look at the ecosystem. `huggingface_hub` is a **dependency** for over **200,000 repositories** on GitHub and **3,000 packages** on PyPI, powering everything from major third-party frameworks like Keras, LangChain, PaddleOCR, ChatTTS, YOLO, Google Generative AI, Moshi, NVIDIA NeMo, and Open Sora, to countless smaller libraries and tools across the ML landscape. Our own ecosystem (transformers, diffusers, datasets, sentence-transformers, lighteval, gradio, peft, trl, smolagents, timm, lerobot, etc.) benefits from this foundation as well. 

The remarkable part? Most of the third-party integrations happened organically, and we played no role in them. The Hugging Face Hub empowers the ML community in countless ways, yet we're continually humbled by how far it has gone and how widely it's used.


## Building for the Next Decade

Version 1.0 isn't just about reaching a milestone. It's about **building the foundation for the next decade of open machine learning**. The breaking changes we've made aren't arbitrary; they're strategic decisions that position `huggingface_hub` to scale with the explosive growth of AI while maintaining the reliability that millions of developers depend on.

### Modern HTTP Infrastructure with httpx and hf_xet

The most significant architectural change in v1.0 is our migration from `requests` to [`httpx`](https://www.python-httpx.org/). This isn't just dependency churn. It's a fundamental upgrade that brings the library into the modern era of HTTP.

**Why httpx?** The benefits are substantial: native HTTP/2 support for better connection efficiency and true thread safety that enables safe connection reuse across multiple threads. Most importantly, `httpx` provides a unified API for both synchronous and asynchronous operations, eliminating the subtle behavioral differences that existed between our sync and async inference clients.

The migration was designed to be as transparent as possible. Most users won't need to change anything. For those with custom HTTP backends, we've provided clear migration paths from `configure_http_backend()` to `set_client_factory()` and `set_async_client_factory()`.

Additionally, `hf_xet` is now the default package for uploading and downloading files to and from the Hub, replacing the previously optional `hf_transfer`, which has now been fully removed.

### Agents Made Simple with MCP and Tiny-Agents

Version [0.32.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.32.0) introduced **Model Context Protocol (MCP) integration** and **tiny-agents**, fundamentally changing how developers build AI agents. What once required complex framework integration now takes approximately 70 lines of Python.

The [`MCPClient`](https://huggingface.co/docs/huggingface_hub/package_reference/mcp) provides a standardized way for AI agents to interact with tools, while the `tiny-agents` CLI lets you run agents directly from the Hub. Connect to local or remote MCP servers, use any Gradio Space as a tool, and build conversational agents that feel natural and responsive.

All of this is built on top of our existing `InferenceClient` and the dozens of Inference Providers it supports. We do believe Agents are the future, and `huggingface_hub` is there to provide the building blocks that enable AI builders to play with them. 

### A Fully-Featured CLI for Modern Workflows

The CLI has evolved from a simple command-line tool into a **comprehensive interface for ML operations**. The streamlined `hf` command replaces the legacy `huggingface-cli` with a modern resource-action pattern:

- `hf auth login` for authentication
- `hf download` and `hf upload` for file transfers
- `hf repo` for repository management
- `hf cache ls` and `hf cache rm` for cache management
- `hf jobs run` for cloud compute

The CLI comes with a [sandboxed installer](https://huggingface.co/docs/huggingface_hub/installation#install-the-hugging-face-cli), making it easy to upgrade without breaking existing dev environments:

```
# On macOS or Linux
curl -LsSf https://hf.co/cli/install.sh | sh

# or on Windows
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

With autocompletion support and an installer that works across platforms, the CLI now feels as polished as any modern developer tool.

### Cleaning House for the Future

Version 1.0 removes legacy patterns that were holding us back. The Git-based `Repository` class is gone. HTTP-based methods like `upload_file()` and `create_commit()` are simpler, more reliable, and better suited for modern workflows. The `HfFolder` token management has been replaced with explicit `login()`, `logout()`, and `get_token()` functions. The old `InferenceApi` class has been superseded by the more feature-complete `InferenceClient`. `hf_transfer` has been fully replaced by `hf_xet` binary package.

These changes weren't made lightly. Most deprecations were announced months in advance with clear warnings and migration guidance. The result is a cleaner, more maintainable codebase that can focus on forward-looking features rather than supporting deprecated patterns.

### The Migration Guide

We understand that breaking changes are disruptive. That's why we've invested heavily in making the migration as smooth as possible. Our [comprehensive migration guide](https://huggingface.co/docs/huggingface_hub/concepts/migration) provides step-by-step instructions for every change with explanations of why each change was necessary.

Most importantly, we've maintained backward compatibility wherever possible. `HfHubHttpError`, for example, inherits from both the old `requests` and new `httpx` base `HTTPError` classes, ensuring that error handling continues to work across versions.
With this release, we're fully committing to the future and we will focus exclusively on v1.0 and beyond, ensuring we can deliver the performance, features, and tools the community needs to interact with the Hugging Face Hub. Previous `v0.*` versions will remain available on PyPI, but they will only receive vulnerability updates.

> [!TIP]
> We’ve worked hard to ensure that `huggingface_hub` v1.0.0 remains backward compatible. In practice, most ML libraries should work seamlessly with both v0.x and v1.x versions. The main exception is `transformers`, which explicitly requires `huggingface_hub` v0.x in its v4 releases and v1.x in its upcoming v5 release. For a detailed compatibility overview across libraries, refer to the table in this [issue](https://github.com/huggingface/huggingface_hub/issues/3340).

## Acknowledgments

To our 280+ contributors who built this library through code, documentation, translations, and community support, thank you! 

We’re also deeply grateful to the entire Hugging Face community for their feedback, bug reports, and suggestions that have shaped this library.

Finally, a huge thank you to our users -from individual developers to large enterprises- for trusting `huggingface_hub` to power your workflows. Your support drives us to keep improving and innovating.

[Please star us on GitHub ⭐](https://github.com/huggingface/huggingface_hub) to show your support and help us continue building the foundation of open machine learning. It has been five years, but it is still only the beginning!
