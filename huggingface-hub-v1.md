TODO:
- header section
- thumbnail
- acknowledgments
- link to release notes
- some snippets? (unsure)
- "please star us on Github"
- add links when relevant

# huggingface_hub v1.0: Five Years of Building the Foundation of Open Machine Learning

**TL;DR:** After five years of development, `huggingface_hub` has reached v1.0—a milestone that marks the library's maturity as the Python package powering **200,000 dependent libraries** and providing core functionality for accessing over 2 million models, 400,000+ datasets, and 600,000+ Spaces. This release introduces breaking changes designed to support the next decade of open machine learning, driven by a global community of 280+ contributors and millions of users.

**🚀 We highly recommend upgrading to v1.0 to benefit from major performance improvements and new capabilities.**

## The Story Behind the Library

Every major library has an origin story. For `huggingface_hub`, it began with a simple idea: **what if sharing machine learning models could be as easy as sharing code on GitHub?**

In the early days of the Hugging Face Hub, researchers and practitioners faced a common frustration. Training a state-of-the-art model required significant compute resources and expertise. Once trained, these models often lived in isolation, stored on local machines and shared via broken Google Drive links. The AI community was duplicating work, wasting resources, and missing opportunities for collaboration.

The Hugging Face Hub emerged as the answer to this challenge. Initially, it was primarily used to share checkpoints compatible with the `transformers` library. All the Python code for interacting with the Hub lived within this library, making it inaccessible for other libraries to reuse.

In late 2020, we shipped `huggingface_hub` v0.0.1 with a simple mission: extract the internal logic from `transformers` and create a dedicated library that would unify how to access and share machine learning models and datasets on the Hugging Face Hub. Initially, the library was as straightforward as a Git wrapper for downloading files and managing repositories. Five years and 35+ releases later, `huggingface_hub` has evolved far beyond its origins. Let's trace that journey.

### The Foundation Years (2020-2021)

The early releases established the basics. Version 0.0.8 introduced our first APIs, wrapping Git commands to interact with repositories. Version 0.0.17 brought token-based authentication, enabling secure access to private repositories and uploads. These were humble beginnings, but they laid the groundwork for everything that followed.

### The Great Shift: Git to HTTP (2022)

In June 2022, version 0.8.1 marked a pivotal moment: we introduced the HTTP Commit API. Instead of requiring Git and Git LFS installations, users could now upload files directly through HTTP requests. The new `create_commit()` API simplified workflows dramatically, especially for large model files that had been cumbersome with Git LFS.

This wasn't just a technical improvement—it was a philosophical shift. We were no longer building a Git wrapper; we were building purpose-built infrastructure for machine learning artifacts.

### An Expanding API Surface (2022–2024)

As the Hub grew from a model repository into a full platform, `huggingface_hub` kept pace with an expanding API surface. Core repository primitives matured: listing trees, browsing refs and commits, reading files or syncing folders, and managing tags, branches, and releases. Repository metadata and webhooks rounded out the offering so teams could react to changes in real time.

Spaces gained full programmatic control to deploy and manage ML apps (hardware requests, secrets, environment configuration, uploads). To deploy models on production-scale infrastructure, Inference Endpoints were integrated as well. The Jobs API came later (Q3 2025) to complete our compute offering.

The social and community layers became first-class: APIs for pull requests and comments, user and organization info, repository likes, following and followers, plus Collections to curate and share sets across the Hub. Everyday ergonomics improved too: seamless authentication in Colab, resumable downloads, reliable uploads of large-scale folders, and more.

Then came version 0.28.0 and the Inference Providers ecosystem. Instead of a single inference backend, we partnered with multiple serverless providers—Together AI, SambaNova, Replicate, Cerebras, Groq, and more. One API, multiple providers, transparent routing. The architecture finally matched how people actually wanted to work.

### Ready. Xet. Go! (2024-2025)

Version 0.30.0 introduced Xet, a groundbreaking new protocol for storing large objects in Git repositories. Unlike Git LFS, which deduplicates at the file level, Xet operates at the chunk level (64KB chunks). When you modify a large model file, only the changed chunks are uploaded or downloaded, not the entire file.

The migration was massive, starting with **20 petabytes** across over 500,000 repositories. Yet it happened transparently, with full backward compatibility. One year later, all repositories have been migrated seamlessly to the Xet backend, allowing for much faster (and smarter!) uploads and downloads.

## Measuring Growth and Impact

Measuring the growth and impact of an open-source library is a tricky task. Numbers tell a story of their own:

- **113.5 million monthly downloads**, **1.6 billion** total (October 2025).
- Powers access to **2M+** public models, **400k+** public datasets, **600k** public Spaces, and about triple when accounting for private repos.
- Used by **60k+** users daily, **550k+** monthly
- Trusted by **200k+ companies** from startups to Fortune 500

But the real scale becomes clear when you look at the ecosystem. `huggingface_hub` is a **dependency** for over **200,000 repositories** on GitHub and **3,000 packages** on PyPI, ranging from our own ecosystem (transformers, diffusers, datasets, sentence-transformers, lighteval, gradio, peft, trl, smolagents, timm, lerobot, etc.) to major third-party frameworks like Keras, LangChain, PaddleOCR, ChatTTS, YOLO, Google Generative AI, Moshi, NVIDIA NeMo, Open Sora, and countless others. The remarkable part? Most of these integrations happened organically, and we played no role in them. The Hugging Face Hub empowers the ML community in countless ways, yet we're continually humbled by how far it has gone and how widely it's used.


## Building for the Next Decade

Version 1.0 isn't just about reaching a milestone—it's about **building the foundation for the next decade of open machine learning**. The breaking changes we've made aren't arbitrary; they're strategic decisions that position `huggingface_hub` to scale with the explosive growth of AI while maintaining the reliability that millions of developers depend on.

### Modern HTTP Infrastructure with httpx and hf_xet

The most significant architectural change in v1.0 is our migration from `requests` to `httpx`. This isn't just dependency churn—it's a fundamental upgrade that brings the library into the modern era of HTTP.

**Why httpx?** The benefits are substantial: native HTTP/2 support for better connection efficiency and true thread safety that enables safe connection reuse across multiple threads. Most importantly, `httpx` provides a unified API for both synchronous and asynchronous operations, eliminating the subtle behavioral differences that existed between our sync and async inference clients.

The migration was designed to be as transparent as possible. Most users won't need to change anything. For those with custom HTTP backends, we've provided clear migration paths from `configure_http_backend()` to `set_client_factory()` and `set_async_client_factory()`.

Additionally, `hf_xet` is now the default package for uploading and downloading files to and from the Hub, replacing `hf_transfer`, which has been removed.

### Agents Made Simple with MCP and Tiny-Agents

Version 0.32.0 introduced **Model Context Protocol (MCP) integration** and **tiny-agents**, fundamentally changing how developers build AI agents. What once required complex framework integration now takes approximately 70 lines of Python.

The `MCPClient` provides a standardized way for AI agents to interact with tools, while the `tiny-agents` CLI lets you run agents directly from the Hub. Connect to local or remote MCP servers, use any Gradio Space as a tool, and build conversational agents that feel natural and responsive.

All of this is built on top of our existing `InferenceClient` and the 12+ Inference Providers integrated.

### A Fully-Featured CLI for Modern Workflows

The CLI has evolved from a simple command-line tool into a **comprehensive interface for ML operations**. The streamlined `hf` command replaces the legacy `huggingface-cli` with a modern resource-action pattern:

- `hf auth login` for seamless authentication
- `hf download` and `hf upload` for file transfers
- `hf repo` for repository management
- `hf cache ls` and `hf cache rm` for cache management
- `hf jobs run` for cloud compute

With autocompletion support and an installer that works across platforms, the CLI now feels as polished as any modern developer tool.

**Note:** The CLI is now included by default - the `[cli]` extra is no longer needed (or available). Just install `huggingface_hub` and you're ready to go.

### Cleaning House for the Future

Version 1.0 removes legacy patterns that were holding us back. The Git-based `Repository` class is gone. HTTP-based methods like `upload_file()` and `create_commit()` are simpler, more reliable, and better suited for modern workflows. The `HfFolder` token management has been replaced with explicit `login()`, `logout()`, and `get_token()` functions. The old `InferenceApi` class has been superseded by the more feature-complete `InferenceClient`.

These changes weren't made lightly. Each deprecation was announced months in advance with clear warnings and migration guidance. The result is a cleaner, more maintainable codebase that can focus on forward-looking features rather than supporting deprecated patterns.

### The Migration Guide

We understand that breaking changes are disruptive. That's why we've invested heavily in making the migration as smooth as possible. Our [comprehensive migration guide](https://huggingface.co/docs/huggingface_hub/concepts/migration) provides step-by-step instructions for every change with explanations of why each change was necessary.

Most importantly, we've maintained backward compatibility wherever possible. `HfHubHttpError` especially inherits from both the old `requests` and new `httpx` base `HTTPError` classes, ensuring that error handling continues to work across versions.
With this release, we're fully committing to the future and we will focus exclusively on v1.0 and beyond, ensuring we can deliver the performance, features, and tools the community needs to interact with the Hugging Face Hub. While previous versions `v0.*` will remain available on PyPI, they will no longer receive updates.

## Acknowledgments

To our 280+ contributors who built this library through code, documentation, translations, and community support—thank you

(thank you everyone, blablabla, to complete)
