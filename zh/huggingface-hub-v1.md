---
title: "huggingface_hub v1.0：开源机器学习基础五周年回顾"
thumbnail: /blog/assets/huggingface-hub-v1/thumbnail.png
authors:
  - user: Wauplin
  - user: celinah
  - user: lysandre
  - user: julien-c
translators:
- user: chenglu
---


# huggingface_hub 1.0 正式版现已发布：开源机器学习基础五周年回顾

**简要总结：** 经过五年的持续开发，`huggingface_hub` 发布 v1.0 正式版！这一里程碑标志着这个库的成熟与稳定。它已成为 Python 生态中支撑 **20 万个依赖库** 的核心组件，并提供访问超过 **200 万公开模型**、**50 万公开数据集** 和 **100 万 Space 应用** 的基础能力。本次更新包含为支持未来十年开源机器学习生态而做出的重大变更，由近 300 位贡献者和数百万用户共同推动发展。

**🚀 强烈建议尽快升级至 v1.0，以体验更优性能和全新功能。**

```bash
pip install --upgrade huggingface_hub
```

此次重大版本更新包括以下内容：

* 使用 `httpx` 作为新后端请求库；
* 全新设计的 `hf` 命令行工具（取代已弃用的 `huggingface-cli`），采用 Typer 构建，功能更加丰富；
* 文件传输全面迁移至 `hf_xet`，彻底淘汰旧的 `hf_transfer` 工具。

查看完整的 [v1.0 发布说明](https://github.com/huggingface/huggingface_hub/releases/tag/v1.0.0)

> [!提示]
> 我们尽可能确保 v1.0.0 与旧版本兼容。大多数机器学习库无需修改即可兼容 v0.x 和 v1.x。主要例外是 `transformers`：v4 版本依赖 v0.x，计划中的 v5 将转向 v1.x。查看此 [issue](https://github.com/huggingface/huggingface_hub/issues/3340) 获取详细的库兼容性表。

## 背后的故事

每个主流库背后都有一段故事。`huggingface_hub` 的故事始于一个简单的想法：**如果共享机器学习模型像在 GitHub 上分享代码一样容易，会怎样？**

在 Hugging Face Hub 的早期阶段，研究人员和开发者常常面临一个困扰：
训练一个先进的模型不仅耗时、耗资源，而且在训练完成后，模型往往“被困”在个人电脑里，只能通过不稳定的 Google Drive 链接进行分享。
这导致社区重复造轮子，资源浪费严重，协作效率极低。

为了解决这一问题，Hugging Face Hub 应运而生。最初，它的功能很简单，只是用于共享和托管与 `transformers` 库兼容的模型检查点。而与 Hub 交互的全部 Python 逻辑代码，也都内置在 `transformers` 库中，其他库无法复用这些功能。

直到 2020 年底，我们推出了 `huggingface_hub` 的首个版本 [v0.0.1](https://github.com/huggingface/huggingface_hub/releases/tag/v0.0.1)，它的初衷是：将原本封装在 `transformers` 库中的内部逻辑独立出来，构建一个专用库，用于统一访问和共享 Hugging Face Hub 上的机器学习模型与数据集。最早的版本非常简洁，它只是一个 Git 操作的封装工具，用于下载文件和管理仓库。但五年过去，历经 35+ 个版本迭代，`huggingface_hub` 已远远超越最初的设想。

让我们一起来回顾这段发展历程。


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

### 奠基阶段（2020–2021）

最初的几个版本为整个库打下了基础。
版本 [0.0.8](https://github.com/huggingface/huggingface_hub/releases/tag/v0.0.8) 引入了第一个 API，通过封装 Git 命令，实现与模型仓库的交互。
接着在版本 [0.0.17](https://github.com/huggingface/huggingface_hub/releases/tag/v0.0.17) 中，加入了基于 token 的认证机制，支持访问私有仓库并安全上传内容。
虽然这些功能看起来很基础，但它们构成了后来所有进步的基石。

### 重要转折：从 Git 到 HTTP（2022）

2022 年 6 月，版本 [0.8.1](https://github.com/huggingface/huggingface_hub/releases/tag/v0.8.1) 发布，这是 Hugging Face Hub 发展史上的一个转折点——我们引入了 HTTP Commit API。

从此，用户无需再安装 Git 和 Git LFS，也能直接通过 HTTP 上传文件。新推出的 `create_commit()` API 极大简化了上传流程，尤其适合处理大型模型文件——这些文件过去通过 Git LFS 操作起来十分繁琐。

此外，该版本还引入了支持 Git 结构感知的缓存机制。所有使用 `huggingface_hub` 的库（无论是官方的 transformers，还是第三方库）现在都能共享同一套缓存系统，具备显式的版本控制和文件去重功能。

这不仅仅是一次技术优化，更是一次理念上的飞跃。
我们不再只是为 transformers 构建 Git 工具，而是在构建一套专为机器学习模型和数据打造的基础设施，面向整个机器学习生态服务。

### API 能力的全面扩展（2022–2024）

随着 Hugging Face Hub 从一个模型仓库逐步发展为一个完整的平台，`huggingface_hub` 的 API 能力也不断拓展，满足更多场景需求。

核心的仓库操作功能不断成熟，支持：

* 列出文件树（list tree）
* 浏览引用（refs）与提交记录（commits）
* 读取文件或同步整个文件夹
* 管理标签、分支和发布周期（release cycle）
* 查询仓库元数据与设置 webhook，帮助团队实时响应变更

与此同时，Hub 上的 [Spaces](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces) 功能开始崭露头角，它是一个简单却强大的方式，可以直接在 Hub 上托管和分享交互式的机器学习演示项目。`huggingface_hub` 也逐步实现了对 Spaces 的完整程序化管理能力，包括硬件资源申请、环境配置、密钥管理、文件上传等。

为了支持模型在生产环境中的部署，我们还集成了 [Inference Endpoints](https://huggingface.co/docs/huggingface_hub/guides/inference_endpoints)。而在 2025 年第三季度，[Jobs API](https://huggingface.co/docs/huggingface_hub/guides/jobs) 的加入，进一步完善了 Hugging Face 的计算服务能力。

在此过程中，社区与社交层也被提升为一等公民。现在支持：

* 创建和管理 [Pull Requests 和评论](https://huggingface.co/docs/huggingface_hub/guides/community)
* 查询用户与组织信息
* 仓库点赞、关注、粉丝功能
* 使用 [Collections](https://huggingface.co/docs/huggingface_hub/guides/collections) 整理和分享资源合集

同时，日常使用体验也得到了显著优化：Colab 中的无缝认证、大型文件夹上传的可靠性提升、支持断点续传等功能，使开发更加高效流畅。

随后，在版本 [v0.28.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.28.0) 中，我们推出了 [推理服务提供方生态](https://huggingface.co/docs/huggingface_hub/guides/inference)。不再依赖单一的推理后端，而是与多家无服务器推理平台合作，包括 Together AI、SambaNova、Replicate、Cerebras、Groq 等，用户通过一个统一的 API 即可调用多个后端，路由透明，按请求计费，真正实现了“按需调用，轻松推理”。

### Ready. Xet. Go!（2024–2025）

在版本 [v0.30.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.30.0) 中，我们发布了 Xet —— 一种颠覆性的 Git 大文件存储协议。

与传统的 Git LFS（只支持文件级去重）不同，Xet 在更精细的粒度（每 64KB 为一块）进行数据去重与传输优化。当你更新一个大型模型或数据文件时，系统只会上传或下载发生变更的部分，而不是整个文件。

这场[大规模迁移](https://huggingface.co/spaces/jsulz/ready-xet-go) 始于 50 多万个仓库，涉及超过 20PB 的数据。但令人惊喜的是，这一迁移过程对用户是完全透明的，100% 向后兼容，无需手动干预，也没有中断现有流程。

一年后，超过 **6,000,000 个仓库**、**77PB+** 的数据已成功迁移至 Xet 后端，带来了更快、更智能的上传与下载体验 🔥

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/huggingface-hub-v1/xet_progress.png)


## 成长与影响力衡量

衡量一个开源库的成长和影响力并不容易，但有时，数字本身就是最好的证明：

* **每月下载量达 1.135 亿次**，**累计下载超 16 亿次**（截至 2025 年 10 月）
* 提供访问 **200 万+ 公共模型**、**50 万+ 公共数据集** 和 **100 万+ 公共 Spaces** —— 如果包括私有仓库，总量大约翻倍
* 每日活跃用户超过 **6 万人**，每月活跃用户超过 **55 万人**
* 被全球 **20 万+ 企业** 信赖使用，从初创公司到《财富》500 强企业

但真正体现其规模的，是整个生态的广度和深度。`huggingface_hub` 已成为 GitHub 上 **超过 20 万个仓库** 和 PyPI 上 **3,000 个软件包** 的 **依赖核心** ，涵盖主流框架如 Keras、LangChain、PaddleOCR、ChatTTS、YOLO、Google Generative AI、Moshi、NVIDIA NeMo、Open Sora 等，还有无数小型工具与项目。Hugging Face 自家生态（如 transformers、diffusers、datasets、sentence-transformers、gradio、peft、trl 等）也都建立在其之上。

最令人欣慰的是，这些第三方集成大多数都是自然发生的，我们并未主动推动。这正是 Hugging Face Hub 释放力量的体现——它让整个机器学习社区能够更加开放、高效地协作与创新，而它如今的广泛使用程度，也远超我们的最初预期。

## 面向未来十年的构建

v1.0 不只是一个版本号的跃迁，它代表的是：**为未来十年开放式机器学习奠定坚实基础**。我们做出的破坏性更新并非随意为之，而是出于战略考虑，为了让 `huggingface_hub` 能够应对 AI 的高速发展，并保持全球数百万开发者所依赖的稳定性与可靠性。

### 现代化 HTTP 架构：httpx 与 hf_xet

v1.0 最重要的架构变更，是将底层 HTTP 请求库从 `requests` 迁移至 [`httpx`](https://www.python-httpx.org/)。这不仅是依赖项的替换，而是一次真正意义上的升级，使整个库正式迈入现代 HTTP 时代。

**为什么选择 httpx？**
它带来的好处非常显著：

* 原生支持 HTTP/2，连接效率更高
* 完整的线程安全，可在多线程间安全复用连接
* 最关键的是：提供统一的同步与异步接口，彻底消除原先同步与异步推理客户端之间的微妙差异

此次迁移的设计尽可能做到“对用户透明”，大多数用户无需做任何修改。对于使用自定义 HTTP 后端的开发者，我们提供了清晰的迁移路径，将 `configure_http_backend()` 替换为 `set_client_factory()` 或 `set_async_client_factory()`。

同时，`hf_xet` 现已成为 Hub 上传和下载文件的默认工具包，完全取代此前的可选方案 `hf_transfer`，后者已被彻底移除。

### MCP 与 Tiny-Agents：让智能体开发触手可及

在 [v0.32.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.32.0) 中，我们引入了 **Model Context Protocol（模型上下文协议，MCP）集成** 和 **tiny-agents 工具链**，这从根本上改变了构建 AI Agent 的方式。曾经需要复杂框架集成的任务，如今只需大约 70 行 Python 代码即可完成。

[`MCPClient`](https://huggingface.co/docs/huggingface_hub/package_reference/mcp) 提供了一个标准化接口，使 AI Agent 能够轻松与各种工具进行交互；而 `tiny-agents` CLI 工具则允许你直接从 Hub 启动 Agent。你可以连接本地或远程 MCP 服务器，将任意 Gradio Space 用作工具，并构建出自然、流畅、响应迅速的对话式智能体。

所有这些，都是在我们现有的 `InferenceClient` 以及其支持的多家推理服务商的基础上构建的。我们坚信 Agent 是未来，而 `huggingface_hub` 将持续提供这些构建 AI 工具的基础模块，助力开发者快速落地创新想法。

### 面向现代工作流的全功能命令行工具

Hugging Face 的 CLI 工具已经从一个简单的命令行工具，发展为一个 **功能全面的机器学习操作接口**。全新设计的 `hf` 命令取代了老旧的 `huggingface-cli`，采用现代化的“资源-动作”模式：

* `hf auth login`：用户认证
* `hf download` 和 `hf upload`：文件上传与下载
* `hf repo`：仓库管理
* `hf cache ls` 和 `hf cache rm`：缓存管理
* `hf jobs run`：运行云端计算任务

CLI 提供了 [沙箱式安装器](https://huggingface.co/docs/huggingface_hub/installation#install-the-hugging-face-cli)，可以在不破坏现有开发环境的前提下快速安装或升级：

```bash
# macOS 或 Linux
curl -LsSf https://hf.co/cli/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

CLI 还支持命令自动补全，并在各大主流平台上都能顺利运行。如今的 Hugging Face CLI，已经具备与现代开发工具媲美的体验和易用性。

### 为未来清理技术债

在 v1.0 中，我们移除了部分阻碍未来发展的旧功能和用法：

* 基于 Git 的 `Repository` 类已被移除。
* HTTP 接口如 `upload_file()` 和 `create_commit()` 变得更简洁、更稳定，也更适应现代化工作流。
* `HfFolder` 的 token 管理方式已被显式的 `login()`、`logout()` 和 `get_token()` 函数所取代，使用方式更直观。
* 原有的 `InferenceApi` 类被功能更完善的 `InferenceClient` 替代。
* 文件传输工具 `hf_transfer` 被彻底淘汰，现已由全新的二进制工具包 `hf_xet` 完全接管。

这些变更并非仓促决策，我们在数月前就已发布弃用通知，附带清晰的迁移指引。最终目标是打造一个 **更清晰、更易维护** 的代码库，让我们能够集中精力开发面向未来的新特性，而不是继续兼容过时的实现方式。

### 迁移指南

我们理解，破坏性更新可能会对现有项目造成困扰。正因如此，我们投入了大量精力，尽可能让迁移过程顺畅无痛。官方的 [迁移指南](https://huggingface.co/docs/huggingface_hub/concepts/migration) 提供了每项变更的 **逐步说明**，并解释了背后的原因。

最重要的是，我们在可能的地方 **保留了向后兼容性**。例如 `HfHubHttpError` 同时继承了旧版 `requests` 和新版 `httpx` 的 `HTTPError` 异常类，确保原有错误处理逻辑依然生效。

从 v1.0 起，我们将**全面聚焦新版**的开发与维护，确保为社区提供更高性能、更丰富功能和更完善的工具。旧版（v0.*）仍可在 PyPI 下载，但今后只会进行安全性补丁更新，不再添加新功能。

> [!提示]
> 我们尽力确保 `huggingface_hub` v1.0.0 与旧版兼容。在实际使用中，大多数机器学习库在 v0.x 与 v1.x 之间都能无缝切换。主要例外是 `transformers`：其 v4 版本明确依赖 v0.x，而即将发布的 v5 将改为依赖 v1.x。想了解各主流库的兼容情况，请参考这个 [兼容性汇总表](https://github.com/huggingface/huggingface_hub/issues/3340)。

## 特别致谢

感谢 280 多位为本库贡献代码、文档、翻译和社区支持的开发者们！

同时也感谢 Hugging Face 社区提供的反馈、Bug 报告与建议，这些都帮助我们不断完善产品。

最后，衷心感谢广大用户 —— 无论是独立开发者还是大型企业 —— 感谢你们信任 `huggingface_hub`，让它成为你们工作流程的一部分。是你们的支持激励我们不断前行。

如果你喜欢这个项目，欢迎到 [GitHub 点个星 ⭐](https://github.com/huggingface/huggingface_hub)，支持我们继续建设开源机器学习的未来！

五年已过，但一切才刚刚开始！
