---
title: "从 GPT2 到 Stable Diffusion：Elixir 社区迎来了 Hugging Face"
thumbnail: /blog/assets/120_elixir-bumblebee/thumbnail.png
authors:
- user: josevalim
  guest: true
translators:
- user: Slinae
- user: chenglu
  proofreader: true
---

# 从 GPT2 到 Stable Diffusion：Elixir 社区迎来了 Hugging Face


上周，[Elixir 社区](https://elixir-lang.org/) 向大家宣布，Elixir 语言社区新增从 GPT2 到 Stable Diffusion 的一系列神经网络模型。这些模型得以实现归功于 [刚刚发布的 Bumblebee 库](https://news.livebook.dev/announcing-bumblebee-gpt2-stable-diffusion-and-more-in-elixir-3Op73O)。Bumblebee 库是使用纯 Elixir 语言实现的 Hugging Face Transformers 库。

为了帮助大家使用开始这些模型， [Livebook](https://livebook.dev/) —— 用于 Elixir 语言的计算 Notebook 平台团队创建了「智能单元」集合，让开发者可以仅用三次点击即搭建各种神经网络模型任务。

<iframe width="100%" style="aspect-ratio: 16 / 9;"src="https://www.youtube.com/embed/g3oyh3g1AtQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

由于 Elixir 运行在支持并发和分布式的 Erlang 虚拟机上，开发者可以将这些模型嵌入 [Phoenix Web 应用]((https://phoenixframework.org/)，作为他们现有 Phoenix Web 应用的一部分，集成在 [使用 Broadway 的数据处理管道中](https://elixir-broadway.org)，将模型和 [Nerves 嵌入式系统]((https://www.nerves-project.org/) 一起部署，而无需依赖第三方软件。在所有场景中，Bumblebee 模型都会编译到 CPU 和 GPU 中。

## 背景

将机器学习模型引入 Elixir 的努力始于大约 2 年前的 [Numerical Elixir (Nx) 项目计划](https://github.com/elixir-nx/nx/tree/main/nx)。Nx 项目实现 Elixir 多维张量和「数值定义」，作为可编译到 CPU/GPU 的 Elixir 子集。Nx 项目没有重造轮子，而是使用 Google XLA 绑定 ([EXLA](https://github.com/elixir-nx/nx/tree/main/exla)) 和 Libtorch ([Torchx](https://github.com/elixir-nx/nx/tree/main/torchx)) 进行 CPU/GPU 编译。

Nx 项目的倡议还催生了其他几个项目。[Axon](https://github.com/elixir-nx/axon) 项目从其他项目，如 [Flax](https://github.com/google/flax) 和 [PyTorch Ignite](https://pytorch.org/ignite/index.html) 项目中获得启发，为 Elixir 引进了可进行功能组合的神经网络。 [Explorer](https://github.com/elixir-nx/explorer) 项目借鉴了 [dplyr](https://dplyr.tidyverse.org/) 和 [Rust's Polars](https://www.pola.rs/)，为 Elixir 社区引进了富有表现力和高性能的数据框 (DataFrame)。

[Bumblebee](https://github.com/elixir-nx/bumblebee) 和 [Tokenizers](https://github.com/elixir-nx/tokenizers) 是我们最新发布的库函数。我们感谢 Hugging Face 对机器学习领域跨社区和跨工具协作的支持，以及 Hugging Face 在加速 Elixir 生态建设中起的关键作用。

下一步，我们计划专注于使用 Elixir 进行神经网络训练和迁移学习，让开发者可以根据业务和应用的需求，增强和专注于预训练模型。我们同时也希望发布更多有关传统机器学习算法的进展。

## 上手实践

如果你想尝试使用 Bumblebee 库，你可以:

  * 下载 [Livebook v0.8](https://livebook.dev/)，从 Notebook 中的 "+ Smart" 单元菜单自动生成 "Neural Networks Tasks"，我们目前正致力于在其他平台和空间上运行 Livebook (敬请期待！😉)
  * 我们同时也提供了 Bumblebee 模型在 Phoenix (+ LiveView) apps 中的应用示例：[单文件 Phoenix 应用](https://github.com/elixir-nx/bumblebee/tree/main/examples/phoenix)。这些示例为将它们集成到您的生产应用中提供了必要的构建模块
  * 想获取更多的实践方法，详阅 [Notebooks](https://github.com/elixir-nx/bumblebee/tree/main/notebooks)

如果你想帮助我们构建 Elixir 机器学习生态系统，欢迎尝试使用以上的项目，并提供一些建议。这里有许多有趣的领域，从编译开发到模型构建。我们非常欢迎你可以参与进来，一起为 Bumblebee 带来更多的模型和模型架构。Elixir 社区的未来发展方向是并发式、分布式和趣味性的。
