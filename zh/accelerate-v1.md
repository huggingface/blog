---
title: "Accelerate 1.0.0"
thumbnail: /blog/assets/186_accelerate_v1/accelerate_v1_thumbnail.png
authors:
- user: muellerzr
- user: marcsun13
- user: BenjaminB
translators:
- user: hugging-hoi2022
---

# Accelerate 1.0.0

## Accelerate 发展概况

在三年半以前、项目发起之初时，[Accelerate](https://github.com/huggingface/accelerate) 的目标还只是制作一个简单框架，通过一个低层的抽象来简化多 GPU 或 TPU 训练，以此替代原生的 PyTorch 训练流程：

![Sylvain's tweet announcing accelerate](https://raw.githubusercontent.com/muellerzr/presentations/master/talks/ai_dev_2024/sylvain_tweet.JPG)

自此，Accelerate 开始不断扩展，逐渐成为一个有多方面能力的代码库。当前，像 Llama 这样的模型已经达到了 405B 参数的量级，而 Accelerate 也致力于应对大模型和大规模训练所面临的诸多难题。这其中的贡献包括：

* [灵活、低层的训练 API](https://huggingface.co/docs/accelerate/basic_tutorials/migration)：支持在六种不同硬件设备（CPU、GPU、TPU、XPU、NPU、MLU）上训练，同时在代码层面保持 99% 原有训练代码不必改动。
* 简单易用的[命令行界面](https://huggingface.co/docs/accelerate/basic_tutorials/launch)：致力于在不同硬件上进行配置，以及运行训练脚本。
* [Big Model Inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) 功能，或者说是 `device_map="auto"`：这使得用户能够在多种不同硬件设备上进行大模型推理，同时现在可以通过诸如高效参数微调（PEFT）等技术以较小计算量来训练大模型。

这三方面的贡献，使得 Accelerate 成为了**几乎所有 Hugging Face 代码库**的基础依赖，其中包括 `transformers`、`diffusers`、`peft`、`trl`。

在 Accelerate 开发趋于稳定将近一年后的今天，我们正式发布了 Accelerate 1.0.0 —— Accelerate 的第一个发布候选版本。

本文将会详细说明以下内容：

1. 为什么我们决定开发 1.0 版本？
2. Accelerate 的未来发展，怎样结合 PyTorch 一同发展?
3. 新版本有哪些重大改变？如何迁移代码到新版本？

## 为什么要开发 1.0

发行这一版本的计划已经进行了一年多。Acceelerate 的 API 集中于 Accelerator 一侧，配置简单，代码扩展性强。但是，我们仍然认识到 Accelerate 还存在诸多有待完成的功能，这包括：

* 为 MS-AMP 和 `TransformerEngine` 集成 FP8 支持（详见[这里](https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8/transformer_engine)和[这里](https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8/ms_amp)）
* 支持在 DeepSpeed 中使用多个模型（详见[这里](https://huggingface.co/docs/accelerate/usage_guides/deepspeed_multiple_model)）
* 使 `torch.compile` 支持大模型推理 API（需要 `torch>=2.5`）
* 集成 `torch.distributed.pipelining` 作为 [替代的分布式推理机制](https://huggingface.co/docs/accelerate/main/en/usage_guides/distributed_inference#memory-efficient-pipeline-parallelism-experimental)
* 集成 `torchdata.StatefulDataLoader` 作为 [替代的数据载入机制](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py)

通过在 1.0 版本中作出的改动，Accelerate 已经有能力在不改变用户 API 接口的情况下不断融入新的技术能力了。

## Accelerate 的未来发展

在 1.0 版本推出以后，我们将重点关注技术社区里的新技术，并寻找方法去融合进 Accelerate 中。可以预见，一些重大的改动将会不久发生在 PyTorch 生态系统中：

* 作为支持 DeepSpeed 多模型的一部分，我们发现虽然当前的 DeepSpeed 方案还能正常工作，但后续可能还是需要大幅度改动整体的 API。因为我们需要为任意多模型训练场景去制作封装类。
* 由于 [torchao](https://github.com/pytorch/ao) 和 [torchtitan](https://github.com/pytorch/torchtitan) 逐渐变得受欢迎，可以推测将来 PyTorch 可能会将这些集成进来成为一个整体。为了致力于更原生的 FP8 训练、新的分布式分片 API，以及支持新版 FSDP（FSDPv2），我们推测 Accelerate 内部和通用的很多 API 也将会更改（希望改动不大）。
* 借助 `torchao`/FP8，很多新框架也带来了不同的理念和实现方法，来使得 FP8 训练有效且稳定（例如 `transformer_engine`、`torchao`、`MS-AMP`、`nanotron`）。针对 Accelerate，我们的目标是把这些实现都集中到一个地方，使用简单的配置方法让用户探索和试用每一种方法，最终我们希望形成稳定灵活的代码架构。这个领域发展迅速，尤其是 NVidia 的 FP4 技术即将问世。我们希望不仅能够支持这些方法，同时也为不同方法提供可靠的基准测试，来和原生的 BF16 训练对比，以显示技术趋势。

我们也对 PyTorch 社区分布式训练的发展感到期待，希望 Accelerate 紧跟步伐，为最近技术提供一个低门槛的入口。也希望社区能够继续探索实验、共同学习，让我们寻找在复杂计算系统上训练、扩展大模型的最佳方案。

## 如何使用 1.0 版本

如想使用 1.0 版本，需要先使用如下方法获取 Accelerate：

* pip:

```bash
pip install --pre accelerate
```

* Docker:

```bash
docker pull huggingface/accelerate:gpu-release-1.0.0rc1
```

可用的版本标记有：
* `gpu-release-1.0.0rc1`
* `cpu-release-1.0.0rc1`
* `gpu-fp8-transformerengine-release-1.0.0rc1`
* `gpu-deepspeed-release-1.0.0rc1`

## 代码迁移指南

下面是关于弃用 API 的详细说明：

* 给 `Accelerator()` 传递 `dispatch_batches`、`split_batches`、`even_batches`、`use_seedable_sampler` 参数的这种方式已经被弃用。新的方法是创建一个 `accelerate.utils.DataLoaderConfiguration()` 然后传给 `Accelerator()`（示例：`Accelerator(dataloader_config=DataLoaderConfiguration(...))`）。 
* `Accelerator().use_fp16` 和 `AcceleratorState().use_fp16` 已被移除。新的替代方式是检查 `accelerator.mixed_precision == "fp16"`。
* `Accelerator().autocast()` 不再接收 `cache_enabled` 参数。该参数被包含在 `AutocastKwargs()` 里（示例：`Accelerator(kwargs_handlers=[AutocastKwargs(cache_enabled=True)])`）。
* `accelerate.utils.is_tpu_available` 被 `accelerate.utils.is_torch_xla_available` 替代。
* `accelerate.utils.modeling.shard_checkpoint` 应被 `huggingface_hub` 里的 `split_torch_state_dict_into_shards` 替代。
* `accelerate.tqdm.tqdm()` 的第一个参数不再是 `True`/`False`，`main_process_only` 需要以命名参数的形式传参。
* `ACCELERATE_DISABLE_RICH` 不再是一个有效的环境变量。用户需通过设置 `ACCELERATE_ENABLE_RICH=1` 手动启动详细的回溯（traceback）信息。
* FSDP 中的 `fsdp_backward_prefetch_policy` 已被 `fsdp_backward_prefetch` 代替。

## 总结

首先感谢使用 Accelerate，看到一个小的想法转变成一个总下载量超过一亿、日均下载量接近三十万的项目还是很令人惊叹的。

通过本版发行，我们希望社区能够踊跃尝试，尽快在官方发行版出现前迁移到 1.0 版本。

请大家持续关注，及时追踪我们 [GitHub](https://github.com/huggingface/accelerate) 和 [社交软件](https://x.com/TheZachMueller) 上的最新信息。
