---
title: "英特尔 Gaudi 加速辅助生成"
thumbnail: /blog/assets/assisted-generation-support-gaudi/thumbnail.png
authors:
- user: haimbarad
  guest: true
  org: Intel
- user: nraste
  guest: true
  org: Intel
- user: joeychou
  guest: true
  org: Intel
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 英特尔 Gaudi 加速辅助生成

随着模型规模的增长，生成式人工智能的实现需要大量的推理资源。这不仅增加了每次生成的成本，而且还增加了用于满足此类请求的功耗。因此，文本生成的推理优化对于降低延迟、基础设施成本以及功耗都至关重要，其可以改善用户体验并提高文本生成任务的效率。

辅助解码是一种用于加速文本生成的流行方法。我们在英特尔 Gaudi2 上对其进行了适配和优化，使得其性能与英伟达 H100 GPU 相当，一如我们在 [之前的博文](https://huggingface.co/blog/zh/bridgetower) 中所展示的，但 Gaudi2 的价格仅相当于英伟达 A100 80GB GPU。这项工作现已集成入 Optimum Habana，Optimum Habana 对 Transformers 和 Diffusers 等各种 Hugging Face 库进行了扩展，以在英特尔 Gaudi 处理器上对用户的工作流进行全面优化。

## 投机采样 - 辅助解码

投机采样是一种用于加速文本生成的技术。其工作原理是用一个草稿模型一次生成 K 个词元，再由目标模型对这 K 个生成词元进行评估。如若草稿模型生成的某个位置的词元被拒绝，则用目标模型来生成该位置的词元，并丢弃草稿模型生成的随后词元，反复执行上述过程直至结束。使用投机采样，可以提高文本生成的速度并得到与原始自回归采样相当的生成质量。使用该技术时，用户可以指定草稿模型。数据证明，推测采样可为基于 transformer 的大模型带来约 2 倍的加速。一句话概括，投机采样可以加速文本生成并提高英特尔 Gaudi 处理器上的文本生成性能。

然而，草稿模型和目标模型 KV 缓存尺寸不同，因此同时分别对这两个模型进行优化显得尤为重要。本文，我们假设目标模型为一个量化模型，并利用 KV 缓存和投机采样对其进行加速。请注意，这里每个模型都有自己的 KV 缓存。我们用草稿模型生成 K 个词元，然后用目标模型对其进行评估; 当草稿模型生成的词元被拒绝时，目标模型会用于生成被拒绝位置的词元，并丢弃草稿模型生成的随后词元; 接着草稿模型继续生成接下来的 K 个词元，如此往复。

请注意，文献 [2] 证明了执行投机采样可以恢复目标模型的分布 - 这从理论上保证了投机采样可以达到与对目标模型自身进行自回归采样相同的采样质量。因此，不采用投机采样的理由仅在于收益，如草稿模型的尺寸并没有足够的比较优势，抑或是草稿模型生成词元的接受比太低。

辅助生成是一种类似于投机采样的技术，其大约与投机采样同一时间被独立发明出来 [3]。其作者将此方法集成到了 Hugging Face Transformers 中，现在模型的 _.generate()_ 的方法中有一个可选的 _assistant\_model_ 参数用于启用辅助生成。

## 用法及实验

在 Gaudi 上使用辅助生成非常简单，我们在 [此](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation#run-speculative-sampling-on-gaudi) 提供了一个示例。

顾名思义，参数 `--assistant_model` 用于指定草稿模型。草稿模型用于生成 K 个词元，然后由目标模型对其进行评估。当草稿模型生成的词元被拒绝时，目标模型会自己生成该位置的词元，并将草稿模型生成的该位置之后的词元丢弃。接着，草稿模型再生成接下来的 K 个词元，如此往复。草稿模型的接受率部分取决于模型选择，部分取决于输入文本。一般情况下，辅助生成能将大型 transformer 族模型的速度提高约 2 倍。

## 总结

Gaudi 现已支持用户简单易用地使用辅助生成加速文本生成，用户可用其进一步提高英特尔 Gaudi 处理器的性能。该方法基于投机采样，已被证明可以有效提高基于大型 transformer 模型的性能。

# 参考文献

[1] N. Shazeer，Fast Transformer Decoding: One Write-Head is All You Need，Nov. 2019，arXiv:1911.02150.

[2] C. Chen，S. Borgeaud，G. Irving，J.B. Lespiau，L. Sifre，J. Jumper, Accelerating Large Language Model Decoding with Speculative Sampling，Feb. 2023，arXiv:2302.01318

[3] J. Gante，辅助生成: 低延迟文本生成的新方向，May 2023，https://huggingface.co/blog/zh/assisted-generation