---
title: "统一多模态大模型评估，加速多模态智能涌现"
thumbnail: /blog/assets/lmms_eval/thumbnail.png
authors:
- user: luodian
  guest: true
  org: lmms-lab
- user: PY007
  guest: true
  org: lmms-lab
- user: kcz358
  guest: true
  org: lmms-lab
- user: pufanyi
  guest: true
  org: lmms-lab
- user: JvThunder
  guest: true
  org: lmms-lab
- user: dododododo
  guest: true
- user: THUdyh
  guest: true
  org: lmms-lab
- user: liuhaotian
  guest: true
  org: lmms-lab
- user: ZhangYuanhan
  guest: true
  org: lmms-lab
- user: zhangysk
  guest: true
- user: Chunyuan24
  guest: true
  org: lmms-lab
- user: liuziwei7
  guest: true
translators:
- user: kcz358
  guest: true
---
# 统一多模态大模型评估，加速多模态智能涌现

**代码仓库** : https://github.com/EvolvingLMMs-Lab/lmms-eval

**官方主页** : https://lmms-lab.github.io/

随着人工智能研究的深入发展，多模态大模型，如GPT-4V和LLaVA等模型，已经成为了学术界和产业界的热点。但是，这些先进的模型需要一个有效的评估框架来准确衡量其性能，而这并非易事。一方面，不同模型采用的提示（prompt）和答案后处理方式多种多样，可能导致性能评估结果大相径庭，正如HuggingFace在其博客中提及的“1001 flavors of MMLU” 所示，即同一评测数据集的不同实现可能会造成极大的分数差异，甚至改变模型在排行榜上的排序。

另一方面，评估过程中的数据集获取与处理也充满挑战，尤其是当面对尚未广泛可用的旧数据集时，研究人员往往需要投入大量时间和精力进行手动搜索、下载和处理。

为解决以上问题，南洋理工大学、字节跳动等机构的研究人员联合开源了[`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval)这是一个专为多模态大型模型设计的评估框架。该框架在[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) 和 [🤗 Accelerate](https://github.com/huggingface/accelerate)的基础上改进和扩展，提供了一个统一的界面来定义模型、数据集和评估指标，为评估大型多模态模型（LMMs）提供了一个高效的解决方案。我们希望通过这个框架共同推动多模态模型的迭代周期，并促进它们在学术界和工业界的更广泛应用。我们真诚期待在多模态人工智能领域见证更多的突破和创新，共同推进人工智能技术向更高效、更智能的未来发展。

![pipeline.jpg](https://huggingface.co/datasets/kcz358/lmms-eval-blog/resolve/main/pipeline.png)

## 主要功能概览

**一键式评估**:  lmms-eval让用户能够通过单一命令轻松在多个数据集上评估其模型性能，无需手动准备数据集。只需一行代码，用户便能在几分钟内获得综合评估结果，包括详尽的日志和样本分析，涵盖模型参数、输入输出、正确答案等，适用于需要使用GPT4等高级模型进行评分的场景。

以下是一个使用 LLaVa 模型在 [MME](https://arxiv.org/abs/2306.13394) 和 [MMBench](https://arxiv.org/abs/2307.06281) 上进行评测的一个例子:

```
# 从源代码安装
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git 

# 从pypi进行安装
# pip install lmms-eval

# 安装llava
# pip install git+https://github.com/haotian-liu/LLaVA.git

# 一行代码运行评测!
accelerate launch --multi_gpu --num_processes=8 -m lmms_eval \
    --model llava   \
    --model_args pretrained="liuhaotian/llava-v1.5-7b"   \
    --tasks mme,mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_mmbenchen \
    --output_path ./logs
```

**并行加速与任务合并g**: 利用Huggingface的accelerator，lmms-eval支持多GPU、模型并行及多batch处理，显著提高评估效率。这一特点尤其在同时测试多个数据集时体现出其优势，大大缩短了评估时间。

以下是使用4 x A100 40G在不同数据集上的总运行时间。


| 数据集 (#数目)          | LLaVA-v1.5-7b | LLaVA-v1.5-13b |
| :---------------------- | :------------ | :------------- |
| mme (2374)              | 2 分 43 秒    | 3 分 27 秒     |
| gqa (12578)             | 10 分 43 秒   | 14 分 23 秒    |
| scienceqa_img (2017)    | 1 分 58 秒    | 2 分 52 秒     |
| ai2d (3088)             | 3 分 17 秒    | 4 分 12 秒     |
| coco2017_cap_val (5000) | 14 分 13 秒   | 19 分 58 秒    |

此外在0.1.1.dev 的更新中，团队支持了 tensor parallelism 能够在4 * 3090 上运行 LLaVA-v1.6-34B 这样更大的模型并且支持高效推理。

**全面的数据集支持：** `lmms-eval` 团队在 Huggingface 的 lmms-lab 上托管了超过 40 个多样化的数据集（数量持续增加），涵盖了从 COCO Captions 到 MMMU 等一系列任务。所有数据集都已经转换为统一的格式进行存档，在团队的 lmms-lab 官方 Huggingface Hub 上直接获取。用户可以查看评估数据的具体细节，并且只需点击一次即可轻松下载和使用。您可以在[此集合](https://huggingface.co/collections/lmms-lab/)下找到我们支持的所有数据集。

![org_dataset.png](https://huggingface.co/datasets/kcz358/lmms-eval-blog/resolve/main/org_dataset.png)

![viewer.png](https://huggingface.co/datasets/kcz358/lmms-eval-blog/resolve/main/viewer.png)

**易于扩展**: 通过统一的接口定义， `lmms-eval` 不仅简化了不同模型和数据集的整合过程，也为引入新的数据集和模型提供了便利。同时，它还支持简便的个性化设置，通过简单的 yaml 文件的配置即可增加新的数据集，也允许用户根据需要简单的修改配置文件来自定义评测配置。

**可对比性**: 我们提供了环境以便于作者能够复现 LLaVA 1.5 模型原本的在论文里 report 的分数。除此之外，我们也完整的提供了 LLaVA 系列模型在所有的评测数据集上的实验结果以及环境参数作为参考（见 Github 内 Readme 部分）。

**可在线同步的日志**: 我们提供详细的日志工具，帮助您理解评估过程和结果。日志包括模型参数、生成参数、输入问题、模型响应和真实答案。您还可以记录每一个细节，并在Weights & Biases的运行中进行可视化展示。用户无论何时何地都可以实时查阅结果，方便快捷。

![wandb_table.jpg](https://huggingface.co/datasets/kcz358/lmms-eval-blog/resolve/main/wandb_table.jpg)

## 结论

总而言之，该框架的实施不仅为多模态模型评估提供了新工具，还为未来的研究和开发铺平了道路，包括视频多模态评估、少样本评估模式和批量推理加速等，展示了其强大的潜力和远见。`lmms-eval` 的推出标志着评估的新时代的到来，为人工智能研究和应用开辟了新的道路。我们希望社区能够发现在这个快速发展的领域中使用它来评估他们自己的模型的价值所在！


## 相关链接

GitHub仓库 : https://github.com/EvolvingLMMs-Lab/lmms-eval

官方网站 : https://lmms-lab.github.io/

Huggingface网站 : https://huggingface.co/lmms-lab

数据集 : https://huggingface.co/collections/lmms-lab/lmms-eval-661d51f70a9d678b6f43f272
