---
title: 使用 Diffusers 通过 Dreambooth 技术来训练 Stable Diffusion
thumbnail: /blog/assets/sd_dreambooth_training/thumbnail.jpg
authors:
- user: valhalla
- user: pcuenq
- user: 9of9
  guest: true
translators:
- user: innovation64
- user: inferjay
  proofreader: true
---

# 使用 Diffusers 通过 Dreambooth 技术来训练 Stable Diffusion


[Dreambooth](https://dreambooth.github.io/) 是一种使用专门的微调形式来训练 [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) 的新概念技术。一些人用他仅仅使用很少的他们的照片训练出了一个很棒的照片，有一些人用他去尝试新的风格。🧨 Diffusers 提供一个 [DreamBooth 训练脚本](https://github.com/huggingface/diffusers/tree/main/examples/DreamBooth)。这使得训练不会花费很长时间，但是他比较难筛选正确的超参数并且容易过拟合。

我们做了许多实验来分析不同设置下 DreamBooth 的效果。本文展示了我们的发现和一些小技巧来帮助你在用 DreamBooth 微调 Stable Diffusion 的时候提升结果。

在开始之前，请注意该方法禁止应用在恶意行为上，来生成一些有害的东西，或者在没有相关背景下冒充某人。该模型的训练参照 [CreativeML Open RAIL-M 许可](https://huggingface.co/spaces/CompVis/stable-diffusion-license)。

注意：该帖子的先前版本已出版为 [W＆B 报告](https://wandb.ai/psuraj/dreambooth/reports/Dreambooth-Training-Analysis--VmlldzoyNzk0NDc3)

TL;DR: 推荐设置
-----------

*   DreamBooth 很容易快速过拟合，为了获取高质量图片，我们必须找到一个 "sweet spot" 在训练步骤和学习率之间。我们推荐使用低学习率和逐步增加步数直到达到比较满意的状态策略； 
*   DreamBooth 需要更多的脸部训练步数。在我们的实验中，当 BS 设置为 2，学习率设置为 1e-6，800-1200 步训练的很好；
*   先前提到的对于当训练脸部时避免过拟合非常重要，但对于其他主题可能影响就没那么大了；
*   如果你看到生成的图片噪声很大质量很低。这通常意味着过拟合了。首先，先尝试上述步骤去避免他，如果生成的图片依旧充满噪声。使用 DDIM 调度器或者运行更多推理步骤 (对于我们的实验大概 100 左右就很好了)；
*   训练文本编码器对于 UNet 的质量有很大影响。我们最优的实验配置包括使用文本编码器微调，低学习率和一个适合的步数。但是，微调文本编码器需要更多的内存，所以理想设置是一个至少 24G 显存的 GPU。使用像 8bit adam、fp 16 或梯度累计技巧有可能在像 Colab 或 Kaggle 提供的 16G 的 GPU 上训练；
*   EMA 对于微调不重要；
*   没有必要用 sks 词汇训练 DreamBooth。最早的实现之一是因为它在词汇中是罕见的 token ，但实际上是一种 rifle。我们的实验或其他像 [@nitrosocke](https://huggingface.co/nitrosocke) 的例子都表明使用自然语言描述你的目标是没问题的。
    

学习率影响
-----

DreamBooth 很容易过拟合，为了获得好的结果，设置针对你数据集合理的学习率和训练步数。在我们的实验中 (细节如下)，我们微调了四种不同的数据集用不同的高或低的学习率。总的来说，我们在低学习率的情况下获得了更好的结果。

实验设置
----

所有的实验使用 [`train_deambooth.py` 脚本](https://github.com/huggingface/diffusers/tree/main/examples/DreamBooth)，使用 `AdamW` 优化器在 2X40G 的 A00 机器上运行。我们采用相同的随机种子和保持所有超参相同，除了学习率，训练步骤和先前保留配置。

对于前三个例子 (不同对象)，我们微调模型配置为 bs = 4 (每个 GPU 分 2 个)，400 步。一个高学习率 = `5e-6`，一个低学习率 = `2e-6`。无先前保留配置。

最后一个实验尝试把人加入模型，我们使用先去保留配置同时 bs = 2 (每个 GPU 分 1 个)，800-1200 步。一个高学习率 = `5e-6`，一个低学习率 = `2e-6`。

你可以使用 8bit adam，`fp16` 精度训练，梯度累计去减少内存的需要，并执行相同的实验在一个 16G 显存的机器上。

### Toy 猫

高学习率 (`5e-6`)

![Toy 猫, 高学习率](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/1_cattoy_hlr.jpg)

低学习率 (`2e-6`)

![Toy 猫, 低学习率](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/2_cattoy_llr.jpg)

### 猪的头

高学习率 (`5e-6`) 请注意，颜色伪影是噪声残留物-运行更多的推理步骤可以帮助解决其中一些细节。

![猪的头, 高学习率](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/3_pighead_hlr.jpg)

低学习率 (`2e-6`)

![猪的头, 低学习率](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/4_pighead_llr.jpg)

### 土豆先生的头

高学习率 (`5e-6`) 请注意，颜色伪像是噪声残余物 - 运行更多的推理步骤可以帮助解决其中一些细节

![土豆先生的头, 高学习率](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/5_potato_hlr.jpg)

低学习率 (`2e-6`)

![土豆先生的头, 低学习率](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/6_potato_llr.jpg)

### 人脸

我们试图将 Seinfeld 的 Kramer 角色纳入 Stable Diffusion 中。如前所述，我们培训了更小的批量尺寸的更多步骤。即使这样，结果也不是出色的。为了简洁起见，我们省略了这些示例图像，并将读者推迟到下一部分，在这里，面部训练成为我们努力的重点。

### 初始化结果总结

为了用 DreamBooth 获取更好的 Stable Diffusion 结果，针对你的数据集调整你的学习率和训练步数非常重要。

*   高学习率多训练步数会导致过拟合。无论使用什么提示，该模型将主要从训练数据中生成图像
*   低学习率少训练步骤会导致欠拟合。该模型将无法生成我们试图组合的概念
    

脸部训练非常困难，在我们的实验中，学习率在 2e-6 同时 400 步对于物体已经很好了，但是脸部需要学习率在 1e-6 (或者 2e-6) 同时 1200 步才行。

如果发生以下情况，模型过度拟合，则图像质量会降低很多:

*   学习率过高
*   训练步数过多
*   对于面部的情况，如下一部分所示，当不使用事先保存时
    

训练脸部使用先前配置
----------

先前的保存是一种使用我们试图训练的同一类的其他图像作为微调过程的一部分。例如，如果我们尝试将新人纳入模型，我们要保留的类可能是人。事先保存试图通过使用新人的照片与其他人的照片相结合来减少过度拟合。好处是，我们可以使用 Stable Diffusion 模型本身生成这些其他类图像！训练脚本如果需要的话会自动处理这一点，但是你还可以为文件夹提供自己的先前保存图像

先前配置，1200 步数，学习率 = `2e-6`

![Faces, 先前配置](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/7_faces_with_prior.jpg)

无先前配置，1200 步数，学习率 = `2e-6`

![Faces, 无先前配置](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/8_faces_no_prior.jpg)

如你所见，当使用先前配置时，结果会更好，但是仍然有嘈杂的斑点。是时候做一些其他技巧了

调度程序的效果
-------

在前面的示例中，我们使用 `PNDM` 调度程序在推理过程中示例图像。我们观察到，当模型过度时，`DDIM` 通常比 `PNDM` 和 `LMSDISCRETE` 好得多。此外，通过推断更多步骤可以提高质量：100 似乎是一个不错的选择。附加步骤有助于将一些噪声贴在图像详细信息中。

PNDM, Kramer 脸

![PNDM Cosmo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/9_cosmo_pndm.jpg)

`LMSDiscrete`, Kramer 脸。结果很糟糕

![LMSDiscrete Cosmo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/a_cosmo_lmsd.jpg)

`DDIM`, Kramer 脸。效果好多了

![DDIM Cosmo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/b_cosmo_ddim.jpg)

对于其他主题，可以观察到类似的行为，尽管程度较小。

`PNDM`, 土豆头

![PNDM 土豆头](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/c_potato_pndm.jpg)

`LMSDiscrete`, 土豆头

![LMSDiscrite 土豆头](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/d_potato_lmsd.jpg)

`DDIM`, 土豆头

![DDIM 土豆头](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/e_potato_ddim.jpg)

微调文本编码器
-------

原始的 DreamBooth 论文讲述了一个微调 UNet 网络部分但是冻结文本编码部分的方法。然而我们观察到微调文本编码会获得更好的效果。在看到其他 DreamBooth 实施中使用的方法后，我们尝试了这种方法，结果令人惊讶！

冻结文本编码器

![冻结文本编码器](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/f_froxen_encoder.jpg)

微调文本编码器

![微调文本编码器](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/g_unfrozen_encoder.jpg)

微调文本编码器会产生最佳结果，尤其是脸。它生成更现实的图像，不太容易过度拟合，并且还可以更好地提示解释性，能够处理更复杂的提示。

后记：Textual Inversion + DreamBooth
---------------------------------

我们还进行了最后一个实验，将 [Textual Inversion](https://textual-inversion.github.io/) 与 DreamBooth 结合在一起。两种技术都有相似的目标，但是它们的方法不同。

在本次实验中我们首先用 Textual Inversion 跑了 2000 步。接着那个模型我们又跑了 DreamBooth 额外的 500 步，学习率为 1e-6。结果如下：

![Textual Inversion + Dreambooth](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/h_textual_inversion_dreambooth.jpg)

我们认为，结果比进行简单的 DreamBooth 要好得多，但不如我们调整整个文本编码器时那样好。它似乎可以更多地复制训练图像的样式，因此对它们可能会过度拟合。我们没有进一步探索这种组合，但是这可能是改善 DreamBooth 适合 16GB GPU 的过程的有趣替代方法。欢迎随时探索并告诉我们你的结果！
