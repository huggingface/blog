---
title: 使用 Diffusers 通过 Dreambooth 技术来训练 Stable Diffusion
thumbnail: /blog/assets/sd_dreambooth_training/thumbnail.jpg
authors:
- user: valhalla
- user: pcuenq
- user: 9of9
- user: youyuanrsq
  guest: true
translators:
- user: innovation64
- user: inferjay
  proofreader: true
---

# 使用 Diffusers 通过 Dreambooth 技术来训练 Stable Diffusion


[Dreambooth](https://dreambooth.github.io/) 是一种使用特殊的微调方式来教会 [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) 新概念的技术。利用这个技术，有的人仅仅用他们自己很少的照片就将自己置身于奇妙的境界之中，而有些人则结合它生成新的风格。🧨Diffusers提供一个 [DreamBooth 训练脚本](https://github.com/huggingface/diffusers/tree/main/examples/DreamBooth)。使用这个脚本训练不会花费很长的时间，但是比较难筛选正确的超参数，并且容易过拟合。

> [Dreambooth](https://dreambooth.github.io/) is a technique to teach new concepts to [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) using a specialized form of fine-tuning. Some people have been using it with a few of their photos to place themselves in fantastic situations, while others are using it to incorporate new styles. [🧨 Diffusers](https://github.com/huggingface/diffusers) provides a Dreambooth [training script](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). It doesn't take long to train, but it's hard to select the right set of hyperparameters and it's easy to overfit.

我们做了许多实验来分析不同参数配置下`DreamBooth`的效果。本文展示了我们的发现和一些小技巧来帮助你在用 `DreamBooth`微调`Stable Diffusion`的时候提升（生成图片的）效果。

> We conducted a lot of experiments to analyze the effect of different settings in Dreambooth. This post presents our findings and some tips to improve your results when fine-tuning Stable Diffusion with Dreambooth.

在开始之前，请注意该方法禁止应用在恶意行为上，来生成一些有害的东西，或者在没有相关背景下冒充某人。该模型的训练参照 [CreativeML Open RAIL-M 许可](https://huggingface.co/spaces/CompVis/stable-diffusion-license)。

> Before we start, please be aware that this method should never be used for malicious purposes, to generate harm in any way, or to impersonate people without their knowledge. Models trained with it are still bound by the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license) that governs distribution of Stable Diffusion models.

注意：该帖子的先前版本已出版为 [W＆B 报告](https://wandb.ai/psuraj/dreambooth/reports/Dreambooth-Training-Analysis--VmlldzoyNzk0NDc3)

> _Note: a previous version of this post was published [as a W&B report](https://wandb.ai/psuraj/dreambooth/reports/Dreambooth-Training-Analysis--VmlldzoyNzk0NDc3)_.

## TL;DR: 推荐的设置（TL;DR: Recommended Settings）

* `DreamBooth`很容易快速过拟合，为了获取高质量图片，我们必须在训练步骤（steps）和学习率之间找到一个 "sweet spot"。我们推荐使用较小的学习率以及逐步增加步数直到得到比较满意的结果的策略；
* > Dreambooth tends to overfit quickly. To get good-quality images, we must find a 'sweet spot' between the number of training steps and the learning rate. We recommend using a low learning rate and progressively increasing the number of steps until the results are satisfactory.
* 对于脸部图像而言`DreamBooth`需要更多的训练步数（steps）。在我们的实验中，当batch size设置为2，学习率设置为`1e-6`时，将steps设置为800-1200步效果不错；
* > Dreambooth needs more training steps for faces. In our experiments, 800-1200 steps worked well when using a batch size of 2 and LR of 1e-6.
* 当针对脸部图像（生成）进行训练时，事先保存（prior perservation）对于避免过拟合非常重要，但对于其他主题可能影响就没那么大了；
* > Prior preservation is important to avoid overfitting when training on faces. For other subjects, it doesn't seem to make a huge difference.
* 如果你看到生成的图片噪声很大且质量很低。这通常意味着过拟合了。首先，先尝试上述步骤去避免他，如果生成的图片依旧充满噪声。使用`DDIM`调度器（scheduler）或者迭代更多推理步骤（steps）（对于我们的实验大概 100 左右就很好了）；
* > If you see that the generated images are noisy or the quality is degraded, it likely means overfitting. First, try the steps above to avoid it. If the generated images are still noisy, use the DDIM scheduler or run more inference steps (~100 worked well in our experiments).
* 训练文本编码器而不是（训练）`Unet`对（图像的）质量有很大影响。我们最好的结果是通过使用文本编码器微调、较低的学习率和适当的步数的组合来获得的。但是，微调文本编码器需要更多的内存，因此至少具有24GB内存的GPU。使用8位`Adam`、fp16训练或梯度累积等技术，可以在像Google Colab或Kaggle提供的16GB GPU上进行训练。
* > Training the text encoder in addition to the UNet has a big impact on quality. Our best results were obtained using a combination of text encoder fine-tuning, low LR, and a suitable number of steps. However, fine-tuning the text encoder requires more memory, so a GPU with at least 24 GB of RAM is ideal. Using techniques like 8-bit Adam, `fp16` training or gradient accumulation, it is possible to train on 16 GB GPUs like the ones provided by Google Colab or Kaggle.
* `EMA`对于微调不重要；
* > Fine-tuning with or without EMA produced similar results.
* 没有必要用`sks`词汇训练`DreamBooth`。最早的实现使用它是因为这个token在词汇中很罕见，但实际上是一种 rifle。我们的实验或其他像 [@nitrosocke](https://huggingface.co/nitrosocke) 的例子都表明使用自然语言描述你的目标就足够了。
* > There's no need to use the `sks` word to train Dreambooth. One of the first implementations used it because it was a rare token in the vocabulary, but it's actually a kind of rifle. Our experiments, and those by for example [@nitrosocke](https://huggingface.co/nitrosocke) show that it's ok to select terms that you'd naturally use to describe your target.

## 学习率的影响（Learning Rate Impact）

`Dreambooth`很容易过拟合。为了获得良好的结果，请调整学习率和训练步数来适配你的数据集。在我们的实验中（详见下文），我们使用较高和较低的学习率对四个不同的数据集进行了微调。在所有情况下，我们都使用较低的学习率获得了更好的结果。

> Dreambooth overfits very quickly. To get good results, tune the learning rate and the number of training steps in a way that makes sense for your dataset. In our experiments (detailed below), we fine-tuned on four different datasets with high and low learning rates. In all cases, we got better results with a low learning rate.

## 实验设置（Experiments Settings）

我们所有的实验都是在使用`AdamW` 优化器的 [`train_deambooth.py` 脚本](https://github.com/huggingface/diffusers/tree/main/examples/DreamBooth)上，使用2个40GB的A100上进行的。我们在所有运行中都使用相同的种子和相同的超参数，除了学习率、训练步数和是否使用事先保存（prior preservation）。

> All our experiments were conducted using the [`train_dreambooth.py`](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) script with the `AdamW` optimizer on 2x 40GB A100s. We used the same seed and kept all hyperparameters equal across runs, except LR, number of training steps and the use of prior preservation.

对于前三个示例（各种物体），我们设置batch size大小为4（每个GPU为2）并进行了400步的微调。我们使用了高学习率`5e-6`和低学习率`2e-6`。没有使用事先保存（prior preservation）。

> For the first 3 examples (various objects), we fine-tuned the model with a batch size of 4 (2 per GPU) for 400 steps. We used a high learning rate of `5e-6` and a low learning rate of `2e-6`. No prior preservation was used.

最后一个实验尝试把人加入模型，我们使用事先保存（prior preservation）并将batch size设置为2 (每个GPU分1个)，训练800-1200步。我们使用的高学习率为`5e-6`，低学习率为`2e-6`。

> The last experiment attempts to add a human subject to the model. We used prior preservation with a batch size of 2 (1 per GPU), 800 and 1200 steps in this case. We used a high learning rate of `5e-6` and a low learning rate of `2e-6`.

你可以使用8bit `Adam`，`fp16` 精度训练或者梯度累计去减少内存的需要，并在一个16G显存的机器上运行相同的实验。

> Note that you can use 8-bit Adam, `fp16` training or gradient accumulation to reduce memory requirements and run similar experiments on GPUs with 16 GB of memory.

### Toy 猫（Cat Toy）

高学习率 High Learning Rate(`5e-6`)

![Cat Toy, High Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/1_cattoy_hlr.jpg)

低学习率 Low Learning Rate (`2e-6`)

![Cat Toy, Low Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/2_cattoy_llr.jpg)

### 猪头（Pighead）

高学习率 (`5e-6`) 请注意，颜色伪影是噪声残留物-运行更多的推理步骤可以帮助解决其中一些细节问题。

> High Learning Rate (`5e-6`). Note that the color artifacts are noise remnants – running more inference steps could help resolve some of those details.

![Pighead, High Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/3_pighead_hlr.jpg)

低学习率 Low Learning Rate(`2e-6`)

![Pighead, Low Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/4_pighead_llr.jpg)

### 土豆先生的头（Mr. Potato Head）

高学习率 (`5e-6`) 请注意，颜色伪像是噪声残余物-运行更多的推理步骤可以帮助解决其中一些细节问题。

> High Learning Rate (`5e-6`). Note that the color artifacts are noise remnants – running more inference steps could help resolve some of those details.

![Potato Head, High Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/5_potato_hlr.jpg)

低学习率 Low Learning Rate(`2e-6`)

![Potato Head, Low Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/6_potato_llr.jpg)

### 人脸（Human Face）

我们尝试将Seinfeld中的Kramer角色融入到`Stable Diffusion`中。正如之前提到的，我们使用更小的batch size训练了更多的步骤。即便如此，结果也不是特别出色。为了简洁起见，我们省略了这些示例图像，并将读者推荐到接下来的部分，其中人脸训练将成为我们努力的重点。

> We tried to incorporate the Kramer character from Seinfeld into Stable Diffusion. As previously mentioned, we trained for more steps with a smaller batch size. Even so, the results were not stellar. For the sake of brevity, we have omitted these sample images and defer the reader to the next sections, where face training became the focus of our efforts.

### 初步结果总结（Summary of Initial Results）

为了用`DreamBooth`获取更好的`Stable Diffusion`结果，针对你的数据集调整你的学习率和训练步数非常重要。

> To get good results training Stable Diffusion with Dreambooth, it's important to tune the learning rate and training steps for your dataset.

* 学习率过高和过多的训练步骤会导致过拟合。无论使用什么提示，模型大多会生成与训练数据相似的图像。
* > High learning rates and too many training steps will lead to overfitting. The model will mostly generate images from your training data, no matter what prompt is used.
* 学习率过低和步骤过少会导致欠拟合：模型无法生成我们尝试融合的概念。
* > Low learning rates and too few steps will lead to underfitting: the model will not be able to generate the concept we were trying to incorporate.

面部更难训练。在我们的实验中，学习率为`2e-6`，400个训练步骤对于物体表现良好，但对于面部需要`1e-6`（或`2e-6`）和约1200步才行。

> Faces are harder to train. In our experiments, a learning rate of `2e-6` with `400` training steps works well for objects but faces required `1e-6` (or `2e-6`) with ~1200 steps.

如果模型过度拟合，图像质量会严重降低，这会发生在以下情况下:

* 学习率过高
* 训练步数过多
* 对于面部而言，如果不使用事先保存（prior preservation），如下一节所示，也会发生过拟合

> Image quality degrades a lot if the model overfits, and this happens if:
> * The learning rate is too high.
> * We run too many training steps.
> * In the case of faces, when no prior preservation is used, as shown in the next section.

## 在训练人脸时使用事先保存（Using Prior Preservation when training Faces）

事先保存（prior preservation）是一种技术，它使用同一类别的额外图像作为微调过程的一部分。例如，如果我们尝试将一个新的人物融入到模型中，我们想要保存的*类别*可能是*人*。事先保存（prior preservation）通过使用新人物的照片与其他人的照片相结合来减少过拟合。好处在于，我们可以使用`Stable Diffusion`模型生成这些额外的类别图像！如果你愿意，训练脚本会自动处理这些额外的类别图像，但你也可以提供一个包含自己先前保存图像的文件夹。

> Prior preservation is a technique that uses additional images of the same class we are trying to train as part of the fine-tuning process. For example, if we try to incorporate a new person into the model, the _class_ we'd want to preserve could be _person_. Prior preservation tries to reduce overfitting by using photos of the new person combined with photos of other people. The nice thing is that we can generate those additional class images using the Stable Diffusion model itself! The training script takes care of that automatically if you want, but you can also provide a folder with your own prior preservation images.

事先保存（prior preservation），1200 步数，学习率 = `2e-6` （Prior preservation, 1200 steps, lr=`2e-6`）

![Faces, prior preservation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/7_faces_with_prior.jpg)

无事先保存，1200 步数，学习率 = `2e-6`（No prior preservation, 1200 steps, lr=`2e-6`）

![Faces, prior preservation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/8_faces_no_prior.jpg)

如你所见，当使用事先保存（prior preservation）时，结果会更好，但是仍然有嘈杂的斑点。是时候使用一些其他技巧了。

> As you can see, results are better when prior preservation is used, but there are still noisy blotches. It's time for some additional tricks!

## 调度器的影响（Effect of Schedulers）

在之前的示例中，我们使用`PNDM`调度器在推理过程中对图像进行采样。我们观察到，当模型出现过拟合时，`DDIM`通常比`PNDM`和`LMSDiscrete`表现更好。此外，可以通过运行更多步骤来提高质量：100步似乎是一个不错的选择。额外的步骤有助于将一些噪声补丁转化为图像细节。

> In the previous examples, we used the `PNDM` scheduler to sample images during the inference process. We observed that when the model overfits, `DDIM` usually works much better than `PNDM` and `LMSDiscrete`. In addition, quality can be improved by running inference for more steps: 100 seems to be a good choice. The additional steps help resolve some of the noise patches into image details.

`PNDM`, Kramer脸部 （`PNDM`, Kramer face）

![PNDM Cosmo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/9_cosmo_pndm.jpg)

`LMSDiscrete`, Kramer脸部。结果很糟糕 （`LMSDiscrete`, Kramer face. Results are terrible!）

![LMSDiscrete Cosmo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/a_cosmo_lmsd.jpg)

`DDIM`, Kramer脸部。效果好多了（`DDIM`, Kramer face. Much better）

![DDIM Cosmo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/b_cosmo_ddim.jpg)

对于其他物体，可以观察到类似的行为，尽管程度较小。

> A similar behaviour can be observed for other subjects, although to a lesser extent.

`PNDM`, 土豆头（`PNDM`, Potato Head）

![PNDM 土豆头](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/c_potato_pndm.jpg)

`LMSDiscrete`, 土豆头（`LMSDiscrete`, Potato Head）

![LMSDiscrite Potato](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/d_potato_lmsd.jpg)

`DDIM`, 土豆头（`DDIM`, Potato Head）

![DDIM Potato](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/e_potato_ddim.jpg)

## 微调文本编码器（Fine-tuning the Text Encoder）

原始的`Dreambooth`论文描述了一种微调模型中`UNet`组件的方法，但是保持文本编码器不变。然而，我们观察到微调编码器可以产生更好的结果。在看到其他`Dreambooth`的实现中使用该方法后，我们进行了实验，结果非常显著！

> The original Dreambooth paper describes a method to fine-tune the UNet component of the model but keeps the text encoder frozen. However, we observed that fine-tuning the encoder produces better results. We experimented with this approach after seeing it used in other Dreambooth implementations, and the results are striking!

冻结文本编码器（Frozen text encoder）

![Frozen text encoder](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/f_froxen_encoder.jpg)

微调文本编码器（Fine-tuned text encoder）

![Fine-tuned text encoder](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/g_unfrozen_encoder.jpg)

微调文本编码器可以产生最佳结果，特别是对于面部。它生成更加逼真的图像，更不容易过拟合，同时还可以实现更好的提示可解释性，能够处理更复杂的提示。

> Fine-tuning the text encoder produces the best results, especially with faces. It generates more realistic images, it's less prone to overfitting and it also achieves better prompt interpretability, being able to handle more complex prompts.

## 后记：Textual Inversion + DreamBooth（Epilogue: Textual Inversion + Dreambooth）

我们还进行了最后一个实验，将[Textual Inversion](https://textual-inversion.github.io/)与`Dreambooth`相结合。这两种技术具有类似的目标，但它们的方法不同。

> We also ran a final experiment where we combined [Textual Inversion](https://textual-inversion.github.io) with Dreambooth. Both techniques have a similar goal, but their approaches are different.

在这个实验中，我们首先运行了2000步的Textual Inversion。然后从该模型中，我们使用学习率为`1e-6`的`Dreambooth`进行了额外的500步训练。以下是结果：

> In this experiment we first ran textual inversion for 2000 steps. From that model, we then ran Dreambooth for an additional 500 steps using a learning rate of `1e-6`. These are the results:

![Textual Inversion + Dreambooth](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/h_textual_inversion_dreambooth.jpg)

我们认为这些结果比仅使用`Dreambooth`要好得多，但不如微调整个文本编码器时那么好。它似乎更多地复制了训练图像的风格，所以可能出现了过拟合现象。我们没有进一步探索这种组合，但它可能是一个有趣的替代方案，可以改进`Dreambooth`并这个方案仍可以在16GB的GPU上运行。请随意探索并告诉我们您的结果！

> We think the results are much better than doing plain Dreambooth but not as good as when we fine-tune the whole text encoder. It seems to copy the style of the training images a bit more, so it could be overfitting to them. We didn't explore this combination further, but it could be an interesting alternative to improve Dreambooth and still fit the process in a 16GB GPU. Feel free to explore and tell us about your results!