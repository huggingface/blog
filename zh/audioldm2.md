---
title: "AudioLDM 2，加速⚡️！" 
thumbnail: /blog/assets/161_audioldm2/thumbnail.png
authors:
- user: sanchit-gandhi
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# AudioLDM 2，加速⚡️！

<!-- {blog_metadata} -->
<!-- {authors} -->

<a target="_blank" href="https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/AudioLDM-2.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt=" 在 Colab 中打开 "/>
</a>

AudioLDM 2 由刘濠赫等人在 [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) 一文中提出。 AudioLDM 2 接受文本提示作为输入并输出对应的音频，其可用于生成逼真的声效、人类语音以及音乐。

虽然生成的音频质量很高，但基于其原始实现进行推理的速度非常慢: 生成一个 10 秒的音频需要 30 秒以上的时间。慢的原因是多重的，包括其使用了多阶段建模、checkpoint 较大以及代码尚未优化等。

本文将展示如何在 Hugging Face 🧨 Diffusers 库中使用 AudioLDM 2，并在此基础上探索一系列代码优化 (如半精度、Flash 注意力、图编译) 以及模型级优化 (如选择合适的调度器及反向提示)。最终我们将推理时间降低了 **10 倍** 多，且对输出音频质量的影响最低。本文还附有一个更精简的 [Colab notebook](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/AudioLDM-2.ipynb)，这里面包含所有代码但精简了很多文字部分。

最终，我们可以在短短 1 秒内生成一个 10 秒的音频！

## 模型概述

受 [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) 的启发，AudioLDM 2 是一种文生音频的 _ 隐扩散模型 (latent diffusion model，LDM)_，其可以将文本嵌入映射成连续的音频表征。

大体的生成流程总结如下:

1. 给定输入文本 $\boldsymbol{x}$，使用两个文本编码器模型来计算文本嵌入: [CLAP](https://huggingface.co/docs/transformers/main/en/model_doc/clap) 的文本分支，以及 [Flan-T5](https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5) 的文本编码器。

    $$\boldsymbol{E} _{1} = \text{CLAP}\left(\boldsymbol{x} \right); \quad \boldsymbol{E}_ {2} = \text{T5}\left(\boldsymbol{x}\right)
    $$

    CLAP 文本嵌入经过训练，可以与对应的音频嵌入对齐，而 Flan-T5 嵌入可以更好地表征文本的语义。

2. 这些文本嵌入通过各自的线性层投影到同一个嵌入空间:

    $$\boldsymbol{P} _{1} = \boldsymbol{W}_ {\text{CLAP}} \boldsymbol{E} _{1}; \quad \boldsymbol{P}_ {2} = \boldsymbol{W} _{\text{T5}}\boldsymbol{E}_ {2}
    $$

    在 `diffusers` 实现中，这些投影由 [AudioLDM2ProjectionModel](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2/AudioLDM2ProjectionModel) 定义。

3. 使用 [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2) 语言模型 (LM) 基于 CLAP 和 Flan-T5 嵌入自回归地生成一个含有 $N$ 个嵌入向量的新序列:

    $$\tilde{\boldsymbol{E}} _{i} = \text{GPT2}\left(\boldsymbol{P}_ {1}, \boldsymbol{P} _{2}, \tilde{\boldsymbol{E}}_ {1:i-1}\right) \qquad \text{for } i=1,\dots,N$$
    
4. 以生成的嵌入向量 $\tilde{\boldsymbol{E}} _{1:N}$ 和 Flan-T5 文本嵌入 $\boldsymbol{E}_ {2}$ 为条件，通过 LDM 的反向扩散过程对随机隐变量进行 _去噪_ 。LDM 在反向扩散过程中运行 $T$ 个步推理:

    $$\boldsymbol{z} _{t} = \text{LDM}\left(\boldsymbol{z}_ {t-1} | \tilde{\boldsymbol{E}} _{1:N}, \boldsymbol{E}_ {2}\right) \qquad \text{for } t = 1, \dots, T$$

    其中初始隐变量 $\boldsymbol{z} _{0}$ 是从正态分布 $\mathcal{N} \left(\boldsymbol{0}, \boldsymbol{I} \right )$ 中采样而得。 LDM 的 [UNet](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2/AudioLDM2UNet2DConditionModel) 的独特之处在于它需要 **两组** 交叉注意力嵌入，来自 GPT2 语言模型的 $\tilde{\boldsymbol{E}}_ {1:N}$ 和来自 Flan-T5 的  $\boldsymbol{E}_{2}$，而其他大多数 LDM 只有一个交叉注意力条件。

5. 把最终去噪后的隐变量 $\boldsymbol{z}_{T}$ 传给 VAE 解码器以恢复梅尔谱图 $\boldsymbol{s}$:

    $$
    \boldsymbol{s} = \text{VAE} _{\text{dec}} \left(\boldsymbol{z}_ {T}\right)
    $$

6. 梅尔谱图被传给声码器 (vocoder) 以获得输出音频波形 $\mathbf{y}$:

    $$
    \boldsymbol{y} = \text{Vocoder}\left(\boldsymbol{s}\right)
    $$

下图展示了文本输入是如何作为条件传递给模型的，可以看到在 LDM 中两个提示嵌入均被用作了交叉注意力的条件:

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/161_audioldm2/audioldm2.png?raw=true" width="600"/>
</p>

有关如何训练 AudioLDM 2 模型的完整的详细信息，读者可以参阅 [AudioLDM 2 论文](https://arxiv.org/abs/2308.05734)。

Hugging Face 🧨 Diffusers 提供了一个端到端的推理流水线类 [`AudioLDM2Pipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2) 以将该模型的多阶段生成过程包装到单个可调用对象中，这样用户只需几行代码即可完成从文本生成音频的过程。

AudioLDM 2 有三个变体。其中两个 checkpoint 适用于通用的文本到音频生成任务，第三个 checkpoint 专门针对文本到音乐生成。三个官方 checkpoint 的详细信息请参见下表，这些 checkpoint 都可以在 [Hugging Face Hub](https://huggingface.co/models?search=cvssp/audioldm2) 上找到:

| checkpoint                                                            | 任务          | 模型大小 | 训练数据（单位：小时） |
|-----------------------------------------------------------------------|---------------|------------|-------------------|
| [cvssp/audioldm2](https://huggingface.co/cvssp/audioldm2)             | 文生音频 | 1.1B       | 1150k             |
| [cvssp/audioldm2-music](https://huggingface.co/cvssp/audioldm2-music) | 文生音乐 | 1.1B       | 665k              |
| [cvssp/audioldm2-large](https://huggingface.co/cvssp/audioldm2-large) | 文生音频 | 1.5B       | 1150k             |

至此，我们已经全面概述了 AudioLDM 2 生成的工作原理，接下来让我们将这一理论付诸实践！

## 加载流水线

我们以基础版模型 [cvssp/audioldm2](https://huggingface.co/cvssp/audioldm2) 为例，首先使用 [`.from_pretrained`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained) 方法来加载整个管道，该方法会实例化管道并加载预训练权重:

```python
from diffusers import AudioLDM2Pipeline

model_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(model_id)
```

**输出:**

```
Loading pipeline components...: 100%|███████████████████████████████████████████| 11/11 [00:01<00:00, 7.62it/s]
```

与 PyTorch 一样，使用 `to` 方法将流水线移至 GPU:

```python
pipe.to("cuda");
```

现在，我们来定义一个随机数生成器并固定一个种子，我们可以通过这种方式来固定 LDM 模型中的起始隐变量从而保证结果的可复现性，并可以观察不同提示对生成过程和结果的影响:

```python
import torch

generator = torch.Generator("cuda").manual_seed(0)
```

现在，我们准备好开始第一次生成了！本文中的所有实验都会使用固定的文本提示以及相同的随机种子来生成音频，并比较不同方案的延时和效果。 [`audio_length_in_s`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2Pipeline.__call__.audio_length_in_s) 参数主要控制所生成音频的长度，这里我们将其设置为默认值，即 LDM 训练时的音频长度: 10.24 秒:

```python
prompt = "The sound of Brazilian samba drums with waves gently crashing in the background"

audio = pipe(prompt, audio_length_in_s=10.24, generator=generator).audios[0]
```

**输出:**

```
100%|███████████████████████████████████████████| 200/200 [00:13<00:00, 15.27it/s]
```

酷！我们花了大约 13 秒最终生成出了音频。我们来听一下:

```python
from IPython.display import Audio

Audio(audio, rate=16000)
```

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/161_audioldm2/sample_1.wav" type="audio/wav">
浏览器不支持音频元素。
</audio>

听起来跟我们的文字提示很吻合！质量很好，但是有一些背景噪音。我们可以为流水线提供 [_反向提示 (negative prompt)_](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2Pipeline.__call__.negative_prompt)，以防止其生成的音频中含有某些不想要特征。这里，我们给模型一个反向提示，以防止模型生成低质量的音频。我们不设 `audio_length_in_s` 参数以使用其默认值:

```python
negative_prompt = "Low quality, average quality."

audio = pipe(prompt, negative_prompt=negative_prompt, generator=generator.manual_seed(0)).audios[0]
```

**输出:**

```
100%|███████████████████████████████████████████| 200/200 [00:12<00:00, 16.50it/s]
```

使用反向提示 ${}^1$ 时，推理时间不变; 我们只需将 LDM 的无条件输入替换为反向提示即可。这意味着我们在音频质量方面获得的任何收益都是免费的。

我们听一下生成的音频:

```python
Audio(audio, rate=16000)
```

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/161_audioldm2/sample_2.wav" type="audio/wav">
浏览器不支持音频元素。
</audio>

显然，整体音频质量有所改善 - 噪声更少，并且音频整体听起来更清晰。

${}^1$ 请注意，在实践中，我们通常会看到第二次生成比第一次生成所需的推理时间有所减少。这是由于我们第一次运行计算时 CUDA 被“预热”了。因此一般进行基准测试时我们会选择第二次推理的时间作为结果。

## 优化 1: Flash 注意力

PyTorch 2.0 及更高版本包含了一个优化过的内存高效的注意力机制的实现，用户可通过 [`torch.nn.function.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) (SDPA) 函数来调用该优化。该函数会根据输入自动使能多个内置优化，因此比普通的注意力实现运行得更快、更节省内存。总体而言，SDPA 函数的优化与 Dao 等人在论文 [Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) 中所提出的 _flash 注意力_ 类似。

如果安装了 PyTorch 2.0 且 `torch.nn.function.scaled_dot_product_attention` 可用，Diffusers 将默认启用该函数。因此，仅需按照 [官方说明](https://pytorch.org/get-started/locally/) 安装 torch 2.0 或更高版本，不需对流水线🚀作任何改动，即能享受提速。

```python
audio = pipe(prompt, negative_prompt=negative_prompt, generator=generator.manual_seed(0)).audios[0]
```

**输出:**

```
100%|███████████████████████████████████████████| 200/200 [00:12<00:00, 16.60it/s]
```

有关在 `diffusers` 中使用 SDPA 的更多详细信息，请参阅相应的 [文档](https://huggingface.co/docs/diffusers/optimization/torch2.0)。

## 优化 2: 半精度

默认情况下， `AudioLDM2Pipeline` 以 float32 (全) 精度方式加载模型权重。所有模型计算也以 float32 精度执行。对推理而言，我们可以安全地将模型权重和计算转换为 float16 (半) 精度，这能改善推理时间和 GPU 内存，同时对生成质量的影响微乎其微。

我们可以通过将 `from_pretrained` 的 [`torch_dtype`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained.torch_dtype) 参数设为 `torch.float16` 来加载半精度权重:

```python
pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.to("cuda");
```

我们运行一下 float16 精度的生成，并听一下输出:

```python
audio = pipe(prompt, negative_prompt=negative_prompt, generator=generator.manual_seed(0)).audios[0]

Audio(audio, rate=16000)
```

**输出:**

```
100%|███████████████████████████████████████████| 200/200 [00:09<00:00, 20.94it/s]
```

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/161_audioldm2/sample_3.wav" type="audio/wav">
浏览器不支持音频元素。
</audio>

音频质量与全精度生成基本没有变化，推理加速了大约 2 秒。根据我们的经验，使用具有 float16 精度的 `diffusers` 流水线，我们可以获得显著的推理加速而无明显的音频质量下降。因此，我们建议默认使用 float16 精度。

## 优化 3: Torch Compile

为了获得额外的加速，我们还可以使用新的 `torch.compile` 功能。由于在流水线中 UNet 通常计算成本最高，因此我们用 `torch.compile` 编译一下 UNet，其余子模型 (文本编码器和 VAE) 保持不变:

```python
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

用 `torch.compile` 包装 UNet 后，由于编译 UNet 的开销，我们运行第一步推理时通常会很慢。所以，我们先运行一步流水线预热，这样后面真正运行的时候就快了。请注意，第一次推理的编译时间可能长达 2 分钟，请耐心等待！

```python
audio = pipe(prompt, negative_prompt=negative_prompt, generator=generator.manual_seed(0)).audios[0]
```

**输出:**

```
100%|███████████████████████████████████████████| 200/200 [01:23<00:00, 2.39it/s]
```

很棒！现在 UNet 已编译完毕，现在可以以更快的速度运行完整的扩散过程了:

```python
audio = pipe(prompt, negative_prompt=negative_prompt, generator=generator.manual_seed(0)).audios[0]
```

**输出:**

```
100%|███████████████████████████████████████████| 200/200 [00:04<00:00, 48.98it/s]
```

只需 4 秒即可生成！在实践中，你只需编译 UNet 一次，然后就可以为后面的所有生成赢得一个更快的推理。这意味着编译模型所花费的时间可以由后续推理时间的收益所均摊。有关 `torch.compile` 的更多信息及选项，请参阅 [torch compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) 文档。

## 优化 4: 调度器

还有一个选项是减少推理步数。选择更高效的调度器可以帮助减少步数，而不会牺牲输出音频质量。你可以调用 [`schedulers.compatibles`](https://huggingface.co/docs/diffusers/v0.20.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 属性来查看哪些调度器与 `AudioLDM2Pipeline` 兼容:

```python
pipe.scheduler.compatibles
```

**输出:**

```
[diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
 diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler,
 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
 diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler,
 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
 diffusers.schedulers.scheduling_pndm.PNDMScheduler,
 diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler,
 diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler,
 diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
 diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler,
 diffusers.utils.dummy_torch_and_torchsde_objects.DPMSolverSDEScheduler,
 diffusers.schedulers.scheduling_ddim.DDIMScheduler,
 diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler,
 diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler]
```

好！现在我们有一长串的调度器备选📝。默认情况下，AudioLDM 2 使用 [`DDIMScheduler`](https://huggingface.co/docs/diffusers/api/schedulers/ddim)，其需要 200 个推理步才能生成高质量的音频。但是，性能更高的调度程序，例如 [`DPMSolverMultistepScheduler`](https://huggingface.co/docs/diffusers/main/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler)，
只需 **20-25 个推理步** 即可获得类似的结果。

让我们看看如何将 AudioLDM 2 调度器从 `DDIM` 切换到 `DPM Multistep` 。我们需要使用 [`ConfigMixin.from_config()`](https://huggingface.co/docs/diffusers/main/en/api/configuration#diffusers.ConfigMixin.from_config) 方法以用原始 [`DDIMScheduler`](https://huggingface.co/docs/diffusers/api/schedulers/ddim) 的配置来加载 [`DPMSolverMultistepScheduler`](https://huggingface.co/docs/diffusers/main/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler):

```python
from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
```

让我们将推理步数设为 20，并使用新的调度器重新生成。由于 LDM 隐变量的形状未更改，因此我们不必重编译:

```python
audio = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=20, generator=generator.manual_seed(0)).audios[0]
```

**输出:**

```
100%|███████████████████████████████████████████| 20/20 [00:00<00:00, 49.14it/s]
```

这次只用了不到 **1 秒** 就生成了音频！我们听下它的生成:

```python
Audio(audio, rate=16000)
```

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/161_audioldm2/sample_4.wav" type="audio/wav">
浏览器不支持音频元素。
</audio>

生成质量与原来的基本相同，但只花了原来时间的一小部分！ 🧨 Diffusers 流水线是“可组合”的，这个设计允许你轻松地替换调度器或其他组件以获得更高性能。

## 内存消耗如何？

我们想要生成的音频的长度决定了 LDM 中待去噪的隐变量的 _宽度_ 。由于 UNet 中交叉注意力层的内存随序列长度 (宽度) 的平方而变化，因此生成非常长的音频可能会导致内存不足错误。我们还可以通过 batch size 来控制生成的样本数，进而控制内存使用。

如前所述，以 float16 半精度加载模型可以节省大量内存。使用 PyTorch 2.0 SDPA 也可以改善内存占用，但这部分改善对超长序列长度来讲可能不够。

我们来试着生成一个 2.5 分钟 (150 秒) 的音频。我们通过设置 [`num_waveforms_per_prompt`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2Pipeline.__call__.num_waveforms_per_prompt) `=4` 来生成 4 个候选音频。一旦 [`num_waveforms_per_prompt`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2Pipeline.__call__.num_waveforms_per_prompt) `>1` ，在生成的音频和文本提示之间会有一个自动评分机制: 将音频和文本提示嵌入到 CLAP 音频文本嵌入空间中，然后根据它们的余弦相似度得分进行排名。生成的音频中第 `0` 个音频就是分数“最高”的音频。

由于我们更改了 UNet 中隐变量的宽度，因此我们必须使用新的隐变量形状再执行一次 torch 编译。为了节省时间，我们就不编译了，直接重新加载管道:

```python
pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.to("cuda")

audio = pipe(prompt, negative_prompt=negative_prompt, num_waveforms_per_prompt=4, audio_length_in_s=150, num_inference_steps=20, generator=generator.manual_seed(0)).audios[0]
```

**输出:**

```
---------------------------------------------------------------------------
OutOfMemoryError Traceback (most recent call last)
<ipython-input-33-c4cae6410ff5> in <cell line: 5>()
      3 pipe.to("cuda")
      4
----> 5 audio = pipe(prompt, negative_prompt=negative_prompt, num_waveforms_per_prompt=4, audio_length_in_s=150, num_inference_steps=20, generator=generator.manual_seed(0)).audios[0]

23 frames
/usr/local/lib/python3.10/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    112
    113 def forward(self, input: Tensor) -> Tensor:
--> 114 return F.linear(input, self.weight, self.bias)
    115
    116 def extra_repr(self) -> str:

OutOfMemoryError: CUDA out of memory. Tried to allocate 1.95 GiB. GPU 0 has a total capacty of 14.75 GiB of which 1.66 GiB is free. Process 414660 has 13.09 GiB memory in use. Of the allocated memory 10.09 GiB is allocated by PyTorch, and 1.92 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation. See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

除非你的 GPU 显存很大，否则上面的代码可能会返回 OOM 错误。虽然 AudioLDM 2 流水线涉及多个组件，但任何时候只有当前正在使用的模型必须在 GPU 上。其余模块均可以卸载到 CPU。该技术称为“CPU 卸载”，可大大减少显存使用，且对推理时间的影响很小。

我们可以使用函数 [enable_model_cpu_offload()](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2Pipeline.enable_model_cpu_offload) 在流水线上启用 CPU 卸载:

```python
pipe.enable_model_cpu_offload()
```

调用 API 生成音频的方式与以前相同:

```python
audio = pipe(prompt, negative_prompt=negative_prompt, num_waveforms_per_prompt=4, audio_length_in_s=150, num_inference_steps=20, generator=generator.manual_seed(0)).audios[0]
```

**输出:**

```
100%|███████████████████████████████████████████| 20/20 [00:36<00:00, 1.82s/it]
```

这样，我们就可以生成 4 个各为 150 秒的样本，所有这些都在一次流水线调用中完成！大版的 AudioLDM 2 checkpoint 比基础版的 checkpoint 总内存使用量更高，因为 UNet 的大小相差两倍多 (750M 参数与 350M 参数相比)，因此这种节省内存的技巧对大版的 checkpoint 特别有用。

## 总结

在本文中，我们展示了 🧨 Diffusers 开箱即用的四种优化方法，并将 AudioLDM 2 的生成时间从 14 秒缩短到不到 1 秒。我们还重点介绍了如何使用内存节省技巧 (例如半精度和 CPU 卸载) 来减少长音频样本或大 checkpoint 场景下的峰值显存使用量。

本文作者 [Sanchit Gandhi](https://huggingface.co/sanchit-gandhi) 非常感谢 [Vaibhav Srivastav](https://huggingface.co/reach-vb) 和 [Sayak Paul](https://huggingface.co/sayakpaul) 的建设性意见。频谱图图像来自于 [Getting to Know the Mel Spectrogram](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0) 一文，波形图来自于 [Aalto Speech Processing](https://speechprocessingbook.aalto.fi/Representations/Waveform.html) 一文。