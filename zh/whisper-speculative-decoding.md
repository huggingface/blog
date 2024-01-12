---
title: "使用推测解码使 Whisper 实现 2 倍的推理加速" 
thumbnail: /blog/assets/whisper-speculative-decoding/thumbnail.png
authors:
- user: sanchit-gandhi
translators:
- user: yaoqih
- user: zhongdongy
  proofreader: true
---

# 使用推测解码使 Whisper 实现 2 倍的推理加速

<a target="_blank" href="https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/speculative_decoding.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Open AI 推出的 [Whisper](https://openai.com/research/whisper) 是一个通用语音转录模型，在各种基准和音频条件下都取得了非常棒的结果。最新的 [large-v3](https://huggingface.co/openai/whisper-large-v3) 模型登顶了 [OpenASR 排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)，被评为最佳的开源英语语音转录模型。该模型在 Common Voice 15 数据集的 58 种语言中也展现出了强大的多语言性能，在 42 种语言上的单词错误率 (WER) 低于 30％。

尽管转录准确度非常优秀，但推理速度非常缓慢。即使利用 [flash attention](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2) 、半精度和 [分块](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline.chunk_length_s) 等优化推理技术，1 小时长度的音频在 16GB T4 GPU 上也需要超过 6 分钟的转录时间。

在本文中，我们将演示如何运用推测解码将 Whisper 的推理时间缩减 **2 倍**，同时在数学上确保完全取得与原模型 **相同的输出**。因此，这种方法可以完美地替换现有的 Whisper 流水线，因为它可以在不降低准确性的情况下免费获得 2 倍的加速。想要看附带有更简洁解释的全部代码，请参阅配套的 [Google Colab](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/speculative_decoding.ipynb)。

## 推测解码

推测解码由 Yaniv Leviathan 等人在 [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) 中提出。其思想是，一个更快的 **辅助模型** 通常会生成和更大的 **主模型** 相同的 token。

首先，辅助模型会通过自回归生成 $N$ 个 _候选 token_ 序列: $\hat{\boldsymbol{y}}_{1:N}$。在下图中，辅助模型生成了一个包含 5 个候选 token 的序列: `The quick brown sock jumps` 。

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        controls playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/whisper-speculative-decoding/split_1.mp4"
    ></video>
</figure>

尽管这些候选 token 可以快速生成，但它们可能与主模型预测的 token 不同。因此，在第二步中，候选 token 被传入主模型以进行“验证”。主模型将候选 token 作为输入，并执行 **单次前馈传播**。主模型的输出是每个步骤中“正确”token 的序列 $ \boldsymbol{y}_{1:N}$。

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        controls playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/whisper-speculative-decoding/split_2.mp4"
    ></video>
</figure>

在上图中，我们看到主模型预测的前三个 token 与辅助模型的 token 一致: `<span style="color:green">` The quick brown 但是，辅助模型的第四个候选 token: “ `<span style="color:red">` sock”与主模型的正确 token: “ `<span style="color:green">` fox”不一致。

我们知道，所有候选 token 一直到第一个不匹配之前都是正确的 ( `<span style="color:green">` The quick brown)，因为这些与主模型的预测一致。但是，在第一个不匹配之后，候选 token 开始偏离主模型实际预测的 token。因此，我们可以用主模型的正确 token ( `<span style="color:green">` fox) 替换第一个不正确的候选 token ( `<span style="color:red">` sock)，并放弃之后所有预测的 token，因为这些已经逐渐偏离主模型的预测。经过校正的序列 `The quick brown fox` 现在成为辅助模型的新输入:

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        controls playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/whisper-speculative-decoding/split_3.mp4"
    ></video>
</figure>

然后，辅助模型再次通过自回归推理，生成一组新的 $N$ 个候选 token，这些 token 再次通过主模型的单次前馈传播进行验证。

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        controls playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/whisper-speculative-decoding/split_4.mp4"
    ></video>
</figure>

由于我们在生成的时候使用的快速的辅助模型进行自回归，并且缓慢的主模型仅用于验证前馈传播，解码过程将大大加快。此外，经过主模型前馈传播验证后可以确保与仅使用主模型时获得完全相同的输出。这使得推测解码可以完美地替换现有的 Whisper 流水线，因为我们可以确定会取得相同质量的输出。

为了最大限度地减少延迟，辅助模型应该比主模型快得多，同时尽可能频繁地预测相同的 token 分布。实际上，这两个属性之间需要权衡: 模型越快，其准确度越低。然而，由于所有预测 token 中的 70-80％ 往往是“较易”的 token，此权衡倾向于选择一个更快的模型，而不是一个更准确的模型。因此，辅助模型应该至少比主模型快 3 倍 (越快越好)，同时在示例中正确预测所有较“易”token。剩余的 20-30％ 更“难”的 token 可以由更大的主模型进行验证。

选择辅助模型的唯一约束是它必须与主模型使用相同的词汇表。也就是说，辅助模型必须使用与主模型完全一对一相同的分词器。因此，如果我们想对诸如 [large-v2](https://huggingface.co/openai/whisper-large-v2) (多语言) 的 Whisper 多语言版本使用推测解码，我们需要选择诸如 [tiny](https://huggingface.co/openai/tiny) 的 Whisper 多语言版本作为辅助模型。而如果我们想对诸如 [medium.en](https://huggingface.co/openai/whisper-medium.en) 的 Whisper 英文版本使用推测解码，我们需要选择诸如 [tiny.en](https://huggingface.co/openai/tiny.en) 的 Whisper 英文版本作为辅助模型。目前，[large-v3](https://huggingface.co/openai/whisper-large-v3) 是唯一一个扩展了词汇量的 Whisper 检查点，因此与以前的 Whisper 检查点不兼容。

现在我们已经了解了推测解码背后的原理，我们准备实际实现它。在 [🤗 Transformers](https://huggingface.co/docs/transformers/index) 库中，推测解码被实现为“辅助生成 (Assisted Generation)”推理策略。欲了解更多实现细节，建议读者阅读 Joao Gante 关于 [辅助生成](https://huggingface.co/blog/assisted-generation) 的精彩博文。

## 英文语音转录

### 基准实现

我们首先使用 Whisper [large-v2](https://huggingface.co/openai/whisper-large-v2) 进行基准测试，以获得推理速度的基准数值。我们可以通过便捷的 [`AutoModelForSpeechSeq2Seq`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSpeechSeq2Seq) 和 [`AutoProcessor`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor) 类加载主模型及其对应的处理器。我们将以 `float16` 精度加载模型，并通过传递 [`low_cpu_mem_usage=True`](https://huggingface.co/docs/transformers/main_classes/model#large-model-loading) 确保加载时间尽可能少。此外，我们要确保模型以 [safetensors](https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors) 格式加载，方法是传递 [`use_safetensors=True`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained.use_safetensors)。最后，我们将传递参数 `attn_implementation="sdpa"` ，以通过 PyTorch 的 [SDPA 注意力内核](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) 进行 Flash 注意力加速。

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
```

让我们加载将用于基准测试的英语语音转录数据集。我们将加载 [LibriSpeech ASR](https://huggingface.co/datasets/librispeech_asr) 中验证数据集的 clean 分组中的 73 个样本组成的小型数据集。这大约有 9MB 的数据，因此非常轻量且可以快速下载到设备上。

```python
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
```

对于基准测试，我们只想测量生成时间，所以让我们编写一个简短的辅助函数来测量此步骤运行的时间。下面的函数将同时返回解码的 token 和运行模型所需的时间:

```python
import time

def generate_with_time(model, inputs, **kwargs):
    start_time = time.time()
    outputs = model.generate(**inputs, **kwargs)
    generation_time = time.time() - start_time
    return outputs, generation_time
```

现在我们可以迭代语音数据集中的音频样本，并统计整体生成时间:

```python
from tqdm import tqdm

all_time = 0
predictions = []
references = []

for sample in tqdm(dataset):
    audio = sample["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    inputs = inputs.to(device=device, dtype=torch.float16)
  
    output, gen_time = generate_with_time(model, inputs)
    all_time += gen_time
    predictions.append(processor.batch_decode(output, skip_special_tokens=True, normalize=True)[0])
    references.append(processor.tokenizer._normalize(sample["text"]))

print(all_time)
```

**Output:**

```
100%|██████████| 73/73 [01:37<00:00,  1.33s/it]
72.99542546272278
```

很好！我们看到转录 73 个样本花了 73 秒。让我们检查一下预测的 WER:

```python
from evaluate import load

wer = load("wer")
print(wer.compute(predictions=predictions, references=references))
```

**Output:**

```
0.03507271171941831
```

我们的最终基准数值为 73 秒，WER 为 3.5％。

### 推测解码

现在让我们加载推测解码的辅助模型。在此示例中，我们将使用 Whisper 蒸馏后的版本 [distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2)。蒸馏模型只使用了 Whisper 中 32 个解码器层中的 2 个编码器。因此，它比 Whisper 快 6 倍，同时在分布测试集上的 WER 性能相比于蒸馏前仅下降了 1％。这使其成为理想的辅助模型，因为它在转录准确性和生成速度方面都非常优秀${}^1$。

---

${}^1$ 我们即将发布 Distil-Whisper 的改进版本，在 token 分布中具有更佳的对齐性，这将进一步提高推测解码性能。关注 [Distil-Whisper 存储库](https://github.com/huggingface/distil-whisper) 来追踪最新的更新信息。

---

由于 Distil-Whisper 使用与 Whisper 模型完全相同的编码器，我们可以在主模型和辅助模型之间共享编码器。然后，我们只需要从 Distil-Whisper 加载 2 层解码器作为“仅解码器”模型。我们可以通过便捷的 [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) 自动类实现这一点。在实践中，相比于仅使用主模型，这仅增加了 8％的 VRAM 占用量。

```python
from transformers import AutoModelForCausalLM

assistant_model_id = "distil-whisper/distil-large-v2"

assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)

assistant_model.to(device)
```

我们可以为推测解码的基准测试定义一个新的函数。与前面的函数唯一的区别是，我们在对 `.generate` 的调用中传递辅助模型:

```python
def assisted_generate_with_time(model, inputs, **kwargs):
    start_time = time.time()
    outputs = model.generate(**inputs, assistant_model=assistant_model, **kwargs)
    generation_time = time.time() - start_time
    return outputs, generation_time
```

让我们使用 Distil-Whisper 作为 Whisper 的助手运行推测解码的基准测试:

```python
all_time = 0
predictions = []
references = []

for sample in tqdm(dataset):
    audio = sample["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    inputs = inputs.to(device=device, dtype=torch.float16)
  
    output, gen_time = assisted_generate_with_time(model, inputs)
    all_time += gen_time
    predictions.append(processor.batch_decode(output, skip_special_tokens=True, normalize=True)[0])
    references.append(processor.tokenizer._normalize(sample["text"]))

print(all_time)
```

**Outputs:**

```
100%|██████████| 73/73 [00:38<00:00,  1.88it/s]
32.69683289527893
```

使用推测解码，推理时间仅为 33 秒，比之前快 2.2 倍！让我们验证一下 WER 是否相同:

```python
print(wer.compute(predictions=predictions, references=references))
```

**Outputs:**

```
0.03507271171941831
```

太完美了！再次达到 3.5％的 WER，因为我们的输出与仅使用主模型的时候完全相同。

推测解码也可以与基础的 🤗 Transformers [pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API 一起用于推理。下面，我们使用模型和处理器实例化管道，然后使用它来转录测试数据集中的第一个样本。这可以扩展为转录任意长度的音频样本，包括进行批处理:

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=4,
    generate_kwargs={"assistant_model": assistant_model},
    torch_dtype=torch_dtype,
    device=device,
)

sample = dataset[0]["audio"]
result = pipe(sample)
print(result["text"])
```

**Outputs:**

```
 Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.
```

使用 Whisper 和 Distil-Whisper 运行推测解码的端到端代码示例可在 [Distil-Whisper 模型卡](https://huggingface.co/distil-whisper/distil-large-v2#speculative-decoding) 中找到。它将本文中涵盖的推理阶段组合成一个代码示例。

## 多语言语音转录

Distil-Whisper 是英语语音转录的最佳辅助模型，因为它与原始 Whisper 模型的 WER 误差率仅相差 1％，而对短长语音样本的推理速度提高了 6 倍。然而，官方的 Distil-Whisper 检查点仅支持英语，这意味着它们无法用于多语言语音转录。

要使用推测解码进行多语言语音转录，您可以使用 [官方 Whisper 多语言检查点](https://huggingface.co/openai/whisper-large-v2#model-details) 之一，或者 Whisper 的微调版本。在撰写本文时，Hugging Face Hub 上已有超过 5000 个微调过的 Whisper 检查点，支持超过 100 种语言。这些为选择表现出色的辅助模型提供了极好的起点。在此示例中，我们将使用最小的官方多语言检查点 Whisper [tiny](https://huggingface.co/openai/whisper-tiny)。您可以使用任意一个您的语言中微调过的不同检查点！

让我们为新的辅助模型 Whisper tiny 加载权重。由于 Whisper tiny 的编码器与 large-v2 不同，这次我们将使用 `AutoModelForSpeechSeq2Seq` 类同时加载编码器和解码器:

```python
assistant_model_id = "openai/whisper-tiny"

assistant_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    assistant_model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)

assistant_model.to(device);
```

我们的基准数据集，将从 [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) 数据集的荷兰语 (“nl”) 部分中加载 73 个样本:

```python
dataset = load_dataset("sanchit-gandhi/voxpopuli_dummy", "nl", split="validation")
```

非常好！现在我们可以像前面一样重新运行我们的 Whisper large-v2 模型的基准测试。我们所做的唯一更改是在 generate 函数中传递语言和任务参数，以确保执行语音转录 (而不是语音翻译)。推测解码完全兼容语音转录和翻译任务。只需如下所示设置任务参数即可:

```python
all_time = 0
predictions = []
references = []

for sample in tqdm(dataset):
    audio = sample["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    inputs = inputs.to(device=device, dtype=torch.float16)
  
    output, gen_time = generate_with_time(model, inputs, language="nl", task="transcribe")
    all_time += gen_time
    predictions.append(processor.batch_decode(output, skip_special_tokens=True, normalize=True)[0])
    references.append(processor.tokenizer._normalize(sample["normalized_text"]))

wer_result = wer.compute(predictions=predictions, references=references)

print("Time:", all_time)
print("WER:", wer_result)
```

**Outputs:**

```
100%|██████████| 73/73 [02:05<00:00,  1.72s/it]
Time: 116.50992178916931
WER: 0.127190136275146
```

没错！我们的基准时间为 117 秒，WER 为 12.8％。让我们使用推测解码重新运行生成过程:

```python
all_time = 0
predictions = []
references = []

for sample in tqdm(dataset):
    audio = sample["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    inputs = inputs.to(device=device, dtype=torch.float16)

    output, gen_time = assisted_generate_with_time(model, inputs, language="nl", task="transcribe")
    all_time += gen_time
    predictions.append(processor.batch_decode(output, skip_special_tokens=True, normalize=True)[0])
    references.append(processor.tokenizer._normalize(sample["normalized_text"]))

wer_result = wer.compute(predictions=predictions, references=references)

print("Time:", all_time)
print("WER:", wer_result)
```

**Outputs:**

```
100%|██████████| 73/73 [01:08<00:00,  1.06it/s]
Time: 62.10229682922363
WER: 0.127190136275146
```

Nice！我们达到了 12.8％ 的 WER，但这次的推理时间只有 62 秒，表示速度提高了 1.9 倍。考虑到加载辅助模型的低开销和确保获得完全相同输出的数学证明，推测解码为现有的 Whisper 管道提供了完美的即插即用的替代方案。

## 高效推测解码的策略

在本最终部分，我们将介绍两种策略，以确保使用推测解码时获得可能最快的推理时间。

#### 辅助模型

我们的目标是选择一个至少比主模型快 3 倍 **并且** 正确转录至少 70-80％ 的预测 token (通常是示例中的“更简单”token) 的辅助模型。如果您想要转录某种特定语言，一种有效的策略是训练两个不同大小的 Whisper 模型，并将其中一个用作另一个的辅助模型:

- 首先，微调 Whisper [large-v3](https://huggingface.co/openai/whisper-large-v3) 以用作主模型
- 其次，在同一数据集上蒸馏 Whisper [large-v3](https://huggingface.co/openai/whisper-large-v3) 以用作快速的辅助模型

微调和蒸馏都可以提高主模型和辅助模型在您选择的语言上的 WER 性能，同时最大化 token 分布的对齐。有关 Whisper 微调的完整指南，请参阅 [此处](https://huggingface.co/blog/fine-tune-whisper)，有关蒸馏的指南请参阅 [此处](https://github.com/huggingface/distil-whisper/tree/main/training)。

#### 批次大小

值得注意的是，使用推测解码获得的最大速度提升来自批次大小为 1。对于批处理推测解码，批处理中的所有候选 token 必须与验证 token 相匹配，才能被接受。如果批处理中给定位置的 token 不一致，则所有在该位置之前的候选 token 将被丢弃。因此，推测解码更倾向于较小的批次大小。在实践中，我们发现推测解码可以提供速度提升，直到批次大小达到 4 为止。当批次大小超过 4 时，推测解码的推理速度比仅用主模型还要慢。有关完整结果，请参阅 [Distil-Whisper 论文](https://arxiv.org/pdf/2311.00430.pdf) 的第 D.3 节。

## 结论

在本博文中，我们介绍了推测解码的推理策略，以及如何将其应用于语音转录的 Whisper 模型。我们展示了如何实现 2 倍的速度提升，同时数学上确保获得与仅使用原始模型相同的输出。我们鼓励您尝试将推测解码用作现有 Whisper 管道的即插即用替代方案，因为使用额外的辅助模型的开销很小，并且可以保证获得相同的转录结果。

## 致谢

本博客由 [Sanchit Gandhi](https://huggingface.co/sanchit-gandhi) 撰写。非常感谢 [Patrick von Platen](https://huggingface.co/patrickvonplaten) 和 [Pedro Cuenca](https://huggingface.co/pcuenq) 的建设性意见，以及 [Joao Gante](https://huggingface.co/joaogante) 在 🤗 Transformers 中实现辅助生成的贡献。