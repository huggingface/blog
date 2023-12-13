---
title: "Speculative Decoding for 2x Faster Whisper Inference" 
thumbnail: /blog/assets/whisper-spec-dec/thumbnail.png
authors:
- user: sanchit-gandhi
---

# Speculative Decoding for 2x Faster Whisper Inference

<a target="_blank" href="https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/speculative_decoding.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Open AI's [Whisper](https://openai.com/research/whisper) is a general 
purpose speech transcription model that achieves state-of-the-art results across a range of different benchmarks and 
audio conditions. The latest [large-v3](https://huggingface.co/openai/whisper-large-v3) model tops the 
[OpenASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), ranking as the best open-source 
speech transcription model for English. The model also demonstrates strong multilingual performance, achieving less than 
30% word error rate (WER) on 42 of the 58 languages tested in the Common Voice 15 dataset.

While the transcription accuracy is exceptional, the inference time is very slow. A 1 hour audio clip takes upwards of 
6 minutes to transcribe on a 16GB T4 GPU, even after leveraging inference optimisations like [flash attention](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2), 
half-precision and chunking. In this Google Colab, we demonstrate how Speculative Decoding can be employed to reduce the 
inference time of Whisper by a **factor of 2**, while mathematically ensuring exactly the **same outputs** are achieved 
from the model. As a result, this method provides a perfect drop-in replacement for existing Whisper pipelines, since it 
provides free 2x speed-up while maintaining the same accuracy.

## Speculative Decoding

Speculative Decoding was proposed in the paper [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) 
by Yaniv Leviathan et. al. from Google. It works on the premise that a faster, **assistant model** can be used to 
boostrap the generation of a larger, **main model**.

First, the assistant model auto-regressively generates a sequence of \\( N \\) *candidate tokens*, \\( \hat{\boldsymbol{y}}_{1:N} \\). 
In the diagram below, the assistant model generates a sequence of 5 candidate tokens: `The quick brown sock jumps`.

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 70%; margin: auto;"
        controls playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/whisper-spec-dec/split_1.mp4"
    ></video>
</figure>

While these candidate tokens are generated quickly, they may differ from those predicted by the main model. Therefore, 
in the second step, the candidate tokens are passed to the main model to be "verified". The main model takes the 
candidate tokens as input and performs a **single forward pass**. The outputs of the main model are the "correct" 
token for each step in the token sequence \\( \boldsymbol{y}_{1:N} \\).

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 70%; margin: auto;"
        controls playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/whisper-spec-dec/split_2.mp4"
    ></video>
</figure>

In the diagram above, we see that the first three tokens predicted by the main model agree with those from the assistant 
model: <span style="color:green">The quick brown</span>. However, the fourth candidate token from the assistant model, 
<span style="color:red">sock</span>, mismatches with the correct token from the main model, <span style="color:green">fox</span>.

We know that all candidate tokens up to the first mismatch are correct (<span style="color:green">The quick brown</span>), 
since these agree with the predictions from the main model. However, after the first mismatch, the candidate tokens 
diverge from the actual tokens predicted by the main model. Therefore, we can replace the first incorrect candidate 
token (<span style="color:red">sock</span>) with the correct token from the main model (<span style="color:green">fox</span>), 
and discard all predicted tokens that come after this, since these have diverged. The corrected sequence, `The quick brown fox`, 
now forms the new input to the assistant model:

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 70%; margin: auto;"
        controls playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/whisper-spec-dec/split_3.mp4"
    ></video>
</figure>

The inference process then repeats, the assistant model generating a new set of \\( N \\) candidate tokens, which are verified 
in a single forward pass by the main model.

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 70%; margin: auto;"
        controls playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/whisper-spec-dec/split_4.mp4"
    ></video>
</figure>

Since we auto-regressively generate using the fast, assistant model, and only perform verification forward passes with 
the slow, main model, the decoding process is sped-up substantially. Furthermore, the verification forward passes 
performed by the main model ensure that **exactly the same outputs** are achieved as using the main model standalone. 
This makes speculative decoding the perfect substitute to existing Whisper pipelines, since one can be certain that the 
same performance will be attained.

To get the biggest speed-up in latency, we want the assistant model to be as fast as possible, while predicting the same 
token distribution as the main model. In practice, these two attributes form a trade-off: the faster the model is, the 
less accurate it is. The only constraint for selecting an assistant model is that it must share the same vocabulary as 
the main model. That is to say, if we want to use speculative decoding with Whisper [large-v2](https://huggingface.co/openai/whisper-large-v2) 
(multilingual), we need to select a multilingual variant of Whisper. Whereas if we want to use speculative decoding with 
Whisper [medium.en](https://huggingface.co/openai/whisper-medium.en) (English-only), we need an English-only variant of 
Whisper. At the current time, Whisper [large-v3](https://huggingface.co/openai/whisper-large-v3) is an exception, since 
it is the only Whisper checkpoint with an expanded vocabulary size, and thus is not compatible with previous Whisper 
checkpoints.

Now that we know the background behind speculative decoding, we're ready to dive into the practical implementation. In 
the [🤗 Transformers](https://huggingface.co/docs/transformers/index) library, speculative decoding is implemented as 
the "assisted generation" inference strategy. For more details about the implementation, the reader is advised to read 
Joao Gante's excellent blog post on [Assisted Generation](https://huggingface.co/blog/assisted-generation). 

## English Speech Transcription

### Baseline Implementation

We start by benchmarking Whisper [large-v2](https://huggingface.co/openai/whisper-large-v2) to get our baseline number 
for inference speed. We can load the main model and it's corresponding processor via the convenient 
[`AutoModelForSpeechSeq2Seq`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSpeechSeq2Seq) 
and [`AutoProcessor`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor) classes. We'll 
load the model in `float16` precision and make sure that loading time takes as little time as possible by passing 
[`low_cpu_mem_usage=True`](https://huggingface.co/docs/transformers/main_classes/model#large-model-loading). In addition, 
we want to make sure that the model is loaded in [safetensors](https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors) 
format by passing [`use_safetensors=True`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained.use_safetensors).
Finally, we'll pass the argument `attn_implementation="sdpa"` to benefit from Flash Attention speed-ups through PyTorch's 
[SDPA attention kernel](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html):

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

Let's load the English speech transcription dataset that we will use for benchmarking. We'll load a small dataset 
consisting of 73 samples from the [LibriSpeech ASR](https://huggingface.co/datasets/librispeech_asr) validation-clean 
dataset. This amounts to ~9MB of data, so it's very lightweight and quick to download on device:

```python
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
```

For the benchmark, we only want to measure the generation time, so let's write a short helper function that measures 
this step. The following function will return both the decoded tokens and the time it took to run the model:

```python
import time

def generate_with_time(model, inputs, **kwargs):
    start_time = time.time()
    outputs = model.generate(**inputs, **kwargs)
    generation_time = time.time() - start_time
    return outputs, generation_time
```

We can now iterate over the audio samples in our dataset and sum up the overall generation time:

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

Alright! We see that transcribing the 73 samples took 73 seconds. Let's check the WER of the predictions:

```python
from evaluate import load

wer = load("wer")
print(wer.compute(predictions=predictions, references=references))
```

Our final baseline number is 73 seconds for a WER of 3.5%.

### Speculative Decoding

Now let's load the assistant model for speculative decoding. In this example, we'll use a distilled variant of Whisper, 
[distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2). The distilled model copies the entire encoder 
from Whisper, but only 2 of the 32 decoder layers. As such, it runs 6x faster than Whisper, while performing to within 
1% WER on our-of-distribution test sets. This makes it the perfect candidate choice of assistant model, since it has both 
high transcription accuracy and fast generation \\({}^1\\).

Since Distil-Whisper uses exactly same encoder as the Whisper model, we can share the encoder across the main and 
assistant models. We then only have to load the 2-layer decoder from Distil-Whisper as a "decoder-only" model. We can do 
this through the convenient [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) 
auto class. In practice, this results in only an 8% increase to VRAM over using the main model alone.

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

assistant_model.to(device);
```

------------------------------------------------------------------------

\\({}^1\\) We intend to release an improved variant of Distil-Whisper with a stronger alignment in the token distribution 
that will improve speculative decoding performance further. Follow the [Distil-Whisper repository](https://github.com/huggingface/distil-whisper) 
for updates.

------------------------------------------------------------------------

We can define a modified function for our speculative decoding benchmark. The only difference from the previous function 
is that we pass the assistant model to our call to `.generate`:

```python
def assisted_generate_with_time(model, inputs, **kwargs):
    start_time = time.time()
    outputs = model.generate(**inputs, assistant_model=assistant_model, **kwargs)
    generation_time = time.time() - start_time
    return outputs, generation_time
```

Let's run the benchmark with speculative decoding, using Distil-Whisper as the assistant to Whisper:

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

With speculative decoding, the inference time was just 33 seconds, 2.2x faster than before! Let's verify we have the same 
WER:

```python
print(wer.compute(predictions=predictions, references=references))
```
**Outputs:**
```
0.03507271171941831
```

Perfect! 3.5% WER again. This confirms we have identical outputs to using the main model standalone.

Speculative decoding can also be incorporated with 🤗 Transformers [pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) 
class for an easy API for inference. Below, we instantiate the pipeline using the model and processor, and then use it to 
transcribe the first sample from the toy dataset. This can be extended to transcribe audio samples of arbitrary length, 
including with the use of batching:

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

An end-to-end codesnippet for running speculative decoding with Whisper and Distil-Whisper can be found on the [Distil-Whisper model card](https://huggingface.co/distil-whisper/distil-large-v2#speculative-decoding). 
It combines the stages of inference covered in this notebook into a single code example.

## Multilingual Speech Transcription

Distil-Whisper is the perfect assistant model for English speech transcription, since it performs to within 1% WER of the 
original Whisper model, while being 6x faster over short and long-form audio samples. However, the official Distil-Whisper 
checkpoints are English only, meaning they cannot be used for multilingual speech transcription. 

To use speculative decoding for multilingual speech transcription, one could either use on of the [official multilingual Whisper checkpoints](https://huggingface.co/openai/whisper-large-v2#model-details), 
or a fine-tuned variant of Whisper. As of the time of writing, there are over 5,000 [fine-tuned Whisper checkpoints](https://huggingface.co/models?other=whisper) 
on the Hugging Face Hub in over 100 languages. These provide an excellent starting point for selecting assistant Whisper 
checkpoints that perform very well on a single language. In this example, we'll use the smallest official multilingual 
checkpoint, Whisper [tiny](https://huggingface.co/openai/whisper-tiny). Feel free to experiment with different checkpoints
fine-tuned in your language!

Let's load the weights for our new assistant model, Whisper tiny. Since the encoder in Whisper tiny differs from that in 
large-v2, we'll load both the encoder and decoder using the `AutoModelForSpeechSeq2Seq` class:

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

For our benchmarking dataset, we'll load 73 samples from the Dutch ("nl") split of the [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) dataset:

```python
dataset = load_dataset("sanchit-gandhi/voxpopuli_dummy", "nl", split="validation")
```

Great! We can now re-run our benchmark for our baseline Whisper large-v2 model as before. The only change we make is that 
we pass the language and task arguments to our generate function, in order to ensure we perform speech transcription 
(not speech translation). Note that speculative decoding is fully compatible with the speech translation task. Simply 
set the task argument as required below:

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

Right! We have our baseline time of 117 seconds and a WER of 12.8%. Let's re-run the generation process using speculative decoding:

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

Again, we achieve 12.8% WER, but this time in just 62 seconds of inference time, representing a speed-up of 1.9x.
Given the low overhead of loading the assistant model and the mathematical property that exactly the same outputs are 
achieved, speculative decoding offers the perfect drop-in replacement to existing Whisper pipelines.

## Strategies for Efficient Speculative Decoding

In this final section, we cover two strategies for ensuring the fastest possible inference time with speculative decoding.

#### Assistant Model

Our objective is to select an assistant model that is both fast **and** maintains the same token distribution as the 
main model. If you have a particular language in which you want to transcribe, an effective strategy is to train two 
Whisper models of different sizes, and use one as the assistant to the other:

* First, fine-tune Whisper [large-v3](https://huggingface.co/openai/whisper-large-v3) to act as your main model
* Second, distil Whisper [large-v3](https://huggingface.co/openai/whisper-large-v3) on the same dataset to act as a fast assistant model

Fine-tuning and distillation can improve the WER performance of both the main and assistant model on your chosen language, 
while maximising the alignment in the token distributions. A complete guide to Whisper fine-tuning can be found 
[here](https://huggingface.co/blog/fine-tune-whisper), and distillation [here](https://github.com/huggingface/distil-whisper/tree/main/training).

#### Batch Size

It is worth noting that the largest speed gains with speculative decoding come with a batch size of 1. For batched 
speculative decoding, all candidate tokens **across the batch** must match the validation tokens in order for the tokens 
to be accepted. If a token in the batch at a given position does not agree, all candidate tokens that precede the position 
are discarded. Consequently, speculative decoding favours lower batch sizes. In practice, we find that speculative decoding 
provides a speed-up until a batch size of 4. Above batch size 4, speculative decoding returns slower inference than the 
main model alone. For full results, refer to Section D.3 of the [Distil-Whisper paper](https://arxiv.org/pdf/2311.00430.pdf).

## Conclusion

In this blog post, we covered the inference strategy of speculative decoding, as applied to the Whisper model for speech 
transcription. We demonstrated how 2x speed-ups can be achieved, while mathematically ensuring the same outputs as using 
the original model alone. We encourage you to try speculative decoding as a drop-in replacement for existing Whisper 
pipelines, given the low overhead of using the additional assistant model and the guarantee of the same transcription results.
