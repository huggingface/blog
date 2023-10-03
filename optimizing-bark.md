---
title: "Optimizing Bark using 🤗 Transformers" 
thumbnail: /blog/assets/bark_optimization/thumbnail.png
authors:
- user: ylacombe
---

# Optimizing a Text-To-Speech model using 🤗 Transformers


<a target="_blank" href="https://colab.research.google.com/github/ylacombe/notebooks/blob/main/Benchmark_Bark_HuggingFace.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"/>
</a>

🤗 Transformers provides many of the latest state-of-the-art (SoTA) models across domains and tasks. To get the best performance from these models, they need to be optimized for inference speed and memory usage.

The 🤗 Hugging Face ecosystem offers precisely such ready & easy to use optimization tools that can be applied across the board to all the models in the library. This makes it easy to **reduce memory footprint** and **improve inference** with just a few extra lines of code.

In this hands-on tutorial, I'll demonstrate how you can optimize [Bark](https://huggingface.co/docs/transformers/main/en/model_doc/bark#overview), a Text-To-Speech (TTS) model supported by 🤗 Transformers, based on three simple optimizations. These optimizations rely solely on the [Transformers](https://github.com/huggingface/transformers), [Optimum](https://github.com/huggingface/optimum) and [Accelerate](https://github.com/huggingface/accelerate) libraries from the 🤗 ecosystem.

This tutorial is also a demonstration of how one can benchmark a non-optimized model and its varying optimizations.

For a more streamlined version of the tutorial 
with fewer explanations but all the code, see the accompanying [Google Colab](https://colab.research.google.com/github/ylacombe/notebooks/blob/main/Benchmark_Bark_HuggingFace.ipynb).

This blog post is organized as follows:

## Table of Contents

1.   A [reminder](#bark-architecture) of Bark architecture
2.   An [overview](#optimization-techniques) of different optimization techniques and their advantages
3.   A [presentation](#benchmark-results) of benchmark results


# Bark Architecture


**Bark** is a transformer-based text-to-speech model proposed by Suno AI in [suno-ai/bark](https://github.com/suno-ai/bark). It is capable of generating a wide range of audio outputs, including speech, music, background noise, and simple sound effects. Additionally, it can produce nonverbal communication sounds such as laughter, sighs, and sobs.

Bark has been available in 🤗 Transformers since v4.31.0 onwards!


You can play around with Bark and discover it's abilities [here](https://colab.research.google.com/github/ylacombe/notebooks/blob/main/Bark_HuggingFace_Demo.ipynb).



Bark is made of 4 main models:

- `BarkSemanticModel` (also referred to as the 'text' model): a causal auto-regressive transformer model that takes as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
- `BarkCoarseModel` (also referred to as the 'coarse acoustics' model): a causal autoregressive transformer, that takes as input the results of the `BarkSemanticModel` model. It aims at predicting the first two audio codebooks necessary for EnCodec.
- `BarkFineModel` (the 'fine acoustics' model), this time a non-causal autoencoder transformer, which iteratively predicts the last codebooks based on the sum of the previous codebooks embeddings.
- having predicted all the codebook channels from the [`EncodecModel`](https://huggingface.co/docs/transformers/v4.31.0/model_doc/encodec), Bark uses it to decode the output audio array.

At the time of writing, two Bark checkpoints are available, a [smaller](https://huggingface.co/suno/bark-small) and a [larger](https://huggingface.co/suno/bark) version.


## Load the Model and its Processor

The pre-trained Bark small and large checkpoints can be loaded from the [pre-trained weights](https://huggingface.co/suno/bark) on the Hugging Face Hub. You can change the repo-id with the checkpoint size that you wish to use.

We'll default to the small checkpoint, to keep it fast. But you can try the large checkpoint by using `"suno/bark"` instead of `"suno/bark-small"`.

```python
from transformers import BarkModel

model = BarkModel.from_pretrained("suno/bark-small")
```

Place the model to an accelerator device to get the most of the optimization techniques:

```python
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

Load the processor, which will take care of tokenization and optional speaker embeddings.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("suno/bark-small")
```

# Optimization techniques

In this section, we'll explore how to use off-the-shelf features from the 🤗 Optimum and 🤗 Accelerate libraries to optimize the Bark model, with minimal changes to the code.

## Some set-ups

Let's prepare the inputs and define a function to measure the latency and GPU memory footprint of the Bark generation method.

```python
text_prompt = "Let's try generating speech, with Bark, a text-to-speech model"
inputs = processor(text_prompt).to(device)
```

Measuring the latency and GPU memory footprint requires the use of specific CUDA methods. We define a utility function that measures both the latency and GPU memory footprint of the model at inference time. To ensure we get an accurate picture of these metrics, we average over a specified number of runs `nb_loops`:

```python
import torch
from transformers import set_seed


def measure_latency_and_memory_use(model, inputs, nb_loops = 5):

  # define Events that measure start and end of the generate pass
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  # reset cuda memory stats and empty cache
  torch.cuda.reset_peak_memory_stats(device)
  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  # get the start time
  start_event.record()

  # actually generate
  for _ in range(nb_loops):
        # set seed for reproducibility
        set_seed(0)
        output = model.generate(**inputs, do_sample = True, fine_temperature = 0.4, coarse_temperature = 0.8)

  # get the end time
  end_event.record()
  torch.cuda.synchronize()

  # measure memory footprint and elapsed time
  max_memory = torch.cuda.max_memory_allocated(device)
  elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3

  print('Execution time:', elapsed_time/nb_loops, 'seconds')
  print('Max memory footprint', max_memory*1e-9, ' GB')

  return output
```

## Base case

Before incorporating any optimizations, let's measure the performance of the baseline model and listen to a generated example. We'll benchmark the model over five iterations and report an average of the metrics:

```python

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**Output:**

```
Execution time: 9.3841625 seconds
Max memory footprint 1.914612224  GB
```

Now, listen to the output:

```python
from IPython.display import Audio

# now, listen to the output
sampling_rate = model.generation_config.sample_rate
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)
```


The output sounds like this ([download audio](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_base.wav)): 

<audio controls> 
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_base.wav" type="audio/wav"> 
Your browser does not support the audio element. 
</audio> 

### Important note:

 Here, the number of iterations is actually quite low. To accurately measure and compare results, one should increase it to at least 100.

One of the main reasons for the importance of increasing `nb_loops` is that the speech lengths generated vary greatly between different iterations, even with a fixed input.

 One consequence of this is that the latency measured by `measure_latency_and_memory_use` may not actually reflect the actual performance of optimization techniques! The benchmark at the end of the blog post reports the results averaged over 100 iterations, which gives a true indication of the performance of the model.

## 1. 🤗 Better Transformer

Better Transformer is an 🤗 Optimum feature that performs kernel fusion under the hood. This means that certain model operations will be better optimized on the GPU and that the model will ultimately be faster.

To be more specific, most models supported by 🤗 Transformers rely on attention, which allows them to selectively focus on certain parts of the input when generating output. This enables the models to effectively handle long-range dependencies and capture complex contextual relationships in the data.

The naive attention technique can be greatly optimized via a technique called [Flash Attention](https://arxiv.org/abs/2205.14135), proposed by the authors Dao et. al. in 2022.

Flash Attention is a faster and more efficient algorithm for attention computations that combines traditional methods (such as tiling and recomputation) to minimize memory usage and increase speed. Unlike previous algorithms, Flash Attention reduces memory usage from quadratic to linear in sequence length, making it particularly useful for applications where memory efficiency is important.

Turns out that Flash Attention is supported by 🤗 Better Transformer out of the box! It requires one line of code to export the model to 🤗 Better Transformer and enable Flash Attention:



```python
model =  model.to_bettertransformer()

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**Output:**

```
Execution time: 5.43284375 seconds
Max memory footprint 1.9151841280000002  GB
```

The output sounds like this ([download audio](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_bettertransformer.wav)): 

<audio controls> 
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_bettertransformer.wav" type="audio/wav"> 
Your browser does not support the audio element. 
</audio> 

**What does it bring to the table?**

There's no performance degradation, which means you can get exactly the same result as without this function, while gaining 20% to 30% in speed! Want to know more? See this [blog post](https://pytorch.org/blog/out-of-the-box-acceleration/).

## 2. Half-precision

Most AI models typically use a storage format called single-precision floating point, i.e. `fp32`. What does it mean in practice? Each number is stored using 32 bits.

You can thus choose to encode the numbers using 16 bits, with what is called half-precision floating point, i.e. `fp16`, and use half as much storage as before! More than that, you also get inference speed-up!

Of course, it also comes with small performance degradation since operations inside the model won't be as precise as using `fp32`.

You can load a 🤗 Transformers model with half-precision by simpling adding `torch_dtype=torch.float16` to the `BarkModel.from_pretrained(...)` line!

In other words:

```python
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**Output:**

```
Execution time: 7.00045390625 seconds
Max memory footprint 2.7436124160000004  GB
```

The output sounds like this ([download audio](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_fp16.wav)): 

<audio controls> 
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_fp16.wav" type="audio/wav"> 
Your browser does not support the audio element. 
</audio> 


**What does it bring to the table?**

With a slight degradation in performance, you benefit from a memory footprint reduced by 50% and a speed gain of 5%.

## 3. CPU offload

As mentioned in the first section of this booklet, Bark comprises 4 sub-models, which are called up sequentially during audio generation. **In other words, while one sub-model is in use, the other sub-models are idle.**

Why is this a problem? GPU memory is precious in AI, because it's where operations are fastest, and it's often a bottleneck.

A simple solution is to unload sub-models from the GPU when inactive. This operation is called CPU offload.

**Good news:** CPU offload for Bark was integrated into 🤗 Transformers and you can use it with only one line of code.

You only need to make sure 🤗 Accelerate is installed!
```python
model = BarkModel.from_pretrained("suno/bark-small")

# Enable CPU offload
model.enable_cpu_offload()

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**Output:**

```
Execution time: 8.97633828125 seconds
Max memory footprint 1.3231160320000002  GB
```

The output sounds like this ([download audio](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_cpu_offload.wav)): 

<audio controls> 
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_cpu_offload.wav" type="audio/wav"> 
Your browser does not support the audio element. 
</audio> 


**What does it bring to the table?**

With a slight degradation in speed (10%), you benefit from a huge memory footprint reduction (60% 🤯).

With this feature enabled, `bark-large` footprint is now only 2GB instead of 5GB.
That's the same memory footprint as `bark-small`!


Want more? With `fp16` enabled, it's even down to 1GB. We'll see this in practice in the next section!

## 4. Combine

Let's bring it all together. The good news is that you can combine optimization techniques, which means you can use CPU offload, as well as half-precision and 🤗 Better Transformer!

```python
# load in fp16
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)

# convert to bettertransformer
model = BetterTransformer.transform(model, keep_original_model=False)

# enable CPU offload
model.enable_cpu_offload()

with torch.inference_mode():
  speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
```

**Output:**

```
Execution time: 7.4496484375000005 seconds
Max memory footprint 0.46871091200000004  GB
```

The output sounds like this ([download audio](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_cpu_offload.wav)): 

<audio controls> 
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_optimized.wav" type="audio/wav"> 
Your browser does not support the audio element. 
</audio> 


**What does it bring to the table?**

Ultimately, you get a 23% speed-up and a huge 80% memory saving!

## Using batching

Want more?

Altogether, the 3 optimization techniques bring even better results when batching.
Batching means combining operations for multiple samples to bring the overall time spent generating the samples lower than generating sample per sample.

Here is a quick example of how you can use it:

```python
text_prompt = [
    "Let's try generating speech, with Bark, a text-to-speech model",
    "Wow, batching is so great!",
    "I love Hugging Face, it's so cool."]

inputs = processor(text_prompt).to(device)


with torch.inference_mode():
  # samples are generated all at once
  speech_output = model.generate(**inputs, do_sample = True, fine_temperature = 0.4, coarse_temperature = 0.8)
```


The output sounds like this (download [first](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_0.wav), [second](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_1.wav), and [last](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_2.wav) audio): 

<audio controls> 
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_0.wav" type="audio/wav"> 
Your browser does not support the audio element. 
</audio> 

<audio controls> 
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_1.wav" type="audio/wav"> 
Your browser does not support the audio element. 
</audio> 


<audio controls> 
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bark_optimization/audio_sample_batch_2.wav" type="audio/wav"> 
Your browser does not support the audio element. 
</audio> 



# Benchmark results

As mentioned above, the little experiment we've carried out is an exercise in thinking and needs to be extended for a better measure of performance. One also needs to warm up the GPU with a few blank iterations before properly measuring performance.

Here are the results of a 100-sample benchmark extending the measurements, **using the large version of Bark**.

The benchmark was run on an NVIDIA TITAN RTX 24GB with a maximum of 256 new tokens.

## How to read the results?

### Latency

It measures the duration of a single call to the generation method, regardless of batch size.

In other words, it's equal to \\(\frac{elapsedTime}{nbLoops}\\).

**A lower latency is preferred.**

### Maximum memory footprint

It measures the maximum memory used during a single call to the generation method.

**A lower footprint is preferred.**


### Throughput

It measures the number of samples generated per second. This time, the batch size is taken into account.

In other words, it's equal to \\(\frac{nbLoops*batchSize}{elapsedTime}\\).

**A higher throughput is preferred.**

## No batching

Here are the results with `batch_size=1`.

| Absolute values             | Latency | Memory  |
|-----------------------------|---------|---------|
| no optimization             |   10.48 | 5025.0M |
| bettertransformer only      |    7.70 | 4974.3M |
| offload + bettertransformer |    8.90 | 2040.7M |
| offload + bettertransformer + fp16            |    8.10 | 1010.4M |

| Relative value              | Latency | Memory |
|-----------------------------|---------|--------|
| no optimization             |      0% |     0% |
| bettertransformer only      |    -27% |    -1% |
| offload + bettertransformer |    -15% |   -59% |
| offload + bettertransformer + fp16            |    -23% |   -80% |

### Comment

As expected, CPU offload greatly reduces memory footprint while slightly increasing latency.

However, combined with bettertransformer and `fp16`, we get the best of both worlds, huge latency and memory decrease!

## Batch size set to 8
And here are the benchmark results but with `batch_size=8` and throughput measurement.

Note that since `bettertransformer` is a free optimization because it does exactly the same operation and has the same memory footprint as the non-optimized model while being faster, the benchmark was run with **this optimization enabled by default**.


| absolute values               | Latency | Memory  | Throghput |
|-------------------------------|---------|---------|-----------|
| base case (bettertransformer) |   19.26 | 8329.2M |      0.42 |
| + fp16                          |   10.32 | 4198.8M |      0.78 |
| + offload                       |   20.46 | 5172.1M |      0.39 |
| + offload + fp16                |   10.91 | 2619.5M |      0.73 |

| Relative value                | Latency | Memory | Throughput |
|-------------------------------|---------|--------|------------|
| + base case (bettertransformer) |      0% |     0% |         0% |
| + fp16                          |    -46% |   -50% |        87% |
| + offload                       |      6% |   -38% |        -6% |
| + offload + fp16                |    -43% |   -69% |       77% |

### Comment

This is where we can see the potential of combining all three optimization features!

The impact of `fp16` on latency is less marked with `batch_size = 1`, but here it is of enormous interest as it can reduce latency by almost half, and almost double throughput!

# Concluding remarks

This blog post showcased a few simple optimization tricks bundled in the 🤗 ecosystem. Using anyone of these techniques, or a combination of all three, can greatly improve Bark inference speed and memory footprint.

* You can use the large version of Bark without any performance degradation and a footprint of just 2GB instead of 5GB, 15% faster, **using 🤗 Better Transformer and CPU offload**.


* Do you prefer high throughput? **Batch by 8 with 🤗 Better Transformer and half-precision**.


* You can get the best of both worlds by using **fp16, 🤗 Better Transformer and CPU offload**!
