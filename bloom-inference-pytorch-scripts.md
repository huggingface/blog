---
title: "Incredibly Fast BLOOM Inference with DeepSpeed and Accelerate"
thumbnail: /blog/assets/bloom-inference-pytorch-scripts/thumbnail.png
authors:
- user: stas
- user: sgugger
---

<h1>Incredibly Fast BLOOM Inference with DeepSpeed and Accelerate</h1>

{blog_metadata}

{authors}

This article shows how to get an incredibly fast per token throughput when generating with the 176B parameter [BLOOM model](https://huggingface.co/bigscience/bloom).

As the model needs 352GB in bf16 (bfloat16) weights (`176*2`), the most efficient set-up is 8x80GB A100 GPUs. Also 2x8x40GB A100s or 2x8x48GB A6000 can be used. The main reason for using these GPUs is that at the time of this writing they provide the largest GPU memory, but other GPUs can be used as well. For example, 24x32GB V100s can be used.

Using a single node will typically deliver a fastest throughput since most of the time intra-node GPU linking hardware is faster than inter-node one, but it's not always the case.

If you don't have that much hardware, it's still possible to run BLOOM inference on smaller GPUs, by using CPU or NVMe offload, but of course, the generation time will be much slower.

We are also going to cover the [8bit quantized solutions](https://huggingface.co/blog/hf-bitsandbytes-integration), which require half the GPU memory at the cost of slightly slower throughput. We will discuss [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) and [Deepspeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) libraries there.

## Benchmarks

Without any further delay let's show some numbers.

For the sake of consistency, unless stated differently, the benchmarks in this article were all done on the same 8x80GB A100 node w/ 512GB of CPU memory on [Jean Zay HPC](http://www.idris.fr/eng/jean-zay/index.html). The JeanZay HPC users enjoy a very fast IO of about 3GB/s read speed (GPFS). This is important for checkpoint loading time. A slow disc will result in slow loading time. Especially since we are concurrently doing IO in multiple processes.

All benchmarks are doing [greedy generation](https://huggingface.co/blog/how-to-generate#greedy-search) of 100 token outputs:
```
Generate args {'max_length': 100, 'do_sample': False}
```
The input prompt is comprised of just a few tokens. The previous token caching is on as well, as it'd be quite slow to recalculate them all the time.

First, let's have a quick look at how long did it take to get ready to generate - i.e. how long did it take to load and prepare the model:

| project                 | secs |
| :---------------------- | :--- |
| accelerate              |  121 |
| ds-inference shard-int8 |   61 |
| ds-inference shard-fp16 |   60 |
| ds-inference unsharded  |  662 |
| ds-zero                 |  462 |

Deepspeed-Inference comes with pre-sharded weight repositories and there the loading takes about 1 minuted. Accelerate's loading time is excellent as well - at just about 2 minutes. The other solutions are much slower here.

The loading time may or may not be of importance, since once loaded you can continually generate tokens again and again without an additional loading overhead.

Next the most important benchmark of token generation throughput. The throughput metric here is a simple - how long did it take to generate 100 new tokens divided by 100 and the batch size (i.e. divided by the total number of generated tokens).

Here is the throughput in msecs on 8x80GB GPUs:

| project      \ bs |      1 |     8 |    16 |    32 |   64 |  128 |  256 | 512  |
| :---------------- | :----- | :---- | :---- | :---- | :--- | :--- | :--- | :--- |
| accelerate   bf16 | 230.38 | 31.78 | 17.84 | 10.89 |  oom |      |      |      |
| accelerate   int8 | 286.56 | 40.92 | 22.65 | 13.27 |  oom |      |      |      |
| ds-inference fp16 |  44.02 |  5.70 |  3.01 |  1.68 | 1.00 | 0.69 |  oom |      |
| ds-inference int8 |  89.09 | 11.44 |  5.88 |  3.09 | 1.71 | 1.02 | 0.71 | oom  |
| ds-zero      bf16 |    283 | 34.88 |   oom |       |      |      |      |      |

where OOM == Out of Memory condition where the batch size was too big to fit into GPU memory.

Getting an under 1 msec throughput with Deepspeed-Inference's Tensor Parallelism (TP) and custom fused CUDA kernels! That's absolutely amazing! Though using this solution for other models that it hasn't been tried on may require some developer time to make it work.

Accelerate is super fast as well. It uses a very simple approach of naive Pipeline Parallelism (PP) and because it's very simple it should work out of the box with any model.

Since Deepspeed-ZeRO can process multiple generate streams in parallel its throughput can be further divided by 8 or 16, depending on whether 8 or 16 GPUs were used during the `generate` call. And, of course, it means that it can process a batch size of 64 in the case of 8x80 A100 (the table above) and thus the throughput is about 4msec - so all 3 solutions are very close to each other.

Let's revisit again how these numbers were calculated. To generate 100 new tokens for a batch size of 128 took 8832 msecs in real time when using Deepspeed-Inference in fp16 mode. So now to calculate the throughput we did: walltime/(batch_size*new_tokens) or `8832/(128*100) = 0.69`.

Now let's look at the power of quantized int8-based models provided by Deepspeed-Inference and BitsAndBytes, as it requires only half the original GPU memory of inference in bfloat16 or float16.

Throughput in msecs 4x80GB A100:

| project       bs  |      1 |     8 |    16 |    32 |   64 | 128  |
| :---------------- | :----- | :---- | :---- | :---- | :--- | :--- |
| accelerate   int8 | 284.15 | 40.14 | 21.97 |  oom  |      |      |
| ds-inference int8 | 156.51 | 20.11 | 10.38 |  5.50 | 2.96 | oom  |

To reproduce the benchmark results simply add `--benchmark` to any of these 3 scripts discussed below.


## Solutions


First checkout the demo repository:

```
git clone https://github.com/huggingface/transformers-bloom-inference
cd transformers-bloom-inference
```

In this article we are going to use 3 scripts located under `bloom-inference-scripts/`.

The framework-specific solutions are presented in an alphabetical order:

## HuggingFace Accelerate

[Accelerate](https://github.com/huggingface/accelerate)

Accelerate handles big models for inference in the following way:
1. Instantiate the model with empty weights.
2. Analyze the size of each layer and the available space on each device (GPUs, CPU) to decide where each layer should go.
3. Load the model checkpoint bit by bit and put each weight on its device

It then ensures the model runs properly with hooks that transfer the inputs and outputs on the right device and that the model weights offloaded on the CPU (or even the disk) are loaded on a GPU just before the forward pass, before being offloaded again once the forward pass is finished.

In a situation where there are multiple GPUs with enough space to accommodate the whole model, it switches control from one GPU to the next until all layers have run. Only one GPU works at any given time, which sounds very inefficient but it does produce decent throughput despite the idling of the GPUs.

It is also very flexible since the same code can run on any given setup. Accelerate will use all available GPUs first, then offload on the CPU until the RAM is full, and finally on the disk. Offloading to CPU or disk will make things slower. As an example, users have reported running BLOOM with no code changes on just 2 A100s with a throughput of 15s per token as compared to 10 msecs on 8x80 A100s.

You can learn more about this solution in [Accelerate documentation](https://huggingface.co/docs/accelerate/big_modeling).

### Setup

```
pip install transformers>=4.21.3 accelerate>=0.12.0
```


### Run

The simple execution is:

```
python bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 1 --benchmark
```

To activate the 8bit quantized solution from [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) first install `bitsandbytes`:

```
pip install bitsandbytes
```

and then add `--dtype int8` to the previous command line:

```
python bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --dtype int8 --batch_size 1 --benchmark
```

if you have more than 4 GPUs you can tell it to use only 4 with:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --dtype int8 --batch_size 1 --benchmark
```

The highest batch size we were able to run without OOM was 40 in this case. If you look inside the script we had to tweak the memory allocation map to free the first GPU to handle only activations and the previous tokens' cache.



## DeepSpeed-Inference

[DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) uses Tensor-Parallelism and efficient fused CUDA kernels to deliver a super-fast <1msec per token inference on a large batch size of 128.


### Setup

```
pip install deepspeed>=0.7.3
```

### Run

1. the fastest approach is to use a TP-pre-sharded (TP = Tensor Parallel) checkpoint that takes only ~1min to load, as compared to 10min for non-pre-sharded bloom checkpoint:


```
deepspeed --num_gpus 8 bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-fp16
```

1a.
if you want to run the original bloom checkpoint, which once loaded will run at the same throughput as the previous solution, but the loading will take 10-20min:

```
deepspeed --num_gpus 8 bloom-inference-scripts/bloom-ds-inference.py --name bigscience/bloom
```

2a. The 8bit quantized version requires you to have only half the GPU memory of the normal half precision version:


```
deepspeed --num_gpus 8 bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8
```

Here we used `microsoft/bloom-deepspeed-inference-int8` and also told the script to run in `int8`.

And of course, just 4x80GB A100 GPUs is now sufficient:

```
deepspeed --num_gpus 4 bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8
```

The highest batch size we were able to run without OOM was 128 in this case.

You can see two factors at play leading to better performance here.

1. The throughput here was improved by using Tensor Parallelism (TP) instead of the Pipeline Parallelism (PP) of Accelerate. Because Accelerate is meant to be very generic it is also unfortunately hard to maximize the GPU usage. All computations are done first on GPU 0, then on GPU 1, etc. until GPU 8, which means 7 GPUs are idle all the time. DeepSpeed-Inference on the other hand uses TP, meaning it will send tensors to all GPUs, compute part of the generation on each GPU and then all GPUs communicate to each other the results, then move on to the next layer. That means all GPUs are active at once but they need to communicate much more.

2. DeepSpeed-Inference also uses custom CUDA kernels to avoid allocating too much memory and doing tensor copying to and from GPUs. The effect of this is lesser memory requirements and fewer kernel starts which improves the throughput and allows for bigger batch sizes leading to higher overall throughput.

If you are interested in more examples you can take a look at [Accelerate GPT-J inference with DeepSpeed-Inference on GPUs](https://www.philschmid.de/gptj-deepspeed-inference) or [Accelerate BERT inference with DeepSpeed-Inference on GPUs](https://www.philschmid.de/bert-deepspeed-inference).

## Deepspeed ZeRO-Inference


[Deepspeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) uses a magical sharding approach which can take almost any model and scale it across a few or hundreds of GPUs and the do training or inference on it.

### Setup

```
pip install deepspeed
```


### Run

Note that the script currently runs the same inputs on all GPUs, but you can run a different stream on each GPU, and get `n_gpu` times faster throughput. You can't do that with Deepspeed-Inference.

```
deepspeed --num_gpus 8 bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 1 --benchmark
```

Please remember that with ZeRO the user can generate multiple unique streams at the same time - and thus the overall performance should be throughput in secs/token divided by number of participating GPUs - so 8x to 16x faster depending on whether 8 or 16 GPUs were used!

You can also try the offloading solutions with just one smallish GPU, which will take a long time to run, but if you don't have 8 huge GPUs this is as good as it gets.


CPU-Offload (1x GPUs):
```
deepspeed --num_gpus 1 bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --cpu_offload --benchmark
```

NVMe-Offload (1x GPUs):
```
deepspeed --num_gpus 1 bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --nvme_offload_path=/path/to/nvme_offload --benchmark
```

make sure to adjust `/path/to/nvme_offload` to somewhere you have ~400GB of free memory on a fast NVMe drive.


## Additional Client and Server Solutions

At [transformers-bloom-inference](https://github.com/huggingface/transformers-bloom-inference) you will find more very efficient solutions, including server solutions.

Here are some previews.

Server solutions:

* [Mayank Mishra](https://github.com/mayank31398) took all the demo scripts discussed in this blog post and turned them into a webserver package, which you can download from [here](https://github.com/huggingface/transformers-bloom-inference)

* [Nicolas Patry](https://github.com/Narsil) has developed a super-efficient [Rust-based webserver solution]((https://github.com/Narsil/bloomserver).

More client-side solutions:

* [Thomas Wang](https://github.com/thomasw21) is developing a very fast [custom CUDA kernel BLOOM model](https://github.com/huggingface/transformers_bloom_parallel).

* The JAX team @HuggingFace has developed a [JAX-based solution](https://github.com/huggingface/bloom-jax-inference)


As this blog post is likely to become outdated if you read this months after it was published please
use [transformers-bloom-inference](https://github.com/huggingface/transformers-bloom-inference) to find the most up-to-date solutions.



## Blog credits

Huge thanks to the following kind folks who asked good questions and helped improve the readability of the article:
Olatunji Ruwase and Philipp Schmid.
