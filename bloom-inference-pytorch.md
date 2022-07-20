---
title: "Efficient BLOOM Inference on PyTorch"
thumbnail: /blog/assets/86_bloom_megatron_deepspeed/thumbnail.png
---

<h1>Efficient BLOOM Inference on PyTorch</h1>

<div class="blog-metadata">
    <small>Published July 26, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/bloom-inference-pytorch.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/stas">
        <img class="avatar avatar-user" src="/blog/assets/bloom-inference-pytorch/stas-bekman-300x300.jpg">
        <div class="bfc">
            <code>stas</code>
            <span class="fullname">Stas Bekman,</span>
        </div>
    </a>
    <a href="/sgugger">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1593126474392-5ef50182b71947201082a4e5.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>sgugger</code>
            <span class="fullname">Sylvain Gugger,</span>
        </div>
    </a>
    and
    <a href="/Narsil">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1608285816082-5e2967b819407e3277369b95.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>narsil</code>
            <span class="fullname">Nicolas Patry</span>
        </div>
    </a>
</div>

This article discusses various PyTorch-based solution to efficient 176B parameter [BLOOM model](https://huggingface.co/bigscience/bloom).

As the model needs 352GB in bf16 weights (`176*2`), the most efficient set-up is 8x80GB A100. The second best is 2x8x40GB A100s. The main reason for using A100s And, is that at as of this writing they provide the largest GPU memory. But other GPUs can be used as well. It'd probably take 24x32GB V100 as another possibility.

Using a single node is ideal since PCIe speed is typically much faster than inter-node network.

If you don't have that much hardware, it's still possible to run BLOOM inference on smaller GPUs, by using CPU or NVME offload, but of course the generation time will be much slower.

For the sake of consistency, unless stated differently, the benchmarks in this article were all done on the same 8x80GB A100 node on [Jean Zay HPC](http://www.idris.fr/eng/jean-zay/index.html). The HPC users enjoy a very fast IO of about 3GB/s read speed (GPFS). This is important for loading speed time. A slow disc will result in slow loading time. Especially since we are concurrently doing IO in multiple processes.

## Performance metrics

For doing occasional inference on someone's desktop, that is a few inputs at a time the two critical metrics are the time to load the framework and the checkpoint shards and then the generation time.

For doing inference on the server the two important metrics for inference is the overall latency and the throughput of tokens. The loading time is less important since once loaded it becomes persistent.

The time to generate a token is straightforward - this is just how long it takes for a new token to be generated. As long as `use_cache=True` (the default argument to `generate`) all the previous logits are saved and therefore each token takes the same amount of time to generate. This of course comes at a memory cost - as the model is huge it can easily take GBs of memory to keep the cache. One can turn the caching off, and save memory, but it'd mean recalculating all the previous logits - which would make a long sequence generation very slow.

The latency is more difficult to define. If the server is idle and the model is ready to generate, and one or more prompts are sent from the same user at once, the latency would be just the time to generate a single token multiplied by the number of new tokens to generate. However, if you have multiple users sending prompts at different times, things become much more complicated. The first user gets the fastest response and thus fast latency, but subsequent users if they are unlucky to submit their request while the previous request is being processed may have to first wait till the first request has been generated and only their request will be attended. It's possible to design a very efficient system that uses a pipeline parallelism, where a new request can join the pipeline at any moment. But this of course will make the overall throughput lower. Then of course there is a small overhead of processing the input, sending data to the server and sending the results back to the user, but this is usually insignificant compared to the cost of generation on a huge model.

This article primarily discusses simple solutions and new more efficient solutions are being worked on as well.

The solutions are presented in alphabetical order:


## Accelerate

[Accelerate](https://github.com/huggingface/accelerate)

Accelerate deploys a naive Pipeline Parallelism to load a model that is much larger than the GPU size, by spreading the layers over multiple GPUs. At inference time it then switches control from one GPU to the next until all layers have run. In this situation only one GPU works at any given time. It sounds very inefficient but it produces excellent throughput despite the idling of the GPUs.


### Setup

```
pip install transformers>=4.20.1
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed/
cd Megatron-DeepSpeed
```

### Run

```
python scripts/inference/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 40 --benchmark
[...]

*** Performance stats:
Throughput per token including tokenize: 10.74 msecs
Start to ready to generate: 107.892 secs
Tokenize and generate 20000 (bs=40) tokens: 52.059 secs
Start to finish: 159.952 secs
```

The highest batch size I was able to run without OOM was 40 in this case.



## DeepSpeed-Inference

[DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) uses Tensor-Parallelism and efficient fused CUDA kernels to deliver an super-fast <1msec per token inference on a large batch size of 128.



### Setup

```
pip install deepspeed>=0.6.7 transformers>=4.20.1
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed/
cd Megatron-DeepSpeed
```

### Run

```
deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 128 --benchmark
[...]
*** Performance stats:
Throughput per token including tokenize: 0.73 msecs
Start to ready to generate: 591.902 secs
Tokenize and generate 64000 (bs=128) tokens: 9.421 secs
Start to finish: 601.323 secs

```

The highest batch size I was able to run without OOM was 128 in this case.






## DeepSpeed-ZeRO

[DeepSpeed-ZeRO](https://www.deepspeed.ai/tutorials/zero/) was primarily developed for training huge models, and wasn't targeting inference, but it still produces pretty decent results. ZeRO shards the weights across multiple GPUs and then gathers the needed weights magically before each `forward` and `backward` calls, so that the model isn't aware that its weights were ever split up.


### Setup

```
pip install deepspeed>=0.6.7 transformers>=4.20.1
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed/
cd Megatron-DeepSpeed
```

### Run

```
deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 8 --benchmark
[...]
*** Performance stats:
Throughput per token including tokenize: 37.86 msecs
Start to ready to generate: 464.724 secs
Tokenize and generate 32000 (bs=8) tokens: 242.090 secs
Start to finish: 706.813 secs
```

The highest batch size I was able to run without OOM was 8 in this case.



## Rust

XXX: Nicolas's work here


### Setup

```
```

### Run

```
```
