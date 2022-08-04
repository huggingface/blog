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

This article discusses various PyTorch-based solutions to run efficiently 176B parameter [BLOOM model](https://huggingface.co/bigscience/bloom).

As the model needs 352GB in bf16 (bfloat16) weights (`176*2`), the most efficient set-up is 8x80GB A100. The second best is 2x8x40GB A100s. The main reason for using A100s And, at the time of this writing, they provide the largest GPU memory. But other GPUs can be used as well. It'd probably take 24x32GB V100 as another possibility.

Using a single node is ideal since PCIe speed is typically much faster than inter-node network.

If you don't have that much hardware, it's still possible to run BLOOM inference on smaller GPUs, by using CPU or NVME offload, but of course, the generation time will be much slower.

For the sake of consistency, unless stated differently, the benchmarks in this article were all done on the same 8x80GB A100 node on [Jean Zay HPC](http://www.idris.fr/eng/jean-zay/index.html). The HPC users enjoy a very fast IO of about 3GB/s read speed (GPFS). This is important for loading speed time. A slow disc will result in slow loading time. Especially since we are concurrently doing IO in multiple processes.

## Performance metrics

For doing occasional inference on someone's desktop, that is a few inputs at a time, the two critical metrics are the time to load the framework and the checkpoint shards and then the generation time.

For doing inference on a server the two important metrics are the overall latency and the throughput of tokens. The loading time is less important since once loaded, the model becomes persistent.

The time to generate a token is straightforward - this is just how long it takes for a new token to be generated. As long as `use_cache=True` (the default argument to `generate`) all the previous logits are saved and therefore each token takes the same amount of time to generate. This of course comes at a memory cost - as the model is huge it can easily take GBs of memory to keep the cache. One can turn the caching off, and save memory, but it'd mean recalculating all the previous logits - which would make a long sequence generation very slow.

The latency is more difficult to define. If the server is idle and the model is ready to generate, and one or more prompts are sent from the same user at once, the latency would be just the time to generate a single token multiplied by the number of new tokens to generate. However, if you have multiple users sending prompts at different times, things become much more complicated. The first user gets the fastest response and thus fast latency, but subsequent users if they are unlucky to submit their request while the previous request is being processed may have to first wait till the first request has been generated and only their request will be attended. It's possible to design a very efficient system that uses a pipeline parallelism, where a new request can join the pipeline at any moment. But this has caveats. Then of course there is a small overhead of processing the input, sending data to the server, and sending the results back to the user, but this is usually insignificant compared to the cost of generation on a huge model.

Let's try and show the metrics in a drawing:

<img class="avatar avatar-user" src="/blog/assets/bloom-inference-pytorch/initial_drawing.png">

In this image, **T** is the simple forward latency. This is determined by how efficiently the **forward** pass is coded. But as you can see, using `8xA100` gives plenty of computing power so we can increase the batch size, and that's (to a point) not going to change **T**. But you will be generating more tokens within **T** so your throughput is effectively going up.
As you increase your batch_size, your memory consumption will likely go up too, so you might hit OOM errors, **before** actually reaching your computational boundary. 

This article primarily discusses simple solutions in PyTorch. More efficient solutions are being worked on as well.

The solutions are presented in alphabetical order:


## Accelerate

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
pip install accelerate transformers>=4.20.1
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed/
cd Megatron-DeepSpeed
git checkout bloom-inference
```

### Run

```
python scripts/inference/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 40 --benchmark
[...]

*** Performance stats:
Latency (T):  104 msecs
Throughput per token including tokenize: 10.74 msecs
Start to ready to generate: 107.892 secs
Tokenize and generate 20000 (bs=40) tokens: 52.059 secs
Start to finish: 159.952 secs
```

The highest batch size we were able to run without OOM was 40 in this case.



## DeepSpeed-Inference

[DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) uses Tensor-Parallelism and efficient fused CUDA kernels to deliver a super-fast <1msec per token inference on a large batch size of 128.



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
Latency (T):  18.8 msecs
Throughput per token including tokenize: 0.73 msecs
Start to ready to generate: 591.902 secs
Tokenize and generate 64000 (bs=128) tokens: 9.421 secs
Start to finish: 601.323 secs

```

The highest batch size we were able to run without OOM was 128 in this case.
You can see two factors at play leading to better performance here.

1/ T was reduced by using Tensor Parallelism (TP) instead of the Pipeline Parallelism (PP) of `accelerate`.
   Because accelerate is meant to be very generic it is also unfortunately hard to maximize the GPU usage.
   All computations are done first on GPU 0, then on GPU 1, etc... until GPU 8, which means
   7 GPU are idle all the time.
   `DeepSpeed` on the other end uses TP meaning it will send tensors to all GPUs, compute part of the computation
   then all GPUs communicate to each other the results, then move on to the next layer. That means all
   GPUs are active at once but they need to communicate much more. Overall that decreases **T** by a theoretical
   factor of `8x` (the theoretical limit would be `1.3s`).

2/ DeepSpeed also uses custom cuda kernels to avoid allocating too much memory and doing tensor copies
   The effect of this is less memory allocation and fewer kernel starts which does reduce **T** but also
   allows for bigger batch sizes (here **128** which also participates in raising the overall throughput.


## Splitting the generation


Another approach we took, was relooking at the overall behavior of the webserver in generation mode.

In practice when users send requests to a `text-generation` model, we don't run a single `forward` loop but multiple ones.
They also don't send the same parameters, some want `greedy`, some `sampling` and some `beam_search` for instance, and probably
with never really the same values.

In theory, if we're using `.generate` directly, we are **not** able to do any batching on different parameter sets. So what happens
is that we're less likely to use batching in production and that as mentioned above is a *big* source of throughput improvement.

Let's take a simple example where there's only 1 query running, and another query comes in while the first is being processed.

<img class="avatar avatar-user" src="/blog/assets/bloom-inference-pytorch/overall_latency.png">

As you can see here, the first request gets a latency of `3 x T` which is what we would expect. But **request 2** has to wait
for `4.5 x T`. which is bigger. If there was a **request 3** with yet another parameter set, then you would have to wait for `7.5 x T`.
And it piles up. The simple solution is to force a single (or handful) parameter sets so that we can ensure we're capping the overall latency.

Another approach is to split entirely the generation loop. We're going to ignore the `use_cache=True` complexity for the sake of readability.

So now, what we're going to do is have each request, handle the `generate` part on its own, and simply send back the required tensors only for
the `forward` pass. A single thread will be responsible for batching those simple requests.

Since each request is handling entirely its own way of choosing next ids, there is no problem in handling an infinite amount of different parameters.

You're getting a compute model like this.

<img class="avatar avatar-user" src="/blog/assets/bloom-inference-pytorch/reducing_server_latency.png">

Here you can see that at most, **request 2** has to wait for 1 forward pass to join the batch. And so even if **request 1** is super long (as in many tokens need to be generated),
**request 2** can start to execute almost immediately, and might even finish before **request 1** is done, and we actually see that happening in practice regularly.

As with any sort of optimization, the trick is using all sources of optimizations at once, for instance here since every request is doing its own `.generate` then
the tensors have to be copied a bunch of times (including the past key values). And any additional overhead adds up to increase **T** you need to make good care that doing this generation split is not offsetting all the other benefits.


Code here in Rust: https://github.com/Narsil/bloomserver 
It's in Rust, mostly because threading/async/multiprocessing have a lot of caveats in Python and was causing a lot of issues.
Rust doesn't provide any performance benefits here (it uses the same PyTorch under the hood) 
This code is not meant to be portable and should run on `16xA100(40)` and it also uses PP (like `accelerate`)
and uses a technique to interleave GPU usage to that the throughput should be more comparable to deepspeed,
even if the latency (**T**) is the same as `accelerate`.


There are other issues you need to take into account, like padding which might also hit your performance but this is
beyond this blogpost (More info [here](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/pipelines#pipeline-batching)).


### Setup

```
git clone https://github.com/Narsil/bloomserver.git
pip install setuptools_rust
pip install -r requirements.txt
python convert_weights.py #libtorch cannot load Python made weights so we have to convert them
TORCH_CUDA_VERSION=cu116 cargo run --release
```

### Run

**This experiment does not have a 1-1 benchmark analysis since it's running a full fledged webserver and was never ran on the same machines as before**

```
*** Performance stats (NUMBERS ARE ESTIMATES FROM QUERIES):
Latency (T):  104 msecs (should be similar to `accelerate`, it's using the roughly same code)
Throughput per token including tokenize: N/A (higher than accelerate, we're able to get 10X throughput compared to `accelerate`, on a similar machine)
Start to ready to generate: 15.902 secs 
Tokenize and generate N/A (bs=N/A) tokens: N/A
Start to finish: N/A
```

The faster load speed is due to `safetensors` (not a production ready library yet) is using `mmap` + there's 1 thread to load per GPU.
Overall the latency should be the same as `accelerate`, throughput is ~10X better and **most importantly, it can support any amount of different parameters**
without any latency cost.
