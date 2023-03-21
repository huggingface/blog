---
title: "Accelerating Stable Diffusion Inference on Intel CPUs"
thumbnail: /blog/assets/xxx/01.png
authors:
- user: juliensimon
- user: ellacharlaix
---

# Accelerating Stable Diffusion Inference on Intel CPUs


<!-- {blog_metadata} -->
<!-- {authors} -->

Recently, we introduced the latest generation of [Intel Xeon](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html) CPUs (code name Sapphire Rapids), its new hardware features for deep learning acceleration, and how to use them to accelerate [distributed fine-tuning](https://huggingface.co/blog/intel-sapphire-rapids) and [inference](https://huggingface.co/blog/intel-sapphire-rapids-inference) for natural language processing Transformers.

In this post, we're going to show you different techniques to accelerate Stable Diffusion models on Sapphire Rapids CPUs. A follow-up post will do the same for distributed fine-tuning.

At the time of writing, the simplest way to get your hands on a Sapphire Rapids server is to use the Amazon EC2 [R7iz](https://aws.amazon.com/ec2/instance-types/r7iz/) instance family. As it's still in preview, you have to [sign up](https://pages.awscloud.com/R7iz-Preview.html) to get access. Like in previous posts, I'm using an `r7iz.metal-16xl` instance (64 vCPU, 512GB RAM) with an Ubunutu 20.04 AMI (`ami-07cd3e6c4915b2d18`).

Let's get started! Code samples are available on [Gitlab](https://gitlab.com/juliensimon/huggingface-demos/-/tree/main/optimum/stable_diffusion_intel).

## The Diffusers library

The [Diffusers](https://huggingface.co/docs/diffusers/index) library makes it extremely simple to generate images with Stable Diffusion models. If you're not familiar with these models, here's a great [illustrated introduction](https://jalammar.github.io/illustrated-stable-diffusion/).

First, let's create a virtual environment with the required libraries: Transformers, Diffusers, Accelerate, and PyTorch.

```
virtualenv sd_inference
source sd_inference/bin/activate
pip install transformers diffusers accelerate torch==1.13.1
```

Then, we write a simple benchmarking function that repeatedly runs inference, and returns the average latency for a single-image generation.

```python
import time

def elapsed_time(pipeline, prompt, nb_pass=10, num_inference_steps=20):
	# warmup
	images = pipeline(prompt, num_inference_steps=10).images
	start = time.time()
	for _ in range(nb_pass):
		_ = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np")
	end = time.time()
	return (end - start) / nb_pass
```

Now, let's build a `StableDiffusionPipeline` with the default `float32` data type, and measure its inference latency.

```python
import torch
from diffusers import StableDiffusionPipeline

prompt = "sailing ship in storm by Rembrandt"
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
latency = elapsed_time(pipe, prompt)
print(latency)
```

The average latency is **32.6 seconds**. As demonstrated by this [Intel Space](https://huggingface.co/spaces/Intel/Stable-Diffusion-Side-by-Side), the same code runs on a previous generation Intel Xeon (code name Ice Lake) in about 45 seconds. 

Out of the box, we can see that Sapphire Rapids CPUs are quite faster without any code change!

Now, let's accelerate!

## Optimum Intel and OpenVINO

[Optimum Intel](https://huggingface.co/docs/optimum/intel/index) accelerates end-to-end pipelines on Intel architectures. Its API is extremely similar to the vanilla [Diffusers](https://huggingface.co/docs/diffusers/index) API, making it trivial to adapt existing code.

Optimum Intel supports [OpenVINO](https://docs.openvino.ai/latest/index.html), an Intel open-source toolkit for high-performance inference. 

Optimum Intel and OpenVINO can be installed as follows:

```
pip install optimum[openvino]
```

Starting from the code above, we only need to replace `StableDiffusionPipeline` with `OVStableDiffusionPipeline`.

```python
from optimum.intel.openvino import OVStableDiffusionPipeline
...
ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)

latency = elapsed_time(ov_pipe, prompt)
print(latency)
```

OpenVINO automatically optimizes the model for the `bfloat16` format. Thanks to this, the average latency is now **16.7 seconds**, a sweet 2x speedup.

The pipeline above support dynamic input shapes, with no restriction on the number of images or their resolution. If your application can support a static input shape (say, always generate a single 512x512 image), you can unlock significant acceleration by reshaping the pipeline.

```python
ov_pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
latency = elapsed_time(ov_pipe, prompt)
```

With a static shape, average latency is slashed to **4.7 seconds**, an additional 3.5x speedup. 

As you can see, OpenVINO is a simple and efficient way to accelerate Stable Diffusion inference. When combined with a Sapphire Rapids CPU, it delivers almost 10x speedup compared to vanilla inference on Ice Lake Xeons.

If you can't or don't want to use OpenVINO, the rest of this post will show you a series of other optimization techniques. Fasten your seatbelt!

## Memory allocation

Diffuser models are large multi-gigabyte models, and image generation is a memory-intensive operation. By installing a high-performance memory allocation library, we should be able to speed up memory operations and parallelize them across the Xeon cores.    Please note that this will change the default memory allocation library on your system. Of course, you can go back to the default library by uninstalling the new one.

[jemalloc](https://jemalloc.net/) and [tcmalloc](https://github.com/gperftools/gperftools) are equally interesting. Here, I'm installing `jemalloc` as my tests give it a slight performance edge. Jemalloc can also be tweaked for a particular workload, for example to maximize CPU utilization. You can refer to the [tuning guide](https://github.com/jemalloc/jemalloc/blob/dev/TUNING.md) for details.

```
# Option 1: jemalloc
sudo apt-get install -y libjemalloc-dev
export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms: 30000,muzzy_decay_ms:30000"

# Option 2: tcmalloc
sudo apt-get install -y google-perftools
export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libtcmalloc.so
```

Running our original Diffusers code, the average latency drops from 32.6 seconds to **11.9 seconds**. That's almost 3x faster, without any code change. Jemalloc is certainly working great on our 32-core Xeon.

We're far from done. Let's add the Intel Extension for PyTorch to the mix.

## IPEX and BF16

The [Intel Extension for Pytorch](https://intel.github.io/intel-extension-for-pytorch/) (IPEX) extends PyTorch and takes advantage of hardware acceleration features present on Intel CPUs, such as [AVX-512](https://en.wikipedia.org/wiki/AVX-512) Vector Neural Network Instructions (AVX512 VNNI) and [Advanced Matrix Extensions](https://en.wikipedia.org/wiki/Advanced_Matrix_Extensions) (AMX).

Let's install it.

```
pip install intel_extension_for_pytorch
```

We then update our code to optimize each pipeline element with IPEX (you can list them by printing the `pipe` object). This requires converting them to the channels-last format.

```python
import intel_extension_for_pytorch as ipex
...
pipe = StableDiffusionPipeline.from_pretrained(model_id)
# to channels last
pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
pipe.safety_checker = pipe.safety_checker.to(memory_format=torch.channels_last)

# optimize with ipex
pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=torch.bfloat16, inplace=True)
pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=torch.bfloat16, inplace=True)
pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
pipe.safety_checker = ipex.optimize(pipe.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)
```

We also enable the `bloat16` data format to leverage the AMX tile matrix multiply unit (TMMU) accelerator present on Sapphire Rapids CPUs.

```python
with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    latency = elapsed_time(pipe, prompt)
    print(latency)
```

With this updated version, inference latency is further reduced from 11.9 seconds to **6.3 seconds**. That's almost an extra 2x acceleration thanks to IPEX and AMX.

Can we extract a bit more performance? Yes, with schedulers!

## Schedulers 

The Diffusers library lets us attach a [scheduler](https://huggingface.co/docs/diffusers/using-diffusers/schedulers) to a Stable Diffusion pipeline. Schedulers try to find the best trade-off between denoising speed and denoising quality.

According to the documentation: "*At the time of writing this doc DPMSolverMultistepScheduler gives arguably the best speed/quality trade-off and can be run with as little as 20 steps.*"

Let's try it.

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
...
dpm = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=dpm)
```

With this final version, inference latency is now down to **6 seconds**. Compared to our initial baseline, this is more than 5x faster (5.4x to be precise).

<kbd>
  <img src="assets/xxx_stable_diffusion_inference_intel/01.png">
</kbd>



## Conclusion

The ability to generate high-quality images in seconds should work well for a lot of use cases, such as customer apps, content generation for marketing and media, or synthetic data for dataset augmentation.

Here are some resources to help you get started:

* Diffusers [documentation](https://huggingface.co/docs/diffusers),
* [Intel IPEX](https://github.com/intel/intel-extension-for-pytorch) on GitHub
* [Developer resources](https://www.intel.com/content/www/us/en/developer/partner/hugging-face.html) from Intel and Hugging Face. 

If you have questions or feedback, we'd love to read them on the [Hugging Face forum](https://discuss.huggingface.co/).

Thanks for reading!



 
