---
title: Goodbye cold boot - how we made LoRA Inference 300% faster
thumbnail: /blog/assets/171_load_lora_adapters/thumbnail3.png
authors:
- user: raphael-gl
---

# Goodbye cold boot - how we made LoRA Inference 300% faster

tl;dr: We swap the Stable Diffusion LoRA adapters per user request, while keeping the base model warm allowing fast LoRA inference across multiple users. You can experience this by browsing our [LoRA catalogue](https://huggingface.co/models?library=diffusers&other=lora) and playing with the inference widget.

![Inference Widget Example](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/171_load_lora_adapters/inference_widget.png)

In this blog we will go in detail over how we achieved that. 

We've been able to drastically speed up inference in the Hub for public LoRAs based on public Diffusion models. This has allowed us to save compute resources and provide a faster and better user experience. 

To perform inference on a given model, there are two steps:
1. Warm up phase - that consists in downloading the model and setting up the service (25s).
2. Then the inference job itself (10s).

With the improvements, we were able to reduce the warm up time from 25s to 3s. We are now able to serve inference for hundreds of distinct LoRAs, with less than 5 A10G GPUs, while the response time to user requests decreased from 35s to 13s.

Let's talk more about how we can leverage some recent features developed in the [Diffusers](https://github.com/huggingface/diffusers/) library to serve many distinct LoRAs in a dynamic fashion with one single service.

## LoRA

LoRA is a fine-tuning technique that belongs to the family of "parameter-efficient" (PEFT) methods, which try to reduce the number of trainable parameters affected by the fine-tuning process. It increases fine-tuning speed while reducing the size of fine-tuned checkpoints.

Instead of fine-tuning the model by performing tiny changes to all its weights, we freeze most of the layers and only train a few specific ones in the attention blocks. Furthermore, we avoid touching the parameters of those layers by adding the product of two smaller matrices to the original weights. Those small matrices are the ones whose weights are updated during the fine-tuning process, and then saved to disk. This means that all of the model original parameters are preserved, and we can load the LoRA weights on top using an adaptation method.

The LoRA name (Low Rank Adaptation) comes from the small matrices we mentioned. For more information about the method, please refer to [this post](https://huggingface.co/blog/lora) or the [original paper](https://arxiv.org/abs/2106.09685).

<div id="diagram"></div>

![LoRA decomposition](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/171_load_lora_adapters/lora_diagram.png)

The diagram above shows two smaller orange matrices that are saved as part of the LoRA adapter. We can later load the LoRA adapter and merge it with the blue base model to obtain the yellow fine-tuned model. Crucially, _unloading_ the adapter is also possible so we can revert back to the original base model at any point.

In other words, the LoRA adapter is like an add-on of a base model that can be added and removed on demand. And because of A and B smaller ranks, it is very light in comparison with the model size. Therefore, loading is much faster than loading the whole base model.

If you look, for example, inside the [Stable Diffusion XL Base 1.0 model repo](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main), which is widely used as a base model for many LoRA adapters, you can see that its size is around **7 GB**. However, typical LoRA adapters like [this one](https://huggingface.co/minimaxir/sdxl-wrong-lora/) take a mere **24 MB** of space !

There are far less blue base models than there are yellow ones on the Hub. If we can go quickly from the blue to yellow one and vice versa, then we have a way serve many distinct yellow models with only a few distinct blue deployments.

For a more exhaustive presentation on what LoRA is, please refer to the following blog post:[Using LoRA for Efficient Stable Diffusion Fine-Tuning](https://huggingface.co/blog/lora), or refer directly to the [original paper](https://arxiv.org/abs/2106.09685).

## Benefits

We have approximately **2500** distinct public LoRAs on the Hub. The vast majority (**~92%**) of them are LoRAs based on the [Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model.

Before this mutualization, this would have meant deploying a dedicated service for all of them (eg. for all the yellow merged matrices in the diagram above); releasing + reserving at least one new GPU. The time to spawn the service and have it ready to serve requests for a specific model is approximately **25s**, then on top of this you have the inference time (**~10s** for a 1024x1024 SDXL inference diffusion with 25 inference steps on an A10G). If an adapter is only occasionally requested, its service gets stopped to free resources preempted by others.

If you were requesting a LoRA that was not so popular, even if it was based on the SDXL model like the vast majority of adapters found on the Hub so far, it would have required **35s** to warm it up and get an answer on the first request (the following ones would have taken the inference time, eg. **10s**).

Now: request time has decreased from 35s to 13s since adapters will use only a few distinct "blue" base models (like 2 significant ones for Diffusion). Even if your adapter is not so popular, there is a good chance that its "blue" service is already warmed up. In other words, there is a good chance that you avoid the 25s warm up time, even if you do not request your model that often. The blue model is already downloaded and ready, all we have to do is unload the previous adapter and load the new one, which takes **3s** as we see [below](#loading-figures).

Overall, this requires less GPUs to serve all distinct models, even though we already had a way to share GPUs between deployments to maximize their compute usage. In a **2min** time frame, there are approximately **10** distinct LoRA weights that are requested. Instead of spawning 10 deployments, and keeping them warm, we simply serve all of them with 1 to 2 GPUs (or more if there is a request burst).


## Implementation

We implemented LoRA mutualization in the Inference API. When a request is performed on a model available in our platform, we first determine whether this is a LoRA or not. We then identify the base model for the LoRA and route the request to a common backend farm, with the ability to serve requests for the said model. Inference requests get served by keeping the base model warm and loading/unloading LoRAs on the fly. This way we can ultimately reuse the same compute resources to serve many distinct models at once.

### LoRA structure

In the Hub, LoRAs can be identified with two attributes:

![Hub](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/171_load_lora_adapters/lora_adapter_hub.png)

A LoRA will have a ```base_model``` attribute. This is simply the model which the LoRA was built for and should be applied to when performing inference.

Because LoRAs are not the only models with such an attribute (any duplicated model will have one), a LoRA will also need a ```lora``` tag to be properly identified.


### Loading/Offloading LoRA for Diffusers 🧨

<div class="alert">
<p>
Note that there is a more seemless way to perform the same as what is presented in this section using the <a href="https://github.com/huggingface/peft">peft</a> library. Please refer to <a href="https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference">the documentation</a> for more details. The principle remains the same as below (going from/to the blue box to/from the yellow one in the <a href="#diagram">diagram</a> above)
</p>
</div>
</br>

4 functions are used in the Diffusers library to load and unload distinct LoRA weights:

```load_lora_weights``` and ```fuse_lora``` for loading and merging weights with the main layers. Note that merging weights with the main model before performing inference can decrease the inference time by 30%.

```unload_lora_weights``` and ```unfuse_lora``` for unloading.

We provide an example below on how one can leverage the Diffusers library to quickly load several LoRA weights on top of a base model:

```py
import torch


from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
)

import time

base = "stabilityai/stable-diffusion-xl-base-1.0"

adapter1 = 'nerijs/pixel-art-xl'
weightname1 = 'pixel-art-xl.safetensors'

adapter2 = 'minimaxir/sdxl-wrong-lora'
weightname2 = None

inputs = "elephant"
kwargs = {}

if torch.cuda.is_available():
    kwargs["torch_dtype"] = torch.float16

start = time.time()

# Load VAE compatible with fp16 created by madebyollin
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)
kwargs["vae"] = vae
kwargs["variant"] = "fp16"

model = DiffusionPipeline.from_pretrained(
    base, **kwargs
)

if torch.cuda.is_available():
    model.to("cuda")

elapsed = time.time() - start

print(f"Base model loaded, elapsed {elapsed:.2f} seconds")


def inference(adapter, weightname):
    start = time.time()
    model.load_lora_weights(adapter, weight_name=weightname)
    # Fusing lora weights with the main layers improves inference time by 30 % !
    model.fuse_lora()
    elapsed = time.time() - start

    print(f"LoRA adapter loaded and fused to main model, elapsed {elapsed:.2f} seconds")

    start = time.time()
    data = model(inputs, num_inference_steps=25).images[0]
    elapsed = time.time() - start
    print(f"Inference time, elapsed {elapsed:.2f} seconds")

    start = time.time()
    model.unfuse_lora()
    model.unload_lora_weights()
    elapsed = time.time() - start
    print(f"LoRA adapter unfused/unloaded from base model, elapsed {elapsed:.2f} seconds")


inference(adapter1, weightname1)
inference(adapter2, weightname2)
```

## Loading figures

All numbers below are in seconds:

<table>
  <tr>
    <th>GPU</th>
    <td>T4</td>
    <td>A10G</td>
  </tr>
  <tr>
    <th>Base model loading - not cached</th>
    <td>20</td>
    <td>20</td>
  </tr>
  <tr>
    <th>Base model loading - cached</th>
    <td>5.95</td>
    <td>4.09</td>
  </tr>
  <tr>
    <th>Adapter 1 loading</th>
    <td>3.07</td>
    <td>3.46</td>
  </tr>
  <tr>
    <th>Adapter 1 unloading</th>
    <td>0.52</td>
    <td>0.28</td>
  </tr>
  <tr>
    <th>Adapter 2 loading</th>
    <td>1.44</td>
    <td>2.71</td>
  </tr>
  <tr>
    <th>Adapter 2 unloading</th>
    <td>0.19</td>
    <td>0.13</td>
  </tr>
  <tr>
    <th>Inference time</th>
    <td>20.7</td>
    <td>8.5</td>
  </tr>
</table>

With 2 to 4 additional seconds per inference, we can serve many distinct LoRAs. However, on an A10G GPU, the inference time decreases by a lot while the adapters loading time does not change much, so the LoRA's loading/unloading is relatively more expensive.

### Serving requests

To serve inference requests, we use [this open source community image](https://github.com/huggingface/api-inference-community/tree/main/docker_images/diffusers)

You can find the previously described mechanism used in the [TextToImagePipeline](https://github.com/huggingface/api-inference-community/blob/main/docker_images/diffusers/app/pipelines/text_to_image.py) class.

When a LoRA is requested, we'll look at the one that is loaded and change it only if required, then we perform inference as usual. This way, we are able to serve requests for the base model and many distinct adapters.

Below is an example on how you can test and request this image:

```
$ git clone https://github.com/huggingface/api-inference-community.git

$ cd api-inference-community/docker_images/diffusers

$ docker build -t test:1.0 -f Dockerfile .

$ cat > /tmp/env_file <<'EOF'
MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
TASK=text-to-image
HF_HUB_ENABLE_HF_TRANSFER=1
EOF

$ docker run --gpus all --rm --name test1 --env-file /tmp/env_file_minimal -p 8888:80 -it test:1.0
```

Then in another terminal perform requests to the base model and/or miscellaneous LoRA adapters to be found on the HF Hub.

```
# Request the base model
$ curl 0:8888 -d '{"inputs": "elephant", "parameters": {"num_inference_steps": 20}}' > /tmp/base.jpg

# Request one adapter
$ curl -H 'lora: minimaxir/sdxl-wrong-lora' 0:8888 -d '{"inputs": "elephant", "parameters": {"num_inference_steps": 20}}' > /tmp/adapter1.jpg

# Request another one
$ curl -H 'lora: nerijs/pixel-art-xl' 0:8888 -d '{"inputs": "elephant", "parameters": {"num_inference_steps": 20}}' > /tmp/adapter2.jpg
```

### What about batching ?

Recently a really interesting [paper](https://arxiv.org/abs/2311.03285) came out, that described how to increase the throughput by performing batched inference on LoRA models. In short, all inference requests would be gathered in a batch, the computation related to the common base model would be done all at once, then the remaining adapter-specific products would be computed. We did not implement such a technique (close to the approach adopted in [text-generation-inference](https://github.com/huggingface/text-generation-inference/) for LLMs). Instead, we stuck to single sequential inference requests. The reason is that we observed that batching was not interesting for diffusers: throughput does not increase significantly with batch size. On the simple image generation benchmark we performed, it only increased 25% for a batch size of 8, in exchange for 6 times increased latency! Comparatively, batching is far more interesting for LLMs because you get 8 times the sequential throughput with only a 10% latency increase. This is the reason why we did not implement batching for diffusers.


## Conclusion: **Time**!

Using dynamic LoRA loading, we were able to save compute resources and improve the user experience in the Hub Inference API. Despite the extra time added by the process of unloading the previously loaded adapter and loading the one we're interested in, the fact that the serving process is most often already up and running makes the inference time response on the whole much shorter.

Note that for a LoRA to benefit from this inference optimization on the Hub, it must both be public, non-gated and based on a non-gated public model. Please do let us know if you apply the same method to your deployment!
