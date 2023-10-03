---
title: Optimizing Stable Diffusion for Intel CPUs with NNCF and 🤗 Optimum
thumbnail: /blog/assets/train_optimize_sd_intel/thumbnail.png
authors:
- user: AlexKoff88
  guest: true
- user: MrOpenVINO
  guest: true
- user: helenai
  guest: true
- user: sayakpaul
- user: echarlaix
---

# Optimizing Stable Diffusion for Intel CPUs with NNCF and 🤗 Optimum


[**Latent Diffusion models**](https://arxiv.org/abs/2112.10752) are game changers when it comes to solving text-to-image generation problems. [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release) is one of the most famous examples that got wide adoption in the community and industry. The idea behind the Stable Diffusion model is simple and compelling: you generate an image from a noise vector in multiple small steps refining the noise to a latent image representation. This approach works very well, but it can take a long time to generate an image if you do not have access to powerful GPUs. 

Through the past five years, [OpenVINO Toolkit](https://docs.openvino.ai/) encapsulated many features for high-performance inference. Initially designed for Computer Vision models, it still dominates in this domain showing best-in-class inference performance for many contemporary models, including [Stable Diffusion](https://huggingface.co/blog/stable-diffusion-inference-intel). However, optimizing Stable Diffusion models for resource-constraint applications requires going far beyond just runtime optimizations. And this is where model optimization capabilities from OpenVINO [Neural Network Compression Framework](https://github.com/openvinotoolkit/nncf) (NNCF) come into play.

In this blog post, we will outline the problems of optimizing Stable Diffusion models and propose a workflow that substantially reduces the latency of such models when running on a resource-constrained HW such as CPU. In particular, we achieved **5.1x** inference acceleration and **4x** model footprint reduction compared to PyTorch.

## Stable Diffusion optimization

In the [Stable Diffusion pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview), the UNet model is computationally the most expensive to run. Thus, optimizing just one model brings substantial benefits in terms of inference speed.

However, it turns out that the traditional model optimization methods, such as post-training 8-bit quantization, do not work for this model. There are two main reasons for that. First, pixel-level prediction models, such as semantic segmentation, super-resolution, etc., are one of the most complicated in terms of model optimization because of the complexity of the task, so tweaking model parameters and the structure breaks the results in numerous ways. The second reason is that the model has a lower level of redundancy because it accommodates a lot of information while being trained on [hundreds of millions of samples](https://laion.ai/blog/laion-5b/). That is why researchers have to employ more sophisticated quantization methods to preserve the accuracy after optimization. For example, Qualcomm used the layer-wise Knowledge Distillation method ([AdaRound](https://arxiv.org/abs/2004.10568)) to [quantize](https://www.qualcomm.com/news/onq/2023/02/worlds-first-on-device-demonstration-of-stable-diffusion-on-android) Stable Diffusion models. It means that model tuning after quantization is required, anyway. If so, why not just use [Quantization-Aware Training](https://arxiv.org/abs/1712.05877) (QAT) which can tune the model and quantization parameters simultaneously in the same way the source model is trained? Thus, we tried this approach in our work using [NNCF](https://github.com/openvinotoolkit/nncf), [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html), and [Diffusers](https://github.com/huggingface/diffusers) and coupled it with [Token Merging](https://arxiv.org/abs/2210.09461).

## Optimization workflow

We usually start the optimization of a model after it's trained. Here, we start from a [model](https://huggingface.co/svjack/Stable-Diffusion-Pokemon-en) fine-tuned on the [Pokemons dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) containing images of Pokemons and their text descriptions.

We used the [text-to-image fine-tuning example](https://huggingface.co/docs/diffusers/training/text2image) for Stable Diffusion from the Diffusers and integrated QAT from NNCF into the following training [script](https://github.com/huggingface/optimum-intel/tree/main/examples/openvino/stable-diffusion). We also changed the loss function to incorporate knowledge distillation from the source model that acts as a teacher in this process while the actual model being trained acts as a student. This approach is different from the classical knowledge distillation method, where the trained teacher model is distilled into a smaller student model. In our case, knowledge distillation is used as an auxiliary method that helps improve the final accuracy of the optimizing model. We also use the Exponential Moving Average (EMA) method for model parameters excluding quantizers which allows us to make the training process more stable. We tune the model for 4096 iterations only.

With some tricks, such as gradient checkpointing and [keeping the EMA model](https://github.com/huggingface/optimum-intel/blob/bbbe7ff0e81938802dbc1d234c3dcdf58ef56984/examples/openvino/stable-diffusion/train_text_to_image_qat.py#L941) in RAM instead of VRAM, we can run the optimization process using one GPU with 24 GB of VRAM. The whole optimization takes less than a day using one GPU!

## Going beyond Quantization-Aware Training

Quantization alone can bring significant enhancements by reducing model footprint, load time, memory consumption, and inference latency. But the great thing about quantization is that it can be applied along with other optimization methods leading to a cumulative speedup.

Recently, Facebook Research introduced a [Token Merging](https://arxiv.org/abs/2210.09461) method for Vision Transformer models. The essence of the method is that it merges redundant tokens with important ones using one of the available strategies (averaging, taking max values, etc.). This is done before the self-attention block, which is the most computationally demanding part of Transformer models. Therefore, reducing the token dimension reduces the overall computation time in the self-attention blocks. This method has also been [adapted](https://arxiv.org/pdf/2303.17604.pdf) for Stable Diffusion models and has shown promising results when optimizing Stable Diffusion pipelines for high-resolution image synthesis running on GPUs.
 
We modified the Token Merging method to be compliant with OpenVINO and stacked it with 8-bit quantization when applied to the Attention UNet model. This also involves all the mentioned techniques including Knowledge Distillation, etc. As for quantization, it requires fine-tuning to be applied to restore the accuracy. We also start optimization and fine-tuning from the [model](https://huggingface.co/svjack/Stable-Diffusion-Pokemon-en) trained on the [Pokemons dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions). The figure below shows an overall optimization workflow.

![overview](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-optimize-sd-intel/overview.png)

The resultant model is highly beneficial when running inference on devices with limited computational resources, such as client or edge CPUs. As it was mentioned, stacking Token Merging with quantization leads to an additional reduction in the inference latency.

<div class="flex flex-row">
<div class="grid grid-cols-2 gap-4">
<figure>
<img class="max-w-full rounded-xl border-2 border-solid border-gray-600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-optimize-sd-intel/image_torch.png" alt="Image 1" />
<figcaption class="mt-2 text-center text-sm text-gray-500">PyTorch FP32, Inference Speed: 230.5 seconds, Memory Footprint: 3.44 GB</figcaption>
</figure>
<figure>
<img class="max-w-full rounded-xl border-2 border-solid border-gray-600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-optimize-sd-intel/image_fp32.png" alt="Image 2" />
<figcaption class="mt-2 text-center text-sm text-gray-500">OpenVINO FP32, Inference Speed: 120 seconds (<b>1.9x</b>), Memory Footprint: 3.44 GB</figcaption>
</figure>
<figure>
<img class="max-w-full rounded-xl border-2 border-solid border-gray-600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-optimize-sd-intel/image_quantized.png" alt="Image 3" />
<figcaption class="mt-2 text-center text-sm text-gray-500">OpenVINO 8-bit, Inference Speed: 59 seconds (<b>3.9x</b>), Memory Footprint: 0.86 GB (<b>0.25x</b>)</figcaption>
</figure>
<figure>
<img class="max-w-full rounded-xl border-2 border-solid border-gray-600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-optimize-sd-intel/image_tome_quantized.png" alt="Image 4" />
<figcaption class="mt-2 text-center text-sm text-gray-500">ToMe + OpenVINO 8-bit, Inference Speed: 44.6 seconds (<b>5.1x</b>), Memory Footprint: 0.86 GB (<b>0.25x</b>)</figcaption>
</figure>
</div>
</div>

Results of image generation [demo](https://huggingface.co/spaces/helenai/stable_diffusion) using different optimized models. Input prompt is “cartoon bird”, seed is 42. The models are with OpenVINO 2022.3 in [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-overview) using a “CPU upgrade” instance which utilizes 3rd Generation Intel® Xeon® Scalable Processors with Intel® Deep Learning Boost technology.

## Results

We used the disclosed optimization workflows to get two types of optimized models, 8-bit quantized and quantized with Token Merging, and compare them to the PyTorch baseline. We also converted the baseline to vanilla OpenVINO floating-point (FP32) model for the comprehensive comparison.

The picture above shows the results of image generation and some model characteristics. As you can see, just conversion to OpenVINO brings a significant decrease in the inference latency ( **1.9x** ). Applying 8-bit quantization boosts inference speed further leading to **3.9x** speedup compared to PyTorch. Another benefit of quantization is a significant reduction of model footprint, **0.25x** of PyTorch checkpoint, which also improves the model load time. Applying Token Merging (ToME) (with a **merging ratio of 0.4** ) on top of quantization brings **5.1x** performance speedup while keeping the footprint at the same level. We didn't provide a thorough analysis of the visual quality of the optimized models, but, as you can see, the results are quite solid.

For the results shown in this blog, we used the default number of 50 inference steps. With fewer inference steps, inference speed will be faster, but this has an effect on the quality of the resulting image. How large this effect is depends on the model and the [scheduler](https://huggingface.co/docs/diffusers/using-diffusers/schedulers). We recommend experimenting with different number of steps and schedulers and find what works best for your use case.

Below we show how to perform inference with the final pipeline optimized to run on Intel CPUs:

```python
from optimum.intel import OVStableDiffusionPipeline

# Load and compile the pipeline for performance.
name = "OpenVINO/stable-diffusion-pokemons-tome-quantized-aggressive"
pipe = OVStableDiffusionPipeline.from_pretrained(name, compile=False)
pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
pipe.compile()

# Generate an image.
prompt = "a drawing of a green pokemon with red eyes"
output = pipe(prompt, num_inference_steps=50, output_type="pil").images[0]
output.save("image.png")
```

You can find the training and quantization [code](https://github.com/huggingface/optimum-intel/tree/main/examples/openvino/stable-diffusion) in the Hugging Face [Optimum Intel](https://huggingface.co/docs/optimum/main/en/intel/index) library. The notebook that demonstrates the difference between optimized and original models is available [here](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/stable_diffusion_optimization.ipynb). You can also find [many models](https://huggingface.co/models?library=openvino&sort=downloads) on the Hugging Face Hub under the [OpenVINO organization](https://huggingface.co/OpenVINO). In addition, we have created a [demo](https://huggingface.co/spaces/helenai/stable_diffusion) on Hugging Face Spaces that is being run on a 3rd Generation Intel Xeon Scalable processor.

## What about the general-purpose Stable Diffusion model?

As we showed with the Pokemon image generation task, it is possible to achieve a high level of optimization of the Stable Diffusion pipeline when using a relatively small amount of training resources. At the same time, it is well-known that training a general-purpose Stable Diffusion model is an [expensive task](https://www.mosaicml.com/blog/training-stable-diffusion-from-scratch-part-2). However, with enough budget and HW resources, it is possible to optimize the general-purpose model using the described approach and tune it to produce high-quality images. The only caveat we have is related to the token merging method that reduces the model capacity substantially. The rule of thumb here is the more complicated the dataset you have for the training, the less merging ratio you should use during the optimization.

If you enjoyed reading this post, you might also be interested in checking out [this post](https://huggingface.co/blog/stable-diffusion-inference-intel) that discusses other complementary approaches to optimize the performance of Stable Diffusion on 4th generation Intel Xeon CPUs.
