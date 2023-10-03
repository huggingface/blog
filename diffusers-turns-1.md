---
title: "Happy 1st anniversary 🤗 Diffusers!" 
thumbnail: /blog/assets/diffusers-turns-1/diffusers-turns-1.png
authors:
- user: stevhliu
- user: sayakpaul
- user: pcuenq
---

# Happy 1st anniversary 🤗 Diffusers!


🤗 Diffusers is happy to celebrate its first anniversary! It has been an exciting year, and we're proud and grateful for how far we've come thanks to our community and open-source contributors. Last year, text-to-image models like DALL-E 2, Imagen, and Stable Diffusion captured the world's attention with their ability to generate stunningly photorealistic images from text, sparking a massive surge of interest and development in generative AI. But access to these powerful models was limited.

At Hugging Face, our mission is to democratize good machine learning by collaborating and helping each other build an open and ethical AI future together. Our mission motivated us to create the 🤗 Diffusers library so *everyone* can experiment, research, or simply play with text-to-image models. That’s why we designed the library as a modular toolbox, so you can customize a diffusion model’s components or just start using it out-of-the-box.

As 🤗 Diffusers turns 1, here’s an overview of some of the most notable features we’ve added to the library with the help of our community. We are proud and immensely grateful for being part of an engaged community that promotes accessible usage, pushes diffusion models beyond just text-to-image generation, and is an all-around inspiration.

**Table of Contents**

* [Striving for photorealism](#striving-for-photorealism)
* [Video pipelines](#video-pipelines)
* [Text-to-3D models](#text-to-3d-models)
* [Image editing pipelines](#image-editing-pipelines)
* [Faster diffusion models](#faster-diffusion-models)
* [Ethics and safety](#ethics-and-safety)
* [Support for LoRA](#support-for-lora)
* [Torch 2.0 optimizations](#torch-20-optimizations)
* [Community highlights](#community-highlights)
* [Building products with 🤗 Diffusers](#building-products-with-🤗-diffusers)
* [Looking forward](#looking-forward)

## Striving for photorealism

Generative AI models are known for creating photorealistic images, but if you look closely, you may notice certain things that don't look right, like generating extra fingers on a hand. This year, the DeepFloyd IF and Stability AI SDXL models made a splash by improving the quality of generated images to be even more photorealistic.

[DeepFloyd IF](https://stability.ai/blog/deepfloyd-if-text-to-image-model) - A modular diffusion model that includes different processes for generating an image (for example, an image is upscaled 3x to produce a higher resolution image). Unlike Stable Diffusion, the IF model works directly on the pixel level, and it uses a large language model to encode text.

[Stable Diffusion XL (SDXL)](https://stability.ai/blog/sdxl-09-stable-diffusion) - The latest Stable Diffusion model from Stability AI, with significantly more parameters than its predecessor Stable Diffusion 2. It generates hyper-realistic images, leveraging a base model for close adherence to the prompt, and a refiner model specialized in the fine details and high-frequency content.

Head over to the DeepFloyd IF [docs](https://huggingface.co/docs/diffusers/v0.18.2/en/api/pipelines/if#texttoimage-generation) and the SDXL [docs](https://huggingface.co/docs/diffusers/v0.18.2/en/api/pipelines/stable_diffusion/stable_diffusion_xl) today to learn how to start generating your own images!

## Video pipelines

Text-to-image pipelines are cool, but text-to-video is even cooler! We currently support two text-to-video pipelines, [VideoFusion](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video) and [Text2Video-Zero](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video_zero).

If you’re already familiar with text-to-image pipelines, using a text-to-video pipeline is very similar:

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=24).frames
video_path = export_to_video(video_frames)
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/darthvader_cerpense.gif" alt="Generated video of Darth Vader surfing."/>
</div>

We expect text-to-video to go through a revolution during 🤗 Diffusers second year, and we are excited to see what the community builds on top of these to push the boundaries of video generation from language!

## Text-to-3D models

In addition to text-to-video, we also have text-to-3D generation now thanks to OpenAI’s [Shap-E](https://hf.co/papers/2305.02463) model. Shap-E is trained by encoding a large dataset of 3D-text pairs, and a diffusion model is conditioned on the encoder’s outputs. You can design 3D assets for video games, interior design, and architecture. 

Try it out today with the [`ShapEPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/shap_e#diffusers.ShapEPipeline) and [`ShapEImg2ImgPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/shap_e#diffusers.ShapEImg2ImgPipeline).

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/cake_out.gif" alt="3D render of a birthday cupcake generated using SHAP-E."/>
</div>

## Image editing pipelines

Image editing is one of the most practical use cases in fashion, material design, and photography. With diffusion models, the possibilities of image editing continue to expand.

We have many [pipelines](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlling_generation) in 🤗 Diffusers to support image editing. There are image editing pipelines that allow you to describe your desired edit as a prompt, removing concepts from an image, and even a pipeline that unifies multiple generation methods to create high-quality images like panoramas. With 🤗 Diffusers, you can experiment with the future of photo editing now!

## Faster diffusion models

Diffusion models are known to be time-intensive because of their iterative steps. With OpenAI’s [Consistency Models](https://huggingface.co/papers/2303.01469), the image generation process is significantly faster. Generating a single 256x256 resolution image only takes 3/4 of a second on a modern CPU! You can try this out in 🤗 Diffusers with the [`ConsistencyModelPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/consistency_models).

On top of speedier diffusion models, we also offer many optimization techniques for faster inference like [PyTorch 2.0’s `scaled_dot_product_attention()` (SDPA) and `torch.compile()`](https://pytorch.org/blog/accelerated-diffusers-pt-20), sliced attention, feed-forward chunking, VAE tiling, CPU and model offloading, and more. These optimizations save memory, which translates to faster generation, and allow you to run inference on consumer GPUs. When you distribute a model with 🤗 Diffusers, all of these optimizations are immediately supported!

In addition to that, we also support specific hardware and formats like ONNX, the `mps` PyTorch device for Apple Silicon computers, Core ML, and others.

To learn more about how we optimize inference with 🤗 Diffusers, check out the [docs](https://huggingface.co/docs/diffusers/optimization/opt_overview)!

## Ethics and safety

Generative models are cool, but they also have the ability to produce harmful and NSFW content. To help users interact with these models responsibly and ethically, we’ve added a [`safety_checker`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) component that flags inappropriate content generated during inference. Model creators can choose to incorporate this component into their models if they want.

In addition, generative models can also be used to produce disinformation. Earlier this year, the [Balenciaga Pope](https://www.theverge.com/2023/3/27/23657927/ai-pope-image-fake-midjourney-computer-generated-aesthetic) went viral for how realistic the image was despite it being fake. This underscores the importance and need for a mechanism to distinguish between generated and human content. That’s why we’ve added an invisible watermark for images generated by the SDXL model, which helps users be better informed.

The development of these features is guided by our [ethical charter](https://huggingface.co/docs/diffusers/main/en/conceptual/ethical_guidelines), which you can find in our documentation.

## Support for LoRA

Fine-tuning diffusion models is expensive and out of reach for most consumer GPUs. We added the Low-Rank Adaptation ([LoRA](https://huggingface.co/papers/2106.09685)) technique to close this gap. With LoRA, which is a method for parameter-efficient fine-tuning, you can fine-tune large diffusion models faster and consume less memory. The resulting model weights are also very lightweight compared to the original model, so you can easily share your custom models. If you want to learn more, [our documentation](https://huggingface.co/docs/diffusers/main/en/training/lora) shows how to perform fine-tuning and inference on Stable Diffusion with LoRA.

In addition to LoRA, we support other [training techniques](https://huggingface.co/docs/diffusers/main/en/training/overview) for personalized generation, including DreamBooth, textual inversion, custom diffusion, and more!

## Torch 2.0 optimizations

PyTorch 2.0 [introduced support](https://pytorch.org/get-started/pytorch-2.0/#pytorch-2x-faster-more-pythonic-and-as-dynamic-as-ever) for `torch.compile()`and `scaled_dot_product_attention()`, a more efficient implementation of the attention mechanism. 🤗 Diffusers [provides first-class support](https://huggingface.co/docs/diffusers/optimization/torch2.0) for these features resulting in massive speedups in inference latency, which can sometimes be more than twice as fast!

In addition to visual content (images, videos, 3D assets, etc.), we also added support for audio! Check out [the documentation](https://huggingface.co/docs/diffusers/using-diffusers/audio) to learn more.

## Community highlights

One of the most gratifying experiences of the past year has been seeing how the community is incorporating 🤗 Diffusers into their projects. From adapting Low-rank adaptation (LoRA) for faster training of text-to-image models to building a state-of-the-art inpainting tool, here are a few of our favorite projects:


<div class="mx-auto max-w-screen-xl py-8">
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">We built Core ML Stable Diffusion to make it easier for developers to add state-of-the-art generative AI capabilities in their iOS, iPadOS and macOS apps with the highest efficiency on Apple Silicon. We built on top of 🤗 Diffusers instead of from scratch as 🤗 Diffusers consistently stays on top of a rapidly evolving field and promotes much needed interoperability of new and old ideas.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/10639145?s=200&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Atila Orhon</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">🤗 Diffusers has been absolutely developer-friendly for me to dive right into stable diffusion models. Main differentiating factor clearly being that 🤗 Diffusers implementation is often not some code from research lab, that are mostly focused on high velocity driven. While research codes are often poorly written and difficult to understand (lack of typing, assertions, inconsistent design patterns and conventions), 🤗 Diffusers was a breeze to use for me to hack my ideas within couple of hours. Without it, I would have needed to invest significantly more amount of time to start hacking. Well-written documentations and examples are extremely helpful as well.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/35953539?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Simo</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">BentoML is the unified framework for for building, shipping, and scaling production-ready AI applications incorporating traditional ML, pre-trained AI models, Generative and Large Language Models. All Hugging Face Diffuser models and pipelines can be seamlessly integrated into BentoML applications, enabling the running of models on the most suitable hardware and independent scaling based on usage.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/49176046?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">BentoML</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Invoke AI is an open-source Generative AI tool built to empower professional creatives, from game designers and photographers to architects and product designers. Invoke recently launched their hosted offering at invoke.ai, allowing users to generate assets from any computer, powered by the latest research in open-source.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/113954515?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">InvokeAI</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">TaskMatrix connects Large Language Model and a series of Visual Models to enable sending and receiving images during chatting.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/6154722?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Chenfei Wu</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Lama Cleaner is a powerful image inpainting tool that uses Stable Diffusion technology to remove unwanted objects, defects, or people from your pictures. It can also erase and replace anything in your images with ease.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://github.com/Sanster/lama-cleaner/raw/main/assets/logo.png" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Qing</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Grounded-SAM combines a powerful Zero-Shot detector Grounding-DINO and Segment-Anything-Model (SAM) to build a strong pipeline to detect and segment everything with text inputs. When combined with 🤗 Diffusers inpainting models, Grounded-SAM can do highly controllable image editing tasks, including replacing specific objects, inpainting the background, etc.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/113572103?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Tianhe Ren</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Stable-Dreamfusion leverages the convenient implementations of 2D diffusion models in 🤗 Diffusers to replicate recent text-to-3D and image-to-3D methods.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/25863658?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">kiui</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">MMagic (Multimodal Advanced, Generative, and Intelligent Creation) is an advanced and comprehensive Generative AI toolbox that provides state-of-the-art AI models (e.g., diffusion models powered by 🤗 Diffusers and GAN) to synthesize, edit and enhance images and videos. In MMagic, users can use rich components to customize their own models like playing with Legos and manage the training loop easily.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/10245193?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">mmagic</p>
      </div>
    </div>
  </div>
  <div class="mb-8 sm:break-inside-avoid">
    <blockquote class="rounded-xl !mb-0 bg-gray-50 p-6 shadow dark:bg-gray-800">
      <p class="leading-relaxed text-gray-700">Tune-A-Video, developed by Jay Zhangjie Wu and his team at Show Lab, is the first to fine-tune a pre-trained text-to-image diffusion model using a single text-video pair and enables changing video content while preserving motion.</p>
    </blockquote>
    <div class="flex items-center gap-4">
      <img src="https://avatars.githubusercontent.com/u/101181824?s=48&v=4" class="h-12 w-12 rounded-full object-cover" />
      <div class="text-sm">
        <p class="font-medium">Jay Zhangjie Wu</p>
      </div>
    </div>
  </div>
</div>

We also collaborated with Google Cloud (who generously provided the compute) to provide technical guidance and mentorship to help the community train diffusion models with TPUs (check out a summary of the event [here](https://opensource.googleblog.com/2023/06/controlling-stable-diffusion-with-jax-diffusers-and-cloud-tpus.html)). There were many cool models such as this [demo](https://huggingface.co/spaces/mfidabel/controlnet-segment-anything) that combines ControlNet with Segment Anything.

<div class="flex justify-center">
  <img src="https://github.com/mfidabel/JAX_SPRINT_2023/blob/8632f0fde7388d7a4fc57225c96ef3b8411b3648/EX_1.gif?raw=true" alt="ControlNet and SegmentAnything demo of a hot air balloon in various styles">
</div>

Finally, we were delighted to receive contributions to our codebase from over 300 contributors, which allowed us to collaborate together in the most open way possible. Here are just a few of the contributions from our community:

- [Model editing](https://github.com/huggingface/diffusers/pull/2721) by [@bahjat-kawar](https://github.com/bahjat-kawar), a pipeline for editing a model’s implicit assumptions
- [LDM3D](https://github.com/huggingface/diffusers/pull/3668) by [@estelleafl](https://github.com/estelleafl), a diffusion model for 3D images
- [DPMSolver](https://github.com/huggingface/diffusers/pull/3314) by [@LuChengTHU](https://github.com/LuChengTHU), improvements for significantly improving inference speed
- [Custom Diffusion](https://github.com/huggingface/diffusers/pull/3031) by [@nupurkmr9](https://github.com/nupurkmr9), a technique for generating personalized images with only a few images of a subject

Besides these, a heartfelt shoutout to the following contributors who helped us ship some of the most powerful features of Diffusers (in no particular order):

* [@takuma104](https://github.com/huggingface/diffusers/commits?author=takuma104)
* [@nipunjindal](https://github.com/huggingface/diffusers/commits?author=nipunjindal)
* [@isamu-isozaki](https://github.com/huggingface/diffusers/commits?author=isamu-isozaki)
* [@piEsposito](https://github.com/huggingface/diffusers/commits?author=piEsposito)
* [@Birch-san](https://github.com/huggingface/diffusers/commits?author=Birch-san)
* [@LuChengTHU](https://github.com/huggingface/diffusers/commits?author=LuChengTHU)
* [@duongna21](https://github.com/huggingface/diffusers/commits?author=duongna21)
* [@clarencechen](https://github.com/huggingface/diffusers/commits?author=clarencechen)
* [@dg845](https://github.com/huggingface/diffusers/commits?author=dg845)
* [@Abhinay1997](https://github.com/huggingface/diffusers/commits?author=Abhinay1997)
* [@camenduru](https://github.com/huggingface/diffusers/commits?author=camenduru)
* [@ayushtues](https://github.com/huggingface/diffusers/commits?author=ayushtues)

## Building products with 🤗 Diffusers

Over the last year, we also saw many companies choosing to build their products on top of 🤗 Diffusers. Here are a couple of products that have caught our attention:

- [PlaiDay](http://plailabs.com/): “PlaiDay is a Generative AI experience where people collaborate, create, and connect. Our platform unlocks the limitless creativity of the human mind, and provides a safe, fun social canvas for expression.”
- [Previs One](https://previs.framer.wiki/): “Previs One is a diffuser pipeline for cinematic storyboarding and previsualization — it understands film and television compositional rules just as a director would speak them.”
- [Zust.AI](https://zust.ai/): “We leverage Generative AI to create studio-quality product photos for brands and marketing agencies.”
- [Dashtoon](https://dashtoon.com/): “Dashtoon is building a platform to create and consume visual content. We have multiple pipelines that load multiple LORAs, multiple control-nets and even multiple models powered by diffusers. Diffusers has made the gap between a product engineer and a ML engineer super low allowing dashtoon to ship user value faster and better.”
- [Virtual Staging AI](https://www.virtualstagingai.app/): "Filling empty rooms with beautiful furniture using generative models.”
- [Hexo.AI](https://www.hexo.ai/): “Hexo AI helps brands get higher ROI on marketing spends through Personalized Marketing at Scale. Hexo is building a proprietary campaign generation engine which ingests customer data and generates brand compliant personalized creatives.”

If you’re building products on top of 🤗 Diffusers, we’d love to chat to understand how we can make the library better together! Feel free to reach out to patrick@hf.co or sayak@hf.co.

## Looking forward

As we celebrate our first anniversary, we're grateful to our community and open-source contributors who have helped us come so far in such a short time. We're happy to share that we'll be presenting a 🤗 Diffusers demo at ICCV 2023 this fall – if you're attending, do come and see us! We'll continue to develop and improve our library, making it easier for everyone to use. We're also excited to see what the community will create next with our tools and resources. Thank you for being a part of our journey so far, and we look forward to continuing to democratize good machine learning together! 🥳

❤️ Diffusers team

---

**Acknowledgements**: Thank you to [Omar Sanseviero](https://huggingface.co/osanseviero), [Patrick von Platen](https://huggingface.co/patrickvonplaten), [Giada Pistilli](https://huggingface.co/giadap) for their reviews, and [Chunte Lee](https://huggingface.co/Chunte) for designing the thumbnail.
