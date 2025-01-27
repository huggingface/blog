---
title: "The AI tools for Art Newsletter - Issue 1"
thumbnail: /blog/assets/ai_art_newsletter_1/thumbnail.png
authors:
- user: linoyts
- user: multimodalart
---

# The AI tools for Art Newsletter

### First issue 🎉

The AI space is moving so fast it’s hard to believe that a year ago we still struggled to generate people with the correct amount of fingers.   
The last couple of years have been pivotal for open source models and tools for artistic usage. 
AI tools for creative expression have never been more accessible, and we’re only scratching the surface. 
Join us as we look back at the key milestones, tools, and breakthroughs in AI & Arts from 2024, 
and forward for what’s to come in 2025 (spoiler 👀: we’re starting a new weekly roundup).


## Major Releases of 2024 

What were the standout releases of creative AI tools in 2024? We'll highlight the major releases across creative and 
artistic fields, with a particular focus on open-source developments in popular tasks like image and video generation. 

**[timeline figure]**

## Image Generation 

Over 2 years since the OG stable diffusion was released and made waves in image generation with open source models, it’s now safe to say that when it comes to image generation from text, image editing and controlled image generation - open source models are giving closed source models a run for their money.   


### Text-to-image generation

2024 was the year we shifted paradigms of diffusion models - from the traditional Unet based architecture to Diffusion Transformer (DiT), as well as an objective switch to flow matching. 

**TD;LR** - diffusion models and **Gaussian** flow matching are equivalent.Flow matching proposes a vector field parametrization of the network output that is different compared to the ones commonly used in diffusion models previously.

* We recommend this [great blog ](https://diffusionflow.github.io)if you’re interested in learning more about flow matching and the connection with diffusion models

 
**Back to practice**: First to announce the shift was Stability AI with Stable Diffusion 3, however it was HunyuanDiT that became the first open source model with DiT architecture.  \
This trend continued with the releases of AuraFlow, Flux.1 and Stable Diffusion 3.5. 

 
Among many pivotal moments in the (not so long) history of open source image generation models, it’s safe to say that the release of Flux.1 was one of them. Flux [dev] achieved a new state-of-the-art, surpassing popular closed source models like Midjourney v6.0, DALL·E 3 (HD) on various benchmarks. 


### Personalization & stylization 

A positive side effect of advancements in image models is the significant improvement in personalization techniques for text-to-image models and controlled generation.

Back in August 2022, transformative works like Textual Inversion and DreamBooth enhanced our ability to teach and introduce new concepts to text-to-image models, drastically expanding what could be done with them.These opened the door to a stream of improvements and enhancements building on top of these techniques (such as LoRA for diffusion models).

However, an upper bound to the quality of the fine-tuned models is naturally the base model from which it was finetuned. In that sense, we can’t neglect Stable Diffusion XL, which was also a significant marker in personalization for open source image generation models. A testimony to that is that even now, many of the popular techniques and models for personalization and controlled generation are based on SDXL.  The advanced abilities of SDXL (and models that were released after with similar quality) together with the growing understanding of the semantic roles of different components in the diffusion model architecture raises the question -  \
what can we achieve without further optimization?

*cue in the rain of zero shot techniques* - 2024 was definitely the year when generating high quality portraits from reference photos was made possible with as little as a single reference image & without any optimization. Training free techniques like IP adapter FaceID, InstantID, photomaker and more came out and demonstrated competitive if not even superior abilities to those of fine tuned models. 

Similarly, image editing and controlled generation - such as image generation with canny / depth / pose constraints made progress too - both thanks to the growing quality of the base models and the community’s growing understanding of the semantic roles different components have ([Instant Style](https://huggingface.co/spaces/InstantX/InstantStyle), [B-LoRA](https://huggingface.co/spaces/Yardenfren/B-LoRA))

**So what’s next?** since the shift of paradigms to DiT and flow matching objectives, 
Similar models came out trying to utilize DiT based models like Flux and SD3.5 for similar purposes, but so far not quite beating the quality of the SDXL based ones despite the superior quality of the underlying base model. This could be attributed to the relative lack of understanding of semantic roles of different components of the DiT for example, compared to the Unet. 2025 could be the year we identify those roles in DiTs as well, unlocking more possibilities with the next generation of image generation models. 


## Video Generation

As opposed to image generation, with video we still have a way to go. But, it’s safe to say that we’re very far away from where we were a year ago. While we’re all about open-source, the credit for (some) of the significant leap in AI video generation goes to OpenAI’s sora for changing our expectations of video model capabilities quite radically. And as fofr put nicely in *[AI video is having its Stable Diffusion moment](https://replicate.com/blog/ai-video-is-having-its-stable-diffusion-moment)* (which we recommend reading 🙂) - it  \
made everyone realize what is possible. 


The recent surge of open-source video generation models, including CogVideoX, Mochi, Allegro, LTX Video, and HunyuanVideo, has also been noteworthy. Video generation is inherently more challenging than image generation due to the need for motion quality, coherence, and consistency. Additionally, video generation requires substantial computational and memory resources, leading to significant generation latency. This often hinders local usage, making many new open video models inaccessible to community hardware without extensive memory optimizations and quantization approaches that impact both inference latency and the quality of generated videos. Nevertheless the open source community has made remarkable progress - which was recently covered in this blog on [the state of open video generation models](https://huggingface.co/blog/video_gen).

While this implies that most community members are still unable to experiment and develop with open-source video models, it also suggests that we can expect significant advancements in 2025.


## Creative Tools that Shined in 2024

The beauty of open source is that it allows the community to experiment, find new usages for existing models / pipelines, improve on and build new tools together. Many of the creative AI tools that were popular this year are the fruit of joint community effort.  



#### Flux finetuning

Many of the amazing [Flux fine-tunes](https://huggingface.co/spaces/multimodalart/flux-lora-the-explorer) created in the last year were trained thanks to the [AI-toolkit ](https://github.com/ostris/ai-toolkit)by [ostris](https://huggingface.co/ostris).

#### Face to all 
Inspired by fofr's [face-to-many](https://github.com/fofr/cog-face-to-many),[Face to All](https://huggingface.co/spaces/multimodalart/face-to-all) combines the viral Instant ID model with added Controlnet depth constraints and community fine-tuned SDXL LoRAs to create training free high quality portraits in creative stylizations.  


#### Flux style shaping 

Based on a ComfyUI workflow by [Nathan Shipley](https://x.com/CitizenPlain), [Flux style shaping](https://huggingface.co/spaces/multimodalart/flux-style-shaping) combines Flux [dev] Redux and Flux [dev] Depth for style transfer and optical illusion creation.


#### Outpainting with diffusers

[Diffusers Image Outpaint](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint) makes use of the diffusers StableDiffusionXL Fill Pipeline together with an SDXL union controlnet to seamlessly expand an input image.  


#### Live portrait, Face Poke

Adding mimics to a static portrait was never easier with [Live Portrait](https://huggingface.co/spaces/KwaiVGI/LivePortrait) and [Face Poke](https://huggingface.co/spaces/jbilcke-hf/FacePoke).


#### TRELLIS

[TRELLIS](https://huggingface.co/spaces/JeffreyXiang/TRELLIS) is a 3D generation model for versatile and high-quality 3D asset creation that took over the 3D landscape with a bang. 


#### IC Light 

[IC-Light](https://huggingface.co/spaces/lllyasviel/IC-Light), which stands for "Imposing Consistent Light", is a tool for relighting with foreground condition.


## What should we expect for AI & Art in 2025?

2025 is the year for open-source to catch up on video, movement, and audio models, making room for more modalities. With advancements in efficient computing and quantization, we can expect significant leaps in open-source video models. As we approach a (natural) plateau in image generation models, we can shift our focus to other tasks and modalities.


## Starting off strong - Open source releases of January 25
[wip]


1. Hunyuan 3D , SD 3D

    [https://huggingface.co/tencent/Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2), [https://huggingface.co/stabilityai/stable-point-aware-3d](https://huggingface.co/stabilityai/stable-point-aware-3d)

2. Moondreamv2

    [https://huggingface.co/vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2)

3. comfy-to-gradio tool

    [https://huggingface.co/blog/run-comfyui-workflows-on-spaces](https://huggingface.co/blog/run-comfyui-workflows-on-spaces)



## Announcing Our Weekly Newsletter 🗞️

Kicking off with this blog, we (Poli & Linoy) will be bringing you a weekly roundup of the latest in the creative 
AI world. In such a fast-evolving space, it’s tough to stay on top of all the new developments, 
let alone sift through them. That’s where we come in & hopefully this way we can make creative AI tools more accessible 
