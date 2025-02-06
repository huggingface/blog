---
title: "The AI tools for Art Newsletter - Issue 1"
thumbnail: /blog/assets/ai_art_newsletter_1/thumbnail.png
authors:
- user: linoyts
- user: multimodalart
---

# The AI tools for Art Newsletter

### First issue 🎉

The AI space is moving so fast it’s hard to believe that a year ago we still struggled to generate people with the correct amount of fingers 😂.

The last couple of years have been pivotal for open source models and tools for artistic usage. 
AI tools for creative expression have never been more accessible, and we’re only scratching the surface. 
Join us as we look back at the key milestones, tools, and breakthroughs in AI & Arts from 2024, 
and forward for what’s to come in 2025 (spoiler 👀: we’re starting a new monthly roundup 👇).

 <iframe src="https://multimodalaiart.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe>

 ## Table of Contents
 - [Major Releases of 2024](#Major-Releases-of-2024)
 - [Image Generation](#Image-Generation)
   * [Text-to-image generation](#Text-to-image-generation)
   * [Personalization & stylization ](#Personalization-&-stylization)
 - [Video Generation](#Video-Generation)
 - [Creative Tools that Shined in 2024](#Creative-Tools-that-Shined-in-2024)
 - [What should we expect for AI & Art in 2025?](#What-should-we-expect-for-AI-&-Art-in-2025?)
 - [Starting off strong - Open source releases of January 25](#Starting-off-strong---Open-source-releases-of-January-25)


## Major Releases of 2024

What were the standout releases of creative AI tools in 2024? We'll highlight the major releases across creative and 
artistic fields, with a particular focus on open-source developments in popular tasks like image and video generation. 

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/timeline_2.png" width="700" height="auto" alt="2024 highlights">

## Image Generation

Over 2 years since the OG stable diffusion was released and made waves in image generation with open source models, it’s now safe to say that when it comes to image generation from text, image editing and controlled image generation - open source models are giving closed source models a run for their money.   
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/finger_meme.png" width="424" height="auto" alt="2024 highlights">

### Text-to-image generation
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/flux_grid.png" width="600" height="auto" alt="flux">
2024 was the year we shifted paradigms of diffusion models - from the traditional Unet based architecture to Diffusion Transformer (DiT), as well as an objective switch to flow matching. 

**TD;LR** - diffusion models and **Gaussian** flow matching are equivalent. Flow matching proposes a vector field parametrization of the network output that is different compared to the ones commonly used in diffusion models previously.

* We recommend this [great blog by Google DeepMind](https://diffusionflow.github.io) if you’re interested in learning more about flow matching and the connection with diffusion models

 
**Back to practice**: First to announce the shift was Stability AI with [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), however it was [HunyuanDiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) that became the first open source model with DiT architecture.  
This trend continued with the releases of [AuraFlow](https://huggingface.co/fal/AuraFlow), [Flux.1](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large). 

 
Among many pivotal moments in the (not so long) history of open source image generation models, it’s safe to say that the release of Flux.1 was one of them. [Flux [dev]](https://huggingface.co/black-forest-labs/FLUX.1-dev) achieved a new state-of-the-art, surpassing popular closed source models like Midjourney v6.0, DALL·E 3 (HD) on various benchmarks. 


### Personalization & stylization

A positive side effect of advancements in image models is the significant improvement in personalization techniques for text-to-image models and controlled generation.

Back in August 2022, transformative works like [Textual Inversion](https://textual-inversion.github.io) and [DreamBooth](https://dreambooth.github.io) enhanced our ability to **teach and introduce new concepts to text-to-image models**, drastically expanding what could be done with them. These opened the door to a stream of improvements and enhancements building on top of these techniques (such as LoRA for diffusion models).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/personalization_1.png" width="424" height="auto" alt="textual inversion - dreambooth">

However, an **upper bound to the quality of the fine-tuned models is naturally the base model** from which it was fine-tuned. In that sense, we can’t neglect Stable Diffusion XL, which was also a significant marker in personalization for open source image generation models. A testimony to that is that even now, many of the popular techniques and models for personalization and controlled generation are based on SDXL.  The advanced abilities of SDXL (and models that were released after with similar quality) together with the growing understanding of the semantic roles of different components in the diffusion model architecture raises the question -  \
what can we achieve without further optimization?

*cue in the rain of zero shot techniques* - 2024 was definitely the year when generating high quality portraits from 
reference photos was made possible with as little as **a single reference image & without any optimization**. Training free 
techniques like [IP adapter FaceID](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID), [InstantID](https://huggingface.co/spaces/InstantX/InstantID), [Photomaker](https://huggingface.co/spaces/TencentARC/PhotoMaker-V2) and more came out and demonstrated competitive if 
not even superior abilities to those of fine-tuned models. 

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/instantid.png" width="600" height="auto" alt="instantid">

Similarly, image editing and controlled generation - such as image generation with canny / depth / pose constraints made progress too - both thanks to the growing quality of the base models and the community’s growing understanding of the semantic roles different components have ([Instant Style](https://huggingface.co/spaces/InstantX/InstantStyle), [B-LoRA](https://huggingface.co/spaces/Yardenfren/B-LoRA))

**So what’s next?** since the shift of paradigms to DiT and flow matching objectives, 
additional models came out trying to utilize DiT-based models like Flux and SD3.5 for similar purposes, but so far not quite beating the quality of the SDXL-based ones despite the superior quality of the underlying base model. This could be attributed to the relative lack of understanding of semantic roles of different components of the DiT compared to the Unet. 2025 could be the year when we identify those roles in DiTs as well, unlocking more possibilities with the next generation of image generation models. 


## Video Generation

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt="demo4.mp4"
       autoplay loop autobuffer muted playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/video_gen/hunyuan-output.mp4" type="video/mp4">
   </video>
 </figure>

As opposed to image generation, with video we still have a way to go. 
But, it’s safe to say that we’re very far away from where we were a year ago. While we’re all about open-source, 
the credit for (some) of the significant leap in AI video generation goes to OpenAI’s sora for changing our 
expectations of video model capabilities quite radically. And as fofr put nicely in *[AI video is having its Stable Diffusion moment](https://replicate.com/blog/ai-video-is-having-its-stable-diffusion-moment)* (which we recommend reading 🙂) - it  \
made everyone realize what is possible.

The recent surge of open-source video generation models, including [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b), [Mochi](https://huggingface.co/genmo/mochi-1-preview), [Allegro](https://huggingface.co/rhymes-ai/Allegro), [LTX Video](https://huggingface.co/Lightricks/LTX-Video), 
and [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo), has also been noteworthy. Video generation is inherently more challenging than image generation due to the need for motion quality, coherence, and consistency. Additionally, video generation requires substantial computational and memory resources, leading to significant generation latency. This often hinders local usage, making many new open video models inaccessible to community hardware without extensive memory optimizations and quantization approaches that impact both inference latency and the quality of generated videos. Nevertheless the open source community has made remarkable progress - which was recently covered in this blog on [the state of open video generation models](https://huggingface.co/blog/video_gen).

While this implies that most community members are still unable to experiment and develop with open-source video models, it also suggests that we can expect significant advancements in 2025.

## Audio Generation

Audio generation has progressed significantly in the past year going from simple sounds to complete songs with lyrics. 
Despite challenges - Audio signals are complex and multifaceted, require more sophisticated mathematical models than 
models that generate text or images and training data quite scarce - 2024 saw open source releases like [OuteTTS](https://huggingface.co/OuteAI/OuteTTS-0.2-500M) and 
[IndicParlerTTS](https://huggingface.co/ai4bharat/indic-parler-tts) for text to speech and openai’s [Whisper large v3 turbo](https://huggingface.co/openai/whisper-large-v3-turbo) for audio speech recognition. 
The year 2025 is already shaping up to be a breakthrough year for audio models, with a remarkable number of releases 
in January alone.  We've seen the release of three new text-to-speech models:  [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), [LLasa TTS](https://huggingface.co/HKUSTAudio/Llasa-3B) and [OuteTTS 0.3](https://huggingface.co/OuteAI/OuteTTS-0.3-1B), 
as well as two new music models: [JASCO](https://huggingface.co/models?search=jasco) and [YuE](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot). With this pace, we can expect even more exciting developments in 
the audio space throughout the year.

This song👇 was generated with YuE 🤯
<figure class="image flex flex-col items-center text-center m-0 w-full">
    <audio
       alt="yue.mp3"
       controls
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/I_wont_back_down_pop.mp3" type="audio/mp3">
   </audio>
 </figure>


## Creative Tools that Shined in 2024

The beauty of open source is that it allows the community to experiment, find new usages for existing models / pipelines, improve on and build new tools together. Many of the creative AI tools that were popular this year are the fruit of joint community effort.

Here are some of our favorites:

#### Flux fine-tuning

Many of the amazing [Flux fine-tunes](https://huggingface.co/spaces/multimodalart/flux-lora-the-explorer) created in the last year were trained thanks to the [AI-toolkit](https://github.com/ostris/ai-toolkit) by [ostris](https://huggingface.co/ostris).

#### Face to all 
Inspired by fofr's [face-to-many](https://github.com/fofr/cog-face-to-many), [Face to All](https://huggingface.co/spaces/multimodalart/face-to-all) combines the viral [Instant ID model](https://huggingface.co/spaces/InstantX/InstantID) with added ControlNet depth constraints and community fine-tuned SDXL LoRAs to create training-free and high-quality portraits in creative stylizations.  

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/face-to-all.png" width="512" height="auto" alt="face to all">

#### Flux style shaping 

Based on a ComfyUI workflow by [Nathan Shipley](https://x.com/CitizenPlain), [Flux style shaping](https://huggingface.co/spaces/multimodalart/flux-style-shaping) combines Flux [dev] Redux and Flux [dev] Depth for style transfer and optical illusion creation.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/styleshaping.jpeg" width="512" height="auto" alt="style shaping">

#### Outpainting with diffusers

[Diffusers Image Outpaint](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint) makes use of the diffusers Stable Diffusion XL Fill Pipeline together with an SDXL union controlnet to seamlessly expand an input image.  


#### Live portrait, Face Poke

Adding mimics to a static portrait was never easier with [Live Portrait](https://huggingface.co/spaces/KwaiVGI/LivePortrait) and [Face Poke](https://huggingface.co/spaces/jbilcke-hf/FacePoke).
<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt="face_poke.mp4"
       autoplay loop autobuffer muted playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/isaac_1.mp4" type="video/mp4">
   </video>
 </figure>


#### TRELLIS

[TRELLIS](https://huggingface.co/spaces/JeffreyXiang/TRELLIS) is a 3D generation model for versatile and high-quality 3D asset creation that took over the 3D landscape with a bang. 

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt="trellis.mp4"
       autoplay loop autobuffer muted playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/trellis.mp4" type="video/mp4">
   </video>
 </figure>

#### IC Light 

[IC-Light](https://huggingface.co/spaces/lllyasviel/IC-Light), which stands for "Imposing Consistent Light", is a tool for relighting with foreground condition.


## What should we expect for AI & Art in 2025?

2025 is the year for open-source to catch up on video, movement, and audio models, making room for more modalities. With advancements in efficient computing and quantization, we can expect significant leaps in open-source video models. As we approach a (natural) plateau in image generation models, we can shift our focus to other tasks and modalities.


## Starting off strong - Open source releases of January 25


1. **YuE - series of open-source music foundation models** for full song generation.
YuE is possibly the best open source model for music generation (with an Apache 2.0 license!), achieving competitive results to closed source models like Suno.

   **try it out & read more**: [demo](https://huggingface.co/spaces/fffiloni/YuE), [model weights](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot).

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt="yue.mp4"
       autobuffer playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/My first YuE (open source Suno) AI generated full song.mp4" type="video/mp4">
   </video>
 </figure>


2. **Hunyuan 3D-2 , SPAR3D, DiffSplat - 3D generation models**. 
3D models are coming in hot - not long after the release of TRELLIS, Hunyuan 3D-2, SPAR3D and DiffSplat are here to 
take over the 3D landscape. 

   **try it out & read more:**
   - [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) 
   - [SPAR3D](https://huggingface.co/stabilityai/stable-point-aware-3d)
   - [DiffSplat](https://huggingface.co/chenguolin/DiffSplat)

3. **Lumina-Image 2.0** - text to image model. 
Lumina is a 2B parameter model competitive with the 12B Flux.1 [dev] and with an Apache 2.0 license(!!).

   **try it out & read more**: [demo](https://huggingface.co/spaces/benjamin-paine/Lumina-Image-2.0), [model weights](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0).

4. **ComfyUI-to-Gradio** - 
a step-by-step guide on how to convert a complex ComfyUI workflow to a simple Gradio application, and how to deploy this application on Hugging Face Spaces ZeroGPU serverless structure, which allows for it to be deployed and run for free in a serverless manner
**read more** [**here**](https://huggingface.co/blog/run-comfyui-workflows-on-spaces).


## Announcing Our Newsletter 🗞️

Kicking off with this blog, we ([Poli](https://huggingface.co/multimodalart) & [Linoy](https://huggingface.co/linoyts)) will be bringing you a monthly roundup of the latest in the creative 
AI world. In such a fast-evolving space, it’s tough to stay on top of all the new developments, 
let alone sift through them. That’s where we come in & hopefully this way we can make creative AI tools more accessible 

<iframe src="https://multimodalaiart.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe>
