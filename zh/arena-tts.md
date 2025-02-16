---
title: "AI艺术工具通讯 - 第1期"
thumbnail: /blog/assets/ai_art_newsletter_1/thumbnail.png
authors:
- user: linoyts
- user: multimodalart
translators:
- user: yaoqih
- user: zhongdongy
  proofreader: true
---

# AI 艺术工具通讯

### 创刊号 🎉

AI 领域的发展速度令人惊叹，回想一年前我们还在为生成正确手指数量的人像而苦苦挣扎的场景，恍如隔世 😂。

过去两年对开源模型和艺术创作工具而言具有里程碑意义。创意表达的 AI 工具从未像现在这般触手可及，然而这仅仅是冰山一角。让我们共同回顾 2024 年 AI 艺术领域的关键突破与创新工具，并展望 2025 年的发展趋势 (剧透预警 👀: 我们将启动月度资讯精选的订阅👇)。

<iframe src="https://multimodalaiart.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe>

## 目录

- [2024 重大发布](#2024-重大发布)
- [图像生成](#图像生成)
  - [文生图](#文生图)
  - [个性化与风格化](#个性化与风格化)
- [视频生成](#视频生成)
- [2024 闪耀创意工具](#2024-闪耀创意工具)
- [2025 年 AI 艺术趋势展望](#2025-年-AI-艺术趋势展望)
- [强势开局: 2025 年 1 月开源新作](#强势开局-2025-年-1-月开源新作)

## 2024 重大发布

2024 年哪些创意 AI 工具最引人注目？我们将重点盘点艺术创作领域的重要发布，特别关注文生图、视频生成等热门任务中的开源进展。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/timeline_2.png" width="700" height="auto" alt="2024 年重要时刻时间轴 ">

## 图像生成

自初代 Stable Diffusion 掀起开源文生图浪潮已逾两年，如今在文本到图像生成、图像编辑和可控生成领域，开源模型已能与闭源产品分庭抗礼。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/finger_meme.png" width="424" height="auto" alt=" 手指生成梗图 ">

### 文生图

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/flux_grid.png" width="600" height="auto" alt="Flux 模型效果展示 ">

2024 年见证了扩散模型的范式转变——从传统 U-Net 架构转向扩散 Transformer (DiT)，同时目标函数也进化为流匹配 (flow matching)。

**技术速览**: 扩散模型与 **高斯** 流匹配本质相通。流匹配通过不同的向量场参数化方式，为网络输出提供了新视角。

- 推荐阅读 [Google DeepMind 的技术博客](https://diffusionflow.github.io)，深入了解流匹配与扩散模型的关联。

**实践进展**: Stability AI 率先推出 [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium)，而 [腾讯混元 DiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) 则成为首个开源的 DiT 架构模型。后续 [AuraFlow](https://huggingface.co/fal/AuraFlow)、[Flux.1](https://huggingface.co/black-forest-labs/FLUX.1-dev) 和 [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) 延续了这一趋势。

在开源图像生成模型的里程碑中，[Flux.1](https://huggingface.co/black-forest-labs/FLUX.1-dev) 的发布堪称革命性。该模型在多项基准测试中超越 Midjourney v6.0、DALL·E 3 (HD) 等闭源模型，刷新了开源模型的性能纪录。

### 个性化与风格化

图像模型的进步带动了个性化生成技术的飞跃。2022 年 8 月，[Textual Inversion](https://textual-inversion.github.io) 和 [DreamBooth](https://dreambooth.github.io) 等开创性工作实现了 **向文生图模型注入概念**，极大扩展了应用边界。这些技术催生了 LoRA 等改进方案，推动个性化生成进入新阶段。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/personalization_1.png" width="424" height="auto" alt=" 个性化技术对比 ">

然而，微调模型的质量受限于基础模型性能。Stable Diffusion XL (SDXL) 的发布为开源个性化生成树立新标杆，当前多数个性化方案仍基于 SDXL 架构。随着对扩散模型各组件语义角色的深入理解，我们不禁思考: **能否实现不进行额外繁琐优化的高质量生成？**

_Zero-shot 技术风暴来袭_ ——2024 年见证了仅需 **单张参考图** 即可生成高质量人像的技术突破。[IP-Adapter FaceID](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID)、[InstantID](https://huggingface.co/spaces/InstantX/InstantID)、[PhotoMaker](https://huggingface.co/spaces/TencentARC/PhotoMaker-V2) 等免训练方案展现出媲美微调模型的实力。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/instantid.png" width="600" height="auto" alt="InstantID 效果展示 ">

图像编辑与可控生成 (如边缘/深度/姿态控制) 也取得长足进步，这既得益于基础模型的发展，也源于社区对模型组件的深入理解 ([Instant Style](https://huggingface.co/spaces/InstantX/InstantStyle)、[B-LoRA](https://huggingface.co/spaces/Yardenfren/B-LoRA))。

**未来展望**: 尽管 DiT 架构模型 (如 Flux、SD3.5) 已开始探索个性化的应用，但对 DiT 组件语义角色的理解尚不及 U-Net 深入。2025 年或将揭开 DiT 的组件奥秘，释放新一代图像模型的全部潜能。

## 视频生成

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt=" 混元视频演示 "
       autoplay loop autobuffer muted playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/video_gen/hunyuan-output.mp4" type="video/mp4">
   </video>
 </figure>

相较图像生成，视频生成仍任重道远。但必须承认，我们已取得显著进步。OpenAI 的 Sora 极大提升了行业预期，正如 fofr 在 [《AI 视频正迎来 Stable Diffusion 时刻》](https://replicate.com/blog/ai-video-is-having-its-stable-diffusion-moment) 中所言——它让人们看到了可能性。

近期开源视频模型的爆发 ([CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b)、[Mochi](https://huggingface.co/genmo/mochi-1-preview)、[Allegro](https://huggingface.co/rhymes-ai/Allegro)、[LTX Video](https://huggingface.co/Lightricks/LTX-Video)、[混元视频](https://huggingface.co/tencent/HunyuanVideo)) 同样值得关注。视频生成面临画面动作是否自然、前后画面是否流畅、人物外观是否保持一致等多重挑战，加之计算资源需求巨大，导致生成延迟较高。尽管内存优化和量化技术可缓解硬件压力，但往往会影响生成的质量。尽管如此，开源社区仍在持续突破，最新进展可参阅 [开源视频生成模型现状](https://huggingface.co/blog/video_gen)。

虽然多数用户仍难以本地运行视频模型，但这也预示着 2025 年将迎来更大突破。

## 音频生成

音频生成在过去一年突飞猛进，从制作简单的声音效果到创作完整的歌曲都取得了很大进步。尽管面临信号复杂度高、训练数据稀缺等挑战，2024 年仍涌现 [OuteTTS](https://huggingface.co/OuteAI/OuteTTS-0.2-500M)、[IndicParlerTTS](https://huggingface.co/ai4bharat/indic-parler-tts) 等开源语音合成模型，以及 OpenAI 的 [Whisper large v3 turbo](https://huggingface.co/openai/whisper-large-v3-turbo) 语音识别模型。2025 年开年即迎来 [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)、[LLasa TTS](https://huggingface.co/HKUSTAudio/Llasa-3B)、[OuteTTS 0.3](https://huggingface.co/OuteAI/OuteTTS-0.3-1B) 等语音模型，以及 [JASCO](https://huggingface.co/models?search=jasco)、[YuE](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot) 音乐模型的集中发布，预示着音频领域将迎来爆发年。

下方歌曲由 YuE 生成🤯

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <audio
       alt="yue.mp3"
       controls
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/I_wont_back_down_pop.mp3" type="audio/mp3">
   </audio>
 </figure>

## 2024 闪耀创意工具

开源之美在于集社区之力探索模型新可能。本年度众多创意工具正是这种协作精神的结晶:

#### Flux fine-tuning

[ostris](https://huggingface.co/ostris) 开发的 [AI 工具包](https://github.com/ostris/ai-toolkit) 助力社区创作出惊艳的 [Flux 微调模型](https://huggingface.co/spaces/multimodalart/flux-lora-the-explorer)。

#### Face to All

受 [face-to-many](https://github.com/fofr/cog-face-to-many) 启发，[Face to All](https://huggingface.co/spaces/multimodalart/face-to-all) 将爆款模型 [Instant ID](https://huggingface.co/spaces/InstantX/InstantID) 与深度 ControlNet、社区微调的 SDXL LoRA 结合，实现免训练的高质量风格化人像生成。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/face-to-all.png" width="512" height="auto" alt="Face to All 效果展示 ">

#### Flux 风格塑形

基于 [Nathan Shipley](https://x.com/CitizenPlain) 的 ComfyUI 工作流，[Flux 风格塑形](https://huggingface.co/spaces/multimodalart/flux-style-shaping) 通过融合 Flux [dev] Redux 与 Depth 模型，实现风格迁移与视错觉创作。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/styleshaping.jpeg" width="512" height="auto" alt=" 风格塑形效果 ">

#### 智能图像外扩

[Diffusers Image Outpaint](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint) 利用 SDXL Fill Pipeline 与联合 ControlNet，实现无缝图像外扩。

#### 动态人像

[Live Portrait](https://huggingface.co/spaces/KwaiVGI/LivePortrait) 与 [Face Poke](https://huggingface.co/spaces/jbilcke-hf/FacePoke) 让静态人像瞬间动起来。

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt=" 面部动画演示 "
       autoplay loop autobuffer muted playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/isaac_1.mp4" type="video/mp4">
   </video>
 </figure>

#### TRELLIS 3D 引擎

[TRELLIS](https://huggingface.co/spaces/JeffreyXiang/TRELLIS) 以惊艳效果重塑 3D 生成格局，支持多样化高质量资产创建。

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt="TRELLIS 演示 "
       autoplay loop autobuffer muted playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/trellis.mp4" type="video/mp4">
   </video>
 </figure>

#### IC Light

[IC-Light](https://huggingface.co/spaces/lllyasviel/IC-Light) 通过前景条件实现智能光影重构。

## 2025 年 AI 艺术趋势展望

2025 年将是开源社区在视频、动态与音频模型领域迎头赶上的一年。随着高效计算与量化技术的突破，开源视频模型有望实现跨越式发展。当图像生成进入自然平台期，我们的目光将转向多模态创新。

## 强势开局: 2025 年 1 月开源新作

1. **YuE 音乐生成模型**

Apache 2.0 协议开源的 [YuE](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot) 在音乐生成质量上比肩 Suno 等闭源产品，[在线体验](https://huggingface.co/spaces/fffiloni/YuE)。

<figure class="image flex flex-col items-center text-center m-0 w-full">
    <video
       alt="YuE 歌曲演示 "
       autobuffer playsinline
     >
     <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai_art_newsletter_1/My first YuE (open source Suno) AI generated full song.mp4" type="video/mp4">
   </video>
 </figure>

2. **3D 生成三剑客**

继 TRELLIS 之后，[混元 3D-2](https://huggingface.co/tencent/Hunyuan3D-2)、[SPAR3D](https://huggingface.co/stabilityai/stable-point-aware-3d)、[DiffSplat](https://huggingface.co/chenguolin/DiffSplat) 持续革新 3D 生成领域。

3. **Lumina-Image 2.0**

这款 20 亿参数的 [文生图模型](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) 以 Apache 2.0 协议开源，性能比肩 80 亿参数的 Flux.1，[在线体验](https://huggingface.co/spaces/benjamin-paine/Lumina-Image-2.0)。

4. **ComfyUI 转 Gradio 指南**

这份 [教程](https://huggingface.co/blog/run-comfyui-workflows-on-spaces) 详细介绍了如何将复杂 ComfyUI 工作流转换为 Gradio 应用，并免费部署于 Hugging Face Spaces。

## 开启资讯新时代 🗞️

从本期开始，我们 ([Poli](https://huggingface.co/multimodalart) 与 [Linoy](https://huggingface.co/linoyts)) 将每月为您精选创意 AI 领域最新动态。在这个快速迭代的领域，我们愿做您的信息顾问，让创意工具触手可及。

<iframe src="https://multimodalaiart.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe>