---
title: "Introducing GenAI-Arena: Image Generation Rankings with Human Preferences"
thumbnail: /blog/assets/arenas-on-the-hub/thumbnail.png
authors:
- user: tianleliphoebe
  guest: true
- user: yuanshengni
  guest: true
- user: DongfuJiang
  guest: true
- user: wenhu
  guest: true
- user: clefourrier
---

# Introducing GenAI-Arena: Image Generation Rankings with Human Preferences

Image generation and manipulation are now used across use cases, from creating stunning artwork to aiding in medical imaging. However, navigating through the multitude of available models and assessing their performance can be a daunting task, as most traditional metrics offer valuable but very specific insights on precise aspects of image generation, and do not assess overall performance.

That’s why we (researchers at the Text and Image GEnerative Research (TIGER) Lab) introduced the GenAI-Arena, a tool to help anyone easily compare image generation models side by side, and provide model rankings back to the communtiy.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.24/gradio.js"> </script>
<gradio-app theme_mode="light" space="TIGER-Lab/GenAI-Arena"></gradio-app>

## Why create the GenAI-Arena?

In the current ecosystem, evaluating image generation models typically involves scouring through research papers to find relevant metrics, running experiments locally, and comparing results manually. However, much like in text (with the [Chatbot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)) and speech (with the [TextToSpeech Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)), using an arena system allows both to streamline this process and make sure than models rankings are aligned with human preferences, instead of just relying on automated metrics only offering a partial view of model capabilities.

We therefore provide a dynamic side-by-side comparison arena, empowering the community to effortlessly generate images, make comparisons, and vote for their preferred model. Our platform is built upon [ImagenHub](https://github.com/TIGER-AI-Lab/ImagenHub), a comprehensive library designed to support inference with various generation and editing models.

## A Holistic Ranking Leaderboard Using Elo

In a spirit of transparency and accountability, we complemented this arena with a leaderboard, using the Elo rating system to rank models.

Widely used in competitive games and sports, the Elo rating system calculates the relative skill levels of players by considering the difference in their ratings. Similarly, in our arena, each model is assigned an Elo rating based on its performance in pairwise battles against other models. While traditional metrics like SSIM, PSNR, LPIPS, FID offer valuable insights into specific aspects of image generation quality, they may not provide a comprehensive assessment of model performance. In contrast, the Elo rating system offers a dynamic and adaptive approach to evaluating models, reflecting their relative performance in direct comparisons. By considering both the strengths and limitations of each approach, researchers and practitioners can gain a more holistic understanding of model capabilities and make informed decisions when selecting models for specific tasks.

## Insights and Results

As the GenAI-Arena community continues to grow, so does the wealth of insights derived from user interactions.

### Image Generation Models

GenAI-Arena focuses on the performance of the open-source cutting-edge diffusion-based generation models.

Leading the board with the highest Arena Elo score of 1097 is Playground v2, sharing the same architecture with SDXL, and developed by the Playground team. Its successor, Playground v2.5, falls slightly behind with an Elo score of 1088, despite offering enhanced color contrast and alignment with human preferences, which may result from the less voting number as it is on-the-shelf for a relatively short time. However, its wide confidence interval indicates it has significant potential for growth and preference within the community.

In the realm of Accelerated Latent Diffusion Models, StableCascade emerges as the frontrunner. It surpasses even some of the non-accelerated counterparts. The superior performance can be attributed to the utilization of Würstchen architecture, which significantly compresses the latent space, demonstrating the advantages of efficiency and compactness.

Notably, PixArtAlpha distinguishes itself as the sole model built on a Transformer backbone, achieving good performance with an Elo score of 1034, despite having the smallest parameter size of 0.6B. This underscores the model's efficiency and the potential of Transformer-based architectures in the field of image generation.

A broader observation from the collected statistics is the apparent correlation between parameter size and model performance. Generally, models with larger parameter sizes tend to perform better, as seen with the top-ranking models, which have parameter sizes of 2.6B or higher. 


| Rank | Model Type                         | Model                                                                                   | Parameter Size | Description                                                                              | Arena Elo | 95% CI | Vote |
| ---- | ---------------------------------- | --------------------------------------------------------------------------------------- |----------------| ---------------------------------------------------------------------------------------- | --------- | ------ | ---- |
| 1    | Latent Diffusion Model             | [Playground v2](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic)     | 2.6B           | same architecture as SDXL trained by Playground team                                     | 1097      | 26/-21 | 645  |
| 2    | Latent Diffusion Model             | [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) | 2.6B           | a successor to Playground v2 with enhanced color contrast and human preference alignment | 1088      | 57/-52 | 148  |
| 3    | Accelerated Latent Diffusion Model | [StableCascade](https://huggingface.co/stabilityai/stable-cascade)                      | 5.1B           | Würstchen architecture at a much smaller latent space                                    | 1056      | 32/-32 | 226  |
| 4    | Latent Diffusion Model             | [PixArtAlpha](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS)                  | 0.6B           | Transformer Latent Diffusion Model                                                       | 1034      | 26/-18 | 702  |
| 5    | Accelerated Latent Diffusion Model | [SDXLLightning](https://huggingface.co/ByteDance/SDXL-Lightning)                        | 2.6B           | SDXL with Progressive and Adversarial Diffusion Distillation                             | 1034      | 35/-37 | 199  |
| 6    | Latent Diffusion Model             | [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)                 | 2.6B           | Base + Refiner latent diffusion model                                                    | 1002      | 23/-18 | 717  |
| 7    | Accelerated Latent Diffusion Model | [SDXLTurbo](https://huggingface.co/stabilityai/sdxl-turbo)                              | 2.6B           | SDXL with Adversarial Diffusion Distillation                                             | 955       | 23/-21 | 736  |
| 8    | Latent Diffusion Model             | [OpenJourney](https://huggingface.co/prompthero/openjourney)                            | 0.8B           | Latent diffusion model fine-tuned on midjourney images                                   | 888       | 24/-25 | 729  |
| 9    | Accelerated Latent Diffusion Model | [LCM](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)                              | 0.8B           | Latent Consistency Model                                                                 | 846       | 22/-24 | 806  |

Overall, these insights from the votes of GenAI-Arena community highlight the dynamic evolution of stable diffusion models and the diverse strategies employed to improve image generation quality and efficiency.

### Image Edition Models

Our GenAI-Arena also support a variety of image editing models that allow users to input a source image along with a target instruction. These models adeptly edit the source image to align with the specified instruction, providing correspondingly edited results that meet user needs.

Among the range of editing models available, attention-guided models such as Prompt2prompt and Plug-and-Play (PNP) stand out for their superior performance. These models harness advanced attention control mechanisms to finely edit images in response to user prompts, achieving high consistency with user expectation.

Instruction-following forward-pass models also excel, offering an impressive balance between image quality and operational efficiency. These models can handle edition without image inversion due to they train with paired annotated data.

On the other hand, latent-space manipulation models currently lag behind in performance when compared to other model types. This category faces challenges in consistently meeting user expectations, highlighting an area needing for further research and development to enhance their effectiveness.


| Rank | Model Type                               | 🤖 Model                                                          | Parameter Size | Description                                                                                        | Arena Elo | 95% CI   | Votes |
| ---- | ---------------------------------------- |-------------------------------------------------------------------|----------------| -------------------------------------------------------------------------------------------------- | --------- | -------- | ----- |
| 1    | Attention-guided Model                   | [Prompt2prompt](https://prompt-to-prompt.github.io/)              | 0.8B           | Prompt-based cross-attention control editing model.                                                | 1188      | 202/-113 | 24    |
| 2    | Attention-guided Model                   | [PNP](https://github.com/MichalGeyer/plug-and-play)               | 0.8B           | Feature and self-attention injection image editing model.                                          | 1134      | 156/-100 | 22    |
| 3    | Instruction-following forward-pass model | [InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix) | 0.8B           | Instruction-based image editing model trained with automatically synthesized data.                 | 1087      | 94/-107  | 29    |
| 4    | Instruction-following forward-pass model | [MagicBrush](https://osu-nlp-group.github.io/MagicBrush)          | 0.8B           | Same structure as InstructPix2Pix but fine-tuned on manually-annotated instruction-guided dataset. | 1085      | 106/-102 | 37    |
| 5    | Attention-guided Model                   | [Pix2PixZero](https://pix2pixzero.github.io/)                     | 0.8B           | Cross-attention guided editing model.                                                              | 984       | 122/-102 | 21    |
| 6    | Latent-space manipulation model          | [CycleDiffusion](https://github.com/ChenWu98/cycle-diffusion)     | 0.8B           | Unpaired image translation model with reconstructable encoder for stochastic DPMs.                 | 848       | 84/-129  | 24    |
| 7    | Latent-space manipulation model          | [SDEdit](https://sde-image-editing.github.io/)                    | 0.8B           | An image synthesis and editing framework with SDEs                                                 | 675       | 157/-227 | 15    |

We hope that continuously analyzing and understanding the performance dynamics of these models will and foster advancements in AI-driven image editing technology and help users make decision about the best models for them.

## How to Get Involved and Join the Arena

Participating in GenAI-Arena is very simple. Visit our [space](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena), explore the available models, and start generating and comparing images. Don’t forget to cast your vote for the model that impresses you the most!

To further engage with us and the community, join discussions on our leaderboard and share your experiences on social media. Whether you’re a seasoned researcher, a budding AI enthusiast, or simply curious about the capabilities of image generation models, we'd love to collaborate and discuss with you!

To witness how your latest model fares against other state-of-the-art competitors, consider contributing it to our [github](https://github.com/TIGER-AI-Lab/ImagenHub) via a PR. This will enable our GenAI-Arena pipeline to incorporate your model, allowing it to be showcased on the leaderboard following community voting.
