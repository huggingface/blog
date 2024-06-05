---
title: "Launching the Artificial Analysis Text to Image Leaderboard & Arena"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_artificialanalysis.png
authors:
- user: mhillsmith
  guest: true
  org: ArtificialAnalysis
- user: georgewritescode
  guest: true
  org: ArtificialAnalysis
---

# Launching the Artificial Analysis Text to Image Leaderboard & Arena

In two short years since the advent of diffusion-based image generators, AI image models have achieved near-photographic quality. How do these models compare? Are the open-source alternatives on par with their proprietary counterparts? 

The Artificial Analysis Text to Image Leaderboard aims to answer these questions with human preference based rankings. The ELO score is informed by over 45,000 human image preferences collected in the Artificial Analysis Image Arena. The leaderboard features the leading open-source and proprietary image models : the latest versions of Midjourney, OpenAI's DALL·E, Stable Diffusion, Playground and more.

![Untitled](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-on-the-hub/artificial_analysis_vision_leaderboard.png)

Check-out the leaderboard here: [https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard](https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard)

You can also take part in the Text to Image Arena, and get your personalized model ranking after 30 votes!

## Methodology

Comparing the quality of image models has traditionally been even more challenging than evaluations in other AI modalities such as language models, in large part due to the inherent variability in people’s preferences for how images should look. Early objective metrics have given way to expensive human preference studies as image models approach very high accuracy. Our Image Arena represents a crowdsourcing approach to gathering human preference data at scale, enabling comparison between key models for the first time. 

We calculate an ELO score for each model via a regression of all preferences, similar to Chatbot Arena. Participants are presented with a prompt and two images, and are asked select the image that best reflects the prompt. To ensure the evaluation reflects a wide-range of use-cases we generate >700 images for each model. Prompts span diverse styles and categories including human portraits, groups of people, animals, nature, art and more. 

## Early Insights From the Results 👀

- **While proprietary models lead, open source is increasingly competitive**: Proprietary models including Midjourney, Stable Diffusion 3 and DALL·E 3 HD lead the leaderboard. However, a number of open-source models, currently led by Playground AI v2.5, are gaining ground and surpass even OpenAI’s DALL·E 3.
- **The space is rapidly advancing:** The landscape of image generation models is rapidly evolving. Just last year, DALL·E 2 was a clear leader in the space. Now, DALL·E 2 is selected in the arena less than 25% of the time and is amongst the lowest ranked models.
- **Stable Diffusion 3 Medium being open sourced may have a big impact on the community**: Stable Diffusion 3 is a contender to the top position on the current leaderboard and Stability AI’s CTO recently announced during a presentation with AMD that Stable Diffusion 3 Medium will be open sourced June 12. Stable Diffusion 3 Medium may offer lower quality performance compared to the Stable Diffusion 3 model served by Stability AI currently (presumably the full-size variant), but the new model may be a major boost to the open source community. As we have seen with Stable Diffusion 1.5 and SDXL, it is likely we will see many fine tuned versions released by the community.

## How to contribute or get in touch

To see the leaderboard, check out the space on Hugging Face here: [https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard](https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard)

To participate in the ranking and contribute your preferences, select the ‘Image Arena’ tab and choose the image which you believe best represents the prompt. After 30 images, select the ‘Personal Leaderboard’ tab to see your own personalized ranking of image models based on your selections. 

For updates, please follow us on [**Twitter**](https://twitter.com/ArtificialAnlys) and [**LinkedIn**](https://linkedin.com/company/artificial-analysis). (We also compare the speed and pricing of Text to Image model API endpoints on our website at [https://artificialanalysis.ai/text-to-image](https://artificialanalysis.ai/text-to-image)). 

We welcome all feedback! We're available via message on Twitter, as well as on [**our website](https://artificialanalysis.ai/contact)** via our contact form.

## Other Image Model Quality Initiatives

The Artificial Analysis Text to Image leaderboard is not the only quality image ranking or crowdsourced preference initiative. We built our leaderboard to focus on covering both proprietary and open source models to give a full picture of how leading Text to Image models compare.

Check out the following for other great initiatives:

- [Open Parti Prompts Leaderboard](https://huggingface.co/spaces/OpenGenAI/parti-prompts-leaderboard)
- [imgsys Arena](https://huggingface.co/spaces/fal-ai/imgsys)
- [GenAI-Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena)
- [Vision Arena](https://huggingface.co/spaces/WildVision/vision-arena)
