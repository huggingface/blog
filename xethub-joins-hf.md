---
title: "XetHub is joining Hugging Face!"
thumbnail: /blog/assets/xethub-joins-hf/thumbnail.png
authors:
- user: yuchenglow
  org: xet-team
- user: julien-c
---

# XetHub is joining Hugging Face!

We are super excited to officially announce that Hugging Face acquired XetHub 🔥

XetHub is a Seattle-based company founded by Yucheng Low, Ajit Banerjee, Rajat Arya who previously worked at Apple where they built and scaled Apple’s internal ML infrastructure. XetHub’s mission is to enable software engineering best practices for AI development. XetHub has developed technologies to enable Git to scale to TB repositories and enable teams to explore, understand and work together on large evolving datasets and models. They were soon joined by a talented team of 12 team members. You should give them a follow at their new org page: [hf.co/xet-team](https://huggingface.co/xet-team)

## Our common goal at HF

> The XetHub team will help us unlock the next 5 years of growth of HF datasets and models by switching to our own, better version of LFS as storage backend for the Hub's repos.
>
> – Julien Chaumond, HF CTO

Back in 2020 when we built the first version of the HF Hub, we decided to build it on top of Git LFS because it was decently well-known and it was a reasonable choice to bootstrap the Hub’s usage.

We knew back then, however, that we would want to switch to our own, more optimized storage and versioning backend at some point. Git LFS – even though it stands for Large File storage – was just never meant for the type of large files we handle in AI, which are not just large, but _very very_ large 😃

## Example future use cases 🔥 – what this will enable on the Hub

Let's say you have a 10GB Parquet file. You add a single row. Today you need to re-upload 10GB. With the chunked files and deduplication from XetHub, you will only need to re-upload the few chunks containing the new row.

Another example for GGUF model files: let’s say [@bartowski](https://huggingface.co/bartowski) wants to update one single metadata value in the GGUF header for a Llama 3.1 405B repo. Well, in the future bartowski can only re-upload a single chunk of a few kilobytes, making the process way more efficient 🔥

As the field moves to trillion parameters models in the coming months (thanks Maxime Labonne for the new <a href="https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct">BigLlama-3.1-1T</a> 🤯) our hope if that this new tech will unlock new scale both in the community, and inside of Enterprise companies.

Finally, with large datasets and large models come challenges with collaboration. How do teams work together on large data, models and code? How do users understand how their data and models are evolving? We will be working to find better solutions to answer these questions.

## Fun current stats on Hub repos 🤯🤯

- number of repos: 1.3m models, 450k datasets, 680k spaces
- total cumulative size: 12PB stored in LFS (280M files) / 7,3 TB stored in git (non-LFS)
- Hub’s daily number of requests: 1B
- daily Cloudfront bandwidth: 6PB 🤯

## A personal word from [@ylow](https://huggingface.co/yuchenglow)

<!-- <i’ll insert a pic of yucheng (hf profile)> -->

I have been part of the AI/ML world for over 15 years, and have seen how deep learning has slowly taken over vision, speech, text and really increasingly every data domain. 

What I have severely underestimated is the power of data. What seemed like impossible tasks just a few years ago (like image generation) turned out to be possible with orders of magnitude more data, and a model with the capacity to absorb it. In hindsight, this is an ML history lesson that has repeated itself many times.

I have been working in the data domain ever since my PhD. First in a startup (GraphLab/Dato/Turi) where I made structured data and ML algorithms scale on a single machine. Then after it was acquired by Apple, worked to scale AI data management to >100PB, supporting 10s of internal teams who shipped 100s of features annually. In 2021, together with my co-founders, supported by Madrona and other angel investors, started XetHub to bring our learnings of achieving collaboration at scale to the world.

XetHub’s goal is to enable ML teams to operate like software teams, by scaling Git file storage to TBs, seamlessly enabling experimentation and reproducibility, and providing the visualization capabilities to understand how datasets and models evolve. 

I, along with the entire XetHub team, are very excited to join Hugging Face and continue this mission to make AI collaboration and development easier - by integrating XetHub technology into Hub - and to release these features to the largest ML Community in the world!

## Finally, our Infrastructure team is hiring 👯

If you like those subjects and you want to build and scale the collaboration platform for the open source AI movement, get in touch!

