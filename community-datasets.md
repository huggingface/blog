---
title: "Data is better together: Enabling communities to collectively build better datasets together using Argilla and Hugging Face Spaces"
thumbnail: /blog/assets/community-datasets/thumbnail.png
authors:
- user: davanstrien
- user: dvilasuero
  guest: true
---

# Data is better together: Enabling communities to collectively build better datasets together using Argilla and Hugging Face Spaces 

Recently, Argilla and Hugging Face [launched](https://huggingface.co/posts/dvilasuero/680660181190026) `Data is Better Together`, an experiment to collectively build a preference dataset of prompt rankings. In a few days, we had:

- 350 community contributors labeling data 
- Over 11,000 prompt ratings

See the [progress dashboard](https://huggingface.co/spaces/DIBT/prompt-collective-dashboard) for the latest stats!

This resulted in the release of [`10k_prompts_ranked`](https://huggingface.co/datasets/DIBT/10k_prompts_ranked), a dataset consisting of 10,000 prompts with user ratings for the quality of the prompt. We want to enable many more projects like this!

In this post, we’ll discuss why we think it’s essential for the community to collaborate on building datasets and share an invitation to join the first cohort of communities [Argilla](https://argilla.io/) and Hugging Face will support to develop better datasets together! 


## Data remains essential for better models

Data continues to be essential for better models: We see continued evidence from [published research](https://huggingface.co/papers/2402.05123), open-source [experiments](https://argilla.io/blog/notus7b/), and from the open-source community that better data can lead to better models. 

<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/17480bfba418032faec37da19e9c678ac9eeed43/blog/community-datasets/why-model-better.png" alt="Screenshot of datasets in the Hugging Face Hub"><br> 
<em>The question.</em> 
 </p> 

<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/17480bfba418032faec37da19e9c678ac9eeed43/blog/community-datasets/data-is-the-answer.png" alt="Screenshot of datasets in the Hugging Face Hub"><br> 
<em>A frequent answer.</em> 
 </p> 

## Why build datasets collectively?

Data is vital for machine learning, but many languages, domains, and tasks still lack high-quality datasets for training, evaluating, and benchmarking — the community already shares thousands of models, datasets, and demos daily via the Hugging Face Hub. As a result of collaboration, the open-access AI community has created amazing things. Enabling the community to build datasets collectively will unlock unique opportunities for building the next generation of datasets to build the next generation of models. 


Empowering the community to build and improve datasets collectively will allow people to:

- Contribute to the development of Open Source ML with no ML or programming skills required.
- Create chat datasets for a particular language.
- Develop benchmark datasets for a specific domain. 
- Create preference datasets from a diverse range of participants.
- Build datasets for a particular task.
- Build completely new types of datasets collectively as a community.

Importantly we believe that building datasets collectively will allow the community to build better datasets abd allow people who don't know how to code to contribute to the development of AI.

### Making it easy for people to contribute 

One of the challenges to many previous efforts to build AI datasets collectively was setting up an efficient annotation task. Argilla is an open-source tool that can help create datasets for LLMs and smaller specialised task-specific models. Hugging Face Spaces is a platform for building and hosting machine learning demos and applications. Recently, Argilla added support for authentication via a Hugging Face account for Argilla instances hosted on Spaces. This means it now takes seconds for users to start contributing to an annotation task. 

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://video.twimg.com/ext_tw_video/1757693043619004416/pu/vid/avc1/1068x720/wh3DyY0nMcRJaMki.mp4?tag=12"
    ></video>
</figure>


Now that we have stress-tested this new workflow when creating the [`10k_prompts_ranked`](https://huggingface.co/datasets/DIBT/10k_prompts_ranked), dataset, we want to support the community in launching new collective dataset efforts. 

## Join our first cohort of communities who want to build better datasets together!

We’re very excited about the possibilities unlocked by this new, simple flow for hosting annotation tasks. To support the community in building better datasets, Hugging Face and Argilla invite interested people and communities to join our initial cohort of community dataset builders. 

People joining this cohort will:

- Be supported in creating an Argilla Space with Hugging Face authentication. Hugging Face will grant free persistent storage and improved CPU spaces for participants. 
- Have their comms and promotion advertising the initiative amplified by Argilla and Hugging Face.
- Be invited to join a cohort community channel 

Our goal is to support the community in building better datasets together. We are open to many ideas and want to support the community as far as possible in building better datasets together.

## What types of projects are we looking for?

We are open to supporting many types of projects, especially those of existing open-source communities. We are particularly interested in projects focusing on building datasets for languages, domains, and tasks that are currently underrepresented in the open-source community. Our only current limitation is that we're primarily focused on text-based datasets. If you have a very cool idea for multimodal datasets, we'd love to hear from you, but we may not be able to support you in this first cohort. 

Tasks can either be fully open or open to members of a particular Hugging Face Hub organization. 

If you want to be part of the first cohort, please join us in the `#data-is-better-together` channel in the [Hugging Face Discord](http://hf.co/join/discord) and let us know what you want to build together! 

We are looking forward to building better datasets together with you!


