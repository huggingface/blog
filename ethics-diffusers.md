---
title: "Ethical Guidelines for developing the Diffusers library" 
thumbnail: /blog/assets/ethics-diffusers/thumbnail.png
authors:
- user: giadap
---

# Ethical guidelines for developing the Diffusers library

We are on a journey to make our libraries more responsible, one commit at a time! 

As part of the [Diffusers library documentation](https://huggingface.co/docs/diffusers/main/en/index), we are proud to announce the publication of an [ethical framework](https://huggingface.co/docs/diffusers/main/en/conceptual/ethical_guidelines). 

Given diffusion models' real case applications in the world and potential negative impacts on society, this initiative aims to guide the technical decisions of the Diffusers library maintainers about community contributions. We wish to be transparent in how we make decisions, and above all, we aim to clarify what values guide those decisions.

We see ethics as a process that leverages guiding values, concrete actions, and continuous adaptation. For this reason, we are committed to adjusting our guidelines over time, following the evolution of the Diffusers project and the valuable feedback from the community that keeps it alive.

## Ethical guidelines

* **Transparency**: we are committed to being transparent in managing PRs, explaining our choices to users, and making technical decisions.
* **Consistency**: we are committed to guaranteeing our users the same level of attention in project management, keeping it technically stable and consistent.
* **Simplicity**: with a desire to make it easy to use and exploit the Diffusers library, we are committed to keeping the project’s goals lean and coherent.
* **Accessibility**: the Diffusers project helps lower the entry bar for contributors who can help run it even without technical expertise. Doing so makes research artifacts more accessible to the community.
* **Reproducibility**: we aim to be transparent about the reproducibility of upstream code, models, and datasets when made available through the Diffusers library.
* **Responsibility**: as a community and through teamwork, we hold a collective responsibility to our users by anticipating and mitigating this technology’s potential risks and dangers.

## Safety features and mechanisms

In addition, we provide a non-exhaustive - and hopefully continuously expanding! - list of safety features and mechanisms implemented by the Hugging Face team and the broader community.

* **[Community tab](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)**: it enables the community to discuss and better collaborate on a project.

* **Tag feature**: authors of a repository can tag their content as being “Not For All Eyes”

* **Bias exploration and evaluation**: the Hugging Face team provides a [Space](https://huggingface.co/spaces/society-ethics/DiffusionBiasExplorer) to demonstrate the biases in Stable Diffusion and DALL-E interactively. In this sense, we support and encourage bias explorers and evaluations.

* **Encouraging safety in deployment**
    * **[Safe Stable Diffusion](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion_safe)**: It mitigates the well-known issue that models, like Stable Diffusion, that are trained on unfiltered, web-crawled datasets tend to suffer from inappropriate degeneration. Related paper: [Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models](https://arxiv.org/abs/2211.05105).

    * **Staged released on the Hub**: in particularly sensitive situations, access to some repositories should be restricted. This staged release is an intermediary step that allows the repository’s authors to have more control over its use.

* **Licensing**: [OpenRAILs](https://huggingface.co/blog/open_rail), a new type of licensing, allow us to ensure free access while having a set of restrictions that ensure more responsible use. 