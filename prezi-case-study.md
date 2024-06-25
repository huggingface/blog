---
title: "Going multimodal: How Prezi is leveraging the Hub and the Expert Support Program to accelerate their ML roadmap"
thumbnail: /blog/assets/70_sempre_health/thumbnailprezi.jpg
authors:
- user: Violette
- user: jeffboudier
- user: MoritzLaurer
- user: bmateusz
  guest: true
---

# Going multimodal: How Prezi is leveraging the Hub and the Expert Support Program to accelerate their ML roadmap

Everybody knows that a great visual is worth a thousand words. The team at Prezi, a visual communications software company, is putting this insight into practice with their Prezi presentations that combine images and text in highly dynamic presentations. 

Prezi has joined the Hugging Face Expert Support Program to fully leverage modern machine learning's potential. Over the past months, Hugging Face has supported Prezi in integrating smaller, more efficient open-source models into their ML workflows. This cooperation started at a perfect time, as multimodal models are becoming increasingly capable. 

We recently sat down with [Máté Börcsök](https://www.linkedin.com/in/mateborcsok/?originalSubdomain=hu), a backend engineer at [Prezi](https://prezi.com/), to talk about their experience in the [Expert Support Program](https://huggingface.co/support). In this short video, Máté walks us through some of their machine learning work and shares their experience collaborating with our team via the Expert Support Program. 

<iframe width="100%" style="aspect-ratio: 16 / 9;" src="https://www.youtube.com/embed/pM6D0tRoIbI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

_If you'd like to accelerate your machine learning roadmap with the help of our experts, as Máté and his team did, visit [hf.co/support](https://huggingface.co/support) to learn more about our Expert Support Program and request a quote._


## Transcript with additional details:

### Introduction

My name is Máté, and I am a backend engineer at Prezi, an online presentation tool that brings your ideas to life.

### How does the HF Expert Support Program help you build AI?

Our flagship AI product at Prezi is Prezi AI, which helps our users create better Prezi presentations faster. Users start by providing a prompt and description of the presentation they want to create. The system then automatically creates a draft presentation for them to get started. It’s a complex system that calls different services and builds up the presentation’s structure using closed models and various asset provider services.

When we joined the program, we already had a version of this system, and our expert reviewed the flow and suggested improvements. Our pipeline includes a search system to find suitable assets (images and texts) for each unique presentation. In this context, an important piece of advice was, for example, to add an open-source re-ranker model to the system, which can find the best images or texts for your presentation cheaper, faster, and better than an LLM.

Our use cases are inherently multi-modal as our presentations combine images and text. There are a lot of models released every week, and our expert helps us cut through the hype and understand which models are useful for us and which are not. This helps us save a lot of time, as we are using a combination of vision models, text models, and vision-language models (VLMs) to solve our unique challenges. Multimodal machine learning is challenging, and the guidance is really appreciated. We are not Machine Learning Engineers, and we are learning this together on the way.

### What’s your favorite feature of Inference Endpoints?

I highly recommend you check out the [Endpoint Model Catalog](https://ui.endpoints.huggingface.co/catalog). It is a curated list of models that work well with Inference Endpoints and require zero configuration. I love that you can set it up so that the Endpoint goes to sleep after a few minutes, so it won’t burn money. It also supports single and quad A100 instances required for some models. Keeping the models updated is also straightforward. Inference Endpoints let us deploy the latest version with a single click or roll back to any older version using the Git hash. None of these features are easily available on AWS, so it was very convenient for us to use them. Even if a model is not in the [catalog](https://ui.endpoints.huggingface.co/catalog) yet, it’s relatively easy to make them work. At least it was easy for me, with our expert supporting us.

### What teams would benefit most from Expert Support?

The Hugging Face partnership opened the doors of machine learning for us. Our dedicated expert gives us access to a community of machine learning experts who can give feedback on our wildest questions. As I said earlier, we are not Machine Learning Engineers. Our expert guides us to work on the right things, sharing best practices and state-of-the-art models for embedding, re-ranking, and object detection and showing us how to fine-tune new vision language models and collect and curate data. These are mostly things we can do ourselves, but his guidance gives a huge speedup and keeps us focused on meaningful tasks for our users.

---

With the he Expert Support Program, we've put together a world-class team to help customers build better ML solutions, faster. Our experts answer questions and find solutions as needed in your machine learning journey from research to production. Visit [hf.co/support](https://huggingface.co/support) to learn more and request a quote.
