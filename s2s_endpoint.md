---
title: "Deploying Speech-to-Speech on Hugging Face Inference Endpoints with a Custom Docker Container" 
thumbnail: 
authors:
- user: andito

---

# Deploying speech-to-speech

[Speech-to-speech](https://github.com/huggingface/speech-to-speech) is an exciting new project from Hugging Face that combines several advanced models to create a seamless, almost magical experience: you speak, and the system responds with a synthesized voice. However, running such a complex system isn’t easy—it demands substantial computational power, and even when run on a high-end laptop, latency can become a problem, especially when using the best-performing models. While a powerful GPU could solve this issue, not everyone wants to build their own cluster.

This is where Hugging Face's Inference Endpoints (IE) come into play. Inference Endpoints allow you to rent a virtual machine equipped with a GPU and pay only for the time your system is running, providing an ideal solution for deploying performance-heavy applications like speech-to-speech.

For simpler models, deploying with Inference Endpoints is straightforward. You can set up a model from Hugging Face's Transformers library with minimal configuration. However, for a more complex pipeline—like mine, which involves voice activity detection (VAD), speech-to-text (STT), language models (LMM), and text-to-speech (TTS)—using a custom endpoint handler quickly revealed limitations. To overcome these, I turned to a more flexible and powerful approach: deploying a custom Docker image.

In this blog post, I’ll guide you step by step through how I built a custom Docker image tailored for my speech-to-speech pipeline and deployed it to a Hugging Face Inference Endpoint. This method not only provided more flexibility but also improved performance by optimizing the build process and bundling necessary data. If you’re dealing with complex model pipelines or want to optimize your application deployment, this guide will offer valuable insights.

# Building the custom Docker image

To start building my custom Docker image, I cloned the default Docker image repository provided by Hugging Face.

```bash
git clone https://github.com/huggingface/huggingface-inference-toolkit
```

Why Clone the Default Repository?
- Solid Foundation: Provides a proven base image optimized for inference tasks.
- Compatibility: Ensures that the custom image aligns with Hugging Face's deployment requirements.
- Ease of Customization: Offers a structured environment to implement specific changes for your application.

You can see all of [our changes here](https://github.com/andimarafioti/speech-to-speech-inference-toolkit/pull/1/files)

With the base repository cloned, the next step was to customize it for my speech-to-speech application.

First, I added my speech-to-speech project as submodules to the project. This approach allows for better version control and integration. 

I also added some data I will copy to the docker container. For this, I created a repository on Hugging Face and dumped my data there, which I then initialized here as a submodule. This saves me from downloading the data each time that I instanciate the endpoint or build the container, and it gives me an easy way to track the data and make the project reproducible.

```bash
git submodule add https://github.com/your-username/speech-to-speech.git
git submodule add https://huggingface.co/andito/fast-unidic
```

Then, I modified some parts of the Dockerfile (removed packages I know I won't need), and moved in the install of the `requirements.txt` for my project into the Dockerfile to avoid having it done on the inference endpoint. Once the Dockerfile was complete, I built and pushed it with:

```bash
DOCKER_DEFAULT_PLATFORM="linux/amd64" docker build -t speech-to-speech -f dockerfiles/pytorch/Dockerfile . 
docker tag speech-to-speech andito/speech-to-speech:latest 
docker push andito/speech-to-speech:latest
```

Once this is done, we can use the docker image directly in the endpoint. 

# Setting up an endpoint with a custom docker image:

Derek, can you fill this?