---
title: "Deploying Speech-to-Speech on Hugging Face Inference Endpoints with a Custom Docker Container" 
thumbnail: 
authors:
- user: andito
- user: derek-thomas
- user: dmaniloff
- user: eustlb

---

# Deploying speech-to-speech

[Speech-to-speech](https://github.com/huggingface/speech-to-speech) is an exciting new project from Hugging Face that combines several advanced models to create a seamless, almost magical experience: you speak, and the system responds with a synthesized voice. However, running such a complex system isn’t easy—it demands substantial computational power, and even when run on a high-end laptop, latency can become a problem, especially when using the best-performing models. While a powerful GPU could solve this issue, not everyone wants to build their own cluster.

This is where Hugging Face's [Inference Endpoints (IE)](https://huggingface.co/inference-endpoints) come into play. Inference Endpoints allow you to rent a virtual machine equipped with a GPU and pay only for the time your system is running, providing an ideal solution for deploying performance-heavy applications like speech-to-speech.

For simpler models, deploying with Inference Endpoints is straightforward. You can set up a model from Hugging Face's Transformers library with minimal configuration. However, for a more complex pipeline—like mine, which involves voice activity detection (VAD), speech-to-text (STT), language models (LMM), and text-to-speech (TTS)—using a custom endpoint handler quickly revealed limitations. To overcome these, I turned to a more flexible and powerful approach: deploying a custom Docker image.

In this blog post, I’ll guide you step by step through how I built a custom Docker image tailored for my speech-to-speech pipeline and deployed it to a Hugging Face Inference Endpoint. This method not only provided more flexibility but also improved performance by optimizing the build process and bundling necessary data. If you’re dealing with complex model pipelines or want to optimize your application deployment, this guide will offer valuable insights.

# Building the custom Docker image

To begin creating my custom Docker image, I started by cloning Hugging Face’s default Docker image repository. This serves as a great starting point for deploying machine learning models in inference tasks.

```bash
git clone https://github.com/huggingface/huggingface-inference-toolkit
```

## Why Clone the Default Repository?
- Solid Foundation: The repository provides a pre-optimized base image designed specifically for inference workloads, which gives a reliable starting point.
- Compatibility: Since the image is built to align with Hugging Face’s deployment environment, this ensures smooth integration when you deploy your own custom image.
- Ease of Customization: The repository offers a clean and structured environment, making it easy to customize the image for the specific requirements of your application.

You can check out all of [our changes here](https://github.com/andimarafioti/speech-to-speech-inference-toolkit/pull/1/files)

## Customizing the Docker Image for the Speech-to-Speech Application

With the repository cloned, the next step was tailoring the image to support my speech-to-speech pipeline.

1. Adding the Speech-to-Speech Project as Submodules

To integrate my project smoothly, I added the speech-to-speech codebase and any required datasets as submodules. This approach offers better version control, ensuring the exact version of the code and data is always available when the Docker image is built.

By including data directly within the Docker container, I avoid having to download it each time the endpoint is instantiated, which significantly reduces startup time and ensures the system is reproducible. The data is stored in a Hugging Face repository, which provides easy tracking and versioning.

```bash
git submodule add https://github.com/your-username/speech-to-speech.git
git submodule add https://huggingface.co/andito/fast-unidic
```

2. Modifying the Dockerfile

Next, I optimized the Dockerfile to suit my needs:

- Streamlining the Image: I removed packages and dependencies that weren’t relevant to my use case. This reduces the image size and cuts down on unnecessary overhead during inference.
- Installing Requirements: I moved the installation of `requirements.txt` from the entry point to the Dockerfile itself. This way, the dependencies are installed when building the Docker image, speeding up deployment since these packages won’t need to be installed at runtime.

3. Deploying the Custom Image
 
Once the modifications were in place, I built and pushed the custom image to Docker Hub:
```bash
DOCKER_DEFAULT_PLATFORM="linux/amd64" docker build -t speech-to-speech -f dockerfiles/pytorch/Dockerfile . 
docker tag speech-to-speech andito/speech-to-speech:latest 
docker push andito/speech-to-speech:latest
```

With the Docker image built and pushed, it’s ready to be used in the Hugging Face Inference Endpoint. By using this pre-built image, the endpoint can launch faster and run more efficiently, as all dependencies and data are pre-packaged within the image.

# Setting up an Inference Endpoint

Using a custom docker image just requires a slightly different configuration, feel free to check out the [documentation](https://huggingface.co/docs/inference-endpoints/en/guides/custom_container). We will walk through the approach to do this in both the GUI and the API.

Pre-Steps
1. Login: https://huggingface.co/login
2. Request access to [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
3. Create a Fine-Graned Token: https://huggingface.co/settings/tokens/new?tokenType=fineGrained
![Fine-Grained Token](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/s2s_endpoint/fine-grained-token.png)
    - Select access to gated repos
    - If you are using the API make sure to select permissions to Manage Inference Endpoints

## Inference Endpoints GUI
1. Navigate to https://ui.endpoints.huggingface.co/new
2. Fill in the relevant information
    - Model Repository - `andito/fast-unidic`
    - Model Name - Feel free to rename if you don't like the generated name 
        - e.g. `speech-to-speech-demo` 
        - Keep it lower-case and short
    - Choose your preferred Cloud and Hardware -  I used `AWS` `GPU` `L4`
        - It's only `$0.80` an hour and is big enough to handle the models
    - Advanced Configuration (click the expansion arrow ➤)
        - Container Type - `Custom`
        - Container Port - `5000`
        - Container URL - `andito/speech-to-speech:latest`
        - Secrets - `HF_TOKEN`|`<your token here>`
<details>
  <summary>Click to show images</summary>
  <p>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/s2s_endpoint/new-inference-endpoint.png" alt="New Inference Endpoint" width="500px">
  </p>
  <p>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/s2s_endpoint/advanced-configuration.png" alt="Advanced Configuration" width="500px">
  </p>
</details>
3. Click `Create Endpoint`

> [!NOTE] The Model Repository doesn't actually matter since the models are specified and downloaded in the container creation, but Inference Endpoints requires a model, so feel free to pick a slim one of your choice.
> [!NOTE] You need to specify `HF_TOKEN` because we need to download gated models in the container creation stage. This won't be necessary if you use models that aren't gated or private.
> [!WARNING] The current [huggingface-inference-toolkit entrypoint](https://github.com/huggingface/huggingface-inference-toolkit/blob/028b8250427f2ab8458ed12c0d8edb50ff914a08/scripts/entrypoint.sh#L4) uses port 5000. You should match this in the **Container Port**

## Inference Endpoints API

Here we will walk through the steps for creating the endpoint with the API. Just use this code in your python environment of choice.

Make sure to use `0.25.1` or greater
```bash
pip install huggingface_hub>=0.25.1
```

Use a token that can write an endpoint (Write or Fine-Grained)
```python
from huggingface_hub import login
login()
```

```python
from huggingface_hub import create_inference_endpoint, get_token
endpoint = create_inference_endpoint(
    # Model Configuration
    "speech-to-speech-demo",
    repository="andito/fast-unidic",
    framework="custom",
    task="custom",
    # Security
    type="protected",
    # Hardware
    vendor="aws",
    accelerator="gpu",
    region="us-east-1",
    instance_size="x1",
    instance_type="nvidia-l4",
    # Image Configuration
    custom_image={
        "health_route": "/health",
        "url": "andito/speech-to-speech:latest", # Pulls from DockerHub
        "port": 5000
    },
    secrets={'HF_TOKEN':get_token()}
)

# Optional
endpoint.wait()
```
# Overview
![Overview](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/s2s_endpoint/overview.png)

Major Componants
- [Speech To Speech](https://github.com/huggingface/speech-to-speech/tree/inference-endpoint)
  - This is a Hugging Face Library, we put some inference-endpoint specific files in the `inference-endpoint` branch
- andito/fast-unidic
- [andimarafioti/speech-to-speech-toolkit](https://github.com/andimarafioti/speech-to-speech-inference-toolkit)
  - This was forked from [huggingface/huggingface-inference-toolkit](https://github.com/huggingface/huggingface-inference-toolkit) to help us build the Custom Container configured as we desire 


## Building the webserver
Basically go over my code on webservice_starlette.py 

- How to handle streaming connections with a websocket / they are not supported in the default example and I added code for them

## Custom handler custom client

Go over the custom handler and the custom client.
