---
title: "Deploying Speech-to-Speech on Hugging Face Inference Endpoints with a Custom Docker Container" 
thumbnail: /blog/assets/s2s_endpoint/thumbnail.png
authors:
- user: andito
- user: derek-thomas
- user: dmaniloff
- user: eustlb

---

# Introduction

[Speech-to-Speech (S2S)](https://github.com/huggingface/speech-to-speech) is an exciting new project from Hugging Face that combines several advanced models to create a seamless, almost magical experience: you speak, and the system responds with a synthesized voice. 

The project implements a cascaded pipeline leveraging models available through the Transformers library on the Hugging Face hub. The pipeline consists of the following components:

1. Voice Activity Detection (VAD)
2. Speech to Text (STT)
3. Language Model (LM)
4. Text to Speech (TTS)

What's more, S2S has multi-language support! It currently supports English, French, Spanish, Chinese, Japanese, and Korean. You can run the pipeline in single-language mode or use the `auto` flag for automatic language detection. Check out the repo for more details [here](https://github.com/huggingface/speech-to-speech). Or even better, check out this demo:

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/<@eustache fill the version in>/gradio.js"
></script>

<gradio-app theme_mode="light" space="<@eustache fill the space in>"></gradio-app>

```
> 👩🏽‍💻: That's all amazing, but how do I run S2S?
> 🤗: Great question!
```

Running Speech-to-Speech requires significant computational resources. Even on a high-end laptop you might encounter latency issues, particularly when using the most advanced models. While a powerful GPU can mitigate these problems, not everyone has the means (or desire!) to set up their own hardware.

This is where Hugging Face's [Inference Endpoints (IE)](https://huggingface.co/inference-endpoints) come into play. Inference Endpoints allow you to rent a virtual machine equipped with a GPU (or other hardware you might need) and pay only for the time your system is running, providing an ideal solution for deploying performance-heavy applications like Speech-to-Speech.

In this blog post, we’ll guide you step by step to deploy Speech-to-Speech to a Hugging Face Inference Endpoint. This is what we'll cover:

- Understanding Inference Endpoints and a quick overview of the different ways to setup IE, including a custom container image (which is what we'll need for S2S)
- Building a custom docker image for S2S
- Deploying the custom image to IE and having some fun with S2S!

# Inference Endpoints

Inference Endpoints provide a scalable and efficient way to deploy machine learning models. These endpoints allow you to serve models with minimal setup, leveraging a variety of powerful hardware. Inference Endpoints are ideal for deploying applications that require high performance and reliability, without the need to manage underlying infrastructure.

Here's a few key features, and be sure to check out the documentation for more:

- **Simplicity** - You can be up and running in minutes thanks to IE's direct support of models in the Hugging Face hub.
- **Scalability** - You don't have to worry about scale, since IE scales automatically, including to zero, in order to handle varying loads and save costs.
- **Customization**: You can customize the setup of your IE to handle new tasks. More on this below.

Inference Endpoints supports all of the Transformers and Sentence-Transformers tasks, but can also support custom tasks. These are the IE setup options:

1. **Pre-built Models**: Quickly deploy models directly from the Hugging Face hub.
2. **Custom Handlers**: Define custom inference logic for more complex pipelines.
3. **Custom Docker Images**: Use your own Docker images to encapsulate all dependencies and custom code.

For simpler models, options 1 and 2 are ideal and make deploying with Inference Endpoints super straightforward. However, for a complex pipeline like S2S, wen will need the flexibility of option 3: deploying our IE using a custom Docker image.

This method not only provided more flexibility but also improved performance by optimizing the build process and bundling necessary data. If you’re dealing with complex model pipelines or want to optimize your application deployment, this guide will offer valuable insights.


# Deploying Speech-to-Speech on Inference Endpoints
Let's get into it!

# Building the custom Docker image

To begin creating a custom Docker image, we started by cloning Hugging Face’s default Docker image repository. This serves as a great starting point for deploying machine learning models in inference tasks.

```bash
git clone https://github.com/huggingface/huggingface-inference-toolkit
```

### Why Clone the Default Repository?
- **Solid Foundation**: The repository provides a pre-optimized base image designed specifically for inference workloads, which gives a reliable starting point.
- **Compatibility**: Since the image is built to align with Hugging Face’s deployment environment, this ensures smooth integration when you deploy your own custom image.
- **Ease of Customization**: The repository offers a clean and structured environment, making it easy to customize the image for the specific requirements of your application.

You can check out all of [our changes here](https://github.com/andimarafioti/speech-to-speech-inference-toolkit/pull/1/files)

### Customizing the Docker Image for the Speech-to-Speech Application

With the repository cloned, the next step was tailoring the image to support our Speech-to-Speech pipeline.

1. Adding the Speech-to-Speech Project

To integrate the project smoothly, we added the speech-to-speech codebase and any required datasets as submodules. This approach offers better version control, ensuring the exact version of the code and data is always available when the Docker image is built.

By including data directly within the Docker container, we avoid having to download it each time the endpoint is instantiated, which significantly reduces startup time and ensures the system is reproducible. The data is stored in a Hugging Face repository, which provides easy tracking and versioning.

```bash
git submodule add https://github.com/huggingface/speech-to-speech.git
git submodule add https://huggingface.co/andito/fast-unidic
```

2. Optimizing the Docker Image

Next, we modified the Dockerfile to suit our needs:

- **Streamlining the Image**: We removed packages and dependencies that weren’t relevant to our use case. This reduces the image size and cuts down on unnecessary overhead during inference.
- **Installing Requirements**: We moved the installation of `requirements.txt` from the entry point to the Dockerfile itself. This way, the dependencies are installed when building the Docker image, speeding up deployment since these packages won’t need to be installed at runtime.

3. Deploying the Custom Image
 
Once the modifications were in place, we built and pushed the custom image to Docker Hub:
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

### Inference Endpoints GUI
1. Navigate to https://ui.endpoints.huggingface.co/new
2. Fill in the relevant information
    - Model Repository - `andito/s2s`
    - Model Name - Feel free to rename if you don't like the generated name 
        - e.g. `speech-to-speech-demo` 
        - Keep it lower-case and short
    - Choose your preferred Cloud and Hardware - We used `AWS` `GPU` `L4`
        - It's only `$0.80` an hour and is big enough to handle the models
    - Advanced Configuration (click the expansion arrow ➤)
        - Container Type - `Custom`
        - Container Port - `80`
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
> [!WARNING] The current [huggingface-inference-toolkit entrypoint](https://github.com/huggingface/huggingface-inference-toolkit/blob/028b8250427f2ab8458ed12c0d8edb50ff914a08/scripts/entrypoint.sh#L4) uses port 5000 as default, but the inference endpoint expects port 80. You should match this in the **Container Port**. We already set it in our dockerfile, but beware if making your own from scratch!

## Inference Endpoints API

Here we will walk through the steps for creating the endpoint with the API. Just use this code in your python environment of choice.

Make sure to use `0.25.1` or greater
```bash
pip install huggingface_hub>=0.25.1
```

Use a [token](https://huggingface.co/docs/hub/en/security-tokens) that can write an endpoint (Write or Fine-Grained)
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

To use the endpoint, we will need to build a small webservice. The code for it is done on `webservice_starlette.py` and `s2s_handler.py`. Normally, you would only have a custom handler for an endpoint, but since we want to have a really low latency, we also built the webservice to support websocket connections instead of normal requests. This sounds intimidating at first, but the webservice is only 32 lines of code!

  <p>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/s2s_endpoint/webservice.png" alt="Webservice code" width="800px">
  </p>

This code will run `prepare_handler` on startup, which will initialize all the models and warm them up. Then, each message will be processed by `inference_handler.process_streaming_data`

  <p>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/s2s_endpoint/process_streaming.png" alt="Process streaming code" width="800px">
  </p>

This method simply receives the audio data from the client, chunks it into small parts for the VAD, and submits it to a queue for processing. Then it checks the output processing queue (the spoken response from the model!) and returns it if there is something. All of the internal processing is handled by [Hugging Face's speech_to_speech library](https://github.com/huggingface/speech-to-speech).

## Custom handler custom client

The webservice receives and returns audio. But there is still a big missing piece, how do we record and play back the audio? For that, we created [a client](https://github.com/huggingface/speech-to-speech/blob/inference-endpoint/audio_streaming_client.py) that connects to the service. The easiest is to divide the analysis in the connection to the webservice and the recording/playing of audio.

  <p>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/s2s_endpoint/client.png" alt="Audio client code" width="800px">
  </p>

Initializing the webservice client requires setting a header for all messages with our Hugging Face Token. When initializing the client, we set what we want to do on common messages (open, close, error, message). This will determine what our client does when the server sends it messages. 

  <p>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/s2s_endpoint/messages.png" alt="Audio client messages code" width="800px">
  </p>

We can see that the reactions to the messages are straight forward, with the `on_message` being the only method with more complexity. This method understands when the server is done responding and starts 'listening' back to the user. Otherwise, it puts the data from the server in the playback queue.

  <p>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/s2s_endpoint/client-audio.png" alt="Client's audio record and playback" width="800px">
  </p>

The client's audio section has 4 tasks:
1. Record the audio
2. Submit the audio recordings
3. Receive the audio responses from the server
4. Playback the audio responses

The audio is recorded on the `audio_input_callback` method, it simply submits all chunks to a queue. Then, it is sent to the server with the `send_audio` method. Here, if there is no audio to send, we still submit an empty array in order to receive a response from the server. The responses from the server are handled by the `on_message` method we saw earlier in the blog. Then, the playback of the audio responses are handled by the `audio_output_callback` method. Here we only need to ensure that the audio is in the range we expect (We don't want to destroy someone eardrum's because of a faulty package!) and ensure that the size of the output array is what the playback library expects. 

# Conclusion

In this post, we walked through the steps of deploying the Speech-to-Speech (S2S) pipeline on Hugging Face Inference Endpoints using a custom Docker image. We built a custom container to handle the complexities of the S2S pipeline and demonstrated how to configure it for scalable, efficient deployment. Hugging Face Inference Endpoints make it easier to bring performance-heavy applications like Speech-to-Speech to life, without the hassle of managing hardware or infrastructure.

If you're interested in trying it out or have any questions, feel free to explore the following resources:

- [Speech-to-Speech GitHub Repository](https://github.com/huggingface/speech-to-speech)
- [Speech-to-Speech Inference Toolkit](https://github.com/andimarafioti/speech-to-speech-inference-toolkit)
- [Base Inference Toolkit](https://github.com/huggingface/huggingface-inference-toolkit)
- [Hugging Face Inference Endpoints Documentation](https://huggingface.co/docs/inference-endpoints/en/guides/custom_container)

Have issues or questions? Open a discussion on the relevant GitHub repository, and we’ll be happy to help!