---
title: "Google Cloud TPUs made available to Hugging Face users" 
thumbnail: /blog/assets/tpu-inference-endpoints-spaces/thumbnail.png
authors:
- user: pagezyhf
- user: michellehbn
- user: philschmid
- user: tengomucho
---

# Google Cloud TPUs made available to Hugging Face users

![Google Cloud TPUs made available to Hugging Face users](/blog/assets/tpu-inference-endpoints-spaces/thumbnail.png)

We're excited to share some great news! AI builders are now able to accelerate their applications with [Google Cloud TPUs](https://cloud.google.com/tpu?hl=en) on Hugging Face [Inference Endpoints](https://ui.endpoints.huggingface.co/) and [Spaces](https://huggingface.co/spaces)!

For those who might not be familiar, TPUs are custom-made AI hardware designed by Google. They are known for their ability to scale cost-effectively and deliver impressive performance across various AI workloads. This hardware has played a crucial role in some of Google's latest innovations, including the development of the Gemma 2 open models. We are excited to announce that TPUs will now be available for use in Inference Endpoints and Spaces.

This is a big step in our ongoing [collaboration](https://huggingface.co/blog/gcp-partnership) to provide you with the best tools and resources for your AI projects. We're really looking forward to seeing what amazing things you'll create with this new capability!

## Hugging Face Inference Endpoints support for TPUs

Hugging Face Inference Endpoints provides a seamless way to deploy Generative AI models  with a few clicks on a dedicated, managed infrastructure using the cloud provider of your choice. Starting today, Google [TPU v5e](https://cloud.google.com/tpu/docs/v5e-inference) is available on Inference Endpoints. Choose the model you want to deploy, select Google Cloud Platform, select us-west1 and you’re ready to pick a TPU configuration:

We have 3 instance configurations, with more to come:

- v5litepod-1 TPU v5e with 1 core and 16 GB memory ($1.375/hour)
- v5litepod-4 TPU v5e with 4 cores and 64 GB memory ($5.50/hour)
- v5litepod-8 TPU v5e with 8 cores and 128 GB memory ($11.00/hour)

![ie-tpu](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tpu-inference-endpoints-spaces/ie-tpu.png)

While you can use v5litepod-1 for models with up to 2 billion parameters without much hassle, we recommend to use v5litepod-4 for larger models to avoid memory budget issues. The larger the configuration, the lower the latency will be.

Together with the product and engineering teams at Google, we're excited to bring the performance and cost efficiency of TPUs to our Hugging Face community. This collaboration has resulted in some great developments:

1. We've created an open-source library called [Optimum TPU](https://github.com/huggingface/optimum-tpu), which makes it super easy for you to train and deploy Hugging Face models on Google TPUs.
2. Inference Endpoints uses Optimum TPU along with Text Generation Inference (TGI) to serve Large Language Models (LLMs) on TPUs.
3. We’re always working on support for a variety of model architectures. Starting today you can deploy Gemma & Llama in a few clicks. (Optimum TPU supported models).

## Hugging Face Spaces support for TPUs

Hugging Face Spaces provide developers with a platform to create, deploy, and share AI-powered demos and applications quickly. We are excited to introduce new TPU v5e instance support for Hugging Face Spaces. To upgrade your Space to run on TPUs, navigate to the Settings button in your Space and select the desired configuration:

- v5litepod-1 TPU v5e with 1 core and 16 GB memory ($1.375/hour)
- v5litepod-4 TPU v5e with 4 cores and 64 GB memory ($5.50/hour)
- v5litepod-8 TPU v5e with 8 cores and 128 GB memory ($11.00/hour)

![spaces-tpu](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tpu-inference-endpoints-spaces/spaces-tpu.png)

Go build and share with the community awesome ML-powered demos on TPUs on [Hugging Face Spaces](https://huggingface.co/spaces)!

We're proud of what we've achieved together with Google and can't wait to see how you'll use TPUs in your projects.