---
title: "How to deploy and fine-tune DeepSeek models on AWS" 
thumbnail: /blog/assets/deepseek-r1-aws/thumbnail.png
authors:
- user: pagezyhf
- user: jeffboudier
- user: dacorvo
---

# How to deploy and fine-tune DeepSeek models on AWS

A running document to showcase how to deploy and fine-tune DeepSeek R1 models with Hugging Face on AWS.

## What is DeepSeek-R1?

If you’ve ever struggled with a tough math problem, you know how useful it is to think a little longer and work through it carefully. [**OpenAI’s o1 model**](https://x.com/polynoamial/status/1834280155730043108) showed that when LLMs are trained to do the same—by using more compute during inference—they get significantly better at solving reasoning tasks like mathematics, coding, and logic.

However, the recipe behind OpenAI’s reasoning models has been a well kept secret. That is, until last week, when DeepSeek released their [**DeepSeek-R1**](https://huggingface.co/deepseek-ai/DeepSeek-R1) model and promptly broke the internet (and the [**stock market!**](https://x.com/KobeissiLetter/status/1883831022149927352)).

DeepSeek AI open-sourced DeepSeek-R1-Zero, DeepSeek-R1, and six dense models distilled from DeepSeek-R1 based on Llama and Qwen architectures. You can find them all in the [DeepSeek R1 collection](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d).

We collaborate with Amazon Web Services to make it easier for developers to deploy the latest Hugging Face models on AWS services to build better generative AI applications.

Let’s review how you can deploy and fine-tune DeepSeek R1 models with Hugging Face on AWS.
- [Deploy DeepSeek R1 models](#deploy-deepseek-r1-models)
    - [Deploy on AWS with Hugging Face Inference Endpoints](#deploy-on-aws-with-hugging-face-inference-endpoints)
    - [Deploy on Amazon Bedrock Marketplace]
    - [Deploy on Amazon SageMaker AI with Hugging Face LLM DLCs](#deploy-on-amazon-sagemaker-ai-with-hugging-face-llm-dlcs)
        - [DeepSeek R1 on GPUs](#deepseek-r1-on-gpus)
        - [Distilled models on GPUs](#distilled-models-on-gpus)
        - [Distilled models on Neuron](#distilled-models-on-neuron)
    - [Deploy on EC2 Neuron with the Hugging Face Neuron Deep Learning AMI](#deploy-on-ec2-neuron-with-the-hugging-face-neuron-deep-learning-ami)
- [Fine-tune DeepSeek R1 models](#fine-tune-deepseek-r1-models)
    - [Fine tune on Amazon SageMaker AI with Hugging Face Training DLCs](#fine-tune-on-amazon-sagemaker-ai-with-hugging-face-training-dlcs)
    - [Fine tune on EC2  Neuron with the Hugging Face Neuron Deep Learning AMI](#fine-tune-on-ec2--neuron-with-the-hugging-face-neuron-deep-learning-ami)

## Deploy DeepSeek R1 models

### Deploy on AWS with Hugging Face Inference Endpoints

[**Hugging Face Inference Endpoints**](https://ui.endpoints.huggingface.co/) offers an easy and secure way to deploy Machine Learning models on dedicated compute for use in production on AWS. Inference Endpoints empower developers and data scientists alike to create AI applications without managing infrastructure: simplifying the deployment process to a few clicks, including handling large volumes of requests with autoscaling, reducing infrastructure costs with scale-to-zero, and offering advanced security.

With Inference Endpoints, you can deploy any of the 6 distilled models from DeepSeek-R1 and also a quantized version of DeepSeek R1 made by Unsloth: https://huggingface.co/unsloth/DeepSeek-R1-GGUF.
On the model page, click on Deploy, then on HF Inference Endpoints. You will be redirected to the Inference Endpoint page, where we selected for you an optimized inference container, and the recommended hardware to run the model. Once you created your endpoint, you can send your queries to DeepSeek R1 for 8.3$ per hour with AWS 🤯.

You can find DeepSeek R1 and distilled models, as well as other popular open LLMs, ready to deploy on optimized configurations in the [Inference Endpoints Model Catalog](https://endpoints.huggingface.co/catalog?task=text-generation).

![deepseek_r1_ie.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deepseek-aws/deepseek_r1_ie.png)

| **Note:** The team is working on enabling DeepSeek models deployment on Inferentia instances. Stay tuned!

### Deploy on Amazon Bedrock Marketplace

You can deploy the Deepseek distilled models on Amazon Bedrock via the marketplace, which will deploy an endpoint in Amazon SageMaker AI under the hood. Here is a video of how you can navigate through the AWS console:

![bedrock-deployment.gif](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deepseek-aws/bedrock-deployment.gif)

### Deploy on Amazon Sagemaker AI with Hugging Face LLM DLCs

#### DeepSeek R1 on GPUs

| **Note:** The team is working on enabling DeepSeek-R1 deployment on Amazon Sagemaker AI with the Hugging Face LLM DLCs on GPU. Stay tuned! 

#### Distilled models on GPUs

You can deploy the Deepseek distilled models on Amazon Sagemaker AI with Hugging Face LLM DLCs using Jumpstart directly or using the Python Sagemaker SDK.
Here is a video of how you can navigate through the AWS console:

![jumpstart-deployment.gif](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deepseek-aws/jumpstart-deployment.gif)

Now we have seen how to deploy usig Jumpstart, let’s walk through the Python Sagemaker SDK deployment of DeepSeek-R1-Distill-Llama-70B. 

Code snippets are available on the model page under the Deploy button! 

![deploy_sagemaker_sdk.gif](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deepseek-aws/deploy_sagemaker_sdk.gif)

Before, let’s start with a few pre-requisites. Make sur you have a Sagemaker Domain [configured](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html), sufficient [quota](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html) in Sagemaker, and a JupyterLab [space](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide-create-space.html). For DeepSeek-R1-Distill-Llama-70B, you should raise the default quota for ml.g6.48xlarge for endpoint usage to 1. 

For reference, here are the hardware configurations we recommend you to use for each of the distilled variants:

| **Model** | **Instance Type** | **# of GPUs per replica** |
| --- | --- | --- |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B | ml.g6.48xlarge | 8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | ml.g6.12xlarge | 4 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | ml.g6.12xlarge | 4 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | ml.g6.2xlarge | 1 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | ml.g6.2xlarge | 1 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | ml.g6.2xlarge | 1 |

Once in a notebook, make sure to install the latest version of SageMaker SDK.

```python
!pip install sagemaker --upgrade
```

Then, instantiate a sagemaker_session which is used to determine the current region and execution role.

```python
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]
```

Create the SageMaker Model object with the Python SDK:

```python
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
model_name = hf_model_id.split("/")[-1].lower()

# Hub Model configuration. https://huggingface.co/models
hub = {
    "HF_MODEL_ID": model_id,
    "SM_NUM_GPUS": json.dumps(8)
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="3.0.1"),
    env=hub,
    role=role,
)
```

Deploy the model to a SageMaker endpoint and test the endpoint:

```python
endpoint_name = f"{model_name}-ep"

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    endpoint_name=endpoint_name,
    initial_instance_count=1,
    instance_type="ml.g6.48xlarge",
    container_startup_health_check_timeout=2400,
)
  
# send request
predictor.predict({"inputs": "What is the meaning of life?"})
```

That’s it, you deployed a Llama 70B reasoning model! 

Because you are using a TGI v3 container under the hood, the most performant parameters for the given hardware will be automatically selected.

Make sure you delete the endpoint once you finished testing it.

```python
predictor.delete_model()
predictor.delete_endpoint()
```

#### Distilled models on Neuron

Let’s walk through the deployment of DeepSeek-R1-Distill-Llama-70B on a Neuron instance, like AWS Trainium 2 and AWS Inferentia 2.

Code snippets are available on the model page under the Deploy button! 

![deploy_neuron.gif](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/deepseek-aws/deploy_neuron.gif)

The pre-requisites to deploy to a Neuron instance are the same. Make sure you have a SageMaker Domain [configured](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html), sufficient [quota](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html) in SageMaker, and a JupyterLab [space](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide-create-space.html). For DeepSeek-R1-Distill-Llama-70B, you should raise the default quota for `ml.inf2.48xlarge` for endpoint usage to 1.

Then, instantiate a `sagemaker_session` which is used to determine the current region and execution role.

```python
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]
```

Create the SageMaker Model object with the Python SDK:

```python
image_uri = get_huggingface_llm_image_uri("huggingface-neuronx", version="0.0.25")
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
model_name = hf_model_id.split("/")[-1].lower()

# Hub Model configuration
hub = {
    "HF_MODEL_ID": model_id,
    "HF_NUM_CORES": "24",
    "HF_AUTO_CAST_TYPE": "bf16",
    "MAX_BATCH_SIZE": "4",
    "MAX_INPUT_TOKENS": "3686",
    "MAX_TOTAL_TOKENS": "4096",
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=image_uri,
    env=hub,
    role=role,
)
```

Deploy the model to a SageMaker endpoint and test the endpoint:

```python
endpoint_name = f"{model_name}-ep"

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    endpoint_name=endpoint_name,
    initial_instance_count=1,
    instance_type="ml.inf2.48xlarge",
    container_startup_health_check_timeout=3600,
    volume_size=512,
)
  
# send request
predictor.predict(
    {
        "inputs": "What is is the capital of France?",
        "parameters": {
            "do_sample": True,
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }
    }
)
```

That’s it, you deployed a Llama 70B reasoning model on a Neuron instance! Under the hood, it downloaded a pre-compiled model from Hugging Face to speed up the endpoint start time.

Make sure you delete the endpoint once you finished testing it.

```python
predictor.delete_model()
predictor.delete_endpoint()
```

### Deploy on EC2 Neuron with the Hugging Face Neuron Deep Learning AMI

This guide will detail how to export, deploy and run DeepSeek-R1-Distill-Llama-70B on a `inf2.48xlarge` AWS EC2 Instance.

Before, let’s start with a few pre-requisites. Make sure you have subscribed to the Hugging Face Neuron Deep Learning AMI on the [Marketplace](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2). It provides you all the necessary dependencies to train and deploy Hugging Face models on Trainium & Inferentia. Then, launch an inf2.48xlarge instance in EC2 with the AMI and connect through SSH. You can check our step-by-step [guide](https://huggingface.co/docs/optimum-neuron/en/guides/setup_aws_instance) if you have never done it.

Once connected through the instance, you can deploy the model on an endpoint with this command:

```bash
docker run -p 8080:80 \
    -v $(pwd)/data:/data \
    --device=/dev/neuron0 \
    --device=/dev/neuron1 \
    --device=/dev/neuron2 \
    --device=/dev/neuron3 \
    --device=/dev/neuron4 \
    --device=/dev/neuron5 \
    --device=/dev/neuron6 \
    --device=/dev/neuron7 \
    --device=/dev/neuron8 \
    --device=/dev/neuron9 \
    --device=/dev/neuron10 \
    --device=/dev/neuron11 \
    -e HF_BATCH_SIZE=4 \
    -e HF_SEQUENCE_LENGTH=4096 \
    -e HF_AUTO_CAST_TYPE="bf16" \
    -e HF_NUM_CORES=24 \
    ghcr.io/huggingface/neuronx-tgi:latest \
    --model-id deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --max-batch-size 4 \
    --max-total-tokens 4096
```

It will take a few minutes to download the compiled model from the Hugging Face cache and launch a TGI endpoint.

Then, you can test the endpoint: 

```bash
curl localhost:8080/generate \
    -X POST \
    -d '{"inputs":"Why is the sky dark at night?"}' \
    -H 'Content-Type: application/json'
```

Make sure you pause the EC2 instance once you are done testing it.

| **Note:** The team is working on enabling DeepSeek R1 deployment on Trainium & Inferentia with the Hugging Face Neuron Deep Learning AMI. Stay tuned! 

## Fine-tune DeepSeek R1 models

### Fine tune on Amazon SageMaker AI with Hugging Face Training DLCs

| **Note:** The team is working on enabling all DeepSeek models fine tuning with the Hugging Face Training DLCs. Stay tuned!

### Fine tune on EC2  Neuron with the Hugging Face Neuron Deep Learning AMI

| **Note:** The team is working on enabling all DeepSeek models fine tuning with the Hugging Face Neuron Deep Learning AMI . Stay tuned!

