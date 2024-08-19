---
title: "How to deploy Meta Llama 3.1 405B on Vertex AI" 
thumbnail: /blog/assets/llama31-on-vertex-ai/thumbnail.jpg
authors:
- user: alvarobartt
- user: philschmid
- user: pagezyhf
- user: jeffboudier
---
# How to deploy Meta Llama 3.1 405B on Vertex AI

[Meta Llama 3.1](https://huggingface.co/blog/llama31) is the latest open LLM from Meta, released in July 2024. Meta Llama 3.1 comes in three sizes: 8B for efficient deployment and development on consumer-size GPU, 70B for large-scale AI native applications, and 405B for synthetic data, LLM as a Judge or distillation; among other use cases. Amongst Meta Llama 3.1 new features, the ones to highlight are: a large context length of 128K tokens (vs original 8K), multilingual capabilities, tool usage capabilities, and a more permissive license.

In this blog you will learn how to deploy [`meta-llama/Meta-Llama-3.1-405B-Instruct-FP8`](https://hf.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8) in a Google Cloud A3 node with 8 x H100 NVIDIA GPUs on Vertex AI with [Text Generation Inference](https://github.com/huggingface/text-generation-inference) (TGI) using the Hugging Face purpose-built Deep Learning Containers (DLCs) for Google Cloud.

This blog will cover:

[Introduction to Vertex AI](#introduction-to-vertex-ai)

1. [Requirements for Meta Llama 3.1 Models on Google Cloud](#1-requirements-for-meta-llama-31-models-on-google-cloud)
2. [Setup Google Cloud for Vertex AI](#2-setup-google-cloud-for-vertex-ai)
3. [Register the Meta Llama 3.1 405B Model on Vertex AI](#3-register-the-meta-llama-31-405b-model-on-vertex-ai)
4. [Deploy Meta Llama 3.1 405B on Vertex AI](#4-deploy-meta-llama-31-405b-on-vertex-ai)
5. [Run online predictions with Meta Llama 3.1 405B](#5-run-online-predictions-with-meta-llama-31-405b)
    5.1 [Via Python](#51-via-python)
        5.1.1 [Within the same session](#511-within-the-same-session)
        5.1.2 [From a different session](#512-from-a-different-session)
    5.2 [Via the Vertex AI Online Prediction UI](#52-via-the-vertex-ai-online-prediction-ui)
6. [Clean up resources](#6-clean-up-resources)

[Conclusion](#conclusion)

Lets get started! 🚀

## Introduction to Vertex AI

Vertex AI is a machine learning (ML) platform that lets you train and deploy ML models and AI applications, and customize Large Language Models (LLMs) for use in your AI-powered applications. Vertex AI combines data engineering, data science, and ML engineering workflows, enabling your teams to collaborate using a common tool-set and scale your applications using the benefits of Google Cloud.

This blog will be focused on deploying an already fine-tuned model from the Hugging Face Hub using a pre-built container to get real-time online predictions.

More information at [Vertex AI - Documentation - Introduction to Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform).

## 1. Requirements for Meta Llama 3.1 Models on Google Cloud

Meta Llama 3.1 brings exciting advancements, however, running those requires careful consideration of your hardware resources. For inference, the memory requirements depend on the model size and the precision of the weights. Here's a table showing the approximate memory needed for different configurations:

<table>
  <tr>
   <td><strong>Model Size</strong>
   </td>
   <td><strong>FP16</strong>
   </td>
   <td><strong>FP8</strong>
   </td>
   <td><strong>INT4</strong>
   </td>
  </tr>
  <tr>
   <td>8B
   </td>
   <td>16 GB
   </td>
   <td>8 GB
   </td>
   <td>4 GB
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td>140 GB
   </td>
   <td>70 GB
   </td>
   <td>35 GB
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>810 GB
   </td>
   <td>405 GB
   </td>
   <td>203 GB
   </td>
  </tr>
</table>

_Note: The above-quoted numbers indicate the GPU VRAM required just to load the model checkpoint. They don’t include torch reserved space for kernels or CUDA graphs._

As an example, an A3 instance (with 8 H100s with 80GiB each) has a total of ~640GB of VRAM, so the 405B model would need to be run in a multi-node setup or run at a lower precision (e.g. FP8), which would be the recommended approach. Read more about it in the [Hugging Face Blog for Meta Llama 3.1](https://huggingface.co/blog/llama31#inference-memory-requirements).

The A3 machine series in Google Cloud has 208 vCPUs, and 1,872 GB of memory. This machine series is optimized for compute and memory intensive, network bound ML training, and HPC workloads. Read more about the A3 accelerator-optimized machines with 8 x NVIDIA H100 80GB GPUs availability announcement at [Announcing A3 supercomputers with NVIDIA H100 GPUs, purpose-built for AI](https://cloud.google.com/blog/products/compute/introducing-a3-supercomputers-with-nvidia-h100-gpus) and about the A3 machine series at [Compute Engine - Accelerator-optimized machine family](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-vms).

Even if the A3 accelerator-optimized machines with 8 x NVIDIA H100 80GB GPUs are available within Google Cloud, you will still need to request a custom quota increase in Google Cloud, as those need a specific approval. Note that the A3 accelerator-optimized machines are only available in some zones, so make sure to check the availability of both A3 High or even A3 Mega per zone at [Compute Engine - GPU regions and zones](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones).

In this case, to request a quota increase to use the A3 High GPU machine type you will need to increase the following quotas:

* `Service: Vertex AI API` and `Name: Custom model serving Nvidia H100 80GB GPUs per region` set to **8**
* `Service: Vertex AI API` and `Name: Custom model serving A3 CPUs per region` set to **208**

![A3 Quota Request in Google Cloud](https://raw.githubusercontent.com/alvarobartt/meta-llama-3-1-on-vertex-ai/main/notebooks/meta-llama-3-1-on-vertex-ai/imgs/a3-quota-request.png)

Read more on how to request a quota increase at [Google Cloud Documentation - View and manage quotas](https://cloud.google.com/docs/quotas/view-manage).

## 2. Setup Google Cloud for Vertex AI

Before proceeding, for convenience we will set the following environment variables:

```python
%env PROJECT_ID=your-project-id
%env LOCATION=your-region
```

First you need to install `gcloud` in your machine following the instructions at [Cloud SDK - Install the gcloud CLI](https://cloud.google.com/sdk/docs/install); and log in into your Google Cloud account, setting your project and preferred Google Compute Engine region.

```bash
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud config set compute/region $LOCATION
```

Once the Google Cloud SDK is installed, you need to enable the Google Cloud APIs required to use Vertex AI from a Deep Learning Container (DLC) within their Artifact Registry for Docker.

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable containerfilesystem.googleapis.com
```

Then you will also need to install [`google-cloud-aiplatform`](https://github.com/googleapis/python-aiplatform), required to programmatically interact with Google Cloud Vertex AI from Python.

```bash
pip install --upgrade --quiet google-cloud-aiplatform
```

To then initialize it via Python as follows:

```python
import os
from google.cloud import aiplatform

aiplatform.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))
```

Finally, as the Meta Llama 3.1 models are gated under the [`meta-llama` organization in the Hugging Face Hub](https://hf.co/meta-llama), you will need to request access to it and wait for approval which shouldn't take longer than 24 hours. Then, you need to install the `huggingface_hub` Python SDK to use the `huggingface-cli` to log in into the Hugging Face Hub to download those models.

```bash
pip install --upgrade --quiet huggingface_hub
```

Alternatively, you can also skip the `huggingface_hub` installation and just generate a [Hugging Face Fine-grained Token](https://hf.co/settings/tokens) with read-only permissions for [`meta-llama/Meta-Llama-3.1-405B-Instruct-FP8`](https://hf.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8) or any other model under the [`meta-llama` organization](https://hf.co/meta-llama), to be selected under e.g. `Repository permissions -> meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 -> Read access to contents of selected repos`. And either set that token within the `HF_TOKEN` environment variable or just provide it manually to the `notebook_login` method as follows:

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 3. Register the Meta Llama 3.1 405B Model on Vertex AI

To register the Meta Llama 3.1 405B model on Vertex AI, you will need to use the `google-cloud-aiplatform` Python SDK. But before proceeding you need to first define which DLC are you going to use, which in this case will be the latest Hugging Face TGI DLC for GPU.

As of the current date, August 2024, the latest available Hugging Face TGI DLC, i.e. [us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu121.2-2.ubuntu2204.py310](us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu121.2-2.ubuntu2204.py310), which uses TGI v2.2 that comes with support for the Meta Llama 3.1 architecture as it needs a different RoPE scaling method than its predecessor, Meta Llama 3.

To check which Hugging Face DLCs are available in Google Cloud you can either navigate to [Google Cloud Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) and filter by "huggingface-text-generation-inference", or use the following `gcloud` command:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-text-generation-inference"
```

Then you need to define the configuration for the container, which are the environment variables that the `text-generation-launcher` expects as arguments (as per the [official documentation](https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/launcher)), which in this case are the following:

* `MODEL_ID` the model ID on the Hugging Face Hub, i.e. `meta-llama/Meta-Llama-3.1-405B-Instruct-FP8`.
* `HUGGING_FACE_HUB_TOKEN` the read-access token over the gated repository `meta-llama/Meta-Llama-3.1-405B-Instruct-FP8`, required to download the weights from the Hugging Face Hub.
* `NUM_SHARD` the number of shards to use i.e. the number of GPUs to use, in this case set to 8 as an A3 instance with 8 x H100 NVIDIA GPUs will be used.

Additionally, as a recommendation you should also define `HF_HUB_ENABLE_HF_TRANSFER=1` to enable a faster download speed via the `hf_transfer` utility, as Meta Llama 3.1 405B is around 400 GiB and downloading the weights may take longer otherwise.

Then you can already register the model within Vertex AI's Model Registry via the `google-cloud-aiplatform` Python SDK as follows:

```python
from huggingface_hub import get_token

model = aiplatform.Model.upload(
    display_name="meta-llama--Meta-Llama-3.1-405B-Instruct-FP8",
    serving_container_image_uri="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu121.2-2.ubuntu2204.py310",
    serving_container_environment_variables={
        "MODEL_ID": "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        "HUGGING_FACE_HUB_TOKEN": get_token(),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "NUM_SHARD": "8",
    },
)
model.wait()
```

![Meta Llama 3.1 405B FP8 registered on Vertex AI](https://raw.githubusercontent.com/alvarobartt/meta-llama-3-1-on-vertex-ai/main/notebooks/meta-llama-3-1-on-vertex-ai/imgs/vertex-ai-model.png)

## 4. Deploy Meta Llama 3.1 405B on Vertex AI

Once Meta Llama 3.1 405B is registered on Vertex AI Model Registry, then you can create a Vertex AI Endpoint and deploy the model to the endpoint, with the Hugging Face DLC for TGI as the serving container.

As mentioned before, since Meta Llama 3.1 405B in FP8 takes ~400 GiB of disk space, that means we need at least 400 GiB of GPU VRAM to load the model, and the GPUs within the node need to support the FP8 data type. In this case, an A3 instance with 8 x NVIDIA H100 80GB with a total of \~640 GiB of VRAM will be used to load the model while also leaving some free VRAM for the KV Cache and the CUDA Graphs.

```python
endpoint = aiplatform.Endpoint.create(display_name="Meta-Llama-3.1-405B-FP8-Endpoint")

deployed_model = model.deploy(
    endpoint=endpoint,
    machine_type="a3-highgpu-8g",
    accelerator_type="NVIDIA_H100_80GB",
    accelerator_count=8,
)
```

> Note that the Meta Llama 3.1 405B deployment on Vertex AI may take around 30 minutes to deploy, as it needs to allocate the resources on Google Cloud, and then download the weights from the Hugging Face Hub (\~10 minutes) and load those for inference in TGI.

![Meta Llama 3.1 405B Instruct FP8 deployed on Vertex AI](https://raw.githubusercontent.com/alvarobartt/meta-llama-3-1-on-vertex-ai/main/notebooks/meta-llama-3-1-on-vertex-ai/imgs/vertex-ai-endpoint.png)

Congrats, you already deployed Meta Llama 3.1 405B in your Google Cloud account! 🔥 Now is time to put the model to the test.

## 5. Run online predictions with Meta Llama 3.1 405B

Vertex AI will expose the endpoint `/predict` that is built on top of the `/vertex` endpoint within the Text Generation Inference (TGI) DLC, which runs the standard `/generate` method from TGI, but making sure that the I/O data is compliant with Vertex AI.

As `/generate` is the endpoint that is being exposed, you will need to format the messages with the chat template before sending the request to Vertex AI, so it's recommended to install 🤗`transformers` to use the `apply_chat_template` method from the `PreTrainedTokenizerFast` tokenizer instance.

```bash
pip install --upgrade --quiet transformers
```

And then apply the chat template to a conversation using the tokenizer as follows:

```python
import os
from huggingface_hub import get_token
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    token=get_token(),
)

messages = [
    {"role": "system", "content": "You are an assistant that responds as a pirate."},
    {"role": "user", "content": "What's the Theory of Relativity?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

So that now you have a string out of the initial conversation messages, formatted using the default chat template for Meta Llama 3.1, so that the above produces:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an assistant that responds as a pirate.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's the Theory of Relativity?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

Which is what you will be sending within the payload to the deployed Vertex AI Endpoint, as well as the generation arguments as in [Consuming Text Generation Inference (TGI) -> Generate](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_generation).

### 5.1 Via Python

#### 5.1.1 Within the same session

If you are willing to run the online prediction within the current session i.e. the same one as the one used to deploy the model, you can send requests programmatically via the `aiplatform.Endpoint` returned as of the `aiplatform.Model.deploy` method as in the following snippet.

```python
output = deployed_model.predict(
    instances=[
        {
            "inputs": inputs,
            "parameters": {
                "max_new_tokens": 128,
                "do_sample": True,
                "top_p": 0.95,
                "temperature": 0.7,
            },
        },
    ]
)
```

Producing the following `output`:

```
Prediction(predictions=["Yer want ta know about them fancy science things, eh? Alright then, matey, settle yerself down with a pint o' grog and listen close. I be tellin' ye about the Theory o' Relativity, as proposed by that swashbucklin' genius, Albert Einstein.\n\nNow, ye see, Einstein said that time and space be connected like the sea and the wind. Ye can't have one without the other, savvy? And he proposed that how ye see time and space depends on how fast ye be movin' and where ye be standin'. That be called relativity, me"], deployed_model_id='***', metadata=None, model_version_id='1', model_resource_name='projects/***/locations/***/models/***', explanations=None)
```

#### 5.1.2 From a different session

If the Vertex AI Endpoint was deployed in a different session and you just want to use it, but don't have access to the `deployed_model` variable returned by the `aiplatform.Model.deploy` method, then you can also run the following snippet to instantiate the deployed `aiplatform.Endpoint` via its resource name that can be found either within the Vertex AI Online Prediction UI, from the `aiplatform.Endpoint` instantiated above, or just replacing the values in `projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}`.

```python
import os
from google.cloud import aiplatform

aiplatform.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))

endpoint = aiplatform.Endpoint(f"projects/{os.getenv('PROJECT_ID')}/locations/{os.getenv('LOCATION')}/endpoints/{ENDPOINT_ID}")
output = endpoint.predict(
    instances=[
        {
            "inputs": inputs,
            "parameters": {
                "max_new_tokens": 128,
                "do_sample": True,
                "top_p": 0.95,
                "temperature": 0.7,
            },
        },
    ],
)
```

Producing the following `output`:

```
Prediction(predictions=["Yer lookin' fer a treasure trove o' knowledge about them fancy physics, eh? Alright then, matey, settle yerself down with a pint o' grog and listen close, as I spin ye the yarn o' Einstein's Theory o' Relativity.\n\nIt be a tale o' two parts, me hearty: Special Relativity and General Relativity. Now, I know what ye be thinkin': what in blazes be the difference? Well, matey, let me break it down fer ye.\n\nSpecial Relativity be the idea that time and space be connected like the sea and the sky."], deployed_model_id='***', metadata=None, model_version_id='1', model_resource_name='projects/***/locations/***/models/***', explanations=None)
```

### 5.2 Via the Vertex AI Online Prediction UI

Alternatively, for testing purposes you can also use the Vertex AI Online Prediction UI, that provides a field that expects the JSON payload formatted according to the Vertex AI specification (as in the examples above) being:

```json
{
    "instances": [
        {
            "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an assistant that responds as a pirate.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's the Theory of Relativity?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "parameters": {
                "max_new_tokens": 128,
                "do_sample": true,
                "top_p": 0.95,
                "temperature": 0.7
            }
        }
    ]
}
```

So that the output is generated and printed within the UI too.

![Meta Llama 3.1 405B Instruct FP8 online prediction on Vertex AI](https://raw.githubusercontent.com/alvarobartt/meta-llama-3-1-on-vertex-ai/main/notebooks/meta-llama-3-1-on-vertex-ai/imgs/vertex-ai-online-prediction.png)

## 6. Clean up resources

Finally, you can already release the resources that you've created as follows, to avoid unnecessary costs:

* `deployed_model.undeploy_all` to undeploy the model from all the endpoints.
* `deployed_model.delete` to delete the endpoint/s where the model was deployed gracefully, after the `undeploy_all` method.
* `model.delete` to delete the model from the registry.

```python
deployed_model.undeploy_all()
deployed_model.delete()
model.delete()
```

Alternatively, you can also remove those from the Google Cloud Console following the steps:

* Go to Vertex AI in Google Cloud
* Go to Deploy and use -> Online prediction
* Click on the endpoint and then on the deployed model/s to "Undeploy model from endpoint"
* Then go back to the endpoint list and remove the endpoint
* Finally, go to Deploy and use -> Model Registry, and remove the model

## Conclusion

That's it! You have already registered and deployed Meta Llama 3.1 405B Instruct FP8 on Google Cloud Vertex AI, then ran online prediction both programmatically and via the Google Cloud Console, and finally cleaned up the resources used to avoid unnecessary costs.

Thanks to the Hugging Face DLCs for Text Generation Inference (TGI), and Google Cloud Vertex AI, deploying a high-performance text generation container for serving Large Language Models (LLMs) has never been easier. And we’re not going to stop here – stay tuned as we enable more experiences to build AI with open models on Google Cloud!
