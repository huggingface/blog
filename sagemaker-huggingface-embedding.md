---
title:  Introducing the Hugging Face Embedding Container for Amazon SageMaker
thumbnail: /blog/assets/sagemaker-huggingface-embedding/thumbnail.jpg
authors:
- user: philschmid
- user: jeffboudier
---

# Introducing the Hugging Face Embedding Container for Amazon SageMaker

We are excited to announce that the new Hugging Face Embedding Container for Amazon SageMaker is now generally available (GA). AWS customers can now efficiently deploy embedding models on SageMaker to build Generative AI applications, including Retrieval-Augmented Generation (RAG) applications. 

In this Blog we will show you how to deploy open Embedding Models, like [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l), [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) or [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) to Amazon SageMaker for inference using the new Hugging Face Embedding Container. We will deploy the [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) one of the best open Embedding Models for retrieval - you can check its rankings on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard). 

The example covers:
- [What is the Hugging Face Embedding Container?](#what-is-hugging-face-embedding-container)
- [1. Setup development environment](#1-setup-development-environment)
- [2. Retrieve the new Hugging Face Embedding Container](#2-retrieve-the-new-hugging-face-embedding-container)
- [3. Deploy Snowflake Arctic to Amazon SageMaker](#3-deploy-snowflake-arctic-to-amazon-sagemaker)
- [4. Run and evaluate Inference performance](#4-run-and-evaluate-inference-performance)
- [5. Delete model and endpoint](#5-delete-model-and-endpoint)

## What is the Hugging Face Embedding Container?

The Hugging Face Embedding Container is a new purpose-built Inference Container to easily deploy Embedding Models in a secure and managed environment. The DLC is powered by [Text Embedding Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) a blazing fast and memory efficient solution for deploying and serving Embedding Models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5. TEI implements many features such as:

* No model graph compilation step
* Small docker images and fast boot times
* Token based dynamic batching
* Optimized transformers code for inference using Flash Attention, Candle and cuBLASLt
* Safetensors weight loading
* Production ready (distributed tracing with Open Telemetry, Prometheus metrics)

TEI supports the following model architectures
* BERT/CamemBERT, e.g. [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) or [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m)
* RoBERTa, [sentence-transformers/all-roberta-large-v1](https://huggingface.co/sentence-transformers/all-roberta-large-v1) 
* XLM-RoBERTa, e.g. [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)
* NomicBert, e.g. [jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)
* JinaBert, e.g. [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

Lets get started!


## 1. Setup development environment

We are going to use the `sagemaker` python SDK to deploy Snowflake Arctic to Amazon SageMaker. We need to make sure to have an AWS account configured and the `sagemaker` python SDK installed. 


```python
!pip install "sagemaker>=2.221.1" --upgrade --quiet
```

If you are going to use Sagemaker in a local environment, you need access to an IAM Role with the required permissions for Sagemaker. You can find out more about it [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).


```python
import sagemaker
import boto3
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it does not exist
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker session region: {sess.boto_region_name}")

```

## 2. Retrieve the new Hugging Face Embedding Container

Compared to deploying regular Hugging Face models we first need to retrieve the container uri and provide it to our `HuggingFaceModel` model class with a `image_uri` pointing to the image. To retrieve the new Hugging Face Embedding Container in Amazon SageMaker, we can use the `get_huggingface_llm_image_uri` method provided by the `sagemaker` SDK. This method allows us to retrieve the URI for the desired Hugging Face Embedding Container. Important to note is that TEI has 2 different versions for cpu and gpu, so we create a helper function to retrieve the correct image uri based on the instance type. 


```python
from sagemaker.huggingface import get_huggingface_llm_image_uri

# retrieve the image uri based on instance type
def get_image_uri(instance_type):
  key = "huggingface-tei" if instance_type.startswith("ml.g") or instance_type.startswith("ml.p") else "huggingface-tei-cpu"
  return get_huggingface_llm_image_uri(key, version="1.2.3")

```

## 3. Deploy Snowflake Arctic to Amazon SageMaker

To deploy  [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) to Amazon SageMaker we create a `HuggingFaceModel` model class and define our endpoint configuration including the `HF_MODEL_ID`, `instance_type` etc. We will use a `c6i.2xlarge` instance type, which has 4 Intel Ice-Lake vCPUs, 8GB of memory and costs around $0.204 per hour. 


```python
import json
from sagemaker.huggingface import HuggingFaceModel

# sagemaker config
instance_type = "ml.g5.xlarge"

# Define Model and Endpoint configuration parameter
config = {
  'HF_MODEL_ID': "Snowflake/snowflake-arctic-embed-m", # model_id from hf.co/models
}

# create HuggingFaceModel with the image uri
emb_model = HuggingFaceModel(
  role=role,
  image_uri=get_image_uri(instance_type),
  env=config
)
```

After we have created the `HuggingFaceModel` we can deploy it to Amazon SageMaker using the `deploy` method. We will deploy the model with the `ml.c6i.2xlarge` instance type.


```python
# Deploy model to an endpoint
# https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
emb = emb_model.deploy(
  initial_instance_count=1,
  instance_type=instance_type,
)

```

SageMaker will now create our endpoint and deploy the model to it. This can take  ~5 minutes. 

## 4. Run and evaluate Inference performance

After our endpoint is deployed we can run inference on it. We will use the `predict` method from the `predictor` to run inference on our endpoint.



```python
data = {
  "inputs": "the mesmerizing performances of the leads keep the film grounded and keep the audience riveted .",
}
 
res = emb.predict(data=data)
 
 
# print some results
print(f"length of embeddings: {len(res[0])}")
print(f"first 10 elements of embeddings: {res[0][:10]}")
```

Awesome we can now generate embeddings with our model, Lets test the performance of our model.

We will send 3,900 requests to our endpoint use threading with 10 concurrent threads. We will measure the average latency and throughput of our endpoint. We are going to sent an input of 256 tokens to have a total of ~1 Million tokens. We decided to use 256 tokens as input length to find the balance between shorter and longer inputs.

Note: When running the load test, the requests are sent from europe and the endpoint is deployed in us-east-1. This adds a network overhead to it.


```python
import threading
import time
number_of_threads = 10
number_of_requests = int(3900 // number_of_threads)
print(f"number of threads: {number_of_threads}")
print(f"number of requests per thread: {number_of_requests}")
 
def send_requests():
    for _ in range(number_of_requests):
        # input counted at https://huggingface.co/spaces/Xenova/the-tokenizer-playground for 100 tokens
        emb.predict(data={"inputs": "Hugging Face is a company and a popular platform in the field of natural language processing (NLP) and machine learning. They are known for their contributions to the development of state-of-the-art models for various NLP tasks and for providing a platform that facilitates the sharing and usage of pre-trained models. One of the key offerings from Hugging Face is the Transformers library, which is an open-source library for working with a variety of pre-trained transformer models, including those for text generation, translation, summarization, question answering, and more. The library is widely used in the research and development of NLP applications and is supported by a large and active community. Hugging Face also provides a model hub where users can discover, share, and download pre-trained models. Additionally, they offer tools and frameworks to make it easier for developers to integrate and use these models in their own projects. The company has played a significant role in advancing the field of NLP and making cutting-edge models more accessible to the broader community. Hugging Face also provides a model hub where users can discover, share, and download pre-trained models. Additionally, they offer tools and frameworks to make it easier for developers and ma"})
 
# Create multiple threads
threads = [threading.Thread(target=send_requests) for _ in range(number_of_threads) ]
# start all threads
start = time.time()
[t.start() for t in threads]
# wait for all threads to finish
[t.join() for t in threads]
print(f"total time: {round(time.time() - start)} seconds")
```

Sending 3,900 requests or embedding 1 million tokens took around 841 seconds. This means we can run around ~5 requests per second. But keep in mind that includes the network latency from europe to us-east-1. When we inspect the latency of the endpoint through cloudwatch we can see that latency for our Embeddings model is 2s at 10 concurrent requests. This is very impressive for a small & old CPU instance, which cost ~150$ per month. You can deploy the model to a GPU instance to get faster inference times.

_Note: We ran the same test on a `ml.g5.xlarge` with 1x NVIDIA A10G GPU. Embedding 1 million tokens took around 30 seconds. This means we can run around ~130 requests per second. The latency for the endpoint is 4ms at 10 concurrent requests. The `ml.g5.xlarge` costs around $1.408 per hour on Amazon SageMaker._

GPU instances are much faster than CPU instances, but they are also more expensive. If you want to bulk process embeddings, you can use a GPU instance. If you want to run a small endpoint with low costs, you can use a CPU instance. We plan to work on a dedicated benchmark for the Hugging Face Embedding Container in the future.


```python
print(f"https://console.aws.amazon.com/cloudwatch/home?region={sess.boto_region_name}#metricsV2:graph=~(metrics~(~(~'AWS*2fSageMaker~'ModelLatency~'EndpointName~'{emb.endpoint_name}~'VariantName~'AllTraffic))~view~'timeSeries~stacked~false~region~'{sess.boto_region_name}~start~'-PT5M~end~'P0D~stat~'Average~period~30);query=~'*7bAWS*2fSageMaker*2cEndpointName*2cVariantName*7d*20{emb.endpoint_name}")

```

![cw](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/cw.png)

## 5. Delete model and endpoint

To clean up, we can delete the model and endpoint


```python
emb.delete_model()
emb.delete_endpoint()
```

## Conclusion

The new Hugging Face Embedding Container allows you to easily deploy open Embedding Models such as [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l) to Amazon SageMaker for inference. We walked through setting up the development environment, retrieving the container, deploying the model, and evaluating its inference performance.

With this new container, customers can now easily deploy high-performance embedding models, enabling the creation of sophisticated Generative AI applications with improved efficiency. We are excited to see what you build with the new Hugging Face Embedding Container for Amazon SageMaker. If you have any questions or feedback, please let us know. 

 