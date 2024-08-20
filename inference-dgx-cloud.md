---
title: "Serverless Inference with Hugging Face and NVIDIA NIM" 
thumbnail: /blog/assets/train-dgx-cloud/thumbnail.jpg
authors:
- user: philschmid
- user: jeffboudier
---

# Serverless Inference with Hugging Face and NVIDIA NIM

Today, we are thrilled to announce the launch of **Hugging Face** **NVIDIA NIM API (serverless)**, a new service on the Hugging Face Hub, available to Enterprise Hub organizations. This new service makes it easy to use open models with the accelerated compute platform, of [NVIDIA DGX Cloud](https://www.nvidia.com/en-us/data-center/dgx-cloud) accelerated compute platform for inference serving. We built this solution so that Enterprise Hub users can easily access the latest NVIDIA AI technology in a serverless way to run inference on popular Generative AI models including Llama and Mistral, using standardized APIs and a few lines of code within the[ Hugging Face Hub](https://huggingface.co/models).


<div align="center"> 
  <img src="/blog/assets/train-dgx-cloud/thumbnail.jpg" alt="Thumbnail"> 
</div>


## Serverless Inference powered by NVIDIA NIM

This new experience builds on our[ collaboration with NVIDIA](https://nvidianews.nvidia.com/news/nvidia-and-hugging-face-to-connect-millions-of-developers-to-generative-ai-supercomputing) to simplify the access and use of open Generative AI models on NVIDIA accelerated computing. One of the main challenges developers and organizations face is the upfront cost of infrastructure and the complexity of optimizing inference workloads for LLM. With Hugging Face NVIDIA NIM API (serverless), we offer an easy solution to these challenges, providing instant access to state-of-the-art open Generative AI models optimized for NVIDIA infrastructure with a simple API for running inference. The pay-as-you-go pricing model ensures that you only pay for the request time you use, making it an economical choice for businesses of all sizes.

NVIDIA NIM API (serverless) complements [Train on DGX Cloud](https://huggingface.co/blog/train-dgx-cloud), an AI training service already available on Hugging Face. 


## How it works

Running serverless inference with Hugging Face models has never been easier. Here’s a step-by-step guide to get you started:

_Note: You need access to an Organization with a [Hugging Face Enterprise Hub](https://huggingface.co/enterprise) subscription to run Inference._

Before you begin, ensure you meet the following requirements:
1. You are member of an Enterprise Hub organization.
2. You have created a fine-grained token for your organization. Follow the steps below to create your token.


### Create a Fine-Grained Token

Fine-grained tokens allow users to create tokens with specific permissions for precise access control to resources and namespaces. First, go to[ Hugging Face Access Tokens](https://huggingface.co/settings/tokens) and click on “Create new Token” and select “fine-grained”.

<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-dgx-cloud/fine-grained-token-1.png" alt="Create Token"> 
</div>



Enter a “Token name” and select your Enterprise organization in “org permissions” as scope and then click “Create token”. You don’t need to select any additional scopes. 


<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-dgx-cloud/fine-grained-token-2.png" alt="Scope Token"> 
</div>


Now, make sure to save this token value to authenticate your requests later.


### **Find your NIM**

You can find “NVIDIA NIM API (serverless)” on the model page of supported Generative AI models. You can find all supported models in this [NVIDIA NIM Collection](https://huggingface.co/collections/nvidia/nim-66a3c6fcdcb5bbc6e975b508), and in the Pricing section. 

We will use the `meta-llama/Meta-Llama-3-8B-Instruct`. Go the [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model card open “Deploy” menu, and select “NVIDIA NIM API (serverless)” - this will open an interface with pre-generated code snippets for Python, Javascript or Curl. 



<div align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-dgx-cloud/inference-modal.png" alt="inference-modal"> 
</div>


### **Send your requests**

NVIDIA NIM API (serverless) is standardized on the OpenAI API. This allows you to use the `openai’` sdk for inference. Replace the `YOUR_FINE_GRAINED_TOKEN_HERE` with your fine-grained token and you are ready to run inference. 

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://huggingface.co/api/integrations/dgx/v1",
    api_key="YOUR_FINE_GRAINED_TOKEN_HERE"
)

chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count to 500"}
    ],
    stream=True,
    max_tokens=1024
)

# Iterate and print stream
for message in chat_completion:
    print(message.choices[0].delta.content, end='')
```

Congrats! 🎉 You can now start building your Generative AI applications using open models. 🔥

NVIDIA NIM API (serverless) currently only supports the `chat.completions.create` and `models.list` API. We are working on extending this while adding more models. The `models.list` can be used to check which models are currently available for Inference. 

```python
models = client.models.list()
for m in models.data:
    print(m.id)
```


## Supported Models and Pricing 

Usage of Hugging Face NVIDIA NIM API (serverless) is billed based on the compute time spent per request. We exclusively use NVIDIA H100 Tensor Core GPUs, which are priced at $8.25 per hour. To make this easier to understand for per-request pricing, we can convert this to a per-second.

$8.25 per hour = $0.0023 per second (rounded to 4 decimal places)

The total cost for a request will depend on the model size, the number of GPUs required, and the time taken to process the request. Here's a breakdown of our current model offerings, their GPU requirements, typical response times, and estimated cost per request:


<table>
  <tr>
   <td><strong>Model ID</strong>
   </td>
   <td><strong>Number of NVIDIA H100 GPUs</strong>
   </td>
   <td><strong>Typical Response Time (500 input tokens, 100 output tokens)</strong>
   </td>
   <td><strong>Estimated Cost per Request</strong>
   </td>
  </tr>
  <tr>
   <td>meta-llama/Meta-Llama-3-8B-Instruct
   </td>
   <td>1
   </td>
   <td>1 seconds
   </td>
   <td>$0.0023
   </td>
  </tr>
  <tr>
   <td>meta-llama/Meta-Llama-3-70B-Instruct
   </td>
   <td>4
   </td>
   <td>2 seconds
   </td>
   <td>$0.0184
   </td>
  </tr>
  <tr>
   <td>meta-llama/Meta-Llama-3.1-405B-Instruct-FP8
   </td>
   <td>8
   </td>
   <td>5 seconds
   </td>
   <td>$0.0917
   </td>
  </tr>
</table>


Usage fees accrue to your Enterprise Hub Organizations’ current monthly billing cycle. You can check your current and past usage at any time within the billing settings of your Enterprise Hub Organization.

**Supported Models **


<table>
  <tr>
   <td><strong>Model ID</strong>
   </td>
   <td><strong>Number of H100 GPUs</strong>
   </td>
  </tr>
  <tr>
   <td><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1</a>
   </td>
   <td>8
   </td>
  </tr>
  <tr>
   <td><a href="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1">mistralai/Mixtral-8x7B-Instruct-v0.1</a>
   </td>
   <td>2
   </td>
  </tr>
  <tr>
   <td><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3</a>
   </td>
   <td>2
   </td>
  </tr>
  <tr>
   <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct">meta-llama/Meta-Llama-3.1-70B-Instruct</a>
   </td>
   <td>4
   </td>
  </tr>
  <tr>
   <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct</a>
   </td>
   <td>1
   </td>
  </tr>
  <tr>
   <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct">meta-llama/Meta-Llama-3-8B-Instruct</a>
   </td>
   <td>1
   </td>
  </tr>
  <tr>
   <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct</a>
   </td>
   <td>4
   </td>
  </tr>
  <tr>
   <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8">meta-llama/Meta-Llama-3.1-405B-Instruct-FP8</a>
   </td>
   <td>8
   </td>
  </tr>
</table>



## Accelerating AI Inference with NVIDIA TensorRT-LLM

We are excited to continue our collaboration with NVIDIA to push the boundaries of AI inference performance and accessibility. A key focus of our ongoing efforts is the integration of the NVIDIA TensorRT-LLM library into Hugging Face's Text Generation Inference (TGI) framework.

We'll be sharing more details, benchmarks, and best practices for using TGI with NVIDIA TensorRT-LLM in the near future. Stay tuned for more exciting developments as we continue to expand our collaboration with NVIDIA and bring more powerful AI capabilities to developers and organizations worldwide!
