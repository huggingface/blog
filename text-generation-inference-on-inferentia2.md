---
title: "Hugging Face Text Generation Inference available for AWS Inferentia2" 
thumbnail: /blog/assets/175_text_generation_inference_on_inferentia2/thumbnail.jpg
authors:
- user: philschmid
- user: dacorvo
---

# Hugging Face Text Generation Inference available for AWS Inferentia2

We are excited to announce the general availability of Hugging Face Text Generation Inference (TGI) on AWS Inferentia2 and Amazon SageMaker.  

**[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)**, is a purpose-built solution for deploying and serving Large Language Models (LLMs) for production workloads at scale. TGI enables high-performance text generation using Tensor Parallelism and continuous batching for the most popular open LLMs, including Llama, Mistral, and more. Text Generation Inference is used in production by companies such as Grammarly, Uber, Deutsche Telekom, and many more.  

The integration of TGI into Amazon SageMaker, in combination with AWS Inferentia2, presents a powerful solution and viable alternative to GPUs for building production LLM applications. The seamless integration ensures easy deployment and maintenance of models, making LLMs more accessible and scalable for a wide range of production use cases.

With the new TGI for AWS Inferentia2 on Amazon SageMaker, AWS customers can benefit from the same technologies that power highly-concurrent, low-latency LLM experiences like [HuggingChat](https://hf.co/chat), [OpenAssistant](https://open-assistant.io/), and Serverless Endpoints for LLMs on the Hugging Face Hub.

## Deploy Zephyr 7B on AWS Inferentia2 using Amazon SageMaker

This tutorial shows how easy it is to deploy a state-of-the-art LLM, such as Zephyr 7B, on AWS Inferentia2 using Amazon SageMaker. Zephyr is a 7B fine-tuned version of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) that was trained on a mix of publicly available and synthetic datasets using [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290), as described in detail in the [technical report](https://arxiv.org/abs/2310.16944). The model is released under the Apache 2.0 license, ensuring wide accessibility and use.

We are going to show you how to:

1. Setup development environment
2. Retrieve the TGI Neuronx Image
3. Deploy Zephyr 7B to Amazon SageMaker
4. Run inference and chat with the model

Let’s get started.

### 1. Setup development environment

We are going to use the `sagemaker` python SDK to deploy Zephyr to Amazon SageMaker. We need to make sure to have an AWS account configured and the `sagemaker` python SDK installed.

```python
!pip install transformers "sagemaker>=2.206.0" --upgrade --quiet
```

If you are going to use Sagemaker in a local environment. You need access to an IAM Role with the required permissions for Sagemaker. You can find out more about it [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

```python
import sagemaker
import boto3
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it doesn't exist
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

### 2. Retrieve TGI Neuronx Image

The new Hugging Face TGI Neuronx DLCs can be used to run inference on AWS Inferentia2. You can use the `get_huggingface_llm_image_uri` method of the `sagemaker` SDK to retrieve the appropriate Hugging Face TGI Neuronx DLC URI based on your desired `backend`, `session`, `region`, and `version`. You can find all the available versions [here](https://github.com/aws/deep-learning-containers/releases?q=tgi+AND+neuronx&expanded=true).

*Note: At the time of writing this blog post the latest version of the Hugging Face LLM DLC is not yet available via the `get_huggingface_llm_image_uri` method. We are going to use the raw container uri instead.*

```python
from sagemaker.huggingface import get_huggingface_llm_image_uri

# retrieve the llm image uri
llm_image = get_huggingface_llm_image_uri(
  "huggingface-neuronx",
  version="0.0.20"
)

# print ecr image uri
print(f"llm image uri: {llm_image}")
```

### 4. Deploy Zephyr 7B to Amazon SageMaker

Text Generation Inference (TGI) on Inferentia2 supports popular open LLMs, including Llama, Mistral, and more. You can check the full list of supported models (text-generation) [here](https://huggingface.co/docs/optimum-neuron/package_reference/export#supported-architectures).

**Compiling LLMs for Inferentia2**  

At the time of writing, [AWS Inferentia2 does not support dynamic shapes for inference](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.6.0/general/arch/neuron-features/dynamic-shapes.html#neuron-dynamic-shapes), which means that we need to specify our sequence length and batch size ahead of time.
To make it easier for customers to utilize the full power of Inferentia2, we created a [neuron model cache](https://huggingface.co/docs/optimum-neuron/guides/cache_system), which contains pre-compiled configurations for the most popular LLMs. A cached configuration is defined through a model architecture (Mistral), model size (7B), neuron version (2.16), number of inferentia cores (2), batch size (2), and sequence length (2048).

This means we don't need to compile the model ourselves, but we can use the pre-compiled model from the cache. Examples of this are [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) and [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta). You can find compiled/cached configurations on the [Hugging Face Hub](https://huggingface.co/aws-neuron/optimum-neuron-cache/tree/main/inference-cache-config). If your desired configuration is not yet cached, you can compile it yourself using the [Optimum CLI](https://huggingface.co/docs/optimum-neuron/cli/compile) or open a request at the [Cache repository](https://huggingface.co/aws-neuron/optimum-neuron-cache/discussions) 

For this post we re-compiled `HuggingFaceH4/zephyr-7b-beta` using the following command and parameters on a `inf2.8xlarge` instance, and pushed it to the Hub at [aws-neuron/zephyr-7b-seqlen-2048-bs-4-cores-2](https://huggingface.co/aws-neuron/zephyr-7b-seqlen-2048-bs-4-cores-2)

```bash
# compile model with optimum for batch size 4 and sequence length 2048
optimum-cli export neuron -m HuggingFaceH4/zephyr-7b-beta --batch_size 4 --sequence_length 2048 --num_cores 2 --auto_cast_type bf16 ./zephyr-7b-beta-neuron
# push model to hub [repo_id] [local_path] [path_in_repo]
huggingface-cli upload  aws-neuron/zephyr-7b-seqlen-2048-bs-4 ./zephyr-7b-beta-neuron ./ --exclude "checkpoint/**"
# Move tokenizer to neuron model repository
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta').push_to_hub('aws-neuron/zephyr-7b-seqlen-2048-bs-4')"
```

If you are trying to compile an LLM with a configuration that is not yet cached, it can take up to 45 minutes.

**Deploying TGI Neuronx Endpoint**  

Before deploying the model to Amazon SageMaker, we must define the TGI Neuronx endpoint configuration. We need to make sure to set the following parameters according to the fixed-shape compilation parameters we used:

Before deploying the model to Amazon SageMaker, we must define the TGI Neuronx endpoint configuration. We need to make sure the following additional parameters are defined: 

- `HF_NUM_CORES`: Number of Neuron Cores used for the compilation.
- `HF_BATCH_SIZE`: The batch size that was used to compile the model.
- `HF_SEQUENCE_LENGTH`: The sequence length that was used to compile the model.
- `HF_AUTO_CAST_TYPE`: The auto cast type that was used to compile the model.

We still need to define traditional TGI parameters with:

- `HF_MODEL_ID`: The Hugging Face model ID.
- `HF_TOKEN`: The Hugging Face API token to access gated models.
- `MAX_BATCH_SIZE`: The maximum batch size that the model can handle, equal to the batch size used for compilation.
- `MAX_INPUT_LENGTH`: The maximum input length that the model can handle. 
- `MAX_TOTAL_TOKENS`: The maximum total tokens the model can generate, equal to the sequence length used for compilation.

```python
import json
from sagemaker.huggingface import HuggingFaceModel

# sagemaker config & model config
instance_type = "ml.inf2.8xlarge"
health_check_timeout = 1800

# Define Model and Endpoint configuration parameter
config = {
    "HF_MODEL_ID": "HuggingFaceH4/zephyr-7b-beta",
    "HF_NUM_CORES": "2",
    "HF_BATCH_SIZE": "4",
    "HF_SEQUENCE_LENGTH": "2048",
    "HF_AUTO_CAST_TYPE": "bf16",  
    "MAX_BATCH_SIZE": "4",
    "MAX_INPUT_LENGTH": "1512",
    "MAX_TOTAL_TOKENS": "2048",
}

# create HuggingFaceModel with the image uri
llm_model = HuggingFaceModel(
  role=role,
  image_uri=llm_image,
  env=config
)
```

After we have created the `HuggingFaceModel` we can deploy it to Amazon SageMaker using the `deploy` method. We will deploy the model with the `ml.inf2.8xlarge` instance type.

```python
# Deploy model to an endpoint
llm = llm_model.deploy(
  initial_instance_count=1,
  instance_type=instance_type,
  container_startup_health_check_timeout=health_check_timeout,
)
```

SageMaker will create our endpoint and deploy the model to it. This can take 10-15 minutes.

### 5. Run inference and chat with the model

After our endpoint is deployed, we can run inference on it, using the `predict` method from `predictor`. We can provide different parameters to impact the generation, adding them to the `parameters` attribute of the payload. You can find the supported parameters [here](https://www.philschmid.de/sagemaker-llama-llm#5-run-inference-and-chat-with-the-model), or in the open API specification of TGI in the [swagger documentation](https://huggingface.github.io/text-generation-inference/)

The `HuggingFaceH4/zephyr-7b-beta` is a conversational chat model, meaning we can chat with it using a prompt structure like the following:

```
<|system|>\nYou are a friendly.</s>\n<|user|>\nInstruction</s>\n<|assistant|>\n
```

Manually preparing the prompt is error prone, so we can use the `apply_chat_template` method from the tokenizer to help with it. It expects a `messages` dictionary in the well-known OpenAI format, and converts it into the correct format for the model. Let's see if Zephyr knows some facts about AWS.

```python
from transformers import AutoTokenizer

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("aws-neuron/zephyr-7b-seqlen-2048-bs-4-cores-2")

# Prompt to generate
messages = [
    {"role": "system", "content": "You are the AWS expert"},
    {"role": "user", "content": "Can you tell me an interesting fact about AWS?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generation arguments
payload = {
    "do_sample": True,
    "top_p": 0.6,
    "temperature": 0.9,
    "top_k": 50,
    "max_new_tokens": 256,
    "repetition_penalty": 1.03,
    "return_full_text": False,
    "stop": ["</s>"]
}
chat = llm.predict({"inputs":prompt, "parameters":payload})

print(chat[0]["generated_text"][len(prompt):])
# Sure, here's an interesting fact about AWS: As of 2021, AWS has more than 200 services in its portfolio, ranging from compute power and storage to databases,

```

Awesome, we have successfully deployed Zephyr to Amazon SageMaker on Inferentia2 and chatted with it.

### 6. Clean up

To clean up, we can delete the model and endpoint.

```python
llm.delete_model()
llm.delete_endpoint()
```

## Conclusion

The integration of Hugging Face Text Generation Inference (TGI) with AWS Inferentia2 and Amazon SageMaker provides a cost-effective alternative solution for deploying Large Language Models (LLMs).

We're actively working on supporting more models, streamlining the compilation process, and refining the caching system.

---

Thanks for reading! If you have any questions, feel free to contact me on [Twitter](https://twitter.com/_philschmid) or [LinkedIn](https://www.linkedin.com/in/philipp-schmid-a6a2bb196/).
