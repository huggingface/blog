---
title: "Deploy GPT-J 6B for inference using  Hugging Face Transformers and Amazon SageMaker"
thumbnail: /blog/assets/45_gptj_sagemaker/thumbnail.png
---

<h1>Deploy GPT-J 6B for inference using  Hugging Face Transformers and Amazon SageMaker</h1>

<div class="blog-metadata">
    <small>Published Jan 11, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/gptj-sagemaker.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/philschmid">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1624629516652-5ff5d596f244529b3ec0fb89.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>philschmid</code>
            <span class="fullname">Philipp Schmid</span>
        </div>
    </a>
</div>

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>


Almost 6 months ago to the day, [EleutherAI](https://www.eleuther.ai/) released [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B), an open-source alternative to [OpenAIs GPT-3](https://openai.com/blog/gpt-3-apps/).  [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) is the 6 billion parameter successor to [EleutherAIs](https://www.eleuther.ai/) GPT-NEO family, a family of transformer-based language models based on the GPT architecture for text generation.

[EleutherAI](https://www.eleuther.ai/)'s primary goal is to train a model that is equivalent in size to GPT⁠-⁠3 and make it available to the public under an open license. 

Over the last 6 months, `GPT-J` gained a lot of interest from Researchers, Data Scientists, and even Software Developers, but it remained very challenging to deploy `GPT-J` into production for real-world use cases and products. 

There are some hosted solutions to use `GPT-J` for production workloads, like the [Hugging Face Inference API](https://huggingface.co/inference-api), or for experimenting using  [EleutherAIs 6b playground](https://6b.eleuther.ai/), but fewer examples on how to easily deploy it into your own environment. 

In this blog post, you will learn how to easily deploy `GPT-J` using [Amazon SageMaker](https://aws.amazon.com/de/sagemaker/) and the [Hugging Face Inference Toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit) with a few lines of code for scalable, reliable, and secure real-time inference using a regular size GPU instance with NVIDIA T4 (~500$/m). 

But before we get into it, I want to explain why deploying `GPT-J` into production is challenging. 

---

## Background

The weights of the 6 billion parameter model represent a ~24GB memory footprint. To load it in float32, one would need at least 2x model size CPU RAM: 1x for initial weights and another 1x to load the checkpoint. So for `GPT-J` it would require at least 48GB of CPU RAM to just load the model.

To make the model more accessible, [EleutherAI](https://www.eleuther.ai/) also provides float16 weights, and `transformers` has new options to reduce the memory footprint when loading large language models. Combining all this it should take roughly 12.1GB of CPU RAM to load the model.

```python
from transformers import GPTJForCausalLM
import torch

model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
		revision="float16",
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True
)
```

The caveat of this example is that it takes a very long time until the model is loaded into memory and ready for use. In my experiments, it took `3 minutes and 32 seconds` to load the model with the code snippet above on a `P3.2xlarge` AWS EC2 instance (the model was not stored on disk). This duration can be reduced by storing the model already on disk, which reduces the load time to `1 minute and 23 seconds`, which is still very long for production workloads where you need to consider scaling and reliability. 

For example, Amazon SageMaker has a [60s limit for requests to respond](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html#sagemaker_region), meaning the model needs to be loaded and the predictions to run within 60s, which in my opinion makes a lot of sense to keep the model/endpoint scalable and reliable for your workload. If you have longer predictions, you could use [batch-transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html).

In [Transformers](https://github.com/huggingface/transformers) the models loaded with the `from_pretrained` method are following PyTorch's [recommended practice](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended), which takes around `1.97 seconds` for BERT [[REF]](https://colab.research.google.com/drive/1-Y5f8PWS8ksoaf1A2qI94jq0GxF2pqQ6?usp=sharing). PyTorch offers an [additional alternative way of saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model) using `torch.save(model, PATH)` and `torch.load(PATH)`.

*“Saving a model in this way will save the entire module using Python’s [pickle](https://docs.python.org/3/library/pickle.html) module. The disadvantage of this approach is that the serialized data is bound to the specific classes and the exact directory structure used when the model is saved.”* 

This means that when we save a model with `transformers==4.13.2` it could be potentially incompatible when trying to load with `transformers==4.15.0`. However, loading models this way reduces the loading time by **~12x,** down to `0.166s` for BERT. 

Applying this to `GPT-J` means that we can reduce the loading time from `1 minute and 23 seconds` down to `7.7 seconds`, which is ~10.5x faster.

<br>
<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Model Load time of BERT and GPTJ" src="assets/45_gptj_sagemaker/model_load_time.png"></medium-zoom>
  <figcaption>Figure 1. Model load time of BERT and GPTJ</figcaption>
</figure>
<br>

## Tutorial

With this method of saving and loading models, we achieved model loading performance for `GPT-J` compatible with production scenarios. But we need to keep in mind that we need to align: 

> Align PyTorch and Transformers version when saving the model with `torch.save(model,PATH)` and loading the model with `torch.load(PATH)` to avoid incompatibility.
>

### Save `GPT-J` using `torch.save`

To create our `torch.load()` compatible model file we load `GPT-J` using Transformers and the `from_pretrained` method, and then save it with `torch.save()`.

```python
from transformers import AutoTokenizer,GPTJForCausalLM
import torch

# load fp 16 model
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16)
# save model with torch.save
torch.save(model, "gptj.pt")
```

Now we are able to load our `GPT-J` model with `torch.load()` to run predictions. 

```python
from transformers import pipeline
import torch

# load model
model = torch.load("gptj.pt")
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# create pipeline
gen = pipeline("text-generation",model=model,tokenizer=tokenizer,device=0)

# run prediction
gen("My Name is philipp")
#[{'generated_text': 'My Name is philipp k. and I live just outside of Detroit....
```

---

### Create `model.tar.gz` for the Amazon SageMaker real-time endpoint

Since we can load our model quickly and run inference on it let’s deploy it to Amazon SageMaker. 

There are two ways you can deploy transformers to Amazon SageMaker. You can either [“Deploy a model from the Hugging Face Hub”](https://huggingface.co/docs/sagemaker/inference#deploy-a-model-from-the-%F0%9F%A4%97-hub) directly or [“Deploy a model with `model_data` stored on S3”](https://huggingface.co/docs/sagemaker/inference#deploy-with-model_data). Since we are not using the default Transformers method we need to go with the second option and deploy our endpoint with the model stored on S3. 

For this, we need to create a `model.tar.gz` artifact containing our model weights and additional files we need for inference, e.g. `tokenizer.json`. 

**We provide uploaded and publicly accessible `model.tar.gz` artifacts, which can be used with the `HuggingFaceModel` to deploy `GPT-J` to Amazon SageMaker.**

See [“Deploy `GPT-J` as Amazon SageMaker Endpoint”](https://www.notion.so/Deploy-GPT-J-6B-for-inference-using-Hugging-Face-Transformers-and-Amazon-SageMaker-ce65921edf2246e6a71bb3073e5b3bc7) on how to use them.

If you still want or need to create your own `model.tar.gz`, e.g. because of compliance guidelines, you can use the helper script [convert_gpt.py](https://github.com/philschmid/amazon-sagemaker-gpt-j-sample/blob/main/convert_gptj.py) for this purpose, which creates the `model.tar.gz` and uploads it to S3. 

```bash
# clone directory
git clone https://github.com/philschmid/amazon-sagemaker-gpt-j-sample.git

# change directory to amazon-sagemaker-gpt-j-sample
cd amazon-sagemaker-gpt-j-sample

# create and upload model.tar.gz
pip3 install -r requirements.txt
python3 convert_gptj.py --bucket_name {model_storage}
```

The `convert_gpt.py` should print out an S3 URI similar to this. `s3://hf-sagemaker-inference/gpt-j/model.tar.gz`.

### Deploy `GPT-J` as Amazon SageMaker Endpoint

To deploy our Amazon SageMaker Endpoint we are going to use the [Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/) and the `HuggingFaceModel` class. 

The snippet below uses the `get_execution_role` which is only available inside Amazon SageMaker Notebook Instances or Studio. If you want to deploy a model outside of it check [the documentation](https://huggingface.co/docs/sagemaker/train#installation-and-setup#). 

The `model_uri` defines the location of our `GPT-J` model artifact. We are going to use the publicly available one provided by us. 

```python
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

# IAM role with permissions to create endpoint
role = sagemaker.get_execution_role()

# public S3 URI to gpt-j artifact
model_uri="s3://huggingface-sagemaker-models/transformers/4.12.3/pytorch/1.9.1/gpt-j/model.tar.gz"

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	model_data=model_uri,
	transformers_version='4.12.3',
	pytorch_version='1.9.1',
	py_version='py38',
	role=role, 
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='ml.g4dn.xlarge' #'ml.p3.2xlarge' # ec2 instance type
)
```

If you want to use your own `model.tar.gz` just replace the `model_uri` with your S3 Uri.

The deployment should take around 3-5 minutes.

### Run predictions

We can run predictions using the `predictor` instances created by our `.deploy` method. To send a request to our endpoint we use the `predictor.predict` with our `inputs`.

```python
predictor.predict({
	"inputs": "Can you please let us know more details about your "
})
```

If you want to customize your predictions using additional `kwargs` like `min_length`, check out  “Usage best practices” below. 

## Usage best practices

When using generative models, most of the time you want to configure or customize your prediction to fit your needs, for example by using beam search, configuring the max or min length of the generated sequence, or adjust the temperature to reduce repetition. The Transformers library provides different strategies and `kwargs` to do this, the Hugging Face Inference toolkit offers the same functionality using the `parameters` attribute of your request payload. Below you can find examples on how to generate text without parameters, with beam search, and using custom configurations. If you want to learn about different decoding strategies check out this [blog post](https://huggingface.co/blog/how-to-generate).

### Default request

This is an example of a default request using `greedy` search.

Inference-time after the first request: `3s`

```python
predictor.predict({
	"inputs": "Can you please let us know more details about your "
})
```

### Beam search request

This is an example of a request using `beam` search with 5 beams.

Inference-time after the first request: `3.3s`

```python
predictor.predict({
	"inputs": "Can you please let us know more details about your ",
  "parameters" : {
    "num_beams": 5,
  }
})
```

### Parameterized request

This is an example of a request using a custom parameter, e.g. `min_length` for generating at least 512 tokens.

Inference-time after the first request: `38s`

```python
predictor.predict({
	"inputs": "Can you please let us know more details about your ",
  "parameters" : {
    "max_length": 512,
    "temperature": 0.9,
  }
})
```

### Few-Shot example (advanced)

This is an example of how you could `eos_token_id` to stop the generation on a certain token, e.g. `\n` ,`.` or `###` for few-shot predictions. Below is a few-shot example for generating tweets for keywords.

Inference-time after the first request: `15-45s`

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

end_sequence="###"
temperature=4
max_generated_token_length=25
prompt= """key: markets
tweet: Take feedback from nature and markets, not from people.
###
key: children
tweet: Maybe we die so we can come back as children.
###
key: startups
tweet: Startups shouldn’t worry about how to put out fires, they should worry about how to start them.
###
key: hugging face
tweet:"""

prompt_token_length = len(tokenizer.encode(str(prompt), return_tensors='pt')[0])  


predictor.predict({
	'inputs': prompt,
  "parameters" : {
    "max_length": int(prompt_token_length + max_generated_token_length),
    "temperature": float(temperature),
    "eos_token_id": int(tokenizer.convert_tokens_to_ids(end_sequence)),
    "return_full_text":False
  }
})
```

---

To delete your endpoint you can run. 

```python
predictor.delete_endpoint()
```

## Conclusion

We successfully managed to deploy `GPT-J`, a 6 billion parameter language model created by [EleutherAI](https://www.eleuther.ai/), using Amazon SageMaker. We reduced the model load time from 3.5 minutes down to 8 seconds to be able to run scalable, reliable inference. 

Remember that using `torch.save()` and `torch.load()` can create incompatibility issues. If you want to learn more about scaling out your Amazon SageMaker Endpoints check out my other blog post: [“MLOps: End-to-End Hugging Face Transformers with the Hub & SageMaker Pipelines”](https://www.philschmid.de/mlops-sagemaker-huggingface-transformers).

---

Thanks for reading! If you have any question, feel free to contact me, through [Github](https://github.com/huggingface/transformers), or on the [forum](https://discuss.huggingface.co/c/sagemaker/17). You can also connect with me on [Twitter](https://twitter.com/_philschmid) or [LinkedIn](https://www.linkedin.com/in/philipp-schmid-a6a2bb196/).
