---
title: "ASR+Diarization on Inference Endpoints" 
thumbnail: /blog/assets/asr-diarization/thumbnail.png
authors:
- user: sergeipetrov
- user: reach-vb
---

# ASR+Diarization on Inference Endpoints

Whisper is one of the best open source speech recognition models and definitely the one most widely used. Hugging Face [Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated) make it very easy to deploy it out-of-the-box. However, if you’d like to couple it with a diarization pipeline to identify speakers or introduce assisted generation for speculative decoding, things may get tricky. 

In this blog, we’ll explore how to deploy the Automatic Speech Recogniton (ASR) + Diarization pipeline on Inference Endpoints. We’ll focus on how to do so using a [custom inference handler](https://huggingface.co/docs/inference-endpoints/guides/custom_handler). The implementation of the pipeline is inspired by the famous [Insanely Fast Whisper](https://github.com/Vaibhavs10/insanely-fast-whisper#insanely-fast-whisper). For those ready to go all the way, we provide a containerized model server that can be deployed anywhere. 

This will also be a demonstration of how flexible Inference Endpoints are and that you can host pretty much anything there. Here is the code to follow along:

- [A custom ASR+Diarization handler](https://huggingface.co/sergeipetrov/asrdiarization-handler-default/blob/main/handler.py). This handler works on Inference Endpoints as-is.
- [A standalone platform-agnostic model server](https://github.com/plaggy/fast-whisper-server/tree/main/model-server). This is a container that you can deploy anywhere - including Inference Endpoints! This is also a good fit if you’d like to do something very custom.

*Starting with [Pytorch 2.2](https://pytorch.org/blog/pytorch2-2/) SDPA supports Flash Attention 2 out-of-the-box so you don’t need to install and pass it explicitly.*


### The main modules

Below is a high-level schema of what the endpoint looks like under the hood:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/363a1c4e-6ac6-4610-9f44-fd5dbb1f9c4e/d469d17e-b358-40f8-9290-627aa2b89fcb.png)

The implementation of ASR and diarization pipelines is modularized to cater to a wider range of use cases - the diarization pipeline operates on top of ASR outputs, and you can use only the ASR part if diarization is not needed. For diarization, we propose using the [Pyannote model](https://huggingface.co/pyannote/speaker-diarization-3.1), which is a current open source SOTA.

We’ll add speculative decoding as a way to speed up inference. The speedup is achieved by using a smaller and faster model to do expensive auto-regressive generation. Learn more about how it works with Whisper specifically in [this](https://huggingface.co/blog/whisper-speculative-decoding) great blog post.

Speculative decoding comes with restrictions:

- at least the decoder part of an assistant model should be the same as that of the main model
- the batch size much be 1

Make sure to take the above into account: at some point you wouldn’t want to trade a larger batch for an assistant model. A great choice for an assistant model for Whisper is a corresponding [distilled version](https://huggingface.co/distil-whisper).


### Set up your own

The easiest way to start is to clone the custom handler repository using the [repo duplicator](https://huggingface.co/spaces/huggingface-projects/repo_duplicator).

Here is the model loading piece from the `handler.py`:

```python
from pyannote.audio import Pipeline
from transformers import pipeline, AutoModelForCausalLM

...

self.asr_pipeline = pipeline(
      "automatic-speech-recognition",
      model=model_settings.asr_model,
      torch_dtype=torch_dtype,
      device=device
  )

  self.assistant_model = AutoModelForCausalLM.from_pretrained(
      model_settings.assistant_model,
      torch_dtype=torch_dtype,
      low_cpu_mem_usage=True,
      use_safetensors=True
  ) 
  
  ...

  self.diarization_pipeline = Pipeline.from_pretrained(
      checkpoint_path=model_settings.diarization_model,
      use_auth_token=model_settings.hf_token,
  ) 
  
  ...
```

You can customize the pipeline based on your needs. `ModelSettings` in the `[config.py](http://config.py)` hold the parameters used at the initialization time that define which models to use:

```python
class ModelSettings(BaseSettings):
    asr_model: str
    assistant_model: Optional[str] = None
    diarization_model: Optional[str] = None
    hf_token: Optional[str] = None
```

The parameters can be adjusted by passing environment variables with corresponding names - this works both with a custom container and an inference handler. It’s a [Pydantic feature](https://docs.pydantic.dev/latest/concepts/pydantic_settings/). To pass environment variables to a container during build time you’ll have to create an endpoint via an API call (not via the interface). 

You could hardcode model names not to pass them as environment variables but *note that the diarization pipeline requires a token to be passed explicitly (`hf_token`).*

For context, all the diarization-related pre- and postprocessing utils are in `diarization_utils.py`

The only required component is an ASR model. Optionally, an assistant model can be specified to be used for speculative decoding, and a diarization model can be used to partition a transcription by speakers.


### Deploy on Inference Endpoints

If you only need the ASR part you could specify `asr_model`/`assistant_model` in the `[config.py](http://config.py)` and deploy with a click of a button:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/f2da181d-f9f9-4258-89f1-c833c7fc8263/1eb75d2a-6957-48c9-9e75-9d29f072f64f.png)

To pass environment variables to containers hosted on Inference Endpoints you’ll need to create an endpoint programmatically using the [provided API](https://api.endpoints.huggingface.cloud/#post-/v2/endpoint/-namespace-). Below is an example call:

```python
body = {
    "compute": {
        "accelerator": "gpu",
        "instanceSize": "medium",
        "instanceType": "g5.2xlarge",
        "scaling": {
            "maxReplica": 1,
            "minReplica": 0
        }
    },
    "model": {
        "framework": "pytorch",
        "image": {
            # a default container
            "huggingface": {
                "env": {
		                # this is where a Hub model gets mounted
                    "HF_MODEL_DIR": "/repository", 
                    "DIARIZATION_MODEL": "pyannote/speaker-diarization-3.1",
                    "HF_TOKEN": "<your_token>",
                    "ASR_MODEL": "openai/whisper-large-v3",
                    "ASSISTANT_MODEL": "distil-whisper/distil-large-v3"
                }
            }
        },
        # a model repository on the Hub
        "repository": "sergeipetrov/asrdiarization-handler",
        "task": "custom"
    },
    # the endpoint name
    "name": "asr-diarization-1",
    "provider": {
        "region": "us-east-1",
        "vendor": "aws"
    },
    "type": "private"
}
```


### When to use an assistant model

To give a better idea on when using an assistant model is beneficial given the speculative decoding restrictions below is benchmarking performed with [k6](https://k6.io/docs/):

```bash
# Setup:
# GPU: A10
ASR_MODEL=openai/whisper-large-v3
ASSISTANT_MODEL=distil-whisper/distil-large-v3

# long: 60s audio; short: 8s audio
long_assisted..................: avg=4.15s    min=3.84s    med=3.95s    max=6.88s    p(90)=4.03s    p(95)=4.89s   
long_not_assisted..............: avg=3.48s    min=3.42s    med=3.46s    max=3.71s    p(90)=3.56s    p(95)=3.61s   
short_assisted.................: avg=326.96ms min=313.01ms med=319.41ms max=960.75ms p(90)=325.55ms p(95)=326.07ms
short_not_assisted.............: avg=784.35ms min=736.55ms med=747.67ms max=2s       p(90)=772.9ms  p(95)=774.1ms
```

As you can see, assisted generation gives dramatic performance gains when an audio is short (batch size is 1). If an audio is longer than one chunk speculative decoding may hurt inference time because of the batch size restriction.


### Payload

Once deployed, send your audio along with parameters your Inference Endpoints like this (in Python):

```python
import base64
import requests

API_URL = "<your endpoint URL>"
filepath = "/path/to/audio"

with open(filepath, "rb") as f:
    audio_encoded = base64.b64encode(f.read())

data = {
    "inputs": audio_encoded,
    "parameters": {
        "batch_size": 24
    }
}

resp = requests.post(API_URL, json=data, headers={"Authorization": "Bearer <your token>"})
print(resp.json())
```

Or with [InferenceClient] (there is also an [async version](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)):(https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client#huggingface_hub.InferenceClient):

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model = "<your endpoint URL>", token="<your token>")

with open("/path/to/audio", "rb") as f:
    audio_encoded = base64.b64encode(f.read())
data = {
    "inputs": audio_encoded,
    "parameters": {
        "batch_size": 24
    }
}

res = client.post(json=data)
```


### Standalone model server

[A standalone containerized server](https://github.com/plaggy/fast-whisper-server/tree/main/model-server) has everything to turn a configured pipeline into an endpoint. It can be deployed with hosting services or on unmanaged infrastructure, depending on your needs.

**Recap**   

To recap, these are the options we made available for hosting the ASR+Diarization pipeline on Inference Endpoints:

- A custom model handler on the Hub. You could hardcode models to use or parametrize with environment variables. Remember that a Pyannote diarization pipeline requires a token so you’ll need to pass it as an environment variable to perform diarization.
- A platform-agnostic containerized model service. Can be hosted on Inference Endpoints and anywhere else.
