---
title: "Deploy MusicGen using Inference Endpoints" 
thumbnail: /blog/assets/musicgen-inference/thumbnail.png
authors:
- user: reach-vb 
- user: merve
---

<h1> Deploy MusicGen using Inference Endpoints </h1>

<!-- {blog_metadata} -->
<!-- {authors} -->

[MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen) is a powerful music generation model that takes in text prompt and melody control and outputs music. In this blog post, we will walk you through how to deploy it using [Inference Endpoints](https://huggingface.co/inference-endpoints). 

`transformers` pipelines are powerful abstractions to do inference over `transformers` based models. The pipeline API also enables easy deployment of models in Inference Endpoints, with only few clicks. If the model we want to deploy in Inference Endpoints lacks pipeline, we can write our own inference function for it, called [custom handlers](https://huggingface.co/docs/inference-endpoints/guides/custom_handler). 

As the time this blog post is released, MusicGen lacks dedicated `transformers` pipeline. To implement customer handler for MusicGen and serve it, we will..
1. Duplicate the MusicGen repository we want to serve,
2. Write custom handler in `handler.py`, and the dependencies in `requirements.txt`,
3. Create Inference Endpoint for that repository.

### Let's go!

First, we will duplicate the [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) repository to our own profile using [repository duplicator](https://huggingface.co/spaces/osanseviero/repo_duplicator).

Then, from the duplicated repository, we will add `handler.py` and `requirements.txt`.
First, let's take a look at how to infer MusicGen.

You can initialize and use MusicGen with below code 👇 

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["80s pop track with bassy drums and synth"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

You can also input a melody to control the output like below. 

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
sample = next(iter(dataset))["audio"]

# take the first half of the audio sample
sample["array"] = sample["array"][: len(sample["array"]) // 2]

inputs = processor(
    audio=sample["array"],
    sampling_rate=sample["sampling_rate"],
    text=["80s blues track with groovy saxophone"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

For `handler.py`, we can use [this template](https://huggingface.co/docs/inference-endpoints/guides/custom_handler#3-customize-endpointhandler) and override `__init__` and `__call__` methods to contain above code. `__init__` will initialize the model and the processor, and `__call__` will take the data and return the generated music. You can find the modified `EndpointHandler` class below. 👇 

```python
from typing import Dict, List, Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch

class EndpointHandler:
    def __init__(self, path="facebook/musicgen-small"):
        # load model and processor from path
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = MusicgenForConditionalGeneration.from_pretrained(path).to("cuda")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Args:
            data (:dict:):
                The payload with the text prompt and generation parameters.
        """
        # process input
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # preprocess
        inputs = self.processor(
            text=[inputs],
            padding=True,
            return_tensors="pt",).to("cuda")

        # pass inputs with all kwargs in data
        if parameters is not None:
            outputs = self.model.generate(**inputs, max_new_tokens=256, **parameters)
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=256)

        # postprocess the prediction
        prediction = outputs[0].cpu().numpy()

        return [{"generated_audio": prediction}]
```

Then, we will create a `requirements.txt` file that contains dependencies to be able to run above code. In this case, it is below.

````
transformers
accelerate
```

Then, simply uploading these two files to our repository will suffice to serve the model.

![inference-files](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ie_musicgen/files.png)

We can now create the Inference Endpoint. Simply head to [Inference Endpoints](https://huggingface.co/inference-endpoints) page and click `Deploy your first model`. Enter the model repository to point at your duplicated repository identifier, select the hardware and create the endpoint. Any hardware with minimum of 16 GB RAM should work for `musicgen-small`.

![Create Endpoint](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ie_musicgen/create_endpoint.png)

After creating the endpoint, it will be up and running. Then we can simply send a request to the endpoint.

![Endpoint Running](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ie_musicgen/endpoint_running.png)