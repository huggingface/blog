---
title: "Deploy MusicGen using Inference Endpoints" 
thumbnail: /blog/assets/musicgen-inference/thumbnail.png
authors:
- user: reach-vb 
- user: merve
---

<h1> Deploy MusicGen in no time with Inference Endpoints </h1>

<!-- {blog_metadata} -->
<!-- {authors} -->

[MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen) is a powerful music generation model that takes in text prompt and melody control and outputs music. This blog post will guide you through deploying it using [Inference Endpoints](https://huggingface.co/inference-endpoints). 

Inference Endpoints allows us to write custom inference functions called [custom handler](https://huggingface.co/docs/inference-endpoints/guides/custom_handler). These are particularly useful when a model is not supported out-of-the-box by the `transformers` high-level abstraction `pipeline`.

Let's demonstrate this process using MusicGen as an example. To implement a custom handler function for MusicGen and deploy it, we will need to:

At the time of this blog post's release, MusicGen lacks a dedicated `transformers` pipeline. To implement a custom handler function for MusicGen and serve it, we will need to:

1. Duplicate the MusicGen repository we want to serve,
2. Write a custom handler in `handler.py` and the dependencies in `requirements.txt` and add them to the duplicated repository,
3. Create Inference Endpoint for that repository.

Or simply use the final result and deploy our [custom MusicGen model repo](https://huggingface.co/reach-vb/musicgen-large-fp16-endpoint), where we just followed the steps above :)

### Let's go!

First, we will duplicate the [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) repository to our own profile using [repository duplicator](https://huggingface.co/spaces/osanseviero/repo_duplicator).

Then, we will add `handler.py` and `requirements.txt` to the duplicated repository.
First, let's take a look at how to run inference with MusicGen.

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

For `handler.py`, we can use [this template](https://huggingface.co/docs/inference-endpoints/guides/custom_handler#3-customize-endpointhandler) and override the `__init__` and `__call__` methods with our custom inference code. `__init__` will initialize the model and the processor, and `__call__` will take the data and return the generated music. You can find the modified `EndpointHandler` class below. 👇 

```python
from typing import Dict, List, Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch

class EndpointHandler:
    def __init__(self, path=""):
        # load model and processor from path
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = MusicgenForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16).to("cuda")

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
            with torch.autocast("cuda"):
                outputs = self.model.generate(**inputs, **parameters)
        else:
            with torch.autocast("cuda"):
                outputs = self.model.generate(**inputs,)

        # postprocess the prediction
        prediction = outputs[0].cpu().numpy().tolist()

        return [{"generated_audio": prediction}]
```

Then, we will create a `requirements.txt` file containing the dependencies we need to run the code:

```
transformers==4.31.0
accelerate>=0.20.3
```

Uploading these two files to our repository will suffice to serve the model.

![inference-files](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ie_musicgen/files.png)

We can now create the Inference Endpoint. Head to [Inference Endpoints](https://huggingface.co/inference-endpoints) page and click `Deploy your first model`. In the "Model repository" field, enter the identifier of your duplicated repository. Then select the hardware you want and create the endpoint. Any instance with a minimum of 16 GB RAM should work for `musicgen-small`.

![Create Endpoint](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ie_musicgen/create_endpoint.png)

After creating the endpoint, it will be automatically launched and ready to receive requests.

![Endpoint Running](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ie_musicgen/endpoint_running.png)

We can query the endpoint with the below snippet.

```bash
curl URL_OF_ENDPOINT \
-X POST \
-d '{"inputs":"alt rock song"}' \
-H "Authorization: {YOUR_TOKEN_HERE}" \
-H "Content-Type: application/json"
```

We can see the following waveform sequence as output.
```
[{"generated_audio":[[-0.024490159,-0.03154691,-0.0079551935,-0.003828604, ...]]}]
```

You can convert the generated sequence to audio however you want. You can use `scipy` in Python to write it to a .wav file. 

```python
import scipy
import numpy as np

output = [{"generated_audio":[[-0.024490159,-0.03154691,-0.0079551935,-0.003828604, ...]]}]
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=np.array(output[0]["generated_audio"][0]))
```

And voila! 

## Conclusion

In this blog post, we have shown how to deploy MusicGen using Inference Endpoints with a custom inference handler. The same technique can be used for any other model in the Hub that does not have an associated pipeline. All you have to do is override the `Endpoint Handler` class in `handler.py`, and add `requirements.txt` to reflect your project's dependencies. 

### Read More
- [Inference Endpoints documentation covering Custom Handler](https://huggingface.co/docs/inference-endpoints/guides/custom_handler)
