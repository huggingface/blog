---
title: Inference for PROs
thumbnail: /blog/assets/inference_pro/thumbnail.png
authors:
- user: osanseviero
- user: pcuenq
- user: victor
---

# Inference for PROs

<!-- {blog_metadata} -->
<!-- {authors} -->

Today, we're introducing Inference for PRO users - a community offering that gives you access to APIs of curated endpoints for some of the most exciting models available, as well as improved rate limits for the usage of free Inference API. 

Hugging Face has provided a free Inference API for over 150,000 Transformers and Diffusers models (among other libraries!). Hugging Face PRO users benefit from higher rate limits, allowing them to use models to build prototypes and proof of concepts more extensively. On top of that, PRO users get exclusive access to Inference API for a curated list of models that benefit from extremely fast inference powered by [text-generation-inference](https://github.com/huggingface/text-generation-inference).

## Contents

## Supported Models

In addition to thousands of public models available in the Hub, PRO users get free access to the following state-of-the-art models:

| Model               |       Size       | Use                                   |
|---------------------|:----------------:|---------------------------------------|
| Llama 2 Chat        | 7B, 13B, and 70B | One of the best conversational models |
| Code Llama Base     |    7B and 13B    | Autocomplete code                     |
| Code Llama Instruct |        70B       | Conversational code assistant         |
| SDXL                |                  | Generate images                       |

Inference for PROs makes it easy to experiment and prototype with new models without having to deploy them on your own infrastructure. It gives PRO access to ready-to-use HTTP endpoints for all the available models listed above. It’s not meant to be used for heavy production applications - for that, we recommend using [Inference Endpoints](https://ui.endpoints.huggingface.co/catalog). Inference for PROs also allows using applications that depend upon an LLM endpoint, such as using a [VS Code extension](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode) for code completion or have their own version of [Hugging Chat](http://hf.co/chat).

## Getting started with Inference For PROs

Using Inference for PROs is as simple as sending a POST request to the API endpoint for the model you want to run. You'll also need to get an authentication token from [your token settings page](https://huggingface.co/settings/tokens) and use it in the request. For example, to generate text using [Llama 2 70B Chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) in a terminal session, you'd do something like:

```bash
curl https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf \
    -X POST \
    -d '{"inputs": "In a surprising turn of events, "}' \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <YOUR_TOKEN>"
```

Which would print something like this:

```json
[{"generated_text":"In a surprising turn of events, 20th Century Fox has released a new trailer for Ridley Scott's Alien"}]
```

You can also use many of the familiar transformers generation parameters, like `temperature` or `max_new_tokens`:

```bash
curl https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf \
    -X POST \
    -d '{"inputs": "In a surprising turn of events, ", "parameters": {"temperature": 0.7, "max_new_tokens": 100}}' \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <YOUR_TOKEN>"
```

```json
[{"generated_text":"In a surprising turn of events, 2K has announced that it will be releasing a new free-to-play game called NBA 2K23 Arcade Edition. This game will be available on Apple iOS devices and will allow players to compete against each other in quick, 3-on-3 basketball matches.\n\nThe game promises to deliver fast-paced, action-packed gameplay, with players able to choose from a variety of NBA teams and players, including some of the biggest"}]
```

For more details on the generation parameters, please take a look at [this section](#generation-parameters).
    
To do the same thing in Python, you can take advantage of the `huggingface_hub` Python library utility functions, such as `InferenceClient`:

```bash
pip install huggingface_hub
```

The `InferenceClient` is a helpful wrapper that allows you to make calls to the Inference API and Inference Endpoints easily:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="meta-llama/Llama-2-70b-chat-hf", token=YOUR_TOKEN)

output = client.text_generation("Can you please let us know more details about your ")
print(output)
```

## Applications

### Chat models with Llama and Code Llama 

#### How to prompt

For chat versions:
Short description of how to prompt + link to article

For instruct:
…

### Code infilling with Code Llama

Code models like Code Llama can be used for code completion using the same generation strategy we used in the previous examples: you provide a starting string that may contain code or comments, and the model will try to continue the sequence with plausible content. Code models can also be used for _infilling_, a more specialized task where you provide prefix and suffix sequences, and the model will predict what should go in between. This is great for applications such as IDE extensions. Let's see an example using Code Llama:

```Python
API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-13b-hf"

prompt = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''

output = text_completion(prompt, {"return_full_text": False, "max_new_tokens": 100})
infilled = output[0]["generated_text"]
print(prompt.replace("<FILL_ME>", infilled))
```

In this example, we set return_full_text to False, so the model only returns the infilled portion and not the prompt we provided. We then replace the special string <FILL_ME> with the model output. For more details on how this task works, please take a look at https://huggingface.co/blog/codellama#code-completion

### Stable Diffusion XL

SDXL is also available for PRO users. The response returned by the endpoint consists of a byte stream representing the generated image. If you use the `InferenceClient`, it will automatically decode to a `PIL` image for you:

```Python
sdxl = InferenceClient(model="stabilityai/stable-diffusion-xl-base-1.0", token=YOUR_TOKEN)
image = sdxl.text_to_image(
    "Dark gothic city in a misty night, lit by street lamps. A man in a cape is walking away from us",
    guidance_scale=9,
)
```

![SDXL example generation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-for-pros/sdxl-example.png)

For more details on how to control generation, please take a look at [this section](#controlling-image-generation).

## Generation Parameters

### Controlling Text Generation

-> write here about temperature, etc

### Controlling Image Generation


### Caching

If you run the same generation multiple times, you’ll see that the result returned by the API is the same (even if you are using sampling instead of greedy decoding). This is because recent results are cached. To force a different response each time, we’ll pass a header to tell the server to run a new generation each time: `x-use-cache: 0`.

```Python
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
headers = {"Authorization": "Bearer <YOUR_TOKEN>", "x-use-cache: 0"}

def text_completion(text, params=None):
    payload = {
       "inputs": text,
       "parameters": params if params is not None else {},
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

text_completion("In a surprising turn of events, ", {"temperature": 0.7})
```

### Streaming
Token streaming is the mode in which the server returns the tokens one by one as the model generates them. This enables showing progressive generations to the user rather than waiting for the whole generation. Streaming is an essential aspect of the end-user experience as it reduces latency, one of the most critical aspects of a smooth experience.

<div class="flex justify-center">
    <img 
        class="block dark:hidden" 
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/streaming-generation-visual_360.gif"
    />
    <img 
        class="hidden dark:block" 
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/streaming-generation-visual-dark_360.gif"
    />
</div>

To stream tokens with `InferenceClient`, simply pass `stream=True` and iterate over the response.

```python
for token in client.text_generation("How do you make cheese?", max_new_tokens=12, stream=True):
    print(token)

# To
# make
# cheese
#,
# you
# need
# to
# start
# with
# milk
```

To use the generate_stream endpoint with curl, you can add the `-N`/`--no-buffer` flag, which disables curl default buffering and shows data as it arrives from the server.

```
curl -N https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf \
    -X POST \
    -d '{"inputs": "In a surprising turn of events, ", "parameters": {"temperature": 0.7, "max_new_tokens": 100}}' \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <YOUR_TOKEN>"
```

### Error handling

When errors occur, they may appear in the following flavors:
- As an HTTP response code.
- As a service-specific error that goes with a successful HTTP response code `200`. For example, if the model was evicted after a period of inactivity, you may encounter the following response:

```json
{'error': 'Model stabilityai/stable-diffusion-2-1-base is currently loading',
 'estimated_time': 726.7694702148438}
```

Please make sure your code handles both cases appropriately.

## Subscribe to PRO

to get access to Inference for PROs

Are your looking for a model not available in Inference for PROs. Please use this discussion: xxx

Hugging Face is and will always be an open community of people, please share your findings with the community and … 
