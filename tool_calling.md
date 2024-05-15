---
title: "Tool calling with Hugging Face" 
thumbnail: /blog/assets/tool_calling/thumbnail.png
authors:
- user: jofthomas
- user: drbh
- user: kkondratenko
  guest: true
---
# Tool Calling in Hugging Face is here!

## Introduction

A few weeks ago, we introduced the new [Messages API](https://huggingface.co/blog/tgi-messages-api) that provided OpenAI compatibility with Text Generation Inference (TGI) and Inference Endpoints.

At the time, the Messages API did not support function calling. This is a limitation that has now been lifted!

Starting with version **1.4.5,** TGI offers an API compatible with the OpenAI Chat Completion API with the addition of the `tools` and the `tools_choice` keys. This change has been propagated in the **`huggingface_hub`** version **0.23.0**, meaning any Hugging Face endpoint can now call some tools if using a newer version.

This new feature is available in Inference Endpoints (dedicated and serverless). We’ll now showcase how you can start building your open-source agents right away.

To get you started quickly, we’ve included detailed code examples of how to:

- Create an Inference Endpoint
- Call tools with the InferenClient
- Use OpenAI’s SDK
- Leverage LangChain and LlamaIndex integrations

## **Create an Inference Endpoint using `huggingface_hub`**

[Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) offers a secure, production solution to easily deploy any Transformers model from the Hub on dedicated infrastructure managed by Hugging Face.

To showcase this newfound power of TGI, we will deploy an 8B instruct tuned model: 

[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

We can deploy the model in just [a few clicks from the UI](https://ui.endpoints.huggingface.co/new?vendor=aws&repository=NousResearch%2FNous-Hermes-2-Mixtral-8x7B-DPO&tgi_max_total_tokens=32000&tgi=true&tgi_max_input_length=1024&task=text-generation&instance_size=2xlarge&tgi_max_batch_prefill_tokens=2048&tgi_max_batch_total_tokens=1024000&no_suggested_compute=true&accelerator=gpu&region=us-east-1) or take advantage of the `huggingface_hub` Python library to programmatically create and manage Inference Endpoints. We demonstrate the use of the Hub library below.

First, we need to specify the endpoint name and model repository, along with the task of text-generation. A protected Inference Endpoint means a valid HF token is required to access the deployed API. We also need to configure the hardware requirements like vendor, region, accelerator, instance type, and size. You can check out the list of available resource options [here](https://api.endpoints.huggingface.cloud/#get-/v2/provider) and view recommended configurations for select models in our catalog [here](https://ui.endpoints.huggingface.co/catalog). 

```python
from huggingface_hub import create_inference_endpoint

endpoint = create_inference_endpoint(
    "llama-3-8b-function-calling",
    repository="meta-llama/Meta-Llama-3-8B-Instruct",
    framework="pytorch",
    task="text-generation",
    accelerator="gpu",
    vendor="aws",
    region="us-east-1",
    type="protected",
    instance_type="nvidia-a10g",
    instance_size="x1",
    custom_image={
        "health_route": "/health",
        "env": {
            "MAX_INPUT_LENGTH": "3500",
            "MAX_BATCH_PREFILL_TOKENS": "3500",
            "MAX_TOTAL_TOKENS": "4096",
            "MAX_BATCH_TOTAL_TOKENS": "4096",
            "HUGGING_FACE_HUB_TOKEN":"<HF_TOKEN>",
            "MODEL_ID": "/repository",
        },
        "url": "ghcr.io/huggingface/text-generation-inference:latest", # use this build or newer
    },
)

endpoint.wait()
print(endpoint.status)

```

Since the model is gated, it will be very important to replace `<HF_TOKEN>` with your own Hugging Face token once you have accepted the terms and conditions of Llama-3-8B-Instruct on the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

It will take a few minutes for our deployment to spin up. We can utilize the `.wait()` utility to block the running thread until the endpoint reaches a final "running" state. Once running, we can confirm its status and take it for a spin via the UI Playground:

![IE UI Overview](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tool_calling/endpoint.png)

Great, we now have a working deployment! 


> ##### 💡 By default, your endpoint will scale-to-zero after 15 minutes of idle time without any requests to optimize cost during periods of inactivity. Check out [the Hub Python Library documentation](https://huggingface.co/docs/huggingface_hub/guides/inference_endpoints) to see all the functionality available for managing your endpoint lifecycle.


## Using Inference Endpoints via OpenAI client libraries

The added support for messages in TGI makes Inference Endpoints directly compatible with the OpenAI Chat Completion API. This means that any existing scripts that use OpenAI models via the OpenAI client libraries can be directly swapped out to use any open LLM running on a TGI endpoint!

With this seamless transition, you can immediately take advantage of the numerous benefits offered by open models:

- Complete control and transparency over models and data
- No more worrying about rate limits
- The ability to fully customize systems according to your specific needs

Let's see how.

### With the InferenceClient from Hugging Face

The function can directly be called with the serverless API or with any endpoint by with the endpoint URL.

```py
from huggingface_hub import InferenceClient

# Ask for weather in the next days using tools
#client = InferenceClient("<ENDPOINT_URL>")
#or
client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
messages = [
    {"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
    {"role": "user", "content": "What's the weather like in Paris, France?"},
]
tools = [
     {
         "type": "function",
         "function": {
             "name": "get_current_weather",
             "description": "Get the current weather",
             "parameters": {
                 "type": "object",
                 "properties": {
                     "location": {
                         "type": "string",
                         "description": "The city and state, e.g. San Francisco, CA",
                     },
                     "format": {
                         "type": "string",
                         "enum": ["celsius", "fahrenheit"],
                         "description": "The temperature unit to use. Infer this from the users location.",
                     },
                 },
                 "required": ["location", "format"],
             },
         },
     },
     
 ]
response = client.chat_completion(
     model="meta-llama/Meta-Llama-3-70B-Instruct",
     messages=messages,
     tools=tools,
     tool_choice="auto",
     max_tokens=500,
 )
response.choices[0].message.tool_calls[0].function
```

```python
ChatCompletionOutputFunctionDefinition(arguments={'format': 'celsius', 'location': 'Paris, France'}, name='get_current_weather', description=None)
```

### With the OpenAI Python client

The example below shows how to make this transition using the [OpenAI Python Library](https://github.com/openai/openai-python). Simply replace the `<ENDPOINT_URL>` with your endpoint URL (be sure to include the `v1/` suffix) and populate the `<HF_API_TOKEN>` field with a valid Hugging Face user token.

We can then use the client as usual, passing a list of messages to stream responses from our Inference Endpoint.

```python
from openai import OpenAI

# initialize the client but point it to TGI
client = OpenAI(
    base_url="<ENDPOINT_URL>" + "/v1/",  # replace with your endpoint url
    api_key="<HF_API_TOKEN>",  # replace with your token
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        },
    }
]
chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Celsius in San Francisco, CA?",
        },
    ],
    tools=tools,
    tool_choice="auto",  # tool selected by caller
    max_tokens=500,
)

called = chat_completion.choices[0]
print(called)
```

```python
Choice(finish_reason='eos_token', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id=0, function=Function(arguments={'format': 'celsius', 'location': 'San Francisco, CA'}, name='get_current_weather', description=None), type='function')]))
```

Behind the scenes, TGI’s Messages API automatically converts the list of messages into the model’s required instruction format using it’s [chat template](https://huggingface.co/docs/transformers/chat_templating). You can learn more about chat templates on the [documentation](https://huggingface.co/docs/transformers/main/en/chat_templating) or on this [space](https://huggingface.co/spaces/Jofthomas/Chat_template_viewer)!

> #####💡 Be mindful that specifying the `auto` parameter will always call a function.


## How to use with LangChain

Now, let’s see how to use functions in the newly created package `langchain_huggingface` 

```python
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace

llm = HuggingFaceEndpoint(
    endpoint_url="https://aac2dhzj35gskpof.us-east-1.aws.endpoints.huggingface.cloud",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
    repetition_penalty=1.03,
)
llm_engine_hf = ChatHuggingFace(llm=llm)

class calculator(BaseModel):
    """Multiply two integers together."""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")
    
llm_with_multiply = llm_engine_hf.bind_tools([calculator], tool_choice="auto")
tool_chain = llm_with_multiply 
tool_chain.invoke("what's 3 * 12")
```

```python
AIMessage(content='', additional_kwargs={'tool_calls': [ChatCompletionOutputToolCall(function=ChatCompletionOutputFunctionDefinition(arguments={'a': 3, 'b': 12}, name='calculator', description=None), id=0, type='function')]}, response_metadata={'token_usage': ChatCompletionOutputUsage(completion_tokens=23, prompt_tokens=154, total_tokens=177), 'model': '', 'finish_reason': 'eos_token'}, id='run-cb823ae4-665e-4c88-b1c6-e69ae5cbbc74-0', tool_calls=[{'name': 'calculator', 'args': {'a': 3, 'b': 12}, 'id': 0}])We’re able to directly leverage the same`ChatOpenAI` class that we would have used with the OpenAI models. This allows all previous code to function with our endpoint by changing just one line of code. 
Let’s now use this declared LLM in a simple RAG pipeline to answer a question over the contents of a HF blog post.
```

## How to use with LlamaIndex

Similarly, you can also use a tool with TGI endpoints in [LLamaIndex](https://www.llamaindex.ai/), but not the serverless API yet

```python
import os
from typing import List, Literal,Optional
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

from llama_index.llms.huggingface import (
    TextGenerationInference,
)

URL = "your_tgi_endpoint"
model = TextGenerationInference(
    model_url=URL, token=False
)  # set token to False in case of public endpoint

def get_current_weather(location: str, format: str):
    """Get the current weather

    Args:
    location (str): The city and state, e.g. San Francisco, CA
    format (str): The temperature unit to use ('celsius' or 'fahrenheit'). Infer this from the users location.
    """
    ...

class WeatherArgs(BaseModel):
    location: str = Field(
        description="The city and region, e.g. Paris, Ile-de-France"
    )
    format: Literal["fahrenheit", "celsius"] = Field(
        description="The temperature unit to use ('fahrenheit' or 'celsius'). Infer this from the location.",
    )

weather_tool = FunctionTool.from_defaults(
    fn=get_current_weather,
    name="get_current_weather",
    description="Get the current weather",
    fn_schema=WeatherArgs,
)

def get_current_weather_n_days(location: str, format: str, num_days: int):
    """Get the weather forecast for the next N days

    Args:
    location (str): The city and state, e.g. San Francisco, CA
    format (str): The temperature unit to use ('celsius' or 'fahrenheit'). Infer this from the users location.
    num_days (int): The number of days for the weather forecast.
    """
    ...

class ForecastArgs(BaseModel):
    location: str = Field(
        description="The city and region, e.g. Paris, Ile-de-France"
    )
    format: Literal["fahrenheit", "celsius"] = Field(
        description="The temperature unit to use ('fahrenheit' or 'celsius'). Infer this from the location.",
    )
    num_days: int = Field(
        description="The duration for the weather forecast in days.",
    )

forecast_tool = FunctionTool.from_defaults(
    fn=get_current_weather_n_days,
    name="get_current_weather_n_days",
    description="Get the current weather for n days",
    fn_schema=ForecastArgs,
)

usr_msg = ChatMessage(
    role=MessageRole.USER,
    content="What's the weather like in Paris over next week?",
)

response = model.chat_with_tools(
    user_msg=usr_msg,
    tools=[
        weather_tool,
        forecast_tool,
    ],
    tool_choice="get_current_weather_n_days",
)

print(response.message.additional_kwargs)
```

```python

{'tool_calls': [{'id': 0, 'type': 'function', 'function': {'description': None, 'name': 'get_current_weather_n_days', 'arguments': {'format': 'celsius', 'location': 'Paris, Ile-de-France', 'num_days': 7}}}]}
```

## Clean up

To clean up our work, we can either pause or delete the model endpoint. This step can alternately be completed via the UI. 

```python
# pause our running endpoint
endpoint.pause()

# optionally delete
endpoint.delete()
```

## Conclusion

Now that you can call some tools with Hugging Face models in the different frameworks, we strongly encourage you to deploy ( and possibly fine-tune) your own models in an Inference Endpoint and experiment with this new feature. We are convinced that the capacity of small LLMs to call some tools will be very beneficial to the community. We can’t wait to see what use cases you will power with open LLMs and tools!
