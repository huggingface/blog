# Tool Use, Unified

## tl;dr

There is now a **unified tool use API** across several popular families of models. This API means the same code is portable - few or no model-specific changes are needed to use tools in chats with [Mistral](https://huggingface.co/mistralai), [Cohere](https://huggingface.co/CohereForAI), [NousResearch](https://huggingface.co/NousResearch) or [Llama](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) models. In addition, Transformers now includes helper functionality to make tool calling even easier, as well as [complete documentation](https://huggingface.co/docs/transformers/main/chat_templating#advanced-tool-use--function-calling) and [examples](https://colab.research.google.com/drive/1NPV5ia3_RZB1ksY0DtngQlD0Q7mDl9ln?usp=sharing) for the entire tool use process. Support for even more models will be added in the near future.

## Introduction

Tool use is a curious feature – everyone thinks it’s great, but most people haven’t tried it themselves. Conceptually, it’s very straightforward: you give some tools (callable functions) to your LLM, and it can decide to call them to help it respond to user queries. Maybe you give it a calculator, so it doesn’t have to rely on its internal, unreliable arithmetic abilities. Maybe you let it search the web, or view your calendar, or you give it (read only!) access to a company database so it can pull up information or search technical documentation.

Tool use overcomes a lot of the core limitations of LLMs. Many LLMs are fluent and loquacious, but they are often imprecise with calculations and facts, and hazy on specific details of more niche topics. They don’t know anything that happened after their training cutoff date. They are generalists; they arrive into the conversation with no idea of you or your workplace beyond what you give them in the system message. Tools give them access to structured, specific, relevant and up-to-date information that can help a lot in making them into a genuinely helpful partner, rather than just a fascinating novelty.

The problems arise, however, when you actually try to implement tool use. Documentation is often sparse and even contradictory - and this is true for both closed-source APIs as well as open-source models! Although tool use is simple in theory, it frequently becomes a nightmare in practice: How do you pass tools to the model? How do you ensure the tool prompts match the formats it was trained with? When the model calls a tool, how do you incorporate that into the chat? If you’ve tried to implement tool use before, you’ve probably found that these questions are surprisingly tricky, and that the documentation wasn’t always complete and helpful.

Worse, different models can have wildly different implementations of tool use. Even at the most basic level of just defining the available tools, some providers expect JSON schemas, others expect Python function headers. Even among the ones that expect JSON schemas, small details often differ and create big API incompatibilities. This creates a lot of friction, and generally just deepens user confusion. So what can we do about all of this?

## Chat Templating

Devoted fans of the Hugging Face Cinematic Universe will remember that the open-source community faced a similar challenge in the past with **chat models**. Chat models models use control tokens like `<|start_of_user_turn|>` or `<|end_of_message|>` to let the model know what’s going on in the chat, but different models were trained with totally different control tokens, which meant that users needed to write specific formatting code for each model they wanted to use. This was a huge headache at the time.

Our solution to this was **chat templates** - essentially, models would come with a tiny Jinja template, which would render chats with the right format and control tokens for each model. Chat templates meant that users could write chats in a universal, model-agnostic format, trusting in the Jinja templates to handle any model-specific formatting required.

The obvious approach to supporting tool use, then, was to extend chat templates to support tools as well. And that’s exactly what we did, but tools created a lot of new challenges for the templating system. Let’s go through what those challenges were and how we solved them, and in the process hopefully you’ll gain a deeper understanding of how the system works, and how you can make it work for you.

## Passing tools to a chat template

Our first criterion when designing the tool use API was that it should be intuitive to define tools and pass them to the chat template. We found that most users wrote their tool functions first, and then figured out how to generate tool definitions from them and pass those to the model. This led to an obvious approach: What if users could simply pass functions directly to the chat template, and let it generate tool definitions for them?

The problem here, though, is that “passing functions” is a very language-specific thing to do, and lots of people access chat models through [JavaScript](https://huggingface.co/docs/transformers.js/en/index) or [Rust](https://huggingface.co/docs/text-generation-inference/en/index) instead of Python. So, we found a compromise that we think offers the best of both worlds: **Chat templates expect tools to be defined as JSON schema, but if you pass Python functions to the template instead, they will be automatically converted to JSON schema for you.** This results in a nice, clean API:

```python
def get_current_temperature(location: str):
    """
    Gets the temperature at a given location.

    Args:
        location: The location to get the temperature for
    """
    return 22.0  # bug: Sometimes the temperature is not 22. low priority

tools = [get_current_temperature]    

chat = [
    {"role": "user", "content": "Hey, what's the weather like in Paris right now?"}
]

tool_prompt = tokenizer.apply_chat_template(
    chat, 
    tools=tools
    add_generation_prompt=True,
    return_tensors="pt"
)
```

Internally, the `get_current_temperature` function will be expanded into a complete JSON schema. If you want to see the generated schema, you can use the `get_json_schema` function:

```python
>>> from transformers.utils import get_json_schema

>>> get_json_schema(get_current_weather)
{
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Gets the temperature at a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the temperature for"
                }
            },
            "required": [
                "location"
            ]
        }
    }
}
```

If you prefer manual control, or you’re coding in a language other than Python, you can pass JSON schemas like these directly to the template. However, when you’re working in Python, you can avoid handling JSON schema directly. All you need to do is define your tool functions with clear **names,** accurate **type hints**, and complete **docstrings,** including **argument docstrings,** since all of these will be used to generate the JSON schema that will be read by the template. Much of this is good Python practice anyway, and if you follow it, then you’ll find that no extra work is required - your functions are already usable as tools!

## Adding tool calls to the chat

One detail that is often overlooked by users (and model documentation 😬) is that when a model calls a tool, this actually requires **two** messages to be added to the chat history. The first message is the assistant **calling** the tool, and the second is the **tool response,** the output of the called function. 

Both tool calls and tool responses are necessary - remember that the model only knows what’s in the chat history, and it will not be able to make sense of a tool response if it can’t also see the call it made and the arguments it passed to get that response. “22” on its own is not very informative, but it’s very helpful if you know that the message preceding it was `get_current_temperature("Paris, France")`.

This is one of the areas that can be extremely divergent between different providers, but the standard we settled on is that **tool calls are a field of assistant messages,** like so:

```python
message = {
    "role": "assistant",
    "tool_calls": [
        {
            "type": "function",
             "function": {
                 "name": "get_current_temperature", 
                 "arguments": {
                     "location": "Paris, France"
                }
            }
        }
    ]
}
chat.append(message)
```

## Adding tool responses to the chat

Tool responses are much simpler, especially when tools only return a single string or number.

```python
message = {
    "role": "tool", 
    "name": "get_current_temperature", 
    "content": "22.0"
}
chat.append(message)
```

## Tool use in action

Let’s take the code we have so far and build a complete example of tool-calling. If you want to use tools in your own projects, we recommend playing around with the code here - try running it yourself, adding or removing tools, swapping models, and tweaking details to get a feel for the system. That familiarity will make things much easier when the time comes to implement tool use in your software! To make that easier, this example is [available as a Colab notebook](https://colab.research.google.com/drive/1NPV5ia3_RZB1ksY0DtngQlD0Q7mDl9ln?usp=sharing) as well.

First, let’s set up our model. We’ll use `Hermes-2-Pro-Llama-3-8B`, because it’s small, capable, ungated, and it supports tool calling. You may get better results on complex tasks if you use a larger model, though!

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

Next, we’ll set up our tool and the chat we want to use. Let’s use the `get_current_temperature` example from above:

```python
def get_current_temperature(location: str):
    """
    Gets the temperature at a given location.

    Args:
        location: The location to get the temperature for, in the format "city, country"
    """
    return 22.0  # bug: Sometimes the temperature is not 22. low priority to fix tho

tools = [get_current_temperature]    

chat = [
    {"role": "user", "content": "Hey, what's the weather like in Paris right now?"}
]

tool_prompt = tokenizer.apply_chat_template(
    chat, 
    tools=tools, 
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True,
)
tool_prompt = tool_prompt.to(model.device)
```

Now we’re ready to generate the model’s response to the user query, given the tools it has access to:

```python
out = model.generate(**tool_prompt, max_new_tokens=128)
generated_text = out[0, tool_prompt['input_ids'].shape[1]:]

print(tokenizer.decode(generated_text))
```

and we get:

```python
<tool_call>
{"arguments": {"location": "Paris, France"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

Success! Note how the model correctly inferred that it should pass the argument “Paris, France” rather than just “Paris”, because that is the format recommended by the function docstring.

Next, let’s add that tool call to the chat:

```python
message = {
    "role": "assistant", 
    "tool_calls": [
        {
            "type": "function", 
            "function": {
                "name": "get_current_temperature", 
                "arguments": {"location": "Paris, France"}
            }
        }
    ]
}
chat.append(message)
```

Now, we actually call the tool, and we add its response to the chat:

```python
message = {
    "role": "tool", 
    "name": "get_current_temperature", 
    "content": "22.0"
}
chat.append(message)
```

And finally, just as we did before, we format the updated chat and pass it to the model, so that it can use the tool response in conversation:

```python
tool_prompt = tokenizer.apply_chat_template(
    chat, 
    tools=tools, 
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True,
)
tool_prompt = tool_prompt.to(model.device)

out = model.generate(**tool_prompt, max_new_tokens=128)
generated_text = out[0, tool_prompt['input_ids'].shape[1]:]

print(tokenizer.decode(generated_text))
```

And we get:

```html
The current temperature in Paris is 22.0 degrees Celsius. Enjoy your day!<|im_end|>
```

## The regrettable disunity of response formats

Even though chat templates can hide model-specific differences when converting from chats and tool definitions to formatted text, the same isn’t true in reverse - when the model emits a tool-call, it will do so in its own format, so you’ll need to parse it out manually for now before adding it to the chat in the universal format. Thankfully, most of the formats are pretty intuitive, so this should only be a couple of lines of `json.loads()` or, at worst, a simple `re.search()` to create the tool call dict you need.

We do have some ideas to eliminate this inconvenience, but they’re not quite ready for prime-time yet. “Let us cook”, as the kids say.

## Conclusion

Despite the minor caveat above, we think this is a big improvement on the previous situation, where tool use was scattered, confusing and poorly documented. We hope this makes it a lot easier for open-source developers to include tool-use in their projects, augmenting powerful LLMs with a range of tools that add amazing new capabilities. From smaller models like [Hermes-2-Pro-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B), to the giant state-of-the-art behemoths like [Mistral-Large](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407), [Command-R-Plus](https://huggingface.co/CohereForAI/c4ai-command-r-plus) or [Llama-3.1-405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct), many of the LLMs at the cutting edge now support tool use. We think tools will be an integral part of the next wave of LLM products, and we hope these changes make it easier for you to use them in your own projects. Good luck!